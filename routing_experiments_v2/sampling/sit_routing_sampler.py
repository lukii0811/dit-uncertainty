"""Linear-flow Euler/Heun sampler for controlled inference-time TREAD.

This sampler never reconstructs or decodes an intermediate X0. It can retain
the state trajectory, current velocity, corrector velocity, and routing masks.
CFG routing can be applied to both branches, the conditional branch only, or
the unconditional branch only.
"""

from collections.abc import Collection
from typing import Optional, Union

import torch


RoutingSteps = Optional[Union[str, int, Collection[int]]]
CFG_ROUTING_MODES = {"both", "conditional", "unconditional"}


def _uses_routing(step_idx, routing_steps):
    if routing_steps is None:
        return False
    if isinstance(routing_steps, str):
        if routing_steps.lower() != "all":
            raise ValueError("routing_steps string must be 'all'")
        return True
    if isinstance(routing_steps, int):
        return step_idx == routing_steps
    return step_idx in routing_steps


def _validate_mask_ratio(mask_ratio):
    value = float(mask_ratio)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1); got {value}")
    return value


def _step_mask_seed(base_seed, step_idx):
    modulus = 2**63 - 1
    return (
        int(base_seed) + 0x1E3779B97F4A7C15 * (int(step_idx) + 1)
    ) % modulus


def sample_ids_keep(
    model,
    batch_size,
    mask_ratio,
    device,
    step_idx,
    mask_seed,
    mask_sample_seeds=None,
):
    """Draw one deterministic, per-sample routing subset for this solver step."""
    mask_ratio = _validate_mask_ratio(mask_ratio)
    num_tokens = int(model.x_embedder.num_patches)
    num_masked = int(num_tokens * mask_ratio)
    num_kept = num_tokens - num_masked
    if num_kept == num_tokens:
        return torch.arange(num_tokens, device=device).expand(batch_size, -1)

    if mask_sample_seeds is None:
        base_seeds = [int(mask_seed) + index for index in range(batch_size)]
    else:
        if len(mask_sample_seeds) != batch_size:
            raise ValueError("mask_sample_seeds must contain one seed per sample")
        base_seeds = [int(value) for value in mask_sample_seeds]

    rows = []
    for base_seed in base_seeds:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(_step_mask_seed(base_seed, step_idx))
        noise = torch.rand(num_tokens, device=device, generator=generator)
        rows.append(torch.argsort(noise)[:num_kept])
    return torch.stack(rows, dim=0)


def _velocity_tensor(model_output):
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    if not torch.is_tensor(model_output):
        raise TypeError("The SiT model must return a velocity tensor or tuple")
    return model_output


def _model_velocity(model, x, time_value, labels, model_dtype, route, ids_keep):
    times = torch.full(
        (x.shape[0],),
        float(time_value),
        device=x.device,
        dtype=model_dtype,
    )
    output = model(
        x.to(dtype=model_dtype),
        times,
        labels,
        force_routing=bool(route),
        routing_ids_keep=ids_keep if route else None,
        return_projector_features=False,
    )
    return _velocity_tensor(output).to(torch.float64)


def predict_guided_velocity(
    model,
    x,
    time_value,
    labels,
    cfg_scale,
    guidance_low,
    guidance_high,
    model_dtype,
    route_this_step=False,
    ids_keep=None,
    cfg_routing_mode="both",
):
    """Evaluate the guided field with independently routable CFG branches."""
    if cfg_routing_mode not in CFG_ROUTING_MODES:
        raise ValueError(
            f"cfg_routing_mode must be one of {sorted(CFG_ROUTING_MODES)}"
        )
    guidance_active = (
        cfg_scale > 1.0
        and float(guidance_low) <= float(time_value) <= float(guidance_high)
    )
    conditional_routes = route_this_step and cfg_routing_mode in {
        "both",
        "conditional",
    }

    # Outside the CFG interval there is only one model evaluation, so branch
    # selection is undefined; route that single pass to preserve Setup 1/2
    # semantics at every requested routing step.
    if not guidance_active:
        return _model_velocity(
            model,
            x,
            time_value,
            labels,
            model_dtype,
            route_this_step,
            ids_keep if route_this_step else None,
        )

    unconditional_routes = route_this_step and cfg_routing_mode in {
        "both",
        "unconditional",
    }
    null_labels = torch.full_like(labels, model.num_classes)

    # Same routing state permits one batched model call. In the routed case the
    # exact same token subset is deliberately shared by both CFG branches.
    if conditional_routes == unconditional_routes:
        model_input = torch.cat([x, x], dim=0)
        model_labels = torch.cat([labels, null_labels], dim=0)
        model_ids = (
            torch.cat([ids_keep, ids_keep], dim=0)
            if conditional_routes
            else None
        )
        velocity = _model_velocity(
            model,
            model_input,
            time_value,
            model_labels,
            model_dtype,
            conditional_routes,
            model_ids,
        )
        conditional, unconditional = velocity.chunk(2, dim=0)
    else:
        conditional = _model_velocity(
            model,
            x,
            time_value,
            labels,
            model_dtype,
            conditional_routes,
            ids_keep if conditional_routes else None,
        )
        unconditional = _model_velocity(
            model,
            x,
            time_value,
            null_labels,
            model_dtype,
            unconditional_routes,
            ids_keep if unconditional_routes else None,
        )

    return unconditional + cfg_scale * (conditional - unconditional)


@torch.no_grad()
def sit_sampler_with_routing(
    model,
    latents,
    class_labels,
    num_steps=50,
    solver="euler",
    cfg_scale=1.5,
    guidance_low=0.0,
    guidance_high=1.0,
    cfg_routing_mode="both",
    routing_steps=None,
    mask_ratio=0.5,
    mask_seed=42,
    mask_sample_seeds=None,
    mask_schedule="per_step",
    record_components=("trajectory", "vcur"),
):
    """Integrate from t=1 to t=0 without constructing intermediate X0."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    if cfg_scale < 1.0:
        raise ValueError("cfg_scale must be >= 1")
    if not 0.0 <= guidance_low <= guidance_high <= 1.0:
        raise ValueError("Guidance interval must satisfy 0 <= low <= high <= 1")
    if class_labels is None or class_labels.shape[0] != latents.shape[0]:
        raise ValueError("One class label is required for every latent")
    if mask_schedule not in {"per_step", "fixed"}:
        raise ValueError("mask_schedule must be 'per_step' or 'fixed'")
    mask_ratio = _validate_mask_ratio(mask_ratio)
    record_components = set(record_components)
    unknown = record_components - {"trajectory", "vcur", "vprime", "routing_masks"}
    if unknown:
        raise ValueError(f"Unknown record components: {sorted(unknown)}")

    model_dtype = next(model.parameters()).dtype
    time_steps = torch.linspace(
        1.0,
        0.0,
        num_steps + 1,
        dtype=torch.float64,
        device=latents.device,
    )
    x_next = latents.to(torch.float64)
    trajectory = [x_next.detach().float().clone()] if "trajectory" in record_components else None
    velocities_cur = [] if "vcur" in record_components else None
    velocities_prime = (
        []
        if "vprime" in record_components and solver == "heun" and num_steps > 1
        else None
    )
    routing_masks = {} if "routing_masks" in record_components else None
    fixed_ids_keep = None

    for step_idx, (t_cur, t_next) in enumerate(
        zip(time_steps[:-1], time_steps[1:])
    ):
        x_cur = x_next
        route_this_step = _uses_routing(step_idx, routing_steps)
        ids_keep = None
        if route_this_step:
            if mask_schedule == "fixed" and fixed_ids_keep is not None:
                ids_keep = fixed_ids_keep
            else:
                ids_keep = sample_ids_keep(
                    model=model,
                    batch_size=x_cur.shape[0],
                    mask_ratio=mask_ratio,
                    device=x_cur.device,
                    step_idx=0 if mask_schedule == "fixed" else step_idx,
                    mask_seed=mask_seed,
                    mask_sample_seeds=mask_sample_seeds,
                )
                if mask_schedule == "fixed":
                    fixed_ids_keep = ids_keep
            if routing_masks is not None:
                routing_masks[step_idx] = ids_keep.detach().cpu()

        velocity_cur = predict_guided_velocity(
            model=model,
            x=x_cur,
            time_value=t_cur,
            labels=class_labels,
            cfg_scale=cfg_scale,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            model_dtype=model_dtype,
            route_this_step=route_this_step,
            ids_keep=ids_keep,
            cfg_routing_mode=cfg_routing_mode,
        )
        if velocities_cur is not None:
            velocities_cur.append(velocity_cur.detach().float().clone())

        dt = t_next - t_cur
        x_euler = x_cur + dt * velocity_cur
        x_next = x_euler
        if solver == "heun" and step_idx < num_steps - 1:
            velocity_prime = predict_guided_velocity(
                model=model,
                x=x_euler,
                time_value=t_next,
                labels=class_labels,
                cfg_scale=cfg_scale,
                guidance_low=guidance_low,
                guidance_high=guidance_high,
                model_dtype=model_dtype,
                route_this_step=route_this_step,
                ids_keep=ids_keep,
                cfg_routing_mode=cfg_routing_mode,
            )
            x_next = x_cur + dt * 0.5 * (velocity_cur + velocity_prime)
            if velocities_prime is not None:
                velocities_prime.append(velocity_prime.detach().float().clone())

        if trajectory is not None:
            trajectory.append(x_next.detach().float().clone())

    return {
        "sample": x_next,
        "time_steps": time_steps.detach().cpu(),
        "xt_trajectory": torch.stack(trajectory) if trajectory is not None else None,
        "velocity_cur": (
            torch.stack(velocities_cur) if velocities_cur is not None else None
        ),
        "velocity_prime": (
            torch.stack(velocities_prime) if velocities_prime is not None else None
        ),
        "routing_masks": routing_masks,
        "solver": solver,
        "cfg_routing_mode": cfg_routing_mode,
        "routing_steps": routing_steps,
        "mask_schedule": mask_schedule,
        "mask_ratio": mask_ratio,
        "selection_ratio": 1.0 - mask_ratio,
    }

