"""Deterministic linear-flow SiT sampler with controlled TREAD routing.

Step 0 is the first, highest-noise solver step at t=1. Setup 1 resamples one
fresh mask per solver step. Setup 2 routes only one selected solver step. In a
Heun step, the predictor and corrector reuse the same routing mask.

``mask_ratio`` is the fraction of tokens that do not traverse the routed
blocks. ``mask_ratio=0`` keeps every token and is supported as an all-token
routing control. The dense baseline is represented by ``routing_steps=None``.
"""

from collections.abc import Collection
from typing import Optional, Union

import torch


RoutingSteps = Optional[Union[str, int, Collection[int]]]


def _uses_routing(step_idx, routing_steps):
    if routing_steps is None:
        return False
    if isinstance(routing_steps, str):
        if routing_steps.lower() != "all":
            raise ValueError(
                "routing_steps must be None, 'all', an int, or a collection"
            )
        return True
    if isinstance(routing_steps, int):
        return step_idx == routing_steps
    return step_idx in routing_steps


def _validate_mask_ratio(mask_ratio):
    value = float(mask_ratio)
    if not 0.0 <= value < 1.0:
        raise ValueError(
            f"mask_ratio must be in [0, 1); got {value}. "
            "Use selection_ratio=1 or mask_ratio=0 for all-token routing."
        )
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
            raise ValueError(
                "mask_sample_seeds must contain one value per batch item"
            )
        base_seeds = [int(value) for value in mask_sample_seeds]

    rows = []
    for base_seed in base_seeds:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(_step_mask_seed(base_seed, step_idx))
        noise = torch.rand(
            num_tokens, device=device, generator=generator
        )
        rows.append(torch.argsort(noise)[:num_kept])
    return torch.stack(rows, dim=0)


def _velocity_tensor(model_output):
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    if not torch.is_tensor(model_output):
        raise TypeError("The SiT model must return a velocity tensor or tuple")
    return model_output


def predict_velocity(
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
):
    guidance_active = (
        cfg_scale > 1.0
        and float(guidance_low) <= float(time_value) <= float(guidance_high)
    )
    if guidance_active:
        model_input = torch.cat([x, x], dim=0)
        null_labels = torch.full_like(labels, model.num_classes)
        model_labels = torch.cat([labels, null_labels], dim=0)
        model_ids = (
            torch.cat([ids_keep, ids_keep], dim=0)
            if route_this_step
            else None
        )
    else:
        model_input = x
        model_labels = labels
        model_ids = ids_keep if route_this_step else None

    times = torch.full(
        (model_input.shape[0],),
        float(time_value),
        device=model_input.device,
        dtype=model_dtype,
    )
    output = model(
        model_input.to(dtype=model_dtype),
        times,
        model_labels,
        force_routing=route_this_step,
        routing_ids_keep=model_ids,
        return_projector_features=False,
    )
    velocity = _velocity_tensor(output).to(torch.float64)
    if guidance_active:
        conditional, unconditional = velocity.chunk(2, dim=0)
        velocity = unconditional + cfg_scale * (
            conditional - unconditional
        )
    return velocity


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
    routing_steps=None,
    mask_ratio=0.5,
    mask_seed=42,
    mask_sample_seeds=None,
    record_xt_trajectory=False,
    record_x0_predictions=False,
    record_routing_masks=False,
):
    """Integrate the SiT velocity field from t=1 to t=0.

    ``x0_predictions`` contains one model-predicted clean latent per solver
    step. For Euler it is ``x_cur - t_cur*v_cur``. For non-final Heun steps it
    is the corrector prediction ``x_euler - t_next*v_prime``. The accompanying
    ``x0_prediction_times`` tensor removes any ambiguity.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    if cfg_scale < 1.0:
        raise ValueError("cfg_scale must be >= 1.0 for this class-conditional sampler")
    if not 0.0 <= guidance_low <= guidance_high <= 1.0:
        raise ValueError("Guidance interval must satisfy 0 <= low <= high <= 1")
    mask_ratio = _validate_mask_ratio(mask_ratio)
    if class_labels is None:
        raise ValueError("class_labels are required")
    if class_labels.shape[0] != latents.shape[0]:
        raise ValueError("Latent and label batch sizes differ")

    model_dtype = next(model.parameters()).dtype
    time_steps = torch.linspace(
        1.0,
        0.0,
        num_steps + 1,
        dtype=torch.float64,
        device=latents.device,
    )
    x_next = latents.to(torch.float64)
    xt_trajectory = (
        [x_next.detach().float().clone()]
        if record_xt_trajectory
        else None
    )
    x0_predictions = [] if record_x0_predictions else None
    x0_prediction_times = [] if record_x0_predictions else None
    routing_masks = {} if record_routing_masks else None

    for step_idx, (t_cur, t_next) in enumerate(
        zip(time_steps[:-1], time_steps[1:])
    ):
        x_cur = x_next
        route_this_step = _uses_routing(step_idx, routing_steps)
        ids_keep = None
        if route_this_step:
            ids_keep = sample_ids_keep(
                model=model,
                batch_size=x_cur.shape[0],
                mask_ratio=mask_ratio,
                device=x_cur.device,
                step_idx=step_idx,
                mask_seed=mask_seed,
                mask_sample_seeds=mask_sample_seeds,
            )
            if record_routing_masks:
                routing_masks[step_idx] = ids_keep.detach().cpu()

        velocity_cur = predict_velocity(
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
        )
        dt = t_next - t_cur
        x_euler = x_cur + dt * velocity_cur
        x_next = x_euler
        x0_for_step = x_cur - t_cur * velocity_cur
        x0_time = t_cur

        if solver == "heun" and step_idx < num_steps - 1:
            velocity_prime = predict_velocity(
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
            )
            x_next = x_cur + dt * 0.5 * (
                velocity_cur + velocity_prime
            )
            x0_for_step = x_euler - t_next * velocity_prime
            x0_time = t_next

        if record_xt_trajectory:
            xt_trajectory.append(x_next.detach().float().clone())
        if record_x0_predictions:
            x0_predictions.append(x0_for_step.detach().float().clone())
            x0_prediction_times.append(x0_time.detach().clone())

    return {
        "sample": x_next,
        "xt_trajectory": (
            torch.stack(xt_trajectory) if record_xt_trajectory else None
        ),
        "x0_predictions": (
            torch.stack(x0_predictions) if record_x0_predictions else None
        ),
        "x0_prediction_times": (
            torch.stack(x0_prediction_times)
            if record_x0_predictions
            else None
        ),
        "time_steps": time_steps.detach().clone(),
        "routing_masks": routing_masks,
        "routing_steps": routing_steps,
        "mask_ratio": mask_ratio,
        "selection_ratio": 1.0 - mask_ratio,
        "mask_seed": int(mask_seed),
        "solver": solver,
    }


def sample_baseline(model, latents, class_labels, **kwargs):
    return sit_sampler_with_routing(
        model,
        latents,
        class_labels,
        routing_steps=None,
        mask_ratio=0.0,
        **kwargs,
    )


def sample_routing_every_step(
    model, latents, class_labels, mask_ratio=0.5, **kwargs
):
    return sit_sampler_with_routing(
        model,
        latents,
        class_labels,
        routing_steps="all",
        mask_ratio=mask_ratio,
        **kwargs,
    )


def sample_routing_single_step(
    model,
    latents,
    class_labels,
    routing_step,
    mask_ratio=0.5,
    **kwargs,
):
    return sit_sampler_with_routing(
        model,
        latents,
        class_labels,
        routing_steps=int(routing_step),
        mask_ratio=mask_ratio,
        **kwargs,
    )
