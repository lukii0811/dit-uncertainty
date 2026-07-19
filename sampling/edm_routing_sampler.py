"""EDM sampling utilities for controlled TREAD routing experiments.

The original training-time sampler in ``edm.py`` always evaluates the model in
its default inference mode.  This module keeps the same EDM/Heun update while
adding a per-step routing schedule and trajectory recording.

Terminology is deliberately explicit: ``mask_ratio`` is the fraction of tokens
that skip the configured TREAD blocks.  Therefore ``mask_ratio=0`` is the dense
baseline (all tokens are used), while ``mask_ratio=0.5`` masks half the tokens.
``mask_ratio=1`` is invalid because it would leave no tokens for attention.
"""

from collections.abc import Collection
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


RoutingSteps = Optional[Union[str, int, Collection[int]]]


def _unwrap_module(module):
    while hasattr(module, "module"):
        module = module.module
    return module


def _get_raw_dit(net):
    """Return the raw DiT inside an EDMPrecond or a light wrapper."""
    module = _unwrap_module(net)
    if hasattr(module, "model"):
        module = _unwrap_module(module.model)
    required = ("x_embedder", "router", "routes")
    if not all(hasattr(module, name) for name in required):
        raise TypeError(
            "The routed EDM sampler expected an EDMPrecond-wrapped TREAD DiT "
            "with x_embedder, router, and routes attributes."
        )
    if not module.routes:
        raise ValueError("The model has no TREAD routes configured.")
    return module


def _uses_routing(step_idx: int, routing_steps: RoutingSteps) -> bool:
    if routing_steps is None:
        return False
    if isinstance(routing_steps, str):
        if routing_steps.lower() != "all":
            raise ValueError("routing_steps must be None, 'all', an int, or a collection of ints.")
        return True
    if isinstance(routing_steps, int):
        return step_idx == routing_steps
    return step_idx in routing_steps


def _validate_mask_ratio(mask_ratio: float) -> float:
    mask_ratio = float(mask_ratio)
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(
            f"mask_ratio must be in [0, 1); got {mask_ratio}. "
            "Use mask_ratio=0 for the all-token baseline."
        )
    return mask_ratio


def _step_mask_seed(base_seed, step_idx):
    # A stable integer mix; avoids dependence on batch size or earlier active steps.
    modulus = 2**63 - 1
    return (int(base_seed) + 0x1E3779B97F4A7C15 * (int(step_idx) + 1)) % modulus


def _sample_ids_keep(
    net,
    batch_size,
    mask_ratio,
    device,
    step_idx,
    mask_seed,
    mask_sample_seeds,
):
    raw_dit = _get_raw_dit(net)
    num_patches = raw_dit.x_embedder.num_patches
    if mask_sample_seeds is None:
        base_seeds = [int(mask_seed) + index for index in range(batch_size)]
    else:
        if len(mask_sample_seeds) != batch_size:
            raise ValueError("mask_sample_seeds must contain one seed per batch item")
        base_seeds = [int(seed) for seed in mask_sample_seeds]

    rows = []
    for base_seed in base_seeds:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(_step_mask_seed(base_seed, step_idx))
        rows.append(torch.rand(num_patches, device=device, generator=generator))
    noise_random = torch.stack(rows, dim=0)
    ids_shuffle = torch.argsort(noise_random, dim=1)
    num_mask = int(num_patches * mask_ratio)
    return ids_shuffle[:, : num_patches - num_mask]


@torch.no_grad()
def edm_sampler_with_routing(
    net,
    latents,
    class_labels=None,
    cfg_scale=None,
    feat=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    *,
    routing_steps: RoutingSteps = None,
    mask_ratio: float = 0.5,
    mask_seed: int = 0,
    mask_sample_seeds=None,
    record_xt_trajectory: bool = False,
    record_x0_predictions: bool = False,
    record_routing_masks: bool = False,
) -> Dict[str, Any]:
    """Sample with the original EDM update and a controlled routing schedule.

    Step index 0 is the first/highest-noise generation step.  Step index
    ``num_steps - 1`` is the final/lowest-noise step.

    A single mask is sampled for each active generation step and reused for the
    Euler evaluation and the second-order Heun correction at that step.  When
    ``routing_steps='all'``, a fresh mask is sampled at the start of every step.

    ``xt_trajectory`` stores the initial noisy state followed by one post-update
    state per generation step. ``x0_predictions`` stores one model-predicted X0
    per generation step. For Heun steps this is the corrector evaluation at
    ``t_next``; for the final Euler-only step it is the current evaluation.

    All intermediate tensors are optional so large final-image-only runs do not
    allocate or save trajectory data.
    """
    mask_ratio = _validate_mask_ratio(mask_ratio)
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2 for the EDM schedule.")
    if class_labels is None:
        raise ValueError("class_labels are required for the class-conditional DiT model.")

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    xt_trajectory = [x_next.detach().float().clone()] if record_xt_trajectory else None
    x0_predictions = [] if record_x0_predictions else None
    routing_masks = {}

    for step_idx, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = (
            x_cur
            + (t_hat ** 2 - t_cur ** 2).sqrt()
            * S_noise
            * randn_like(x_cur)
        )

        route_kwargs = {}
        route_this_step = _uses_routing(step_idx, routing_steps) and mask_ratio > 0
        if route_this_step:
            ids_keep = _sample_ids_keep(
                net,
                batch_size=x_hat.shape[0],
                mask_ratio=mask_ratio,
                device=x_hat.device,
                step_idx=step_idx,
                mask_seed=mask_seed,
                mask_sample_seeds=mask_sample_seeds,
            )
            if record_routing_masks:
                routing_masks[step_idx] = ids_keep.detach().cpu()
            route_kwargs = {
                "force_routing": True,
                "overwrite_selection_ratio": mask_ratio,
                "routing_ids_keep": ids_keep,
            }

        denoised_cur = net(
            x_hat.float(),
            t_hat,
            class_labels.long(),
            cfg_scale,
            feat=feat,
            **route_kwargs,
        ).to(torch.float64)
        d_cur = (x_hat - denoised_cur) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        x0_for_step = denoised_cur

        if step_idx < num_steps - 1:
            denoised_corrector = net(
                x_next.float(),
                t_next,
                class_labels.long(),
                cfg_scale,
                feat=feat,
                **route_kwargs,
            ).to(torch.float64)
            d_prime = (x_next - denoised_corrector) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            x0_for_step = denoised_corrector

        if record_xt_trajectory:
            xt_trajectory.append(x_next.detach().float().clone())
        if record_x0_predictions:
            x0_predictions.append(x0_for_step.detach().float().clone())

    return {
        "sample": x_next,
        "xt_trajectory": (
            torch.stack(xt_trajectory) if record_xt_trajectory else None
        ),
        "x0_predictions": (
            torch.stack(x0_predictions) if record_x0_predictions else None
        ),
        "sigma_steps": t_steps.detach().clone(),
        "routing_masks": routing_masks if record_routing_masks else None,
        "routing_steps": routing_steps,
        "mask_ratio": mask_ratio,
        "mask_seed": int(mask_seed),
        "mask_sample_seeds": (
            [int(seed) for seed in mask_sample_seeds]
            if mask_sample_seeds is not None
            else None
        ),
    }


def sample_baseline(net, latents, class_labels, **kwargs):
    """Dense EDM baseline with no routing at any generation step."""
    return edm_sampler_with_routing(
        net,
        latents,
        class_labels,
        routing_steps=None,
        mask_ratio=0.0,
        **kwargs,
    )


def sample_routing_every_step(net, latents, class_labels, mask_ratio=0.5, **kwargs):
    """Setup 1: resample one fresh routing mask per EDM generation step."""
    return edm_sampler_with_routing(
        net,
        latents,
        class_labels,
        routing_steps="all",
        mask_ratio=mask_ratio,
        **kwargs,
    )


def sample_routing_single_step(
    net,
    latents,
    class_labels,
    routing_step,
    mask_ratio=0.5,
    **kwargs,
):
    """Setup 2: apply routing only at one zero-based EDM generation step."""
    return edm_sampler_with_routing(
        net,
        latents,
        class_labels,
        routing_steps=int(routing_step),
        mask_ratio=mask_ratio,
        **kwargs,
    )
