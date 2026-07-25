"""Epistemic / aleatoric maps via law of total variance over routes and seeds.

Uncertainty is computed directly on the predicted velocity v(x_t, t) (no
one-step x0 extrapolation), so it reflects the model's instantaneous output
at the given noisy latent rather than a denoising preview.
"""

import torch

from sampling.sit_routing_sampler import predict_guided_velocity, sample_ids_keep


def patchify(latents, patch_size):
    """(..., C, H, W) -> (..., N, D) with D = C * p * p, N = (H/p)*(W/p)."""
    p = int(patch_size)
    *leading, c, h, w = latents.shape
    if h % p or w % p:
        raise ValueError(f"H,W=({h},{w}) must be divisible by patch_size={p}")
    x = latents.reshape(-1, c, h, w).float()
    x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, gh, gw, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(*leading, (h // p) * (w // p), c * p * p)
    return x


def decompose_route_seed_variance(patches_mk):
    """Law of total variance over routes r and seeds z.

    patches_mk: (M, K, N, D) patchified velocity.
    Returns U_epi, U_ale, U_tot each (N,):
      U_ale = mean_D E_r[Var_z], U_epi = mean_D Var_r[E_z].
    """
    if patches_mk.ndim != 4:
        raise ValueError("patches_mk must have shape (M, K, N, D)")
    mean_z = patches_mk.float().mean(dim=1)
    var_z = patches_mk.float().var(dim=1, unbiased=False)
    u_epi = mean_z.var(dim=0, unbiased=False).mean(dim=-1)
    u_ale = var_z.mean(dim=0).mean(dim=-1)
    return u_epi, u_ale, u_epi + u_ale


@torch.no_grad()
def collect_velocity_mk(
    model, latents_k, labels_k, time_value, mask_ratio, num_routes, mask_seed,
    cfg_scale=1.5, guidance_low=0.0, guidance_high=1.0, cfg_routing_mode="both",
):
    """Evaluate M routes × K seeds at one t; return predicted velocity (M, K, C, H, W)."""
    model_dtype = next(model.parameters()).dtype
    rows = []
    for route_idx in range(num_routes):
        ids = sample_ids_keep(
            model, 1, mask_ratio, latents_k.device, route_idx, mask_seed,
        )
        cols = []
        for k in range(latents_k.shape[0]):
            x = latents_k[k : k + 1]
            v = predict_guided_velocity(
                model, x, time_value, labels_k[k : k + 1],
                cfg_scale, guidance_low, guidance_high, model_dtype,
                route_this_step=True, ids_keep=ids, cfg_routing_mode=cfg_routing_mode,
            )
            cols.append(v.float()[0])
        rows.append(torch.stack(cols, dim=0))
    return torch.stack(rows, dim=0)


@torch.no_grad()
def estimate_uncertainty_maps(
    model, latents_k, labels_k, time_value, mask_ratio, num_routes, mask_seed,
    cfg_scale=1.5, guidance_low=0.0, guidance_high=1.0, cfg_routing_mode="both",
):
    """M×K velocity predictions at one t → patchify → (U_epi, U_ale, U_tot) each (N,)."""
    v_mk = collect_velocity_mk(
        model, latents_k, labels_k, time_value, mask_ratio, num_routes, mask_seed,
        cfg_scale, guidance_low, guidance_high, cfg_routing_mode,
    )
    patch_size = int(model.x_embedder.patch_size[0])
    patches = patchify(v_mk, patch_size)
    return decompose_route_seed_variance(patches)
