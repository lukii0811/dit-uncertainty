"""Load Setup-3 shard latents at a solver step for uncertainty estimation."""

from pathlib import Path

import torch


def load_shards(run_dir):
    root = Path(run_dir).expanduser().resolve() / "tensor_shards"
    return sorted(root.glob("*.pt"))


def records_from_shards(shard_paths):
    """Flatten shards into dicts with sample_id, seed, class_idx, xt, time_steps."""
    records = []
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "xt_trajectory" not in payload:
            raise ValueError(f"Missing xt_trajectory in {path}")
        xt = payload["xt_trajectory"]  # (B, T+1, C, H, W)
        for i in range(xt.shape[0]):
            records.append(
                {
                    "sample_id": int(payload["sample_id"][i]),
                    "seed": int(payload["seed"][i]),
                    "class_idx": int(payload["class_idx"][i]),
                    "xt_trajectory": xt[i],
                    "time_steps": payload["time_steps"],
                }
            )
    return records


def group_by_class(records, num_seeds=4):
    """class_idx -> list of records (exactly num_seeds each)."""
    buckets = {}
    for rec in records:
        buckets.setdefault(rec["class_idx"], []).append(rec)
    grouped = {}
    for class_idx, items in sorted(buckets.items()):
        items = sorted(items, key=lambda r: r["sample_id"])[:num_seeds]
        if len(items) != num_seeds:
            raise ValueError(
                f"class {class_idx}: need {num_seeds} seeds, got {len(items)}"
            )
        grouped[class_idx] = items
    return grouped


def stack_latents_at_step(class_records, step_idx):
    """K trajectories -> latents_k (K,C,H,W), labels_k (K,), time_value."""
    xt = torch.stack([r["xt_trajectory"][step_idx] for r in class_records], dim=0)
    labels = torch.tensor(
        [r["class_idx"] for r in class_records], dtype=torch.long
    )
    time_value = float(class_records[0]["time_steps"][step_idx])
    return xt, labels, time_value


def save_heatmap(vector_n, path, title=None):
    """Reshape (N,) -> sqrt(N)×sqrt(N) grid and save PNG."""
    import math

    import matplotlib.pyplot as plt

    n = int(vector_n.numel())
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"N={n} is not a square grid")
    grid = vector_n.detach().float().cpu().reshape(side, side).numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(grid, origin="upper")
    plt.colorbar(fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def upsample_heatmap(vector_n, out_h, out_w):
    """Reshape (N,) -> sqrt(N)×sqrt(N) grid and nearest-upsample to (out_h, out_w)."""
    import math

    import numpy as np
    from PIL import Image

    n = int(vector_n.numel())
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"N={n} is not a square grid")
    grid = vector_n.detach().float().cpu().reshape(side, side).numpy().astype(np.float32)
    resized = Image.fromarray(grid, mode="F").resize((out_w, out_h), Image.NEAREST)
    return np.array(resized)


def save_overlay(image_hwc, vector_n, path, title=None, alpha=0.45):
    """Overlay an (N,) uncertainty map on top of a decoded HxWx3 uint8 image."""
    import matplotlib.pyplot as plt

    h, w = image_hwc.shape[:2]
    heat = upsample_heatmap(vector_n, h, w)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_hwc)
    im = plt.imshow(heat, cmap="jet", alpha=alpha)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def decode_class_reference_image(base_module, vae, class_records, vae_dtype, latent_scale, latent_bias):
    """Decode the final (t=0) latent of the first seed in the group to an HxWx3 uint8 image."""
    final_latent = class_records[0]["xt_trajectory"][-1].unsqueeze(0)
    decoded = base_module.decode_latents_to_uint8(
        vae, final_latent, 1, vae_dtype, latent_scale, latent_bias
    )
    return decoded[0].permute(1, 2, 0).numpy()


def run_class_step(
    model, class_records, step_idx, out_dir, mask_ratio, num_routes, mask_seed,
    base_image=None, **cfg,
):
    """Estimate Uepi/Uale/Utot for one class at one step and save PNGs."""
    from uncertainty import estimate_uncertainty_maps

    device = next(model.parameters()).device
    xt, labels, t = stack_latents_at_step(class_records, step_idx)
    u_epi, u_ale, u_tot = estimate_uncertainty_maps(
        model, xt.to(device), labels.to(device), t, mask_ratio, num_routes, mask_seed, **cfg,
    )
    class_idx = class_records[0]["class_idx"]
    stem = Path(out_dir) / f"class_{class_idx:04d}_step_{step_idx:03d}"
    save_heatmap(u_epi, f"{stem}_uepi.png", f"Uepi c={class_idx} step={step_idx}")
    save_heatmap(u_ale, f"{stem}_uale.png", f"Uale c={class_idx} step={step_idx}")
    save_heatmap(u_tot, f"{stem}_utot.png", f"Utot c={class_idx} step={step_idx}")
    if base_image is not None:
        save_overlay(base_image, u_epi, f"{stem}_uepi_overlay.png", f"Uepi overlay c={class_idx} step={step_idx}")
        save_overlay(base_image, u_ale, f"{stem}_uale_overlay.png", f"Uale overlay c={class_idx} step={step_idx}")
        save_overlay(base_image, u_tot, f"{stem}_utot_overlay.png", f"Utot overlay c={class_idx} step={step_idx}")
    return {
        "class_idx": class_idx,
        "step_idx": step_idx,
        "time": t,
        "uepi_mean": float(u_epi.mean()),
        "uale_mean": float(u_ale.mean()),
        "utot_mean": float(u_tot.mean()),
    }


def main():
    import argparse
    import csv
    import json

    import inference_routing_experiments as base
    from checkpoint import build_model_from_checkpoint

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", default="0,25,49")
    p.add_argument("--num-routes", type=int, default=16)
    p.add_argument("--num-seeds", type=int, default=4)
    p.add_argument("--mask-seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    p.add_argument("--classes", default=None, help="Comma list of class ids to process (default: all in run-dir)")
    p.add_argument("--overlay", action="store_true", help="Also save heatmaps overlaid on the decoded final image")
    p.add_argument("--vae-source", choices=["modelscope", "local", "huggingface"], default="local")
    p.add_argument("--vae-model", default="/data1/huanghaitao/TREAD/parameters/sdvae/sd-vae-ft-mse/")
    p.add_argument("--modelscope-vae-id", default="q2792046875/sd-vae-ft-mse")
    p.add_argument("--modelscope-cache-dir", default=None)
    p.add_argument("--modelscope-local-dir", default=None)
    p.add_argument("--vae-dtype", default="model")
    p.add_argument("--vae-latent-scale", type=float, default=None)
    p.add_argument("--vae-latent-bias", type=float, default=None)
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    manifest = json.loads((Path(args.run_dir) / "run_manifest.json").read_text())
    mask_ratio = float(manifest["mask_ratio"])
    device = base.resolve_device(args.device)
    model, _ = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        allow_unsafe_pickle=args.allow_unsafe_checkpoint_load,
    )
    model = model.to(device=device).eval()

    vae, vae_dtype = None, None
    latent_scale = args.vae_latent_scale
    latent_bias = args.vae_latent_bias
    if args.overlay:
        if latent_scale is None:
            latent_scale = float(manifest.get("vae_latent_scale", 0.18215))
        if latent_bias is None:
            latent_bias = float(manifest.get("vae_latent_bias", 0.0))
        vae, vae_dtype, _ = base.load_vae(args, device, next(model.parameters()).dtype)

    grouped = group_by_class(records_from_shards(load_shards(args.run_dir)), args.num_seeds)
    if args.classes:
        wanted = {int(c) for c in args.classes.split(",")}
        grouped = {k: v for k, v in grouped.items() if k in wanted}
    steps = [int(s) for s in args.steps.split(",")]
    rows = []
    out = Path(args.output_dir)
    for class_idx, recs in grouped.items():
        base_image = None
        if args.overlay:
            base_image = decode_class_reference_image(base, vae, recs, vae_dtype, latent_scale, latent_bias)
        for step in steps:
            rows.append(
                run_class_step(
                    model, recs, step, out, mask_ratio, args.num_routes, args.mask_seed,
                    base_image=base_image,
                )
            )
            print(f"class={class_idx} step={step} done")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
