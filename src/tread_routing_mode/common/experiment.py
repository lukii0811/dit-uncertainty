"""Shared execution and artifact helpers for routing experiments."""

from pathlib import Path

import torch

from .artifacts import save_image, tensor_sha256, write_json
from .sampling import sample_flow


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_subdir(root, class_id, noise_seed):
    return Path(root) / f"class_{class_id:04d}" / f"seed_{noise_seed:010d}"


def make_noise(model, seed, device):
    generator_device = device if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    side = int(model.pos_embed.shape[1] ** 0.5) * model.patch_size
    noise = torch.randn((1, model.in_channels, side, side), generator=generator,
                        device=generator_device)
    return noise.to(device)


def load_vae(name, device):
    from diffusers import AutoencoderKL
    project_root = Path(__file__).resolve().parents[3]
    cache_dir = project_root / "weights" / "huggingface" / "hub"
    local_vae_cache = (
        cache_dir / f"models--{name.replace('/', '--')}"
    ).exists()
    return AutoencoderKL.from_pretrained(
        name, cache_dir=cache_dir, use_safetensors=True,
        local_files_only=local_vae_cache).to(device).eval()


def run_and_save(*, model, vae, device, output_dir, class_id, noise_seed,
                 schedule, steps, cfg_scale, solver, save_every, decode_every):
    output_dir = sample_subdir(output_dir, class_id, noise_seed)
    states_dir, images_dir = output_dir / "states", output_dir / "images"
    states_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    noise = make_noise(model, noise_seed, device)
    label = torch.tensor([class_id], device=device)
    final, states = sample_flow(
        model, noise, label, steps=steps, cfg_scale=cfg_scale,
        save_every=save_every, solver=solver, routing_schedule=schedule,
    )
    records = []
    for state in states:
        name = f"step_{state['step']:04d}"
        state_path = states_dir / f"{name}.pt"
        torch.save(state, state_path)
        record = {
            "step": state["step"], "t": state["t"],
            "state": str(state_path),
            "latent_sha256": tensor_sha256(state["latent"]),
        }
        should_decode = state["step"] == 0 or (
            vae is not None and decode_every and state["step"] % decode_every == 0
        )
        if vae is not None and should_decode:
            with torch.inference_mode():
                decoded = vae.decode(state["latent"].to(device) / 0.18215).sample[0]
            image_path = images_dir / f"{name}.png"
            save_image(decoded, image_path)
            record["image"] = str(image_path)
        records.append(record)
    final_path = output_dir / "final.pt"
    torch.save({"latent": final.cpu(), "class_id": class_id, "seed": noise_seed}, final_path)
    metadata = {
        "class_id": class_id, "noise_seed": noise_seed,
        "mask_mode": schedule.mode, "mask_seed": schedule.mask_seed,
        "route_step": schedule.route_step,
        "keep_ratio": schedule.keep_ratio,
        "steps": steps, "cfg_scale": cfg_scale, "solver": solver,
        "initial_noise_sha256": tensor_sha256(noise),
        "final": str(final_path), "final_sha256": tensor_sha256(final),
        "states": records, "mask_history": schedule.history,
    }
    write_json(metadata, output_dir / "metadata.json")
    return output_dir, metadata
