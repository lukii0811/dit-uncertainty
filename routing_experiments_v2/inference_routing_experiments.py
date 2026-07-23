"""Paired SiT inference for Baseline, Setup 1, and Setup 2.

This is the self-contained v2 counterpart of the familiar inference script.
It preserves the seed pairing, output shards, VAE loading, and run-manifest
workflow while replacing intermediate-X0 storage with optional trajectory and
velocity-field storage.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

try:
    from .checkpoint import build_model_from_checkpoint
    from .sampling.sit_routing_sampler import sit_sampler_with_routing
except ImportError:
    from checkpoint import build_model_from_checkpoint
    from sampling.sit_routing_sampler import sit_sampler_with_routing


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_seed_class_pairs(path, num_classes, start_index=0, limit=None):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    normalized = []
    seen_ids = set()
    for position, pair in enumerate(pairs):
        sample_id = int(pair.get("sample_id", position))
        seed = int(pair["seed"])
        class_idx = int(pair["class_idx"])
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        if not 0 <= class_idx < num_classes:
            raise ValueError(
                f"class_idx {class_idx} is outside [0, {num_classes - 1}]"
            )
        seen_ids.add(sample_id)
        normalized.append(
            {"sample_id": sample_id, "seed": seed, "class_idx": class_idx}
        )
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    selected = normalized[start_index:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("The selected seed-class range is empty")
    return selected, hashlib.sha256(raw).hexdigest(), path


def make_initial_latents(pairs, resolution, device):
    shape = (4, resolution // 8, resolution // 8)
    latents = []
    for pair in pairs:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(pair["seed"]))
        latents.append(
            torch.randn(
                shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        )
    labels = torch.tensor(
        [pair["class_idx"] for pair in pairs],
        device=device,
        dtype=torch.long,
    )
    return torch.stack(latents, dim=0), labels


def make_mask_sample_seeds(pairs, mask_seed):
    modulus = 2**63 - 1
    return [
        (
            int(mask_seed)
            + int(pair["seed"]) * 6364136223846793005
            + int(pair["class_idx"]) * 1442695040888963407
        )
        % modulus
        for pair in pairs
    ]


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_model_dtype(name, device):
    dtype = DTYPES[name]
    if device.type == "cpu" and dtype != torch.float32:
        print(f"CPU inference: overriding --dtype {name} with float32")
        return torch.float32
    return dtype


def resolve_save_policy(args):
    if args.save_mode == "paired":
        return {
            "xt": True,
            "vcur": True,
            "vprime": False,
            "routing_masks": True,
            "final_latents": True,
            "final_images_tensor": True,
            "final_png": False,
        }
    if args.save_mode == "fid":
        return {
            "xt": False,
            "vcur": False,
            "vprime": False,
            "routing_masks": False,
            "final_latents": False,
            "final_images_tensor": False,
            "final_png": True,
        }
    policy = {
        "xt": args.save_xt,
        "vcur": args.save_vcur,
        "vprime": args.save_vprime,
        "routing_masks": args.save_routing_masks,
        "final_latents": args.save_final_latents,
        "final_images_tensor": args.save_final_images_tensor,
        "final_png": args.save_final_png,
    }
    if not any(policy.values()):
        raise ValueError("Custom save mode needs at least one --save-* option")
    return policy


def policy_record_components(policy):
    components = []
    if policy["xt"]:
        components.append("trajectory")
    if policy["vcur"]:
        components.append("vcur")
    if policy["vprime"]:
        components.append("vprime")
    if policy["routing_masks"]:
        components.append("routing_masks")
    return components


def resolve_routing(args):
    if args.mask_ratio is not None and args.selection_ratio is not None:
        raise ValueError("Pass only one of --mask-ratio and --selection-ratio")
    if args.mask_ratio is not None:
        mask_ratio = float(args.mask_ratio)
        ratio_source = "mask_ratio"
    else:
        selection_ratio = (
            0.5 if args.selection_ratio is None else float(args.selection_ratio)
        )
        if not 0.0 < selection_ratio <= 1.0:
            raise ValueError("selection_ratio must be in (0, 1]")
        mask_ratio = 1.0 - selection_ratio
        ratio_source = "selection_ratio"
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in [0, 1)")
    if args.experiment == "baseline":
        return None, 0.0, ratio_source
    if args.experiment in ("every-step", "fixed-mask"):
        return "all", mask_ratio, ratio_source
    if args.routing_step is None:
        raise ValueError("single-step requires --routing-step")
    if not 0 <= args.routing_step < args.num_steps:
        raise ValueError(f"routing_step must be in [0, {args.num_steps - 1}]")
    return int(args.routing_step), mask_ratio, ratio_source


def resolve_mask_schedule(experiment):
    return "fixed" if experiment == "fixed-mask" else "per_step"


def _resolve_path_from_script(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative.resolve()
    return path.resolve()


def _find_diffusers_vae_directory(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VAE directory does not exist: {root}")

    def is_vae_config(config_path):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        class_name = str(config.get("_class_name", "")).lower()
        return "autoencoder" in class_name or (
            "latent_channels" in config and "block_out_channels" in config
        )

    direct = root / "config.json"
    if direct.is_file() and is_vae_config(direct):
        return root
    candidates = sorted(
        {
            path.parent
            for path in root.rglob("config.json")
            if is_vae_config(path)
        },
        key=lambda path: (len(path.parts), str(path)),
    )
    if len(candidates) == 1:
        print(f"Found nested Diffusers VAE directory: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple Diffusers VAE directories found; pass the exact --vae-model"
        )
    raise FileNotFoundError(f"No Diffusers AutoencoderKL config found under {root}")


def resolve_vae_reference(args):
    if args.vae_source == "local":
        requested = _resolve_path_from_script(args.vae_model)
        path = _find_diffusers_vae_directory(requested)
        return str(path), True, {
            "source": "local",
            "requested": args.vae_model,
            "resolved_path": str(path),
        }
    if args.vae_source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as error:
            raise ImportError(
                "--vae-source modelscope requires: pip install modelscope"
            ) from error
        kwargs = {}
        if args.modelscope_local_dir:
            kwargs["local_dir"] = str(
                Path(args.modelscope_local_dir).expanduser().resolve()
            )
        elif args.modelscope_cache_dir:
            kwargs["cache_dir"] = str(
                Path(args.modelscope_cache_dir).expanduser().resolve()
            )
        try:
            resolved = snapshot_download(args.modelscope_vae_id, **kwargs)
        except TypeError:
            if "local_dir" not in kwargs:
                raise
            resolved = snapshot_download(
                args.modelscope_vae_id, cache_dir=kwargs["local_dir"]
            )
        resolved = _find_diffusers_vae_directory(resolved)
        return str(resolved), True, {
            "source": "modelscope",
            "model_id": args.modelscope_vae_id,
            "resolved_path": str(resolved),
        }
    return args.vae_model, args.local_files_only, {
        "source": "huggingface",
        "model_id": args.vae_model,
        "local_files_only": args.local_files_only,
    }


def load_vae(args, device, model_dtype):
    try:
        from diffusers.models import AutoencoderKL
    except ImportError as error:
        raise ImportError("Final image decoding requires diffusers") from error
    vae_dtype = (
        model_dtype if args.vae_dtype == "model" else DTYPES[args.vae_dtype]
    )
    if device.type == "cpu":
        vae_dtype = torch.float32
    reference, local_only, report = resolve_vae_reference(args)
    vae = AutoencoderKL.from_pretrained(
        reference,
        torch_dtype=vae_dtype,
        local_files_only=local_only,
    ).to(device)
    report["dtype"] = str(vae_dtype)
    return vae.eval(), vae_dtype, report


@torch.no_grad()
def decode_latents_to_uint8(
    vae, latents, decode_batch_size, vae_dtype, latent_scale, latent_bias
):
    outputs = []
    for start in range(0, latents.shape[0], decode_batch_size):
        batch = latents[start : start + decode_batch_size].to(
            device=next(vae.parameters()).device,
            dtype=vae_dtype,
        )
        batch = (batch - latent_bias) / latent_scale
        decoded = vae.decode(batch).sample
        decoded = (
            ((decoded.float() + 1.0) / 2.0)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
        )
        outputs.append(decoded)
    return torch.cat(outputs, dim=0)


def save_png_batch(images, pairs, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for image, pair in zip(images, pairs):
        array = image.permute(1, 2, 0).numpy()
        Image.fromarray(array).save(directory / f"{pair['sample_id']:06d}.png")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_manifest(output_dir, manifest, resume):
    path = output_dir / "run_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"Existing run manifest at {path} does not match")
        if not resume:
            raise FileExistsError(
                f"{path} exists; pass --resume or choose another output directory"
            )
        return
    write_json(path, manifest)


def tensor_output_required(policy):
    return any(
        policy[key]
        for key in (
            "xt",
            "vcur",
            "vprime",
            "routing_masks",
            "final_latents",
            "final_images_tensor",
        )
    )


def tensor_shard_path(output_dir, pairs):
    return output_dir / "tensor_shards" / (
        f"shard_{pairs[0]['sample_id']:06d}_{pairs[-1]['sample_id']:06d}.pt"
    )


def batch_is_complete(output_dir, pairs, policy):
    tensor_done = (
        tensor_shard_path(output_dir, pairs).exists()
        if tensor_output_required(policy)
        else True
    )
    png_done = (
        all(
            (output_dir / "images" / f"{pair['sample_id']:06d}.png").exists()
            for pair in pairs
        )
        if policy["final_png"]
        else True
    )
    return tensor_done and png_done


def save_tensor_shard(output_dir, pairs, result, policy, final_images):
    payload = {
        "format": "tread-sit-paired-inference-shard-v2",
        "sample_id": torch.tensor([p["sample_id"] for p in pairs]),
        "seed": torch.tensor([p["seed"] for p in pairs]),
        "class_idx": torch.tensor([p["class_idx"] for p in pairs]),
        "time_steps": result["time_steps"].cpu(),
    }
    if policy["xt"]:
        payload["xt_trajectory"] = result["xt_trajectory"].permute(
            1, 0, 2, 3, 4
        ).cpu()
    if policy["vcur"]:
        payload["velocity_cur"] = result["velocity_cur"].permute(
            1, 0, 2, 3, 4
        ).cpu()
    if policy["vprime"] and result["velocity_prime"] is not None:
        payload["velocity_prime"] = result["velocity_prime"].permute(
            1, 0, 2, 3, 4
        ).cpu()
    if policy["routing_masks"]:
        payload["routing_masks"] = result["routing_masks"]
    if policy["final_latents"]:
        payload["final_latents"] = result["sample"].float().cpu()
    if policy["final_images_tensor"]:
        payload["final_decoded_uint8"] = final_images
    path = tensor_shard_path(output_dir, pairs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_run_manifest(
    args,
    experiment,
    routing_steps,
    mask_ratio,
    ratio_source,
    policy,
    pairs,
    pair_path,
    pair_sha256,
    checkpoint_report,
    device,
    model_dtype,
):
    return {
        "format": "tread-sit-paired-inference-run-v2",
        "experiment": experiment,
        "routing_steps": routing_steps,
        "mask_schedule": resolve_mask_schedule(experiment),
        "mask_ratio": mask_ratio,
        "selection_ratio": 1.0 - mask_ratio,
        "ratio_argument_used": ratio_source,
        "mask_ratio_definition": "fraction of tokens that skip routed blocks",
        "route_layer_semantics": "start/end indices are inclusive",
        "step_definition": "step 0 starts at t=1 (highest noise)",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_state_key": checkpoint_report["state_key"],
        "seed_pair_file": str(pair_path),
        "seed_pair_sha256": pair_sha256,
        "selected_sample_ids": [pairs[0]["sample_id"], pairs[-1]["sample_id"]],
        "selected_count": len(pairs),
        "model": args.model,
        "resolution": args.resolution,
        "path_type": "linear",
        "prediction": "v",
        "num_steps": args.num_steps,
        "solver": args.solver,
        "cfg_scale": args.cfg_scale,
        "cfg_routing_mode": args.cfg_routing_mode,
        "guidance_interval": [args.guidance_low, args.guidance_high],
        "mask_seed": args.mask_seed,
        "route": {
            "start_layer_idx": args.start_layer_idx,
            "end_layer_idx": args.end_layer_idx,
        },
        "qk_norm": args.qk_norm,
        "fused_attn": args.fused_attn,
        "ops_head": args.ops_head,
        "batch_size": args.batch_size,
        "save_mode": args.save_mode,
        "save_policy": policy,
        "intermediate_x0": False,
        "intermediate_decoding": False,
        "final_decoding": policy["final_images_tensor"] or policy["final_png"],
        "vae_model": args.vae_model,
        "vae_latent_scale": args.vae_latent_scale,
        "vae_latent_bias": args.vae_latent_bias,
        "runtime_device": str(device),
        "runtime_dtype": str(model_dtype),
    }


def run(args):
    if args.resolution % 8:
        raise ValueError("resolution must be divisible by 8")
    device = resolve_device(args.device)
    model_dtype = resolve_model_dtype(args.dtype, device)
    policy = resolve_save_policy(args)
    routing_steps, mask_ratio, ratio_source = resolve_routing(args)
    pairs, pair_sha256, pair_path = load_seed_class_pairs(
        args.seed_pairs, args.num_classes, args.start_index, args.limit
    )
    ignored = () if args.load_projectors else ("projectors.",)
    model, report = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        state_key=args.state_key,
        resolution=args.resolution,
        num_classes=args.num_classes,
        encoder_depth=args.encoder_depth,
        start_layer_idx=args.start_layer_idx,
        end_layer_idx=args.end_layer_idx,
        qk_norm=args.qk_norm,
        fused_attn=args.fused_attn,
        ignored_state_prefixes=ignored,
        allow_unsafe_pickle=args.allow_unsafe_checkpoint_load,
    )
    model = model.to(device=device, dtype=model_dtype).eval()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report["runtime_device"] = str(device)
    report["runtime_dtype"] = str(model_dtype)
    write_json(output_dir / "checkpoint_load_report.json", report)
    manifest = build_run_manifest(
        args,
        args.experiment,
        routing_steps,
        mask_ratio,
        ratio_source,
        policy,
        pairs,
        pair_path,
        pair_sha256,
        report,
        device,
        model_dtype,
    )
    write_run_manifest(output_dir, manifest, args.resume)

    needs_vae = policy["final_images_tensor"] or policy["final_png"]
    vae = vae_dtype = None
    if needs_vae:
        vae, vae_dtype, vae_report = load_vae(args, device, model_dtype)
        write_json(output_dir / "vae_load_report.json", vae_report)
    components = policy_record_components(policy)
    for start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[start : start + args.batch_size]
        if args.resume and batch_is_complete(output_dir, batch_pairs, policy):
            print(f"Skipping completed batch at sample {batch_pairs[0]['sample_id']}")
            continue
        latents, labels = make_initial_latents(batch_pairs, args.resolution, device)
        result = sit_sampler_with_routing(
            model=model,
            latents=latents,
            class_labels=labels,
            num_steps=args.num_steps,
            solver=args.solver,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            cfg_routing_mode=args.cfg_routing_mode,
            routing_steps=routing_steps,
            mask_ratio=mask_ratio,
            mask_seed=args.mask_seed,
            mask_sample_seeds=make_mask_sample_seeds(batch_pairs, args.mask_seed),
            mask_schedule=resolve_mask_schedule(args.experiment),
            record_components=components,
        )
        final_images = None
        if needs_vae:
            final_images = decode_latents_to_uint8(
                vae,
                result["sample"],
                args.decode_batch_size,
                vae_dtype,
                args.vae_latent_scale,
                args.vae_latent_bias,
            )
        if tensor_output_required(policy):
            save_tensor_shard(output_dir, batch_pairs, result, policy, final_images)
        if policy["final_png"]:
            save_png_batch(final_images, batch_pairs, output_dir / "images")
        print(
            f"Completed samples {batch_pairs[0]['sample_id']}.."
            f"{batch_pairs[-1]['sample_id']}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/data1/huanghaitao/TREAD/parameters/0400000.pt")
    parser.add_argument("--seed-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment",
        choices=["baseline", "every-step", "single-step", "fixed-mask"],
        required=True,
    )
    ratios = parser.add_mutually_exclusive_group()
    ratios.add_argument("--mask-ratio", type=float, default=None)
    ratios.add_argument("--selection-ratio", type=float, default=None)
    parser.add_argument("--routing-step", type=int, default=None)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--start-layer-idx", type=int, default=2)
    parser.add_argument("--end-layer-idx", type=int, default=8)
    parser.add_argument("--model", default="SiT-B/2")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--ops-head", type=int, default=16)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--solver", choices=["euler", "heun"], default="heun")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument(
        "--cfg-routing-mode",
        choices=["both", "conditional", "unconditional"],
        default="both",
    )
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--state-key", default="auto")
    parser.add_argument("--load-projectors", action="store_true")
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=list(DTYPES), default="float16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--vae-source",
        choices=["modelscope", "local", "huggingface"],
        default="local",
    )
    parser.add_argument(
        "--vae-model",
        default="/data1/huanghaitao/TREAD/parameters/sdvae/sd-vae-ft-mse/",
    )
    parser.add_argument("--modelscope-vae-id", default="q2792046875/sd-vae-ft-mse")
    parser.add_argument("--modelscope-cache-dir", default=None)
    parser.add_argument("--modelscope-local-dir", default=None)
    parser.add_argument("--vae-dtype", choices=["model", *DTYPES], default="model")
    parser.add_argument("--vae-latent-scale", type=float, default=0.18215)
    parser.add_argument("--vae-latent-bias", type=float, default=0.0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-mode", choices=["paired", "fid", "custom"], default="paired")
    parser.add_argument("--save-xt", action="store_true")
    parser.add_argument("--save-vcur", action="store_true")
    parser.add_argument("--save-vprime", action="store_true")
    parser.add_argument("--save-routing-masks", action="store_true")
    parser.add_argument("--save-final-latents", action="store_true")
    parser.add_argument("--save-final-images-tensor", action="store_true")
    parser.add_argument("--save-final-png", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    run(build_parser().parse_args())
