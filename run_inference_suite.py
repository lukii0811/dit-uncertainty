"""Run a complete TREAD-SiT experiment suite with one model/VAE load.

The suite can run the dense baseline, Setup 1 (fresh routing every solver
step), and Setup 2 at many individual solver steps. All configurations share
one checkpoint load, one VAE load, one seed-class manifest, and one process.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import inference_routing_experiments as base
from checkpoint import build_model_from_checkpoint
from sampling.sit_routing_sampler import sit_sampler_with_routing


def parse_int_spec(value, upper_bound):
    """Parse `all`, comma-separated indices, and inclusive ranges."""
    if value.strip().lower() == "all":
        return list(range(upper_bound))
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-", 1)
            start, end = int(pieces[0]), int(pieces[1])
            if end < start:
                raise ValueError(f"Descending step range is invalid: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    if not indices:
        raise ValueError("No Setup 2 steps were selected")
    unique = []
    seen = set()
    for index in indices:
        if not 0 <= index < upper_bound:
            raise ValueError(
                f"Setup 2 step {index} is outside [0, {upper_bound - 1}]"
            )
        if index not in seen:
            seen.add(index)
            unique.append(index)
    return unique


def parse_ratio_list(args):
    if args.suite_selection_ratios is not None:
        values = [
            float(item.strip())
            for item in args.suite_selection_ratios.split(",")
            if item.strip()
        ]
    elif args.selection_ratio is not None:
        values = [float(args.selection_ratio)]
    elif args.mask_ratio is not None:
        values = [1.0 - float(args.mask_ratio)]
    else:
        values = [0.5]
    if not values:
        raise ValueError("At least one selection ratio is required")
    unique = []
    for value in values:
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"Each selection ratio must be in (0, 1]; got {value}"
            )
        if value not in unique:
            unique.append(value)
    return unique


def ratio_tag(selection_ratio):
    text = f"{selection_ratio:.6f}".rstrip("0").rstrip(".")
    return "keep_" + text.replace(".", "p")


def build_configurations(args):
    requested = set(args.experiments)
    ratios = parse_ratio_list(args)
    single_steps = parse_int_spec(args.single_steps, args.num_steps)
    configurations = []

    if "baseline" in requested:
        configurations.append(
            {
                "name": "baseline",
                "experiment": "baseline",
                "routing_steps": None,
                "mask_ratio": 0.0,
                "selection_ratio": 1.0,
                "relative_output": Path("baseline"),
            }
        )

    if "every-step" in requested:
        for selection_ratio in ratios:
            configurations.append(
                {
                    "name": f"setup1_every_step_{ratio_tag(selection_ratio)}",
                    "experiment": "every-step",
                    "routing_steps": "all",
                    "mask_ratio": 1.0 - selection_ratio,
                    "selection_ratio": selection_ratio,
                    "relative_output": (
                        Path("setup1_every_step") / ratio_tag(selection_ratio)
                    ),
                }
            )

    if "single-step" in requested:
        for selection_ratio in ratios:
            for step in single_steps:
                configurations.append(
                    {
                        "name": (
                            f"setup2_step_{step:03d}_{ratio_tag(selection_ratio)}"
                        ),
                        "experiment": "single-step",
                        "routing_steps": step,
                        "mask_ratio": 1.0 - selection_ratio,
                        "selection_ratio": selection_ratio,
                        "relative_output": (
                            Path("setup2_single_step")
                            / ratio_tag(selection_ratio)
                            / f"step_{step:03d}"
                        ),
                    }
                )
    if not configurations:
        raise ValueError("The suite contains no configurations")
    return configurations, ratios, single_steps


def build_run_manifest(
    args,
    config,
    policy,
    pairs,
    pair_path,
    pair_sha256,
    checkpoint_report,
    device,
    model_dtype,
):
    return {
        "format": "tread-sit-paired-inference-run-v1",
        "suite_format": "tread-sit-multi-experiment-suite-v1",
        "experiment": config["experiment"],
        "routing_steps": config["routing_steps"],
        "mask_ratio": config["mask_ratio"],
        "selection_ratio": config["selection_ratio"],
        "ratio_argument_used": "suite_selection_ratio",
        "mask_ratio_definition": "fraction of tokens that skip routed blocks",
        "route_layer_semantics": "start and end layer indices are inclusive",
        "step_definition": "step 0 starts at t=1 (highest noise)",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_state_key": checkpoint_report["state_key"],
        "seed_pair_file": str(pair_path),
        "seed_pair_sha256": pair_sha256,
        "selected_sample_ids": [
            pairs[0]["sample_id"],
            pairs[-1]["sample_id"],
        ],
        "selected_count": len(pairs),
        "model": args.model,
        "resolution": args.resolution,
        "path_type": "linear",
        "prediction": "v",
        "num_steps": args.num_steps,
        "solver": args.solver,
        "cfg_scale": args.cfg_scale,
        "guidance_interval": [args.guidance_low, args.guidance_high],
        "mask_seed": args.mask_seed,
        "route": {
            "start_layer_idx": args.start_layer_idx,
            "end_layer_idx": args.end_layer_idx,
        },
        "qk_norm": args.qk_norm,
        "fused_attn": args.fused_attn,
        "ops_head": args.ops_head,
        "ops_head_note": "recorded from mentor args; not used by official SiT baseline",
        "batch_size": args.batch_size,
        "save_mode": args.save_mode,
        "save_policy": policy,
        "x0_definition": "x0_pred = xt - t * v_pred",
        "x0_record_definition": (
            "Euler: current evaluation at t_cur. Heun: corrector evaluation "
            "at t_next except the final Euler-only step."
        ),
        "vae_model": args.vae_model,
        "vae_latent_scale": args.vae_latent_scale,
        "vae_latent_bias": args.vae_latent_bias,
        "runtime_device": str(device),
        "runtime_dtype": str(model_dtype),
    }


def tensor_output_required(policy):
    return any(
        policy[key]
        for key in (
            "xt",
            "x0_latents",
            "x0_images",
            "routing_masks",
            "final_latents",
            "final_images_tensor",
        )
    )


def run_one_configuration(
    args,
    config,
    suite_output_dir,
    model,
    vae,
    vae_dtype,
    policy,
    pairs,
    pair_path,
    pair_sha256,
    checkpoint_report,
    device,
    model_dtype,
):
    output_dir = suite_output_dir / config["relative_output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        args=args,
        config=config,
        policy=policy,
        pairs=pairs,
        pair_path=pair_path,
        pair_sha256=pair_sha256,
        checkpoint_report=checkpoint_report,
        device=device,
        model_dtype=model_dtype,
    )
    base.write_run_manifest(output_dir, manifest, args.resume)

    needs_x0 = policy["x0_latents"] or policy["x0_images"]
    started = time.perf_counter()
    completed_batches = 0
    skipped_batches = 0
    for batch_start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]
        if args.resume and base.batch_is_complete(
            output_dir, batch_pairs, policy
        ):
            skipped_batches += 1
            continue

        latents, labels = base.make_initial_latents(
            batch_pairs, args.resolution, device
        )
        result = sit_sampler_with_routing(
            model=model,
            latents=latents,
            class_labels=labels,
            num_steps=args.num_steps,
            solver=args.solver,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            routing_steps=config["routing_steps"],
            mask_ratio=config["mask_ratio"],
            mask_seed=args.mask_seed,
            mask_sample_seeds=base.make_mask_sample_seeds(
                batch_pairs, args.mask_seed
            ),
            record_xt_trajectory=policy["xt"],
            record_x0_predictions=needs_x0,
            record_routing_masks=policy["routing_masks"],
        )

        final_images = None
        if policy["final_images_tensor"] or policy["final_png"]:
            final_images = base.decode_latents_to_uint8(
                vae,
                result["sample"],
                args.decode_batch_size,
                vae_dtype,
                args.vae_latent_scale,
                args.vae_latent_bias,
            )

        x0_images = None
        if policy["x0_images"]:
            x0_images = base.decode_x0_predictions(
                vae,
                result["x0_predictions"],
                args.decode_batch_size,
                vae_dtype,
                args.vae_latent_scale,
                args.vae_latent_bias,
            )

        if tensor_output_required(policy):
            base.save_tensor_shard(
                output_dir,
                batch_pairs,
                result,
                policy,
                final_images,
                x0_images,
            )
        if policy["final_png"]:
            base.save_png_batch(
                final_images, batch_pairs, output_dir / "images"
            )
        completed_batches += 1
        print(
            f"  samples {batch_pairs[0]['sample_id']}.."
            f"{batch_pairs[-1]['sample_id']} complete"
        )

        del result, latents, labels, final_images, x0_images

    elapsed = time.perf_counter() - started
    return {
        "name": config["name"],
        "output_dir": str(output_dir),
        "status": "complete",
        "completed_batches": completed_batches,
        "skipped_batches": skipped_batches,
        "elapsed_seconds": elapsed,
    }


def write_or_validate_suite_manifest(path, payload, resume):
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        for dynamic_key in (
            "results",
            "shared_load_seconds",
            "suite_elapsed_seconds",
        ):
            comparable_existing.pop(dynamic_key, None)
        comparable_payload = dict(payload)
        for dynamic_key in (
            "results",
            "shared_load_seconds",
            "suite_elapsed_seconds",
        ):
            comparable_payload.pop(dynamic_key, None)
        if comparable_existing != comparable_payload:
            raise ValueError(
                f"Existing suite manifest at {path} does not match this run"
            )
        if not resume:
            raise FileExistsError(
                f"{path} exists; pass --resume or use another output directory"
            )
        payload["results"] = existing.get("results", [])
        return
    base.write_json(path, payload)


def run_suite(args):
    if args.resolution % 8:
        raise ValueError("resolution must be divisible by 8")
    configurations, ratios, single_steps = build_configurations(args)
    suite_output_dir = Path(args.output_dir).expanduser().resolve()

    print("Planned configurations:")
    for index, config in enumerate(configurations, start=1):
        print(
            f"  {index:03d}/{len(configurations):03d} {config['name']} "
            f"-> {config['relative_output']}"
        )
    if args.dry_run:
        print("Dry run complete; no model or VAE was loaded.")
        return

    device = base.resolve_device(args.device)
    model_dtype = base.resolve_model_dtype(args.dtype, device)
    policy = base.resolve_save_policy(args)
    pairs, pair_sha256, pair_path = base.load_seed_class_pairs(
        args.seed_pairs,
        num_classes=args.num_classes,
        start_index=args.start_index,
        limit=args.limit,
    )

    suite_output_dir.mkdir(parents=True, exist_ok=True)
    suite_manifest_path = suite_output_dir / "suite_manifest.json"
    suite_manifest = {
        "format": "tread-sit-multi-experiment-suite-v1",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "seed_pair_file": str(pair_path),
        "seed_pair_sha256": pair_sha256,
        "selected_count": len(pairs),
        "experiments": list(args.experiments),
        "selection_ratios": ratios,
        "single_steps": single_steps,
        "num_steps": args.num_steps,
        "solver": args.solver,
        "save_mode": args.save_mode,
        "configurations": [
            {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in config.items()
            }
            for config in configurations
        ],
        "results": [],
    }
    write_or_validate_suite_manifest(
        suite_manifest_path, suite_manifest, args.resume
    )

    print("Loading checkpoint and constructing SiT once...")
    load_started = time.perf_counter()
    ignored_prefixes = () if args.load_projectors else ("projectors.",)
    model, checkpoint_report = build_model_from_checkpoint(
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
        ignored_state_prefixes=ignored_prefixes,
        allow_unsafe_pickle=args.allow_unsafe_checkpoint_load,
    )
    model = model.to(device=device, dtype=model_dtype).eval()
    checkpoint_report["runtime_device"] = str(device)
    checkpoint_report["runtime_dtype"] = str(model_dtype)
    checkpoint_report["ops_head_metadata_only"] = args.ops_head
    base.write_json(
        suite_output_dir / "checkpoint_load_report.json", checkpoint_report
    )

    needs_vae = (
        policy["x0_images"]
        or policy["final_images_tensor"]
        or policy["final_png"]
    )
    vae = None
    vae_dtype = None
    if needs_vae:
        print("Loading SD-VAE once...")
        vae, vae_dtype, vae_report = base.load_vae(
            args, device, model_dtype
        )
        base.write_json(
            suite_output_dir / "vae_load_report.json", vae_report
        )
    load_elapsed = time.perf_counter() - load_started
    print(f"Shared model/VAE loading finished in {load_elapsed:.2f}s")

    suite_started = time.perf_counter()
    results_by_name = {
        item["name"]: item for item in suite_manifest.get("results", [])
    }
    for index, config in enumerate(configurations, start=1):
        print(
            f"[{index}/{len(configurations)}] Running {config['name']}"
        )
        try:
            result = run_one_configuration(
                args=args,
                config=config,
                suite_output_dir=suite_output_dir,
                model=model,
                vae=vae,
                vae_dtype=vae_dtype,
                policy=policy,
                pairs=pairs,
                pair_path=pair_path,
                pair_sha256=pair_sha256,
                checkpoint_report=checkpoint_report,
                device=device,
                model_dtype=model_dtype,
            )
        except Exception as error:
            result = {
                "name": config["name"],
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            results_by_name[config["name"]] = result
            suite_manifest["results"] = list(results_by_name.values())
            base.write_json(suite_manifest_path, suite_manifest)
            if not args.continue_on_error:
                raise
            print(f"  FAILED: {type(error).__name__}: {error}")
            continue

        results_by_name[config["name"]] = result
        suite_manifest["results"] = list(results_by_name.values())
        base.write_json(suite_manifest_path, suite_manifest)
        print(
            f"  complete in {result['elapsed_seconds']:.2f}s "
            f"({result['skipped_batches']} batches skipped)"
        )

    suite_manifest["shared_load_seconds"] = load_elapsed
    suite_manifest["suite_elapsed_seconds"] = (
        time.perf_counter() - suite_started
    )
    base.write_json(suite_manifest_path, suite_manifest)
    print(
        "Suite complete. Total experiment time: "
        f"{suite_manifest['suite_elapsed_seconds']:.2f}s"
    )


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "experiment":
            action.required = False
            action.default = "baseline"
            action.help = "Ignored by the suite; use --experiments"
        elif action.dest == "routing_step":
            action.help = "Ignored by the suite; use --single-steps"
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["baseline", "every-step", "single-step"],
        default=["baseline", "every-step", "single-step"],
        help="Experiment families to run in this process",
    )
    parser.add_argument(
        "--single-steps",
        default="all",
        help="Setup 2 steps: all, 0-31, or 0,4,8,12",
    )
    parser.add_argument(
        "--suite-selection-ratios",
        default=None,
        help="Comma-separated kept-token ratios, e.g. 1.0,0.75,0.5",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    run_suite(build_parser().parse_args())
