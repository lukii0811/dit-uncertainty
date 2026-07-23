"""Run a complete TREAD-SiT experiment suite with one model/VAE load.

The suite can run the dense baseline, Setup 1 (fresh routing every solver
step), and Setup 2 at many individual solver steps. All configurations share
one checkpoint load, one VAE load, one seed-class manifest, and one process.
"""

import itertools
import json
import time
import traceback
from pathlib import Path

import torch

try:
    from . import inference_routing_experiments as base
    from .checkpoint import build_model_from_checkpoint
    from .sampling.sit_routing_sampler import sit_sampler_with_routing
except ImportError:
    import inference_routing_experiments as base
    from checkpoint import build_model_from_checkpoint
    from sampling.sit_routing_sampler import sit_sampler_with_routing


def parse_int_spec(value, upper_bound):
    """Parse `all`, comma-separated indices, and inclusive ranges."""
    values = value if isinstance(value, (list, tuple)) else [value]
    if any(str(item).strip().lower() == "all" for item in values):
        return list(range(upper_bound))
    indices = []
    for value_item in values:
        for part in str(value_item).split(","):
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
        raw = args.suite_selection_ratios
        values = [
            float(item.strip())
            for value in raw
            for item in str(value).split(",")
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


def value_tag(value):
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def list_or_scalar(values, scalar):
    return list(dict.fromkeys(values if values is not None else [scalar]))


def parse_guidance_intervals(args):
    if args.suite_guidance_intervals is None:
        return [(args.guidance_low, args.guidance_high)]
    result = []
    for value in args.suite_guidance_intervals:
        low, high = map(float, value.split(":"))
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"Invalid guidance interval: {value}")
        if (low, high) not in result:
            result.append((low, high))
    return result


def sampling_tag(solver, num_steps, cfg_scale, cfg_mode, interval, mask_seed=None):
    low, high = interval
    parts = [
        solver,
        f"steps_{num_steps:04d}",
        f"cfg_{value_tag(cfg_scale)}",
        f"guidance_{value_tag(low)}_{value_tag(high)}",
    ]
    if cfg_mode is not None:
        parts.append(f"cfgroute_{cfg_mode}")
    if mask_seed is not None:
        parts.append(f"maskseed_{mask_seed}")
    return Path(*parts)


def build_configurations(args):
    requested = set(args.experiments)
    ratios = parse_ratio_list(args)
    solvers = list_or_scalar(args.suite_solvers, args.solver)
    num_steps_values = [
        int(value) for value in list_or_scalar(args.suite_num_steps, args.num_steps)
    ]
    cfg_scales = [
        float(value) for value in list_or_scalar(args.suite_cfg_scales, args.cfg_scale)
    ]
    cfg_modes = list_or_scalar(args.suite_cfg_routing_modes, args.cfg_routing_mode)
    mask_seeds = [
        int(value) for value in list_or_scalar(args.suite_mask_seeds, args.mask_seed)
    ]
    intervals = parse_guidance_intervals(args)
    if any(value < 1 for value in num_steps_values):
        raise ValueError("Every sampling step count must be positive")
    if any(value < 1.0 for value in cfg_scales):
        raise ValueError("Every CFG scale must be >= 1")
    configurations = []
    all_single_steps = set()
    base_grid = itertools.product(solvers, num_steps_values, cfg_scales, intervals)
    for solver, num_steps, cfg_scale, interval in base_grid:
        baseline_path = Path("baseline") / sampling_tag(
            solver, num_steps, cfg_scale, None, interval
        )
        baseline_name = "baseline__" + "__".join(baseline_path.parts[1:])
        common = {
            "solver": solver,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "guidance_interval": list(interval),
            "baseline_relative_output": baseline_path,
            "baseline_name": baseline_name,
        }
        if "baseline" in requested:
            configurations.append(
                {
                    **common,
                    "name": baseline_name,
                    "experiment": "baseline",
                    "routing_steps": None,
                    "mask_ratio": 0.0,
                    "selection_ratio": 1.0,
                    "cfg_routing_mode": "none",
                    "mask_seed": mask_seeds[0],
                    "mask_schedule": "per_step",
                    "relative_output": baseline_path,
                }
            )

        target_grid = itertools.product(ratios, cfg_modes, mask_seeds)
        for selection_ratio, cfg_mode, mask_seed in target_grid:
            suffix = sampling_tag(
                solver,
                num_steps,
                cfg_scale,
                cfg_mode,
                interval,
                mask_seed,
            )
            target_common = {
                **common,
                "mask_ratio": 1.0 - selection_ratio,
                "selection_ratio": selection_ratio,
                "cfg_routing_mode": cfg_mode,
                "mask_seed": mask_seed,
                "mask_schedule": "per_step",
            }
            if "every-step" in requested:
                path = Path("setup1_every_step") / ratio_tag(selection_ratio) / suffix
                configurations.append(
                    {
                        **target_common,
                        "name": "setup1__" + "__".join(path.parts[1:]),
                        "experiment": "every-step",
                        "routing_steps": "all",
                        "relative_output": path,
                    }
                )
            if "fixed-mask" in requested:
                path = Path("setup3_fixed_mask") / ratio_tag(selection_ratio) / suffix
                configurations.append(
                    {
                        **target_common,
                        "name": "setup3__" + "__".join(path.parts[1:]),
                        "experiment": "fixed-mask",
                        "routing_steps": "all",
                        "mask_schedule": "fixed",
                        "relative_output": path,
                    }
                )
            if "single-step" in requested:
                steps = parse_int_spec(args.single_steps, num_steps)
                all_single_steps.update(steps)
                for step in steps:
                    path = (
                        Path("setup2_single_step")
                        / ratio_tag(selection_ratio)
                        / f"step_{step:03d}"
                        / suffix
                    )
                    configurations.append(
                        {
                            **target_common,
                            "name": "setup2__" + "__".join(path.parts[1:]),
                            "experiment": "single-step",
                            "routing_steps": step,
                            "relative_output": path,
                        }
                    )
    if not configurations:
        raise ValueError("The suite contains no configurations")
    return configurations, ratios, sorted(all_single_steps)


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
        "format": "tread-sit-paired-inference-run-v2",
        "suite_format": "tread-sit-multi-experiment-suite-v2",
        "name": config["name"],
        "baseline_name": config["baseline_name"],
        "baseline_relative_output": str(config["baseline_relative_output"]),
        "experiment": config["experiment"],
        "routing_steps": config["routing_steps"],
        "mask_schedule": config.get("mask_schedule", "per_step"),
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
        "num_steps": config["num_steps"],
        "solver": config["solver"],
        "cfg_scale": config["cfg_scale"],
        "cfg_routing_mode": config["cfg_routing_mode"],
        "guidance_interval": config["guidance_interval"],
        "mask_seed": config["mask_seed"],
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
        "intermediate_x0": False,
        "intermediate_decoding": False,
        "velocity_recording": (
            "Euler Vcur is reconstructable from trajectory. Heun Vprime is "
            "reconstructable from trajectory and Vcur."
        ),
        "vae_model": args.vae_model,
        "vae_latent_scale": args.vae_latent_scale,
        "vae_latent_bias": args.vae_latent_bias,
        "runtime_device": str(device),
        "runtime_dtype": str(model_dtype),
    }


def tensor_output_required(policy):
    return base.tensor_output_required(policy)


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

    record_components = base.policy_record_components(policy)
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
            num_steps=config["num_steps"],
            solver=config["solver"],
            cfg_scale=config["cfg_scale"],
            guidance_low=config["guidance_interval"][0],
            guidance_high=config["guidance_interval"][1],
            cfg_routing_mode=(
                "both"
                if config["experiment"] == "baseline"
                else config["cfg_routing_mode"]
            ),
            routing_steps=config["routing_steps"],
            mask_ratio=config["mask_ratio"],
            mask_seed=config["mask_seed"],
            mask_sample_seeds=base.make_mask_sample_seeds(
                batch_pairs, config["mask_seed"]
            ),
            mask_schedule=config.get("mask_schedule", "per_step"),
            record_components=record_components,
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

        if tensor_output_required(policy):
            base.save_tensor_shard(
                output_dir,
                batch_pairs,
                result,
                policy,
                final_images,
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

        del result, latents, labels, final_images

    elapsed = time.perf_counter() - started
    return {
        "name": config["name"],
        "output_dir": str(output_dir),
        "status": "complete",
        "completed_batches": completed_batches,
        "skipped_batches": skipped_batches,
        "elapsed_seconds": elapsed,
    }


IMMUTABLE_SUITE_KEYS = (
    "format",
    "checkpoint",
    "seed_pair_sha256",
    "selected_sample_ids",
    "selected_count",
    "model",
    "resolution",
    "path_type",
    "prediction",
    "route",
    "qk_norm",
    "fused_attn",
    "batch_size",
    "save_mode",
    "save_policy",
    "vae_model",
    "vae_latent_scale",
    "vae_latent_bias",
    "runtime_dtype",
)


def _validate_immutable_suite_settings(path, existing, payload):
    """Reject changes that would break paired comparison in one suite."""
    reference_run = None
    run_manifests = sorted(path.parent.rglob("run_manifest.json"))
    if run_manifests:
        reference_run = json.loads(
            run_manifests[0].read_text(encoding="utf-8")
        )
    mismatches = []
    for key in IMMUTABLE_SUITE_KEYS:
        expected = payload.get(key)
        if key in existing:
            actual = existing.get(key)
        elif reference_run is not None:
            actual = reference_run.get(key)
        else:
            continue
        if actual != expected:
            mismatches.append(
                f"{key}: existing={actual!r}, requested={expected!r}"
            )
    if mismatches:
        raise ValueError(
            f"Existing suite manifest at {path} uses incompatible sampling "
            "settings:\n  " + "\n  ".join(mismatches)
        )


def write_or_validate_suite_manifest(path, payload, resume):
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(
                f"{path} exists; pass --resume or use another output directory"
            )
        _validate_immutable_suite_settings(path, existing, payload)

        existing_configs = {
            item["name"]: item for item in existing.get("configurations", [])
        }
        for item in payload.get("configurations", []):
            name = item["name"]
            if name in existing_configs and existing_configs[name] != item:
                raise ValueError(
                    f"Configuration {name!r} already exists with different settings"
                )
            existing_configs[name] = item

        merged = dict(existing)
        for key in IMMUTABLE_SUITE_KEYS:
            if key in payload:
                merged[key] = payload[key]
        merged["experiments"] = list(
            dict.fromkeys(
                [*existing.get("experiments", []), *payload.get("experiments", [])]
            )
        )
        merged["selection_ratios"] = sorted(
            set(existing.get("selection_ratios", []))
            | set(payload.get("selection_ratios", [])),
            reverse=True,
        )
        merged["single_steps"] = sorted(
            set(existing.get("single_steps", []))
            | set(payload.get("single_steps", []))
        )
        for key in (
            "num_steps_list",
            "solvers",
            "cfg_scales",
            "cfg_routing_modes",
            "mask_seeds",
        ):
            combined = [*existing.get(key, []), *payload.get(key, [])]
            merged[key] = sorted(set(combined))
        guidance = {
            tuple(value)
            for value in [
                *existing.get("guidance_intervals", []),
                *payload.get("guidance_intervals", []),
            ]
        }
        merged["guidance_intervals"] = [list(value) for value in sorted(guidance)]
        merged["configurations"] = list(existing_configs.values())
        merged["results"] = existing.get("results", [])
        base.write_json(path, merged)
        print(
            "Extending existing suite: "
            f"{len(existing.get('configurations', []))} existing + "
            f"{len(payload.get('configurations', []))} requested configurations"
        )
        return merged
    base.write_json(path, payload)
    return payload


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
        "format": "tread-sit-multi-experiment-suite-v2",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
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
        "experiments": list(args.experiments),
        "selection_ratios": ratios,
        "single_steps": single_steps,
        "num_steps_list": sorted({item["num_steps"] for item in configurations}),
        "solvers": sorted({item["solver"] for item in configurations}),
        "cfg_scales": sorted({item["cfg_scale"] for item in configurations}),
        "cfg_routing_modes": sorted(
            {
                item["cfg_routing_mode"]
                for item in configurations
                if item["cfg_routing_mode"] != "none"
            }
        ),
        "guidance_intervals": sorted(
            {tuple(item["guidance_interval"]) for item in configurations}
        ),
        "mask_seeds": sorted({item["mask_seed"] for item in configurations}),
        "route": {
            "start_layer_idx": args.start_layer_idx,
            "end_layer_idx": args.end_layer_idx,
        },
        "qk_norm": args.qk_norm,
        "fused_attn": args.fused_attn,
        "batch_size": args.batch_size,
        "save_mode": args.save_mode,
        "save_policy": policy,
        "vae_model": args.vae_model,
        "vae_latent_scale": args.vae_latent_scale,
        "vae_latent_bias": args.vae_latent_bias,
        "runtime_dtype": str(model_dtype),
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
    suite_manifest = write_or_validate_suite_manifest(
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

    needs_vae = policy["final_images_tensor"] or policy["final_png"]
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
        choices=["baseline", "every-step", "single-step", "fixed-mask"],
        default=["baseline", "every-step", "single-step"],
        help="Experiment families to run in this process",
    )
    parser.add_argument(
        "--single-steps",
        nargs="+",
        default=["all"],
        help="Setup 2 steps: all, 0-31, or 0,4,8,12",
    )
    parser.add_argument(
        "--suite-selection-ratios",
        nargs="+",
        default=None,
        help="Kept-token ratios as a list; comma values remain accepted",
    )
    parser.add_argument(
        "--suite-num-steps", nargs="+", type=int, default=None
    )
    parser.add_argument(
        "--suite-solvers",
        nargs="+",
        choices=["euler", "heun"],
        default=None,
    )
    parser.add_argument(
        "--suite-cfg-scales", nargs="+", type=float, default=None
    )
    parser.add_argument(
        "--suite-cfg-routing-modes",
        nargs="+",
        choices=["both", "conditional", "unconditional"],
        default=None,
    )
    parser.add_argument(
        "--suite-guidance-intervals",
        nargs="+",
        default=None,
        help="List such as 0.0:1.0 0.2:0.8",
    )
    parser.add_argument(
        "--suite-mask-seeds", nargs="+", type=int, default=None
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    run_suite(build_parser().parse_args())
