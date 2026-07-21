#!/usr/bin/env python3
"""Run TREAD routing at exactly one Euler timestep over a ratio/timestep grid."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

SETUP_DIR = Path(__file__).resolve().parent
STAGE_DIR = SETUP_DIR.parent
PROJECT_ROOT = STAGE_DIR.parents[1]
DEFAULT_CONFIG = STAGE_DIR / "config.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tread_routing_mode.common.artifacts import file_sha256, write_json
from tread_routing_mode.common.checkpoint import load_checkpoint
from tread_routing_mode.common.experiment import (
    load_vae,
    run_and_save,
    sample_subdir,
)
from tread_routing_mode.common.image_metrics import ImageMetricComparator
from tread_routing_mode.common.routing import RoutingSchedule


METRICS = ("dino_similarity", "lpips_distance")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        help="Generation device. Defaults to the device recorded by baseline.",
    )
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument(
        "--route-steps",
        type=int,
        nargs="+",
        help="Optional subset of Euler steps. Defaults to every step N..1.",
    )
    parser.add_argument(
        "--keep-ratios",
        type=float,
        nargs="+",
        help="Optional subset of configured keep ratios.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate completed trials instead of resuming them.",
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(rows, path):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def ratio_slug(keep_ratio):
    return f"{keep_ratio:.3f}".replace(".", "p")


def canonical_float_list(values):
    return [float(value) for value in values]


def validate_design(config, args):
    fixed = config["fixed_generation"]
    design = config["setup_2_single_timestep"]
    if fixed["solver"] != "euler":
        raise ValueError("Setup 2 is fixed to the Euler ODE solver")
    if fixed["save_every"] != 1 or fixed["decode_every"] != 1:
        raise ValueError("Every diffusion state and image must be saved")

    configured_ratios = canonical_float_list(design["keep_ratios"])
    if len(configured_ratios) != 5 or len(set(configured_ratios)) != 5:
        raise ValueError("Setup 2 must define exactly five unique keep_ratios")
    if not all(0.0 < ratio <= 1.0 for ratio in configured_ratios):
        raise ValueError("Every keep_ratio must be in (0, 1]")
    if not any(math.isclose(ratio, 1.0) for ratio in configured_ratios):
        raise ValueError("The five keep_ratios must include the ratio=1 control")

    if design["route_steps"] == "all":
        configured_steps = list(range(int(fixed["sample_steps"]), 0, -1))
    else:
        configured_steps = [int(step) for step in design["route_steps"]]
    expected_steps = set(range(1, int(fixed["sample_steps"]) + 1))
    if set(configured_steps) != expected_steps or len(configured_steps) != len(expected_steps):
        raise ValueError("route_steps must cover every Euler model evaluation N..1")

    selected_steps = args.route_steps or configured_steps
    if len(selected_steps) != len(set(selected_steps)):
        raise ValueError("--route-steps contains duplicates")
    if not set(selected_steps).issubset(expected_steps):
        raise ValueError(f"--route-steps must be within [1, {fixed['sample_steps']}]")
    selected_steps = sorted(selected_steps, reverse=True)

    selected_ratios = canonical_float_list(args.keep_ratios or configured_ratios)
    if len(selected_ratios) != len(set(selected_ratios)):
        raise ValueError("--keep-ratios contains duplicates")
    for ratio in selected_ratios:
        if not any(math.isclose(ratio, configured) for configured in configured_ratios):
            raise ValueError(
                f"keep_ratio={ratio} is not in the configured grid {configured_ratios}"
            )
    selected_ratios = [
        next(configured for configured in configured_ratios if math.isclose(ratio, configured))
        for ratio in selected_ratios
    ]
    return fixed, design, configured_steps, configured_ratios, selected_steps, selected_ratios


def state_hashes(metadata):
    return {int(row["step"]): row["latent_sha256"] for row in metadata["states"]}


def image_sequence_sha256(sample_dir):
    digest = hashlib.sha256()
    paths = sorted((sample_dir / "images").glob("step_*.png"))
    if not paths:
        raise RuntimeError(f"No decoded images found in {sample_dir}")
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def validate_saved_artifacts(sample_dir, expected_steps):
    state_steps = {
        int(path.stem.removeprefix("step_"))
        for path in (sample_dir / "states").glob("step_*.pt")
    }
    image_steps = {
        int(path.stem.removeprefix("step_"))
        for path in (sample_dir / "images").glob("step_*.png")
    }
    if state_steps != expected_steps:
        raise RuntimeError(
            f"Saved latent steps are incomplete in {sample_dir}: {sorted(state_steps)}"
        )
    if image_steps != expected_steps:
        raise RuntimeError(
            f"Saved image steps are incomplete in {sample_dir}: {sorted(image_steps)}"
        )


def validate_mask_history(metadata, *, steps, route_step, keep_ratio, num_tokens):
    history = metadata["mask_history"]
    if len(history) != steps:
        raise RuntimeError(f"Expected {steps} schedule records, got {len(history)}")
    if [row["step"] for row in history] != list(range(steps, 0, -1)):
        raise RuntimeError("Routing history does not cover Euler steps N..1")

    selected = [row for row in history if row["routing_selected"]]
    if len(selected) != 1 or selected[0]["step"] != route_step:
        raise RuntimeError("Routing was not selected at exactly the requested timestep")
    active = selected[0]
    expected_kept = max(1, int(num_tokens * keep_ratio))
    if not active["routing_active"]:
        raise RuntimeError("The selected timestep did not enter the routing path")
    if not math.isclose(active["keep_ratio"], keep_ratio):
        raise RuntimeError("keep_ratio changed at the selected timestep")
    if active["num_kept"] != expected_kept:
        raise RuntimeError("Unexpected number of kept tokens at the selected timestep")
    if active["num_routed"] != num_tokens - expected_kept:
        raise RuntimeError("Unexpected number of masked tokens at the selected timestep")

    masked = keep_ratio < 1.0
    if active["masking_active"] != masked:
        raise RuntimeError("masking_active disagrees with keep_ratio")
    if masked and active["mask_draw_index"] != 1:
        raise RuntimeError("The selected timestep must perform exactly one mask draw")
    if not masked and active["mask_draw_index"] is not None:
        raise RuntimeError("The ratio=1 control must not draw a random mask")

    for row in history:
        if row is active:
            continue
        if row["routing_active"] or row["routing_selected"] or row["masking_active"]:
            raise RuntimeError("Routing was active outside the selected timestep")
        if row["num_kept"] != num_tokens or row["num_routed"] != 0:
            raise RuntimeError("An inactive timestep did not use the full token sequence")
        if row["mask_draw_index"] is not None or row["mask_sha256"] is not None:
            raise RuntimeError("An inactive timestep unexpectedly sampled a mask")
    return active


def validate_against_baseline(
    metadata, baseline_metadata, *, route_step, keep_ratio
):
    if metadata["initial_noise_sha256"] != baseline_metadata["initial_noise_sha256"]:
        raise RuntimeError("Initial noise differs from the paired baseline")
    candidate = state_hashes(metadata)
    baseline = state_hashes(baseline_metadata)
    if candidate.keys() != baseline.keys():
        raise RuntimeError("Candidate and baseline do not contain identical state steps")
    mismatched_pre_route = [
        step for step in candidate
        if step >= route_step and candidate[step] != baseline[step]
    ]
    if mismatched_pre_route:
        raise RuntimeError(
            f"States before routing differ from baseline: {sorted(mismatched_pre_route)}"
        )
    if math.isclose(keep_ratio, 1.0):
        mismatched = [step for step in candidate if candidate[step] != baseline[step]]
        if mismatched or metadata["final_sha256"] != baseline_metadata["final_sha256"]:
            raise RuntimeError(
                "The ratio=1 control is not bit-identical to baseline; "
                f"mismatched_steps={sorted(mismatched)}"
            )


def add_trajectory_context(
    comparison, *, route_step, keep_ratio, mask_seed, num_kept, num_tokens,
    total_steps
):
    rows = []
    immediate_step = route_step - 1
    for metric in comparison:
        observation_step = int(metric["step"])
        if observation_step >= route_step:
            phase = "pre_routing"
        elif observation_step == immediate_step:
            phase = "immediate"
        else:
            phase = "downstream"
        rows.append({
            "route_step": route_step,
            "route_t": route_step / total_steps,
            "keep_ratio": keep_ratio,
            "masked_ratio": 1.0 - keep_ratio,
            "num_kept": num_kept,
            "num_masked": num_tokens - num_kept,
            "mask_seed": mask_seed,
            "observation_step": observation_step,
            "observation_t": float(metric.get("t", observation_step / total_steps)),
            "phase": phase,
            "steps_after_routing": immediate_step - observation_step,
            "dino_similarity": metric["dino_similarity"],
            "lpips_distance": metric["lpips_distance"],
        })
    return rows


def summary_by_ratio(final_rows):
    grouped = defaultdict(list)
    for row in final_rows:
        grouped[row["keep_ratio"]].append(row)
    result = []
    for keep_ratio in sorted(grouped, reverse=True):
        rows = grouped[keep_ratio]
        entry = {
            "keep_ratio": keep_ratio,
            "masked_ratio": 1.0 - keep_ratio,
            "num_route_steps": len(rows),
        }
        for prefix in ("immediate", "final"):
            for metric in METRICS:
                key = f"{prefix}_{metric}"
                values = [row[key] for row in rows]
                entry[key] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values),
                    "min": min(values),
                    "max": max(values),
                }
        result.append(entry)
    return result


def load_or_run_trial(
    *,
    model,
    vae,
    comparator,
    device,
    trial_root,
    baseline_dir,
    baseline_metadata,
    fixed,
    route_step,
    keep_ratio,
    mask_seed,
    num_tokens,
    fixed_parameter_sha256,
    overwrite,
):
    sample_dir = sample_subdir(trial_root, fixed["class_id"], fixed["noise_seed"])
    trial_manifest_path = trial_root / "manifest.json"
    metric_path = sample_dir / "image_metrics_to_baseline.json"

    if overwrite and trial_root.exists():
        shutil.rmtree(trial_root)
    complete = trial_manifest_path.exists() and metric_path.exists()
    if complete:
        trial_manifest = read_json(trial_manifest_path)
        if trial_manifest["fixed_parameter_sha256"] != fixed_parameter_sha256:
            raise RuntimeError(
                f"Completed trial has incompatible fixed parameters: {trial_root}. "
                "Use --overwrite to regenerate it."
            )
        if trial_manifest["route_step"] != route_step or not math.isclose(
            trial_manifest["keep_ratio"], keep_ratio
        ):
            raise RuntimeError(f"Completed trial metadata disagrees with path: {trial_root}")
        metadata = read_json(sample_dir / "metadata.json")
        comparison = read_json(metric_path)
        resumed = True
    else:
        if trial_root.exists():
            print(f"Discarding incomplete trial: {trial_root}", flush=True)
            shutil.rmtree(trial_root)
        schedule = RoutingSchedule(
            mode="single_timestep",
            keep_ratio=keep_ratio,
            mask_seed=mask_seed,
            total_steps=fixed["sample_steps"],
            num_tokens=num_tokens,
            route_step=route_step,
        )
        sample_dir, metadata = run_and_save(
            model=model,
            vae=vae,
            device=device,
            output_dir=trial_root,
            class_id=fixed["class_id"],
            noise_seed=fixed["noise_seed"],
            schedule=schedule,
            steps=fixed["sample_steps"],
            cfg_scale=fixed["cfg_scale"],
            solver=fixed["solver"],
            save_every=fixed["save_every"],
            decode_every=fixed["decode_every"],
        )
        comparison = comparator.compare_sample_dirs(baseline_dir, sample_dir)
        write_json(comparison, metric_path)
        resumed = False

    active = validate_mask_history(
        metadata,
        steps=fixed["sample_steps"],
        route_step=route_step,
        keep_ratio=keep_ratio,
        num_tokens=num_tokens,
    )
    validate_against_baseline(
        metadata,
        baseline_metadata,
        route_step=route_step,
        keep_ratio=keep_ratio,
    )
    expected_steps = set(range(fixed["sample_steps"] + 1))
    if {int(row["step"]) for row in comparison} != expected_steps:
        raise RuntimeError("DINO/LPIPS comparison is missing trajectory steps")
    validate_saved_artifacts(sample_dir, expected_steps)
    if any(
        row.get("dino_model") != comparator.dino_model
        or row.get("lpips_net") != comparator.lpips_net
        for row in comparison
    ):
        raise RuntimeError("Saved metrics were computed with different DINO/LPIPS models")
    candidate_images_sha256 = image_sequence_sha256(sample_dir)
    image_metrics_sha256 = file_sha256(metric_path)
    if complete:
        if trial_manifest.get("candidate_images_sha256") != candidate_images_sha256:
            raise RuntimeError(
                f"Decoded images changed after metrics were computed: {trial_root}. "
                "Use --overwrite to regenerate the trial."
            )
        if trial_manifest.get("image_metrics_sha256") != image_metrics_sha256:
            raise RuntimeError(
                f"Metric JSON changed after the trial completed: {trial_root}. "
                "Use --overwrite to regenerate the trial."
            )

    immediate_step = route_step - 1
    immediate = next(row for row in comparison if int(row["step"]) == immediate_step)
    final = next(row for row in comparison if int(row["step"]) == 0)
    final_row = {
        "route_step": route_step,
        "route_t": route_step / fixed["sample_steps"],
        "keep_ratio": keep_ratio,
        "masked_ratio": 1.0 - keep_ratio,
        "num_kept": active["num_kept"],
        "num_masked": active["num_routed"],
        "mask_seed": mask_seed,
        "immediate_step": immediate_step,
        "immediate_dino_similarity": immediate["dino_similarity"],
        "immediate_lpips_distance": immediate["lpips_distance"],
        "final_dino_similarity": final["dino_similarity"],
        "final_lpips_distance": final["lpips_distance"],
        "active_mask_sha256": active["mask_sha256"],
        "final_sha256": metadata["final_sha256"],
        "candidate_images_sha256": candidate_images_sha256,
        "image_metrics_sha256": image_metrics_sha256,
        "sample_dir": str(sample_dir.relative_to(PROJECT_ROOT)),
    }
    trajectory_rows = add_trajectory_context(
        comparison,
        route_step=route_step,
        keep_ratio=keep_ratio,
        mask_seed=mask_seed,
        num_kept=active["num_kept"],
        num_tokens=num_tokens,
        total_steps=fixed["sample_steps"],
    )
    trial_manifest = {
        "experiment": "setup_2_single_timestep",
        "fixed_parameter_sha256": fixed_parameter_sha256,
        "only_variables": {
            "route_step": route_step,
            "keep_ratio": keep_ratio,
        },
        "route_step": route_step,
        "affected_state_step": immediate_step,
        "keep_ratio": keep_ratio,
        "masked_ratio": 1.0 - keep_ratio,
        "mask_seed": mask_seed,
        "initial_noise_sha256": metadata["initial_noise_sha256"],
        "active_mask_sha256": active["mask_sha256"],
        "final_sha256": metadata["final_sha256"],
        "candidate_images_sha256": candidate_images_sha256,
        "image_metrics_sha256": image_metrics_sha256,
        "sample_dir": str(sample_dir.relative_to(PROJECT_ROOT)),
        "immediate_metrics": {
            "dino_similarity": immediate["dino_similarity"],
            "lpips_distance": immediate["lpips_distance"],
        },
        "final_metrics": {
            "dino_similarity": final["dino_similarity"],
            "lpips_distance": final["lpips_distance"],
        },
    }
    write_json(trial_manifest, trial_manifest_path)
    return final_row, trajectory_rows, resumed


def main():
    args = parse_args()
    config = read_json(args.config)
    (
        fixed,
        design,
        configured_steps,
        configured_ratios,
        route_steps,
        keep_ratios,
    ) = validate_design(config, args)
    metrics_config = config["metrics"]
    mask_seed = int(design["mask_seed"])

    baseline_root = STAGE_DIR / "baseline" / "results"
    baseline_manifest_path = baseline_root / "manifest.json"
    if not baseline_manifest_path.exists():
        raise FileNotFoundError("Run baseline/run.py before Setup 2")
    baseline_manifest = read_json(baseline_manifest_path)
    if baseline_manifest["fixed_generation"] != fixed:
        raise ValueError("Baseline and Setup 2 fixed-generation parameters differ")
    generation_device = args.device or baseline_manifest["runtime"]["device"]
    if baseline_manifest["runtime"]["device"] != str(generation_device):
        raise ValueError("Baseline and Setup 2 must use the same generation device")

    checkpoint_path = (PROJECT_ROOT / fixed["checkpoint"]).resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != baseline_manifest["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint bytes differ from the baseline checkpoint")
    model, checkpoint_args, checkpoint_step = load_checkpoint(
        checkpoint_path, generation_device, fixed["checkpoint_weights"])
    num_tokens = int(model.pos_embed.shape[1])
    training_keep_ratio = 1.0 - float(checkpoint_args.selection_ratio)
    if not any(math.isclose(ratio, training_keep_ratio) for ratio in configured_ratios):
        raise ValueError(
            "The ratio sweep must include the checkpoint training keep_ratio "
            f"{training_keep_ratio}"
        )
    if min(configured_ratios) < training_keep_ratio and not math.isclose(
        min(configured_ratios), training_keep_ratio
    ):
        raise ValueError(
            "Configured keep_ratios must not extrapolate below the checkpoint "
            f"training value {training_keep_ratio}"
        )

    baseline_dir = sample_subdir(
        baseline_root, fixed["class_id"], fixed["noise_seed"])
    baseline_metadata = read_json(baseline_dir / "metadata.json")
    if int(baseline_metadata["steps"]) != int(fixed["sample_steps"]):
        raise ValueError("Baseline was generated with a different number of steps")
    expected_state_steps = set(range(int(fixed["sample_steps"]) + 1))
    if set(state_hashes(baseline_metadata)) != expected_state_steps:
        raise ValueError("Baseline does not contain every saved latent state N..0")
    baseline_image_steps = {
        int(path.stem.removeprefix("step_"))
        for path in (baseline_dir / "images").glob("step_*.png")
    }
    if baseline_image_steps != expected_state_steps:
        raise ValueError("Baseline does not contain every decoded image N..0")
    baseline_images_sha256 = image_sequence_sha256(baseline_dir)

    fixed_payload = {
        "fixed_generation": fixed,
        "generation_device": str(generation_device),
        "model_dtype": str(next(model.parameters()).dtype),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint_step,
        "route_start": checkpoint_args.start_layer_idx,
        "route_end": checkpoint_args.end_layer_idx,
        "mask_seed": mask_seed,
        "metrics": metrics_config,
        "metrics_device": str(args.metrics_device),
        "torch_version": torch.__version__,
        "baseline_initial_noise_sha256": baseline_metadata["initial_noise_sha256"],
        "baseline_final_sha256": baseline_metadata["final_sha256"],
        "baseline_images_sha256": baseline_images_sha256,
    }
    fixed_parameter_sha256 = hashlib.sha256(
        json.dumps(fixed_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    vae = load_vae(fixed["vae"], generation_device)
    comparator = ImageMetricComparator(
        metrics_config["dino_model"],
        args.metrics_device,
        metrics_config["batch_size"],
        metrics_config["lpips_net"],
    )

    output_root = SETUP_DIR / "results"
    comparison_root = SETUP_DIR / "comparison_results"
    final_rows = []
    trajectory_rows = []
    resumed_count = 0
    total = len(route_steps) * len(keep_ratios)
    trial_index = 0
    for route_step in route_steps:
        for keep_ratio in keep_ratios:
            trial_index += 1
            trial_root = (
                output_root
                / f"route_step_{route_step:04d}"
                / f"keep_ratio_{ratio_slug(keep_ratio)}"
            )
            print(
                f"[{trial_index}/{total}] route_step={route_step}, "
                f"keep_ratio={keep_ratio:.3f}",
                flush=True,
            )
            final_row, trial_trajectory, resumed = load_or_run_trial(
                model=model,
                vae=vae,
                comparator=comparator,
                device=generation_device,
                trial_root=trial_root,
                baseline_dir=baseline_dir,
                baseline_metadata=baseline_metadata,
                fixed=fixed,
                route_step=route_step,
                keep_ratio=keep_ratio,
                mask_seed=mask_seed,
                num_tokens=num_tokens,
                fixed_parameter_sha256=fixed_parameter_sha256,
                overwrite=args.overwrite,
            )
            final_rows.append(final_row)
            trajectory_rows.extend(trial_trajectory)
            resumed_count += int(resumed)

    control_rows = [row for row in final_rows if math.isclose(row["keep_ratio"], 1.0)]
    control_exact = bool(control_rows) and all(
        row["final_sha256"] == baseline_metadata["final_sha256"]
        for row in control_rows
    )
    mask_hashes_by_ratio = {}
    for keep_ratio in keep_ratios:
        hashes = {
            row["active_mask_sha256"]
            for row in final_rows
            if math.isclose(row["keep_ratio"], keep_ratio)
        }
        mask_hashes_by_ratio[str(keep_ratio)] = sorted(hashes)
        if len(hashes) != 1:
            raise RuntimeError(
                f"The active mask changed across route steps for keep_ratio={keep_ratio}"
            )

    analysis = {
        "comparison": "setup_2_single_timestep_vs_unmasked_baseline",
        "design": {
            "route_steps": route_steps,
            "keep_ratios": keep_ratios,
            "masked_ratios": [1.0 - ratio for ratio in keep_ratios],
            "mask_seed": mask_seed,
            "num_trials": total,
            "checkpoint_training_keep_ratio": training_keep_ratio,
            "grid_complete": (
                route_steps == configured_steps and keep_ratios == configured_ratios
            ),
            "ratio_semantics": "keep_ratio; 1.0 means all tokens and no masking",
            "timestep_semantics": (
                "routing at step k changes transition k->k-1; "
                "the immediate observation is state k-1"
            ),
        },
        "metric_directions": {
            "dino_similarity": "higher_is_closer",
            "lpips_distance": "lower_is_closer",
        },
        "fixed_parameter_sha256": fixed_parameter_sha256,
        "control_ratio_one": {
            "num_trials": len(control_rows),
            "all_final_hashes_equal_baseline": control_exact,
        },
        "active_mask_hashes_by_ratio": mask_hashes_by_ratio,
        "summary_by_keep_ratio": summary_by_ratio(final_rows),
        "trials": final_rows,
    }
    write_json(analysis, comparison_root / "analysis.json")
    write_csv(final_rows, comparison_root / "final_metrics.csv")
    write_csv(trajectory_rows, comparison_root / "trajectory_metrics.csv")

    manifest = {
        "experiment": "setup_2_single_timestep",
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "fixed_generation": fixed,
        "routing": {
            "mode": "single_timestep",
            "route_steps": route_steps,
            "keep_ratios": keep_ratios,
            "mask_seed": mask_seed,
            "checkpoint_training_keep_ratio": training_keep_ratio,
            "only_variables": ["route_step", "keep_ratio"],
        },
        "runtime": {
            "generation_device": str(generation_device),
            "metrics_device": str(args.metrics_device),
            "model_dtype": str(next(model.parameters()).dtype),
            "torch_version": torch.__version__,
            "resumed_trials": resumed_count,
            "generated_trials": total - resumed_count,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "weights": fixed["checkpoint_weights"],
            "step": checkpoint_step,
            "args": vars(checkpoint_args),
        },
        "baseline_manifest": str(baseline_manifest_path.relative_to(PROJECT_ROOT)),
        "fixed_parameter_sha256": fixed_parameter_sha256,
        "analysis": str((comparison_root / "analysis.json").relative_to(PROJECT_ROOT)),
    }
    write_json(manifest, output_root / "manifest.json")
    print(
        f"Finished {total} paired trials ({resumed_count} resumed): {output_root}",
        flush=True,
    )
    print(f"DINO/LPIPS analysis: {comparison_root}", flush=True)


if __name__ == "__main__":
    main()
