#!/usr/bin/env python3
"""Run paired TREAD inference while varying only the mask seed."""

import argparse
import csv
import hashlib
import json
import math
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
    default_device,
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
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--metrics-device", default="cpu")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sequence_sha256(mask_history):
    digest = hashlib.sha256()
    for row in mask_history:
        digest.update(bytes.fromhex(row["mask_sha256"]))
    return digest.hexdigest()


def validate_mask_history(metadata, *, steps, keep_ratio, num_tokens):
    history = metadata["mask_history"]
    if len(history) != steps:
        raise RuntimeError(f"Expected {steps} masks, got {len(history)}")
    if [row["step"] for row in history] != list(range(steps, 0, -1)):
        raise RuntimeError("Mask history does not cover every Euler timestep")
    if [row["mask_draw_index"] for row in history] != list(range(1, steps + 1)):
        raise RuntimeError("A new mask draw was not performed at every timestep")
    expected_kept = max(1, int(num_tokens * keep_ratio))
    for row in history:
        if not row["routing_active"]:
            raise RuntimeError("Routing must be active for the complete trajectory")
        if row["num_kept"] != expected_kept:
            raise RuntimeError("Unexpected number of processed tokens")
        if not math.isclose(row["keep_ratio"], keep_ratio):
            raise RuntimeError("keep_ratio changed inside a trajectory")
    return sequence_sha256(history)


def metric_stats(rows):
    result = {}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        result[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def trajectory_summary(comparisons):
    grouped = defaultdict(list)
    for rows in comparisons:
        for row in rows:
            grouped[row["step"]].append(row)
    summary = []
    for step in sorted(grouped, reverse=True):
        rows = grouped[step]
        summary.append({
            "step": step,
            "num_mask_seeds": len(rows),
            "mean_dino_similarity": statistics.fmean(
                row["dino_similarity"] for row in rows),
            "std_dino_similarity": statistics.pstdev(
                row["dino_similarity"] for row in rows),
            "mean_lpips_distance": statistics.fmean(
                row["lpips_distance"] for row in rows),
            "std_lpips_distance": statistics.pstdev(
                row["lpips_distance"] for row in rows),
        })
    return summary


def main():
    args = parse_args()
    config = read_json(args.config)
    fixed = config["fixed_generation"]
    tread = config["tread"]
    metrics_config = config["metrics"]
    mask_seeds = tread["mask_seeds"]
    if len(mask_seeds) < 2 or len(mask_seeds) != len(set(mask_seeds)):
        raise ValueError("mask_seeds must contain at least two unique values")
    if fixed["solver"] != "euler":
        raise ValueError("Setup 1 is fixed to the Euler ODE solver")
    if fixed["save_every"] != 1 or fixed["decode_every"] != 1:
        raise ValueError("Every diffusion state and image must be saved")

    baseline_root = STAGE_DIR / "baseline" / "results"
    baseline_manifest_path = baseline_root / "manifest.json"
    if not baseline_manifest_path.exists():
        raise FileNotFoundError("Run baseline/run.py before Setup 1")
    baseline_manifest = read_json(baseline_manifest_path)
    if baseline_manifest["fixed_generation"] != fixed:
        raise ValueError("Baseline and Setup 1 fixed-generation parameters differ")
    if baseline_manifest["runtime"]["device"] != str(args.device):
        raise ValueError("Baseline and Setup 1 must use the same generation device")

    checkpoint_path = (PROJECT_ROOT / fixed["checkpoint"]).resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != baseline_manifest["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint bytes differ from the baseline checkpoint")
    model, checkpoint_args, checkpoint_step = load_checkpoint(
        checkpoint_path, args.device, fixed["checkpoint_weights"])
    expected_keep_ratio = 1.0 - float(checkpoint_args.selection_ratio)
    keep_ratio = float(tread["keep_ratio"])
    if not math.isclose(keep_ratio, expected_keep_ratio):
        raise ValueError(
            "keep_ratio must match checkpoint training: "
            f"expected {expected_keep_ratio}, got {keep_ratio}"
        )

    vae = load_vae(fixed["vae"], args.device)
    comparator = ImageMetricComparator(
        metrics_config["dino_model"],
        args.metrics_device,
        metrics_config["batch_size"],
        metrics_config["lpips_net"],
    )
    baseline_dir = sample_subdir(
        baseline_root, fixed["class_id"], fixed["noise_seed"])
    baseline_metadata = read_json(baseline_dir / "metadata.json")
    baseline_noise_hash = baseline_metadata["initial_noise_sha256"]

    fixed_payload = {
        "fixed_generation": fixed,
        "generation_device": str(args.device),
        "model_dtype": str(next(model.parameters()).dtype),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint_step,
        "route_start": checkpoint_args.start_layer_idx,
        "route_end": checkpoint_args.end_layer_idx,
        "keep_ratio": keep_ratio,
    }
    fixed_parameter_sha256 = hashlib.sha256(
        json.dumps(fixed_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    output_root = SETUP_DIR / "results"
    trials = []
    final_rows = []
    comparisons = []
    for mask_seed in mask_seeds:
        trial_name = f"mask_seed_{mask_seed:04d}"
        trial_root = output_root / trial_name
        print(f"Starting {trial_name}", flush=True)
        schedule = RoutingSchedule(
            mode="resample",
            keep_ratio=keep_ratio,
            mask_seed=mask_seed,
            total_steps=fixed["sample_steps"],
            num_tokens=model.pos_embed.shape[1],
        )
        sample_dir, metadata = run_and_save(
            model=model,
            vae=vae,
            device=args.device,
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
        if metadata["initial_noise_sha256"] != baseline_noise_hash:
            raise RuntimeError("Initial noise differs from the paired baseline")
        mask_sequence_hash = validate_mask_history(
            metadata,
            steps=fixed["sample_steps"],
            keep_ratio=keep_ratio,
            num_tokens=model.pos_embed.shape[1],
        )
        image_comparison = comparator.compare_sample_dirs(baseline_dir, sample_dir)
        expected_image_steps = set(range(fixed["sample_steps"] + 1))
        if {row["step"] for row in image_comparison} != expected_image_steps:
            raise RuntimeError("DINO/LPIPS comparison is missing diffusion steps")
        comparisons.append(image_comparison)
        write_json(
            image_comparison,
            sample_dir / "image_metrics_to_baseline.json",
        )
        final_metrics = next(row for row in image_comparison if row["step"] == 0)
        trial = {
            "trial": trial_name,
            "only_variable": {"mask_seed": mask_seed},
            "fixed_parameter_sha256": fixed_parameter_sha256,
            "sample_dir": str(sample_dir.relative_to(PROJECT_ROOT)),
            "initial_noise_sha256": metadata["initial_noise_sha256"],
            "mask_sequence_sha256": mask_sequence_hash,
            "num_unique_masks": len({
                row["mask_sha256"] for row in metadata["mask_history"]
            }),
            "final_sha256": metadata["final_sha256"],
            "final_dino_similarity": final_metrics["dino_similarity"],
            "final_lpips_distance": final_metrics["lpips_distance"],
        }
        write_json(trial, trial_root / "manifest.json")
        trials.append(trial)
        final_rows.append({
            "mask_seed": mask_seed,
            "dino_similarity": final_metrics["dino_similarity"],
            "lpips_distance": final_metrics["lpips_distance"],
            "mask_sequence_sha256": mask_sequence_hash,
            "final_sha256": metadata["final_sha256"],
        })

    trajectory = trajectory_summary(comparisons)
    analysis = {
        "comparison": "setup_1_resampled_masks_vs_unmasked_baseline",
        "num_mask_seeds": len(mask_seeds),
        "only_variable": "mask_seed",
        "fixed_parameter_sha256": fixed_parameter_sha256,
        "metric_directions": {
            "dino_similarity": "higher_is_closer",
            "lpips_distance": "lower_is_closer",
        },
        "final_per_mask_seed": final_rows,
        "final_summary": metric_stats(final_rows),
        "trajectory_summary": trajectory,
    }
    analysis_root = SETUP_DIR / "comparison_results"
    write_json(analysis, analysis_root / "analysis.json")
    write_csv(final_rows, analysis_root / "final_metrics.csv")
    write_csv(trajectory, analysis_root / "trajectory_metrics.csv")

    manifest = {
        "experiment": "setup_1_resampled_masks",
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "fixed_generation": fixed,
        "routing": {
            "mode": "new_random_mask_at_every_timestep",
            "keep_ratio": keep_ratio,
            "selection_ratio": 1.0 - keep_ratio,
            "mask_seeds": mask_seeds,
            "only_variable": "mask_seed",
        },
        "runtime": {
            "generation_device": str(args.device),
            "metrics_device": str(args.metrics_device),
            "model_dtype": str(next(model.parameters()).dtype),
            "torch_version": torch.__version__,
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
        "analysis": str((analysis_root / "analysis.json").relative_to(PROJECT_ROOT)),
        "trials": trials,
    }
    write_json(manifest, output_root / "manifest.json")
    print(f"Finished {len(trials)} paired TREAD runs: {output_root}")
    print(f"DINO/LPIPS analysis: {analysis_root}")


if __name__ == "__main__":
    main()
