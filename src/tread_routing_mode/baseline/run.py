#!/usr/bin/env python3
"""Create the shared unmasked reference for all routing experiments."""

import argparse
import json
import sys
from pathlib import Path

import torch

BASELINE_DIR = Path(__file__).resolve().parent
STAGE_DIR = BASELINE_DIR.parent
PROJECT_ROOT = STAGE_DIR.parents[1]
DEFAULT_CONFIG = STAGE_DIR / "config.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tread_routing_mode.common.artifacts import file_sha256, write_json
from tread_routing_mode.common.checkpoint import load_checkpoint
from tread_routing_mode.common.experiment import default_device, load_vae, run_and_save
from tread_routing_mode.common.routing import RoutingSchedule


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def validate_fixed_generation(fixed):
    if fixed["solver"] != "euler":
        raise ValueError("This experiment is fixed to the Euler ODE solver")
    if fixed["save_every"] != 1 or fixed["decode_every"] != 1:
        raise ValueError("Every diffusion state and image must be saved")
    if fixed["sample_steps"] <= 0:
        raise ValueError("sample_steps must be positive")


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_generation"]
    validate_fixed_generation(fixed)

    checkpoint_path = (PROJECT_ROOT / fixed["checkpoint"]).resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    model, checkpoint_args, checkpoint_step = load_checkpoint(
        checkpoint_path, args.device, fixed["checkpoint_weights"])
    vae = load_vae(fixed["vae"], args.device)

    schedule = RoutingSchedule(
        mode="baseline",
        keep_ratio=1.0,
        mask_seed=0,
        total_steps=fixed["sample_steps"],
        num_tokens=model.pos_embed.shape[1],
    )
    output_root = BASELINE_DIR / "results"
    sample_dir, metadata = run_and_save(
        model=model,
        vae=vae,
        device=args.device,
        output_dir=output_root,
        class_id=fixed["class_id"],
        noise_seed=fixed["noise_seed"],
        schedule=schedule,
        steps=fixed["sample_steps"],
        cfg_scale=fixed["cfg_scale"],
        solver=fixed["solver"],
        save_every=fixed["save_every"],
        decode_every=fixed["decode_every"],
    )
    if len(metadata["mask_history"]) != fixed["sample_steps"]:
        raise RuntimeError("Baseline did not record every diffusion step")
    if any(row["routing_active"] for row in metadata["mask_history"]):
        raise RuntimeError("Routing unexpectedly activated in the baseline")

    manifest = {
        "experiment": "unmasked_baseline",
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "fixed_generation": fixed,
        "runtime": {
            "device": str(args.device),
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
        "reference": {
            "sample_dir": str(sample_dir.relative_to(PROJECT_ROOT)),
            "class_id": fixed["class_id"],
            "noise_seed": fixed["noise_seed"],
            "keep_ratio": 1.0,
            "initial_noise_sha256": metadata["initial_noise_sha256"],
            "final_sha256": metadata["final_sha256"],
        },
    }
    write_json(manifest, output_root / "manifest.json")
    print(f"Baseline created: {sample_dir}")


if __name__ == "__main__":
    main()
