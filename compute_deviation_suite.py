"""Compare TREAD Setup 1/2 outputs against a paired dense baseline.

The evaluator discovers run directories created by ``run_inference_suite.py``
and strictly pairs tensor shards by sample ID, seed, and class. It computes
trajectory and final-output deviations in latent space, decoded pixel space,
and optionally local CLIP/DINOv2 feature spaces.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def discover_runs(suite_root, baseline_dir=None, explicit_targets=None):
    suite_root = Path(suite_root).expanduser().resolve()
    baseline = (
        Path(baseline_dir).expanduser().resolve()
        if baseline_dir
        else suite_root / "baseline"
    )
    if not (baseline / "run_manifest.json").is_file():
        raise FileNotFoundError(
            f"Baseline run_manifest.json not found under {baseline}"
        )

    if explicit_targets:
        targets = [Path(value).expanduser().resolve() for value in explicit_targets]
    else:
        targets = []
        for manifest_path in suite_root.rglob("run_manifest.json"):
            run_dir = manifest_path.parent.resolve()
            if run_dir == baseline.resolve():
                continue
            manifest = read_json(manifest_path)
            if manifest.get("experiment") in {"every-step", "single-step"}:
                targets.append(run_dir)
    targets = sorted(
        set(targets),
        key=lambda path: (
            read_json(path / "run_manifest.json").get("experiment", ""),
            float(read_json(path / "run_manifest.json").get("selection_ratio", 0)),
            str(read_json(path / "run_manifest.json").get("routing_steps", "")),
            str(path),
        ),
    )
    if not targets:
        raise FileNotFoundError("No Setup 1 or Setup 2 run directories were found")
    return suite_root, baseline.resolve(), targets


COMPATIBILITY_KEYS = (
    "checkpoint",
    "checkpoint_state_key",
    "seed_pair_sha256",
    "selected_sample_ids",
    "selected_count",
    "model",
    "resolution",
    "path_type",
    "prediction",
    "num_steps",
    "solver",
    "cfg_scale",
    "guidance_interval",
    "mask_seed",
    "route",
)


def validate_manifests(baseline_manifest, target_manifest, target_dir):
    mismatches = []
    for key in COMPATIBILITY_KEYS:
        if baseline_manifest.get(key) != target_manifest.get(key):
            mismatches.append(
                f"{key}: baseline={baseline_manifest.get(key)!r}, "
                f"target={target_manifest.get(key)!r}"
            )
    if mismatches:
        raise ValueError(
            f"Run {target_dir} is not paired with the baseline:\n  "
            + "\n  ".join(mismatches)
        )


def shard_map(run_dir):
    shard_dir = Path(run_dir) / "tensor_shards"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"tensor_shards directory missing: {shard_dir}")
    shards = {path.name: path for path in shard_dir.glob("shard_*.pt")}
    if not shards:
        raise FileNotFoundError(f"No tensor shards found in {shard_dir}")
    return shards


def paired_shards(baseline_dir, target_dir, allow_incomplete=False):
    baseline = shard_map(baseline_dir)
    target = shard_map(target_dir)
    baseline_names = set(baseline)
    target_names = set(target)
    if baseline_names != target_names and not allow_incomplete:
        raise ValueError(
            f"Shard mismatch for {target_dir}: "
            f"missing target={sorted(baseline_names-target_names)}, "
            f"extra target={sorted(target_names-baseline_names)}"
        )
    common = sorted(baseline_names & target_names)
    if not common:
        raise ValueError(f"No paired shards for {target_dir}")
    return [(name, baseline[name], target[name]) for name in common]


def validate_shard_pair(reference, target, name):
    for key in ("sample_id", "seed", "class_idx"):
        if key not in reference or key not in target:
            raise KeyError(f"{name}: required key {key!r} is missing")
        if not torch.equal(reference[key], target[key]):
            raise ValueError(f"{name}: {key} differs from the baseline")
    for key in ("time_steps", "x0_prediction_times"):
        if key in reference and key in target:
            ref_value, target_value = reference[key], target[key]
            if ref_value is None and target_value is None:
                continue
            if ref_value is None or target_value is None or not torch.equal(
                ref_value, target_value
            ):
                raise ValueError(f"{name}: {key} differs from the baseline")


def add_step_axis(tensor):
    return tensor if tensor.ndim >= 5 else tensor.unsqueeze(1)


def numeric_metrics(reference, target, image_range=False):
    reference = reference.float()
    target = target.float()
    if image_range:
        reference = reference / 255.0
        target = target / 255.0
    if reference.shape != target.shape:
        raise ValueError(
            f"Tensor shapes differ: baseline={reference.shape}, target={target.shape}"
        )
    if reference.ndim < 3:
        raise ValueError("Expected [batch, step, ...] tensors")
    reduce_dims = tuple(range(2, reference.ndim))
    difference = target - reference
    mse = difference.square().mean(dim=reduce_dims)
    rmse = mse.sqrt()
    mae = difference.abs().mean(dim=reduce_dims)
    flat_reference = reference.flatten(2)
    flat_target = target.flatten(2)
    flat_difference = difference.flatten(2)
    reference_norm = torch.linalg.vector_norm(flat_reference, dim=-1)
    relative_l2 = torch.linalg.vector_norm(flat_difference, dim=-1) / reference_norm.clamp_min(1e-12)
    cosine_similarity = torch.nn.functional.cosine_similarity(
        flat_reference, flat_target, dim=-1, eps=1e-12
    )
    result = {
        "mae": mae,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "cosine_distance": 1.0 - cosine_similarity,
    }
    if image_range:
        result["psnr"] = -20.0 * torch.log10(rmse.clamp_min(1e-12))
    return result


def feature_metrics(reference, target):
    if reference.shape != target.shape:
        raise ValueError("Feature tensor shapes differ")
    reference = torch.nn.functional.normalize(reference.float(), dim=-1)
    target = torch.nn.functional.normalize(target.float(), dim=-1)
    return {
        "cosine_distance": 1.0 - (reference * target).sum(dim=-1),
        "normalized_l2": torch.linalg.vector_norm(target - reference, dim=-1),
    }


class LocalImageEncoder:
    def __init__(self, kind, model_path, device, dtype, batch_size):
        try:
            from transformers import AutoImageProcessor, AutoModel
            from transformers import CLIPVisionModelWithProjection
        except ImportError as error:
            raise ImportError(
                "CLIP/DINO metrics require the transformers package"
            ) from error
        self.kind = kind
        self.name = kind
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.batch_size = int(batch_size)
        self.processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        if kind == "clip":
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                model_path, local_files_only=True
            )
        elif kind == "dinov2":
            self.model = AutoModel.from_pretrained(
                model_path, local_files_only=True
            )
        else:
            raise ValueError(f"Unknown feature encoder: {kind}")
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def encode(self, images):
        prefix = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:]).cpu()
        outputs = []
        for start in range(0, flat.shape[0], self.batch_size):
            batch = flat[start : start + self.batch_size]
            arrays = [image.permute(1, 2, 0).numpy() for image in batch]
            inputs = self.processor(images=arrays, return_tensors="pt")
            moved = {}
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    moved[key] = value.to(self.device, dtype=self.dtype)
                else:
                    moved[key] = value.to(self.device)
            model_output = self.model(**moved)
            if self.kind == "clip":
                features = model_output.image_embeds
            elif getattr(model_output, "pooler_output", None) is not None:
                features = model_output.pooler_output
            else:
                features = model_output.last_hidden_state[:, 0]
            outputs.append(features.float().cpu())
        features = torch.cat(outputs, dim=0)
        return features.reshape(*prefix, features.shape[-1])


def make_feature_encoders(args):
    if not args.dinov2_model and not args.clip_model:
        return []
    device = args.feature_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.feature_dtype]
    encoders = []
    if args.dinov2_model:
        encoders.append(
            LocalImageEncoder(
                "dinov2",
                args.dinov2_model,
                device,
                dtype,
                args.feature_batch_size,
            )
        )
    if args.clip_model:
        encoders.append(
            LocalImageEncoder(
                "clip",
                args.clip_model,
                device,
                dtype,
                args.feature_batch_size,
            )
        )
    return encoders


def append_metrics(destination, modality, metrics):
    for metric_name, values in metrics.items():
        destination[modality][metric_name].append(values.cpu())


def summarize_values(values):
    return {
        "mean": values.mean(dim=0),
        "std": values.std(dim=0, unbiased=False),
        "median": values.median(dim=0).values,
        "p95": torch.quantile(values, 0.95, dim=0),
        "max": values.max(dim=0).values,
        "count": int(values.shape[0]),
    }


def tensor_to_list(value):
    return value.tolist() if torch.is_tensor(value) else value


def safe_name(path, root):
    try:
        relative = str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        relative = str(Path(path).resolve())
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", relative)


def evaluate_run(
    args,
    suite_root,
    baseline_dir,
    target_dir,
    encoders,
    baseline_feature_cache,
):
    baseline_manifest = read_json(baseline_dir / "run_manifest.json")
    target_manifest = read_json(target_dir / "run_manifest.json")
    validate_manifests(baseline_manifest, target_manifest, target_dir)
    shard_pairs = paired_shards(
        baseline_dir, target_dir, allow_incomplete=args.allow_incomplete
    )
    collected = defaultdict(lambda: defaultdict(list))
    sample_ids, seeds, classes = [], [], []
    modality_times = {}
    solver = target_manifest.get("solver", "euler")

    for shard_name, baseline_path, target_path in shard_pairs:
        reference = safe_torch_load(baseline_path)
        target = safe_torch_load(target_path)
        validate_shard_pair(reference, target, shard_name)
        sample_ids.append(reference["sample_id"])
        seeds.append(reference["seed"])
        classes.append(reference["class_idx"])

        if "xt_trajectory" in reference and "xt_trajectory" in target:
            append_metrics(
                collected,
                "xt_latent",
                numeric_metrics(reference["xt_trajectory"], target["xt_trajectory"]),
            )
            modality_times["xt_latent"] = reference["time_steps"].float()

        if "x0_predictions" in reference and "x0_predictions" in target:
            append_metrics(
                collected,
                "x0_latent",
                numeric_metrics(reference["x0_predictions"], target["x0_predictions"]),
            )
            modality_times["x0_latent"] = reference["x0_prediction_times"].float()

        if "x0_decoded_uint8" in reference and "x0_decoded_uint8" in target:
            append_metrics(
                collected,
                "x0_image_pixel",
                numeric_metrics(
                    reference["x0_decoded_uint8"],
                    target["x0_decoded_uint8"],
                    image_range=True,
                ),
            )
            modality_times["x0_image_pixel"] = reference["x0_prediction_times"].float()

            for encoder in encoders:
                cache_key = (shard_name, "x0_decoded_uint8", encoder.name)
                if cache_key not in baseline_feature_cache:
                    baseline_feature_cache[cache_key] = encoder.encode(
                        reference["x0_decoded_uint8"]
                    )
                target_features = encoder.encode(target["x0_decoded_uint8"])
                modality = f"x0_image_{encoder.name}"
                append_metrics(
                    collected,
                    modality,
                    feature_metrics(
                        baseline_feature_cache[cache_key], target_features
                    ),
                )
                modality_times[modality] = reference["x0_prediction_times"].float()

        if "final_latents" in reference and "final_latents" in target:
            append_metrics(
                collected,
                "final_latent",
                numeric_metrics(
                    add_step_axis(reference["final_latents"]),
                    add_step_axis(target["final_latents"]),
                ),
            )
            modality_times["final_latent"] = torch.tensor([0.0])

        if "final_decoded_uint8" in reference and "final_decoded_uint8" in target:
            reference_final = add_step_axis(reference["final_decoded_uint8"])
            target_final = add_step_axis(target["final_decoded_uint8"])
            append_metrics(
                collected,
                "final_image_pixel",
                numeric_metrics(reference_final, target_final, image_range=True),
            )
            modality_times["final_image_pixel"] = torch.tensor([0.0])
            for encoder in encoders:
                cache_key = (shard_name, "final_decoded_uint8", encoder.name)
                if cache_key not in baseline_feature_cache:
                    baseline_feature_cache[cache_key] = encoder.encode(reference_final)
                target_features = encoder.encode(target_final)
                modality = f"final_image_{encoder.name}"
                append_metrics(
                    collected,
                    modality,
                    feature_metrics(
                        baseline_feature_cache[cache_key], target_features
                    ),
                )
                modality_times[modality] = torch.tensor([0.0])

        # For Euler, the saved X0 at step s corresponds to Xt[s], so the
        # guided velocity can be reconstructed exactly as (Xt-X0)/t.
        if (
            solver == "euler"
            and "xt_trajectory" in reference
            and "x0_predictions" in reference
            and "xt_trajectory" in target
            and "x0_predictions" in target
        ):
            times = reference["x0_prediction_times"].float()
            if torch.all(times > 0):
                shape = (1, times.shape[0]) + (1,) * (
                    reference["x0_predictions"].ndim - 2
                )
                expanded_time = times.reshape(shape)
                reference_v = (
                    reference["xt_trajectory"][:, :-1]
                    - reference["x0_predictions"]
                ) / expanded_time
                target_v = (
                    target["xt_trajectory"][:, :-1]
                    - target["x0_predictions"]
                ) / expanded_time
                append_metrics(
                    collected,
                    "velocity_reconstructed",
                    numeric_metrics(reference_v, target_v),
                )
                modality_times["velocity_reconstructed"] = times

    if not collected:
        raise ValueError(
            f"No comparable saved tensors were found for {target_dir}. "
            "Use save-mode paired when generating samples."
        )

    concatenated = {
        modality: {
            metric: torch.cat(chunks, dim=0)
            for metric, chunks in metrics.items()
        }
        for modality, metrics in collected.items()
    }
    statistics = {
        modality: {
            metric: summarize_values(values)
            for metric, values in metrics.items()
        }
        for modality, metrics in concatenated.items()
    }

    routing_step = target_manifest.get("routing_steps")
    sanity = {}
    if isinstance(routing_step, int):
        if "x0_latent" in statistics and routing_step > 0:
            pre = statistics["x0_latent"]["rmse"]["mean"][:routing_step]
            sanity["x0_pre_intervention_max_mean_rmse"] = float(pre.max())
        if "xt_latent" in statistics:
            pre = statistics["xt_latent"]["rmse"]["mean"][: routing_step + 1]
            sanity["xt_pre_intervention_max_mean_rmse"] = float(pre.max())
        sanity["tolerance"] = args.sanity_tolerance
        sanity["passed"] = all(
            value <= args.sanity_tolerance
            for key, value in sanity.items()
            if key.endswith("mean_rmse")
        )

    run_name = safe_name(target_dir, suite_root)
    per_sample_path = Path(args.output_dir) / "per_run" / f"{run_name}.pt"
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "tread-deviation-per-sample-v1",
            "target_dir": str(target_dir),
            "experiment": target_manifest.get("experiment"),
            "routing_step": routing_step,
            "selection_ratio": target_manifest.get("selection_ratio"),
            "sample_id": torch.cat(sample_ids),
            "seed": torch.cat(seeds),
            "class_idx": torch.cat(classes),
            "times": modality_times,
            "metrics": concatenated,
        },
        per_sample_path,
    )

    json_statistics = {
        modality: {
            metric: {
                key: tensor_to_list(value)
                for key, value in summary.items()
            }
            for metric, summary in metrics.items()
        }
        for modality, metrics in statistics.items()
    }
    return {
        "name": run_name,
        "target_dir": str(target_dir),
        "experiment": target_manifest.get("experiment"),
        "routing_step": routing_step,
        "selection_ratio": target_manifest.get("selection_ratio"),
        "mask_ratio": target_manifest.get("mask_ratio"),
        "sample_count": int(torch.cat(sample_ids).numel()),
        "per_sample_file": str(per_sample_path),
        "times": {key: value.tolist() for key, value in modality_times.items()},
        "statistics": json_statistics,
        "sanity": sanity,
    }, statistics, modality_times


def write_step_csv(path, run_summaries):
    fieldnames = [
        "run",
        "experiment",
        "routing_step",
        "selection_ratio",
        "modality",
        "eval_step",
        "time",
        "metric",
        "mean",
        "std",
        "median",
        "p95",
        "max",
        "count",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in run_summaries:
            for modality, metrics in run["statistics"].items():
                times = run["times"][modality]
                for metric, stats in metrics.items():
                    for step, time_value in enumerate(times):
                        writer.writerow(
                            {
                                "run": run["name"],
                                "experiment": run["experiment"],
                                "routing_step": run["routing_step"],
                                "selection_ratio": run["selection_ratio"],
                                "modality": modality,
                                "eval_step": (
                                    -1 if modality.startswith("final_") else step
                                ),
                                "time": time_value,
                                "metric": metric,
                                "mean": stats["mean"][step],
                                "std": stats["std"][step],
                                "median": stats["median"][step],
                                "p95": stats["p95"][step],
                                "max": stats["max"][step],
                                "count": stats["count"],
                            }
                        )


def build_setup2_matrices(run_summaries):
    groups = defaultdict(list)
    for run in run_summaries:
        if run["experiment"] == "single-step":
            groups[str(run["selection_ratio"])].append(run)
    output = {"format": "tread-setup2-deviation-matrices-v1", "groups": {}}
    for ratio, runs in groups.items():
        runs.sort(key=lambda item: int(item["routing_step"]))
        group = {
            "selection_ratio": float(ratio),
            "routing_steps": torch.tensor(
                [int(run["routing_step"]) for run in runs]
            ),
            "run_names": [run["name"] for run in runs],
            "times": {},
            "matrices": {},
        }
        common_modalities = set.intersection(
            *(set(run["statistics"]) for run in runs)
        )
        for modality in sorted(common_modalities):
            group["times"][modality] = torch.tensor(runs[0]["times"][modality])
            common_metrics = set.intersection(
                *(set(run["statistics"][modality]) for run in runs)
            )
            for metric in sorted(common_metrics):
                key = f"{modality}/{metric}/mean"
                group["matrices"][key] = torch.tensor(
                    [
                        run["statistics"][modality][metric]["mean"]
                        for run in runs
                    ]
                )
        output["groups"][ratio] = group
    return output


def run(args):
    suite_root, baseline_dir, targets = discover_runs(
        args.suite_root, args.baseline_dir, args.target
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else suite_root / "deviation"
    )
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoders = make_feature_encoders(args)
    print(f"Baseline: {baseline_dir}")
    print(f"Targets: {len(targets)}")
    print(
        "Feature encoders: "
        + (", ".join(encoder.name for encoder in encoders) or "none")
    )

    baseline_feature_cache = {}
    run_summaries = []
    for index, target_dir in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target_dir}")
        summary, _, _ = evaluate_run(
            args,
            suite_root,
            baseline_dir,
            target_dir,
            encoders,
            baseline_feature_cache,
        )
        run_summaries.append(summary)
        write_json(
            output_dir / "deviation_summary.json",
            {
                "format": "tread-deviation-summary-v1",
                "suite_root": str(suite_root),
                "baseline_dir": str(baseline_dir),
                "runs": run_summaries,
            },
        )
        if summary["sanity"] and not summary["sanity"].get("passed", True):
            print(f"  WARNING: pre-intervention sanity check failed: {summary['sanity']}")

    write_step_csv(output_dir / "step_metrics.csv", run_summaries)
    torch.save(
        build_setup2_matrices(run_summaries),
        output_dir / "setup2_matrices.pt",
    )
    write_json(
        output_dir / "deviation_manifest.json",
        {
            "format": "tread-deviation-evaluation-v1",
            "suite_root": str(suite_root),
            "baseline_dir": str(baseline_dir),
            "targets": [str(path) for path in targets],
            "dinov2_model": args.dinov2_model,
            "clip_model": args.clip_model,
            "interpretation": {
                "x0": "model-implied clean latent: Xt - t*Vpred",
                "decoded_x0": "VAE decode of predicted X0, not decode of Xt",
                "velocity": (
                    "reconstructed from Xt and X0 for Euler runs; unavailable "
                    "from current saved tensors for non-final Heun correctors"
                ),
            },
        },
    )
    print(f"Deviation evaluation complete: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Explicit target run directory; repeat as needed",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--sanity-tolerance", type=float, default=1e-6)

    parser.add_argument(
        "--dinov2-model",
        default=None,
        help="Local Hugging Face/Transformers DINOv2 model directory",
    )
    parser.add_argument(
        "--clip-model",
        default=None,
        help="Local Hugging Face/Transformers CLIP model directory",
    )
    parser.add_argument("--feature-device", default="auto")
    parser.add_argument(
        "--feature-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--feature-batch-size", type=int, default=64)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
