"""Evaluate v2 routing deviations without any intermediate-X0 metrics.

Intermediate comparisons are velocity-field comparisons only. Final outputs
are compared in latent/pixel space and optionally with local DINOv2 features
and LPIPS.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def discover(suite_root, target_names=None, baseline_override=None):
    suite_root = Path(suite_root).expanduser().resolve()
    suite = read_json(suite_root / "suite_manifest.json")
    configs = {item["name"]: item for item in suite["configurations"]}
    requested_names = set()
    requested_paths = set()
    for value in target_names or []:
        path = Path(value).expanduser()
        if path.exists():
            requested_paths.add(path.resolve())
        else:
            requested_names.add(value)
    targets = []
    for config in configs.values():
        if config["experiment"] == "baseline":
            continue
        target_dir = suite_root / config["relative_output"]
        if (
            (requested_names or requested_paths)
            and config["name"] not in requested_names
            and target_dir.resolve() not in requested_paths
        ):
            continue
        baseline_dir = (
            Path(baseline_override).expanduser().resolve()
            if baseline_override
            else suite_root / config["baseline_relative_output"]
        )
        if not (target_dir / "run_manifest.json").is_file():
            raise FileNotFoundError(f"Target run is missing: {target_dir}")
        if not (baseline_dir / "run_manifest.json").is_file():
            raise FileNotFoundError(
                f"Paired baseline {config['baseline_name']} is missing for "
                f"{config['name']}"
            )
        targets.append((config, baseline_dir, target_dir))
    if not targets:
        raise ValueError("No target runs were selected")
    return suite_root, targets


def validate_manifests(reference, target):
    paired_keys = (
        "solver",
        "num_steps",
        "cfg_scale",
        "guidance_interval",
        "checkpoint",
        "seed_pair_sha256",
        "selected_sample_ids",
        "route",
    )
    mismatches = []
    for key in paired_keys:
        expected = reference.get(key)
        if target.get(key) != expected:
            mismatches.append(f"{key}: baseline={expected!r}, target={target.get(key)!r}")
    if mismatches:
        raise ValueError("Run is not paired with its baseline:\n  " + "\n  ".join(mismatches))
    if target.get("baseline_name") != reference.get("name"):
        raise ValueError(
            f"Target expects baseline {target.get('baseline_name')!r}, but "
            f"loaded {reference.get('name')!r}"
        )


def paired_shards(baseline_dir, target_dir):
    baseline = {path.name: path for path in (baseline_dir / "tensor_shards").glob("*.pt")}
    target = {path.name: path for path in (target_dir / "tensor_shards").glob("*.pt")}
    if baseline.keys() != target.keys():
        missing_target = sorted(baseline.keys() - target.keys())
        missing_baseline = sorted(target.keys() - baseline.keys())
        raise ValueError(
            f"Shard mismatch; missing target={missing_target}, missing baseline={missing_baseline}"
        )
    if not baseline:
        raise ValueError(f"No tensor shards found under {target_dir}")
    return [(name, baseline[name], target[name]) for name in sorted(baseline)]


def validate_shards(reference, target, name):
    for key in ("sample_id", "seed", "class_idx", "time_steps"):
        if key not in reference or key not in target:
            raise ValueError(f"{name} lacks required key {key}")
        if not torch.equal(reference[key], target[key]):
            raise ValueError(f"{name} differs in {key}")


def numeric_metrics(reference, target):
    if reference.shape != target.shape:
        raise ValueError(f"Tensor shapes differ: {reference.shape} vs {target.shape}")
    reference = reference.float()
    target = target.float()
    difference = target - reference
    flat_ref = reference.flatten(2)
    flat_target = target.flatten(2)
    flat_diff = difference.flatten(2)
    return {
        "rmse": flat_diff.square().mean(-1).sqrt(),
        "mae": flat_diff.abs().mean(-1),
        "relative_l2": torch.linalg.vector_norm(flat_diff, dim=-1)
        / torch.linalg.vector_norm(flat_ref, dim=-1).clamp_min(1e-12),
        "cosine_similarity": F.cosine_similarity(
            flat_ref, flat_target, dim=-1, eps=1e-12
        ),
    }


def final_tensor_metrics(reference, target, scale=1.0):
    reference = reference.float() / scale
    target = target.float() / scale
    difference = (target - reference).flatten(1)
    ref_flat = reference.flatten(1)
    target_flat = target.flatten(1)
    rmse = difference.square().mean(-1).sqrt()
    return {
        "rmse": rmse,
        "mae": difference.abs().mean(-1),
        "relative_l2": torch.linalg.vector_norm(difference, dim=-1)
        / torch.linalg.vector_norm(ref_flat, dim=-1).clamp_min(1e-12),
        "cosine_similarity": F.cosine_similarity(
            ref_flat, target_flat, dim=-1, eps=1e-12
        ),
    }


def reconstruct_velocities(payload, solver):
    result = {}
    time_steps = payload["time_steps"].double()
    trajectory = payload.get("xt_trajectory")
    velocity_cur = payload.get("velocity_cur")
    if trajectory is not None:
        dt = (time_steps[1:] - time_steps[:-1]).float()
        shape = (1, dt.shape[0]) + (1,) * (trajectory.ndim - 2)
        velocity_mean = (trajectory[:, 1:] - trajectory[:, :-1]) / dt.reshape(shape)
        result["xt_trajectory"] = (trajectory, time_steps.float())
    else:
        velocity_mean = None

    if velocity_cur is None and solver == "euler" and velocity_mean is not None:
        velocity_cur = velocity_mean
    if velocity_cur is not None:
        result["velocity_cur"] = (velocity_cur, time_steps[:-1].float())

    if solver == "heun":
        velocity_prime = payload.get("velocity_prime")
        if velocity_prime is None and velocity_mean is not None and velocity_cur is not None:
            velocity_prime = 2.0 * velocity_mean[:, :-1] - velocity_cur[:, :-1]
        if velocity_prime is not None:
            result["velocity_prime"] = (
                velocity_prime,
                time_steps[1:-1].float(),
            )
        if velocity_mean is not None:
            result["velocity_heun_mean"] = (
                velocity_mean,
                ((time_steps[:-1] + time_steps[1:]) * 0.5).float(),
            )
    return result


def append_metrics(destination, modality, metrics):
    for name, values in metrics.items():
        destination[modality][name].append(values.cpu())


def distribution(values):
    values = values.float()
    return {
        "mean": values.mean(0),
        "std": values.std(0, unbiased=False),
        "median": values.median(0).values,
        "p95": torch.quantile(values, 0.95, dim=0),
        "max": values.max(0).values,
        "count": int(values.shape[0]),
    }


def to_json(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    return value


class DinoV2Encoder:
    def __init__(self, model_path, device, dtype, batch_size):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise ImportError("DINOv2 evaluation requires transformers") from error
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.batch_size = int(batch_size)
        self.processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True
        ).to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def encode(self, images):
        images = images.cpu()
        outputs = []
        for start in range(0, images.shape[0], self.batch_size):
            batch = images[start : start + self.batch_size]
            arrays = [image.permute(1, 2, 0).numpy() for image in batch]
            inputs = self.processor(images=arrays, return_tensors="pt")
            moved = {}
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    moved[key] = value.to(self.device, dtype=self.dtype)
                else:
                    moved[key] = value.to(self.device)
            output = self.model(**moved)
            features = (
                output.pooler_output
                if getattr(output, "pooler_output", None) is not None
                else output.last_hidden_state[:, 0]
            )
            outputs.append(features.float().cpu())
        return torch.cat(outputs, dim=0)


class LPIPSEncoder:
    def __init__(self, net, device, batch_size):
        try:
            import lpips
        except ImportError as error:
            raise ImportError("LPIPS evaluation requires `pip install lpips`") from error
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

    @torch.inference_mode()
    def distance(self, reference, target):
        values = []
        for start in range(0, reference.shape[0], self.batch_size):
            ref = reference[start : start + self.batch_size].to(
                self.device, dtype=torch.float32
            )
            tgt = target[start : start + self.batch_size].to(
                self.device, dtype=torch.float32
            )
            ref = ref / 127.5 - 1.0
            tgt = tgt / 127.5 - 1.0
            values.append(self.model(ref, tgt).flatten().float().cpu())
        return torch.cat(values)


def feature_metrics(reference, target):
    difference = target.float() - reference.float()
    return {
        "rmse": difference.square().mean(-1).sqrt(),
        "cosine_similarity": F.cosine_similarity(
            reference.float(), target.float(), dim=-1, eps=1e-12
        ),
    }


def build_optional_metrics(args):
    device = args.feature_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.feature_dtype]
    dino = (
        DinoV2Encoder(args.dinov2_model, device, dtype, args.feature_batch_size)
        if args.dinov2_model
        else None
    )
    lpips_metric = (
        LPIPSEncoder(args.lpips_net, device, args.feature_batch_size)
        if args.lpips
        else None
    )
    return dino, lpips_metric


def evaluate_target(config, baseline_dir, target_dir, dino, lpips_metric, caches):
    baseline_manifest = read_json(baseline_dir / "run_manifest.json")
    target_manifest = read_json(target_dir / "run_manifest.json")
    validate_manifests(baseline_manifest, target_manifest)
    collected = defaultdict(lambda: defaultdict(list))
    final_collected = defaultdict(list)
    modality_times = {}

    for shard_name, baseline_path, target_path in paired_shards(
        baseline_dir, target_dir
    ):
        reference = safe_torch_load(baseline_path)
        target = safe_torch_load(target_path)
        validate_shards(reference, target, shard_name)
        ref_velocities = reconstruct_velocities(reference, config["solver"])
        tgt_velocities = reconstruct_velocities(target, config["solver"])
        for modality in sorted(ref_velocities.keys() & tgt_velocities.keys()):
            ref_value, times = ref_velocities[modality]
            target_value, target_times = tgt_velocities[modality]
            if not torch.equal(times, target_times):
                raise ValueError(f"Velocity times differ for {modality}")
            append_metrics(
                collected,
                modality,
                numeric_metrics(ref_value, target_value),
            )
            modality_times[modality] = times

        if "final_latents" in reference and "final_latents" in target:
            for name, values in final_tensor_metrics(
                reference["final_latents"], target["final_latents"]
            ).items():
                final_collected[f"final_latent_{name}"].append(values)
        if (
            "final_decoded_uint8" in reference
            and "final_decoded_uint8" in target
        ):
            reference_images = reference["final_decoded_uint8"]
            target_images = target["final_decoded_uint8"]
            for name, values in final_tensor_metrics(
                reference_images, target_images, scale=255.0
            ).items():
                final_collected[f"final_pixel_{name}"].append(values)

            cache_key = (config["baseline_name"], shard_name)
            if dino is not None:
                if ("dino", cache_key) not in caches:
                    caches[("dino", cache_key)] = dino.encode(reference_images)
                target_features = dino.encode(target_images)
                for name, values in feature_metrics(
                    caches[("dino", cache_key)], target_features
                ).items():
                    final_collected[f"final_dinov2_{name}"].append(values)
            if lpips_metric is not None:
                final_collected["final_lpips"].append(
                    lpips_metric.distance(reference_images, target_images)
                )

    step_statistics = {
        modality: {
            metric: distribution(torch.cat(chunks, dim=0))
            for metric, chunks in metrics.items()
        }
        for modality, metrics in collected.items()
    }
    final_statistics = {
        metric: distribution(torch.cat(chunks, dim=0))
        for metric, chunks in final_collected.items()
    }
    if not step_statistics:
        raise ValueError(
            "No velocity can be compared. Euler needs trajectory or Vcur; "
            "Heun needs Vcur, or trajectory for mean-field comparison."
        )
    return {
        "id": config["name"],
        "baseline_id": config["baseline_name"],
        "setup": config["experiment"],
        "solver": config["solver"],
        "num_steps": config["num_steps"],
        "routing_steps": config["routing_steps"],
        "selection_ratio": config["selection_ratio"],
        "cfg_scale": config["cfg_scale"],
        "cfg_routing_mode": config["cfg_routing_mode"],
        "velocity_times": {key: value.tolist() for key, value in modality_times.items()},
        "velocity_metrics": to_json(step_statistics),
        "final_metrics": to_json(final_statistics),
    }


def write_step_csv(path, summaries):
    rows = []
    for summary in summaries:
        for modality, metrics in summary["velocity_metrics"].items():
            times = summary["velocity_times"][modality]
            for metric, stats in metrics.items():
                for step, time_value in enumerate(times):
                    rows.append(
                        {
                            "id": summary["id"],
                            "setup": summary["setup"],
                            "solver": summary["solver"],
                            "routing_steps": summary["routing_steps"],
                            "selection_ratio": summary["selection_ratio"],
                            "cfg_scale": summary["cfg_scale"],
                            "cfg_routing_mode": summary["cfg_routing_mode"],
                            "modality": modality,
                            "metric": metric,
                            "step": step,
                            "time": time_value,
                            "mean": stats["mean"][step],
                            "std": stats["std"][step],
                            "median": stats["median"][step],
                            "p95": stats["p95"][step],
                            "max": stats["max"][step],
                            "count": stats["count"],
                        }
                    )
    write_csv(path, rows)


def write_final_csv(path, summaries):
    rows = []
    for summary in summaries:
        for metric, stats in summary["final_metrics"].items():
            rows.append(
                {
                    "id": summary["id"],
                    "setup": summary["setup"],
                    "solver": summary["solver"],
                    "routing_steps": summary["routing_steps"],
                    "selection_ratio": summary["selection_ratio"],
                    "cfg_scale": summary["cfg_scale"],
                    "cfg_routing_mode": summary["cfg_routing_mode"],
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "median": stats["median"],
                    "p95": stats["p95"],
                    "max": stats["max"],
                    "count": stats["count"],
                }
            )
    write_csv(path, rows)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    suite_root, targets = discover(
        args.suite_root, args.target, args.baseline_dir
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else suite_root / "deviation_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dino, lpips_metric = build_optional_metrics(args)
    caches = {}
    summaries = []
    for index, (config, baseline_dir, target_dir) in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {config['name']}")
        summaries.append(
            evaluate_target(
                config,
                baseline_dir,
                target_dir,
                dino,
                lpips_metric,
                caches,
            )
        )
        write_json(
            output_dir / "deviation_summary.json",
            {
                "format": "tread-routing-deviation-v2",
                "suite_root": str(suite_root),
                "intermediate_metrics": "velocity fields only; no X0 metrics",
                "runs": summaries,
            },
        )
    write_step_csv(output_dir / "step_metrics.csv", summaries)
    write_final_csv(output_dir / "final_image_metrics.csv", summaries)
    print(f"Deviation evaluation complete: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help=(
            "Configuration name from suite_manifest.json or an explicit run "
            "directory; repeat as needed"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--dinov2-model",
        default=None,
        help="Local Transformers DINOv2 directory (no network download)",
    )
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument(
        "--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex"
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
