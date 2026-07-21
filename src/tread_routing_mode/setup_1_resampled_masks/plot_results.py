from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dit_uncertainty_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
TREAD_ROOT = HERE.parent
DEFAULT_RESULTS_ROOT = HERE / "results"
DEFAULT_OUTPUT_DIR = HERE / "comparison_results"
CONFIG_PATH = TREAD_ROOT / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Setup 1 DINO and LPIPS comparisons against baseline."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Setup 1 results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG and PDF figures.",
    )
    return parser.parse_args()


def load_trajectories(results_root: Path) -> dict[int, list[dict[str, float]]]:
    paths = sorted(results_root.glob("mask_seed_*/class_*/seed_*/image_metrics_to_baseline.json"))
    if not paths:
        raise FileNotFoundError(
            f"No image_metrics_to_baseline.json files found under {results_root}"
        )

    trajectories: dict[int, list[dict[str, float]]] = {}
    expected_steps: list[int] | None = None
    for path in paths:
        mask_seed_dir = path.parents[2].name
        mask_seed = int(mask_seed_dir.removeprefix("mask_seed_"))
        if mask_seed in trajectories:
            raise ValueError(f"More than one trajectory found for mask_seed={mask_seed}")

        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        rows = sorted(rows, key=lambda row: int(row["step"]), reverse=True)
        steps = [int(row["step"]) for row in rows]
        if expected_steps is None:
            expected_steps = steps
        elif steps != expected_steps:
            raise ValueError(
                f"Metric steps for mask_seed={mask_seed} are {steps}, expected {expected_steps}"
            )
        trajectories[mask_seed] = rows

    return dict(sorted(trajectories.items()))


def add_seed_lines(
    ax: plt.Axes,
    timesteps: np.ndarray,
    values: np.ndarray,
    colors: np.ndarray,
) -> None:
    for index, seed_values in enumerate(values):
        ax.plot(
            timesteps,
            seed_values,
            color=colors[index],
            linewidth=1.15,
            alpha=0.38,
            label=(
                f"individual mask sequences (n={len(values)})"
                if index == 0
                else None
            ),
            zorder=1,
        )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_trajectory_panel(
    ax: plt.Axes,
    timesteps: np.ndarray,
    values: np.ndarray,
    colors: np.ndarray,
    *,
    title: str,
    ylabel: str,
    baseline_value: float,
    baseline_label: str,
) -> None:
    add_seed_lines(ax, timesteps, values, colors)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ax.fill_between(
        timesteps,
        mean - std,
        mean + std,
        color="#1f4e79",
        alpha=0.16,
        linewidth=0,
        label="mean ± 1 std",
        zorder=2,
    )
    ax.plot(
        timesteps,
        mean,
        color="#143d59",
        marker="o",
        markersize=4.5,
        linewidth=2.4,
        label="mean over mask seeds",
        zorder=3,
    )
    ax.axhline(
        baseline_value,
        color="#333333",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label=baseline_label,
        zorder=0,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(
        f"Diffusion timestep t ({int(timesteps.max())} = initial noise, "
        "0 = final image)"
    )
    ax.set_ylabel(ylabel)
    tick_span = int(timesteps.max() - timesteps.min())
    tick_stride = max(1, (tick_span + 9) // 10)
    ax.set_xticks(
        np.arange(int(timesteps.min()), int(timesteps.max()) + 1, tick_stride)
    )
    ax.invert_xaxis()
    style_axis(ax)


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.results_root.resolve())
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    seeds = np.asarray(list(trajectories), dtype=int)
    rows_by_seed = list(trajectories.values())
    timesteps = np.asarray([int(row["step"]) for row in rows_by_seed[0]], dtype=int)
    sample_steps = int(timesteps.max())
    dino = np.asarray(
        [[float(row["dino_similarity"]) for row in rows] for rows in rows_by_seed]
    )
    lpips = np.asarray(
        [[float(row["lpips_distance"]) for row in rows] for rows in rows_by_seed]
    )
    colors = plt.colormaps["viridis"](np.linspace(0.08, 0.92, len(seeds)))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlepad": 9,
            "axes.labelcolor": "#303030",
            "text.color": "#202020",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    plot_trajectory_panel(
        axes[0],
        timesteps,
        dino,
        colors,
        title="A. DINO similarity along the generation trajectory",
        ylabel="DINO cosine similarity ↑",
        baseline_value=1.0,
        baseline_label="identity with baseline = 1",
    )
    plot_trajectory_panel(
        axes[1],
        timesteps,
        lpips,
        colors,
        title="B. LPIPS distance along the generation trajectory",
        ylabel="LPIPS distance ↓",
        baseline_value=0.0,
        baseline_label="identity with baseline = 0",
    )
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[1].legend(loc="upper right", frameon=False, fontsize=8.5)

    fixed = config["fixed_generation"]
    tread = config["tread"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "metrics_by_timestep.png"
    pdf_path = args.output_dir / "metrics_by_timestep.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
