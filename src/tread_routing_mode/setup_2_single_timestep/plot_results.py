#!/usr/bin/env python3
"""Plot final and trajectory metrics for single-timestep TREAD routing."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dit_uncertainty_matplotlib")
)

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_COMPARISON_DIR = HERE / "comparison_results"

FINAL_COLUMNS = {
    "route_step",
    "route_t",
    "keep_ratio",
    "masked_ratio",
    "num_kept",
    "num_masked",
    "mask_seed",
    "immediate_step",
    "immediate_dino_similarity",
    "immediate_lpips_distance",
    "final_dino_similarity",
    "final_lpips_distance",
}
TRAJECTORY_COLUMNS = {
    "route_step",
    "route_t",
    "keep_ratio",
    "masked_ratio",
    "num_kept",
    "num_masked",
    "mask_seed",
    "observation_step",
    "observation_t",
    "phase",
    "dino_similarity",
    "lpips_distance",
}

FINAL_INT_COLUMNS = {
    "route_step",
    "num_kept",
    "num_masked",
    "mask_seed",
    "immediate_step",
}
FINAL_FLOAT_COLUMNS = FINAL_COLUMNS - FINAL_INT_COLUMNS
TRAJECTORY_INT_COLUMNS = {
    "route_step",
    "num_kept",
    "num_masked",
    "mask_seed",
    "observation_step",
}
TRAJECTORY_FLOAT_COLUMNS = TRAJECTORY_COLUMNS - TRAJECTORY_INT_COLUMNS - {"phase"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot DINO and LPIPS effects for routing at one diffusion timestep."
        )
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help="Directory containing final_metrics.csv and trajectory_metrics.csv.",
    )
    return parser.parse_args()


def _read_csv(
    path: Path,
    required_columns: Iterable[str],
    int_columns: Iterable[str],
    float_columns: Iterable[str],
) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metrics file does not exist: {path}")

    required = set(required_columns)
    integer = set(int_columns)
    floating = set(float_columns)
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")

        for line_number, raw in enumerate(reader, start=2):
            row: Dict[str, object] = {}
            try:
                for name in required:
                    value = raw[name]
                    if value is None or value.strip() == "":
                        raise ValueError(f"empty value in column {name!r}")
                    if name in integer:
                        row[name] = int(value)
                    elif name in floating:
                        number = float(value)
                        if not math.isfinite(number):
                            raise ValueError(f"non-finite value in column {name!r}")
                        row[name] = number
                    else:
                        row[name] = value
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid row {line_number} in {path}: {error}") from error
            rows.append(row)

    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")
    return rows


def _as_int(row: Mapping[str, object], name: str) -> int:
    return int(row[name])


def _as_float(row: Mapping[str, object], name: str) -> float:
    return float(row[name])


def _validate_common_row(row: Mapping[str, object], source: str) -> None:
    route_step = _as_int(row, "route_step")
    route_t = _as_float(row, "route_t")
    keep_ratio = _as_float(row, "keep_ratio")
    masked_ratio = _as_float(row, "masked_ratio")
    num_kept = _as_int(row, "num_kept")
    num_masked = _as_int(row, "num_masked")

    if route_step < 1:
        raise ValueError(f"{source}: route_step must be positive, got {route_step}")
    if not 0.0 < route_t <= 1.0:
        raise ValueError(f"{source}: route_t must be in (0, 1], got {route_t}")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"{source}: keep_ratio must be in (0, 1], got {keep_ratio}")
    if not math.isclose(masked_ratio, 1.0 - keep_ratio, abs_tol=1e-9):
        raise ValueError(
            f"{source}: masked_ratio={masked_ratio} is inconsistent with "
            f"keep_ratio={keep_ratio}"
        )
    if num_kept < 1 or num_masked < 0:
        raise ValueError(
            f"{source}: invalid token counts kept={num_kept}, masked={num_masked}"
        )


def validate_metrics(
    final_rows: Sequence[Mapping[str, object]],
    trajectory_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[int], List[float], List[int]]:
    for row in final_rows:
        _validate_common_row(row, "final_metrics.csv")
        if _as_int(row, "immediate_step") != _as_int(row, "route_step") - 1:
            raise ValueError(
                "final_metrics.csv: immediate_step must equal route_step - 1"
            )
    for row in trajectory_rows:
        _validate_common_row(row, "trajectory_metrics.csv")
        if _as_int(row, "observation_step") < 0:
            raise ValueError("trajectory_metrics.csv: observation_step must be non-negative")
        observation_t = _as_float(row, "observation_t")
        if not 0.0 <= observation_t <= 1.0:
            raise ValueError(
                "trajectory_metrics.csv: observation_t must be in [0, 1]"
            )

    final_seeds = {_as_int(row, "mask_seed") for row in final_rows}
    trajectory_seeds = {_as_int(row, "mask_seed") for row in trajectory_rows}
    if len(final_seeds) != 1 or final_seeds != trajectory_seeds:
        raise ValueError(
            "Expected one fixed mask_seed shared by final and trajectory metrics"
        )

    route_steps = sorted({_as_int(row, "route_step") for row in final_rows}, reverse=True)
    expected_route_steps = list(range(max(route_steps), 0, -1))
    if route_steps != expected_route_steps:
        raise ValueError(
            f"route_step grid is incomplete: got {route_steps}, "
            f"expected {expected_route_steps}"
        )

    ratios = sorted({_as_float(row, "keep_ratio") for row in final_rows}, reverse=True)
    trajectory_ratios = sorted(
        {_as_float(row, "keep_ratio") for row in trajectory_rows}, reverse=True
    )
    if ratios != trajectory_ratios:
        raise ValueError("Final and trajectory CSV files contain different keep ratios")
    if len(ratios) < 2 or not any(math.isclose(ratio, 1.0) for ratio in ratios):
        raise ValueError("Expected ratio=1 control and at least one masked keep ratio")

    final_keys = [
        (_as_int(row, "route_step"), _as_float(row, "keep_ratio"))
        for row in final_rows
    ]
    if len(final_keys) != len(set(final_keys)):
        raise ValueError("final_metrics.csv contains duplicate route_step/keep_ratio rows")
    expected_final_keys = {
        (route_step, ratio) for route_step in route_steps for ratio in ratios
    }
    if set(final_keys) != expected_final_keys:
        missing = sorted(expected_final_keys - set(final_keys))
        raise ValueError(f"final_metrics.csv grid is incomplete; missing={missing}")

    sample_steps = max(route_steps)
    observation_steps = list(range(sample_steps, -1, -1))
    trajectory_keys = [
        (
            _as_int(row, "route_step"),
            _as_float(row, "keep_ratio"),
            _as_int(row, "observation_step"),
        )
        for row in trajectory_rows
    ]
    if len(trajectory_keys) != len(set(trajectory_keys)):
        raise ValueError(
            "trajectory_metrics.csv contains duplicate route_step/keep_ratio/observation_step rows"
        )
    expected_trajectory_keys = {
        (route_step, ratio, observation_step)
        for route_step in route_steps
        for ratio in ratios
        for observation_step in observation_steps
    }
    if set(trajectory_keys) != expected_trajectory_keys:
        missing_count = len(expected_trajectory_keys - set(trajectory_keys))
        extra_count = len(set(trajectory_keys) - expected_trajectory_keys)
        raise ValueError(
            "trajectory_metrics.csv grid is incomplete or inconsistent: "
            f"missing={missing_count}, extra={extra_count}"
        )

    token_totals = {
        _as_int(row, "num_kept") + _as_int(row, "num_masked")
        for row in [*final_rows, *trajectory_rows]
    }
    if len(token_totals) != 1:
        raise ValueError(f"Token total changes across trials: {sorted(token_totals)}")

    for ratio in ratios:
        control = math.isclose(ratio, 1.0)
        rows = [row for row in final_rows if _as_float(row, "keep_ratio") == ratio]
        expected_kept = max(1, int(next(iter(token_totals)) * ratio))
        for row in rows:
            if _as_int(row, "num_kept") != expected_kept:
                raise ValueError(
                    f"keep_ratio={ratio} has inconsistent num_kept in final metrics"
                )
            if control:
                if abs(_as_float(row, "final_dino_similarity") - 1.0) > 1e-4:
                    raise ValueError("ratio=1 control DINO similarity is not identity")
                if abs(_as_float(row, "final_lpips_distance")) > 1e-6:
                    raise ValueError("ratio=1 control LPIPS distance is not zero")

    return route_steps, ratios, observation_steps


def _format_ratio(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _tick_positions(values: Sequence[int], maximum_ticks: int = 11) -> Tuple[List[int], List[int]]:
    if len(values) <= maximum_ticks:
        positions = list(range(len(values)))
    else:
        stride = int(math.ceil((len(values) - 1) / float(maximum_ticks - 1)))
        positions = list(range(0, len(values), stride))
        if positions[-1] != len(values) - 1:
            positions.append(len(values) - 1)
    return positions, [values[position] for position in positions]


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_final_metrics(
    rows: Sequence[Mapping[str, object]],
    route_steps: Sequence[int],
    ratios: Sequence[float],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    masked_ratios = [ratio for ratio in ratios if not math.isclose(ratio, 1.0)]
    color_values = plt.colormaps["viridis"](
        np.linspace(0.12, 0.88, max(1, len(masked_ratios)))
    )
    colors = {ratio: color_values[index] for index, ratio in enumerate(masked_ratios)}

    by_key = {
        (_as_float(row, "keep_ratio"), _as_int(row, "route_step")): row
        for row in rows
    }
    x = np.asarray(sorted(route_steps), dtype=int)
    for ratio in ratios:
        control = math.isclose(ratio, 1.0)
        ratio_rows = [by_key[(ratio, int(step))] for step in x]
        color = "#333333" if control else colors[ratio]
        label = (
            "keep=1 (unmasked control)"
            if control
            else f"keep={_format_ratio(ratio)}, masked={_format_ratio(1.0 - ratio)}"
        )
        line_style = "--" if control else "-"
        axes[0].plot(
            x,
            [_as_float(row, "final_dino_similarity") for row in ratio_rows],
            color=color,
            linestyle=line_style,
            marker="o",
            markersize=4,
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            x,
            [_as_float(row, "final_lpips_distance") for row in ratio_rows],
            color=color,
            linestyle=line_style,
            marker="o",
            markersize=4,
            linewidth=2.0,
            label=label,
        )

    axes[0].set_title("A. Final DINO similarity", loc="left", fontweight="bold")
    axes[0].set_ylabel("DINO cosine similarity ↑")
    axes[1].set_title("B. Final LPIPS distance", loc="left", fontweight="bold")
    axes[1].set_ylabel("LPIPS distance ↓")
    tick_positions = list(range(1, max(route_steps) + 1, 2))
    if max(route_steps) not in tick_positions:
        tick_positions.append(max(route_steps))
    for ax in axes:
        ax.set_xlabel(
            f"Routing step k (transition k → k−1; {max(route_steps)} = early, 1 = late)"
        )
        ax.set_xticks(sorted(set(tick_positions)))
        ax.set_xlim(max(route_steps) + 0.5, min(route_steps) - 0.5)
        _style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=min(3, len(ratios)),
        frameon=False,
        fontsize=9,
    )

    for suffix in ("png", "pdf"):
        path = output_dir / f"final_metrics_by_route_step.{suffix}"
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Saved {path}")
    plt.close(fig)


def _trajectory_matrix(
    rows_by_key: Mapping[Tuple[float, int, int], Mapping[str, object]],
    ratio: float,
    route_steps: Sequence[int],
    observation_steps: Sequence[int],
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            [
                _as_float(rows_by_key[(ratio, route_step, observation_step)], metric)
                for observation_step in observation_steps
            ]
            for route_step in route_steps
        ],
        dtype=float,
    )


def plot_trajectory_heatmaps(
    rows: Sequence[Mapping[str, object]],
    route_steps: Sequence[int],
    ratios: Sequence[float],
    observation_steps: Sequence[int],
    output_dir: Path,
) -> None:
    masked_ratios = [ratio for ratio in ratios if not math.isclose(ratio, 1.0)]
    if not masked_ratios:
        raise ValueError("At least one keep_ratio below 1 is required for heatmaps")

    rows_by_key = {
        (
            _as_float(row, "keep_ratio"),
            _as_int(row, "route_step"),
            _as_int(row, "observation_step"),
        ): row
        for row in rows
    }
    dino_matrices = [
        _trajectory_matrix(
            rows_by_key,
            ratio,
            route_steps,
            observation_steps,
            "dino_similarity",
        )
        for ratio in masked_ratios
    ]
    lpips_matrices = [
        _trajectory_matrix(
            rows_by_key,
            ratio,
            route_steps,
            observation_steps,
            "lpips_distance",
        )
        for ratio in masked_ratios
    ]

    dino_min = min(float(matrix.min()) for matrix in dino_matrices)
    dino_max = max(1.0, max(float(matrix.max()) for matrix in dino_matrices))
    if math.isclose(dino_min, dino_max):
        dino_min = dino_max - 1e-6
    lpips_max = max(float(matrix.max()) for matrix in lpips_matrices)
    if math.isclose(lpips_max, 0.0):
        lpips_max = 1e-6

    width = max(12.0, 4.1 * len(masked_ratios))
    fig, axes = plt.subplots(
        2,
        len(masked_ratios),
        figsize=(width, 10),
        squeeze=False,
        constrained_layout=True,
    )
    dino_image = None
    lpips_image = None
    x_positions, x_labels = _tick_positions(observation_steps)
    y_positions, y_labels = _tick_positions(route_steps)

    for column, ratio in enumerate(masked_ratios):
        dino_ax = axes[0, column]
        lpips_ax = axes[1, column]
        dino_image = dino_ax.imshow(
            dino_matrices[column],
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap="viridis",
            vmin=dino_min,
            vmax=dino_max,
        )
        lpips_image = lpips_ax.imshow(
            lpips_matrices[column],
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap="magma_r",
            vmin=0.0,
            vmax=lpips_max,
        )
        dino_ax.set_title(
            f"keep={_format_ratio(ratio)}\nmasked={_format_ratio(1.0 - ratio)}",
            fontweight="bold",
        )
        for ax in (dino_ax, lpips_ax):
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels)
            ax.set_yticks(y_positions)
            if column == 0:
                ax.set_yticklabels(y_labels)
                ax.set_ylabel("Routing step k")
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Observed state step")

    axes[0, 0].text(
        -0.28,
        0.5,
        "DINO similarity ↑",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontweight="bold",
    )
    axes[1, 0].text(
        -0.28,
        0.5,
        "LPIPS distance ↓",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontweight="bold",
    )
    if dino_image is not None:
        fig.colorbar(
            dino_image,
            ax=list(axes[0, :]),
            shrink=0.78,
            pad=0.015,
            label="DINO cosine similarity",
        )
    if lpips_image is not None:
        fig.colorbar(
            lpips_image,
            ax=list(axes[1, :]),
            shrink=0.78,
            pad=0.015,
            label="LPIPS distance",
        )
    fig.suptitle(
        "Single-timestep TREAD: immediate and downstream deviation from baseline",
        y=1.02,
    )

    for suffix in ("png", "pdf"):
        path = output_dir / f"trajectory_heatmaps.{suffix}"
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comparison_dir = args.comparison_dir.expanduser().resolve()
    final_rows = _read_csv(
        comparison_dir / "final_metrics.csv",
        FINAL_COLUMNS,
        FINAL_INT_COLUMNS,
        FINAL_FLOAT_COLUMNS,
    )
    trajectory_rows = _read_csv(
        comparison_dir / "trajectory_metrics.csv",
        TRAJECTORY_COLUMNS,
        TRAJECTORY_INT_COLUMNS,
        TRAJECTORY_FLOAT_COLUMNS,
    )
    route_steps, ratios, observation_steps = validate_metrics(
        final_rows, trajectory_rows
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_final_metrics(final_rows, route_steps, ratios, comparison_dir)
    plot_trajectory_heatmaps(
        trajectory_rows,
        route_steps,
        ratios,
        observation_steps,
        comparison_dir,
    )


if __name__ == "__main__":
    main()
