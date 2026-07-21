# Setup 2: routing at one timestep

This experiment changes exactly two variables: the Euler timestep at which
TREAD routing is enabled and the fraction of patch tokens kept by the routed
blocks. The model, checkpoint, class, initial noise, CFG, solver, VAE,
`mask_seed`, and every other generation parameter remain fixed to the baseline.

`keep_ratio` is the fraction of the 256 tokens processed inside the routed
block range. It is not the masked fraction. The configured five-level sweep is:

| keep_ratio | masked_ratio | kept / 256 tokens |
|---:|---:|---:|
| 1.000 | 0.000 | 256 |
| 0.875 | 0.125 | 224 |
| 0.750 | 0.250 | 192 |
| 0.625 | 0.375 | 160 |
| 0.500 | 0.500 | 128 |

The lower bound, `keep_ratio=0.5`, matches the checkpoint's training regime.
`keep_ratio=1` is an explicit unmasked control and must be bit-identical to the
baseline.

With 20 Euler updates, selectable routing steps are `20..1`. Routing at step
`k` changes the transition `k -> k-1`: state `k` is still pre-routing, state
`k-1` is the immediate response, and state `0` is the final image. One fixed
`mask_seed=0` is reset for every trial, so timestep effects are not mixed with
mask randomness and token subsets are controlled across the ratio sweep.

## Run

Run from the repository root after creating the 20-step baseline:

```bash
.venv/bin/python src/tread_routing_mode/setup_2_single_timestep/run.py
.venv/bin/python src/tread_routing_mode/setup_2_single_timestep/plot_results.py
```

The full grid contains 100 paired runs (20 timesteps x 5 ratios). Completed
trials are validated and resumed. An interrupted, incomplete trial is discarded
and regenerated. Use `--overwrite` only when completed trials must also be
regenerated.

A smaller diagnostic subset can be selected without changing the config:

```bash
.venv/bin/python src/tread_routing_mode/setup_2_single_timestep/run.py \
  --route-steps 20 10 1 --keep-ratios 1.0 0.5
```

Run the full command afterward to complete the grid; matching trials will be
reused. A subset run writes an explicitly partial aggregate
(`grid_complete=false`); the plotting script intentionally requires the full
grid so that partial figures cannot be mistaken for the final experiment.

## Outputs

```text
setup_2_single_timestep/
  results/
    route_step_0020/
      keep_ratio_1p000/
      keep_ratio_0p875/
      ...
  comparison_results/
    analysis.json
    final_metrics.csv
    trajectory_metrics.csv
    final_metrics_by_route_step.png
    final_metrics_by_route_step.pdf
    trajectory_heatmaps.png
    trajectory_heatmaps.pdf
```

Every candidate image is compared with the same saved timestep from the
baseline using DINOv2 cosine similarity (higher is closer) and LPIPS AlexNet
distance (lower is closer). Runtime checks enforce one selected routing step,
one mask draw for `keep_ratio<1`, no mask draw for the ratio-one control,
identity with baseline before routing, and a complete `20..0` metric trajectory.
