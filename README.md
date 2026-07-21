# DiT uncertainty: controlled TREAD routing

This repository compares an unmasked flow-matching baseline with controlled
TREAD inference experiments.

## Fixed experiment

The single source of truth is `src/tread_routing_mode/config.json`:

- checkpoint: `weights/0400000.pt`, EMA weights;
- ImageNet class: `0`;
- initial-noise seed: `0`;
- Euler ODE, 20 steps, CFG 3.0;
- TREAD route: blocks 2 through 8;
- Setup 1: `keep_ratio=0.5` and mask seeds `0` through `9`;
- Setup 2: one routed timestep, fixed `mask_seed=0`, and five keep ratios from
  `1.0` to the training value `0.5`.

`keep_ratio` is the fraction of 256 patch tokens processed inside the routed
block range. At `0.5`, 128 tokens are processed and 128 bypass the route. The
baseline uses all 256 tokens. For each TREAD run, one new mask is drawn at every
Euler timestep in Setup 1 and shared by the conditional and unconditional CFG
branches. In Setup 2, exactly one mask is drawn for one transition `k -> k-1`;
all other model evaluations run with routing disabled.

## Structure

```text
src/tread_routing_mode/
  config.json                    # all fixed parameters + mask seeds
  common/                        # model, sampler, routing, DINO and LPIPS
  baseline/
    run.py
    results/
  setup_1_resampled_masks/
    run.py
    plot_results.py
    comparison_results/          # metrics, analysis and plots
    results/
  setup_2_single_timestep/
    run.py
    plot_results.py
    comparison_results/
    results/
```

## Run

Run from the repository root. Baseline and both setups must use the same
generation device. Setup 2 defaults to the device recorded in the baseline.

```bash
.venv/bin/python src/tread_routing_mode/baseline/run.py
.venv/bin/python src/tread_routing_mode/setup_1_resampled_masks/run.py
.venv/bin/python src/tread_routing_mode/setup_1_resampled_masks/plot_results.py
.venv/bin/python src/tread_routing_mode/setup_2_single_timestep/run.py
.venv/bin/python src/tread_routing_mode/setup_2_single_timestep/plot_results.py
```

Both setups refuse to run if the checkpoint bytes, EMA selector, class, initial
noise, solver, CFG, step count, VAE, save frequency, device, or dtype differ
from the baseline. Every run records the initial-noise hash, mask hashes, and a
fixed-parameter fingerprint.

## Comparison

Every decoded timestep is paired with the same timestep from the baseline using
exactly two image metrics:

- `dino_similarity`: cosine similarity between DINOv2 CLS embeddings; higher
  means closer;
- `lpips_distance`: LPIPS v0.1 with AlexNet; lower means closer.

Per-seed trajectories are stored in `image_metrics_to_baseline.json`. Aggregate
Setup 1 outputs are:

```text
src/tread_routing_mode/setup_1_resampled_masks/comparison_results/
  analysis.json
  final_metrics.csv
  trajectory_metrics.csv
  metrics_by_timestep.png
  metrics_by_timestep.pdf
```

Setup 2 stores its aggregate tables and final/trajectory plots in:

```text
src/tread_routing_mode/setup_2_single_timestep/comparison_results/
```
