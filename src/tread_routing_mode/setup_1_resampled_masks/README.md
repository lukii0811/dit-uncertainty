# Setup 1: resampled mask at every timestep

Runs ten paired TREAD trajectories using the fixed parameters from
`../config.json`. Only `mask_seed` changes. Each Euler timestep performs a new
mask draw; the mask is shared by the conditional and unconditional CFG calls.

All decoded states are compared with the unmasked baseline using
`dino_similarity` and `lpips_distance`.

```bash
../../../.venv/bin/python run.py
../../../.venv/bin/python plot_results.py
```

The plotting command reads the saved per-seed metric trajectories and writes
`comparison_results/metrics_by_timestep.png` and `.pdf` inside this setup. The
figure contains exactly two plots: DINO similarity and LPIPS distance against
the diffusion timestep, from initial noise at `t=20` to the final image at
`t=0`.
