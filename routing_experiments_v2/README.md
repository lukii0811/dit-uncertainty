# Self-contained TREAD-SiT Routing Experiments v2

This folder can be copied and run independently. It preserves the familiar
file names, CLI conventions, seed pairing, manifests, tensor shards, resume
behavior, output layout, checkpoint loading, and VAE loading from the previous
version.

The intentional changes are limited to:

1. intermediate predicted X0 is never constructed, saved, decoded, or scored;
2. trajectory, Vcur, Vprime, and routing masks can be saved instead;
3. CFG routing supports both branches, conditional only, or unconditional only;
4. final decoded images can be scored with local DINOv2 and LPIPS;
5. experiment-specific sampling settings accept lists and are swept in one
   process without reloading SiT or the VAE.

## Changes from the previous version

The model backbone and the established experiment workflow were deliberately
kept familiar. `sit_tread.py` retains the previous TREAD-SiT gather/skip/scatter
implementation, while checkpoint loading, seed-class pairing, VAE loading,
resume behavior, manifests, and tensor sharding follow the previous logic. All
Python source needed at runtime is now copied into this folder, so it does not
import project code from its parent directory.

The main logic changes are concentrated in the following locations:

| File and location | Change relative to the previous version |
| --- | --- |
| `sampling/sit_routing_sampler.py`: `predict_guided_velocity()` | CFG can route both branches, only the conditional branch, or only the unconditional branch. When the two branches require different routing, they are evaluated separately and then combined with the usual CFG equation. |
| `sampling/sit_routing_sampler.py`: `sit_sampler_with_routing()` | Intermediate predicted X0 is no longer reconstructed or returned. The sampler can instead record the latent trajectory, current velocity `Vcur`, optional Heun correction velocity `Vprime`, and routing masks. Euler/Heun state-update equations and the existing Setup 1/Setup 2 routing behavior remain unchanged. |
| `inference_routing_experiments.py`: `resolve_save_policy()`, `save_tensor_shard()`, and `run()` | The old intermediate-X0 save/decode path was removed. Only the completed final latent is decoded. New controls select trajectory, `Vcur`, `Vprime`, routing-mask, final-latent, and final-image storage. |
| `run_inference_suite.py`: `build_configurations()` | Setup, keep ratio, routed step, CFG scale, CFG interval, CFG routing branch, solver, and sampling-step arguments may be supplied as lists. Their Cartesian product is run after loading SiT and the VAE only once. Existing scalar arguments remain valid fallbacks. |
| `compute_deviation_suite.py`: `reconstruct_velocities()` | Intermediate comparison is velocity-only. For Euler, `Vcur` may be recovered from adjacent trajectory states. For Heun, a saved trajectory plus `Vcur` can recover `Vprime` and the effective mean velocity. |
| `compute_deviation_suite.py`: `evaluate_target()` | Intermediate predicted-X0, decoded-X0, CLIP, and DINO comparisons were removed. Final images may now be compared with DINOv2 embedding RMSE/cosine similarity and LPIPS, in addition to final latent and pixel errors. |

The tensor-shard schema therefore changes as follows:

- Removed: `x0_prediction_times`, `x0_predictions`, and
  `x0_decoded_uint8`.
- Added: `velocity_cur` and optional `velocity_prime`.
- Retained where enabled: `sample_id`, `seed`, `class_idx`, `time_steps`,
  `xt_trajectory`, `routing_masks`, `final_latents`, and
  `final_decoded_uint8`.

For a focused review, teammates only need to inspect the functions named in the
table above. The TREAD-SiT architecture itself does not contain a new CFG
implementation; branch-specific CFG routing is isolated in the sampler.

## Files

- `sit_tread.py`: complete SiT-B/2 and TREAD gather/scatter model;
- `checkpoint.py`: safe mentor-checkpoint inspection and strict model loading;
- `sampling/sit_routing_sampler.py`: Euler/Heun and branch-selective CFG;
- `inference_routing_experiments.py`: familiar single-configuration entrypoint;
- `run_inference_suite.py`: familiar one-load Baseline/Setup1/Setup2 suite;
- `compute_deviation_suite.py`: paired flow-field/final-image evaluation;
- `generate_seed_class_pairs.py`: seed-class manifest generator;
- `requirements.txt`: complete Python dependency list.

No Python function is imported from outside this folder.

## Recommended suite command

Run inside this folder:

```bash
python run_inference_suite.py \
  --checkpoint ../parameters/0400000.pt \
  --seed-pairs ../seed_pairs_64.json \
  --output-dir ../runs/routing_v2 \
  --experiments baseline every-step single-step fixed-mask \
  --suite-selection-ratios 0.5 \
  --single-steps 0-31 \
  --suite-num-steps 32 \
  --suite-solvers heun \
  --suite-cfg-scales 1.5 \
  --suite-cfg-routing-modes both conditional unconditional \
  --suite-guidance-intervals 0.0:1.0 \
  --suite-mask-seeds 42 \
  --save-mode paired \
  --batch-size 64 \
  --decode-batch-size 64 \
  --vae-source local \
  --vae-model ../parameters/sdvae/sd-vae-ft-mse \
  --allow-unsafe-checkpoint-load
```

The familiar scalar arguments remain supported. If a suite-list option is not
provided, its corresponding scalar value is used:

- `--num-steps` -> `--suite-num-steps`;
- `--solver` -> `--suite-solvers`;
- `--cfg-scale` -> `--suite-cfg-scales`;
- `--cfg-routing-mode` -> `--suite-cfg-routing-modes`;
- `--mask-seed` -> `--suite-mask-seeds`;
- `--guidance-low/--guidance-high` -> `--suite-guidance-intervals`.

Every suite list may contain multiple values. The suite runs their Cartesian
product. Use `--dry-run` before a large sweep and `--resume` after interruption.

## Familiar output layout

The top-level experiment layout remains:

```text
output_dir/
  baseline/
  setup1_every_step/
    keep_0p5/
  setup2_single_step/
    keep_0p5/
      step_000/
  setup3_fixed_mask/
    keep_0p5/
```

Setup 3 (`--experiment fixed-mask` / suite `fixed-mask`) applies TREAD on every
solver step like Setup 1, but samples the routing mask once and reuses it
(`mask_schedule=fixed`). Setup 1 keeps resampling (`mask_schedule=per_step`).

Patch-level epistemic/aleatoric maps from a Setup 3 paired run:

```bash
python visualize_mask_uncertainty.py \
  --run-dir ../runs/suite_run/setup3_fixed_mask/keep_0p5/<sampling_tag> \
  --checkpoint ../parameters/0400000.pt \
  --output-dir ../runs/uncertainty_viz \
  --steps 0,25,49 \
  --num-routes 16 --num-seeds 4 \
  --allow-unsafe-checkpoint-load
```

Uses \(\hat{x}_0=x_t-tv\), TREAD patchify, then
\(U_{epi}=\mathrm{Var}_r[\mathbb{E}_z]\), \(U_{ale}=\mathbb{E}_r[\mathrm{Var}_z]\)
(component-wise over patch features, then mean). Writes heatmaps and `profiles.csv`.

Solver, step count, CFG, branch-routing mode, guidance interval, and mask seed
are appended below these familiar directories only when needed to distinguish
list combinations. Every run still contains:

```text
run_manifest.json
tensor_shards/shard_XXXXXX_XXXXXX.pt
images/                         # FID/final-PNG mode only
```

## Recording policy

`--save-mode paired` now saves:

- Xt trajectory;
- guided Vcur;
- routing masks;
- final latent;
- final decoded uint8 image tensor.

It does not save any intermediate X0. For custom storage use the familiar
flags:

```bash
--save-mode custom \
--save-xt \
--save-vcur \
--save-routing-masks \
--save-final-latents \
--save-final-images-tensor
```

Euler can recover `Vcur = (Xnext-Xcur)/dt` from trajectory alone. Heun should
retain trajectory plus Vcur; then, for every non-final corrector step:

```text
Vprime = 2 * (Xnext-Xcur)/dt - Vcur
```

The final Heun step remains Euler-only, matching the previous sampler.

## CFG routing modes

- `both`: route conditional and unconditional branches with the same mask;
- `conditional`: route only the conditional branch;
- `unconditional`: route only the unconditional branch.

When both branches have the same routing state, they are evaluated in one
batched model call as before. A one-branch routing mode requires two separate
model calls. Outside the active CFG interval only one evaluation exists, and it
is routed whenever Setup1/Setup2 requests routing.

## Deviation evaluation

```bash
python compute_deviation_suite.py \
  --suite-root ../runs/routing_v2 \
  --dinov2-model ../parameters/dinov2-base \
  --lpips \
  --lpips-net alex
```

The evaluator writes the familiar `deviation_summary.json` and
`step_metrics.csv`, plus `final_image_metrics.csv`. Intermediate metrics cover
only Vcur, Vprime, and the Heun mean field. Final metrics cover latent/pixel
differences, DINOv2 feature RMSE/cosine similarity, and optional LPIPS.

DINOv2 must be a local Transformers-format directory. On an offline server,
LPIPS and its selected backbone weights must already be installed/cached.
