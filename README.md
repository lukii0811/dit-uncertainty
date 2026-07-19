# TREAD-SiT Routing Inference v4

This package is a provisional inference implementation for the Mentor
checkpoint described by the supplied training arguments:

- `SiT-B/2`, ImageNet-1K, 256 x 256;
- linear flow-matching path;
- velocity (`v`) prediction;
- DINOv2-B REPA alignment during training;
- TREAD routing from block 2 through block 8;
- `selection_ratio=0.5`, `qk_norm=False`, `fused_attn=True`.

It does not use DINOv2 during sampling. DINOv2 supplied a training-only REPA
target. Images are decoded with a Stable Diffusion VAE; this version defaults
to `stabilityai/sd-vae-ft-mse` and latent scale `0.18215`.

## Important safety behavior

The loader automatically finds common checkpoint keys (`ema`, `model`,
`state_dict`, `ema_state_dict`, or `model_state_dict`) and infers REPA projector
dimensions from tensor shapes. Projector weights are ignored by default because
they are not executed during generation. Every non-projector parameter must
match the provisional SiT model. A mismatch raises an error instead of silently
leaving generative layers randomly initialized.

`ops_head=16` is recorded in every run manifest but is not used, because it is
not an argument in the official REPA SiT model. If the checkpoint contains
unknown `ops_head` parameters, validation will stop and print their keys.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Validate the checkpoint before downloading a VAE

```bash
python inspect_checkpoint.py --checkpoint /path/to/mentor_checkpoint.pt
```

Success prints JSON containing:

```text
"model_name": "SiT-B/2"
"strict_generative_backbone_match": true
```

If safe loading fails because the trusted checkpoint contains an `argparse`
namespace or other Python metadata, add:

```text
--allow-unsafe-checkpoint-load
```

Only use that option for a checkpoint whose source you trust.

## 2. Generate a reusable 16 x 4 manifest

Choose 16 random classes and four independent starting noises per class:

```bash
python generate_seed_class_pairs.py \
  --output seed_pairs_16x4.json \
  --num-selected-classes 16 \
  --samples-per-class 4
```

Choose the classes explicitly instead:

```bash
python generate_seed_class_pairs.py \
  --output seed_pairs_16x4.json \
  --classes 0,7,42,65,88,123,200,301,402,503,604,705,806,907,950,999 \
  --samples-per-class 4
```

Use exactly the same JSON file for every paired experiment.

## 3. Paired baseline

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/mentor_checkpoint.pt \
  --seed-pairs seed_pairs_16x4.json \
  --output-dir runs/baseline \
  --experiment baseline \
  --save-mode paired
```

Defaults are 50 Euler steps, CFG scale 1.5, FP16, and SD-VAE-FT-MSE.

## 4. Setup 1: resample routing every step

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/mentor_checkpoint.pt \
  --seed-pairs seed_pairs_16x4.json \
  --output-dir runs/every_step_keep_050 \
  --experiment every-step \
  --selection-ratio 0.5 \
  --save-mode paired
```

A fresh per-sample mask is generated at every ODE step. For Heun, the same mask
is reused by the predictor and corrector within that step.

## 5. Setup 2: routing at one selected step

Step index 0 is the first/highest-noise step at `t=1`.

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/mentor_checkpoint.pt \
  --seed-pairs seed_pairs_16x4.json \
  --output-dir runs/single_step_10_keep_050 \
  --experiment single-step \
  --routing-step 10 \
  --selection-ratio 0.5 \
  --save-mode paired
```

Equivalent mask-ratio notation is also accepted:

```text
--mask-ratio 0.5
```

Do not pass both ratio arguments.

## 6. Extra all-token routing control

This invokes the routing gather/scatter structure at step 10 while retaining
all tokens in their original order:

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/mentor_checkpoint.pt \
  --seed-pairs seed_pairs_16x4.json \
  --output-dir runs/single_step_10_keep_100 \
  --experiment single-step \
  --routing-step 10 \
  --selection-ratio 1.0 \
  --save-mode paired
```

Compare this run to `selection_ratio < 1` and to the dense baseline.

## 7. Final-image-only generation for FID

Generate 1,000 classes x 10 noises:

```bash
python generate_seed_class_pairs.py \
  --output seed_pairs_1000x10.json \
  --classes 0-999 \
  --samples-per-class 10
```

Then run with `--save-mode fid`. This saves only final PNG files and does not
retain Xt, X0, decoded trajectories, or masks:

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/mentor_checkpoint.pt \
  --seed-pairs seed_pairs_1000x10.json \
  --output-dir runs/fid_baseline \
  --experiment baseline \
  --save-mode fid \
  --batch-size 32 \
  --decode-batch-size 16
```

## Paired tensor contents

Each `tensor_shards/shard_*.pt` contains sample IDs, seeds, class IDs, time
steps, and the enabled fields:

- `xt_trajectory`: `[B, num_steps+1, 4, 32, 32]`;
- `x0_predictions`: `[B, num_steps, 4, 32, 32]`;
- `x0_prediction_times`: time corresponding to each X0 prediction;
- `x0_decoded_uint8`: `[B, num_steps, 3, 256, 256]`;
- `routing_masks`: kept-token indices by solver step;
- `final_latents` and `final_decoded_uint8`.

X0 is reconstructed from the model velocity as:

```text
x0_pred = xt - t * v_pred
```

Xt itself is never sent to the VAE for the trajectory image record.

## Useful alternatives

- Use Heun: `--solver heun`.
- Use the other common SD VAE: `--vae-model stabilityai/sd-vae-ft-ema`.
- Use a cached VAE only: add `--local-files-only`.
- Run a subset: `--start-index 0 --limit 8`.
- Resume completed shards: `--resume`.
- Disable fused attention if unsupported: `--no-fused-attn`.
- Load and validate projector weights too: `--load-projectors`.

Every run writes `run_manifest.json` and `checkpoint_load_report.json` so that
the exact assumptions and loader result are preserved with the samples.
