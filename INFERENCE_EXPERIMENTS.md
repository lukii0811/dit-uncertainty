# Paired EDM/TREAD Inference Experiments

## 1. What is recorded

The experimental sampler distinguishes two latent quantities:

- `Xt`: the noisy state propagated by the EDM solver;
- `X0`: the clean latent predicted by the model (`denoised` in `edm.py`).

The `paired` save mode records both, and VAE-decodes **X0 predictions**, not Xt.
For Heun steps, the recorded X0 is the corrector prediction at `t_next`. For the
last Euler-only step, it is the final current prediction. There is exactly one
recorded X0 per EDM generation step.

The original repository does not save per-step decoded X0 predictions.
`EDMDiffusion.generate()` only decodes the final sample.

## 2. Create persistent, class-balanced seed–class pairs

The generator lets you control the selected classes and the number of starting
noise seeds assigned to every class. It writes only a manifest; all image
generation is performed later by the inference runner.

Select 16 reproducible random classes and generate four noises per class
(16 × 4 = 64 pairs):

```bash
python generate_seed_class_pairs.py \
  --output seed_pairs_64.json \
  --num-selected-classes 16 \
  --class-selection random \
  --class-selection-seed 20260717 \
  --samples-per-class 4 \
  --noise-master-seed 20260718
```

Select exact classes instead:

```bash
python generate_seed_class_pairs.py \
  --output seed_pairs_64.json \
  --classes 0,7,42,100-112 \
  --samples-per-class 4 \
  --noise-master-seed 20260718
```

The `--classes` syntax accepts individual IDs and inclusive ranges. You can also
provide the same syntax in a text file with `--class-file classes.txt`.

For a class-balanced 10,000-image ImageNet experiment, select all 1,000 classes
and generate ten noise seeds per class (1,000 × 10 = 10,000 pairs):

```bash
python generate_seed_class_pairs.py \
  --output seed_pairs_10000.json \
  --num-selected-classes 1000 \
  --class-selection first \
  --samples-per-class 10 \
  --noise-master-seed 20260718
```

Every baseline/routing run must read the same JSON file. Each record contains a
sample ID, unique latent-noise seed, class index, and within-class noise index.

## 3. Paired 64-sample experiment

### Dense baseline

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/latest.pt \
  --seed-pairs seed_pairs_64.json \
  --output-dir results/paired/baseline \
  --experiment baseline \
  --save-mode paired
```

### Setup 1: fresh mask at every generation step

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/latest.pt \
  --seed-pairs seed_pairs_64.json \
  --output-dir results/paired/setup1_mask_0p5 \
  --experiment every-step \
  --mask-ratio 0.5 \
  --save-mode paired
```

### Setup 2: route at one step only

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/latest.pt \
  --seed-pairs seed_pairs_64.json \
  --output-dir results/paired/setup2_step_20_mask_0p5 \
  --experiment single-step \
  --routing-step 20 \
  --mask-ratio 0.5 \
  --save-mode paired
```

With 40 sampling steps, step 0 is the first/highest-noise step and step 39 is
the final/lowest-noise step.

## 4. Saved paired-data format

Data is saved in batch-sized shards under `tensor_shards/`. Each `.pt` file can
contain:

- `sample_id`, `seed`, and `class_idx`;
- `xt_trajectory`: `[B, T+1, 4, 32, 32]`, float32;
- `x0_predictions`: `[B, T, 4, 32, 32]`, float32;
- `x0_decoded_uint8`: `[B, T, 3, 256, 256]`, uint8;
- `routing_masks`: retained token indices by generation step;
- `final_latents`: `[B, 4, 32, 32]`, float32;
- `final_decoded_uint8`: `[B, 3, 256, 256]`, uint8.

The decoded X0 tensors can be converted to float and normalized for CLIP or DINO
feature extraction. Saving them as uint8 tensors substantially reduces disk
usage without requiring individual image files.

The default paired batch size is 4. A complete 64-sample run therefore writes
16 independently readable shards.

## 5. FID-only generation

Use `save-mode fid` to generate final PNG images only. No Xt trajectory, X0
prediction, decoded-X0 tensor, routing mask, or final latent tensor is retained.

Dense baseline:

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/latest.pt \
  --seed-pairs seed_pairs_10000.json \
  --output-dir results/fid/baseline \
  --experiment baseline \
  --batch-size 32 \
  --save-mode fid
```

Every-step routing:

```bash
python inference_routing_experiments.py \
  --checkpoint /path/to/latest.pt \
  --seed-pairs seed_pairs_10000.json \
  --output-dir results/fid/setup1_mask_0p5 \
  --experiment every-step \
  --mask-ratio 0.5 \
  --batch-size 32 \
  --save-mode fid
```

The PNG files appear under `<output-dir>/images/` and can be passed to `fid.py`.

```bash
python fid.py \
  --mode calc \
  --image_path results/fid/baseline/images \
  --ref_path /path/to/imagenet256_reference_stats.npz \
  --inception_path /path/to/inception-2015-12-05.pkl \
  --num_expected 10000 \
  --batch 64
```

The supplied patch also repairs the original `fid.py` CLI argument handling.

## 6. Custom save policy

Use `--save-mode custom` with any combination of:

- `--save-xt`
- `--save-x0-latents`
- `--save-x0-images`
- `--save-routing-masks`
- `--save-final-latents`
- `--save-final-images-tensor`
- `--save-final-png`

For example, to retain only decoded per-step X0 tensors and final latents:

```bash
--save-mode custom --save-x0-images --save-final-latents
```

## 7. Pairing and routing-mask determinism

Initial latent noise is generated independently from each record's seed, so it
does not depend on batch size or processing order.

Routing randomness is deterministically derived from:

- the global `--mask-seed`;
- the sample's latent seed;
- the sample's class index;
- the EDM generation-step index.

Therefore the same sample and step receive the same random token ordering across
Setup 1 and Setup 2. Changing only the mask ratio with the same seed produces
nested retained-token subsets rather than unrelated masks.

Within one EDM step, the mask is reused for the Euler and Heun-corrector model
evaluations. Setup 1 resamples it at the next generation step.

## 8. Ratio convention

`mask_ratio` is the fraction of tokens that skip the configured TREAD route:

- `mask_ratio=0`: all tokens are used; dense baseline;
- `mask_ratio=0.5`: half the tokens are routed out;
- `mask_ratio=1`: invalid because it keeps zero tokens.

If another document uses `keep_ratio=1` for the dense case, convert it with:

```text
mask_ratio = 1 - keep_ratio
```

## 9. Resume and reproducibility

Pass `--resume` to skip complete tensor shards or complete PNG batches. The
runner verifies that the existing `run_manifest.json` exactly matches the new
invocation, including the SHA-256 hash of the seed-pair file.

The runner currently requires `S_churn=0`, ensuring that routing is the only
per-step stochastic intervention. Fixed checkpoint, pair file, mask seed, and
configuration give repeatable paired inputs subject to normal GPU kernel
reproducibility limitations.
