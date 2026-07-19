# TREAD deviation evaluation

## What the decoded trajectory means

For the linear flow path used by this checkpoint,

```text
Xt = (1-t) X0 + t epsilon
V  = epsilon - X0
```

therefore the clean-latent estimate implied by a velocity prediction is

```text
X0_pred = Xt - t V_pred
```

The saved `x0_decoded_uint8` tensor is the SD-VAE decode of `X0_pred`. It is not
a decode of noisy `Xt`, and it is not a ground-truth intermediate image. It is
the model's current estimate of the clean endpoint. CLIP/DINO comparisons of
this trajectory are meaningful as model-belief/semantic deviation measures,
but should be reported together with latent and pixel deviations.

For Euler runs, the evaluator also reconstructs the saved guided velocity as
`(Xt-X0_pred)/t`. For Heun runs, most saved X0 predictions are evaluated at the
temporary Euler predictor state, which is not currently saved, so the velocity
cannot be reconstructed exactly from the existing shards.

## Basic evaluation

```bash
python compute_deviation_suite.py \
  --suite-root runs/full_suite_32steps
```

This requires paired-mode tensor shards. It automatically discovers:

```text
baseline/
setup1_every_step/keep_*/
setup2_single_step/keep_*/step_*/
```

and writes:

```text
deviation/
  deviation_manifest.json
  deviation_summary.json
  step_metrics.csv
  setup2_matrices.pt
  per_run/*.pt
```

## Optional DINOv2 and CLIP metrics

Download Transformers-format models locally from ModelScope:

```bash
modelscope download \
  --model facebook/dinov2-base \
  --local_dir ./parameters/dinov2-base

modelscope download \
  --model openai-mirror/clip-vit-base-patch32 \
  --local_dir ./parameters/clip-vit-base-patch32
```

Then run:

```bash
python compute_deviation_suite.py \
  --suite-root runs/full_suite_32steps \
  --dinov2-model ./parameters/dinov2-base \
  --clip-model ./parameters/clip-vit-base-patch32 \
  --feature-batch-size 64
```

The feature models use `local_files_only=True` and do not access Hugging Face.

## Main modalities

- `xt_latent`: deviation of the actual ODE states;
- `x0_latent`: deviation of `Xt-t*Vpred`;
- `velocity_reconstructed`: exact reconstruction for saved Euler trajectories;
- `x0_image_pixel`: decoded-X0 pixel deviation and PSNR;
- `x0_image_dinov2` / `x0_image_clip`: semantic feature deviation;
- `final_latent` and `final_image_*`: final-output deviations.

For Setup 2, `setup2_matrices.pt` contains matrices whose rows are intervention
steps and whose columns are evaluation steps, ready for heatmap plotting.
