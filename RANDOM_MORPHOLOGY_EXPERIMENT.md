# Random-morphology frozen-3DINO denoising experiment

This experiment keeps the pretrained 3DINO encoder frozen and trains only the
existing contour decoder. Training corruptions are generated online from:

- random erosion or dilation (1–4 voxels),
- random translation in each axis (-3 to +3 voxels),
- one random spherical local addition or removal (radius 3–10 voxels),
- 20% clean identity samples.

The fixed probe benchmark is not used to generate training samples. In
particular, half-space cuts and the benchmark's fixed severity grid remain
held out for evaluation.

Run the full Lung_L pipeline with:

```bash
bash scripts/dino_shape/run_dino_shape_random_morphology_full_pipeline_Lung_L_v1.sh
```
