# Lung_L NSR-QA corruption-vs-no-corruption experiment

This experiment tests the original NSR-QA training principle on real Lung_L
autosegmentations before adding any anatomy-specific cut/truncation corruption.

## Arms

1. **nsrqa_corruption** — original final NSR-QA family set and probabilities,
   with only physical severity ranges replaced by the current Lung_L calibration.
2. **no_corruption** — same data, frozen 3DINO encoder, decoder architecture,
   optimizer, loss weights, epochs, validation/checkpoint logic and seeds, but
   `clean_probability=1.0`; therefore no synthetic corruption is inserted.

Both arms use seeds 91, 92 and 93.

The frozen original corruption mixture is:

- clean 0.20
- smooth warp 0.10
- signed-distance noise 0.20
- combined smooth + signed-distance noise 0.15
- stochastic regional foreground dropout 0.20
- continuous global signed-distance boundary displacement 0.15

No directional cut, deterministic erosion/dilation, translation, bite or blob
probe is added to training.

## Run

```bash
bash scripts/dino_shape/submit_lung_l_nsrqa_corruption_ablation.sh
```

The pipeline prepares six configs, preprocesses testing1 real autosegmentations
once with autoseg-centered localization, trains six models (two arms x three
seeds), evaluates both actual autosegs and clean approved contours, and then
aggregates the rankings.

Study root:

```text
/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_v1
```

Primary score for initial ranking is:

```text
1 - Dice(real autoseg, model reconstruction)
```

HD95, ASSD and relative-volume disagreement are retained as secondary signals.
Approved contours are also passed through every model so that autoseg QA can be
contrasted with each patient's approved-contour identity reconstruction.

No metric threshold is used to label a testing patient good/bad in this stage.
The postprocess step writes `analysis/inspection_shortlist.csv` containing the
top 20% by each arm and the largest arm-to-arm rank disagreements.  Manual
contour-behavior/domain labels should be assigned only after these scores are
frozen.
