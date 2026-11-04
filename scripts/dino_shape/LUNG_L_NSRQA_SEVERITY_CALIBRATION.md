# Lung_L NSR-QA severity calibration

This stage keeps the previous Lung_L NSR-QA corruption **families and mixture**
fixed and recalibrates only corruption severity from the direct real-error-field
characterization of `training1`.

Run:

```bash
bash scripts/dino_shape/submit_lung_l_nsrqa_severity_calibration.sh
```

Default output:

```text
/scratch/shaohan/contour_QA/lung_l_nsrqa_severity_calibration_v1
```

Important outputs:

- `severity_calibration.csv`: physical P10/P50/P90 severity estimates.
- `recommended_training_config.yaml`: full previous NSR-QA config with only
  severity ranges replaced. Training is **not** launched by this stage.
- `random_training_distribution_audit.csv`: random draws from the exact current
  NSR-QA sampler with the proposed config.
- `real_vs_synthetic_quantiles.csv` and `figures/`: distribution checks.
- `example_bank/`: low / typical / high examples for visual review.

## Slicer inspection

Use the existing `TrainingCorruptionExampleViewer` module and point its
`Examples root` to:

```text
.../lung_l_nsrqa_severity_calibration_v1/example_bank
```

It will expose three groups:

- `Lung_L_low` = real-error P10-derived severity
- `Lung_L_typical` = P50-derived severity
- `Lung_L_high` = P90-derived severity

Each selected patient contains the clean target, smooth warp, signed-distance
noise, combined corruption, stochastic region dropout, inward global offset,
outward global offset, and changed/missing/excess masks.

Do not start the next training run until these examples and the synthetic-vs-real
metric distributions have been inspected. In particular, inspect whether the
combined branch becomes implausibly strong even though its two component ranges
are individually calibrated.
