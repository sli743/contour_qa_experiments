# Lung_L NSR-QA realized synthetic-severity fit

This stage fits corruption **severity only**. It keeps the original final NSR-QA
corruption families and their sampling probabilities fixed.

## Fit criterion

For each candidate severity setting, the exact corruption implementation is run
on a common set of training patients/branches/random seeds. Intentional clean
samples are excluded from severity fitting. Non-clean draws that produce no
voxel change are penalized.

The objective matches P25/P50/P75/P90 of realized synthetic errors to the same
quantiles of real approved-vs-autoseg errors, with each residual normalized by
the robust spread of that real metric.

For Lung_L the fitted metrics/weights are:

- 1-Dice: 1
- ASSD: 1
- **HD95: 2**
- HD99: **0 (audit only)**
- missing fraction: 1
- excess fraction: 1

Only four multipliers are optimized: smooth warp, signed-distance noise,
regional dropout, and global boundary offset. The combined corruption inherits
the fitted smooth/noise scales.

## Run

```bash
bash scripts/dino_shape/submit_lung_l_nsrqa_synthetic_severity_fit.sh
```

Main output root:

```text
/scratch/shaohan/contour_QA/lung_l_nsrqa_synthetic_severity_fit_v1
```

Inspect `report.md`, `real_vs_synthetic_quantiles.csv`, the plots under
`figures/`, and `example_bank/` in the existing `TrainingCorruptionExampleViewer`
before using `best_fit_training_config.yaml` for model training.
