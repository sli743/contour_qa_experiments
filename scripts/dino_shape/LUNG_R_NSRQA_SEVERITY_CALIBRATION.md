# Lung_R NSR-QA severity calibration

Re-estimates the NSR-QA corruption severities from Lung_R's own real errors,
using exactly the method behind the Lung_L results in
`lung_l_nsrqa_corruption_ablation_month_temporal_v1`:

| Stage | Lung_L (used) | Lung_R (this) |
|---|---|---|
| Error fields | `lung_l_error_field_characterization_v1` | `lung_r_error_field_characterization_v1` |
| Severity calibration | `lung_l_nsrqa_severity_calibration_v1` | `lung_r_nsrqa_severity_calibration_v1` |
| Script | `analysis/calibrate_lung_l_nsrqa_severity.py` | same script, `roi: Lung_R` |

Not used for either: `fit_lung_l_nsrqa_synthetic_severity*.py`
(`synthetic_severity_fit_v1`, `v3_family_validated`) and
`calibrate_lung_corruption_severity.py` (`lung_l_corruption_severity_calibration_v1`).

Base config is the same Lung_L v3 NSR-QA config, so families, clean/branch
probabilities, correlation settings and trainer settings are unchanged; only the
severity ranges and the ROI data block differ.

## Run

```bash
bash scripts/dino_shape/submit_lung_r_nsrqa_severity_calibration.sh
```

Then check `severity_calibration.csv`, `real_vs_synthetic_quantiles.csv` and
`example_bank/` (Slicer `TrainingCorruptionExampleViewer`, groups
`Lung_R_low/typical/high`), and train both arms on the Month-temporal split:

```bash
CALIBRATED_CONFIG=/scratch/shaohan/contour_QA/lung_r_nsrqa_severity_calibration_v1/recommended_training_config.yaml \
STUDY_ROOT=/scratch/shaohan/contour_QA/lung_r_nsrqa_corruption_ablation_month_temporal_rcal_v1 \
bash scripts/dino_shape/submit_lung_r_nsrqa_month_temporal_ablation.sh
```

## v4: realized-metric fit (recommended over v1 severities)

The v1 severities above produce corruptions ~10x milder than real errors
(Lung_R synthetic 1-Dice P90 0.0024 vs real 0.023; HD99 P90 1.5 vs 8.8 mm).
v4 reuses the Lung_L v3 family-validated fit (which on Lung_L Month 3 improved
HD95 ranking: OOD one-sided p 0.085 -> 0.018) with two fixes:

- dropout targets = each patient's deepest missing component >= 3 mm deep
  (v3: largest component of any depth; mostly 1-voxel slivers, 17% no-op draws);
- HD99 fitted with weight 1 (v3: audit only).

```bash
bash scripts/dino_shape/submit_lung_r_nsrqa_synthetic_severity_fit_v4.sh
```

Check `calibration_status.json` and `real_vs_synthetic_quantiles.csv`, then train
with `best_fit_training_config.yaml` (written only if every family passes the
<=5% no-op audit; otherwise `best_candidate_training_config.yaml`):

```bash
CALIBRATED_CONFIG=/scratch/shaohan/contour_QA/lung_r_nsrqa_synthetic_severity_fit_v4/best_fit_training_config.yaml \
STUDY_ROOT=/scratch/shaohan/contour_QA/lung_r_nsrqa_corruption_ablation_month_temporal_fit_v4 \
bash scripts/dino_shape/submit_lung_r_nsrqa_month_temporal_ablation.sh
```
