# Lung_L corruption-severity calibration

This analysis keeps the historical corruption types fixed and calibrates their
severity to **Lung_L training1** real approved-vs-autoseg errors. It never uses
Month3/testing1 to choose corruption ranges.

## Run

```bash
bash scripts/dino_shape/submit_lung_l_corruption_severity_calibration.sh
```

Default output:

```text
/scratch/shaohan/contour_QA/lung_l_corruption_severity_calibration_v1/
```

Important outputs:

- `real_error_features.csv`
- `severity_sweep.csv`
- `case_family_matches.csv`
- `family_severity_recommendations.csv`
- `recommended_corruption_ranges.yaml`
- `size_conditioning_analysis.csv`
- `cut_direction_orientation_audit.csv`
- `synthetic_validation_bank_features.csv`
- `same_patient_nn_coverage.csv`
- `real_vs_synthetic_classifier.csv`
- `report.md`
- `inspection/` (Slicer-compatible cases)

## Cut direction

Historical benchmark names are retained for reproducibility, but they are not
used by this calibration. On an LPS-canonical NumPy `[z,y,x]` array:

- `cut_inferior_portion`: remove the low-z foreground extent.
- `cut_superior_portion`: remove the high-z foreground extent.
- `cut_posterior_portion`: remove the high-y foreground extent.

The source orientation of every training case is audited in
`cut_direction_orientation_audit.csv`. This explicitly records what the legacy
`cut_upper_half`, `cut_lower_half`, and `cut_posterior_half` names actually do.

## Inspection

Load `slicer/LungCorruptionCalibrationViewer.py` as a scripted module in 3D
Slicer. Point it to the output `inspection/` directory, then choose patient,
family, and `best`/`lower`/`upper` severity.

Do not turn `recommended_corruption_ranges.yaml` into a training distribution
until the numerical matches have been visually reviewed.
