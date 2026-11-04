# All-ROI NSR-QA workflow

1. Run `prepare_all_roi_nsrqa_study.py`. Any ROI missing a required training/validation/testing label or CT directory is recorded in `skipped_rois.tsv` and omitted.
2. Generate the mandatory one-patient standardized NSR-QA training-corruption examples for every active ROI and inspect them in Slicer **before** training.
3. Submit the four-model training Slurm array. The study trains Clean AE, benchmark-overlap DAE, Random Morphology DAE, and NSR-QA from scratch under a common study root.
4. After all checkpoints exist, submit the per-ROI evaluation/report array. Every ROI uses the same deterministic probe-bank format. NSR-QA evaluation exports ProbeContourComparer-compatible masks plus `original_ct_patient_<id>.nii.gz`.
5. Generate the cross-ROI Markdown summary after all per-ROI reports finish.

Bad-segmentation thresholds are predefined by anatomy profile: large solid 20% volume / 10 mm local HD95; medium tubular 15% / 6 mm; thin/branching 10% / 4 mm; complex multi-component 15% / 6 mm.

## Actual-autoseg regression experiment

The clinical-autoseg regression pipeline is separate from the deterministic
synthetic-probe benchmark and uses **NSR-QA only** as the QA model by default.

Primary localization is `autoseg_centered`: the fixed-size crop center is built
from available candidate/autoseg center ROIs, and the approved contour is only
resampled after that grid has been fixed. `approved_reference` reuses the
existing approved-derived crop and is retained as an oracle-localization
sensitivity analysis.

For each ROI the pipeline:

1. preprocesses raw autoseg contours for `training1`, `validation1`, and
   `testing1` in both localization modes;
2. runs the frozen NSR-QA checkpoint and saves CT, approved, autoseg input,
   reconstruction, and missing/excess diagnostic masks per patient;
3. computes true error (autoseg vs approved) and QA disagreement (autoseg vs
   NSR-QA reconstruction) for 1-Dice, HD95, ASSD, and relative volume change;
4. fits the primary simple regression on `training1 + validation1` and evaluates
   the frozen fit on `testing1`;
5. also reports a `validation1 -> testing1` sensitivity fit and 4x4 exploratory
   cross-metric correlations.

Submit all active ROIs with:

```bash
bash scripts/dino_shape/submit_actual_autoseg_regression.sh
```

The default output root is:

```text
/scratch/shaohan/contour_QA/all_roi_NSRQA_study/actual_autoseg_regression
```

After all ROI jobs complete, aggregate ROI-level summaries without pooling
patient observations across organs:

```bash
bash scripts/dino_shape/generate_actual_autoseg_regression_summary.sh
```

The Slicer module `ActualAutosegNSRQAViewer` browses the saved cases by
localization mode, ROI, split, and patient and displays the per-case metrics.
