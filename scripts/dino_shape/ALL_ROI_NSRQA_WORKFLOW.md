# All-ROI NSR-QA workflow

1. Run `prepare_all_roi_nsrqa_study.py`. Any ROI missing a required training/validation/testing label or CT directory is recorded in `skipped_rois.tsv` and omitted.
2. Generate the mandatory one-patient standardized NSR-QA training-corruption examples for every active ROI and inspect them in Slicer **before** training.
3. Submit the four-model training Slurm array. The study trains Clean AE, benchmark-overlap DAE, Random Morphology DAE, and NSR-QA from scratch under a common study root.
4. After all checkpoints exist, submit the per-ROI evaluation/report array. Every ROI uses the same deterministic probe-bank format. NSR-QA evaluation exports ProbeContourComparer-compatible masks plus `original_ct_patient_<id>.nii.gz`.
5. Generate the cross-ROI Markdown summary after all per-ROI reports finish.

Bad-segmentation thresholds are predefined by anatomy profile: large solid 20% volume / 10 mm local HD95; medium tubular 15% / 6 mm; thin/branching 10% / 4 mm; complex multi-component 15% / 6 mm.
