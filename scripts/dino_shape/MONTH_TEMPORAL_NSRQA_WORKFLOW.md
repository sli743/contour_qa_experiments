# Month-temporal all-ROI NSR-QA study

This is a parallel reproduction of the August-13 all-ROI four-model study with
only the patient partition changed.  It intentionally reuses the original
training/evaluation/analysis modules so output CSVs, plots, Markdown reports,
and directory names remain comparable. The ROI set is locked to the same 16
completed August-13 ROIs (64 configs total); `NS_Chestwall` remains excluded.

## Split

The month lists are defined in
`configs/dino_shape/month_temporal_nsrqa_split.yaml`.

- `testing1` = **month3**, held out from all model fitting/selection.
- `training1 + validation1` = month1 + month2.
- Validation is sampled separately within month1 and month2 at 20%, seed 88.
- Patient 16 (month1/month3) and patient 150 (month2/month3) are assigned to
  **month3/testing1 only**.  They are excluded from development to prevent
  patient leakage.

The compatibility names `training1`, `validation1`, and `testing1` are kept on
purpose.  Existing regression/ranking code therefore produces exactly the same
columns and report format, while `testing1` now means the month3 temporal
holdout.

Default new study root:

```text
/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study
```

The old study `/scratch/shaohan/contour_QA/all_roi_NSRQA_study` is not modified.

## 1. Prepare split, configs, manifests, and training examples

```bash
cd /home/shaohan/contour_qa_experiments
bash scripts/dino_shape/prepare_month_temporal_nsrqa_study.sh
```

The temporal preprocessed tree is symlink-only and lives under the new study:

```text
all_roi_NSRQA_month_temporal_study/
  data/preprocessed_roiwise_allrois_1p5mm_nozoom_temporal/
  configs/<ROI>/{pilot,benchmark,random_morphology,nsrqa}.yaml
  manifests/
  training_corruption_examples/
```

Inspect `training_corruption_examples/` before training, as in the original
workflow.  The split audit is in:

- `manifests/temporal_split_patients.tsv`
- `manifests/temporal_split_overlap_resolution.tsv`
- `manifests/temporal_roi_split_counts.tsv`
- `manifests/temporal_source_file_audit.tsv`
- `manifests/temporal_split_summary.{json,md}`
- `manifests/temporal_reference_registry.yaml` (the frozen 16-ROI registry used for this rerun)

## 2. Submit the full pipeline

```bash
bash scripts/dino_shape/submit_month_temporal_full_pipeline.sh
```

Every Slurm submission explicitly specifies partition `hugo` and account
`geoffrey_hugo`; the underlying job files retain the same settings and mail
directives.

The dependency graph is:

```text
four-model training
  |-- shared-probe four-model evaluation -> all-ROI probe summary
  `-- clinical actual-autoseg metrics/regression -> regression summary
                                           `-----> ranking/classification -> ranking report
                                                                      `-----> domain report (if installed)
```

## Output structure

The new root mirrors the original study:

```text
all_roi_NSRQA_month_temporal_study/
  configs/
  manifests/
  models/
    pilot/
    benchmark_overlap/
    random_morphology/
    nsrqa/
  shared_probe_banks/
  evaluations/<ROI>/...
  reports/<ROI>_four_models_NSRQA_shared_probe/
  all_roi_summary/
    all_roi_pooled_summary.csv
    all_roi_probe_summary.csv
    report.md
  actual_autoseg_regression/
    autoseg_centered/
    approved_reference/
    metrics/<ROI>/
      autoseg_centered_metrics.csv
      approved_reference_metrics.csv
    regression/<ROI>/
    summary/
      all_roi_regression_summary.csv
      primary_autoseg_centered_testing1_summary.csv
      primary_cross_roi_medians.csv
    ranking_classification/<ROI>/
      ranking_summary.csv
      classification_summary.csv
      ranking_cases.csv
      classification_cases.csv
    ranking_classification_summary/
    ranking_classification_report/
      report.md
  domain_analysis/                 # when the domain-analysis patch is present
  logs/
```

Because the original analysis scripts are reused, fields containing `testing1`
remain named that way.  In this study they must be interpreted as **month3**.
