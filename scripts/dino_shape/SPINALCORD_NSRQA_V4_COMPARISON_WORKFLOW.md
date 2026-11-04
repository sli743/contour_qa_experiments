# SpinalCord previous-generation vs v4 NSR-QA comparison

## Scientific question

Does replacing only the previous generic NSR-QA training corruption distribution
with the development-calibrated SpinalCord v4 distribution improve QA signals on
**actual held-out Month3 autosegmentations**?

The primary outcome is not synthetic-probe performance. Synthetic probes are a
secondary mechanism check shared identically by the two models.

## Frozen design

Source study:

`/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study`

The comparison reuses its:

- Month1+Month2 development / Month3 testing split;
- coverage-safe 1.5-mm SpinalCord FOV and bbox-midpoint localization policy;
- frozen pretrained 3DINO encoder;
- decoder channels `[256, 128, 64, 32]`;
- symmetric changed-region loss;
- optimizer, learning rate, epochs, early stopping, threshold, and seeds.

Only the corruption backend differs:

- `previous_generation`: existing `stochastic_region_dropout` NSR-QA sampler.
- `v4_corruption`: `spinalcord_proposed_distribution_v4.yaml` through the generic
  corruption dispatcher.

Before training, the preparer also audits longitudinal FOV headroom against the
v4 inferior-overextension support using **development rows only**. It fails if
even the configured lower bound cannot fit. If only the extreme upper tail can
touch the crop face, it records a warning in
`manifests/v4_endpoint_fov_audit.json` rather than changing the FOV or sampler
using Month3 information.

Actual autosegmentations are not training targets. The neural network is still
trained to reconstruct approved contours from synthetic corruptions.

## Primary actual-autoseg evaluation

Primary localization: `autoseg_centered`.

Primary Month3 endpoints:

1. P75 high-error AUROC/AUPRC for true `1-Dice` and true HD95. P75 cutoffs are
   estimated from training1+validation1 truth and frozen before testing1.
2. Spearman association of corresponding raw QA disagreement with true error:
   `qa_1_minus_dice` vs `true_1_minus_dice` and `qa_hd95_mm` vs `true_hd95_mm`.

Old-vs-v4 uncertainty uses paired patient bootstrap resampling (default 5000).
P50 classification, fixed development-fit regression, ASSD, relative volume,
and `approved_reference` localization are secondary/sensitivity analyses.

When the full Radformation `metrics.csv` is supplied, original full-volume Dice
and HD95 replace crop-derived truth for those endpoints, while the QA signal is
still computed from the deployment crop/reconstruction.

## Domain robustness

If `OOD_Domain_Assignment_MAD_TOTAL.csv` is available, the paired report adds:

- QA-score distributions by domain;
- within-domain true-error vs QA-signal Spearman association;
- within-domain P75 classification where both classes exist;
- exploratory one-vs-rest domain separability of each scalar QA signal;
- Kruskal-Wallis score-distribution tests across domains.

Domain labels are never used in shape-model training. Domain analysis is
secondary/exploratory because some domains are small and labels come from the
OOD-review process.

## 3D Slicer outputs

Each model gets a separate self-contained ActualAutosegNSRQAViewer root:

```
<COMPARE_ROOT>/actual_autoseg_results/previous_generation/
<COMPARE_ROOT>/actual_autoseg_results/v4_corruption/
```

Each case contains exactly the viewer-compatible files:

```
ct.nii.gz
approved.nii.gz
autoseg_input.nii.gz
nsrqa_reconstruction.nii.gz
missing_autoseg_vs_approved.nii.gz
excess_autoseg_vs_approved.nii.gz
missing_input_vs_reconstruction.nii.gz
excess_input_vs_reconstruction.nii.gz
metrics.json
```

`metrics.json` additionally records model/config/checkpoint provenance.

## Run

The source month-temporal study must already be prepared and audited.

```bash
bash scripts/dino_shape/submit_spinalcord_nsrqa_v4_comparison_pipeline.sh
```

To force both models to retrain from scratch:

```bash
FRESH_TRAINING=1 bash scripts/dino_shape/submit_spinalcord_nsrqa_v4_comparison_pipeline.sh
```

To rerun inference/reporting while retaining checkpoints:

```bash
FORCE_EVAL=1 bash scripts/dino_shape/submit_spinalcord_nsrqa_v4_comparison_pipeline.sh
```

The default comparison root is:

`/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare`

Primary report:

`reports/previous_vs_v4_actual_autoseg/report.md`

Each arm also receives the existing regression, P75/P50 ranking/classification,
and ranking/classification report outputs under its own actual-autoseg result
root.

## Probe-bank inspection change

`analysis/export_spinalcord_v3_probe_examples.py` now treats `endpoint_matched`
as a strict retrieval category. A random-bank sample must:

1. contain the same dominant endpoint error direction as the real autoseg, and
2. fall within `max(3 mm, 35% of the real endpoint magnitude)` by default,

before the full phenotype distance is used for ranking. If the finite random
bank has no such sample, the `endpoint_matched` category is intentionally empty;
forced coverage examples remain the correct way to test whether the generator
can represent that phenotype.
