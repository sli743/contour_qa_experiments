# Generic NSR-QA pipeline (any ROI)

The design was frozen on 2026-11-23, after the Lung_L / Lung_R studies. The Lung_L / Lung_R method is used unchanged, with no ROI-specific repairs.

| Part | Setting |
|---|---|
| Model | **C1I1**: frozen 3DINO; CT added through gated token fusion and a zero-initialised full-resolution stem on the last two decoder stages; CT dropout 0.3 (`LUNG_L_CT_CONTOUR_NSRQA.md`) |
| Corruption | The ROI's own fit, using the Lung_L v3 realized-severity method with the same corruption types. The leak branch and the mask stem stay off. |
| Comparison | **C0I0**: contour only, no corruption, same network. `WITH_CT_ONLY=1` adds the CT-only segmenter. `DESIGN=full` runs the whole 2×2 instead. |
| Seeds | 91, 92, 93 |
| Score | Volume of deep model change. The depth cutoff is calibrated per arm by the label-free clean-floor rule. Out-of-scan padding is masked. |
| Evaluation | Region detection with depth-matched truth: AP, ρ, recall@k; region recall split into missing and extra; false regions on approved contours. |

Why this design:
- Lung_L: C1I1 ranks best (ρ 0.41 vs 0.17 for C0I0; AP 0.71).
- Lung_R: it ties C0I0 on ranking, but finds the errors' location (region recall 0.43 vs 0.10).
- The early-fusion mask stem and the leak corruption did not help.
- **Known limit:** extra volume that looks plausible on the CT is not trimmed.

## Run

```bash
STAGE=prep  ROI=Heart bash scripts/dino_shape/submit_nsrqa_roi_pipeline.sh
# check the crop audit + fit report / example bank, then
STAGE=train ROI=Heart bash scripts/dino_shape/submit_nsrqa_roi_pipeline.sh
```

`prep` runs these steps (CPU only):
1. `analysis.make_roi_pipeline_configs` writes `<roi>_error_field_characterization_v1/config.yaml` and `<roi>_nsrqa_synthetic_severity_fit_v3method/config.yaml`. Both are the Lung_R configs with only the ROI, names, data paths and output directories replaced.
2. The crop clipping audit runs, writing to `<roi>_fov_clipping_audit_v1`.
3. Error-field characterization runs on training1 only.
4. The severity fit runs.
5. The contour-only study is prepared, including the Month3 autoseg-centred crops of the ROI's crop group.

**Starting point of the fit:** the ROI's month-temporal all-ROI config (`all_roi_NSRQA_month_temporal_study/configs/<ROI>/nsrqa.yaml`). It has the same corruption mix and training recipe as the Lung_L v3 fallback config, with that ROI's anatomy-profile severities.

**Note:** the lungs started from a first-pass severity calibration instead. The fit config allows either starting point.

**End-truncation guard**, the one addition to the lung method:
- **Why:** truncation is a held-out error type. On Esophagus, about 1 in 5 fitted dropouts, and some noise draws (which keep only the largest piece), cut a whole end segment off: HD95 15–109 mm, against a real 95th percentile of 8 mm.
- **What it does:** any draw that shortens the structure's top-to-bottom extent by more than the limit is redrawn, up to 8 times, then left clean.
- **The limit:** the ROI's real 95th-percentile endpoint error (`max_abs_endpoint_error_mm`, at least 3 mm): about 6 mm for Esophagus, 21 mm for Heart.
- **Where:** the guard lives in `training/corruptions/stochastic_region_dropout.redraw_if_truncated` and is used by both the fit (`forced_branch`) and training.
- **Configs:** it is enabled by the fit config's `end_truncation_guard`, which stores `max_end_truncation_mm` in the fitted config. Configs without that key are unchanged.

To refit an already prepared ROI (before training):
```bash
REFIT=1 STAGE=prep ROI=Esophagus bash scripts/dino_shape/submit_nsrqa_roi_pipeline.sh
```

`train` (GPU) runs, for each seed:
1. Train and evaluate C1I1 and C0I0: on Month3 autoseg, on the clean approved contour, and C1I1 with the CT off.
2. Per-seed scoring and depth calibration (`<roi>_ct_contour_nsrqa_v1/evaluations_floor_corrected/seed<S>*`).

Then pooled region detection, with and without `EXCLUDE`, writes `region_detection{,_assessable}`.

## Truth settings by anatomy profile (`analysis/make_roi_pipeline_configs.py`)

| Profile | Truth / score depth | Region core | Positive patient | Min component | Score depth grid |
|---|---|---|---|---|---|
| large_solid (Lung, Heart, Liver, Stomach) | 4.5 mm | 0.5 mL | 2.0 mL | 0.05 mL | 3–7.5 mm |
| medium_tubular (Esophagus, Trachea, Aorta, …) | 3.0 mm | 0.1 mL | 0.5 mL | 0.02 mL | 1.5–4.5 mm |

Thin and branching ROIs (Bronchus, Carina, brachial plexus, SpinalCord) are refused for now. At 1.5 mm voxels, "deep change" barely exists in them, and SpinalCord has its own large-FOV preparation.
