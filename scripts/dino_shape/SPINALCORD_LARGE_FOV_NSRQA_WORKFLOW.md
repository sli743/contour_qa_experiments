# SpinalCord coverage-safe large-FOV NSR-QA experiment

## Purpose

The original SpinalCord study used `96 x 96 x 192` voxels at 1.5 mm
(`144 x 144 x 288 mm`) and centered crops on the foreground centroid. FOV
inspection showed that clinically relevant S/I extent errors can disappear from
the QA input. A first revision (`96 x 96 x 384`) still clipped approved contours
in some cases because a foreground centroid is not the midpoint of a long/curved
structure and because clipping can occur on X/Y as well as Z.

This revision fixes the geometry rather than increasing Z blindly:

- center each training/reference crop on the **approved SpinalCord bounding-box midpoint**;
- treat `96 x 96 x 384` as a minimum FOV;
- by default auto-size one global FOV from **training1 + validation1 only**, adding
  15 mm physical margin per side in X/Y/Z and rounding each dimension up to a
  multiple of 16;
- audit all three splits per axis/per face and fail before training if any raw
  approved foreground remains outside the resolved FOV;
- use **autoseg bounding-box midpoint** localization for the primary clinical
  `autoseg_centered` evaluation, preserving candidate-only/deployment-safe
  localization while avoiding foreground-centroid endpoint bias.

Patient splits, 1.5-mm spacing, 3DINO encoder, decoder definitions, random seed,
training settings, corruption profile, and losses remain unchanged.

## 1. Prepare data and configs

```bash
cd /home/shaohan/contour_qa_experiments
FRESH_DATA=1 bash scripts/dino_shape/prepare_spinalcord_large_fov_nsrqa_study.sh
```

The default shell settings are:

```text
minimum crop = 96 x 96 x 384
spacing      = 1.5 mm
sizing data  = training1 validation1
margin/side  = 15, 15, 15 mm
round-to     = 16 voxels
```

The actual resolved global size is printed and recorded. Outputs are isolated
under:

```text
/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study
```

Before training, inspect:

```text
manifests/spinalcord_fov_sizing_audit.csv
manifests/spinalcord_large_fov_coverage_audit.csv
manifests/spinalcord_large_fov_spec.json
training_corruption_examples/SpinalCord/
```

`spinalcord_fov_sizing_audit.csv` reports raw approved bbox spans and the
per-case crop size required with margin. `spinalcord_large_fov_coverage_audit.csv`
reports per-axis inside fractions, per-face outside voxel counts, and output-face
touch flags for the resolved crop.

Testing1 is **not used to choose the FOV**. If the development-derived FOV clips
a testing1 case, preparation fails and reports the all-split requirement. Do not
silently bypass this check for the primary experiment.

To disable auto-sizing for a sensitivity run:

```bash
AUTO_SIZE=0 CROP_X=128 CROP_Y=128 CROP_Z=384 FRESH_DATA=1 \
  bash scripts/dino_shape/prepare_spinalcord_large_fov_nsrqa_study.sh
```

## 2. Train the same four models

```bash
bash scripts/dino_shape/submit_spinalcord_large_fov_training.sh
```

This submits the same Clean AE, benchmark-overlap DAE, random-morphology DAE,
and NSR-QA models used in the original all-ROI study. The launcher uses the
existing `hugo` partition / `geoffrey_hugo` account job definition and retains
the existing Slurm mail directives.

## 3. Synthetic probe evaluation

After all four checkpoints exist:

```bash
bash scripts/dino_shape/submit_spinalcord_large_fov_evaluation.sh
```

A study-local shared probe bank is used so the old SpinalCord probe bank cannot
be accidentally reused with the revised geometry.

## 4. Real clinical autosegmentation evaluation

After the revised NSR-QA checkpoint exists:

```bash
bash scripts/dino_shape/submit_spinalcord_large_fov_actual_autoseg.sh
```

For this revised study, `autoseg_centered` uses `bbox_midpoint` localization.
The approved contour is still accessed only after that candidate-derived grid is
fixed. `approved_reference` remains the oracle-localized sensitivity analysis.

The generic clinical pipeline keeps `foreground_centroid` as its default, so
rerunning the original all-ROI study is unchanged unless
`AUTOSEG_CENTER_METHOD=bbox_midpoint` is explicitly supplied.

After that job completes:

```bash
STUDY_ROOT=/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study \
REG_ROOT=/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study/actual_autoseg_regression \
  bash scripts/dino_shape/generate_actual_autoseg_regression_summary.sh

bash scripts/dino_shape/submit_spinalcord_large_fov_ranking_classification.sh
```

Then generate the ranking/classification summary/report with the same
`STUDY_ROOT` and `REG_ROOT` environment variables.

## Interpretation

This is now a **coverage-safe FOV/localization ablation**, not a pure Z-size
ablation. The first 96x96x384 attempt established that Z enlargement alone did
not guarantee 3D containment. Compare the original study against this revised
study while explicitly reporting the resolved crop size and centering rule.

## 5. Reviewed-month domain analysis

After `actual_autoseg_regression` has produced both SpinalCord metrics CSVs, run:

```bash
bash scripts/dino_shape/submit_spinalcord_large_fov_domain_analysis.sh
```

This submits two whole-cohort domain analyses using the same trained NSR-QA
checkpoint but different inference-time localization conditions:

- `autoseg_centered`: primary/deployable candidate-only bbox-midpoint localization;
- `approved_reference`: oracle-localization control.

Outputs are written separately under:

```text
${STUDY_ROOT}/nsrqa_domain_analysis_whole_reviewed_months/
  autoseg_centered/
  approved_reference/
```

The existing reviewed-month rule is retained: month1 and patients outside all
three declared month periods are `Unknown`; only month2/month3 are review
eligible, and unrecorded patient/ROI pairs inside that reviewed cohort are
`Normal`. `Unknown` remains descriptive-only. Clinical Dice and Hausdorff/HD95
truth comes from the original Radformation `metrics.csv`, while NSR-QA scores
come from the large-FOV `actual_autoseg_regression/metrics/SpinalCord` outputs.

Because this is `--analysis_scope whole`, training and validation cases are
included for mechanism/domain coverage and the report is explicitly
leakage-exposed. Use it for domain/FOV sensitivity interpretation rather than as
an unbiased test-performance estimate.
