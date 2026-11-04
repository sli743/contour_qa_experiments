# SpinalCord NSR-QA four-level diagnostics

This workflow diagnoses *where* the previous-generation and v4 SpinalCord NSR-QA pipeline fails without retraining either model.

## Levels

1. **Clean approved reconstruction** — approved contour is fed unchanged to each network. This measures identity preservation and generalization. Both `approved_reference` and `autoseg_centered` crop geometries are tested. Outputs are compatible with `ActualAutosegNSRQAViewer`.
2. **Shared synthetic-probe correction** — uses the existing deterministic probe bank and reports error reduction / fraction corrected. The optional deep geometry pass also measures endpoint correction and beneficial-vs-harmful voxel changes.
3. **Actual-autoseg correction** — decomposes the reconstruction into beneficial addition, harmful addition, beneficial removal, and harmful removal. Reports missing/excess correction recall, change precision, endpoint residuals, and reconstruction gain.
4. **Final QA signal** — recomputes Dice, ASSD, HD95, **HD99**, max-HD, and endpoint metrics directly from the masks and evaluates QA-disagreement vs true autoseg error. Month1+2 defines P75 thresholds; Month3 is evaluated both with all cases and after removing confirmed/likely incomplete acquisitions.

The external clinical metric called `haus` is **not** assumed to be HD95. The diagnostic HD95/HD99/max-HD definitions are explicit and are recomputed from the binary masks.

## Incomplete acquisition audit

`analysis/audit_spinalcord_incomplete_acquisition.py` audits every unique patient appearing in month1/month2/month3 using the **raw CT acquisition geometry**. It reports:

- raw CT S/I extent;
- approved and autoseg SpinalCord S/I spans;
- approved/autoseg inferior and superior endpoint distances to the raw CT edges;
- sensitive `boundary_limited_review` flag;
- narrower `likely_incomplete` flag requiring boundary limitation plus unusually short CT or approved-cord coverage relative to Month1+2;
- `confirmed_incomplete` for manually verified cases supplied to the script.

Patients 95 and 129 are passed as manually confirmed by the default run script based on the current visual audit.

Outputs:

```text
<COMPARE_ROOT>/diagnostics/acquisition_audit/
  all_months_acquisition_audit.csv
  boundary_limited_review_list.csv
  incomplete_acquisition_patient_list.csv
  incomplete_acquisition_patient_list.txt
  acquisition_counts_by_month.csv
  report.md
```

The broader boundary-limited list is intentionally **not** automatically excluded from QA statistics; only confirmed/likely incomplete cases are removed in the filtered sensitivity analysis.

## Run

No retraining is required. The previous-vs-v4 comparison experiment must already exist.

```bash
bash scripts/dino_shape/submit_spinalcord_nsrqa_diagnostics.sh
```

For an interactive run:

```bash
bash scripts/dino_shape/run_spinalcord_nsrqa_diagnostics.sh
```

Set `FORCE_DIAGNOSTICS=1` to regenerate outputs. Set `SKIP_DEEP_PROBE_GEOMETRY=1` for a faster run that skips the NIfTI-level synthetic endpoint/correction decomposition.

## Main outputs

```text
<COMPARE_ROOT>/diagnostics/four_level_report/report.md
<COMPARE_ROOT>/diagnostics/four_level_report/clean_reconstruction_cases.csv
<COMPARE_ROOT>/diagnostics/four_level_report/synthetic_probe_correction_cases.csv
<COMPARE_ROOT>/diagnostics/four_level_report/synthetic_probe_geometry_diagnostics.csv
<COMPARE_ROOT>/diagnostics/four_level_report/actual_autoseg_correction_diagnostics.csv
<COMPARE_ROOT>/diagnostics/four_level_report/qa_signal_performance_diagnostics.csv
```

Clean viewer roots:

```text
<COMPARE_ROOT>/clean_approved_reconstruction/previous_generation/
<COMPARE_ROOT>/clean_approved_reconstruction/v4_corruption/
```

Point the existing `ActualAutosegNSRQAViewer` at either root.

## Interpretation fork

- Poor clean reconstruction on all splits → decoder/identity-preservation bottleneck.
- Good train clean reconstruction but worse validation/Month3 → generalization bottleneck.
- Strong synthetic correction but weak actual correction → synthetic-to-real transfer/context bottleneck.
- Weak synthetic and actual correction → model/training objective bottleneck rather than further corruption-distribution tuning.
- Good actual correction but weak QA correlation → the reconstruction can be useful but raw input↔reconstruction disagreement is a poor QA readout.
- Large improvement after excluding incomplete acquisition → valid CT support/FOV must be modeled explicitly.

## Actual V4 corruption-family correction diagnostic

The diagnostics job also runs `analysis.eval_spinalcord_v4_family_diagnostics` on
validation1 and testing1 approved shapes. It forces the actual frozen V4 endpoint
families at fixed physical severities, tests the V4 surface families and local
bite, and evaluates endpoint families both in isolation and after a production-
style V4 surface corruption. The identical corrupted masks are reconstructed by
both the previous-generation and V4 models.

Outputs are written to:

```
<COMPARE_ROOT>/diagnostics/v4_family_correction/
  detailed_metrics.csv
  family_summary.csv
  legacy_cut_orientation_audit.csv
  endpoint_fraction_corrected.png
  dice_fraction_corrected.png
  hd99_fraction_corrected.png
  report.md
```

The legacy shared-probe names are not treated as anatomical labels. In the old
probe implementation `cut_upper_half` removes the low array-z endpoint and
`cut_lower_half` removes the high array-z endpoint. The audit derives which is
inferior/superior from each NIfTI's physical geometry and reports anatomical
endpoint labels separately.
