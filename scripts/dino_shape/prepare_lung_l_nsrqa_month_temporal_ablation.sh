#!/usr/bin/env bash
set -euo pipefail

# ROI defaults to Lung_L; set ROI=Lung_R (with its own STUDY_ROOT) to prepare the
# same two-arm month-temporal ablation for another lungs_chest ROI.
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ROI="${ROI:-Lung_L}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_month_temporal_v1}"
TEMPORAL_STUDY_ROOT="${TEMPORAL_STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study}"
SPLIT_CONFIG="${SPLIT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/month_temporal_nsrqa_split.yaml}"
CALIBRATED_CONFIG="${CALIBRATED_CONFIG:-/scratch/shaohan/contour_QA/lung_l_nsrqa_severity_calibration_v1/recommended_training_config.yaml}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
SEED="${SEED:-91}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TEMPORAL_SUMMARY="${TEMPORAL_STUDY_ROOT}/manifests/temporal_split_summary.json"
TEMPORAL_ROI_CONFIG="${TEMPORAL_STUDY_ROOT}/configs/${ROI}/nsrqa.yaml"

if [[ ! -s "${TEMPORAL_SUMMARY}" || ! -s "${TEMPORAL_ROI_CONFIG}" ]]; then
  echo "Month-temporal study has not been materialized; preparing it now."
  python scripts/dino_shape/prepare_month_temporal_nsrqa_study.py \
    --split-config "${SPLIT_CONFIG}" \
    --study-root "${TEMPORAL_STUDY_ROOT}"
else
  echo "Reusing existing temporal split: ${TEMPORAL_SUMMARY}"
fi
[[ -s "${TEMPORAL_ROI_CONFIG}" ]] || { echo "Missing temporal ${ROI} config: ${TEMPORAL_ROI_CONFIG}" >&2; exit 2; }

if [[ ! -s "${CALIBRATED_CONFIG}" ]]; then
  echo "Missing calibrated corruption config: ${CALIBRATED_CONFIG}" >&2
  echo "Override CALIBRATED_CONFIG if you want to use a newer validated severity fit." >&2
  exit 2
fi

python -m analysis.prepare_lung_l_nsrqa_month_temporal_ablation \
  --calibrated-config "${CALIBRATED_CONFIG}" \
  --temporal-reference-config "${TEMPORAL_ROI_CONFIG}" \
  --temporal-split-summary "${TEMPORAL_SUMMARY}" \
  --study-root "${STUDY_ROOT}" \
  --seed "${SEED}" \
  --roi "${ROI}"

REFERENCE_CONFIG="${STUDY_ROOT}/configs/nsrqa_corruption_seed${SEED}.yaml"
python -m evaluation.actual_autoseg.preprocess_actual_autoseg \
  --config "${REFERENCE_CONFIG}" \
  --roi "${ROI}" \
  --group lungs_chest \
  --raw-data-root "${RAW_DATA_ROOT}" \
  --output-root "${STUDY_ROOT}/actual_autoseg_preprocessed" \
  --mode autoseg_centered \
  --splits testing1

python - "${STUDY_ROOT}" "${REFERENCE_CONFIG}" "${ROI}" <<'PY'
import csv
import json
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
cfg = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
roi = sys.argv[3]
test_dir = Path(cfg["data"]["test_label_dir"])
expected = len(list(test_dir.glob("*.nii.gz")))
manifest = root / "actual_autoseg_preprocessed" / "manifests" / f"{roi}_preprocess_manifest.csv"
with manifest.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
test_rows = [r for r in rows if r.get("split") == "testing1" and r.get("localization_mode") == "autoseg_centered"]
ok = [r for r in test_rows if r.get("status") == "ok"]
skipped = [r for r in test_rows if r.get("status") != "ok"]
if expected < 40:
    raise SystemExit(f"ERROR: only {expected} {roi} testing1 labels; temporal Month3 remap was not used")
if not ok:
    raise SystemExit(f"ERROR: no Month3 {roi} actual-autoseg cases were successfully preprocessed")
audit = {
    f"expected_month3_{roi.lower()}_labels": expected,
    "preprocessed_ok": len(ok),
    "preprocessed_skipped": len(skipped),
    "skipped": [{"patient_id": r.get("patient_id"), "status": r.get("status"), "reason": r.get("reason")} for r in skipped],
}
(root / "month3_actual_autoseg_preprocess_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(f"Month3 {roi} evaluation cohort: labels={expected}, preprocessed_ok={len(ok)}, skipped={len(skipped)}")
PY
