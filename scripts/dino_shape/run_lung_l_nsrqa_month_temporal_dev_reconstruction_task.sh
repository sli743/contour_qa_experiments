#!/usr/bin/env bash
set -euo pipefail

# Development-set (Month1+Month2 training1/validation1) real-autoseg reconstruction
# for the trained Lung_L month-temporal ablation arms. Diagnostic only: these
# patients were used to fit the shape model, so the outputs show what the model
# learned (e.g. whether it removes stomach/large-bowel gas contoured as lung)
# and are NOT a held-out result.
#
#   prepare      preprocess training1+validation1 raw autosegs (autoseg_centered)
#                into a separate root, leaving the Month3 testing1 inputs untouched
#   TASK_INDEX   run experiment_manifest.csv row TASK_INDEX on those inputs

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_month_temporal_v1}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
MODE="${MODE:-autoseg_centered}"
DEV_INPUT_ROOT="${DEV_INPUT_ROOT:-${STUDY_ROOT}/actual_autoseg_preprocessed_dev}"
DEV_EVAL_ROOT="${DEV_EVAL_ROOT:-${STUDY_ROOT}/evaluations_dev}"
TASK="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${TASK}" ]]; then
  echo "Usage: $0 prepare|TASK_INDEX" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
MANIFEST="${STUDY_ROOT}/experiment_manifest.csv"
[[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 2; }

manifest_fields() {
  python - "${MANIFEST}" "$1" <<'PY'
import csv, sys
path, idx = sys.argv[1], int(sys.argv[2])
with open(path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
r = rows[idx]
for key in ("arm", "seed", "config", "checkpoint"):
    print(r[key])
PY
}

if [[ "${TASK}" == "prepare" ]]; then
  # Same reference config the Month3 preprocessing used (task 0); both arms share data dirs.
  readarray -t FIELDS < <(manifest_fields 0)
  REFERENCE_CONFIG="${FIELDS[2]}"
  echo "[dev-prepare] config=${REFERENCE_CONFIG}"
  echo "[dev-prepare] output=${DEV_INPUT_ROOT}"
  python -m evaluation.actual_autoseg.preprocess_actual_autoseg \
    --config "${REFERENCE_CONFIG}" \
    --roi Lung_L \
    --group lungs_chest \
    --raw-data-root "${RAW_DATA_ROOT}" \
    --output-root "${DEV_INPUT_ROOT}" \
    --mode "${MODE}" \
    --splits training1 validation1

  python - "${DEV_INPUT_ROOT}" "${MODE}" <<'PY'
import csv, sys
from collections import Counter
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
with (root / "manifests" / "Lung_L_preprocess_manifest.csv").open(newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r.get("localization_mode") == mode]
for split in ("training1", "validation1"):
    c = Counter(r.get("status") for r in rows if r.get("split") == split)
    print(f"[dev-prepare] {split}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    for r in rows:
        if r.get("split") == split and r.get("status") != "ok":
            print(f"    patient {r.get('patient_id')}: {r.get('status')} {r.get('reason')}")
PY
  exit 0
fi

readarray -t FIELDS < <(manifest_fields "${TASK}")
ARM="${FIELDS[0]}"
SEED="${FIELDS[1]}"
CONFIG="${FIELDS[2]}"
CHECKPOINT="${FIELDS[3]}"
[[ -s "${CHECKPOINT}" ]] || { echo "Missing checkpoint for ${ARM}: ${CHECKPOINT}" >&2; exit 2; }
[[ -d "${DEV_INPUT_ROOT}/${MODE}/Lung_L" ]] || { echo "Run the prepare step first: ${DEV_INPUT_ROOT}/${MODE}/Lung_L missing" >&2; exit 2; }
OUT="${DEV_EVAL_ROOT}/${ARM}/seed${SEED}"
mkdir -p "${OUT}"

echo "[dev-eval] task=${TASK} arm=${ARM} seed=${SEED} mode=${MODE}"
echo "[dev-eval] checkpoint=${CHECKPOINT}"
python -m evaluation.actual_autoseg.eval_actual_autoseg_nsrqa \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --input-root "${DEV_INPUT_ROOT}" \
  --roi Lung_L \
  --mode "${MODE}" \
  --output-csv "${OUT}/actual_autoseg_dev.csv" \
  --case-output-root "${OUT}/actual_autoseg_cases" \
  --model-name "${ARM}_seed${SEED}" \
  --corruption-label "${ARM}"

python -m evaluation.actual_autoseg.eval_clean_approved_reconstruction \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --input-root "${DEV_INPUT_ROOT}" \
  --roi Lung_L \
  --mode "${MODE}" \
  --output-csv "${OUT}/clean_approved_dev.csv" \
  --case-output-root "${OUT}/clean_approved_cases" \
  --model-name "${ARM}_seed${SEED}" \
  --corruption-label "${ARM}"

echo "[dev-eval] done: ${OUT}"
