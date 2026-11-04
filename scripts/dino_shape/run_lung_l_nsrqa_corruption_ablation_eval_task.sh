#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_v1}"
TASK_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
MODE="${MODE:-autoseg_centered}"
if [[ -z "${TASK_INDEX}" ]]; then
  echo "Usage: $0 TASK_INDEX (0..5)" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
MANIFEST="${STUDY_ROOT}/experiment_manifest.csv"
readarray -t FIELDS < <(python - "${MANIFEST}" "${TASK_INDEX}" <<'PY'
import csv, sys
path, idx = sys.argv[1], int(sys.argv[2])
with open(path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
r = rows[idx]
for key in ("arm", "seed", "config", "checkpoint"):
    print(r[key])
PY
)
ARM="${FIELDS[0]}"
SEED="${FIELDS[1]}"
CONFIG="${FIELDS[2]}"
CHECKPOINT="${FIELDS[3]}"
INPUT_ROOT="${STUDY_ROOT}/actual_autoseg_preprocessed"
OUT="${STUDY_ROOT}/evaluations/${ARM}/seed${SEED}"
mkdir -p "${OUT}"

echo "[eval] task=${TASK_INDEX} arm=${ARM} seed=${SEED} mode=${MODE}"
python -m evaluation.actual_autoseg.eval_actual_autoseg_nsrqa \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --input-root "${INPUT_ROOT}" \
  --roi Lung_L \
  --mode "${MODE}" \
  --output-csv "${OUT}/actual_autoseg_testing.csv" \
  --case-output-root "${OUT}/actual_autoseg_cases" \
  --model-name "${ARM}_seed${SEED}" \
  --corruption-label "${ARM}"

python -m evaluation.actual_autoseg.eval_clean_approved_reconstruction \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --input-root "${INPUT_ROOT}" \
  --roi Lung_L \
  --mode "${MODE}" \
  --output-csv "${OUT}/clean_approved_testing.csv" \
  --case-output-root "${OUT}/clean_approved_cases" \
  --model-name "${ARM}_seed${SEED}" \
  --corruption-label "${ARM}"
