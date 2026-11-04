#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_v1}"
TASK_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${TASK_INDEX}" ]]; then
  echo "Usage: $0 TASK_INDEX (0..5)" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
MANIFEST="${STUDY_ROOT}/experiment_manifest.csv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing experiment manifest: ${MANIFEST}" >&2
  exit 1
fi

readarray -t FIELDS < <(python - "${MANIFEST}" "${TASK_INDEX}" <<'PY'
import csv, sys
path, idx = sys.argv[1], int(sys.argv[2])
with open(path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
if idx < 0 or idx >= len(rows):
    raise SystemExit(f"task index {idx} outside 0..{len(rows)-1}")
r = rows[idx]
for key in ("arm", "seed", "config", "checkpoint"):
    print(r[key])
PY
)
ARM="${FIELDS[0]}"
SEED="${FIELDS[1]}"
CONFIG="${FIELDS[2]}"
CHECKPOINT="${FIELDS[3]}"

echo "[train] task=${TASK_INDEX} arm=${ARM} seed=${SEED}"
python -m training.train_dino_shape_stochastic_region_dropout_decoder \
  --config "${CONFIG}" --fresh

test -f "${CHECKPOINT}" || { echo "Missing trained checkpoint: ${CHECKPOINT}" >&2; exit 1; }
