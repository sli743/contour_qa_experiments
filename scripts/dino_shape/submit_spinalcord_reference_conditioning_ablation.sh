#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
SOURCE_COMPARE_ROOT="${SOURCE_COMPARE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_reference_ablation}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
FRESH_TRAINING="${FRESH_TRAINING:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
N_BOOT="${N_BOOT:-5000}"
SEED="${SEED:-88}"
EXCLUDE_PATIENTS="${EXCLUDE_PATIENTS:-95,129,145}"
TRAIN_TIME="${TRAIN_TIME:-2-00:00:00}"
EVAL_TIME="${EVAL_TIME:-1-00:00:00}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python scripts/dino_shape/prepare_spinalcord_reference_conditioning_ablation.py \
  --source-compare-root "${SOURCE_COMPARE_ROOT}" \
  --output-root "${REFERENCE_ROOT}" \
  --raw-data-root "${RAW_DATA_ROOT}"

mkdir -p "${REFERENCE_ROOT}/logs"
train_export="ALL,PROJECT_ROOT=${PROJECT_ROOT},REFERENCE_ROOT=${REFERENCE_ROOT},FRESH_TRAINING=${FRESH_TRAINING}"
train_raw="$(sbatch --parsable \
  -p hugo --account=geoffrey_hugo --time="${TRAIN_TIME}" \
  --mail-type=ALL --mail-user=shaohan@wustl.edu \
  --output="${REFERENCE_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${REFERENCE_ROOT}/logs/slurm-%x-%A_%a.err" \
  --export="${train_export}" \
  jobs/spinalcord_reference_conditioning_train.job)"
train_id="${train_raw%%;*}"

eval_export="ALL,PROJECT_ROOT=${PROJECT_ROOT},REFERENCE_ROOT=${REFERENCE_ROOT},RAW_DATA_ROOT=${RAW_DATA_ROOT},FORCE_EVAL=${FORCE_EVAL},N_BOOT=${N_BOOT},SEED=${SEED},EXCLUDE_PATIENTS=${EXCLUDE_PATIENTS}"
eval_raw="$(sbatch --parsable \
  -p hugo --account=geoffrey_hugo --time="${EVAL_TIME}" \
  --dependency="afterok:${train_id}" \
  --mail-type=ALL --mail-user=shaohan@wustl.edu \
  --output="${REFERENCE_ROOT}/logs/slurm-%x-%j.out" \
  --error="${REFERENCE_ROOT}/logs/slurm-%x-%j.err" \
  --export="${eval_export}" \
  jobs/spinalcord_reference_conditioning_eval.job)"
eval_id="${eval_raw%%;*}"

cat <<EOF
Submitted SpinalCord reference-conditioning ablation.

Training array: ${train_id}_[0-3]
  0 centered
  1 centered + spatial context
  2 centered + FOV jitter
  3 centered + spatial context + FOV jitter
Evaluation job (after all training succeeds): ${eval_id}

Study root: ${REFERENCE_ROOT}
Primary report after evaluation:
  ${REFERENCE_ROOT}/reports/reference_conditioning_ablation/report.md
EOF
