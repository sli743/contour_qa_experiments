#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_month_temporal_v1}"
TEMPORAL_STUDY_ROOT="${TEMPORAL_STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study}"
SPLIT_CONFIG="${SPLIT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/month_temporal_nsrqa_split.yaml}"
CALIBRATED_CONFIG="${CALIBRATED_CONFIG:-/scratch/shaohan/contour_QA/lung_l_nsrqa_severity_calibration_v1/recommended_training_config.yaml}"
SEED="${SEED:-91}"
MAX_TRAIN_CONCURRENT="${MAX_TRAIN_CONCURRENT:-2}"
MAX_EVAL_CONCURRENT="${MAX_EVAL_CONCURRENT:-2}"

cd "${PROJECT_ROOT}"
mkdir -p "${STUDY_ROOT}/logs"

EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},TEMPORAL_STUDY_ROOT=${TEMPORAL_STUDY_ROOT},SPLIT_CONFIG=${SPLIT_CONFIG},CALIBRATED_CONFIG=${CALIBRATED_CONFIG},SEED=${SEED}"
jobid() { printf '%s' "$1" | cut -d';' -f1; }

PREP_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/lung_l_nsrqa_month_temporal_ablation_prepare.job)"
PREP="$(jobid "${PREP_RAW}")"
echo "Submitted Month-temporal prepare job: ${PREP}"

# Exactly two tasks: task 0 = NSR-QA corruption, task 1 = no-corruption baseline.
TRAIN_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${PREP}" \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="0-1%${MAX_TRAIN_CONCURRENT}" \
  jobs/lung_l_nsrqa_month_temporal_ablation_train.job)"
TRAIN="$(jobid "${TRAIN_RAW}")"
echo "Submitted two-arm single-seed training array: ${TRAIN}"

EVAL_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${TRAIN}" \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="0-1%${MAX_EVAL_CONCURRENT}" \
  jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)"
EVAL="$(jobid "${EVAL_RAW}")"
echo "Submitted Month3 real-autoseg evaluation array: ${EVAL}"

POST_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${EVAL}" \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/lung_l_nsrqa_month_temporal_ablation_postprocess.job)"
POST="$(jobid "${POST_RAW}")"
echo "Submitted postprocess job: ${POST}"

cat <<EOF

Month-temporal Lung_L ablation submitted
  prepare:     ${PREP}
  train:       ${TRAIN}
  evaluate:    ${EVAL}
  postprocess: ${POST}

Study root:
  ${STUDY_ROOT}

Split semantics:
  training1   = Month1+Month2 development training subset
  validation1 = Month1+Month2 development validation subset
  testing1    = Month3 temporal holdout

Arms / seed:
  NSR-QA corruption, seed ${SEED}
  no-corruption baseline, seed ${SEED}

By default this uses the same calibrated corruption config as the earlier
lung_l_nsrqa_corruption_ablation_v1 experiment so the split is the main change.
To use a newer validated severity config, override CALIBRATED_CONFIG when submitting.
EOF
