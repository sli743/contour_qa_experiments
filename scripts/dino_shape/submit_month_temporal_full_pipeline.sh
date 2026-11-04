#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study}"
BANK_ROOT="${BANK_ROOT:-${STUDY_ROOT}/shared_probe_banks}"
DOMAIN_CSV="${DOMAIN_CSV:-/home/shaohan/radformation-data/OOD_Domain_Assignment_MAD_TOTAL.csv}"
MAX_TRAIN_CONCURRENT="${MAX_TRAIN_CONCURRENT:-4}"
MAX_EVAL_CONCURRENT="${MAX_EVAL_CONCURRENT:-2}"
MAX_ACTUAL_CONCURRENT="${MAX_ACTUAL_CONCURRENT:-2}"
MAX_RANK_CONCURRENT="${MAX_RANK_CONCURRENT:-4}"
SUBMIT_DOMAIN="${SUBMIT_DOMAIN:-1}"

cd "${PROJECT_ROOT}"
required_files=(
  jobs/all_roi_four_model_train_array.job
  jobs/all_roi_four_model_eval_array.job
  jobs/all_roi_actual_autoseg_regression_array.job
  jobs/all_roi_actual_autoseg_ranking_classification_array.job
  scripts/dino_shape/run_all_roi_training_task.sh
  scripts/dino_shape/run_all_roi_evaluation_task.sh
  scripts/dino_shape/run_actual_autoseg_regression_task.sh
  scripts/dino_shape/run_actual_autoseg_ranking_classification.sh
  scripts/dino_shape/generate_all_roi_summary.sh
  scripts/dino_shape/generate_actual_autoseg_regression_summary.sh
  scripts/dino_shape/generate_actual_autoseg_ranking_classification_summary.sh
  scripts/dino_shape/generate_actual_autoseg_ranking_classification_report.sh
)
for f in "${required_files[@]}"; do
  [[ -f "${f}" ]] || { echo "Missing required August-13 workflow file: ${f}" >&2; exit 2; }
done

TASKS="${STUDY_ROOT}/manifests/training_tasks.tsv"
ACTIVE="${STUDY_ROOT}/manifests/active_rois.tsv"
SPLIT_META="${STUDY_ROOT}/manifests/temporal_split_summary.json"
[[ -s "${TASKS}" ]] || { echo "Missing ${TASKS}; run prepare_month_temporal_nsrqa_study.sh first." >&2; exit 2; }
[[ -s "${ACTIVE}" ]] || { echo "Missing ${ACTIVE}" >&2; exit 2; }
[[ -s "${SPLIT_META}" ]] || { echo "Missing ${SPLIT_META}; refusing to submit a non-temporal study by mistake." >&2; exit 2; }

mkdir -p "${STUDY_ROOT}/logs" "${BANK_ROOT}"
n_train=$(( $(wc -l < "${TASKS}") - 1 ))
n_roi=$(( $(wc -l < "${ACTIVE}") - 1 ))
(( n_train > 0 && n_roi > 0 )) || { echo "Empty study manifests." >&2; exit 2; }
last_train=$((n_train - 1)); last_roi=$((n_roi - 1))

export_args="ALL,STUDY_ROOT=${STUDY_ROOT},BANK_ROOT=${BANK_ROOT},PROJECT_ROOT=${PROJECT_ROOT}"
jobid() { printf '%s' "$1" | cut -d';' -f1; }

echo "Temporal NSR-QA study root: ${STUDY_ROOT}"
echo "Training tasks: ${n_train}; active ROIs: ${n_roi}"
echo "All sbatch calls explicitly use partition=hugo and account=geoffrey_hugo."

echo "Submitting four-model training array..."
TRAIN_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --export="${export_args}" \
  --job-name=monthtemp_4model_train \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="0-${last_train}%${MAX_TRAIN_CONCURRENT}" \
  jobs/all_roi_four_model_train_array.job)"
TRAIN_JOB="$(jobid "${TRAIN_RAW}")"
echo "  training job: ${TRAIN_JOB}"

echo "Submitting synthetic shared-probe evaluation/report array after training..."
EVAL_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${TRAIN_JOB}" \
  --export="${export_args}" \
  --job-name=monthtemp_4model_eval \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="0-${last_roi}%${MAX_EVAL_CONCURRENT}" \
  jobs/all_roi_four_model_eval_array.job)"
EVAL_JOB="$(jobid "${EVAL_RAW}")"
echo "  evaluation job: ${EVAL_JOB}"

echo "Submitting clinical actual-autoseg NSR-QA metrics/regression after training..."
ACTUAL_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${TRAIN_JOB}" \
  --export="${export_args}" \
  --job-name=monthtemp_autoseg_reg \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="0-${last_roi}%${MAX_ACTUAL_CONCURRENT}" \
  jobs/all_roi_actual_autoseg_regression_array.job)"
ACTUAL_JOB="$(jobid "${ACTUAL_RAW}")"
echo "  actual-autoseg job: ${ACTUAL_JOB}"

echo "Submitting cross-ROI synthetic summary..."
SYN_SUM_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${EVAL_JOB}" \
  --export="${export_args}" \
  --job-name=monthtemp_probe_summary \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/month_temporal_nsrqa_postprocess.job synthetic_summary)"
SYN_SUM_JOB="$(jobid "${SYN_SUM_RAW}")"

echo "Submitting regression summary..."
REG_SUM_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${ACTUAL_JOB}" \
  --export="${export_args}" \
  --job-name=monthtemp_reg_summary \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/month_temporal_nsrqa_postprocess.job regression_summary)"
REG_SUM_JOB="$(jobid "${REG_SUM_RAW}")"

echo "Submitting clinical ranking/classification AUC array..."
RANK_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${ACTUAL_JOB}" \
  --export="${export_args}" \
  --job-name=monthtemp_rankcls \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="0-${last_roi}%${MAX_RANK_CONCURRENT}" \
  jobs/all_roi_actual_autoseg_ranking_classification_array.job)"
RANK_JOB="$(jobid "${RANK_RAW}")"
echo "  ranking/classification job: ${RANK_JOB}"

echo "Submitting ranking/classification summary + report..."
RANK_SUM_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${RANK_JOB}" \
  --export="${export_args}" \
  --job-name=monthtemp_rank_report \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/month_temporal_nsrqa_postprocess.job ranking_summary)"
RANK_SUM_JOB="$(jobid "${RANK_SUM_RAW}")"

DOMAIN_JOB=""
if [[ "${SUBMIT_DOMAIN}" == "1" ]]; then
  if [[ -s "${DOMAIN_CSV}" && -f analysis/analyze_nsrqa_domain_performance.py ]]; then
    echo "Submitting held-out month3 ROI-specific domain analysis..."
    DOMAIN_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
      --dependency="afterok:${RANK_JOB}" \
      --export="${export_args},DOMAIN_CSV=${DOMAIN_CSV}" \
      --job-name=monthtemp_domain \
      --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
      --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
      jobs/month_temporal_nsrqa_postprocess.job domain_report)"
    DOMAIN_JOB="$(jobid "${DOMAIN_RAW}")"
  else
    echo "Domain analysis not submitted: DOMAIN_CSV or analysis/analyze_nsrqa_domain_performance.py is missing."
  fi
fi

cat <<EOF

Submitted temporal NSR-QA pipeline
  training:              ${TRAIN_JOB}
  synthetic evaluation:  ${EVAL_JOB}
  synthetic summary:     ${SYN_SUM_JOB}
  actual-autoseg:        ${ACTUAL_JOB}
  regression summary:    ${REG_SUM_JOB}
  ranking/classification:${RANK_JOB}
  ranking report:        ${RANK_SUM_JOB}
  domain report:         ${DOMAIN_JOB:-not submitted}

Study outputs stay under:
  ${STUDY_ROOT}

Important semantic mapping retained for compatibility:
  training1   = month1+month2 development training subset
  validation1 = month1+month2 development validation subset
  testing1    = month3 held-out temporal test
EOF
