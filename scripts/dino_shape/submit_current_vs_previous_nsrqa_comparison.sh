#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:?ROI required}"
VARIANT="${VARIANT:-large_decoder_ft_last2}"
QA_N_BOOT="${QA_N_BOOT:-5000}"
cd "${PROJECT_ROOT}"
for FOLD in 0 1 2 3 4; do
  C="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}/spatial_qa_head_analysis/spatial_qa_regression_summary.csv"
  N="${STUDY_ROOT}/matched_previous_nsrqa_5fold/${ROI}/fold_${FOLD}/analysis/nsrqa_regression_summary.csv"
  [[ -f "${C}" ]] || { echo "Missing current result: ${C}" >&2; exit 2; }
  [[ -f "${N}" ]] || { echo "Missing matched NSR-QA result: ${N}" >&2; exit 2; }
done
sbatch --parsable \
  --export="ALL,ROI=${ROI},VARIANT=${VARIANT},STUDY_ROOT=${STUDY_ROOT},QA_N_BOOT=${QA_N_BOOT}" \
  -p hugo --account geoffrey_hugo jobs/current_vs_previous_nsrqa_compare.job
