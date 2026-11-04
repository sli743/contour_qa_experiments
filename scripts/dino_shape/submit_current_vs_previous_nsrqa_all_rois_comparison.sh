#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ALLROI_ROOT="${ALLROI_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
VARIANT="${VARIANT:-large_decoder_ft_last2}"
ROIS="${ROIS:-}"
ACTIVE="${ALLROI_ROOT}/manifests/active_rois.tsv"
cd "${PROJECT_ROOT}"
if [[ -n "${ROIS}" ]]; then IFS=',' read -r -a ARR <<< "${ROIS}"; else mapfile -t ARR < <(awk -F '\t' 'NR>1{print $3}' "${ACTIVE}"); fi
JOBS=(); SELECTED=()
for ROI_RAW in "${ARR[@]}"; do
  ROI="$(echo "${ROI_RAW}"|xargs)"; [[ -n "${ROI}" ]] || continue; SELECTED+=("${ROI}")
  for FOLD in 0 1 2 3 4; do
    C="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}/spatial_qa_head_analysis/spatial_qa_regression_summary.csv"
    N="${STUDY_ROOT}/matched_previous_nsrqa_5fold/${ROI}/fold_${FOLD}/analysis/nsrqa_regression_summary.csv"
    [[ -f "${C}" ]] || { echo "Missing current result: ${C}" >&2; exit 2; }
    [[ -f "${N}" ]] || { echo "Missing NSR-QA result: ${N}" >&2; exit 2; }
  done
  RAW=$(sbatch --parsable --export="ALL,ROI=${ROI},VARIANT=${VARIANT},STUDY_ROOT=${STUDY_ROOT}" -p hugo --account geoffrey_hugo jobs/current_vs_previous_nsrqa_compare.job)
  JOBS+=("$(echo "${RAW}"|cut -d';' -f1)")
done
D=$(IFS=:;echo "${JOBS[*]}"); R=$(IFS=:;echo "${SELECTED[*]}")
sbatch --parsable --dependency="afterok:${D}" \
  --export="ALL,VARIANT=${VARIANT},STUDY_ROOT=${STUDY_ROOT},ALLROI_ROOT=${ALLROI_ROOT},ROIS_EXPORT=${R}" \
  -p hugo --account geoffrey_hugo jobs/current_vs_previous_nsrqa_all_rois_aggregate.job
