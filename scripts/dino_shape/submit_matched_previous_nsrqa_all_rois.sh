#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ALLROI_ROOT="${ALLROI_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROIS="${ROIS:-}"
ACTIVE="${ALLROI_ROOT}/manifests/active_rois.tsv"
cd "${PROJECT_ROOT}"
[[ -f "${ACTIVE}" ]] || { echo "Missing ${ACTIVE}" >&2; exit 2; }
if [[ -n "${ROIS}" ]]; then IFS=',' read -r -a ARR <<< "${ROIS}"; else mapfile -t ARR < <(awk -F '\t' 'NR>1{print $3}' "${ACTIVE}"); fi
for ROI_RAW in "${ARR[@]}"; do
  ROI="$(echo "${ROI_RAW}"|xargs)"; [[ -n "${ROI}" ]] || continue
  echo "=== matched previous NSR-QA: ${ROI} ==="
  bash scripts/dino_shape/submit_matched_previous_nsrqa_5fold.sh "${ROI}"
done
