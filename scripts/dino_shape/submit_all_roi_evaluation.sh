#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
MANIFEST="${STUDY_ROOT}/manifests/active_rois.tsv"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
[[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 2; }
n=$(( $(wc -l < "${MANIFEST}") - 1 )); (( n > 0 )) || { echo "No active ROIs" >&2; exit 2; }
last=$((n-1)); cd "${PROJECT_ROOT}"
extra=(); [[ -n "${DEPENDENCY_JOB_ID:-}" ]] && extra+=(--dependency="afterok:${DEPENDENCY_JOB_ID}")
# Partition/account are deliberately explicit here AND in the job file.
sbatch -p hugo --account geoffrey_hugo --array="0-${last}%${MAX_CONCURRENT}" "${extra[@]}" jobs/all_roi_four_model_eval_array.job
