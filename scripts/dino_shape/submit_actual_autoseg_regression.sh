#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
MANIFEST="${STUDY_ROOT}/manifests/active_rois.tsv"
[[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 2; }
n="$(awk 'END{print NR-1}' "${MANIFEST}")"
(( n > 0 )) || { echo "No active ROIs in ${MANIFEST}" >&2; exit 2; }
last=$((n - 1))
echo "Submitting ${n} actual-autoseg NSR-QA regression tasks, max concurrent=${MAX_CONCURRENT}"
sbatch -p hugo --account geoffrey_hugo --array="0-${last}%${MAX_CONCURRENT}" jobs/all_roi_actual_autoseg_regression_array.job
