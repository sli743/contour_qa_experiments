#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
MANIFEST="${STUDY_ROOT}/manifests/active_rois.tsv"
[[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 2; }
n="$(awk 'NR>1 && NF {n++} END {print n+0}' "${MANIFEST}")"
(( n > 0 )) || { echo "No active ROIs in ${MANIFEST}" >&2; exit 2; }
last=$((n-1))
echo "Submitting ${n} ranking/classification tasks, max concurrent=${MAX_CONCURRENT}"
sbatch -p hugo --account geoffrey_hugo --array="0-${last}%${MAX_CONCURRENT}" jobs/all_roi_actual_autoseg_ranking_classification_array.job
