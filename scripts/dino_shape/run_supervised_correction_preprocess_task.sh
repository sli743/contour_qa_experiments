#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ALLROI_ROOT="${ALLROI_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
idx="${1:?ROI array index required}"
manifest="${ALLROI_ROOT}/manifests/active_rois.tsv"
line=$(awk -F '\t' -v i="$idx" 'NR>1 && $1==i {print; exit}' "$manifest")
[[ -n "$line" ]] || { echo "No active ROI index $idx" >&2; exit 2; }
IFS=$'\t' read -r roi_index registry_index roi group rest <<< "$line"
python -m evaluation.supervised_correction.preprocess_autoseg_correction --roi "$roi" --group "$group" --output-root "${STUDY_ROOT}/preprocessed"
