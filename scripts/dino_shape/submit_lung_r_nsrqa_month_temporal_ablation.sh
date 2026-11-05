#!/usr/bin/env bash
set -euo pipefail

# Lung_R version of the two-arm month-temporal NSR-QA ablation
# (task 0 = nsrqa_corruption, task 1 = no_corruption, single seed): same
# prepare -> train -> Month3 eval -> postprocess chain as the Lung_L run.
#
# Corruption severities default to the Lung_L-calibrated config used by
# lung_l_nsrqa_corruption_ablation_month_temporal_v1, because no Lung_R
# calibration exists yet. For the original (uncalibrated) Lung_R profile use:
#   CALIBRATED_CONFIG=/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study/configs/Lung_R/nsrqa.yaml \
#   STUDY_ROOT=/scratch/shaohan/contour_QA/lung_r_nsrqa_corruption_ablation_month_temporal_uncalibrated_v1 \
#   bash scripts/dino_shape/submit_lung_r_nsrqa_month_temporal_ablation.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
export PROJECT_ROOT
export ROI=Lung_R
export JOB_TAG="${JOB_TAG:-lungR}"
export STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_r_nsrqa_corruption_ablation_month_temporal_v1}"

if [[ -s "${STUDY_ROOT}/experiment_manifest.csv" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "ERROR: ${STUDY_ROOT} already holds a prepared study (experiment_manifest.csv)." >&2
  echo "Training uses --fresh and would overwrite its models. Pick a new STUDY_ROOT," >&2
  echo "or rerun with ALLOW_EXISTING=1 to replace it." >&2
  exit 2
fi

exec bash "${PROJECT_ROOT}/scripts/dino_shape/submit_lung_l_nsrqa_month_temporal_ablation.sh"
