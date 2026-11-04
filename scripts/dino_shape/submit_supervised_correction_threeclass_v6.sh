#!/usr/bin/env bash
set -euo pipefail
ROI="${1:-Bronchus}"
FOLD="${2:-0}"
ACTION_WEIGHT="${3:-4.0}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
cd "${PROJECT_ROOT}"
JOB_ID=$(sbatch \
  -p hugo \
  --account geoffrey_hugo \
  --export="ALL,ROI=${ROI},FOLD=${FOLD},ACTION_WEIGHT=${ACTION_WEIGHT}" \
  --parsable \
  jobs/supervised_correction_threeclass_v6.job)
echo "Submitted three-class correction v6: ROI=${ROI} fold=${FOLD} action_weight=${ACTION_WEIGHT} job=${JOB_ID}"
