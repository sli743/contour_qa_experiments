#!/usr/bin/env bash
set -euo pipefail

ROI="${1:-Bronchus}"
FOLD="${2:-0}"
MODEL_VERSION="${3:-v6}"
SPLIT="${4:-validation}"
LOCAL_RADIUS="${5:-6}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
cd "${PROJECT_ROOT}"

JOB_ID=$(sbatch \
  -p hugo \
  --account geoffrey_hugo \
  --export="ALL,ROI=${ROI},FOLD=${FOLD},MODEL_VERSION=${MODEL_VERSION},SPLIT=${SPLIT},LOCAL_RADIUS=${LOCAL_RADIUS}" \
  --parsable \
  jobs/supervised_correction_patient_gating_diagnostic.job)

echo "Submitted patient-gating diagnostic: ROI=${ROI} fold=${FOLD} model=${MODEL_VERSION} split=${SPLIT} job=${JOB_ID}"
