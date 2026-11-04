#!/bin/bash
set -euo pipefail

REPO="${REPO:-/home/shaohan/contour_qa_experiments}"
cd "$REPO"

ERROR_ROOT="/scratch/shaohan/contour_QA/lung_l_error_field_characterization_v1"
if [[ ! -f "$ERROR_ROOT/case_error_field_summary.csv" || ! -f "$ERROR_ROOT/error_components.csv" ]]; then
  echo "Missing direct Lung_L error-field characterization under: $ERROR_ROOT" >&2
  echo "Run scripts/dino_shape/submit_lung_l_error_field_characterization.sh first." >&2
  exit 2
fi

sbatch -p hugo --account=geoffrey_hugo jobs/lung_l_nsrqa_severity_calibration.job
