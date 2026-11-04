#!/bin/bash
set -euo pipefail

REPO="${REPO:-/home/shaohan/contour_qa_experiments}"
cd "$REPO"

ERROR_ROOT="/scratch/shaohan/contour_QA/lung_l_error_field_characterization_v1"
if [[ ! -f "$ERROR_ROOT/case_error_field_summary.csv" ]]; then
  echo "Missing Lung_L direct error-field summary: $ERROR_ROOT/case_error_field_summary.csv" >&2
  echo "Run scripts/dino_shape/submit_lung_l_error_field_characterization.sh first." >&2
  exit 2
fi
if [[ ! -f "$ERROR_ROOT/error_components.csv" ]]; then
  echo "Missing Lung_L error-component table: $ERROR_ROOT/error_components.csv" >&2
  echo "The component-calibrated dropout fit requires error_components.csv." >&2
  exit 2
fi

sbatch -p hugo --account=geoffrey_hugo jobs/lung_l_nsrqa_synthetic_severity_fit_family_validated.job
