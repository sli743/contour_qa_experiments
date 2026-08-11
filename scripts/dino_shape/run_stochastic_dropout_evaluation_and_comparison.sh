#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ROI="${1:-Lung_L}"
case "${ROI}" in
  Lung_L|Bronchus)
    bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_stochastic_region_dropout_evaluation.sh" "${ROI}"
    bash "${PROJECT_ROOT}/scripts/dino_shape/run_generate_four_model_stochastic_dropout_report.sh" "${ROI}" ;;
  all)
    bash "${PROJECT_ROOT}/scripts/dino_shape/run_stochastic_dropout_evaluation_and_comparison.sh" Lung_L
    bash "${PROJECT_ROOT}/scripts/dino_shape/run_stochastic_dropout_evaluation_and_comparison.sh" Bronchus ;;
  *) echo "Usage: $0 {Lung_L|Bronchus|all}" >&2; exit 2 ;;
esac
