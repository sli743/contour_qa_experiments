#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
STAGE="${1:-}"
[[ -n "${STAGE}" ]] || { echo "Usage: $0 {synthetic_summary|regression_summary|ranking_summary|domain_report}" >&2; exit 2; }
cd "${PROJECT_ROOT}"

case "${STAGE}" in
  synthetic_summary)
    STUDY_ROOT="${STUDY_ROOT}" bash scripts/dino_shape/generate_all_roi_summary.sh
    ;;
  regression_summary)
    STUDY_ROOT="${STUDY_ROOT}" REG_ROOT="${REG_ROOT}" bash scripts/dino_shape/generate_actual_autoseg_regression_summary.sh
    ;;
  ranking_summary)
    STUDY_ROOT="${STUDY_ROOT}" REG_ROOT="${REG_ROOT}" bash scripts/dino_shape/generate_actual_autoseg_ranking_classification_summary.sh
    STUDY_ROOT="${STUDY_ROOT}" ACTUAL_ROOT="${REG_ROOT}" bash scripts/dino_shape/generate_actual_autoseg_ranking_classification_report.sh
    ;;
  domain_report)
    DOMAIN_CSV="${DOMAIN_CSV:-/home/shaohan/radformation-data/OOD_Domain_Assignment_MAD_TOTAL.csv}"
    [[ -s "${DOMAIN_CSV}" ]] || { echo "Missing DOMAIN_CSV=${DOMAIN_CSV}" >&2; exit 3; }
    [[ -f analysis/analyze_nsrqa_domain_performance.py ]] || {
      echo "analysis/analyze_nsrqa_domain_performance.py is not present. Apply the domain-analysis patches first." >&2
      exit 3
    }
    source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    python analysis/analyze_nsrqa_domain_performance.py \
      --ranking_root "${REG_ROOT}/ranking_classification" \
      --domain_csv "${DOMAIN_CSV}" \
      --output_dir "${STUDY_ROOT}/domain_analysis" \
      --localization_mode autoseg_centered \
      --true_error_percentile 75 \
      --roi_selection auto
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
