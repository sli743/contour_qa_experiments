#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_v1}"
CALIBRATED_CONFIG="${CALIBRATED_CONFIG:-/scratch/shaohan/contour_QA/lung_l_nsrqa_severity_calibration_v1/recommended_training_config.yaml}"
cd "${PROJECT_ROOT}"

python -m analysis.prepare_lung_l_nsrqa_corruption_ablation \
  --calibrated-config "${CALIBRATED_CONFIG}" \
  --study-root "${STUDY_ROOT}" \
  --seeds 91 92 93

# All arms use identical crop geometry, so materialize real test autosegs once.
REFERENCE_CONFIG="${STUDY_ROOT}/configs/nsrqa_corruption_seed91.yaml"
python -m evaluation.actual_autoseg.preprocess_actual_autoseg \
  --config "${REFERENCE_CONFIG}" \
  --roi Lung_L \
  --group lungs_chest \
  --raw-data-root /home/shaohan/radformation-data/data \
  --output-root "${STUDY_ROOT}/actual_autoseg_preprocessed" \
  --mode autoseg_centered \
  --splits testing1
