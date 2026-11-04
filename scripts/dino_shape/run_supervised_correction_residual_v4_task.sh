#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Bronchus}"
FOLD="${2:-0}"
cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CONFIG=$(python scripts/dino_shape/prepare_supervised_correction_residual_v4.py \
  --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}" | tail -n 1)
OUT="${STUDY_ROOT}/residual_directional_v4/${ROI}/fold_${FOLD}"
echo "Directional residual v4 config: ${CONFIG}"
python -m training.train_supervised_autoseg_correction_decoder --config "${CONFIG}" --fresh
python -m evaluation.supervised_correction.eval_supervised_correction_diagnostic \
  --config "${CONFIG}" --output-dir "${OUT}/evaluation"
python -m analysis.summarize_supervised_correction_residual_v4 \
  --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}"
echo "Done. Results: ${OUT}"
