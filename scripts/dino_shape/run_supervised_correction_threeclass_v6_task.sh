#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Bronchus}"
FOLD="${2:-0}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CONFIG=$(python scripts/dino_shape/prepare_supervised_correction_threeclass_v6.py \
  --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}" \
  --action-weight "${ACTION_WEIGHT}" | tail -n 1)
OUT="${STUDY_ROOT}/threeclass_correction_v6_weighted_ce/${ROI}/fold_${FOLD}"
echo "Three-class correction v6 config: ${CONFIG}"
echo "Action class weight: ${ACTION_WEIGHT} (preserve=1.0)"
python -m training.train_supervised_autoseg_correction_threeclass --config "${CONFIG}" --fresh
python -m evaluation.supervised_correction.eval_supervised_correction_threeclass \
  --config "${CONFIG}" --output-dir "${OUT}/evaluation"
python -m analysis.summarize_supervised_correction_threeclass_v6 \
  --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}"
echo "Done. Results: ${OUT}"
