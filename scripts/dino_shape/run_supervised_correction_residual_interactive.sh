#!/usr/bin/env bash
# Run residual + unchanged-region preservation diagnostic in the current interactive GPU allocation.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Bronchus}"
FOLD="${2:-0}"
cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
CONFIG=$(python scripts/dino_shape/prepare_supervised_correction_residual.py --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}" | tail -n 1)
OUT="${STUDY_ROOT}/residual_diagnostics/${ROI}/fold_${FOLD}"
echo "Residual correction config: ${CONFIG}"
python -m training.train_supervised_autoseg_correction_decoder --config "${CONFIG}" --fresh
python -m evaluation.supervised_correction.eval_supervised_correction_diagnostic --config "${CONFIG}" --output-dir "${OUT}/evaluation"
python -m analysis.summarize_supervised_correction_residual --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}"
echo "Done. Results: ${OUT}"
