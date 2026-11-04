#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Bronchus}"
FOLD="${2:-0}"
MODEL_VERSION="${3:-v5}"
SPLIT="${SPLIT:-validation}"
LOCAL_RADIUS="${LOCAL_RADIUS:-6}"

case "${MODEL_VERSION}" in
  v5)
    MODEL_DIR="threeclass_correction_v5"
    ;;
  v6)
    MODEL_DIR="threeclass_correction_v6_weighted_ce"
    ;;
  *)
    echo "MODEL_VERSION must be v5 or v6, got: ${MODEL_VERSION}" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG="${STUDY_ROOT}/${MODEL_DIR}/${ROI}/fold_${FOLD}/config.yaml"
OUT="${STUDY_ROOT}/action_score_diagnostics/${MODEL_DIR}/${ROI}/fold_${FOLD}/${SPLIT}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

python -m analysis.diagnose_supervised_correction_action_scores \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --local-radius "${LOCAL_RADIUS}" \
  --output-dir "${OUT}"

echo "Done. Diagnostic outputs: ${OUT}"
