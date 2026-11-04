#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
VARIANT="${3:-large_decoder_ft_last2}"
LOCAL_RADIUS="${LOCAL_RADIUS:-6}"
FORCE_ACTION_DIAG="${FORCE_ACTION_DIAG:-0}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
CONFIG="${OUT}/config.yaml"
CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
DIAG_OUT="${OUT}/action_score_diagnostic"
SENTINEL="${DIAG_OUT}/action_score_discrimination.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 3; }

if [[ "${FORCE_ACTION_DIAG}" != "1" && -f "${SENTINEL}" ]]; then
  echo "Reusing action-score discrimination: ${SENTINEL}"
  exit 0
fi

mkdir -p "${DIAG_OUT}"
echo "Computing lightweight validation action discrimination"
echo "ROI=${ROI} fold=${FOLD} variant=${VARIANT} local_radius=${LOCAL_RADIUS}"
python -m analysis.diagnose_supervised_correction_action_scores \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split validation \
  --local-radius "${LOCAL_RADIUS}" \
  --discrimination-only \
  --output-dir "${DIAG_OUT}"

echo "Saved: ${SENTINEL}"
