#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
QA_ROOT="${QA_ROOT:-${STUDY_ROOT}/qa_metric_regression}"
ROI="${1:?Usage: $0 ROI FOLD [CORRECTION_MODE]}"
FOLD="${2:?Usage: $0 ROI FOLD [CORRECTION_MODE]}"
CORRECTION_MODE="${3:-auto}"
LOCAL_RADIUS="${LOCAL_RADIUS:-6}"
DINO_BOUNDARY_RADIUS="${DINO_BOUNDARY_RADIUS:-3}"
N_BOOT="${N_BOOT:-2000}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF || true

BASE_CONFIG="${STUDY_ROOT}/configs/${ROI}/fold_${FOLD}.yaml"
V6_CONFIG="${STUDY_ROOT}/threeclass_correction_v6_weighted_ce/${ROI}/fold_${FOLD}/config.yaml"
OUT="${QA_ROOT}/${ROI}/fold_${FOLD}"
FEATURES="${OUT}/features.csv"
mkdir -p "${OUT}/analysis"

[[ -s "${BASE_CONFIG}" ]] || { echo "Missing base fold config: ${BASE_CONFIG}" >&2; exit 3; }

echo "[$(date)] ROI=${ROI} fold=${FOLD} correction_mode=${CORRECTION_MODE}"
python -m evaluation.qa_regression.extract_actual_autoseg_qa_features \
  --config "${BASE_CONFIG}" \
  --v6-config "${V6_CONFIG}" \
  --correction-mode "${CORRECTION_MODE}" \
  --local-radius "${LOCAL_RADIUS}" \
  --dino-boundary-radius "${DINO_BOUNDARY_RADIUS}" \
  --output-csv "${FEATURES}"

python -m analysis.fit_actual_autoseg_qa_regression \
  --features-csv "${FEATURES}" \
  --output-dir "${OUT}/analysis" \
  --n-boot "${N_BOOT}"

echo "Done: ${OUT}"
