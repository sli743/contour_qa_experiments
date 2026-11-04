#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ALLROI_ROOT="${ALLROI_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
BASELINE_ROOT="${BASELINE_ROOT:-${STUDY_ROOT}/matched_previous_nsrqa_5fold}"
ROI="${1:?ROI required}"
FOLD="${2:?fold required}"
MODE="${3:-all}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG=$(python scripts/dino_shape/prepare_matched_nsrqa_fold.py \
  --roi "${ROI}" --fold "${FOLD}" --study-root "${STUDY_ROOT}" \
  --all-roi-root "${ALLROI_ROOT}" --baseline-root "${BASELINE_ROOT}" | tail -n1)
FOLD_ROOT="${BASELINE_ROOT}/${ROI}/fold_${FOLD}"
MANIFEST="${STUDY_ROOT}/manifests/${ROI}/fold_${FOLD}.csv"
CHECKPOINT="${FOLD_ROOT}/models/matched_previous_nsrqa_${ROI}_fold${FOLD}/checkpoints/best_decoder.pth"

if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  echo "Training matched previous NSR-QA: ROI=${ROI} fold=${FOLD}"
  python -m training.train_dino_shape_stochastic_region_dropout_decoder --config "${CONFIG}" --fresh
fi

if [[ "${MODE}" == "evaluate" || "${MODE}" == "all" ]]; then
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
  mkdir -p "${FOLD_ROOT}/evaluation" "${FOLD_ROOT}/analysis"
  python -m evaluation.actual_autoseg.eval_matched_previous_nsrqa_fold \
    --config "${CONFIG}" --manifest "${MANIFEST}" \
    --output-csv "${FOLD_ROOT}/evaluation/patient_metrics.csv"
  python -m analysis.analyze_matched_previous_nsrqa_fold \
    --metrics-csv "${FOLD_ROOT}/evaluation/patient_metrics.csv" \
    --output-dir "${FOLD_ROOT}/analysis" \
    --target-specificity "${TARGET_SPECIFICITY:-0.95}"
fi
