#!/usr/bin/env bash
set -euo pipefail

# Lung_L standard ROC analysis for volume-change > 20%.
# Run from /home/shaohan/VariantVAE.

SCRIPT="${SCRIPT:-roc_analysis_lungL_volume_gt20_ops_fixed.py}"
FDVC_ROOT="${FDVC_ROOT:-../radformation-data/probe_auc_sensitivity_assd/first_detectable_volume_custom/}"
CSV="${CSV:-${FDVC_ROOT}/input_with_volume_and_scores.csv}"
OUT_DIR="${OUT_DIR:-${FDVC_ROOT}/roc_analysis_volume_gt20/Lung_L}"

ROI="${ROI:-Lung_L}"
VOLUME_THRESHOLD_PCT="${VOLUME_THRESHOLD_PCT:-20}"
METRICS="${METRICS:-}"
INCLUDE_SHIFT="${INCLUDE_SHIFT:-0}"
N_BOOT="${N_BOOT:-1000}"
SEED="${SEED:-123}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing script: ${SCRIPT}"
  exit 1
fi

if [[ ! -f "${CSV}" ]]; then
  echo "ERROR: missing CSV: ${CSV}"
  echo "Set CSV=/path/to/input_with_volume_and_scores.csv or probe_eval_sweep_with_detector_scores.csv"
  exit 1
fi

CMD=(
  python "${SCRIPT}"
  --csvs "${CSV}"
  --out_dir "${OUT_DIR}"
  --roi "${ROI}"
  --volume_threshold_pct "${VOLUME_THRESHOLD_PCT}"
  --n_boot "${N_BOOT}"
  --seed "${SEED}"
)

if [[ -n "${METRICS}" ]]; then
  # shellcheck disable=SC2206
  METRIC_ARGS=(${METRICS})
  CMD+=(--metrics "${METRIC_ARGS[@]}")
fi

if [[ "${INCLUDE_SHIFT}" == "1" ]]; then
  CMD+=(--include_shift)
fi

"${CMD[@]}"

echo
echo "Done."
echo "Output:"
echo "  ${OUT_DIR}"
