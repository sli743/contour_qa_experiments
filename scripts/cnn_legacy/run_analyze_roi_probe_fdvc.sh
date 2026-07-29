#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ROI × probe FDVC analysis + heatmaps
# ============================================================
# Run from:
#   /home/shaohan/VariantVAE
#
# Required:
#   analyze_roi_probe_fdvc.py
#
# Default assumes outputs from compute_first_detectable_volume_change.py:
#   ../radformation-data/probe_auc_sensitivity_assd/all/first_detectable_volume_change/

SCRIPT="${SCRIPT:-analyze_roi_probe_fdvc.py}"

FDVC_ROOT="${FDVC_ROOT:-../radformation-data/probe_auc_sensitivity_assd/first_detectable_volume_custom}"

FIRST_CSV="${FIRST_CSV:-${FDVC_ROOT}/first_detectable_volume_change_by_probe.csv}"
VOLUME_AUC_CSV="${VOLUME_AUC_CSV:-${FDVC_ROOT}/volume_auc_clean_vs_corruption_binned_long.csv}"
OUT_DIR="${OUT_DIR:-${FDVC_ROOT}/roi_probe_fdvc_analysis}"

AUC_THRESHOLD="${AUC_THRESHOLD:-0.8}"
PENALTY_FDVC_PCT="${PENALTY_FDVC_PCT:-105}"
SELECTION_RULE="${SELECTION_RULE:-coverage_first}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT}"
  exit 1
fi

if [[ ! -f "${FIRST_CSV}" ]]; then
  echo "ERROR: missing FIRST_CSV: ${FIRST_CSV}"
  exit 1
fi

CMD=(
  python "${SCRIPT}"
  --first_csv "${FIRST_CSV}"
  --out_dir "${OUT_DIR}"
  --auc_threshold "${AUC_THRESHOLD}"
  --penalty_fdvc_pct "${PENALTY_FDVC_PCT}"
  --selection_rule "${SELECTION_RULE}"
)

if [[ -f "${VOLUME_AUC_CSV}" ]]; then
  CMD+=(--volume_auc_csv "${VOLUME_AUC_CSV}")
fi

"${CMD[@]}"

echo
echo "Done."
echo "Output:"
echo "  ${OUT_DIR}"
