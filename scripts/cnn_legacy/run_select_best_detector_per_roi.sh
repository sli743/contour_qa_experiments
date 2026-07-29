#!/usr/bin/env bash
set -euo pipefail

# Select best detector per ROI from already computed AUC outputs.
# Run from /home/shaohan/VariantVAE.

SCRIPT="${SCRIPT:-select_best_detector_per_roi.py}"

AUC_ROOT="${AUC_ROOT:-../radformation-data/probe_auc_sensitivity/all}"
OUT_DIR="${OUT_DIR:-${AUC_ROOT}/best_detector_per_roi}"

AUC_LONG="${AUC_LONG:-${AUC_ROOT}/auc_clean_vs_corruption_long.csv}"
THRESHOLDS="${THRESHOLDS:-${AUC_ROOT}/auc_detection_thresholds.csv}"
SCORE_DEFS="${SCORE_DEFS:-${AUC_ROOT}/detector_score_definitions.csv}"

AUC_THRESHOLD="${AUC_THRESHOLD:-0.8}"
SELECTION_RULE="${SELECTION_RULE:-composite}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT}"
  exit 1
fi

for f in "${AUC_LONG}" "${THRESHOLDS}" "${SCORE_DEFS}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing input: ${f}"
    exit 1
  fi
done

python "${SCRIPT}" \
  --auc_long "${AUC_LONG}" \
  --thresholds "${THRESHOLDS}" \
  --score_defs "${SCORE_DEFS}" \
  --out_dir "${OUT_DIR}" \
  --auc_threshold "${AUC_THRESHOLD}" \
  --selection_rule "${SELECTION_RULE}"

echo
echo "Done."
echo "Output:"
echo "  ${OUT_DIR}"
