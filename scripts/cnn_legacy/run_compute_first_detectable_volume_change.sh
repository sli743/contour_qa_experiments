#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Compute first detectable volume-change percentage
# ============================================================
# Run from:
#   /home/shaohan/VariantVAE
#
# Required:
#   compute_first_detectable_volume_change.py
#
# Input should be one or more:
#   probe_eval_sweep_with_detector_scores.csv
#
# Examples:
#   MODE=all bash run_compute_first_detectable_volume_change.sh
#   MODE=bronchus64 bash run_compute_first_detectable_volume_change.sh
#   INPUT_CSVS="/path/a.csv /path/b.csv" bash run_compute_first_detectable_volume_change.sh

SCRIPT="${SCRIPT:-compute_first_detectable_volume_change.py}"
EXP_ROOT="${EXP_ROOT:-/scratch/shaohan/experiments/VariantVAE}"
OUT_BASE="${OUT_BASE:-../radformation-data/probe_auc_sensitivity_assd}"
MODE="${MODE:-all}"

AUC_THRESHOLD="${AUC_THRESHOLD:-0.8}"
BIN_WIDTH_PCT="${BIN_WIDTH_PCT:-2.5}"
SELECTION_RULE="${SELECTION_RULE:-composite}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT}"
  exit 1
fi

mkdir -p "${OUT_BASE}"

if [[ -n "${INPUT_CSVS:-}" ]]; then
  # shellcheck disable=SC2206
  CSVS=(${INPUT_CSVS})
  OUT_DIR="${OUT_BASE}/first_detectable_volume_custom"
else
  case "${MODE}" in
    bronchus64)
      CSVS=(
        "/scratch/shaohan/experiments/VariantVAE/label_ae_bronchus_roiwise_fov_fov_64x64x64_bronchus_alpha0.8_kld0.001_wmuniform/probe_eval_sweep/probe_eval_sweep_with_detector_scores.csv"
      )
      OUT_DIR="${OUT_BASE}/bronchus64/first_detectable_volume_change"
      ;;

    all)
      mapfile -t CSVS < <(
        find "${EXP_ROOT}" -maxdepth 4 \
          -path "*probe_eval_sweep/probe_eval_sweep_with_detector_scores.csv" \
          | sort
      )
      if (( ${#CSVS[@]} == 0 )); then
        echo "No probe_eval_sweep_with_detector_scores.csv files found under ${EXP_ROOT}"
        echo "If your combined CSV is in ../radformation-data/probe_auc_sensitivity_assd/all/, use:"
        echo "  INPUT_CSVS=../radformation-data/probe_auc_sensitivity_assd/all/probe_eval_sweep_with_detector_scores.csv bash $0"
        exit 1
      fi
      OUT_DIR="${OUT_BASE}/all/first_detectable_volume_change"
      ;;

    *)
      echo "Unknown MODE=${MODE}; use all, bronchus64, or INPUT_CSVS=..."
      exit 1
      ;;
  esac
fi

python "${SCRIPT}" \
  --csvs "${CSVS[@]}" \
  --out_dir "${OUT_DIR}" \
  --auc_threshold "${AUC_THRESHOLD}" \
  --bin_width_pct "${BIN_WIDTH_PCT}" \
  --selection_rule "${SELECTION_RULE}"

echo
echo "Done."
echo "Output:"
echo "  ${OUT_DIR}"
