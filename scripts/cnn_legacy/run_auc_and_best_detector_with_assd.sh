#!/usr/bin/env bash
set -euo pipefail

# Compute AUC sensitivity including ASSD, then select best detector per ROI.
# Run from /home/shaohan/VariantVAE.
#
# ASSD is included only if probe_eval_sweep_detailed.csv contains:
#   assd_input_vs_recon
#   assd_input_vs_target
#   assd_recon_vs_target

COMPUTE_SCRIPT="${COMPUTE_SCRIPT:-compute_probe_auc_sensitivity_assd.py}"
SELECT_SCRIPT="${SELECT_SCRIPT:-select_best_detector_per_roi_assd.py}"

EXP_ROOT="${EXP_ROOT:-/scratch/shaohan/experiments/VariantVAE}"
OUT_BASE="${OUT_BASE:-../radformation-data/probe_auc_sensitivity_assd}"
MODE="${MODE:-all}"

AUC_THRESHOLD="${AUC_THRESHOLD:-0.8}"
SELECTION_RULE="${SELECTION_RULE:-composite}"

if [[ ! -f "${COMPUTE_SCRIPT}" ]]; then
  echo "ERROR: missing ${COMPUTE_SCRIPT}"
  exit 1
fi

if [[ ! -f "${SELECT_SCRIPT}" ]]; then
  echo "ERROR: missing ${SELECT_SCRIPT}"
  exit 1
fi

mkdir -p "${OUT_BASE}"

case "${MODE}" in
  bronchus64)
    CSV="/scratch/shaohan/experiments/VariantVAE/label_ae_bronchus_roiwise_fov_fov_64x64x64_bronchus_alpha0.8_kld0.001_wmuniform/probe_eval_sweep/probe_eval_sweep_detailed.csv"
    OUT_DIR="${OUT_BASE}/bronchus64"
    python "${COMPUTE_SCRIPT}" --csvs "${CSV}" --out_dir "${OUT_DIR}"
    ;;

  lungL)
    mapfile -t CSVS < <(
      find "${EXP_ROOT}" -maxdepth 3         \( -path "*Lung_L*probe_eval_sweep/probe_eval_sweep_detailed.csv" -o            -path "*lungL*probe_eval_sweep/probe_eval_sweep_detailed.csv" -o            -path "*lung_L*probe_eval_sweep/probe_eval_sweep_detailed.csv" \)         | sort
    )
    [[ ${#CSVS[@]} -gt 0 ]] || { echo "No Lung_L CSVs found"; exit 1; }
    OUT_DIR="${OUT_BASE}/lungL"
    python "${COMPUTE_SCRIPT}" --csvs "${CSVS[@]}" --out_dir "${OUT_DIR}"
    ;;

  all)
    mapfile -t CSVS < <(
      find "${EXP_ROOT}" -maxdepth 3         -path "*probe_eval_sweep/probe_eval_sweep_detailed.csv"         | sort
    )
    [[ ${#CSVS[@]} -gt 0 ]] || { echo "No probe sweep CSVs found"; exit 1; }
    OUT_DIR="${OUT_BASE}/all"
    python "${COMPUTE_SCRIPT}" --csvs "${CSVS[@]}" --out_dir "${OUT_DIR}"
    ;;

  *)
    echo "Unknown MODE=${MODE}; use bronchus64, lungL, or all."
    exit 1
    ;;
esac

BEST_DIR="${OUT_DIR}/best_detector_per_roi"

python "${SELECT_SCRIPT}"   --auc_long "${OUT_DIR}/auc_clean_vs_corruption_long.csv"   --thresholds "${OUT_DIR}/auc_detection_thresholds.csv"   --score_defs "${OUT_DIR}/detector_score_definitions.csv"   --out_dir "${BEST_DIR}"   --auc_threshold "${AUC_THRESHOLD}"   --selection_rule "${SELECTION_RULE}"

echo
echo "Done."
echo "AUC output:"
echo "  ${OUT_DIR}"
echo "Best detector output:"
echo "  ${BEST_DIR}"
