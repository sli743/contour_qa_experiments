#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${SCRIPT:-compute_probe_auc_sensitivity.py}"
EXP_ROOT="${EXP_ROOT:-/scratch/shaohan/experiments/VariantVAE}"
OUT_BASE="${OUT_BASE:-../radformation-data/probe_auc_sensitivity}"
MODE="${MODE:-all}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT}"
  exit 1
fi

mkdir -p "${OUT_BASE}"

case "${MODE}" in
  bronchus64)
    CSV="/scratch/shaohan/experiments/VariantVAE/label_ae_bronchus_roiwise_fov_fov_64x64x64_bronchus_alpha0.8_kld0.001_wmuniform/probe_eval_sweep/probe_eval_sweep_detailed.csv"
    OUT_DIR="${OUT_BASE}/bronchus64"
    python "${SCRIPT}" --csvs "${CSV}" --out_dir "${OUT_DIR}"
    ;;
  lungL)
    mapfile -t CSVS < <(find "${EXP_ROOT}" -maxdepth 3 \( -path "*Lung_L*probe_eval_sweep/probe_eval_sweep_detailed.csv" -o -path "*lungL*probe_eval_sweep/probe_eval_sweep_detailed.csv" -o -path "*lung_L*probe_eval_sweep/probe_eval_sweep_detailed.csv" \) | sort)
    [[ ${#CSVS[@]} -gt 0 ]] || { echo "No Lung_L CSVs found"; exit 1; }
    OUT_DIR="${OUT_BASE}/lungL"
    python "${SCRIPT}" --csvs "${CSVS[@]}" --out_dir "${OUT_DIR}"
    ;;
  all)
    mapfile -t CSVS < <(find "${EXP_ROOT}" -maxdepth 3 -path "*probe_eval_sweep/probe_eval_sweep_detailed.csv" | sort)
    [[ ${#CSVS[@]} -gt 0 ]] || { echo "No probe sweep CSVs found"; exit 1; }
    OUT_DIR="${OUT_BASE}/all"
    python "${SCRIPT}" --csvs "${CSVS[@]}" --out_dir "${OUT_DIR}"
    ;;
  *)
    echo "Unknown MODE=${MODE}; use bronchus64, lungL, or all."
    exit 1
    ;;
esac

echo "Done: ${OUT_DIR}"
