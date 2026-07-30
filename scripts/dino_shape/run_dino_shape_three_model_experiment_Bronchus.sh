#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
RUN_PILOT_TRAIN="${RUN_PILOT_TRAIN:-1}"
RUN_BENCHMARK_TRAIN="${RUN_BENCHMARK_TRAIN:-1}"
RUN_RANDOM="${RUN_RANDOM:-0}"
RUN_EXPORTS="${RUN_EXPORTS:-1}"

cd "${PROJECT_ROOT}"

bash "${PROJECT_ROOT}/scripts/dino_shape/materialize_dino_shape_Bronchus_pilot_and_denoising.sh"

if [[ "${RUN_PILOT_TRAIN}" == "1" ]]; then
  echo "[1/7] Training Bronchus pilot AE without denoising"
  bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_pilot_Bronchus.sh"
else
  echo "[1/7] Skipping pilot training"
fi

echo "[2/7] Evaluating Bronchus pilot on fixed benchmark probes"
bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_probe_sweep_Bronchus.sh"

if [[ "${RUN_BENCHMARK_TRAIN}" == "1" ]]; then
  echo "[3/7] Training Bronchus benchmark-leakage denoising AE"
  bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_denoising_Bronchus_v1.sh"
else
  echo "[3/7] Skipping benchmark-leakage denoising training"
fi

echo "[4/7] Evaluating benchmark-leakage denoiser on fixed benchmark probes"
bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_denoising_probe_sweep_Bronchus_v1.sh"

if [[ "${RUN_EXPORTS}" == "1" ]]; then
  echo "[5/7] Exporting benchmark-leakage denoiser case visualizations"
  bash "${PROJECT_ROOT}/scripts/dino_shape/run_export_dino_shape_denoising_probe_contours_Bronchus_v1.sh" \
    --layout flat --save_metrics
else
  echo "[5/7] Skipping benchmark-leakage case exports"
fi

if [[ "${RUN_RANDOM}" == "1" ]]; then
  echo "[6/7] Running random-morphology pipeline"
  bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_random_morphology_full_pipeline_Bronchus_v1.sh"
else
  echo "[6/7] Reusing existing random-morphology results"
fi

echo "[7/7] Building three-model Bronchus ROC comparison"
bash "${PROJECT_ROOT}/scripts/dino_shape/run_compare_dino_shape_three_models_roc_Bronchus.sh"
