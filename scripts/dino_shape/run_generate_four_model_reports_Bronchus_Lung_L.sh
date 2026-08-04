#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
OUT_ROOT="${OUT_ROOT:-/scratch/shaohan/contour_QA/roc_comparison}"
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

run_roi() {
  local roi="$1" threshold="$2" pilot="$3" benchmark="$4" random_morph="$5" deformation="$6"
  local out="${OUT_ROOT}/${roi}_four_models_no_shift"
  mkdir -p "${out}"
  local models=(
    --model "Pilot AE=${pilot}"
    --model "Benchmark-leakage DAE=${benchmark}"
    --model "Random-morphology DAE=${random_morph}"
    --model "Deformation-noise DAE=${deformation}"
  )
  python -m analysis.plot_multi_model_roc_by_metric \
    "${models[@]}" --roi "${roi}" --volume-threshold-pct "${threshold}" \
    --exclude-shifts --output-dir "${out}"
  python -m analysis.summarize_multi_model_probe_results \
    "${models[@]}" --roi "${roi}" --volume-threshold-pct "${threshold}" \
    --exclude-shifts --output-dir "${out}"
}

run_roi Bronchus 10 \
  "${BRONCHUS_PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Bronchus/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
  "${BRONCHUS_BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
  "${BRONCHUS_RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
  "${BRONCHUS_DEFORMATION_CSV:-/scratch/shaohan/contour_QA/dino_deformation_noise/dino_shape_deformation_noise_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"

run_roi Lung_L 20 \
  "${LUNGL_PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_sweep/probe_eval_sweep_detailed.csv}" \
  "${LUNGL_BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
  "${LUNGL_RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
  "${LUNGL_DEFORMATION_CSV:-/scratch/shaohan/contour_QA/dino_deformation_noise/dino_shape_deformation_noise_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
