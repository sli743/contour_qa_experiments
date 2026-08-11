#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
OUT_ROOT="${OUT_ROOT:-/scratch/shaohan/contour_QA/roc_comparison}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
ROI="${1:-Lung_L}"
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
run_one(){
  local roi="$1" thr="$2" pilot="$3" benchmark="$4" random="$5" stochastic="$6"
  local out="${OUT_ROOT}/${roi}_four_models_stochastic_dropout_no_shift"
  python -m analysis.generate_four_model_probe_report \
    --model "Pilot AE=${pilot}" \
    --model "Benchmark-leakage DAE=${benchmark}" \
    --model "Random-morphology DAE=${random}" \
    --model "Stochastic-region-dropout DAE=${stochastic}" \
    --roi "${roi}" --volume-threshold-pct "${thr}" --exclude-shifts --output-dir "${out}"
}
case "${ROI}" in
  Lung_L)
    run_one Lung_L 20 \
      "${LUNGL_PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_sweep/probe_eval_sweep_detailed.csv}" \
      "${LUNGL_BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${LUNGL_RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${LUNGL_STOCHASTIC_CSV:-/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" ;;
  Bronchus)
    run_one Bronchus 10 \
      "${BRONCHUS_PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Bronchus/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${BRONCHUS_BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${BRONCHUS_RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${BRONCHUS_STOCHASTIC_CSV:-/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" ;;
  all) "$0" Lung_L; "$0" Bronchus ;;
  *) echo "Usage: $0 {Lung_L|Bronchus|all}" >&2; exit 2 ;;
esac
