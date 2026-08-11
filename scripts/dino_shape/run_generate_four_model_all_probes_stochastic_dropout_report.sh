#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
OUT_ROOT="${OUT_ROOT:-/scratch/shaohan/contour_QA/roc_comparison}"
N_BOOT="${N_BOOT:-1000}"
SEED="${SEED:-88}"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

run_roi() {
  local roi="$1" global_threshold="$2" local_depth_threshold="$3" pilot="$4" benchmark="$5" random_morph="$6" stochastic="$7"
  local out="${OUT_ROOT}/${roi}_four_models_stochastic_dropout_all_probes_two_criteria"

  python -m analysis.generate_four_model_all_probes_two_criteria \
    --model "Pilot AE=${pilot}" \
    --model "Benchmark-leakage DAE=${benchmark}" \
    --model "Random-morphology DAE=${random_morph}" \
    --model "Stochastic-region-dropout DAE=${stochastic}" \
    --roi "${roi}" \
    --volume-threshold-pct "${global_threshold}" \
    --local-depth-threshold-mm "${local_depth_threshold}" \
    --exclude-shifts \
    --n-boot "${N_BOOT}" \
    --seed "${SEED}" \
    --output-dir "${out}"

  if command -v pandoc >/dev/null 2>&1; then
    (
      cd "${out}"
      pandoc all_probes_four_model_comparison.md --standalone --embed-resources -o all_probes_four_model_comparison.html
    )
  fi
  echo "All-probes four-model report: ${out}/all_probes_four_model_comparison.md"
}

case "${1:-all}" in
  Lung_L|lung|lung_l)
    run_roi Lung_L "${LUNGL_THRESHOLD_PCT:-20}" "${LUNGL_LOCAL_DEPTH_MM:-10}" \
      "${LUNGL_PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_sweep/probe_eval_sweep_detailed.csv}" \
      "${LUNGL_BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${LUNGL_RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${LUNGL_STOCHASTIC_CSV:-/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
    ;;
  Bronchus|bronchus)
    run_roi Bronchus "${BRONCHUS_THRESHOLD_PCT:-10}" "${BRONCHUS_LOCAL_DEPTH_MM:-4}" \
      "${BRONCHUS_PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Bronchus/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${BRONCHUS_BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${BRONCHUS_RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}" \
      "${BRONCHUS_STOCHASTIC_CSV:-/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
    ;;
  all)
    "$0" Lung_L
    "$0" Bronchus
    ;;
  *)
    echo "Usage: $0 [Lung_L|Bronchus|all]" >&2
    exit 2
    ;;
esac
