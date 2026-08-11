#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ROI="${1:-Lung_L}"
ROOT="${ROOT:-/scratch/shaohan/contour_QA}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/roc_comparison}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${ROI}" in
  Lung_L)
    PILOT_CSV="${LUNGL_PILOT_CSV:-${ROOT}/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv}"
    BENCHMARK_CSV="${LUNGL_BENCHMARK_CSV:-${ROOT}/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv}"
    RANDOM_CSV="${LUNGL_RANDOM_CSV:-${ROOT}/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv}"
    STOCHASTIC_CSV="${LUNGL_STOCHASTIC_CSV:-${ROOT}/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Lung_L_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv}"
    BAD_THRESHOLD="${BAD_THRESHOLD:-20}"
    ;;
  Bronchus)
    PILOT_CSV="${BRONCHUS_PILOT_CSV:-${ROOT}/dino_frozen/dino_shape_pilot_Bronchus/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv}"
    BENCHMARK_CSV="${BRONCHUS_BENCHMARK_CSV:-${ROOT}/dino_denoising/dino_shape_denoising_Bronchus_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv}"
    RANDOM_CSV="${BRONCHUS_RANDOM_CSV:-${ROOT}/dino_random_morphology/dino_shape_random_morphology_Bronchus_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv}"
    STOCHASTIC_CSV="${BRONCHUS_STOCHASTIC_CSV:-${ROOT}/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Bronchus_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv}"
    BAD_THRESHOLD="${BAD_THRESHOLD:-10}"
    ;;
  *)
    echo "Usage: $0 {Lung_L|Bronchus}" >&2
    exit 2
    ;;
esac

OUT="${OUT_ROOT}/${ROI}_four_model_dilation_behavior_diagnostic"

python -m analysis.diagnose_four_model_dilation_behavior \
  --model "Pilot AE=${PILOT_CSV}" \
  --model "Benchmark-leakage DAE=${BENCHMARK_CSV}" \
  --model "Random-morphology DAE=${RANDOM_CSV}" \
  --model "Stochastic-region-dropout DAE=${STOCHASTIC_CSV}" \
  --probe dilated \
  --bad-volume-threshold-pct "${BAD_THRESHOLD}" \
  --output-dir "${OUT}"

echo "Done: ${OUT}"
