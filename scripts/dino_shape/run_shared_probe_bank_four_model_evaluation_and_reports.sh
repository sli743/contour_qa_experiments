#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
ROI="${1:-Lung_L}"
SEED="${SEED:-88}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BANK_ROOT="${BANK_ROOT:-/scratch/shaohan/contour_QA/shared_probe_banks}"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${ROI}" in
  Lung_L)
    BANK_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Lung_L_v1_eval.yaml"
    PILOT_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml"
    BENCH_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Lung_L_v1_eval.yaml"
    RANDOM_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Lung_L_v1_eval.yaml"
    STOCH_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v1_eval.yaml"
    PILOT_OUT="/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_shared_seed_test"
    BENCH_OUT="/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_shared_seed_test"
    RANDOM_OUT="/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_shared_seed_test"
    STOCH_OUT="/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Lung_L_v1/probe_eval_shared_seed_test"
    ;;
  Bronchus)
    BANK_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Bronchus_v1_eval.yaml"
    PILOT_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_Bronchus_probe_eval.yaml"
    BENCH_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Bronchus_v1_eval.yaml"
    RANDOM_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Bronchus_v1_eval.yaml"
    STOCH_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Bronchus_v1_eval.yaml"
    PILOT_OUT="/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Bronchus/probe_eval_shared_seed_test"
    BENCH_OUT="/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Bronchus_v1/probe_eval_shared_seed_test"
    RANDOM_OUT="/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Bronchus_v1/probe_eval_shared_seed_test"
    STOCH_OUT="/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Bronchus_v1/probe_eval_shared_seed_test"
    ;;
  *) echo "Usage: $0 {Lung_L|Bronchus}" >&2; exit 2 ;;
esac

BANK_DIR="${BANK_ROOT}/${ROI}_test_seed${SEED}_v1"

echo "[1/7] Building/reusing deterministic shared probe bank"
python -m evaluation.probes.build_shared_probe_bank \
  --config "${BANK_CONFIG}" --split test --output-dir "${BANK_DIR}" --seed "${SEED}"

run_model() {
  local name="$1" config="$2" out="$3" save_vis="$4"
  echo "Evaluating ${name} -> ${out}"
  local vis_args=()
  if [[ "${save_vis}" == "yes" ]]; then
    vis_args=(--save-visualization)
  fi
  python -m evaluation.probes.eval_dino_shape_shared_probe_bank \
    --config "${config}" \
    --probe-bank "${BANK_DIR}" \
    --output-dir "${out}" \
    --inference-batch-size "${BATCH_SIZE}" \
    "${vis_args[@]}"
}

echo "[2/7] Pilot AE"; run_model "Pilot AE" "${PILOT_CONFIG}" "${PILOT_OUT}" no
echo "[3/7] Benchmark-leakage DAE"; run_model "Benchmark-leakage DAE" "${BENCH_CONFIG}" "${BENCH_OUT}" no
echo "[4/7] Random-morphology DAE"; run_model "Random-morphology DAE" "${RANDOM_CONFIG}" "${RANDOM_OUT}" no
echo "[5/7] Stochastic-region-dropout DAE"; run_model "Stochastic-region-dropout DAE" "${STOCH_CONFIG}" "${STOCH_OUT}" yes

# The comparison runners use explicit ROI-specific variable names.
if [[ "${ROI}" == "Lung_L" ]]; then
  export LUNGL_PILOT_CSV="${PILOT_OUT}/probe_eval_sweep_detailed.csv"
  export LUNGL_BENCHMARK_CSV="${BENCH_OUT}/probe_eval_sweep_detailed.csv"
  export LUNGL_RANDOM_CSV="${RANDOM_OUT}/probe_eval_sweep_detailed.csv"
  export LUNGL_STOCHASTIC_CSV="${STOCH_OUT}/probe_eval_sweep_detailed.csv"
else
  export BRONCHUS_PILOT_CSV="${PILOT_OUT}/probe_eval_sweep_detailed.csv"
  export BRONCHUS_BENCHMARK_CSV="${BENCH_OUT}/probe_eval_sweep_detailed.csv"
  export BRONCHUS_RANDOM_CSV="${RANDOM_OUT}/probe_eval_sweep_detailed.csv"
  export BRONCHUS_STOCHASTIC_CSV="${STOCH_OUT}/probe_eval_sweep_detailed.csv"
fi

echo "[6/7] Probe-separated four-model comparison"
bash scripts/dino_shape/run_generate_four_model_per_probe_stochastic_dropout_reports.sh "${ROI}"

echo "[7/7] All-probes pooled four-model comparison"
bash scripts/dino_shape/run_generate_four_model_all_probes_stochastic_dropout_report.sh "${ROI}"

echo "Done. Shared bank: ${BANK_DIR}"
