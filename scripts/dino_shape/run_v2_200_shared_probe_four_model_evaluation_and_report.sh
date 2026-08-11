#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
SEED="${SEED:-88}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_BOOT="${N_BOOT:-1000}"
FORCE_BASELINE_EVAL="${FORCE_BASELINE_EVAL:-0}"

ROI="Lung_L"
GLOBAL_THRESHOLD_PCT="${LUNGL_THRESHOLD_PCT:-20}"
LOCAL_DEPTH_MM="${LUNGL_LOCAL_DEPTH_MM:-10}"

BANK_ROOT="${BANK_ROOT:-/scratch/shaohan/contour_QA/shared_probe_banks}"
BANK_DIR="${BANK_ROOT}/Lung_L_test_seed${SEED}_v1"

PILOT_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml"
BENCH_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Lung_L_v1_eval.yaml"
RANDOM_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Lung_L_v1_eval.yaml"
V2_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v2_global_boundary.yaml"

PILOT_OUT="/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_shared_seed_test"
BENCH_OUT="/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_shared_seed_test"
RANDOM_OUT="/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_shared_seed_test"
V2_EXP="/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Lung_L_v2_global_boundary"
V2_CHECKPOINT="${V2_CHECKPOINT:-${V2_EXP}/checkpoints/best_decoder.pth}"
V2_OUT="${V2_EXP}/probe_eval_shared_seed_test"
V2_VIS="${V2_OUT}/probe_visualization_shared"

REPORT_ROOT="${REPORT_ROOT:-/scratch/shaohan/contour_QA/roc_comparison/Lung_L_four_models_v2_200_shared_probe}"
POOLED_OUT="${REPORT_ROOT}/pooled"
BY_PROBE_OUT="${REPORT_ROOT}/by_probe"
MASTER_REPORT="${REPORT_ROOT}/report.md"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# The 3DINO environment uses an older PyTorch build that does not understand this newer allocator option.
unset PYTORCH_CUDA_ALLOC_CONF || true

if [[ ! -f "${V2_CHECKPOINT}" ]]; then
  echo "ERROR: v2-200 checkpoint not found: ${V2_CHECKPOINT}" >&2
  exit 1
fi

echo "[1/8] Build/reuse deterministic shared probe bank: ${BANK_DIR}"
python -m evaluation.probes.build_shared_probe_bank \
  --config "${BENCH_CONFIG}" \
  --split test \
  --output-dir "${BANK_DIR}" \
  --seed "${SEED}"

run_eval_if_needed() {
  local name="$1" config="$2" out="$3"
  local csv="${out}/probe_eval_sweep_detailed.csv"
  if [[ "${FORCE_BASELINE_EVAL}" != "1" && -s "${csv}" ]]; then
    echo "Reusing ${name}: ${csv}"
    return 0
  fi
  echo "Evaluating ${name} -> ${out}"
  python -m evaluation.probes.eval_dino_shape_shared_probe_bank \
    --config "${config}" \
    --probe-bank "${BANK_DIR}" \
    --output-dir "${out}" \
    --inference-batch-size "${BATCH_SIZE}"
}

echo "[2/8] Pilot AE"
run_eval_if_needed "Pilot AE" "${PILOT_CONFIG}" "${PILOT_OUT}"
echo "[3/8] Benchmark-leakage DAE"
run_eval_if_needed "Benchmark-leakage DAE" "${BENCH_CONFIG}" "${BENCH_OUT}"
echo "[4/8] Random-morphology DAE"
run_eval_if_needed "Random-morphology DAE" "${RANDOM_CONFIG}" "${RANDOM_OUT}"

echo "[5/8] Evaluate stochastic-region-dropout v2-200 using best_decoder.pth"
rm -rf "${V2_OUT}"
python -m evaluation.probes.eval_dino_shape_shared_probe_bank \
  --config "${V2_CONFIG}" \
  --checkpoint "${V2_CHECKPOINT}" \
  --probe-bank "${BANK_DIR}" \
  --output-dir "${V2_OUT}" \
  --inference-batch-size "${BATCH_SIZE}" \
  --save-visualization \
  --visualization-dir "${V2_VIS}"

PILOT_CSV="${PILOT_OUT}/probe_eval_sweep_detailed.csv"
BENCH_CSV="${BENCH_OUT}/probe_eval_sweep_detailed.csv"
RANDOM_CSV="${RANDOM_OUT}/probe_eval_sweep_detailed.csv"
V2_CSV="${V2_OUT}/probe_eval_sweep_detailed.csv"
for f in "${PILOT_CSV}" "${BENCH_CSV}" "${RANDOM_CSV}" "${V2_CSV}"; do
  [[ -s "${f}" ]] || { echo "ERROR: missing evaluation CSV: ${f}" >&2; exit 1; }
done

echo "Verifying that all four CSVs use identical deterministic probe masks"
python -m analysis.verify_shared_probe_evaluations \
  --csv "Pilot AE=${PILOT_CSV}" \
  --csv "Benchmark-leakage DAE=${BENCH_CSV}" \
  --csv "Random-morphology DAE=${RANDOM_CSV}" \
  --csv "Stochastic-region-dropout DAE v2-200=${V2_CSV}"

rm -rf "${REPORT_ROOT}"
mkdir -p "${POOLED_OUT}" "${BY_PROBE_OUT}"

echo "[6/8] Pooled all-probes four-model report"
python -m analysis.generate_four_model_all_probes_two_criteria \
  --model "Pilot AE=${PILOT_CSV}" \
  --model "Benchmark-leakage DAE=${BENCH_CSV}" \
  --model "Random-morphology DAE=${RANDOM_CSV}" \
  --model "Stochastic-region-dropout DAE v2-200=${V2_CSV}" \
  --roi "${ROI}" \
  --volume-threshold-pct "${GLOBAL_THRESHOLD_PCT}" \
  --local-depth-threshold-mm "${LOCAL_DEPTH_MM}" \
  --exclude-shifts \
  --n-boot "${N_BOOT}" \
  --seed "${SEED}" \
  --output-dir "${POOLED_OUT}"

echo "[7/8] Probe-separated four-model reports"
python -m analysis.generate_four_model_per_probe_reports \
  --model "Pilot AE=${PILOT_CSV}" \
  --model "Benchmark-leakage DAE=${BENCH_CSV}" \
  --model "Random-morphology DAE=${RANDOM_CSV}" \
  --model "Stochastic-region-dropout DAE v2-200=${V2_CSV}" \
  --roi "${ROI}" \
  --volume-threshold-pct "${GLOBAL_THRESHOLD_PCT}" \
  --local-depth-threshold-mm "${LOCAL_DEPTH_MM}" \
  --probe-visualization-dir "${V2_VIS}" \
  --geometry-volume-tolerance-pct 0.5 \
  --exclude-shifts \
  --n-boot "${N_BOOT}" \
  --seed "${SEED}" \
  --output-dir "${BY_PROBE_OUT}"

echo "[8/8] Compose one Markdown report containing pooled + separate-probe results"
python -m analysis.compose_v2_200_four_model_report \
  --pooled-dir "${POOLED_OUT}" \
  --by-probe-dir "${BY_PROBE_OUT}" \
  --output "${MASTER_REPORT}" \
  --roi "${ROI}"

if command -v pandoc >/dev/null 2>&1; then
  (
    cd "${REPORT_ROOT}"
    pandoc report.md --standalone --embed-resources -o report.html || true
  )
fi

echo
echo "Done."
echo "v2 evaluation CSV: ${V2_CSV}"
echo "Pooled report:       ${POOLED_OUT}/all_probes_four_model_comparison.md"
echo "Per-probe index:     ${BY_PROBE_OUT}/index.md"
echo "Combined report:     ${MASTER_REPORT}"
