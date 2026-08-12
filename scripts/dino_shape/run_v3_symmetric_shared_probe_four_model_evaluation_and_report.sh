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
V3_CONFIG="${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v3_symmetric_changed_loss.yaml"

PILOT_OUT="/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_shared_seed_test"
BENCH_OUT="/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_shared_seed_test"
RANDOM_OUT="/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_shared_seed_test"
V3_EXP="/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Lung_L_v3_symmetric_changed_loss"
# Default to the checkpoint selected by validation loss; override V3_CHECKPOINT to evaluate a final/other epoch.
V3_CHECKPOINT="${V3_CHECKPOINT:-${V3_EXP}/checkpoints/best_decoder.pth}"
V3_OUT="${V3_EXP}/probe_eval_shared_seed_test"
V3_VIS="${V3_OUT}/probe_visualization_shared"

REPORT_ROOT="${REPORT_ROOT:-/scratch/shaohan/contour_QA/roc_comparison/Lung_L_four_models_v3_symmetric_shared_probe}"
POOLED_OUT="${REPORT_ROOT}/pooled"
BY_PROBE_OUT="${REPORT_ROOT}/by_probe"
MASTER_REPORT="${REPORT_ROOT}/report.md"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF || true

if [[ ! -f "${V3_CHECKPOINT}" ]]; then
  echo "ERROR: v3 checkpoint not found: ${V3_CHECKPOINT}" >&2
  exit 1
fi

echo "[1/9] Build/reuse deterministic shared probe bank: ${BANK_DIR}"
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

echo "[2/9] Pilot AE"
run_eval_if_needed "Pilot AE" "${PILOT_CONFIG}" "${PILOT_OUT}"
echo "[3/9] Benchmark-leakage DAE"
run_eval_if_needed "Benchmark-leakage DAE" "${BENCH_CONFIG}" "${BENCH_OUT}"
echo "[4/9] Random-morphology DAE"
run_eval_if_needed "Random-morphology DAE" "${RANDOM_CONFIG}" "${RANDOM_OUT}"

echo "[5/9] Evaluate stochastic-region-dropout v3 symmetric-loss model"
rm -rf "${V3_OUT}"
python -m evaluation.probes.eval_dino_shape_shared_probe_bank \
  --config "${V3_CONFIG}" \
  --checkpoint "${V3_CHECKPOINT}" \
  --probe-bank "${BANK_DIR}" \
  --output-dir "${V3_OUT}" \
  --inference-batch-size "${BATCH_SIZE}" \
  --save-visualization \
  --visualization-dir "${V3_VIS}"

PILOT_CSV="${PILOT_OUT}/probe_eval_sweep_detailed.csv"
BENCH_CSV="${BENCH_OUT}/probe_eval_sweep_detailed.csv"
RANDOM_CSV="${RANDOM_OUT}/probe_eval_sweep_detailed.csv"
V3_CSV="${V3_OUT}/probe_eval_sweep_detailed.csv"
for f in "${PILOT_CSV}" "${BENCH_CSV}" "${RANDOM_CSV}" "${V3_CSV}"; do
  [[ -s "${f}" ]] || { echo "ERROR: missing evaluation CSV: ${f}" >&2; exit 1; }
done

echo "[6/9] Verify shared masks and Probe Contour Comparer visualization layout"
python -m analysis.verify_shared_probe_evaluations \
  --csv "Pilot AE=${PILOT_CSV}" \
  --csv "Benchmark-leakage DAE=${BENCH_CSV}" \
  --csv "Random-morphology DAE=${RANDOM_CSV}" \
  --csv "Stochastic-region-dropout DAE v3 symmetric=${V3_CSV}"
python -m analysis.verify_probe_contour_comparer_visualization \
  --manifest "${BANK_DIR}/probe_bank_manifest.csv" \
  --visualization-dir "${V3_VIS}"

rm -rf "${REPORT_ROOT}"
mkdir -p "${POOLED_OUT}" "${BY_PROBE_OUT}"

echo "[7/9] Pooled all-probes four-model report"
python -m analysis.generate_four_model_all_probes_two_criteria \
  --model "Pilot AE=${PILOT_CSV}" \
  --model "Benchmark-leakage DAE=${BENCH_CSV}" \
  --model "Random-morphology DAE=${RANDOM_CSV}" \
  --model "Stochastic-region-dropout DAE v3 symmetric=${V3_CSV}" \
  --roi "${ROI}" \
  --volume-threshold-pct "${GLOBAL_THRESHOLD_PCT}" \
  --local-depth-threshold-mm "${LOCAL_DEPTH_MM}" \
  --exclude-shifts \
  --n-boot "${N_BOOT}" \
  --seed "${SEED}" \
  --output-dir "${POOLED_OUT}"

echo "[8/9] Probe-separated four-model reports"
python -m analysis.generate_four_model_per_probe_reports \
  --model "Pilot AE=${PILOT_CSV}" \
  --model "Benchmark-leakage DAE=${BENCH_CSV}" \
  --model "Random-morphology DAE=${RANDOM_CSV}" \
  --model "Stochastic-region-dropout DAE v3 symmetric=${V3_CSV}" \
  --roi "${ROI}" \
  --volume-threshold-pct "${GLOBAL_THRESHOLD_PCT}" \
  --local-depth-threshold-mm "${LOCAL_DEPTH_MM}" \
  --probe-visualization-dir "${V3_VIS}" \
  --geometry-volume-tolerance-pct 0.5 \
  --exclude-shifts \
  --n-boot "${N_BOOT}" \
  --seed "${SEED}" \
  --output-dir "${BY_PROBE_OUT}"

echo "[9/9] Compose pooled + separate-probe Markdown report"
python -m analysis.compose_four_model_shared_probe_report \
  --pooled-dir "${POOLED_OUT}" \
  --by-probe-dir "${BY_PROBE_OUT}" \
  --output "${MASTER_REPORT}" \
  --roi "${ROI}" \
  --comparison-name "stochastic v3 symmetric changed-region loss" \
  --model-label "Pilot AE" \
  --model-label "Benchmark-leakage DAE" \
  --model-label "Random-morphology DAE" \
  --model-label "Stochastic-region-dropout DAE v3 symmetric"

if command -v pandoc >/dev/null 2>&1; then
  (
    cd "${REPORT_ROOT}"
    pandoc report.md --standalone --embed-resources -o report.html || true
  )
fi

echo
echo "Done."
echo "v3 evaluation CSV: ${V3_CSV}"
echo "Slicer visualization: ${V3_VIS}"
echo "Pooled report: ${POOLED_OUT}/all_probes_four_model_comparison.md"
echo "Per-probe index: ${BY_PROBE_OUT}/index.md"
echo "Combined report: ${MASTER_REPORT}"
