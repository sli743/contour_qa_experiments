#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
BANK_ROOT="${BANK_ROOT:-/scratch/shaohan/contour_QA/shared_probe_banks}"
SEED="${SEED:-88}"
N_BOOT="${N_BOOT:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FORCE_EVAL="${FORCE_EVAL:-0}"
ROI_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "${ROI_INDEX}" ]] || { echo "Usage: $0 ROI_INDEX" >&2; exit 2; }
MANIFEST="${STUDY_ROOT}/manifests/active_rois.tsv"
line="$(awk -F '\t' -v idx="${ROI_INDEX}" 'NR>1 && $1==idx {print; exit}' "${MANIFEST}")"
[[ -n "${line}" ]] || { echo "No active ROI index ${ROI_INDEX}" >&2; exit 2; }
line="${line%$'\r'}"
IFS=$'\t' read -r roi_index registry_index ROI data_group profile GLOBAL_THRESHOLD_PCT LOCAL_DEPTH_MM PILOT_CONFIG BENCH_CONFIG RANDOM_CONFIG NSRQA_CONFIG <<< "${line}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF || true

BANK_DIR="${BANK_ROOT}/${ROI}_test_seed${SEED}_allroi_v1"
PILOT_EXP="${STUDY_ROOT}/models/pilot/dino_shape_pilot_${ROI}_allroi"
BENCH_EXP="${STUDY_ROOT}/models/benchmark_overlap/dino_shape_benchmark_overlap_${ROI}_allroi"
RANDOM_EXP="${STUDY_ROOT}/models/random_morphology/dino_shape_random_morphology_${ROI}_allroi"
NSRQA_EXP="${STUDY_ROOT}/models/nsrqa/dino_shape_NSRQA_${ROI}_allroi"
for ckpt in "${PILOT_EXP}/checkpoints/best_decoder.pth" "${BENCH_EXP}/checkpoints/best_decoder.pth" "${RANDOM_EXP}/checkpoints/best_decoder.pth" "${NSRQA_EXP}/checkpoints/best_decoder.pth"; do
  [[ -s "${ckpt}" ]] || { echo "Missing checkpoint: ${ckpt}" >&2; exit 3; }
done

EVAL_ROOT="${STUDY_ROOT}/evaluations/${ROI}"
PILOT_OUT="${EVAL_ROOT}/pilot/probe_eval_shared_seed_test"
BENCH_OUT="${EVAL_ROOT}/benchmark_overlap/probe_eval_shared_seed_test"
RANDOM_OUT="${EVAL_ROOT}/random_morphology/probe_eval_shared_seed_test"
NSRQA_OUT="${EVAL_ROOT}/nsrqa/probe_eval_shared_seed_test"
PILOT_VIS="${PILOT_OUT}/probe_visualization_shared"
BENCH_VIS="${BENCH_OUT}/probe_visualization_shared"
RANDOM_VIS="${RANDOM_OUT}/probe_visualization_shared"
NSRQA_VIS="${NSRQA_OUT}/probe_visualization_shared"
REPORT_ROOT="${STUDY_ROOT}/reports/${ROI}_four_models_NSRQA_shared_probe"
POOLED_OUT="${REPORT_ROOT}/pooled"; BY_PROBE_OUT="${REPORT_ROOT}/by_probe"; MASTER_REPORT="${REPORT_ROOT}/report.md"

echo "[${ROI}] Build/reuse shared probe bank"
python -m evaluation.probes.build_shared_probe_bank --config "${NSRQA_CONFIG}" --split test --output-dir "${BANK_DIR}" --seed "${SEED}"

run_eval() {
  local name="$1" config="$2" out="$3" vis="$4"
  local csv="${out}/probe_eval_sweep_detailed.csv"
  local have_vis=0

  if [[ -d "${vis}" ]] && compgen -G "${vis}/reconstruction_*.nii.gz" >/dev/null; then
    have_vis=1
  fi

  if [[ "${FORCE_EVAL}" != "1" && -s "${csv}" && "${have_vis}" == "1" ]]; then
    echo "Reusing ${name}: ${csv}"
    echo "Reusing ${name} visualization: ${vis}"
    return
  fi

  if [[ "${FORCE_EVAL}" != "1" && -s "${csv}" && "${have_vis}" != "1" ]]; then
    echo "${name} metrics exist but visualization is missing; rerunning inference to export visualization."
  fi

  rm -rf "${out}"
  python -m evaluation.probes.eval_dino_shape_shared_probe_bank \
    --config "${config}" \
    --probe-bank "${BANK_DIR}" \
    --output-dir "${out}" \
    --inference-batch-size "${BATCH_SIZE}" \
    --save-visualization \
    --visualization-dir "${vis}"
}
run_eval "Clean AE" "${PILOT_CONFIG}" "${PILOT_OUT}" "${PILOT_VIS}"
run_eval "Benchmark-overlap DAE" "${BENCH_CONFIG}" "${BENCH_OUT}" "${BENCH_VIS}"
run_eval "Random-morphology DAE" "${RANDOM_CONFIG}" "${RANDOM_OUT}" "${RANDOM_VIS}"
run_eval "NSR-QA" "${NSRQA_CONFIG}" "${NSRQA_OUT}" "${NSRQA_VIS}"

PILOT_CSV="${PILOT_OUT}/probe_eval_sweep_detailed.csv"; BENCH_CSV="${BENCH_OUT}/probe_eval_sweep_detailed.csv"; RANDOM_CSV="${RANDOM_OUT}/probe_eval_sweep_detailed.csv"; NSRQA_CSV="${NSRQA_OUT}/probe_eval_sweep_detailed.csv"
python -m analysis.verify_shared_probe_evaluations --csv "Clean AE=${PILOT_CSV}" --csv "Benchmark-overlap DAE=${BENCH_CSV}" --csv "Random-morphology DAE=${RANDOM_CSV}" --csv "NSR-QA=${NSRQA_CSV}"
for vis in "${PILOT_VIS}" "${BENCH_VIS}" "${RANDOM_VIS}" "${NSRQA_VIS}"; do
  python -m analysis.verify_probe_contour_comparer_visualization \
    --manifest "${BANK_DIR}/probe_bank_manifest.csv" \
    --visualization-dir "${vis}"
done

rm -rf "${REPORT_ROOT}"; mkdir -p "${POOLED_OUT}" "${BY_PROBE_OUT}"
models=(--model "Clean AE=${PILOT_CSV}" --model "Benchmark-overlap DAE=${BENCH_CSV}" --model "Random-morphology DAE=${RANDOM_CSV}" --model "NSR-QA=${NSRQA_CSV}")
python -m analysis.generate_four_model_all_probes_two_criteria "${models[@]}" --roi "${ROI}" --volume-threshold-pct "${GLOBAL_THRESHOLD_PCT}" --local-depth-threshold-mm "${LOCAL_DEPTH_MM}" --exclude-shifts --n-boot "${N_BOOT}" --seed "${SEED}" --output-dir "${POOLED_OUT}"
python -m analysis.generate_four_model_per_probe_reports "${models[@]}" --roi "${ROI}" --volume-threshold-pct "${GLOBAL_THRESHOLD_PCT}" --local-depth-threshold-mm "${LOCAL_DEPTH_MM}" --probe-visualization-dir "${NSRQA_VIS}" --geometry-volume-tolerance-pct 0.5 --exclude-shifts --n-boot "${N_BOOT}" --seed "${SEED}" --output-dir "${BY_PROBE_OUT}"
python -m analysis.compose_four_model_shared_probe_report --pooled-dir "${POOLED_OUT}" --by-probe-dir "${BY_PROBE_OUT}" --output "${MASTER_REPORT}" --roi "${ROI}" --comparison-name "${ROI} all-ROI NSR-QA study" --model-label "Clean AE" --model-label "Benchmark-overlap DAE" --model-label "Random-morphology DAE" --model-label "NSR-QA"
if command -v pandoc >/dev/null 2>&1; then (cd "${REPORT_ROOT}" && pandoc report.md --standalone --embed-resources -o report.html || true); fi

echo "Done ${ROI}: ${MASTER_REPORT}"
echo "Slicer/CT visualizations:"
echo "  Clean AE:              ${PILOT_VIS}"
echo "  Benchmark-overlap DAE: ${BENCH_VIS}"
echo "  Random-morphology DAE: ${RANDOM_VIS}"
echo "  NSR-QA:                ${NSRQA_VIS}"
