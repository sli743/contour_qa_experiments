#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
SOURCE_STUDY_ROOT="${SOURCE_STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
COMPARE_ROOT="${COMPARE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare}"
V4_CONFIG="${V4_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/spinalcord_proposed_corruption_distribution_v4.yaml}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
CLINICAL_METRICS_CSV="${CLINICAL_METRICS_CSV:-${RAW_DATA_ROOT}/metrics.csv}"
DOMAIN_CSV="${DOMAIN_CSV:-/home/shaohan/radformation-data/OOD_Domain_Assignment_MAD_TOTAL.csv}"
CLINICAL_METRICS_DUPLICATE_POLICY="${CLINICAL_METRICS_DUPLICATE_POLICY:-exclude}"
AUTOSEG_CENTER_METHOD="${AUTOSEG_CENTER_METHOD:-bbox_midpoint}"
SEED="${SEED:-88}"
N_BOOT="${N_BOOT:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FRESH_TRAINING="${FRESH_TRAINING:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF || true

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fail() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -s "$1" ]] || fail "required file missing or empty: $1"; }

require_file "${SOURCE_STUDY_ROOT}/manifests/temporal_split_summary.json"
require_file "${SOURCE_STUDY_ROOT}/manifests/spinalcord_large_fov_spec.json"
require_file "${V4_CONFIG}"
require_file "${CLINICAL_METRICS_CSV}"

mkdir -p "${COMPARE_ROOT}" "${COMPARE_ROOT}/logs"

log "Stage 0: materialize controlled old-vs-v4 configs"
python scripts/dino_shape/prepare_spinalcord_nsrqa_v4_comparison.py \
  --source-study-root "${SOURCE_STUDY_ROOT}" \
  --output-root "${COMPARE_ROOT}" \
  --v4-config "${V4_CONFIG}"

MODEL_MANIFEST="${COMPARE_ROOT}/manifests/model_configs.tsv"
require_file "${MODEL_MANIFEST}"
mapfile -t MODEL_ROWS < <(awk -F '\t' 'NR>1 && NF {gsub(/\r/, ""); print $1 "\t" $2 "\t" $3 "\t" $4}' "${MODEL_MANIFEST}")
[[ ${#MODEL_ROWS[@]} -eq 2 ]] || fail "expected two comparison model rows; found ${#MODEL_ROWS[@]}"

log "Stage 1: train previous-generation and v4 NSR-QA under identical non-corruption settings"
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r model_key model_label config checkpoint <<< "${row}"
  require_file "${config}"
  if [[ "${FRESH_TRAINING}" == "1" ]]; then
    echo "Training ${model_label} from scratch"
    python -m training.train_dino_shape_stochastic_region_dropout_decoder --config "${config}" --fresh
  elif [[ -s "${checkpoint}" ]]; then
    echo "Reusing completed checkpoint for ${model_label}: ${checkpoint}"
  else
    echo "Training/resuming ${model_label}"
    python -m training.train_dino_shape_stochastic_region_dropout_decoder --config "${config}" --resume auto
  fi
  require_file "${checkpoint}"
done

PREV_CONFIG="$(awk -F '\t' 'NR>1 && $1=="previous_generation" {print $3; exit}' "${MODEL_MANIFEST}")"
PREV_CKPT="$(awk -F '\t' 'NR>1 && $1=="previous_generation" {print $4; exit}' "${MODEL_MANIFEST}")"
V4_TRAIN_CONFIG="$(awk -F '\t' 'NR>1 && $1=="v4_corruption" {print $3; exit}' "${MODEL_MANIFEST}")"
V4_CKPT="$(awk -F '\t' 'NR>1 && $1=="v4_corruption" {print $4; exit}' "${MODEL_MANIFEST}")"

log "Stage 2: shared deterministic synthetic-probe evaluation (secondary)"
BANK_DIR="${COMPARE_ROOT}/shared_probe_bank/SpinalCord_test_seed${SEED}"
if [[ "${FORCE_EVAL}" == "1" || ! -s "${BANK_DIR}/probe_bank_manifest.csv" ]]; then
  rm -rf "${BANK_DIR}"
  python -m evaluation.probes.build_shared_probe_bank \
    --config "${PREV_CONFIG}" --split test --output-dir "${BANK_DIR}" --seed "${SEED}"
fi
require_file "${BANK_DIR}/probe_bank_manifest.csv"
for model_key in previous_generation v4_corruption; do
  if [[ "${model_key}" == "previous_generation" ]]; then config="${PREV_CONFIG}"; checkpoint="${PREV_CKPT}"; else config="${V4_TRAIN_CONFIG}"; checkpoint="${V4_CKPT}"; fi
  out="${COMPARE_ROOT}/synthetic_probe/${model_key}"
  vis="${out}/probe_visualization_shared"
  if [[ "${FORCE_EVAL}" == "1" || ! -s "${out}/probe_eval_sweep_detailed.csv" ]]; then
    rm -rf "${out}"
    python -m evaluation.probes.eval_dino_shape_shared_probe_bank \
      --config "${config}" --checkpoint "${checkpoint}" \
      --probe-bank "${BANK_DIR}" --output-dir "${out}" \
      --inference-batch-size "${BATCH_SIZE}" \
      --save-visualization --visualization-dir "${vis}"
  fi
  require_file "${out}/probe_eval_sweep_detailed.csv"
done
python -m analysis.verify_shared_probe_evaluations \
  --csv "Previous-generation NSR-QA=${COMPARE_ROOT}/synthetic_probe/previous_generation/probe_eval_sweep_detailed.csv" \
  --csv "V4 corruption NSR-QA=${COMPARE_ROOT}/synthetic_probe/v4_corruption/probe_eval_sweep_detailed.csv"

log "Stage 3: preprocess actual autosegs once (shared input geometry)"
INPUT_ROOT="${COMPARE_ROOT}/actual_autoseg_inputs"
PREPROCESS_MANIFEST="${INPUT_ROOT}/manifests/SpinalCord_preprocess_manifest.csv"
if [[ "${FORCE_EVAL}" == "1" || ! -s "${PREPROCESS_MANIFEST}" ]]; then
  rm -rf "${INPUT_ROOT}"
  python -m evaluation.actual_autoseg.preprocess_actual_autoseg \
    --config "${PREV_CONFIG}" --roi SpinalCord --group spine \
    --raw-data-root "${RAW_DATA_ROOT}" --output-root "${INPUT_ROOT}" \
    --mode both --center-method "${AUTOSEG_CENTER_METHOD}" \
    --splits training1 validation1 testing1
fi
require_file "${PREPROCESS_MANIFEST}"

log "Stage 4: actual-autoseg inference for both models and localization modes"
for model_key in previous_generation v4_corruption; do
  if [[ "${model_key}" == "previous_generation" ]]; then
    config="${PREV_CONFIG}"; checkpoint="${PREV_CKPT}"; model_label="Previous-generation NSR-QA"; corr_label="legacy_stochastic_region_dropout"
  else
    config="${V4_TRAIN_CONFIG}"; checkpoint="${V4_CKPT}"; model_label="V4 corruption NSR-QA"; corr_label="spinalcord_proposed_v4"
  fi
  model_root="${COMPARE_ROOT}/actual_autoseg_results/${model_key}"
  for mode in autoseg_centered approved_reference; do
    csv="${model_root}/metrics/SpinalCord/${mode}_metrics.csv"
    if [[ "${FORCE_EVAL}" == "1" || ! -s "${csv}" ]]; then
      python -m evaluation.actual_autoseg.eval_actual_autoseg_nsrqa \
        --config "${config}" --checkpoint "${checkpoint}" \
        --input-root "${INPUT_ROOT}" --case-output-root "${model_root}" \
        --roi SpinalCord --mode "${mode}" --output-csv "${csv}" \
        --model-name "${model_label}" --corruption-label "${corr_label}"
    fi
    require_file "${csv}"
  done

  log "Stage 5 (${model_key}): standard regression and P75/P50 classification reports"
  python -m analysis.analyze_actual_autoseg_regression \
    --roi SpinalCord \
    --metrics "autoseg_centered=${model_root}/metrics/SpinalCord/autoseg_centered_metrics.csv" \
    --metrics "approved_reference=${model_root}/metrics/SpinalCord/approved_reference_metrics.csv" \
    --output-dir "${model_root}/regression/SpinalCord" --n-boot "${N_BOOT}" --seed "${SEED}"

  python -m analysis.analyze_actual_autoseg_ranking_classification \
    --roi SpinalCord \
    --metrics "autoseg_centered=${model_root}/metrics/SpinalCord/autoseg_centered_metrics.csv" \
    --metrics "approved_reference=${model_root}/metrics/SpinalCord/approved_reference_metrics.csv" \
    --output-dir "${model_root}/ranking_classification/SpinalCord" \
    --clinical-metrics-csv "${CLINICAL_METRICS_CSV}" \
    --clinical-site thorax \
    --clinical-metrics-duplicate-policy "${CLINICAL_METRICS_DUPLICATE_POLICY}" \
    --n-boot "${N_BOOT}" --seed "${SEED}"

  python analysis/generate_actual_autoseg_ranking_classification_report.py \
    --root "${model_root}" \
    --output-dir "${model_root}/ranking_classification_report"
done

log "Stage 6: paired primary comparison + domain robustness report"
PAIR_REPORT_DIR="${COMPARE_ROOT}/reports/previous_vs_v4_actual_autoseg"
domain_args=()
if [[ -s "${DOMAIN_CSV}" ]]; then
  domain_args=(--domain-csv "${DOMAIN_CSV}")
else
  echo "WARNING: domain CSV not found; domain analysis will be skipped: ${DOMAIN_CSV}" >&2
fi
python -m analysis.analyze_spinalcord_nsrqa_v4_comparison \
  --previous-primary "${COMPARE_ROOT}/actual_autoseg_results/previous_generation/metrics/SpinalCord/autoseg_centered_metrics.csv" \
  --v4-primary "${COMPARE_ROOT}/actual_autoseg_results/v4_corruption/metrics/SpinalCord/autoseg_centered_metrics.csv" \
  --previous-oracle "${COMPARE_ROOT}/actual_autoseg_results/previous_generation/metrics/SpinalCord/approved_reference_metrics.csv" \
  --v4-oracle "${COMPARE_ROOT}/actual_autoseg_results/v4_corruption/metrics/SpinalCord/approved_reference_metrics.csv" \
  --clinical-metrics-csv "${CLINICAL_METRICS_CSV}" \
  --clinical-metrics-duplicate-policy "${CLINICAL_METRICS_DUPLICATE_POLICY}" \
  "${domain_args[@]}" \
  --previous-probe-csv "${COMPARE_ROOT}/synthetic_probe/previous_generation/probe_eval_sweep_detailed.csv" \
  --v4-probe-csv "${COMPARE_ROOT}/synthetic_probe/v4_corruption/probe_eval_sweep_detailed.csv" \
  --output-dir "${PAIR_REPORT_DIR}" --n-boot "${N_BOOT}" --seed "${SEED}"
require_file "${PAIR_REPORT_DIR}/report.md"
if command -v pandoc >/dev/null 2>&1; then
  (cd "${PAIR_REPORT_DIR}" && pandoc report.md --standalone --embed-resources -o report.html || true)
fi

log "Comparison pipeline complete"
echo "Primary paired report: ${PAIR_REPORT_DIR}/report.md"
echo "Previous-generation viewer root: ${COMPARE_ROOT}/actual_autoseg_results/previous_generation"
echo "V4 viewer root:                  ${COMPARE_ROOT}/actual_autoseg_results/v4_corruption"
echo "Previous-generation report:     ${COMPARE_ROOT}/actual_autoseg_results/previous_generation/ranking_classification_report/report.md"
echo "V4 report:                      ${COMPARE_ROOT}/actual_autoseg_results/v4_corruption/ranking_classification_report/report.md"
echo "Domain tables:                  ${PAIR_REPORT_DIR}/domain_analysis/"
