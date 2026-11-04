#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
COMPARE_ROOT="${COMPARE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
CENTER_METHOD="${CENTER_METHOD:-bbox_midpoint}"
FORCE_REFERENCE_FRAME="${FORCE_REFERENCE_FRAME:-0}"
MANUAL_INCOMPLETE_PATIENTS="${MANUAL_INCOMPLETE_PATIENTS:-95 129 145}"
MODE="autoseg_xy_ct_z"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF || true

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fail() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -s "$1" ]] || fail "required file missing or empty: $1"; }

MODEL_MANIFEST="${COMPARE_ROOT}/manifests/model_configs.tsv"
INPUT_ROOT="${COMPARE_ROOT}/reference_frame_inputs"
RESULT_ROOT="${COMPARE_ROOT}/actual_autoseg_results"
CLEAN_ROOT="${COMPARE_ROOT}/clean_approved_reconstruction"
REPORT_ROOT="${COMPARE_ROOT}/diagnostics/reference_frame"
mkdir -p "${INPUT_ROOT}" "${REPORT_ROOT}" "${COMPARE_ROOT}/logs"
require_file "${MODEL_MANIFEST}"

PREV_CONFIG="$(awk -F '\t' 'NR>1 && $1=="previous_generation" {print $3; exit}' "${MODEL_MANIFEST}")"
require_file "${PREV_CONFIG}"

log "Stage 1: materialize candidate-X/Y + CT-slice-axis reference-frame inputs"
HYBRID_MANIFEST="${INPUT_ROOT}/manifests/SpinalCord_preprocess_manifest.csv"
if [[ "${FORCE_REFERENCE_FRAME}" == "1" || ! -s "${HYBRID_MANIFEST}" ]]; then
  rm -rf "${INPUT_ROOT}"
  python -m evaluation.actual_autoseg.preprocess_actual_autoseg \
    --config "${PREV_CONFIG}" --roi SpinalCord --group spine \
    --raw-data-root "${RAW_DATA_ROOT}" --output-root "${INPUT_ROOT}" \
    --mode "${MODE}" --center-method "${CENTER_METHOD}" \
    --splits training1 validation1 testing1
fi
require_file "${HYBRID_MANIFEST}"

log "Stage 2: evaluate existing previous-generation and v4 checkpoints in the hybrid frame"
mapfile -t MODEL_ROWS < <(awk -F '\t' 'NR>1 && NF {gsub(/\r/, ""); print $1 "\t" $2 "\t" $3 "\t" $4}' "${MODEL_MANIFEST}")
[[ ${#MODEL_ROWS[@]} -eq 2 ]] || fail "expected two comparison model rows; found ${#MODEL_ROWS[@]}"
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r model_key model_label config checkpoint <<< "${row}"
  require_file "${config}"; require_file "${checkpoint}"
  if [[ "${model_key}" == "previous_generation" ]]; then
    corr_label="legacy_stochastic_region_dropout"
  else
    corr_label="spinalcord_proposed_v4"
  fi

  metrics_csv="${RESULT_ROOT}/${model_key}/metrics/SpinalCord/${MODE}_metrics.csv"
  if [[ "${FORCE_REFERENCE_FRAME}" == "1" || ! -s "${metrics_csv}" ]]; then
    python -m evaluation.actual_autoseg.eval_actual_autoseg_nsrqa \
      --config "${config}" --checkpoint "${checkpoint}" \
      --input-root "${INPUT_ROOT}" --case-output-root "${RESULT_ROOT}/${model_key}" \
      --roi SpinalCord --mode "${MODE}" --output-csv "${metrics_csv}" \
      --model-name "${model_label}" --corruption-label "${corr_label}"
  fi
  require_file "${metrics_csv}"

  clean_csv="${CLEAN_ROOT}/${model_key}/metrics/SpinalCord/${MODE}_metrics.csv"
  if [[ "${FORCE_REFERENCE_FRAME}" == "1" || ! -s "${clean_csv}" ]]; then
    python -m evaluation.actual_autoseg.eval_clean_approved_reconstruction \
      --config "${config}" --checkpoint "${checkpoint}" \
      --input-root "${INPUT_ROOT}" --roi SpinalCord --mode "${MODE}" \
      --case-output-root "${CLEAN_ROOT}/${model_key}" --output-csv "${clean_csv}" \
      --model-name "${model_label}" --corruption-label "${corr_label}"
  fi
  require_file "${clean_csv}"
done

log "Stage 3: paired reference-frame analysis"
read -r -a excluded <<< "${MANUAL_INCOMPLETE_PATIENTS}"
python -m analysis.analyze_spinalcord_reference_frame \
  --compare-root "${COMPARE_ROOT}" \
  --output-dir "${REPORT_ROOT}" \
  --exclude-patients "${excluded[@]}"
require_file "${REPORT_ROOT}/report.md"
if command -v pandoc >/dev/null 2>&1; then
  (cd "${REPORT_ROOT}" && pandoc report.md --standalone --embed-resources -o report.html || true)
fi

log "Reference-frame diagnostic complete"
echo "Report: ${REPORT_ROOT}/report.md"
echo "Hybrid input manifest: ${HYBRID_MANIFEST}"
echo "V4 viewer root (hybrid mode is a new subfolder): ${RESULT_ROOT}/v4_corruption"
echo "V4 clean viewer root: ${CLEAN_ROOT}/v4_corruption"
echo "Patient comparison: ${REPORT_ROOT}/v4_testing1_patient_localization_comparison.csv"
