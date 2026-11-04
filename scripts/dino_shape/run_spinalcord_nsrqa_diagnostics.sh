#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
COMPARE_ROOT="${COMPARE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
TEMPORAL_SPLIT_YAML="${TEMPORAL_SPLIT_YAML:-${PROJECT_ROOT}/configs/dino_shape/month_temporal_nsrqa_split.yaml}"
FORCE_DIAGNOSTICS="${FORCE_DIAGNOSTICS:-0}"
SKIP_DEEP_PROBE_GEOMETRY="${SKIP_DEEP_PROBE_GEOMETRY:-0}"
BOUNDARY_MARGIN_MM="${BOUNDARY_MARGIN_MM:-4.5}"
SHORT_PERCENTILE="${SHORT_PERCENTILE:-10}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF || true

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fail() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -s "$1" ]] || fail "required file missing or empty: $1"; }

MODEL_MANIFEST="${COMPARE_ROOT}/manifests/model_configs.tsv"
INPUT_ROOT="${COMPARE_ROOT}/actual_autoseg_inputs"
DIAG_ROOT="${COMPARE_ROOT}/diagnostics"
ACQ_ROOT="${DIAG_ROOT}/acquisition_audit"
CLEAN_ROOT="${COMPARE_ROOT}/clean_approved_reconstruction"
REPORT_ROOT="${DIAG_ROOT}/four_level_report"
mkdir -p "${ACQ_ROOT}" "${CLEAN_ROOT}" "${REPORT_ROOT}" "${COMPARE_ROOT}/logs"

require_file "${MODEL_MANIFEST}"
require_file "${TEMPORAL_SPLIT_YAML}"
require_file "${INPUT_ROOT}/manifests/SpinalCord_preprocess_manifest.csv"

log "Stage 1: all-month raw-CT acquisition audit"
if [[ "${FORCE_DIAGNOSTICS}" == "1" || ! -s "${ACQ_ROOT}/all_months_acquisition_audit.csv" ]]; then
  python -m analysis.audit_spinalcord_incomplete_acquisition \
    --raw-data-root "${RAW_DATA_ROOT}" \
    --temporal-split-yaml "${TEMPORAL_SPLIT_YAML}" \
    --output-dir "${ACQ_ROOT}" \
    --roi SpinalCord \
    --boundary-margin-mm "${BOUNDARY_MARGIN_MM}" \
    --short-percentile "${SHORT_PERCENTILE}" \
    --confirmed-incomplete 95 129
fi
require_file "${ACQ_ROOT}/all_months_acquisition_audit.csv"

log "Stage 2: clean approved-contour reconstruction for both models and both localization geometries"
mapfile -t MODEL_ROWS < <(awk -F '\t' 'NR>1 && NF {gsub(/\r/, ""); print $1 "\t" $2 "\t" $3 "\t" $4}' "${MODEL_MANIFEST}")
[[ ${#MODEL_ROWS[@]} -eq 2 ]] || fail "expected two comparison model rows; found ${#MODEL_ROWS[@]}"
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r model_key model_label config checkpoint <<< "${row}"
  require_file "${config}"; require_file "${checkpoint}"
  if [[ "${model_key}" == "previous_generation" ]]; then corr_label="legacy_stochastic_region_dropout"; else corr_label="spinalcord_proposed_v4"; fi
  for mode in approved_reference autoseg_centered; do
    csv="${CLEAN_ROOT}/${model_key}/metrics/SpinalCord/${mode}_metrics.csv"
    if [[ "${FORCE_DIAGNOSTICS}" == "1" || ! -s "${csv}" ]]; then
      python -m evaluation.actual_autoseg.eval_clean_approved_reconstruction \
        --config "${config}" --checkpoint "${checkpoint}" \
        --input-root "${INPUT_ROOT}" --roi SpinalCord --mode "${mode}" \
        --case-output-root "${CLEAN_ROOT}/${model_key}" --output-csv "${csv}" \
        --model-name "${model_label}" --corruption-label "${corr_label}"
    fi
    require_file "${csv}"
  done
done

log "Stage 2.5: actual V4-family correction diagnostic + legacy cut-orientation audit"
V4_FAMILY_ROOT="${DIAG_ROOT}/v4_family_correction"
if [[ "${FORCE_DIAGNOSTICS}" == "1" || ! -s "${V4_FAMILY_ROOT}/report.md" ]]; then
  python -m analysis.eval_spinalcord_v4_family_diagnostics \
    --compare-root "${COMPARE_ROOT}" \
    --output-dir "${V4_FAMILY_ROOT}" \
    --splits validation1 testing1 \
    --surface-repeats 2 \
    --composite-repeats 2
fi
require_file "${V4_FAMILY_ROOT}/report.md"

log "Stage 3: four-level diagnostic analysis"
probe_bank="$(find "${COMPARE_ROOT}/shared_probe_bank" -name probe_bank_manifest.csv -type f | head -n 1 || true)"
[[ -n "${probe_bank}" ]] || fail "shared probe-bank manifest not found under ${COMPARE_ROOT}/shared_probe_bank"
extra=()
if [[ "${SKIP_DEEP_PROBE_GEOMETRY}" == "1" ]]; then extra+=(--skip-deep-probe-geometry); fi
python -m analysis.analyze_spinalcord_nsrqa_diagnostics \
  --compare-root "${COMPARE_ROOT}" \
  --acquisition-audit-csv "${ACQ_ROOT}/all_months_acquisition_audit.csv" \
  --probe-bank-manifest "${probe_bank}" \
  --output-dir "${REPORT_ROOT}" \
  "${extra[@]}"
require_file "${REPORT_ROOT}/report.md"
if command -v pandoc >/dev/null 2>&1; then
  (cd "${REPORT_ROOT}" && pandoc report.md --standalone --embed-resources -o report.html || true)
fi

log "Diagnostics complete"
echo "Four-level report:            ${REPORT_ROOT}/report.md"
echo "Acquisition report:           ${ACQ_ROOT}/report.md"
echo "Incomplete patient CSV:       ${ACQ_ROOT}/incomplete_acquisition_patient_list.csv"
echo "Boundary-review patient CSV:  ${ACQ_ROOT}/boundary_limited_review_list.csv"
echo "All-patient acquisition CSV:  ${ACQ_ROOT}/all_months_acquisition_audit.csv"
echo "Previous clean viewer root:   ${CLEAN_ROOT}/previous_generation"
echo "V4 clean viewer root:         ${CLEAN_ROOT}/v4_corruption"
echo "V4-family diagnostic report:  ${V4_FAMILY_ROOT}/report.md"
echo
echo "Confirmed/likely incomplete-acquisition list:"
cat "${ACQ_ROOT}/incomplete_acquisition_patient_list.txt" || true
