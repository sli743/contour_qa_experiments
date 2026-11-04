#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_reference_ablation}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
AUTOSEG_CENTER_METHOD="${AUTOSEG_CENTER_METHOD:-bbox_midpoint}"
FORCE_EVAL="${FORCE_EVAL:-0}"
N_BOOT="${N_BOOT:-5000}"
SEED="${SEED:-88}"
EXCLUDE_PATIENTS="${EXCLUDE_PATIENTS:-95,129,145}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF || true

fail() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -s "$1" ]] || fail "required file missing or empty: $1"; }
log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

manifest="${REFERENCE_ROOT}/manifests/model_configs.tsv"
require_file "${manifest}"
mkdir -p "${REFERENCE_ROOT}/logs"

base_config="$(awk -F '\t' 'NR>1 && $2=="centered" {print $7; exit}' "${manifest}")"
require_file "${base_config}"

log "Stage 1: preprocess actual autosegs once using deployment and oracle reference grids"
input_root="${REFERENCE_ROOT}/actual_autoseg_inputs"
prep_manifest="${input_root}/manifests/SpinalCord_preprocess_manifest.csv"
if [[ "${FORCE_EVAL}" == "1" || ! -s "${prep_manifest}" ]]; then
  rm -rf "${input_root}"
  python -m evaluation.actual_autoseg.preprocess_actual_autoseg \
    --config "${base_config}" --roi SpinalCord --group spine \
    --raw-data-root "${RAW_DATA_ROOT}" --output-root "${input_root}" \
    --mode both --center-method "${AUTOSEG_CENTER_METHOD}" \
    --splits training1 validation1 testing1
fi
require_file "${prep_manifest}"

log "Stage 2: actual-autoseg and clean-identity evaluation for all four arms"
while IFS=$'\t' read -r arm_index arm model_key model_label use_context use_jitter config checkpoint; do
  [[ "${arm_index}" == "arm_index" ]] && continue
  require_file "${config}"
  require_file "${checkpoint}"
  model_root="${REFERENCE_ROOT}/actual_autoseg_results/${arm}"
  clean_root="${REFERENCE_ROOT}/clean_reconstruction/${arm}"
  for mode in autoseg_centered approved_reference; do
    csv="${model_root}/metrics/SpinalCord/${mode}_metrics.csv"
    if [[ "${FORCE_EVAL}" == "1" || ! -s "${csv}" ]]; then
      python -m evaluation.actual_autoseg.eval_actual_autoseg_nsrqa \
        --config "${config}" --checkpoint "${checkpoint}" \
        --input-root "${input_root}" --case-output-root "${model_root}" \
        --roi SpinalCord --mode "${mode}" --output-csv "${csv}" \
        --model-name "${model_label}" --corruption-label "spinalcord_proposed_v4"
    fi
    require_file "${csv}"

    clean_csv="${clean_root}/metrics/SpinalCord/${mode}_clean.csv"
    if [[ "${FORCE_EVAL}" == "1" || ! -s "${clean_csv}" ]]; then
      python -m evaluation.actual_autoseg.eval_clean_approved_reconstruction \
        --config "${config}" --checkpoint "${checkpoint}" \
        --input-root "${input_root}" --roi SpinalCord --mode "${mode}" \
        --output-csv "${clean_csv}" --case-output-root "${clean_root}" \
        --model-name "${model_label}" --corruption-label "spinalcord_proposed_v4"
    fi
    require_file "${clean_csv}"
  done
done < "${manifest}"

log "Stage 3: exact V4-family synthetic correction diagnostic under deployment-matched recentering"
family_out="${REFERENCE_ROOT}/diagnostics/v4_family_correction"
if [[ "${FORCE_EVAL}" == "1" || ! -s "${family_out}/family_summary.csv" ]]; then
  rm -rf "${family_out}"
  python -m analysis.eval_spinalcord_v4_family_diagnostics \
    --compare-root "${REFERENCE_ROOT}" --output-dir "${family_out}" \
    --splits validation1 testing1 --surface-repeats 2 --composite-repeats 2 \
    --batch-size 4 --seed "${SEED}"
fi
require_file "${family_out}/family_summary.csv"

log "Stage 4: aggregate P75 QA, correction, clean identity, paired bootstrap, and sentinel cases"
report_dir="${REFERENCE_ROOT}/reports/reference_conditioning_ablation"
python -m analysis.analyze_spinalcord_reference_conditioning_ablation \
  --root "${REFERENCE_ROOT}" --output-dir "${report_dir}" \
  --exclude-patients "${EXCLUDE_PATIENTS}" --n-boot "${N_BOOT}" --seed "${SEED}"
require_file "${report_dir}/report.md"

echo "Primary report: ${report_dir}/report.md"
echo "Family diagnostic: ${family_out}/report.md"
echo "Viewer roots: ${REFERENCE_ROOT}/actual_autoseg_results/<arm>"
