#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
N_BOOT="${N_BOOT:-5000}"
SEED="${SEED:-88}"
CLINICAL_METRICS_CSV="${CLINICAL_METRICS_CSV:-}"
CLINICAL_METRICS_DUPLICATE_POLICY="${CLINICAL_METRICS_DUPLICATE_POLICY:-exclude}"
CLINICAL_SITE="${CLINICAL_SITE:-thorax}"
ROI_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "${ROI_INDEX}" ]] || { echo "Usage: $0 ROI_INDEX" >&2; exit 2; }
MANIFEST="${STUDY_ROOT}/manifests/active_rois.tsv"
line="$(awk -F '\t' -v idx="${ROI_INDEX}" 'NR>1 && $1==idx {print; exit}' "${MANIFEST}")"
[[ -n "${line}" ]] || { echo "No active ROI index ${ROI_INDEX}" >&2; exit 2; }
line="${line%$'\r'}"
IFS=$'\t' read -r roi_index registry_index ROI rest <<< "${line}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PRIMARY_CSV="${REG_ROOT}/metrics/${ROI}/autoseg_centered_metrics.csv"
ORACLE_CSV="${REG_ROOT}/metrics/${ROI}/approved_reference_metrics.csv"
[[ -s "${PRIMARY_CSV}" ]] || { echo "Missing ${PRIMARY_CSV}; run actual-autoseg NSR-QA evaluation first" >&2; exit 3; }
[[ -s "${ORACLE_CSV}" ]] || { echo "Missing ${ORACLE_CSV}; run actual-autoseg NSR-QA evaluation first" >&2; exit 3; }
extra_args=()
if [[ -n "${CLINICAL_METRICS_CSV}" ]]; then
  [[ -s "${CLINICAL_METRICS_CSV}" ]] || { echo "Missing clinical metrics CSV: ${CLINICAL_METRICS_CSV}" >&2; exit 4; }
  extra_args+=(
    --clinical-metrics-csv "${CLINICAL_METRICS_CSV}"
    --clinical-site "${CLINICAL_SITE}"
    --clinical-metrics-duplicate-policy "${CLINICAL_METRICS_DUPLICATE_POLICY}"
  )
fi
python -m analysis.analyze_actual_autoseg_ranking_classification \
  --roi "${ROI}" \
  --metrics "autoseg_centered=${PRIMARY_CSV}" \
  --metrics "approved_reference=${ORACLE_CSV}" \
  --output-dir "${REG_ROOT}/ranking_classification/${ROI}" \
  --n-boot "${N_BOOT}" \
  --seed "${SEED}" \
  "${extra_args[@]}"
echo "Done ${ROI}: ${REG_ROOT}/ranking_classification/${ROI}"
