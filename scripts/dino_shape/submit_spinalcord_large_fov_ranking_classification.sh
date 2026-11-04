#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
CLINICAL_METRICS_CSV="${CLINICAL_METRICS_CSV:-${PROJECT_ROOT}/../radformation-data/data/metrics.csv}"
CLINICAL_METRICS_DUPLICATE_POLICY="${CLINICAL_METRICS_DUPLICATE_POLICY:-exclude}"
CLINICAL_SITE="${CLINICAL_SITE:-thorax}"
N_BOOT="${N_BOOT:-5000}"
SEED="${SEED:-88}"

MANIFEST="${STUDY_ROOT}/manifests/active_rois.tsv"
[[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 2; }
[[ -s "${CLINICAL_METRICS_CSV}" ]] || { echo "Missing ${CLINICAL_METRICS_CSV}" >&2; exit 2; }

mapfile -t roi_rows < <(awk -F '\t' 'NR>1 && NF {gsub(/\r/, ""); print $1 "\t" $3}' "${MANIFEST}")
[[ ${#roi_rows[@]} -eq 1 ]] || { echo "Expected exactly one active ROI for the SpinalCord large-FOV study; found ${#roi_rows[@]}" >&2; exit 2; }
IFS=$'\t' read -r ROI_INDEX ROI <<< "${roi_rows[0]}"
[[ "${ROI}" == "SpinalCord" ]] || { echo "Expected SpinalCord in ${MANIFEST}, found ${ROI}" >&2; exit 2; }

for mode in autoseg_centered approved_reference; do
  csv="${REG_ROOT}/metrics/SpinalCord/${mode}_metrics.csv"
  [[ -s "${csv}" ]] || { echo "Missing ${csv}; run the large-FOV actual-autoseg evaluation first" >&2; exit 3; }
done

mkdir -p "${PROJECT_ROOT}/jobs/logs"

echo "Submitting SpinalCord large-FOV percentile ranking/classification"
echo "  QA metrics root: ${REG_ROOT}/metrics/SpinalCord"
echo "  Full-volume clinical truth: ${CLINICAL_METRICS_CSV}"
echo "  Clinical duplicate policy: ${CLINICAL_METRICS_DUPLICATE_POLICY}"
echo "  P75 primary / P50 secondary; cutoffs learned from training1+validation1 and frozen on testing1"

cd "${PROJECT_ROOT}"
sbatch \
  -p hugo \
  --account=geoffrey_hugo \
  --array="${ROI_INDEX}" \
  --job-name=spcord_rankcls \
  --mail-type=ALL \
  --mail-user=shaohan@wustl.edu \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},REG_ROOT=${REG_ROOT},CLINICAL_METRICS_CSV=${CLINICAL_METRICS_CSV},CLINICAL_METRICS_DUPLICATE_POLICY=${CLINICAL_METRICS_DUPLICATE_POLICY},CLINICAL_SITE=${CLINICAL_SITE},N_BOOT=${N_BOOT},SEED=${SEED}" \
  jobs/all_roi_actual_autoseg_ranking_classification_array.job

echo "Output after completion:"
echo "  ${REG_ROOT}/ranking_classification/SpinalCord/classification_summary.csv"
echo "  ${REG_ROOT}/ranking_classification/SpinalCord/classification_cases.csv"
echo "  ${REG_ROOT}/ranking_classification/SpinalCord/clinical_truth_join_audit.csv"
