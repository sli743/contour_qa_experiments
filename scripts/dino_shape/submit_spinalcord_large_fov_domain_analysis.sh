#!/usr/bin/env bash
# Submit reviewed-month domain analyses for the coverage-safe large-FOV
# SpinalCord NSR-QA experiment.
#
# This is a whole-data/domain-mechanism sensitivity analysis over the one
# SpinalCord training1+validation1+testing1 cohort. It intentionally runs both
# localization modes:
#   autoseg_centered  - candidate-only/deployment-safe bbox-midpoint localization
#   approved_reference - oracle localization control
#
# Domain rules are implemented in analysis/analyze_nsrqa_domain_performance.py:
#   month1                         -> Unknown
#   month2/month3 unrecorded pair  -> Normal
#   outside all three months       -> Unknown
#   abnormal mechanism domains     -> I, AV, MI, AC, H
# Unknown cases remain descriptive-only and are excluded from Normal
# calibration and abnormal-domain inferential comparisons.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
DOMAIN_CSV="${DOMAIN_CSV:-${PROJECT_ROOT}/../radformation-data/OOD_Domain_Assignment_MAD_TOTAL.csv}"
CLINICAL_METRICS_CSV="${CLINICAL_METRICS_CSV:-${PROJECT_ROOT}/../radformation-data/data/metrics.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STUDY_ROOT}/nsrqa_domain_analysis_whole_reviewed_months}"
WHOLE_FOLD="${WHOLE_FOLD:-1}"
CLINICAL_METRICS_DUPLICATE_POLICY="${CLINICAL_METRICS_DUPLICATE_POLICY:-exclude}"

METRICS_ROOT="${REG_ROOT}/metrics"
RANKING_ROOT="${STUDY_ROOT}/ranking_classification"
SUBMITTER="${PROJECT_ROOT}/scripts/dino_shape/submit_nsrqa_domain_analysis_whole_data.sh"

[[ -f "${DOMAIN_CSV}" ]] || { echo "Missing domain CSV: ${DOMAIN_CSV}" >&2; exit 2; }
[[ -f "${CLINICAL_METRICS_CSV}" ]] || { echo "Missing clinical metrics CSV: ${CLINICAL_METRICS_CSV}" >&2; exit 2; }
[[ -x "${SUBMITTER}" ]] || { echo "Missing submitter: ${SUBMITTER}" >&2; exit 2; }

for mode in autoseg_centered approved_reference; do
    metrics_csv="${METRICS_ROOT}/SpinalCord/${mode}_metrics.csv"
    if [[ ! -s "${metrics_csv}" ]]; then
        echo "Missing ${mode} SpinalCord metrics:" >&2
        echo "  ${metrics_csv}" >&2
        echo "Run and finish scripts/dino_shape/submit_spinalcord_large_fov_actual_autoseg.sh first." >&2
        exit 3
    fi
done

mkdir -p "${OUTPUT_ROOT}"

for mode in autoseg_centered approved_reference; do
    out="${OUTPUT_ROOT}/${mode}"
    echo "Submitting SpinalCord large-FOV domain analysis: ${mode}"
    echo "  metrics: ${METRICS_ROOT}/SpinalCord/${mode}_metrics.csv"
    echo "  output:  ${out}"
    "${SUBMITTER}" \
        "${STUDY_ROOT}" \
        "${DOMAIN_CSV}" \
        "${out}" \
        "${WHOLE_FOLD}" \
        --localization_mode "${mode}" \
        --clinical_metrics_csv "${CLINICAL_METRICS_CSV}" \
        --clinical_metrics_duplicate_policy "${CLINICAL_METRICS_DUPLICATE_POLICY}" \
        --site thorax \
        --roi_selection manual \
        --rois SpinalCord
done

cat <<MSG
Submitted both SpinalCord localization analyses.
Primary/deployable report:
  ${OUTPUT_ROOT}/autoseg_centered/report.md
Oracle-localization control:
  ${OUTPUT_ROOT}/approved_reference/report.md

Clinical Dice/Hausdorff truth comes from:
  ${CLINICAL_METRICS_CSV}
NSR-QA scores come from:
  ${METRICS_ROOT}/SpinalCord/<localization>_metrics.csv
MSG
