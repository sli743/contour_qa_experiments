#!/bin/bash
# Submit the exploratory whole-data NSR-QA domain analysis.
#
# This deliberately includes training + validation + testing cases from ONE fold.
# It is a leakage-exposed sensitivity analysis and must be interpreted alongside
# the held-out/OOF report, not as a replacement for it.
#
# Usage:
#   scripts/dino_shape/submit_nsrqa_domain_analysis_whole_data.sh \
#     /scratch/shaohan/contour_QA/all_roi_NSRQA_study \
#     ../radformation-data/OOD_Domain_Assignment_MAD_TOTAL.csv \
#     /scratch/shaohan/contour_QA/all_roi_NSRQA_study/nsrqa_domain_analysis_whole \
#     [fold_number] [extra analysis args...]

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <study_root> <domain_csv> <output_dir> [fold_number] [extra analysis args...]" >&2
    exit 2
fi

STUDY_ROOT="$1"
DOMAIN_CSV="$2"
OUTPUT_DIR="$3"
shift 3

WHOLE_FOLD="1"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    WHOLE_FOLD="$1"
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SUBMIT="${SCRIPT_DIR}/submit_nsrqa_domain_analysis.sh"
RANKING_ROOT="${STUDY_ROOT}/ranking_classification"
METRICS_ROOT="${STUDY_ROOT}/actual_autoseg_regression/metrics"

if [[ ! -d "${METRICS_ROOT}" ]]; then
    echo "ERROR: expected whole-data metrics root does not exist:" >&2
    echo "  ${METRICS_ROOT}" >&2
    echo "Locate metrics with:" >&2
    echo "  find '${STUDY_ROOT}' -type f -name autoseg_centered_metrics.csv -print | head -100" >&2
    exit 2
fi

# The delegated submitter calls sbatch with explicit
#   -p hugo --account=geoffrey_hugo
# and the job file independently contains the same partition/account plus the
# standard mail directives.
"${BASE_SUBMIT}" \
    "${RANKING_ROOT}" \
    "${DOMAIN_CSV}" \
    "${OUTPUT_DIR}" \
    --analysis_scope whole \
    --metrics_root "${METRICS_ROOT}" \
    --whole_fold "${WHOLE_FOLD}" \
    "$@"
