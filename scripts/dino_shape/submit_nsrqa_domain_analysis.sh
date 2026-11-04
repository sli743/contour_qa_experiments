#!/bin/bash
# Submit NSR-QA domain analysis with the project's required scheduler settings.
#
# Usage:
#   scripts/dino_shape/submit_nsrqa_domain_analysis.sh \
#     /scratch/.../ranking_classification \
#     ../radformation-data/OOD_Domain_Assignment_MAD_TOTAL.csv \
#     /scratch/.../nsrqa_domain_analysis \
#     [extra analyze_nsrqa_domain_performance.py args]
#
# For whole-data analysis, prefer submit_nsrqa_domain_analysis_whole_data.sh.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <ranking_root> <domain_csv> <output_dir> [extra analysis args...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="${SCRIPT_DIR}/run_nsrqa_domain_analysis.slurm"

RANKING_ROOT="$1"
DOMAIN_CSV="$2"
OUTPUT_DIR="$3"
shift 3
ANALYSIS_ARGS=("$@")

# Whole-data analysis reads case rows directly from actual_autoseg_regression
# metrics. Ranking/classification outputs are optional there and are consulted
# only to recover an already-frozen development threshold when available. OOF
# analysis still requires classification_cases.csv.
ANALYSIS_SCOPE="oof"
for ((i=0; i<${#ANALYSIS_ARGS[@]}; i++)); do
    arg="${ANALYSIS_ARGS[$i]}"
    if [[ "${arg}" == "--analysis_scope" && $((i + 1)) -lt ${#ANALYSIS_ARGS[@]} ]]; then
        ANALYSIS_SCOPE="${ANALYSIS_ARGS[$((i + 1))]}"
    elif [[ "${arg}" == --analysis_scope=* ]]; then
        ANALYSIS_SCOPE="${arg#--analysis_scope=}"
    fi
done

# Preflight ranking outputs on the login node so a bad path fails before a Slurm
# allocation is consumed.  If the conventional ranking_classification path is
# absent, search only the immediate study parent.  The Python analysis repeats
# this resolution and performs duplicate/conflict checks before combining files.
count_case_files() {
    local root="$1"
    [[ -d "${root}" ]] || { echo 0; return; }
    find "${root}" -type f -name classification_cases.csv -print -quit | grep -q . && echo 1 || echo 0
}

if [[ "${ANALYSIS_SCOPE}" != "whole" && "$(count_case_files "${RANKING_ROOT}")" -eq 0 ]]; then
    PARENT_ROOT="$(dirname "${RANKING_ROOT}")"
    if [[ "$(count_case_files "${PARENT_ROOT}")" -eq 1 ]]; then
        echo "No classification_cases.csv under requested ranking root:" >&2
        echo "  ${RANKING_ROOT}" >&2
        echo "Using immediate study parent instead:" >&2
        echo "  ${PARENT_ROOT}" >&2
        RANKING_ROOT="${PARENT_ROOT}"
    else
        echo "ERROR: no classification_cases.csv found under:" >&2
        echo "  ${RANKING_ROOT}" >&2
        echo "or its immediate parent:" >&2
        echo "  ${PARENT_ROOT}" >&2
        echo "Locate the real outputs with:" >&2
        echo "  find '${PARENT_ROOT}' -type f -name classification_cases.csv -print | head -100" >&2
        exit 2
    fi
elif [[ "${ANALYSIS_SCOPE}" == "whole" && "$(count_case_files "${RANKING_ROOT}")" -eq 0 ]]; then
    echo "Whole-data mode: no ranking/classification outputs found under ${RANKING_ROOT}." >&2
    echo "Proceeding from actual-autoseg metrics; the analyzer will use its documented" >&2
    echo "Normal training+validation threshold fallback if no frozen ranking threshold is available." >&2
fi

# Keep partition/account explicit on submission as well as in the job file.
sbatch -p hugo --account=geoffrey_hugo "${JOB}" \
    "${RANKING_ROOT}" "${DOMAIN_CSV}" "${OUTPUT_DIR}" "${ANALYSIS_ARGS[@]}"
