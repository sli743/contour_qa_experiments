#!/bin/bash
#SBATCH --job-name=nsrqa_domain
#SBATCH -p hugo
#SBATCH --account=geoffrey_hugo
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shaohan@wustl.edu
#SBATCH --output=nsrqa_domain_%j.out
#SBATCH --error=nsrqa_domain_%j.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch -p hugo --account=geoffrey_hugo $0 <ranking_root> <domain_csv> <output_dir> [extra analysis args...]" >&2
    echo "Whole-data mode: add --analysis_scope whole --metrics_root <.../actual_autoseg_regression/metrics>." >&2
    exit 2
fi

RANKING_ROOT="$1"
DOMAIN_CSV="$2"
OUTPUT_DIR="$3"
shift 3

# Preserve analysis options before sourcing the interactive shell setup.
# Some user .bashrc files reference "$@" while initializing/activating Conda;
# leaving our analysis CLI options in the shell positional parameters can make
# `conda activate` interpret them as environment arguments.
ANALYSIS_ARGS=("$@")
set --

# Match the environment used by the current 3DINO/NSR-QA workflow.
source ~/.bashrc
conda activate 3DINO

cd /home/shaohan/contour_qa_experiments

python analysis/analyze_nsrqa_domain_performance.py \
    --ranking_root "${RANKING_ROOT}" \
    --domain_csv "${DOMAIN_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    "${ANALYSIS_ARGS[@]}"
