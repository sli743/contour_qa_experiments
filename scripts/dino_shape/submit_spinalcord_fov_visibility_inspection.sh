#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
cd "${PROJECT_ROOT}"

# Scheduler/account are explicit here as well as in the job file so an incorrect
# inherited sbatch configuration cannot silently route this job elsewhere.
sbatch -p hugo --account geoffrey_hugo jobs/spinalcord_fov_visibility_inspection.job
