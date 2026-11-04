#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
cd "${PROJECT_ROOT}"

PREP=$(sbatch --parsable jobs/lung_l_nsrqa_corruption_ablation_prepare.job)
echo "Submitted prepare job: ${PREP}"
TRAIN=$(sbatch --parsable --dependency=afterok:${PREP} --array=0-5%2 jobs/lung_l_nsrqa_corruption_ablation_train.job)
echo "Submitted training array: ${TRAIN}"
EVAL=$(sbatch --parsable --dependency=afterok:${TRAIN} --array=0-5%2 jobs/lung_l_nsrqa_corruption_ablation_eval.job)
echo "Submitted evaluation array: ${EVAL}"
POST=$(sbatch --parsable --dependency=afterok:${EVAL} jobs/lung_l_nsrqa_corruption_ablation_postprocess.job)
echo "Submitted postprocess job: ${POST}"

echo "Pipeline: prepare=${PREP} -> train=${TRAIN} -> eval=${EVAL} -> postprocess=${POST}"
