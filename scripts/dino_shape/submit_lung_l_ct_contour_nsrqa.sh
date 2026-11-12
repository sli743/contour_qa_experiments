#!/usr/bin/env bash
set -euo pipefail

# CT + contour NSR-QA on the Lung_L Month1+2 -> Month3 split.
#
# Arms (manifest task index):
#   0 ct_contour_corruption     CT + contour, v3 family-validated corruption (main)
#   1 ct_contour_no_corruption  CT + contour, identity training
#   2 ct_only_segmenter         contour input empty -> plain CT segmenter
# Contour-only v3 and contour-only no-corruption are linked, not retrained.
#
# Jobs: prepare (inline) -> train array -> Month3 eval array
#       + CT-absent reliance eval (arms 0,1) -> floor-corrected scoring.
#
# SMOKE=1: 2-epoch run of arm 0 only into a separate study root, with eval, to
# check the pipeline on the cluster before the full submission.

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_ct_contour_nsrqa_smoke}"
else
  STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_ct_contour_nsrqa_v1}"
fi
V3_STUDY="${V3_STUDY:-/scratch/shaohan/contour_QA/lung_l_nsrqa_month_temporal_family_validated_v1}"
NOCORR_STUDY="${NOCORR_STUDY:-/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_month_temporal_v1}"
SEED="${SEED:-91}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${STUDY_ROOT}/logs"

PREP_ARGS=(
  --source-config "${V3_STUDY}/configs/nsrqa_corruption_seed${SEED}.yaml"
  --source-actual-autoseg "${V3_STUDY}/actual_autoseg_preprocessed"
  --study-root "${STUDY_ROOT}"
  --seed "${SEED}"
)
if [[ "${SMOKE}" == "1" ]]; then
  PREP_ARGS+=(--arms ct_contour_corruption --epochs 2)
  TRAIN_ARRAY="0"; RELIANCE_ARRAY="0"
else
  TRAIN_ARRAY="0-2"; RELIANCE_ARRAY="0-1"
fi
python -m analysis.prepare_lung_l_ct_contour_nsrqa_study "${PREP_ARGS[@]}"

EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},V3_STUDY=${V3_STUDY},NOCORR_STUDY=${NOCORR_STUDY},SEED=${SEED}"
jobid() { printf '%s' "$1" | cut -d';' -f1; }
LOGS=(--output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err")

TRAIN="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name=lungL_ctc_train \
  --export="${EXPORTS}" "${LOGS[@]}" --array="${TRAIN_ARRAY}" \
  jobs/lung_l_nsrqa_month_temporal_ablation_train.job)")"
echo "Submitted training (${TRAIN_ARRAY}): ${TRAIN}"

EVAL="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name=lungL_ctc_eval \
  --dependency="aftercorr:${TRAIN}" \
  --export="${EXPORTS},IMAGE_MODE=present" "${LOGS[@]}" --array="${TRAIN_ARRAY}" \
  jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"
echo "Submitted Month3 evaluation: ${EVAL}"

RELY="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name=lungL_ctc_ctabsent \
  --dependency="aftercorr:${TRAIN}" \
  --export="${EXPORTS},IMAGE_MODE=absent" "${LOGS[@]}" --array="${RELIANCE_ARRAY}" \
  jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"
echo "Submitted CT-absent reliance evaluation: ${RELY}"

if [[ "${SMOKE}" == "1" ]]; then
  echo "Smoke run: no scoring job. Check ${STUDY_ROOT}/logs and"
  echo "  ${STUDY_ROOT}/models/ct_contour_corruption/*/logs/image_guidance.csv"
  exit 0
fi

POST="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${EVAL}:${RELY}" \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/lung_l_ct_contour_nsrqa_postprocess.job)")"
echo "Submitted floor-corrected scoring: ${POST}"

cat <<EOF2

CT + contour NSR-QA submitted
  train:     ${TRAIN}
  eval:      ${EVAL}
  ct-absent: ${RELY}
  scoring:   ${POST}
Study root: ${STUDY_ROOT}
EOF2
