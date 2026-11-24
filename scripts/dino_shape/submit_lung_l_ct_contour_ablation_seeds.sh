#!/usr/bin/env bash
set -euo pipefail

# Training seeds for the corruption x CT 2x2 ablation (ROI=Lung_L or Lung_R).
#
# Lung_L: extra seeds (default SEEDS="92 93"); seed 91 ran earlier.
# Lung_R: all seeds (default SEEDS="91 92 93"). V3_STUDY must already be prepared
#   (experiment_metadata.json + Month3 crops): scripts/dino_shape/submit_lung_r_ct_contour_stage1.sh.
#
# Per seed S, all five models are trained:
#   contour-only study (V3_STUDY, family-validated corruption config):
#     0 nsrqa_corruption   -> C1I0
#     1 no_corruption      -> C0I0
#   CT + contour study (STUDY_ROOT):
#     0 ct_contour_corruption     -> C1I1
#     1 ct_contour_no_corruption  -> C0I1
#     2 ct_only_segmenter         -> reference
#   + CT-absent re-evaluation of the two CT + contour arms.
# Then per-seed scoring + ablation (jobs/lung_l_ct_contour_nsrqa_postprocess.job).
#
# Existing seeds are untouched: configs are prepared inline with --per-seed-files, so each
# seed gets its own experiment_manifest_seed<S>.csv and the Month3 inputs that
# already exist are reused (no re-preprocessing).
#
# After all seeds finish, pool them with analysis.ablation_corruption_ct_seeds
# (printed at the end).

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ROI="${ROI:-Lung_L}"
ROI_LC="${ROI,,}"
if [[ "${ROI}" == Lung_? ]]; then JOB_TAG="${JOB_TAG:-lung${ROI:5:1}}"; else JOB_TAG="${JOB_TAG:-${ROI_LC:0:6}}"; fi
# DESIGN=full trains all five models (the 2x2 + ct_only). DESIGN=core (the generic
# pipeline, submit_nsrqa_roi_pipeline.sh) trains only C1I1 (ct_contour_corruption)
# and C0I0 (no_corruption), plus ct_only with WITH_CT_ONLY=1.
DESIGN="${DESIGN:-full}"
WITH_CT_ONLY="${WITH_CT_ONLY:-0}"
# REGIONS=1 also submits pooled region detection after the last post job;
# REGION_ARGS / EXCLUDE / DEPTHS / MIN_COMPONENT_ML pass the ROI's truth settings.
REGIONS="${REGIONS:-0}"
REGION_ARGS="${REGION_ARGS:-}"
EXCLUDE="${EXCLUDE:-}"
DEPTHS="${DEPTHS:-3,4.5,5.25,6,7.5}"
MIN_COMPONENT_ML="${MIN_COMPONENT_ML:-0.05}"
if [[ "${DESIGN}" == "full" ]]; then
  C_IDX="0-1"; T_IDX="0-2"; A_IDX="0-1"
else
  C_IDX="1"; A_IDX="0"
  if [[ "${WITH_CT_ONLY}" == "1" ]]; then T_IDX="0,2"; else T_IDX="0"; fi
fi
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/${ROI_LC}_ct_contour_nsrqa_v1}"
V3_STUDY="${V3_STUDY:-/scratch/shaohan/contour_QA/${ROI_LC}_nsrqa_month_temporal_family_validated_v1}"
# Lung_L seed 91 was run before this script existed; a new ROI runs all seeds here.
if [[ "${ROI}" == "Lung_L" ]]; then SEEDS="${SEEDS:-92 93}"; else SEEDS="${SEEDS:-91 92 93}"; fi
# Optional cap on concurrent GPU tasks per array, e.g. ARRAY_LIMIT=2.
ARRAY_LIMIT="${ARRAY_LIMIT:-}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${STUDY_ROOT}/logs" "${V3_STUDY}/logs"

# Reuse exactly the inputs seed 91's contour-only prepare used.
META="${V3_STUDY}/experiment_metadata.json"
[[ -s "${META}" ]] || { echo "Missing ${META}" >&2; exit 2; }
readarray -t SRC < <(python - "${META}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("source_calibrated_config", "temporal_reference_config", "temporal_split_summary"):
    print(m[k])
PY
)
CALIBRATED_CONFIG="${SRC[0]}"; TEMPORAL_REF="${SRC[1]}"; TEMPORAL_SUMMARY="${SRC[2]}"
for f in "${CALIBRATED_CONFIG}" "${TEMPORAL_REF}" "${TEMPORAL_SUMMARY}" "${V3_STUDY}/actual_autoseg_preprocessed/autoseg_centered"; do
  [[ -e "${f}" ]] || { echo "Missing seed-91 input: ${f}" >&2; exit 2; }
done
echo "Corruption config (as seed 91): ${CALIBRATED_CONFIG}"

jobid() { printf '%s' "$1" | cut -d';' -f1; }
arr() { [[ -n "${ARRAY_LIMIT}" ]] && printf '%s%%%s' "$1" "${ARRAY_LIMIT}" || printf '%s' "$1"; }
SUMMARY=()
POSTS=()
# Values with commas (DEPTHS) cannot go through sbatch --export=<list>; every job
# here uses --export=ALL, so these reach the post / region jobs from the environment.
export DESIGN DEPTHS MIN_COMPONENT_ML EXCLUDE

for SEED in ${SEEDS}; do
  echo
  echo "=== seed ${SEED} ==="
  for d in "${V3_STUDY}/models/nsrqa_corruption/${ROI_LC}_nsrqa_corruption_seed${SEED}" \
           "${STUDY_ROOT}/models/ct_contour_corruption/${ROI_LC}_ct_contour_corruption_seed${SEED}"; do
    [[ -e "${d}" ]] && { echo "Seed ${SEED} already has models (${d}); remove them or pick another seed." >&2; exit 2; }
  done

  # Contour-only configs (both arms) for this seed.
  python -m analysis.prepare_lung_l_nsrqa_month_temporal_ablation \
    --calibrated-config "${CALIBRATED_CONFIG}" \
    --temporal-reference-config "${TEMPORAL_REF}" \
    --temporal-split-summary "${TEMPORAL_SUMMARY}" \
    --study-root "${V3_STUDY}" \
    --seed "${SEED}" --roi "${ROI}" --per-seed-files
  # CT + contour configs, derived from this seed's contour-only corruption config.
  python -m analysis.prepare_lung_l_ct_contour_nsrqa_study \
    --source-config "${V3_STUDY}/configs/nsrqa_corruption_seed${SEED}.yaml" \
    --source-actual-autoseg "${V3_STUDY}/actual_autoseg_preprocessed" \
    --study-root "${STUDY_ROOT}" \
    --seed "${SEED}" --roi "${ROI}" --per-seed-files

  V3_EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${V3_STUDY},ROI=${ROI},MANIFEST=${V3_STUDY}/experiment_manifest_seed${SEED}.csv"
  CT_EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},ROI=${ROI},MANIFEST=${STUDY_ROOT}/experiment_manifest_seed${SEED}.csv"
  V3_LOGS=(--output="${V3_STUDY}/logs/slurm-%x-%A_%a.out" --error="${V3_STUDY}/logs/slurm-%x-%A_%a.err")
  CT_LOGS=(--output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err")

  C_TRAIN="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_c_train" \
    --export="${V3_EXPORTS}" "${V3_LOGS[@]}" --array="$(arr "${C_IDX}")" \
    jobs/lung_l_nsrqa_month_temporal_ablation_train.job)")"
  C_EVAL="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_c_eval" \
    --dependency="aftercorr:${C_TRAIN}" \
    --export="${V3_EXPORTS}" "${V3_LOGS[@]}" --array="$(arr "${C_IDX}")" \
    jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"

  T_TRAIN="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_ctc_train" \
    --export="${CT_EXPORTS}" "${CT_LOGS[@]}" --array="$(arr "${T_IDX}")" \
    jobs/lung_l_nsrqa_month_temporal_ablation_train.job)")"
  T_EVAL="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_ctc_eval" \
    --dependency="aftercorr:${T_TRAIN}" \
    --export="${CT_EXPORTS},IMAGE_MODE=present" "${CT_LOGS[@]}" --array="$(arr "${T_IDX}")" \
    jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"
  T_ABS="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_ctc_ctabsent" \
    --dependency="aftercorr:${T_TRAIN}" \
    --export="${CT_EXPORTS},IMAGE_MODE=absent" "${CT_LOGS[@]}" --array="$(arr "${A_IDX}")" \
    jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"

  POST="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_post" \
    --dependency="afterok:${C_EVAL}:${T_EVAL}:${T_ABS}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},V3_STUDY=${V3_STUDY},NOCORR_STUDY=${V3_STUDY},SEED=${SEED},ROI=${ROI}" \
    --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
    jobs/lung_l_ct_contour_nsrqa_postprocess.job)")"

  SUMMARY+=("seed ${SEED}: contour train ${C_TRAIN} eval ${C_EVAL} | CT train ${T_TRAIN} eval ${T_EVAL} ct-absent ${T_ABS} | post ${POST}")
  POSTS+=("${POST}")
done

echo
echo "Submitted:"
printf '  %s\n' "${SUMMARY[@]}"

if [[ "${REGIONS}" == "1" ]]; then
  if [[ "${DESIGN}" == "full" ]]; then R_ARMS="C0I0 C1I0 C0I1 C1I1 ct_only"
  else R_ARMS="C0I0 C1I1"; [[ "${WITH_CT_ONLY}" == "1" ]] && R_ARMS+=" ct_only"; fi
  export SEEDS EXTRA_ARGS="--arms ${R_ARMS} ${REGION_ARGS}"
  REG="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_regions" \
    --dependency="afterok$(printf ':%s' "${POSTS[@]}")" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ROI=${ROI},STUDY_ROOT=${STUDY_ROOT}" \
    --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
    jobs/nsrqa_region_detection.job)")"
  echo "  region detection ${REG} -> ${STUDY_ROOT}/evaluations_floor_corrected/region_detection"
fi
[[ "${DESIGN}" == "full" ]] || exit 0
ALL_SEEDS="$(printf '%s\n' 91 ${SEEDS} | sort -un | tr '\n' ' ')"
DIRS=""
for SEED in ${ALL_SEEDS}; do
  DIRS+=" --ablation-dir seed${SEED}=${STUDY_ROOT}/evaluations_floor_corrected/seed${SEED}_ablation"
done
cat <<EOF2

When every post job is done, pool the seeds:
  python -m analysis.ablation_corruption_ct_seeds ${DIRS} \\
    --output-dir ${STUDY_ROOT}/evaluations_floor_corrected/ablation_seeds
  python -m analysis.plot_ablation_corruption_ct \\
    --ablation-dir ${STUDY_ROOT}/evaluations_floor_corrected/ablation_seeds
EOF2
