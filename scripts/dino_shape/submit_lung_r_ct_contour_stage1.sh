#!/usr/bin/env bash
set -euo pipefail

# Lung_R corruption x CT study, stage 1 of 2: prepare the contour-only study that
# every Lung_R model is built from (experiment_metadata.json, seed-91 configs and
# the Month3 crops with CT). Nothing is trained here.
#
# Corruption: Lung_R's own calibration. By default the existing v4 fit's
# best_candidate_training_config.yaml (as Lung_L used its v3 fit's best
# candidate; both fail only the dropout no-op audit). v4 uses the Lung_L v3 fit
# machinery with two settings changed: deepest dropout targets >= 3 mm and HD99
# in the objective.
# RUN_FIT=1 instead fits Lung_R first with the Lung_L v3 settings unchanged
# (configs/dino_shape/lung_r_nsrqa_synthetic_severity_fit_v3method.yaml, CPU job,
# up to 12 h) and prepares from that.
#
# Stage 2, once the prepare job has finished, trains all five models for seeds
# 91, 92, 93 and runs the per-seed ablation:
#   ROI=Lung_R bash scripts/dino_shape/submit_lung_l_ct_contour_ablation_seeds.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
V3_STUDY="${V3_STUDY:-/scratch/shaohan/contour_QA/lung_r_nsrqa_month_temporal_family_validated_v1}"
RUN_FIT="${RUN_FIT:-0}"
FIT_CONFIG="${FIT_CONFIG:-configs/dino_shape/lung_r_nsrqa_synthetic_severity_fit_v3method.yaml}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"

if [[ -s "${V3_STUDY}/experiment_metadata.json" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "ERROR: ${V3_STUDY} is already prepared. Pick another V3_STUDY or set ALLOW_EXISTING=1." >&2
  exit 2
fi
mkdir -p "${V3_STUDY}/logs"
jobid() { printf '%s' "$1" | cut -d';' -f1; }

DEP=()
if [[ "${RUN_FIT}" == "1" ]]; then
  read -r EF INIT FIT_OUT < <(python - "${FIT_CONFIG}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["error_field_root"], c["initial_training_config"], c["output"]["directory"])
PY
)
  for f in "${EF}/case_error_field_summary.csv" "${EF}/error_components.csv" "${INIT}"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
  done
  mkdir -p "${FIT_OUT}"
  FIT="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo \
    --job-name=lungR_sevfit_v3method \
    --export="ALL,REPO=${PROJECT_ROOT},CONFIG=${FIT_CONFIG}" \
    --output="${FIT_OUT}/slurm-%x-%j.out" --error="${FIT_OUT}/slurm-%x-%j.err" \
    jobs/lung_l_nsrqa_synthetic_severity_fit_family_validated.job)")"
  echo "Submitted Lung_R severity fit (v3 method): ${FIT} -> ${FIT_OUT}"
  # The fit exits 0 even when the no-op audit fails and always writes the best candidate.
  CALIBRATED_CONFIG="${FIT_OUT}/best_candidate_training_config.yaml"
  DEP=(--dependency="afterok:${FIT}")
else
  CALIBRATED_CONFIG="${CALIBRATED_CONFIG:-/scratch/shaohan/contour_QA/lung_r_nsrqa_synthetic_severity_fit_v4/best_candidate_training_config.yaml}"
  [[ -s "${CALIBRATED_CONFIG}" ]] || { echo "ERROR: missing ${CALIBRATED_CONFIG}" >&2; exit 2; }
  python - "${CALIBRATED_CONFIG}" <<'PY'
import sys, yaml
roi = (yaml.safe_load(open(sys.argv[1])).get("data") or {}).get("roi_name")
if roi not in (None, "Lung_R"):
    sys.exit(f"ERROR: {sys.argv[1]} is calibrated for {roi}, not Lung_R")
PY
fi

PREP="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --job-name=lungR_ctc_prep ${DEP[@]+"${DEP[@]}"} \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ROI=Lung_R,STUDY_ROOT=${V3_STUDY},SEED=91,CALIBRATED_CONFIG=${CALIBRATED_CONFIG}" \
  --output="${V3_STUDY}/logs/slurm-%x-%j.out" --error="${V3_STUDY}/logs/slurm-%x-%j.err" \
  jobs/lung_l_nsrqa_month_temporal_ablation_prepare.job)")"

cat <<EOF
Submitted Lung_R stage 1
  corruption config:    ${CALIBRATED_CONFIG}
  contour-only prepare: ${PREP} -> ${V3_STUDY}

When job ${PREP} has finished, start stage 2 (all five models, seeds 91 92 93):
  ROI=Lung_R bash scripts/dino_shape/submit_lung_l_ct_contour_ablation_seeds.sh
EOF
