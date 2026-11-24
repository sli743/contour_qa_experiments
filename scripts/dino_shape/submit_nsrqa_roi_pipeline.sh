#!/usr/bin/env bash
set -euo pipefail

# Generic NSR-QA pipeline for one ROI (frozen design, 2026-11-23): CT + contour with the
# ROI's own fitted corruption (C1I1) against contour-only without corruption (C0I0),
# seeds 91-93, scored by deep-change volume and region-level detection.
# Method = the Lung_L/Lung_R one; no ROI-specific repairs (no leak branch, no mask stem).
#
#   STAGE=prep ROI=Heart bash scripts/dino_shape/submit_nsrqa_roi_pipeline.sh
#     1. writes the ROI's error-field and fit configs (analysis.make_roi_pipeline_configs)
#     2. crop clipping audit (CPU)                       -> <roi>_fov_clipping_audit_v1
#     3. error-field characterization, training1 only    -> <roi>_error_field_characterization_v1
#     4. realized-severity fit, Lung_L v3 method          -> <roi>_nsrqa_synthetic_severity_fit_v3method
#     5. contour-only study prepare (configs, Month3 crops) -> <roi>_nsrqa_month_temporal_family_validated_v1
#   CHECK before stage 2: the audit report (how much of the ROI the crop cuts off),
#   the fit report and its example bank (do the corruptions look plausible?).
#   The fit enables the end-truncation guard (draws that cut an S/I end segment off are
#   redrawn; limit = the ROI's real p95 endpoint error). Existing audit / error fields
#   are reused. REFIT=1 redoes fit + prepare of an already prepared ROI (before any
#   training), keeping the previous fit and contour study as *_prev_<date>.
#
#   STAGE=train ROI=Heart bash scripts/dino_shape/submit_nsrqa_roi_pipeline.sh
#     C1I1 + C0I0 (+ ct_only with WITH_CT_ONLY=1) x seeds, Month3 / approved / CT-absent
#     evals, per-seed scoring, then pooled region detection -> <roi>_ct_contour_nsrqa_v1
#
# Options: SEEDS ("91 92 93"), WITH_CT_ONLY (0), EXCLUDE (patient ids, after inspection),
# ARRAY_LIMIT, DESIGN=full for the whole 2x2 instead of the core pair.

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STAGE="${STAGE:?set STAGE=prep or STAGE=train}"
ROI="${ROI:?set ROI, e.g. Heart}"
ROI_LC="${ROI,,}"
SCRATCH="${SCRATCH:-/scratch/shaohan/contour_QA}"
TEMPORAL_STUDY="${TEMPORAL_STUDY:-${SCRATCH}/all_roi_NSRQA_month_temporal_study}"
V3_STUDY="${V3_STUDY:-${SCRATCH}/${ROI_LC}_nsrqa_month_temporal_family_validated_v1}"
CT_STUDY="${CT_STUDY:-${SCRATCH}/${ROI_LC}_ct_contour_nsrqa_v1}"
AUDIT_DIR="${AUDIT_DIR:-${SCRATCH}/${ROI_LC}_fov_clipping_audit_v1}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
TAG="${ROI_LC:0:6}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
jobid() { printf '%s' "$1" | cut -d';' -f1; }

# GROUP PROFILE START_CONFIG EF_CONFIG EF_DIR FIT_CONFIG FIT_DIR TRUTH_T MIN_REGION_ML POSITIVE_ML MIN_COMPONENT_ML DEPTHS
CFG_VARS="$(python -m analysis.make_roi_pipeline_configs --roi "${ROI}" --scratch "${SCRATCH}" --temporal-study "${TEMPORAL_STUDY}")"
eval "${CFG_VARS}"
echo "${ROI}: group ${GROUP}, profile ${PROFILE}; truth depth ${TRUTH_T} mm, region core >= ${MIN_REGION_ML} mL, positive >= ${POSITIVE_ML} mL; score depths ${DEPTHS}"

if [[ "${STAGE}" == "prep" ]]; then
  if [[ -s "${V3_STUDY}/experiment_metadata.json" ]]; then
    [[ "${REFIT:-0}" == "1" ]] || { echo "ERROR: ${V3_STUDY} is already prepared (REFIT=1 redoes the fit + prepare)." >&2; exit 2; }
    [[ -e "${CT_STUDY}/models" ]] && { echo "ERROR: ${CT_STUDY} already has models; refitting would mismatch them." >&2; exit 2; }
    # Keep the previous fit and contour study for comparison, then regenerate the configs.
    STAMP="prev_$(date +%Y%m%d_%H%M)"
    mv "${FIT_DIR}" "${FIT_DIR}_${STAMP}"
    mv "${V3_STUDY}" "${V3_STUDY}_${STAMP}"
    echo "Moved the previous fit and contour study to *_${STAMP}"
    CFG_VARS="$(python -m analysis.make_roi_pipeline_configs --roi "${ROI}" --scratch "${SCRATCH}" --temporal-study "${TEMPORAL_STUDY}")"
    eval "${CFG_VARS}"
  fi
  mkdir -p "${AUDIT_DIR}" "${V3_STUDY}/logs"

  # Crop audit and error fields do not depend on the fit: reuse them if present.
  if [[ -s "${AUDIT_DIR}/${ROI}_fov_clipping_audit.md" ]]; then
    AUDIT="reused"
  else
    AUDIT="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${TAG}_fov_audit" \
      --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ROI=${ROI},GROUP=${GROUP},CONFIG=${START_CONFIG},OUTPUT_DIR=${AUDIT_DIR},RAW_DATA_ROOT=${RAW_DATA_ROOT}" \
      --output="${AUDIT_DIR}/slurm-%x-%j.out" --error="${AUDIT_DIR}/slurm-%x-%j.err" \
      jobs/lung_l_fov_clipping_audit.job)")"
  fi
  FIT_DEP=()
  if [[ -s "${EF_DIR}/case_error_field_summary.csv" && -s "${EF_DIR}/error_components.csv" ]]; then
    EF="reused"
  else
    EF="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${TAG}_errfield" \
      --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONFIG=${EF_CONFIG}" \
      --output="${EF_DIR}/slurm-%x-%j.out" --error="${EF_DIR}/slurm-%x-%j.err" \
      jobs/lung_l_error_field_characterization.job)")"
    FIT_DEP=(--dependency="afterok:${EF}")
  fi

  # The fit exits 0 even when the dropout no-op audit fails and always writes the best candidate.
  FIT="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${TAG}_sevfit" \
    ${FIT_DEP[@]+"${FIT_DEP[@]}"} \
    --export="ALL,REPO=${PROJECT_ROOT},CONFIG=${FIT_CONFIG}" \
    --output="${FIT_DIR}/slurm-%x-%j.out" --error="${FIT_DIR}/slurm-%x-%j.err" \
    jobs/lung_l_nsrqa_synthetic_severity_fit_family_validated.job)")"

  PREP="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${TAG}_prep" \
    --dependency="afterok:${FIT}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ROI=${ROI},GROUP=${GROUP},STUDY_ROOT=${V3_STUDY},SEED=91,CALIBRATED_CONFIG=${FIT_DIR}/best_candidate_training_config.yaml,TEMPORAL_STUDY_ROOT=${TEMPORAL_STUDY},RAW_DATA_ROOT=${RAW_DATA_ROOT}" \
    --output="${V3_STUDY}/logs/slurm-%x-%j.out" --error="${V3_STUDY}/logs/slurm-%x-%j.err" \
    jobs/lung_l_nsrqa_month_temporal_ablation_prepare.job)")"

  cat <<EOF
Submitted ${ROI} stage prep
  crop audit       ${AUDIT} -> ${AUDIT_DIR}/${ROI}_fov_clipping_audit.md
  error fields     ${EF} -> ${EF_DIR}
  severity fit     ${FIT} -> ${FIT_DIR} (report + example bank)
  contour prepare  ${PREP} -> ${V3_STUDY} (month3_actual_autoseg_preprocess_audit.json)
Check the crop audit and the fit before training, then:
  STAGE=train ROI=${ROI} bash scripts/dino_shape/submit_nsrqa_roi_pipeline.sh
EOF
  exit 0
fi

[[ "${STAGE}" == "train" ]] || { echo "STAGE must be prep or train" >&2; exit 2; }
[[ -s "${V3_STUDY}/experiment_metadata.json" ]] || { echo "Run STAGE=prep first (${V3_STUDY} not prepared)" >&2; exit 2; }
python - "${V3_STUDY}/experiment_metadata.json" "${ROI}" <<'PY'
import json, sys, yaml
m = json.load(open(sys.argv[1]))
roi = (yaml.safe_load(open(m["source_calibrated_config"])).get("data") or {}).get("roi_name")
if roi not in (None, sys.argv[2]):
    sys.exit(f"ERROR: corruption config {m['source_calibrated_config']} is for {roi}, not {sys.argv[2]}")
PY

ROI="${ROI}" V3_STUDY="${V3_STUDY}" STUDY_ROOT="${CT_STUDY}" SEEDS="${SEEDS:-91 92 93}" \
DESIGN="${DESIGN:-core}" WITH_CT_ONLY="${WITH_CT_ONLY:-0}" REGIONS=1 EXCLUDE="${EXCLUDE:-}" \
DEPTHS="${DEPTHS}" MIN_COMPONENT_ML="${MIN_COMPONENT_ML}" JOB_TAG="${TAG}" ARRAY_LIMIT="${ARRAY_LIMIT:-}" \
REGION_ARGS="--t ${TRUTH_T} --min-region-ml ${MIN_REGION_ML} --positive-ml ${POSITIVE_ML} --min-component-ml ${MIN_COMPONENT_ML}" \
  bash scripts/dino_shape/submit_lung_l_ct_contour_ablation_seeds.sh
