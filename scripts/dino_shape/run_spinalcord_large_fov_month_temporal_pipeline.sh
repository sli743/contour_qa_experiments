#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
BANK_ROOT="${BANK_ROOT:-${STUDY_ROOT}/shared_probe_banks}"
CLINICAL_METRICS_CSV="${CLINICAL_METRICS_CSV:-${PROJECT_ROOT}/../radformation-data/data/metrics.csv}"
CLINICAL_METRICS_DUPLICATE_POLICY="${CLINICAL_METRICS_DUPLICATE_POLICY:-exclude}"
CLINICAL_SITE="${CLINICAL_SITE:-thorax}"
AUTOSEG_CENTER_METHOD="${AUTOSEG_CENTER_METHOD:-bbox_midpoint}"
SEED="${SEED:-88}"
N_BOOT="${N_BOOT:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PREPARE_STUDY="${PREPARE_STUDY:-0}"
FRESH_DATA="${FRESH_DATA:-1}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
PIPELINE_STATE_DIR="${PIPELINE_STATE_DIR:-${STUDY_ROOT}/pipeline_state/month_temporal_single_job}"

export PROJECT_ROOT STUDY_ROOT REG_ROOT BANK_ROOT
export CLINICAL_METRICS_CSV CLINICAL_METRICS_DUPLICATE_POLICY CLINICAL_SITE
export AUTOSEG_CENTER_METHOD SEED N_BOOT BATCH_SIZE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

require_file() {
  [[ -s "$1" ]] || fail "required file missing or empty: $1"
}

stage_complete() {
  local marker="$1"
  shift
  [[ "${REUSE_COMPLETED}" == "1" && -s "${marker}" ]] || return 1
  local f
  for f in "$@"; do
    [[ -s "${f}" ]] || return 1
  done
  return 0
}

mark_done() {
  local marker="$1"
  mkdir -p "$(dirname "${marker}")"
  {
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID:-none}"
    echo "host=$(hostname)"
  } > "${marker}"
}

on_error() {
  local rc=$?
  echo >&2
  echo "Pipeline failed with exit code ${rc}." >&2
  echo "Completed-stage markers are kept under: ${PIPELINE_STATE_DIR}" >&2
  echo "Resubmit the same job to reuse only stages that have both a success marker and expected outputs." >&2
  exit "${rc}"
}
trap on_error ERR

log "SpinalCord large-FOV month-temporal single-job pipeline"
echo "  project: ${PROJECT_ROOT}"
echo "  study:   ${STUDY_ROOT}"
echo "  job:     ${SLURM_JOB_ID:-interactive}"
echo "  host:    $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# Optional convenience mode. By default preparation is deliberately kept outside
# the GPU job so the temporal split/FOV audits can be inspected before training.
if [[ "${PREPARE_STUDY}" == "1" ]]; then
  log "Stage 0: prepare temporal large-FOV study"
  FRESH_DATA="${FRESH_DATA}" \
    STUDY_ROOT="${STUDY_ROOT}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    bash scripts/dino_shape/prepare_spinalcord_large_fov_month_temporal_study.sh
fi

TASKS="${STUDY_ROOT}/manifests/training_tasks.tsv"
ACTIVE="${STUDY_ROOT}/manifests/active_rois.tsv"
TEMPORAL_META="${STUDY_ROOT}/manifests/temporal_split_summary.json"
FOV_SPEC="${STUDY_ROOT}/manifests/spinalcord_large_fov_spec.json"

require_file "${TASKS}"
require_file "${ACTIVE}"
require_file "${TEMPORAL_META}"
require_file "${FOV_SPEC}"
require_file "${CLINICAL_METRICS_CSV}"

mapfile -t roi_rows < <(awk -F '\t' 'NR>1 && NF {gsub(/\r/, ""); print $1 "\t" $3}' "${ACTIVE}")
[[ ${#roi_rows[@]} -eq 1 ]] || fail "expected exactly one active ROI, found ${#roi_rows[@]}"
IFS=$'\t' read -r ROI_INDEX ROI <<< "${roi_rows[0]}"
[[ "${ROI}" == "SpinalCord" ]] || fail "expected SpinalCord, found ${ROI}"

mapfile -t train_rows < <(awk -F '\t' 'NR>1 && NF {gsub(/\r/, ""); print $1 "\t" $2 "\t" $3 "\t" $4}' "${TASKS}")
[[ ${#train_rows[@]} -eq 4 ]] || fail "expected four SpinalCord training tasks, found ${#train_rows[@]}"

mkdir -p "${PIPELINE_STATE_DIR}"

# ---------------------------------------------------------------------------
# Stage 1: four models, sequentially on the same GPU allocation.
# A task is reusable only after the task runner returned successfully and this
# pipeline wrote its marker. A merely existing early best checkpoint is not
# enough to skip an interrupted training run.
# ---------------------------------------------------------------------------
log "Stage 1: sequential four-model training"
for row in "${train_rows[@]}"; do
  IFS=$'\t' read -r task_index task_roi model config <<< "${row}"
  [[ "${task_roi}" == "SpinalCord" ]] || fail "unexpected training ROI ${task_roi} in task ${task_index}"
  require_file "${config}"

  experiment_dir="$(python - "${config}" <<'PYCFG'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["output_root"]) / cfg["experiment_name"])
PYCFG
)"
  checkpoint="${experiment_dir}/checkpoints/best_decoder.pth"
  marker="${PIPELINE_STATE_DIR}/train_task_${task_index}_${model}.done"

  if stage_complete "${marker}" "${checkpoint}"; then
    echo "Reusing completed training task ${task_index}: ${model}"
    continue
  fi

  log "Training task ${task_index}/3: ${model}"
  FRESH=1 bash scripts/dino_shape/run_all_roi_training_task.sh "${task_index}"
  require_file "${checkpoint}"
  mark_done "${marker}"
done

# ---------------------------------------------------------------------------
# Stage 2: shared synthetic probe bank and four-model evaluation/report.
# ---------------------------------------------------------------------------
SYN_REPORT="${STUDY_ROOT}/reports/SpinalCord_four_models_NSRQA_shared_probe/report.md"
EVAL_MARKER="${PIPELINE_STATE_DIR}/synthetic_evaluation.done"
if stage_complete "${EVAL_MARKER}" "${SYN_REPORT}"; then
  log "Stage 2: reuse completed synthetic evaluation"
else
  log "Stage 2: synthetic shared-probe evaluation"
  FORCE_EVAL="${FORCE_EVAL:-0}" \
    BANK_ROOT="${BANK_ROOT}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    bash scripts/dino_shape/run_all_roi_evaluation_task.sh "${ROI_INDEX}"
  require_file "${SYN_REPORT}"
  mark_done "${EVAL_MARKER}"
fi

# ---------------------------------------------------------------------------
# Stage 3: real clinical autoseg preprocessing/reconstruction/regression.
# ---------------------------------------------------------------------------
PRIMARY_CSV="${REG_ROOT}/metrics/SpinalCord/autoseg_centered_metrics.csv"
ORACLE_CSV="${REG_ROOT}/metrics/SpinalCord/approved_reference_metrics.csv"
REG_SUMMARY="${REG_ROOT}/regression/SpinalCord/regression_summary.csv"
ACTUAL_MARKER="${PIPELINE_STATE_DIR}/actual_autoseg.done"
if stage_complete "${ACTUAL_MARKER}" "${PRIMARY_CSV}" "${ORACLE_CSV}" "${REG_SUMMARY}"; then
  log "Stage 3: reuse completed actual-autoseg evaluation"
else
  log "Stage 3: actual-autoseg NSR-QA reconstruction/regression"
  AUTOSEG_CENTER_METHOD="${AUTOSEG_CENTER_METHOD}" \
    REG_ROOT="${REG_ROOT}" \
    bash scripts/dino_shape/run_actual_autoseg_regression_task.sh "${ROI_INDEX}"
  require_file "${PRIMARY_CSV}"
  require_file "${ORACLE_CSV}"
  require_file "${REG_SUMMARY}"
  mark_done "${ACTUAL_MARKER}"
fi

# ---------------------------------------------------------------------------
# Stage 4: P75/P50 real-autoseg ranking/classification using full-volume
# clinical Dice/HD95 truth from metrics.csv.
# ---------------------------------------------------------------------------
CLASS_DIR="${REG_ROOT}/ranking_classification/SpinalCord"
CLASS_SUMMARY="${CLASS_DIR}/classification_summary.csv"
RANK_SUMMARY="${CLASS_DIR}/ranking_summary.csv"
CLASS_AUDIT="${CLASS_DIR}/clinical_truth_join_audit.csv"
RANK_MARKER="${PIPELINE_STATE_DIR}/ranking_classification.done"
if stage_complete "${RANK_MARKER}" "${CLASS_SUMMARY}" "${RANK_SUMMARY}" "${CLASS_AUDIT}"; then
  log "Stage 4: reuse completed ranking/classification"
else
  log "Stage 4: percentile ranking/classification"
  CLINICAL_METRICS_CSV="${CLINICAL_METRICS_CSV}" \
    CLINICAL_METRICS_DUPLICATE_POLICY="${CLINICAL_METRICS_DUPLICATE_POLICY}" \
    CLINICAL_SITE="${CLINICAL_SITE}" \
    REG_ROOT="${REG_ROOT}" \
    bash scripts/dino_shape/run_actual_autoseg_ranking_classification.sh "${ROI_INDEX}"
  require_file "${CLASS_SUMMARY}"
  require_file "${RANK_SUMMARY}"
  require_file "${CLASS_AUDIT}"
  mark_done "${RANK_MARKER}"
fi

# ---------------------------------------------------------------------------
# Stage 5: manuscript-friendly ranking/classification report, same format as
# the August experiment.
# ---------------------------------------------------------------------------
REPORT_MD="${REG_ROOT}/ranking_classification_report/report.md"
REPORT_MARKER="${PIPELINE_STATE_DIR}/ranking_classification_report.done"
if stage_complete "${REPORT_MARKER}" "${REPORT_MD}"; then
  log "Stage 5: reuse completed ranking/classification report"
else
  log "Stage 5: ranking/classification report"
  ACTUAL_ROOT="${REG_ROOT}" \
    OUTPUT_DIR="${REG_ROOT}/ranking_classification_report" \
    bash scripts/dino_shape/generate_actual_autoseg_ranking_classification_report.sh
  require_file "${REPORT_MD}"
  mark_done "${REPORT_MARKER}"
fi

log "Pipeline completed successfully"
echo "Synthetic report: ${SYN_REPORT}"
echo "Clinical reconstructions: ${REG_ROOT}/{autoseg_centered,approved_reference}/SpinalCord/<split>/patient_<id>/"
echo "Regression: ${REG_SUMMARY}"
echo "Classification: ${CLASS_SUMMARY}"
echo "Ranking/classification report: ${REPORT_MD}"
echo "Pipeline state: ${PIPELINE_STATE_DIR}"
