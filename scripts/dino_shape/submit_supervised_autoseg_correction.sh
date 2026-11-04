#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ALLROI_ROOT="${ALLROI_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
cd "$PROJECT_ROOT"
stage="${1:-}"
case "$stage" in
  preprocess)
    n=$(($(wc -l < "${ALLROI_ROOT}/manifests/active_rois.tsv")-1)); [[ $n -gt 0 ]]
    sbatch -p hugo --account geoffrey_hugo --array="0-$((n-1))%${MAX_CONCURRENT:-4}" jobs/supervised_correction_preprocess_array.job
    ;;
  prepare)
    source scripts/dino_shape/activate_3dino_env.sh
    python scripts/dino_shape/prepare_supervised_autoseg_correction.py --study-root "$STUDY_ROOT" --all-roi-study-root "$ALLROI_ROOT"
    ;;
  train|eval)
    tasks="${STUDY_ROOT}/manifests/training_tasks.tsv"; [[ -f "$tasks" ]] || { echo "Run prepare first" >&2; exit 2; }
    n=$(($(wc -l < "$tasks")-1)); [[ $n -gt 0 ]]
    if [[ "$stage" == train ]]; then job=jobs/supervised_correction_train_array.job; conc="${MAX_CONCURRENT:-4}"; else job=jobs/supervised_correction_eval_array.job; conc="${MAX_CONCURRENT:-2}"; fi
    sbatch -p hugo --account geoffrey_hugo --array="0-$((n-1))%${conc}" "$job"
    ;;
  summarize) bash scripts/dino_shape/summarize_supervised_correction_all_rois.sh ;;
  *) echo "Usage: $0 {preprocess|prepare|train|eval|summarize}" >&2; exit 2;;
esac
