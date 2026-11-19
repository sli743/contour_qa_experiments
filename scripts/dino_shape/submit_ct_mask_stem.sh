#!/usr/bin/env bash
set -euo pipefail

# Early-fusion test: CT + contour two-channel stem (image_skip_inputs: ct_mask).
#
# The current CT + contour model (C1I1) sees the contour only as 16^3 3DINO tokens;
# its full-resolution stem sees CT alone. This retrains C1I1 with the stem fed
# [CT, contour], so the decoder can compare the input boundary with the image voxel
# by voxel. Everything else is the existing C1I1 config of the same seed (copied
# from OLD_STUDY/configs), so the comparison with C1I1 is exact.
#
# Per seed: train (1 GPU task), Month3 + clean-approved eval, CT-absent eval.
# Then region detection over all six arms, with the five existing arms read from
# OLD_STUDY/evaluations and the new arm (C1I1m) from STUDY_ROOT. EXCLUDE drops
# cases that cannot be judged (Lung_L 152: approved contour includes stomach gas).
#
#   ROI=Lung_L bash scripts/dino_shape/submit_ct_mask_stem.sh
#   ROI=Lung_R bash scripts/dino_shape/submit_ct_mask_stem.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ROI="${ROI:-Lung_L}"
ROI_LC="${ROI,,}"
JOB_TAG="${JOB_TAG:-lung${ROI:5:1}}"
OLD_STUDY="${OLD_STUDY:-/scratch/shaohan/contour_QA/${ROI_LC}_ct_contour_nsrqa_v1}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/${ROI_LC}_ct_contour_nsrqa_maskstem}"
SEEDS="${SEEDS:-91 92 93}"
if [[ "${ROI}" == "Lung_L" ]]; then EXCLUDE="${EXCLUDE-152}"; else EXCLUDE="${EXCLUDE-26 100}"; fi
CT_ABSENT="${CT_ABSENT:-1}"
ARRAY_LIMIT="${ARRAY_LIMIT:-}"
ARM=ct_mask_contour_corruption

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${STUDY_ROOT}/configs" "${STUDY_ROOT}/logs"

# Same Month3 crops (with CT) as the existing study.
SRC_CROPS="$(readlink -f "${OLD_STUDY}/actual_autoseg_preprocessed")"
[[ -d "${SRC_CROPS}/autoseg_centered" ]] || { echo "Missing ${SRC_CROPS}/autoseg_centered" >&2; exit 2; }
LINK="${STUDY_ROOT}/actual_autoseg_preprocessed"
if [[ -e "${LINK}" || -L "${LINK}" ]]; then
  [[ "$(readlink -f "${LINK}")" == "${SRC_CROPS}" ]] || { echo "${LINK} points elsewhere" >&2; exit 2; }
else
  ln -s "${SRC_CROPS}" "${LINK}"
fi

# The existing arms must already be evaluated for every seed.
for SEED in ${SEEDS}; do
  for a in contour_only_no_corruption contour_only_v3 ct_contour_no_corruption ct_contour_corruption ct_only_segmenter; do
    [[ -d "${OLD_STUDY}/evaluations/${a}/seed${SEED}/actual_autoseg_cases" ]] || {
      echo "Missing ${OLD_STUDY}/evaluations/${a}/seed${SEED} (run the postprocess job for that seed first)" >&2; exit 2; }
  done
done

jobid() { printf '%s' "$1" | cut -d';' -f1; }
arr() { [[ -n "${ARRAY_LIMIT}" ]] && printf '%s%%%s' "$1" "${ARRAY_LIMIT}" || printf '%s' "$1"; }
EVALS=()
SUMMARY=()

for SEED in ${SEEDS}; do
  SRC_CFG="${OLD_STUDY}/configs/ct_contour_corruption_seed${SEED}.yaml"
  MANIFEST="${STUDY_ROOT}/experiment_manifest_seed${SEED}.csv"
  python - "${SRC_CFG}" "${STUDY_ROOT}" "${ARM}" "${SEED}" "${ROI_LC}" "${MANIFEST}" <<'PY'
import copy, csv, sys
from pathlib import Path
import yaml

src, study, arm, seed, roi_lc, manifest = sys.argv[1:]
study = Path(study)
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
img = cfg.get("image_conditioning") or {}
if not img.get("enabled") or img.get("contour_input", "mask") != "mask":
    sys.exit(f"{src} is not a CT + contour config")
if int(img.get("image_skip_last_n_stages", 2)) <= 0:
    sys.exit(f"{src} has no full-resolution stem (image_skip_last_n_stages <= 0)")
new = copy.deepcopy(cfg)
new["image_conditioning"]["image_skip_inputs"] = "ct_mask"
new["experiment_name"] = f"{roi_lc}_{arm}_seed{seed}"
new["output_root"] = str(study / "models" / arm)
meta = dict(new.get("study_metadata") or {})
meta.update({
    "arm": arm,
    "intervention": "as ct_contour_corruption, but the full-resolution stem sees [CT, contour] (early fusion)",
    "source_ct_contour_config": str(Path(src).resolve()),
})
new["study_metadata"] = meta
model_dir = Path(new["output_root"]) / new["experiment_name"]
if model_dir.exists():
    sys.exit(f"{model_dir} exists; remove it or pick another STUDY_ROOT")
cfg_path = study / "configs" / f"{arm}_seed{seed}.yaml"
cfg_path.write_text(yaml.safe_dump(new, sort_keys=False), encoding="utf-8")
with open(manifest, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["task_index", "arm", "seed", "config", "experiment_name", "model_dir", "checkpoint"])
    w.writeheader()
    w.writerow({"task_index": 0, "arm": arm, "seed": seed, "config": str(cfg_path),
                "experiment_name": new["experiment_name"], "model_dir": str(model_dir),
                "checkpoint": str(model_dir / "checkpoints" / "best_decoder.pth")})
print(f"seed {seed}: {cfg_path}  (from {src})")
PY

  EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},ROI=${ROI},MANIFEST=${MANIFEST}"
  LOGS=(--output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err")
  TRAIN="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_ms_train" \
    --export="${EXPORTS}" "${LOGS[@]}" --array="$(arr 0)" \
    jobs/lung_l_nsrqa_month_temporal_ablation_train.job)")"
  EVAL="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_ms_eval" \
    --dependency="aftercorr:${TRAIN}" \
    --export="${EXPORTS},IMAGE_MODE=present" "${LOGS[@]}" --array="$(arr 0)" \
    jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"
  EVALS+=("${EVAL}")
  LINE="seed ${SEED}: train ${TRAIN} eval ${EVAL}"
  if [[ "${CT_ABSENT}" == "1" ]]; then
    ABS="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_ms_ctabsent" \
      --dependency="aftercorr:${TRAIN}" \
      --export="${EXPORTS},IMAGE_MODE=absent" "${LOGS[@]}" --array="$(arr 0)" \
      jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"
    LINE+=" ct-absent ${ABS}"
  fi
  SUMMARY+=("${LINE}")
done

# Region detection: five existing arms from OLD_STUDY + C1I1m from STUDY_ROOT.
# Values contain spaces and '=', so they go through the environment (--export=ALL).
DEP="afterok$(printf ':%s' "${EVALS[@]}")"
export PROJECT_ROOT ROI SEEDS EXCLUDE
export OUT="${STUDY_ROOT}/evaluations_floor_corrected/region_detection"
export EXTRA_ARGS="--arms C0I0 C1I0 C0I1 C1I1 C1I1m ct_only --arm-root C1I1m=${STUDY_ROOT}/evaluations/${ARM}"
mkdir -p "$(dirname "${OUT}")"
REG="$(jobid "$(STUDY_ROOT="${OLD_STUDY}" sbatch --parsable -p hugo --account=geoffrey_hugo \
  --job-name="${JOB_TAG}_ms_regions" --dependency="${DEP}" --export=ALL \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/nsrqa_region_detection.job)")"

echo
echo "Submitted ${ROI} mask-stem study -> ${STUDY_ROOT}"
printf '  %s\n' "${SUMMARY[@]}"
echo "  region detection ${REG} -> ${OUT}{,_assessable} (excluded: ${EXCLUDE:-none})"
