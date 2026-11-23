#!/usr/bin/env bash
set -euo pipefail

# Leak test: add the region-growth corruption branch (deep extra lung) and retrain
# the two corruption models of the 2x2 for each seed:
#   0 ct_contour_corruption_growth  (C1I1g)  from OLD_STUDY/configs/ct_contour_corruption_seed<S>.yaml
#   1 contour_only_growth           (C1I0g)  from V3_STUDY/configs/nsrqa_corruption_seed<S>.yaml
# The only change to each config is the growth branch (analysis.add_growth_corruption):
# size/depth window from the ROI's own excess error components, weight = dropout's.
# Then Month3 + clean-approved eval, CT-absent eval of C1I1g, and region detection
# over the existing arms plus both growth arms. EXCLUDE as in the mask-stem test.
#
#   ROI=Lung_L bash scripts/dino_shape/submit_growth_corruption.sh
#   ROI=Lung_R bash scripts/dino_shape/submit_growth_corruption.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ROI="${ROI:-Lung_L}"
ROI_LC="${ROI,,}"
JOB_TAG="${JOB_TAG:-lung${ROI:5:1}}"
OLD_STUDY="${OLD_STUDY:-/scratch/shaohan/contour_QA/${ROI_LC}_ct_contour_nsrqa_v1}"
V3_STUDY="${V3_STUDY:-/scratch/shaohan/contour_QA/${ROI_LC}_nsrqa_month_temporal_family_validated_v1}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/${ROI_LC}_ct_contour_nsrqa_growth}"
EC="${EC:-/scratch/shaohan/contour_QA/${ROI_LC}_error_field_characterization_v1/error_components.csv}"
SEEDS="${SEEDS:-91 92 93}"
if [[ "${ROI}" == "Lung_L" ]]; then EXCLUDE="${EXCLUDE-152}"; else EXCLUDE="${EXCLUDE-26 100}"; fi
GROWTH_ARGS="${GROWTH_ARGS:-}"   # e.g. "--weight 0.1 --min-depth-mm 6"
ARRAY_LIMIT="${ARRAY_LIMIT:-}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${STUDY_ROOT}/configs" "${STUDY_ROOT}/logs"
[[ -s "${EC}" ]] || { echo "Missing ${EC} (set EC=...)" >&2; exit 2; }

SRC_CROPS="$(readlink -f "${OLD_STUDY}/actual_autoseg_preprocessed")"
[[ -d "${SRC_CROPS}/autoseg_centered" ]] || { echo "Missing ${SRC_CROPS}/autoseg_centered" >&2; exit 2; }
LINK="${STUDY_ROOT}/actual_autoseg_preprocessed"
if [[ -e "${LINK}" || -L "${LINK}" ]]; then
  [[ "$(readlink -f "${LINK}")" == "${SRC_CROPS}" ]] || { echo "${LINK} points elsewhere" >&2; exit 2; }
else
  ln -s "${SRC_CROPS}" "${LINK}"
fi
for SEED in ${SEEDS}; do
  for a in contour_only_no_corruption contour_only_v3 ct_contour_no_corruption ct_contour_corruption ct_only_segmenter; do
    [[ -d "${OLD_STUDY}/evaluations/${a}/seed${SEED}/actual_autoseg_cases" ]] || {
      echo "Missing ${OLD_STUDY}/evaluations/${a}/seed${SEED}" >&2; exit 2; }
  done
done

jobid() { printf '%s' "$1" | cut -d';' -f1; }
arr() { [[ -n "${ARRAY_LIMIT}" ]] && printf '%s%%%s' "$1" "${ARRAY_LIMIT}" || printf '%s' "$1"; }
EVALS=()
SUMMARY=()

for SEED in ${SEEDS}; do
  MANIFEST="${STUDY_ROOT}/experiment_manifest_seed${SEED}.csv"
  # shellcheck disable=SC2086
  python - "${STUDY_ROOT}" "${SEED}" "${ROI_LC}" "${MANIFEST}" "${EC}" \
      "${OLD_STUDY}/configs/ct_contour_corruption_seed${SEED}.yaml" \
      "${V3_STUDY}/configs/nsrqa_corruption_seed${SEED}.yaml" ${GROWTH_ARGS} <<'PY'
import argparse, csv, json, sys
from pathlib import Path
import yaml
from analysis.add_growth_corruption import add_growth, growth_targets

study, seed, roi_lc, manifest, ec, ct_src, c_src, *rest = sys.argv[1:]
ap = argparse.ArgumentParser()
ap.add_argument("--weight", type=float, default=None)
ap.add_argument("--min-depth-mm", type=float, default=4.5)
ap.add_argument("--quantiles", default="0.1,0.9")
opt = ap.parse_args(rest)
study = Path(study)
t = growth_targets(ec, opt.min_depth_mm, tuple(float(v) for v in opt.quantiles.split(",")))
if seed == "91":
    print("growth window:", json.dumps({k: t[k] for k in ("added_fraction", "min_max_depth_mm", "max_max_depth_mm")}),
          f"({t['summary']['n_patients']} patients)")
rows = []
for idx, (arm, src, want_ct) in enumerate([("ct_contour_corruption_growth", ct_src, True),
                                           ("contour_only_growth", c_src, False)]):
    if not Path(src).is_file():
        sys.exit(f"missing source config {src}")
    cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    has_ct = bool((cfg.get("image_conditioning") or {}).get("enabled", False))
    if has_ct != want_ct:
        sys.exit(f"{src}: image_conditioning.enabled={has_ct}, expected {want_ct}")
    if float(cfg["stochastic_region_dropout"].get("branch_probabilities", {}).get("stochastic_region_growth", 0)) > 0:
        sys.exit(f"{src} already has a growth branch")
    new = add_growth(cfg, t, opt.weight)
    new["experiment_name"] = f"{roi_lc}_{arm}_seed{seed}"
    new["output_root"] = str(study / "models" / arm)
    meta = dict(new.get("study_metadata") or {})
    meta.update({"arm": arm, "source_config": str(Path(src).resolve()),
                 "intervention": "as the source corruption model plus the region-growth (leak) branch"})
    new["study_metadata"] = meta
    model_dir = Path(new["output_root"]) / new["experiment_name"]
    if model_dir.exists():
        sys.exit(f"{model_dir} exists; remove it or pick another STUDY_ROOT")
    path = study / "configs" / f"{arm}_seed{seed}.yaml"
    path.write_text(yaml.safe_dump(new, sort_keys=False), encoding="utf-8")
    rows.append({"task_index": idx, "arm": arm, "seed": seed, "config": str(path),
                 "experiment_name": new["experiment_name"], "model_dir": str(model_dir),
                 "checkpoint": str(model_dir / "checkpoints" / "best_decoder.pth")})
    print(f"seed {seed}: {path}")
with open(manifest, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
PY

  EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},ROI=${ROI},MANIFEST=${MANIFEST}"
  LOGS=(--output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err")
  TRAIN="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_gr_train" \
    --export="${EXPORTS}" "${LOGS[@]}" --array="$(arr 0-1)" \
    jobs/lung_l_nsrqa_month_temporal_ablation_train.job)")"
  EVAL="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_gr_eval" \
    --dependency="aftercorr:${TRAIN}" \
    --export="${EXPORTS},IMAGE_MODE=present" "${LOGS[@]}" --array="$(arr 0-1)" \
    jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"
  ABS="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo --job-name="${JOB_TAG}_s${SEED}_gr_ctabsent" \
    --dependency="aftercorr:${TRAIN}" \
    --export="${EXPORTS},IMAGE_MODE=absent" "${LOGS[@]}" --array="$(arr 0)" \
    jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)")"
  EVALS+=("${EVAL}")
  SUMMARY+=("seed ${SEED}: train ${TRAIN} eval ${EVAL} ct-absent ${ABS}")
done

DEP="afterok$(printf ':%s' "${EVALS[@]}")"
export PROJECT_ROOT ROI SEEDS EXCLUDE
export OUT="${STUDY_ROOT}/evaluations_floor_corrected/region_detection"
export EXTRA_ARGS="--arms C0I0 C1I0 C0I1 C1I1 C1I0g C1I1g ct_only --arm-root C1I1g=${STUDY_ROOT}/evaluations/ct_contour_corruption_growth --arm-root C1I0g=${STUDY_ROOT}/evaluations/contour_only_growth"
mkdir -p "$(dirname "${OUT}")"
REG="$(jobid "$(STUDY_ROOT="${OLD_STUDY}" sbatch --parsable -p hugo --account=geoffrey_hugo \
  --job-name="${JOB_TAG}_gr_regions" --dependency="${DEP}" --export=ALL \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/nsrqa_region_detection.job)")"

echo
echo "Submitted ${ROI} growth-corruption study -> ${STUDY_ROOT}"
printf '  %s\n' "${SUMMARY[@]}"
echo "  region detection ${REG} -> ${OUT}{,_assessable} (excluded: ${EXCLUDE:-none})"
echo "Leak acceptance per epoch: ${STUDY_ROOT}/models/*/*/logs/training_corruption_summary.csv (branch_counts_json: stochastic_region_growth vs rejected_to_clean)"
