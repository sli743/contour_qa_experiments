#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Separate generated plot files by metric comparison type
# ============================================================
#
# Usage:
#   bash organize_score_distribution_plots.sh
#
# Optional:
#   PLOT_ROOT=/path/to/score_distribution_analysis_all/plots bash organize_score_distribution_plots.sh
#   MODE=copy bash organize_score_distribution_plots.sh
#
# Default input:
#   ../radformation-data/probe_auc_sensitivity_assd/first_detectable_volume_custom/score_distribution_analysis_all/plots
#
# Output:
#   ${PLOT_ROOT}/organized_by_metric/input_vs_recon/
#   ${PLOT_ROOT}/organized_by_metric/input_vs_target/
#   ${PLOT_ROOT}/organized_by_metric/recon_vs_target/
#   ${PLOT_ROOT}/organized_by_metric/other/
#
# MODE:
#   copy     copy files, keep original plots in place
#   symlink  create symbolic links instead of copying
#   move     move files out of original folders

FDVC_ROOT="${FDVC_ROOT:-../radformation-data/probe_auc_sensitivity_assd/first_detectable_volume_custom/}"
PLOT_ROOT="${PLOT_ROOT:-${FDVC_ROOT}/score_distribution_analysis_all/plots}"
OUT_ROOT="${OUT_ROOT:-${PLOT_ROOT}/organized_by_metric}"
MODE="${MODE:-copy}"

if [[ ! -d "${PLOT_ROOT}" ]]; then
  echo "ERROR: PLOT_ROOT does not exist: ${PLOT_ROOT}"
  exit 1
fi

case "${MODE}" in
  copy|symlink|move) ;;
  *)
    echo "ERROR: MODE must be copy, symlink, or move. Got: ${MODE}"
    exit 1
    ;;
esac

mkdir -p \
  "${OUT_ROOT}/input_vs_recon" \
  "${OUT_ROOT}/input_vs_target" \
  "${OUT_ROOT}/recon_vs_target" \
  "${OUT_ROOT}/other"

count_input_recon=0
count_input_target=0
count_recon_target=0
count_other=0

put_file () {
  local src="$1"
  local category="$2"

  local rel
  rel="$(realpath --relative-to="${PLOT_ROOT}" "${src}")"

  # Avoid recursively organizing already organized files.
  if [[ "${rel}" == organized_by_metric/* ]]; then
    return 0
  fi

  local dst="${OUT_ROOT}/${category}/${rel}"
  mkdir -p "$(dirname "${dst}")"

  case "${MODE}" in
    copy)
      cp -f "${src}" "${dst}"
      ;;
    symlink)
      ln -sfn "$(realpath "${src}")" "${dst}"
      ;;
    move)
      mv -f "${src}" "${dst}"
      ;;
  esac
}

while IFS= read -r -d '' f; do
  base="$(basename "${f}")"

  if [[ "${base}" == *"input_vs_recon"* ]] || \
     [[ "${base}" == *"one_minus_dice_input_vs_recon"* ]] || \
     [[ "${base}" == *"abs_input_recon_volume_diff_frac"* ]]; then
    put_file "${f}" "input_vs_recon"
    count_input_recon=$((count_input_recon + 1))

  elif [[ "${base}" == *"input_vs_target"* ]] || \
       [[ "${base}" == *"input_target"* ]]; then
    put_file "${f}" "input_vs_target"
    count_input_target=$((count_input_target + 1))

  elif [[ "${base}" == *"recon_vs_target"* ]] || \
       [[ "${base}" == *"recon_target"* ]]; then
    put_file "${f}" "recon_vs_target"
    count_recon_target=$((count_recon_target + 1))

  else
    put_file "${f}" "other"
    count_other=$((count_other + 1))
  fi
done < <(
  find "${PLOT_ROOT}" \
    -type f \( -iname "*.png" -o -iname "*.pdf" -o -iname "*.svg" \) \
    -not -path "${OUT_ROOT}/*" \
    -print0
)

cat > "${OUT_ROOT}/README_organized_by_metric.txt" <<EOF
Organized plots by metric comparison type.

Source:
  ${PLOT_ROOT}

Mode:
  ${MODE}

Categories:
  input_vs_recon:
    Metrics comparing corrupted/input contour with AE reconstruction.
    Examples: dice_input_vs_recon, one_minus_dice_input_vs_recon,
              hd95_input_vs_recon, assd_input_vs_recon,
              abs_input_recon_volume_diff_frac

  input_vs_target:
    Metrics comparing corrupted/input contour with clean ground truth.
    Examples: dice_input_vs_target, hd95_input_vs_target, assd_input_vs_target

  recon_vs_target:
    Metrics comparing AE reconstruction with clean ground truth.
    Examples: dice_recon_vs_target, hd95_recon_vs_target, assd_recon_vs_target

  other:
    Files whose metric type could not be inferred from filename.

Counts:
  input_vs_recon:  ${count_input_recon}
  input_vs_target: ${count_input_target}
  recon_vs_target: ${count_recon_target}
  other:           ${count_other}
EOF

echo "Done."
echo "Output root:"
echo "  ${OUT_ROOT}"
echo
echo "Counts:"
echo "  input_vs_recon:  ${count_input_recon}"
echo "  input_vs_target: ${count_input_target}"
echo "  recon_vs_target: ${count_recon_target}"
echo "  other:           ${count_other}"
