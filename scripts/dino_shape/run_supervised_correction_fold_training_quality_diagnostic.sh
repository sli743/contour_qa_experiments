#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Lung_L}"
VARIANT="${2:-large_decoder_ft_last2}"
FOLDS="${FOLDS:-0,1,2,3,4}"
QA_METHOD="${QA_METHOD:-joint_spatial_action_head}"
BAD_PERCENTILE="${BAD_PERCENTILE:-75}"
OUTPUT_DIR="${OUTPUT_DIR:-${STUDY_ROOT}/training_quality_diagnostics/${ROI}/${VARIANT}}"
OUTPUT_ZIP="${OUTPUT_ZIP:-${STUDY_ROOT}/training_quality_diagnostics/${ROI}/${VARIANT}_training_quality_diagnostic.zip}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m analysis.diagnose_supervised_correction_fold_training_quality \
  --study-root "${STUDY_ROOT}" \
  --roi "${ROI}" \
  --variant "${VARIANT}" \
  --folds "${FOLDS}" \
  --method "${QA_METHOD}" \
  --bad-percentile "${BAD_PERCENTILE}" \
  --output-dir "${OUTPUT_DIR}"

python - "${OUTPUT_DIR}" "${OUTPUT_ZIP}" <<'PY'
from pathlib import Path
import sys, zipfile
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(src.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(src))
print(dst)
PY

echo "Training-quality diagnostic complete: ${OUTPUT_DIR}"
echo "Uploadable ZIP: ${OUTPUT_ZIP}"
