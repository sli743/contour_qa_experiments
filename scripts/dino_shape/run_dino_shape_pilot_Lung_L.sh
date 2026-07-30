#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/shaohan/3DINO}"
CONFIG="${CONFIG:-${REPO}/dino_shape_pilot_Lung_L.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
MAX_CASES="${MAX_CASES:-6}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# 1. Train decoder only; pretrained 3DINO encoder remains frozen.
python -m training.train_dino_shape_decoder --config "${CONFIG}"

# 2. Export clean test reconstructions.
python -m evaluation.eval_dino_shape_clean \
  --config "${CONFIG}" \
  --split test

# 3. Generate a small probe set and export clean/corrupted/reconstruction masks
#    with exact filenames expected by ProbeContourComparer.py.
python -m evaluation.probes.eval_dino_shape_probes_visual \
  --config "${CONFIG}" \
  --split test \
  --max_cases "${MAX_CASES}"

echo
echo "Pilot complete."
echo "Open this folder in the Slicer Probe Contour Comparer:"
python - "${CONFIG}" <<'PY'
from pathlib import Path
import yaml, sys
with open(sys.argv[1]) as f:
    c=yaml.safe_load(f)
print(Path(c["output_root"])/c["experiment_name"]/"probe_visualization_test")
PY
