#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
FORCE="${FORCE:-0}"

cd "${PROJECT_ROOT}"

python3 - "${PROJECT_ROOT}" "${FORCE}" <<'PY'
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
force = sys.argv[2] == "1"

pairs = [
    ("configs/dino_shape/dino_shape_pilot_Lung_L.yaml",
     "configs/dino_shape/dino_shape_pilot_Bronchus.yaml"),
    ("configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml",
     "configs/dino_shape/dino_shape_Bronchus_probe_eval.yaml"),
    ("configs/dino_shape/dino_shape_denoising_Lung_L_v1.yaml",
     "configs/dino_shape/dino_shape_denoising_Bronchus_v1.yaml"),
    ("configs/dino_shape/dino_shape_denoising_Lung_L_v1_eval.yaml",
     "configs/dino_shape/dino_shape_denoising_Bronchus_v1_eval.yaml"),
    ("scripts/dino_shape/run_dino_shape_pilot_Lung_L.sh",
     "scripts/dino_shape/run_dino_shape_pilot_Bronchus.sh"),
    ("scripts/dino_shape/run_dino_shape_probe_sweep_Lung_L.sh",
     "scripts/dino_shape/run_dino_shape_probe_sweep_Bronchus.sh"),
    ("scripts/dino_shape/run_dino_shape_denoising_Lung_L_v1.sh",
     "scripts/dino_shape/run_dino_shape_denoising_Bronchus_v1.sh"),
    ("scripts/dino_shape/run_dino_shape_denoising_probe_sweep_Lung_L_v1.sh",
     "scripts/dino_shape/run_dino_shape_denoising_probe_sweep_Bronchus_v1.sh"),
    ("scripts/dino_shape/run_export_dino_shape_denoising_probe_contours_Lung_L_v1.sh",
     "scripts/dino_shape/run_export_dino_shape_denoising_probe_contours_Bronchus_v1.sh"),
]

missing = [str(root / src) for src, _ in pairs if not (root / src).is_file()]
if missing:
    raise SystemExit("Missing Lung_L templates:\n  " + "\n  ".join(missing))

replacements = [
    ("dino_shape_denoising_Lung_L_v1", "dino_shape_denoising_Bronchus_v1"),
    ("dino_shape_pilot_Lung_L", "dino_shape_pilot_Bronchus"),
    ("dino_shape_Lung_L_probe_eval", "dino_shape_Bronchus_probe_eval"),
    ("run_dino_shape_denoising_probe_sweep_Lung_L_v1", "run_dino_shape_denoising_probe_sweep_Bronchus_v1"),
    ("run_dino_shape_denoising_Lung_L_v1", "run_dino_shape_denoising_Bronchus_v1"),
    ("run_dino_shape_probe_sweep_Lung_L", "run_dino_shape_probe_sweep_Bronchus"),
    ("run_dino_shape_pilot_Lung_L", "run_dino_shape_pilot_Bronchus"),
    ("run_export_dino_shape_denoising_probe_contours_Lung_L_v1", "run_export_dino_shape_denoising_probe_contours_Bronchus_v1"),
    ("Lung_L_1p5mm_nozoom", "Bronchus_1p5mm_nozoom"),
    ("/lungs_chest/Lung_L/", "/airway/Bronchus/"),
    ("Lung_L", "Bronchus"),
]

for src_rel, dst_rel in pairs:
    src = root / src_rel
    dst = root / dst_rel
    if dst.exists() and not force:
        print(f"keep existing: {dst_rel}")
        continue
    text = src.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    if dst.suffix in {".yaml", ".yml"}:
        # Current evaluators consume spacing_zyx. Keep generated configs aligned.
        text = text.replace("\n  spacing: [", "\n  spacing_zyx: [")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    src_mode = stat.S_IMODE(src.stat().st_mode)
    os.chmod(dst, src_mode)
    print(f"created: {dst_rel}")
PY

echo
echo "Bronchus pilot and benchmark-leakage files are ready."
echo "Review them with: git diff -- configs/dino_shape scripts/dino_shape"
