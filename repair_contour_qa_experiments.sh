#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: project directory not found: $PROJECT_ROOT" >&2
  exit 1
fi

export PROJECT_ROOT

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

root = Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()

required = [
    root / "models/dino_shape/dino_shape_autoencoder.py",
    root / "training/train_dino_shape_decoder.py",
    root / "evaluation/eval_dino_shape_clean.py",
    root / "evaluation/probes/eval_post_dae_probe_sitk_morph_sweep_assd.py",
    root / "configs/dino_shape/dino_shape_pilot_Lung_L.yaml",
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))

# ---------------------------------------------------------------------
# Package markers
# ---------------------------------------------------------------------
for rel in [
    "models/__init__.py",
    "models/dino_shape/__init__.py",
    "training/__init__.py",
    "evaluation/__init__.py",
    "evaluation/probes/__init__.py",
    "evaluation/metrics/__init__.py",
    "analysis/__init__.py",
    "analysis/roc/__init__.py",
    "analysis/fdvc/__init__.py",
    "analysis/plots/__init__.py",
]:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)

# ---------------------------------------------------------------------
# Model: add a standard reconstruction API and one shared checkpoint loader.
# ---------------------------------------------------------------------
model_path = root / "models/dino_shape/dino_shape_autoencoder.py"
text = model_path.read_text()
text = text.replace(
    "from typing import Sequence, Tuple",
    "from typing import Any, Mapping, Sequence, Tuple",
)
old = """        return logits, features
"""
new = """        return logits, features

    @torch.no_grad()
    def reconstruct(
        self,
        binary_mask: torch.Tensor,
        threshold: float = 0.5,
        return_probability: bool = False,
    ):
        \"\"\"Reconstruct a binary mask through the shared evaluator interface.\"\"\"
        logits, features = self(binary_mask)
        probability = torch.sigmoid(logits)
        reconstruction = (probability >= float(threshold)).float()
        if return_probability:
            return reconstruction, probability, features
        return reconstruction


def load_trained_dino_shape_model(
    config: Mapping[str, Any],
    decoder_checkpoint: str | Path,
    device: torch.device | str,
) -> Frozen3DINOShapeAutoencoder:
    \"\"\"Build the frozen encoder and load a trained contour decoder.\"\"\"
    dino_cfg = config[\"dino\"]
    decoder_cfg = config.get(\"decoder\", {})
    model = Frozen3DINOShapeAutoencoder(
        repo_path=Path(dino_cfg[\"repo_path\"]).expanduser(),
        checkpoint_path=Path(dino_cfg[\"checkpoint_path\"]).expanduser(),
        config_name=dino_cfg.get(\"config_name\", \"train/vit3d_highres\"),
        decoder_channels=decoder_cfg.get(\"channels\", [256, 128, 64, 32]),
    ).to(device)

    checkpoint = Path(decoder_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f\"Trained decoder checkpoint not found: {checkpoint}\")
    state = torch.load(checkpoint, map_location=\"cpu\")
    decoder_state = state.get(\"decoder_state_dict\", state)
    model.decoder.load_state_dict(decoder_state, strict=True)
    model.eval()
    return model
"""
if "def load_trained_dino_shape_model(" not in text:
    if old not in text:
        raise RuntimeError("Could not locate model return statement for patching")
    text = text.replace(old, new, 1)
model_path.write_text(text)

# ---------------------------------------------------------------------
# Training script: package-safe imports and cwd-independent execution.
# ---------------------------------------------------------------------
train_path = root / "training/train_dino_shape_decoder.py"
text = train_path.read_text()
text = text.replace(
    "from dino_shape_autoencoder import Frozen3DINOShapeAutoencoder",
    "from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder",
)
train_path.write_text(text)

# ---------------------------------------------------------------------
# Clean evaluator: overwrite with a package-safe, path-explicit version that
# adds HD95 and ASSD while preserving historical output names.
# ---------------------------------------------------------------------
clean_path = root / "evaluation/eval_dino_shape_clean.py"
clean_path.write_text(r'''#!/usr/bin/env python3
"""Evaluate and export clean binary ROI reconstructions from a trained DINO decoder."""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from models.dino_shape.dino_shape_autoencoder import load_trained_dino_shape_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def patient_id_from_name(name: str) -> str:
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    patterns = [
        r"(?:label|labels|mask|seg|proj|ct)_patient_?(.+)$",
        r"patient_?(.+?)(?:_(?:label|labels|mask|seg|approved|ct))?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else stem


def find_ct(ct_dir: Path, patient_id: str) -> Optional[Path]:
    candidates = list(ct_dir.rglob(f"*patient*{patient_id}*.nii.gz"))
    if not candidates:
        candidates = [
            p for p in ct_dir.rglob("*.nii.gz")
            if patient_id_from_name(p.name) == patient_id
        ]
    return sorted(candidates)[0] if candidates else None


def dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    aa, bb = a.astype(bool), b.astype(bool)
    return float((2.0 * np.logical_and(aa, bb).sum() + eps) / (aa.sum() + bb.sum() + eps))


def surface_metrics(a: np.ndarray, b: np.ndarray, spacing_zyx: Tuple[float, float, float]):
    from surface_distance import metrics
    aa, bb = a.astype(bool), b.astype(bool)
    if not aa.any() and not bb.any():
        return 0.0, 0.0
    if not aa.any() or not bb.any():
        return float("inf"), float("inf")
    distances = metrics.compute_surface_distances(aa, bb, spacing_zyx)
    hd95 = float(metrics.compute_robust_hausdorff(distances, 95.0))
    assd = float(np.mean(metrics.compute_average_surface_distance(distances)))
    return hd95, assd


def save_like(array, reference, path, pixel_id=sitk.sitkUInt8):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    image = sitk.Cast(image, pixel_id)
    sitk.WriteImage(image, str(path), True)


def resolve_checkpoint(cfg, cli_checkpoint: Optional[str]) -> Path:
    if cli_checkpoint:
        return Path(cli_checkpoint).expanduser().resolve()
    configured = cfg.get("paths", {}).get("trained_decoder_checkpoint")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(cfg["output_root"]) / cfg["experiment_name"] / "checkpoints/best_decoder.pth").resolve()


def resolve_output_dir(cfg, split: str, cli_output: Optional[str]) -> Path:
    if cli_output:
        return Path(cli_output).expanduser().resolve()
    base = cfg.get("evaluation", {}).get("output_root")
    if base:
        return Path(base).expanduser().resolve() / f"clean_visualization_{split}"
    return Path(cfg["output_root"]) / cfg["experiment_name"] / f"clean_visualization_{split}"


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    label_dir = Path(cfg["data"][f"{args.split}_label_dir"]).expanduser()
    ct_dir = Path(cfg["data"][f"{args.split}_ct_dir"]).expanduser()
    vis_dir = resolve_output_dir(cfg, args.split, args.output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_dino_shape_model(cfg, checkpoint, device)
    threshold = float(
        args.threshold if args.threshold is not None
        else cfg.get("evaluation", {}).get("threshold", 0.5)
    )
    spacing = tuple(float(v) for v in cfg.get("evaluation", {}).get("spacing_zyx", [1.5, 1.5, 1.5]))

    rows = []
    for label_path in sorted(label_dir.rglob("*.nii.gz")):
        pid = patient_id_from_name(label_path.name)
        label_img = sitk.ReadImage(str(label_path))
        clean = (sitk.GetArrayFromImage(label_img) > 0).astype(np.uint8)
        tensor = torch.from_numpy(clean.astype(np.float32))[None, None].to(device)
        with torch.inference_mode():
            recon_t, prob_t, _ = model.reconstruct(tensor, threshold, return_probability=True)
        prob = prob_t[0, 0].float().cpu().numpy()
        recon = recon_t[0, 0].byte().cpu().numpy()

        save_like(clean, label_img, vis_dir / f"clean_input_patient_{pid}.nii.gz")
        save_like(recon, label_img, vis_dir / f"reconstruction_clean_patient_{pid}.nii.gz")
        save_like(prob.astype(np.float32), label_img, vis_dir / f"reconstruction_probability_clean_patient_{pid}.nii.gz", sitk.sitkFloat32)

        ct_path = find_ct(ct_dir, pid)
        if ct_path:
            shutil.copy2(ct_path, vis_dir / f"original_ct_patient_{pid}.nii.gz")

        hd95, assd = surface_metrics(clean, recon, spacing)
        clean_voxels, recon_voxels = int(clean.sum()), int(recon.sum())
        row = {
            "patient_id": pid,
            "dice_clean_recon": dice(clean, recon),
            "hd95_clean_recon_mm": hd95,
            "assd_clean_recon_mm": assd,
            "clean_voxels": clean_voxels,
            "recon_voxels": recon_voxels,
            "volume_diff_fraction": abs(recon_voxels - clean_voxels) / max(clean_voxels, 1),
            "ct_found": bool(ct_path),
        }
        rows.append(row)
        print(row)

    if not rows:
        raise RuntimeError(f"No NIfTI labels found under {label_dir}")
    metrics_path = vis_dir.parent / f"clean_reconstruction_metrics_{args.split}.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved clean visualizations to: {vis_dir}")
    print(f"Saved clean metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
''')

# ---------------------------------------------------------------------
# Full DINO severity-sweep evaluator. Build it from the canonical ASSD script,
# preserving corruption and metric implementations and replacing only the
# CNN-specific model/data/config sections.
# ---------------------------------------------------------------------
canonical = root / "evaluation/probes/eval_post_dae_probe_sitk_morph_sweep_assd.py"
out = root / "evaluation/probes/eval_dino_shape_probe_sweep_assd.py"
src = canonical.read_text()

src = src.replace(
    "Severity-sweep probe evaluation for vanilla AE shape models.",
    "Severity-sweep probe evaluation for a frozen-3DINO shape autoencoder.",
    1,
)
src = src.replace("from torch.utils.data import DataLoader", "from torch.utils.data import DataLoader, Dataset")
src = src.replace(
    "from dataset.dataset_generator import ProjectionDataset\n"
    "from dataset.pre_processing import compose_transforms\n"
    "from network.models import VAE\n"
    "from network.training import apply_label_mask, NormalizeImage\n",
    "from models.dino_shape.dino_shape_autoencoder import load_trained_dino_shape_model\n",
)

start = src.index("def load_config(config_path: str) -> Dict:")
end = src.index("# -----------------------------\n# Metrics", start)
replacement = r'''def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_patient_id(path: str) -> str:
    fname = os.path.basename(path)
    match = re.match(r"(?:label|labels|ct)_patient_?(.+)\.nii\.gz$", fname, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    stem = fname[:-7] if fname.endswith(".nii.gz") else Path(fname).stem
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else stem


def infer_fov_name(cf: Dict) -> str:
    configured = cf.get("evaluation", {}).get("fov_name")
    if configured:
        return str(configured)
    return "native_roi_crop"


def resolve_checkpoint(cf: Dict) -> Path:
    configured = cf.get("paths", {}).get("trained_decoder_checkpoint")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(cf["output_root"]) / cf["experiment_name"] / "checkpoints/best_decoder.pth").resolve()


def make_model(cf: Dict, device: torch.device) -> torch.nn.Module:
    checkpoint = resolve_checkpoint(cf)
    print(f"> Loading trained DINO decoder: {checkpoint}")
    return load_trained_dino_shape_model(cf, checkpoint, device)


class BinaryROIDataset(Dataset):
    def __init__(self, label_dir: str | Path) -> None:
        self.label_dir = Path(label_dir).expanduser().resolve()
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        self.files = sorted(self.label_dir.rglob("*.nii.gz"))
        if not self.files:
            raise RuntimeError(f"No NIfTI files found under {self.label_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        image = sitk.ReadImage(str(path))
        array = (sitk.GetArrayFromImage(image) > 0).astype(np.float32)
        return {
            "target_image": torch.from_numpy(array[None]),
            "proj_image": torch.from_numpy(array[None]),
            "proj_name": str(path),
        }


def get_eval_loader(cf: Dict, split: str = "test") -> DataLoader:
    split_alias = "validation" if split in {"valid", "validation"} else split
    key = f"{split_alias}_label_dir"
    if key not in cf["data"]:
        raise KeyError(f"Config data section has no {key}")
    dataset = BinaryROIDataset(cf["data"][key])
    workers = int(cf.get("evaluation", {}).get("num_workers", 2))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def preprocess_clean_target(x_target: torch.Tensor, cf: Dict) -> torch.Tensor:
    return (x_target > 0.5).float()


@torch.no_grad()
def reconstruct_binary(model: torch.nn.Module, x_in: torch.Tensor, cf: Dict, threshold: float) -> torch.Tensor:
    return model.reconstruct(x_in, threshold=float(threshold))


'''
src = src[:start] + replacement + src[end:]

src = src.replace(
    'p = argparse.ArgumentParser(description="Severity-sweep probe evaluation for vanilla AE shape models")',
    'p = argparse.ArgumentParser(description="Severity-sweep probe evaluation for frozen 3DINO shape models")',
)
src = src.replace(
    'p.add_argument("-c", "--config", required=True, help="VariantVAE YAML config")',
    'p.add_argument("-c", "--config", required=True, help="DINO shape evaluation YAML config")',
)
src = src.replace(
    'choices=["train", "valid", "test"]',
    'choices=["train", "valid", "validation", "test"]',
)
old_main_cfg = '''    cf = resolve_run_paths(load_config(args.config))
    if "ori_size" not in cf:
        cf["ori_size"] = cf["proj_size"][::-1]
    device = torch.device(cf.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    cf["device"] = device

    out_dir = Path(cf["output_folder"]) / args.out_subdir
'''
new_main_cfg = '''    cf = load_config(args.config)
    device_name = cf.get("evaluation", {}).get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cf["device"] = str(device)
    output_root = Path(cf.get("evaluation", {}).get(
        "output_root",
        Path(cf["output_root"]) / cf["experiment_name"],
    )).expanduser().resolve()
    out_dir = output_root / args.out_subdir
'''
if old_main_cfg not in src:
    raise RuntimeError("Could not patch canonical evaluator main config block")
src = src.replace(old_main_cfg, new_main_cfg, 1)
src = src.replace(
    'spacing = tuple(float(x) for x in cf.get("proj_spacing", [1.0, 1.0, 1.0]))',
    'spacing = tuple(float(x) for x in cf.get("evaluation", {}).get("spacing_zyx", [1.5, 1.5, 1.5]))',
)
src = src.replace('"exp_name": cf["exp_name"],', '"exp_name": cf.get("experiment_name", "dino_shape"),')
out.write_text(src)
out.chmod(0o755)

# ---------------------------------------------------------------------
# Evaluation config. Historical training output is referenced, while all new
# evaluation products go to the dedicated contour-QA scratch tree.
# ---------------------------------------------------------------------
eval_cfg = root / "configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml"
eval_cfg.write_text('''experiment_name: dino_shape_pilot_Lung_L
output_root: /scratch/shaohan/3DINO
seed: 88

paths:
  trained_decoder_checkpoint: /scratch/shaohan/3DINO/dino_shape_pilot_Lung_L/checkpoints/best_decoder.pth

dino:
  repo_path: /home/shaohan/3DINO
  checkpoint_path: /home/shaohan/3DINO/pretrained/3dino_vit_weights.pth
  config_name: train/vit3d_highres

data:
  roi_name: Lung_L
  train_label_dir: /home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom/lungs_chest/Lung_L/training1/lungs_chest/Lung_L/labels
  validation_label_dir: /home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom/lungs_chest/Lung_L/validation1/lungs_chest/Lung_L/labels
  test_label_dir: /home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom/lungs_chest/Lung_L/testing1/lungs_chest/Lung_L/labels
  train_ct_dir: /home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom/lungs_chest/Lung_L/training1/lungs_chest/Lung_L/ct
  validation_ct_dir: /home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom/lungs_chest/Lung_L/validation1/lungs_chest/Lung_L/ct
  test_ct_dir: /home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom/lungs_chest/Lung_L/testing1/lungs_chest/Lung_L/ct

decoder:
  channels: [256, 128, 64, 32]

evaluation:
  threshold: 0.5
  spacing_zyx: [1.5, 1.5, 1.5]
  fov_name: Lung_L_roiwise_1p5mm_nozoom
  device: cuda:0
  num_workers: 2
  output_root: /scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L
''')

# ---------------------------------------------------------------------
# CWD-independent runner scripts.
# ---------------------------------------------------------------------
scripts = root / "scripts/dino_shape"
scripts.mkdir(parents=True, exist_ok=True)

(scripts / "run_dino_shape_clean_eval_Lung_L.sh").write_text(r'''#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
python -m evaluation.eval_dino_shape_clean --config "$CONFIG" --split test
''')

(scripts / "run_dino_shape_probe_sweep_Lung_L.sh").write_text(r'''#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
python -m evaluation.probes.eval_dino_shape_probe_sweep_assd \
  --config "$CONFIG" \
  --split test \
  --out_subdir probe_eval_sweep \
  --threshold 0.5 \
  --sample_ratio 1.0 \
  --seed 88 \
  --save_vis \
  --vis_max_cases 3 \
  --drop_inf_in_plots
''')

(scripts / "run_dino_shape_analysis_Lung_L.sh").write_text(r'''#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
RESULT_DIR="${RESULT_DIR:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_sweep}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo "Probe results are expected at: $RESULT_DIR"
echo "Run the existing AUC/FDVC scripts after confirming their CLI arguments against this CSV:"
echo "  $RESULT_DIR/probe_eval_sweep_detailed.csv"
''')

for p in scripts.glob("run_dino_shape_*_Lung_L.sh"):
    p.chmod(0o755)

# Repair historical pilot runner only enough to use project-root modules.
pilot_runner = scripts / "run_dino_shape_pilot_Lung_L.sh"
if pilot_runner.exists():
    t = pilot_runner.read_text()
    t = t.replace("REPO=/home/shaohan/3DINO", "PROJECT_ROOT=/home/shaohan/contour_qa_experiments\nDINO_REPO=/home/shaohan/3DINO")
    t = t.replace('cd "${REPO}"', 'cd "${PROJECT_ROOT}"\nexport PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"')
    t = t.replace("python train_dino_shape_decoder.py", "python -m training.train_dino_shape_decoder")
    t = t.replace("python eval_dino_shape_clean.py", "python -m evaluation.eval_dino_shape_clean")
    t = t.replace("python eval_dino_shape_probes_visual.py", "python -m evaluation.probes.eval_dino_shape_probes_visual")
    pilot_runner.write_text(t)
    pilot_runner.chmod(0o755)

# Pilot visual evaluator import repair.
visual = root / "evaluation/probes/eval_dino_shape_probes_visual.py"
if visual.exists():
    t = visual.read_text().replace(
        "from dino_shape_autoencoder import Frozen3DINOShapeAutoencoder",
        "from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder",
    )
    visual.write_text(t)

# Record repair inventory.
manifest = root / "manifests/immediate_repairs.txt"
manifest.write_text("""Immediate repairs applied:
- package __init__.py files
- shared trained-DINO model loader and reconstruct() API
- package-safe training import
- repaired clean evaluator with Dice, HD95, ASSD, and volume difference
- full DINO severity-sweep evaluator based on canonical ASSD probe script
- Lung_L evaluation config referencing existing best_decoder.pth
- clean/probe/analysis runner scripts
- package-safe pilot visual evaluator import
""")

print(f"Repairs written under: {root}")
PY

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Syntax validation does not import heavy dependencies or load checkpoints.
"$PYTHON_BIN" -m py_compile \
  models/dino_shape/dino_shape_autoencoder.py \
  training/train_dino_shape_decoder.py \
  evaluation/eval_dino_shape_clean.py \
  evaluation/probes/eval_dino_shape_probe_sweep_assd.py \
  evaluation/probes/eval_dino_shape_probes_visual.py

echo "Immediate repairs completed and Python syntax checks passed."
echo "Next smoke test:"
echo "  bash scripts/dino_shape/run_dino_shape_clean_eval_Lung_L.sh"
