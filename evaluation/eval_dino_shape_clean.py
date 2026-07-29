#!/usr/bin/env python3
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
