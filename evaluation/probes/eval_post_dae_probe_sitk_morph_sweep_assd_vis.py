#!/usr/bin/env python3
"""
Severity-sweep probe evaluation for vanilla AE shape models.

This is a drop-in replacement/extension for eval_post_dae_probe_sitk_morph.py.
It keeps the same project conventions:
  - config is a VariantVAE YAML
  - model='ae'
  - trained run path follows main.py naming:
      {exp_name}_alpha{alpha}_kld{kld_weight}_wm{weight_mode}
  - evaluation output is written under:
      {output_folder}/probe_eval_sweep/

New behavior
------------
Instead of testing one corruption setting, this script sweeps corruption severity:
  cut_upper_half / cut_lower_half / cut_posterior_half:
      cut ratio from 0.0 to 0.5 by default
  big_blob / big_bite:
      local box fraction from 0.0 to 0.5 by default, no fixed min block
  shifted / eroded / dilated:
      integer voxels/iterations from 0 to 5 by default

For each probe type and severity level, the script computes:
  - Dice(input, target)
  - Dice(input, recon)
  - Dice(recon, target)
  - HD95(input, target)
  - HD95(input, recon)
  - HD95(recon, target)
  - ASSD(input, target)
  - ASSD(input, recon)
  - ASSD(recon, target)

It also generates boxplots with x-axis = corruption severity level.
If --save_vis is given, it saves input-corrupted/recon/target NIfTI examples
and 3-plane PNG panels for each probe type/severity level.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.dataset_generator import ProjectionDataset
from dataset.pre_processing import compose_transforms
from network.models import VAE
from network.training import apply_label_mask, NormalizeImage


# -----------------------------
# Config / run-path utilities
# -----------------------------

def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _single_value(x, name: str):
    if isinstance(x, list):
        if len(x) != 1:
            raise ValueError(f"Expected a single {name} value for evaluation, got {x}")
        return x[0]
    return x


def resolve_run_paths(cf: Dict) -> Dict:
    """Reproduce main.py run_experiment naming logic."""
    cf = copy.deepcopy(cf)
    base_exp_name = cf["exp_name"]
    alpha = _single_value(cf.get("alpha", None), "alpha")
    kld_weight = _single_value(cf.get("kld_weight", None), "kld_weight")
    weight_mode = _single_value(cf.get("weight_mode", "uniform"), "weight_mode")
    cf["alpha"] = alpha
    cf["kld_weight"] = kld_weight
    cf["weight_mode"] = weight_mode
    cf["exp_name"] = f"{base_exp_name}_alpha{alpha}_kld{kld_weight}_wm{weight_mode}"
    cf["output_folder"] = os.path.join(cf["save_dir"], cf["exp_name"])
    return cf


def extract_patient_id(path: str) -> str:
    fname = os.path.basename(path)
    m = re.match(r"(?:labels|ct)_patient_(.+)\.nii\.gz$", fname)
    if m is None:
        return Path(fname).name.replace(".nii.gz", "")
    return m.group(1)


def infer_fov_name(cf: Dict) -> str:
    sz = cf.get("proj_size", cf.get("ori_size", None))
    if sz is None:
        return "unknown_fov"
    return "x".join(str(int(v)) for v in sz)


# -----------------------------
# Model / data
# -----------------------------

def make_model(cf: Dict, device: torch.device) -> torch.nn.Module:
    if cf.get("model") != "ae":
        raise ValueError(f"This script is intended for model='ae', got {cf.get('model')}")
    model = VAE(cf, embedding=False).to(device)
    ckpt = os.path.join(cf["output_folder"], "checkpoints", cf["ckpt_file_load"])
    print(f"> Loading weights: {ckpt}")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def get_eval_loader(cf: Dict, split: str = "test") -> DataLoader:
    eval_cf = copy.deepcopy(cf)
    eval_cf["train"] = False
    pre_processing = compose_transforms(eval_cf)
    key = f"{split}_dir"
    if key not in cf:
        raise KeyError(f"Config has no {key}")
    ds = ProjectionDataset(cf[key], transform=pre_processing, cf=eval_cf)
    return DataLoader(
        ds,
        batch_size=cf.get("batch_size", 1),
        shuffle=False,
        num_workers=min(os.cpu_count() or 1, 8),
        pin_memory=True,
    )


def preprocess_clean_target(x_target: torch.Tensor, cf: Dict) -> torch.Tensor:
    x_target = NormalizeImage(x_target, cf)
    x_target = apply_label_mask(x_target, cf)
    return (x_target == 1).float()


@torch.no_grad()
def reconstruct_binary(model: torch.nn.Module, x_in: torch.Tensor, cf: Dict, threshold: float) -> torch.Tensor:
    x_hat, _, _, _ = model(x_in, deterministic=True)
    # model output can be slightly larger in some configs; crop to target spatial size
    target_size = tuple(x_in.shape[2:])
    if tuple(x_hat.shape[2:]) != target_size:
        x_hat = x_hat[:, :, : target_size[0], : target_size[1], : target_size[2]]
    if x_hat.shape[1] == 1:
        return (torch.sigmoid(x_hat) > threshold).float()
    # defensive fallback for multiclass
    return (torch.argmax(x_hat, dim=1, keepdim=True) > 0).float()


# -----------------------------
# Metrics
# -----------------------------

def dice_score_binary(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (a * b).sum(dim=(1, 2, 3, 4))
    den = a.sum(dim=(1, 2, 3, 4)) + b.sum(dim=(1, 2, 3, 4)) + eps
    return (2.0 * inter + eps) / den


def hd95_binary(a: torch.Tensor, b: torch.Tensor, spacing: Tuple[float, float, float]) -> torch.Tensor:
    from surface_distance import metrics

    aa = a.detach().cpu().numpy().astype(bool)
    bb = b.detach().cpu().numpy().astype(bool)
    vals: List[float] = []
    for i in range(aa.shape[0]):
        m1 = aa[i, 0]
        m2 = bb[i, 0]
        if not m1.any() and not m2.any():
            vals.append(0.0)
        elif not m1.any() or not m2.any():
            vals.append(np.inf)
        else:
            try:
                surf = metrics.compute_surface_distances(m1, m2, spacing)
                vals.append(float(metrics.compute_robust_hausdorff(surf, 95.0)))
            except Exception as e:
                print(f"HD95 failed: {e}")
                vals.append(np.inf)
    return torch.tensor(vals, dtype=torch.float32)


def assd_binary(a: torch.Tensor, b: torch.Tensor, spacing: Tuple[float, float, float]) -> torch.Tensor:
    """
    Average symmetric surface distance, in mm.

    Returns:
      0.0 if both masks are empty
      inf if exactly one mask is empty
    """
    from surface_distance import metrics

    aa = a.detach().cpu().numpy().astype(bool)
    bb = b.detach().cpu().numpy().astype(bool)
    vals: List[float] = []
    for i in range(aa.shape[0]):
        m1 = aa[i, 0]
        m2 = bb[i, 0]
        if not m1.any() and not m2.any():
            vals.append(0.0)
        elif not m1.any() or not m2.any():
            vals.append(np.inf)
        else:
            try:
                surf = metrics.compute_surface_distances(m1, m2, spacing)
                avg = metrics.compute_average_surface_distance(surf)
                vals.append(float(np.mean(avg)))
            except Exception as e:
                print(f"ASSD failed: {e}")
                vals.append(np.inf)
    return torch.tensor(vals, dtype=torch.float32)


def volume_vox(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().sum(dim=(1, 2, 3, 4))


# -----------------------------
# SITK morphology
# -----------------------------

def _sitk_from_binary_tensor_1(x: torch.Tensor) -> sitk.Image:
    arr = (x.detach().cpu().numpy()[0] > 0.5).astype(np.uint8)  # [D,H,W]
    return sitk.GetImageFromArray(arr)


def _tensor_from_sitk_binary(img: sitk.Image, device: torch.device) -> torch.Tensor:
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.float32)
    return torch.from_numpy(arr[None]).to(device)


def sitk_binary_erode_adaptive(x: torch.Tensor, iterations: int, kernel_radius: int = 1) -> Tuple[torch.Tensor, int, bool]:
    """Erode, decreasing iterations if erosion removes all voxels."""
    x = (x > 0.5).float()
    if iterations <= 0 or x.sum().item() == 0:
        return x, 0, False

    for it in range(iterations, -1, -1):
        if it == 0:
            return x, 0, True
        img = _sitk_from_binary_tensor_1(x)
        out = img
        filt = sitk.BinaryErodeImageFilter()
        filt.SetKernelType(sitk.sitkBall)
        filt.SetKernelRadius([kernel_radius] * 3)
        filt.SetForegroundValue(1)
        filt.SetBackgroundValue(0)
        filt.SetBoundaryToForeground(False)
        for _ in range(it):
            out = filt.Execute(out)
        y = _tensor_from_sitk_binary(out, x.device)
        if y.sum().item() > 0:
            return y.float(), it, (it != iterations)
    return x, 0, True


def sitk_binary_dilate(x: torch.Tensor, iterations: int, kernel_radius: int = 1) -> torch.Tensor:
    x = (x > 0.5).float()
    if iterations <= 0 or x.sum().item() == 0:
        return x
    img = _sitk_from_binary_tensor_1(x)
    out = img
    filt = sitk.BinaryDilateImageFilter()
    filt.SetKernelType(sitk.sitkBall)
    filt.SetKernelRadius([kernel_radius] * 3)
    filt.SetForegroundValue(1)
    filt.SetBackgroundValue(0)
    filt.SetBoundaryToForeground(False)
    for _ in range(iterations):
        out = filt.Execute(out)
    return _tensor_from_sitk_binary(out, x.device).float()


# -----------------------------
# Corruptions
# -----------------------------

def foreground_bbox(x: torch.Tensor):
    fg = (x[0] > 0.5).nonzero(as_tuple=False)
    if fg.numel() == 0:
        return None, None, None
    mins = fg.min(dim=0).values
    maxs = fg.max(dim=0).values
    center = fg.float().mean(dim=0)
    return center, mins, maxs


def cut_by_ratio(x: torch.Tensor, axis: int, side: str, ratio: float) -> torch.Tensor:
    """
    Remove a fraction of the foreground bounding-box extent.
    axis: 0=z, 1=y, 2=x in tensor spatial coordinates.
    side='low' removes lower indices; side='high' removes higher indices.
    """
    ratio = float(ratio)
    y = x.clone()
    if ratio <= 0:
        return y
    _, mins, maxs = foreground_bbox(y)
    if mins is None:
        return y
    span = int((maxs[axis] - mins[axis] + 1).item())
    n_cut = int(round(span * ratio))
    if n_cut <= 0:
        return y
    lo = int(mins[axis].item())
    hi = int(maxs[axis].item())
    if side == "low":
        end = min(hi + 1, lo + n_cut)
        if axis == 0:
            y[:, lo:end, :, :] = 0
        elif axis == 1:
            y[:, :, lo:end, :] = 0
        else:
            y[:, :, :, lo:end] = 0
    elif side == "high":
        start = max(lo, hi + 1 - n_cut)
        if axis == 0:
            y[:, start : hi + 1, :, :] = 0
        elif axis == 1:
            y[:, :, start : hi + 1, :] = 0
        else:
            y[:, :, :, start : hi + 1] = 0
    else:
        raise ValueError(side)
    return y


def random_shift_fixed(x: torch.Tensor, voxels: int, rng: random.Random) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Shift by exactly `voxels` in a random signed axis-combination direction."""
    x = (x > 0.5).float()
    if voxels <= 0:
        return x, (0, 0, 0)
    # Random direction vector with at least one nonzero component; scale each nonzero component by voxels.
    choices = [-1, 0, 1]
    dz = dy = dx = 0
    while dz == dy == dx == 0:
        dz, dy, dx = (rng.choice(choices), rng.choice(choices), rng.choice(choices))
    dz, dy, dx = dz * voxels, dy * voxels, dx * voxels

    out = torch.zeros_like(x)
    _, D, H, W = x.shape
    z_src0, z_src1 = max(0, -dz), min(D, D - dz) if dz >= 0 else D
    y_src0, y_src1 = max(0, -dy), min(H, H - dy) if dy >= 0 else H
    x_src0, x_src1 = max(0, -dx), min(W, W - dx) if dx >= 0 else W
    z_dst0, y_dst0, x_dst0 = max(0, dz), max(0, dy), max(0, dx)
    z_dst1 = z_dst0 + (z_src1 - z_src0)
    y_dst1 = y_dst0 + (y_src1 - y_src0)
    x_dst1 = x_dst0 + (x_src1 - x_src0)
    if z_src1 > z_src0 and y_src1 > y_src0 and x_src1 > x_src0:
        out[:, z_dst0:z_dst1, y_dst0:y_dst1, x_dst0:x_dst1] = x[:, z_src0:z_src1, y_src0:y_src1, x_src0:x_src1]
    return out, (dz, dy, dx)


def surface_voxels(x: torch.Tensor) -> Optional[torch.Tensor]:
    x = (x > 0.5).float()
    if x.sum().item() == 0:
        return None
    eroded, _, _ = sitk_binary_erode_adaptive(x, iterations=1, kernel_radius=1)
    surf = (x > 0.5) & ~(eroded > 0.5)
    coords = surf[0].nonzero(as_tuple=False)
    if coords.numel() == 0:
        coords = x[0].nonzero(as_tuple=False)
    return coords


def choose_surface_anchor(x: torch.Tensor, rng: random.Random) -> Optional[Tuple[int, int, int]]:
    coords = surface_voxels(x)
    if coords is None or coords.numel() == 0:
        return None
    idx = rng.randrange(coords.shape[0])
    return tuple(int(v) for v in coords[idx].tolist())


def connected_components_touching_original(merged_mask: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    merged_mask = merged_mask.astype(bool)
    original_mask = original_mask.astype(bool)
    struct = ndimage.generate_binary_structure(3, 1)
    lab, nlab = ndimage.label(merged_mask, structure=struct)
    if nlab == 0:
        return merged_mask
    touching = np.unique(lab[original_mask])
    touching = touching[touching != 0]
    if len(touching) == 0:
        return merged_mask
    return np.isin(lab, touching)


def local_box_bite_or_blob(
    x: torch.Tensor,
    frac: float,
    rng: random.Random,
    mode: str,
    inward_fraction: float = 0.35,
    keep_connected: bool = True,
) -> torch.Tensor:
    """Local corruption with box size = frac * foreground bbox; no min block."""
    frac = float(frac)
    y = x.clone()
    if frac <= 0:
        return y
    center, mins, maxs = foreground_bbox(y)
    if center is None:
        return y
    spans = (maxs - mins + 1).float()
    dz = int(round(spans[0].item() * frac))
    dy = int(round(spans[1].item() * frac))
    dx = int(round(spans[2].item() * frac))
    if dz <= 0 or dy <= 0 or dx <= 0:
        return y

    anchor = choose_surface_anchor(y, rng)
    if anchor is None:
        return y
    anchor_t = torch.tensor(anchor, dtype=torch.float32, device=center.device)

    if mode == "blob":
        direction = center - anchor_t
        direction = direction / (torch.norm(direction) + 1e-6)
        shift = torch.tensor(
            [inward_fraction * dz * direction[0], inward_fraction * dy * direction[1], inward_fraction * dx * direction[2]],
            dtype=torch.float32,
            device=center.device,
        )
        box_center = anchor_t + shift
    elif mode == "bite":
        box_center = anchor_t
    else:
        raise ValueError(mode)

    zc, yc, xc = [int(round(v)) for v in box_center.detach().cpu().tolist()]
    z0, y0, x0 = max(0, zc - dz // 2), max(0, yc - dy // 2), max(0, xc - dx // 2)
    z1, y1, x1 = min(y.shape[1], z0 + dz), min(y.shape[2], y0 + dy), min(y.shape[3], x0 + dx)

    if mode == "bite":
        y[:, z0:z1, y0:y1, x0:x1] = 0
        return y

    original = (x > 0.5).float()
    y[:, z0:z1, y0:y1, x0:x1] = 1
    if keep_connected:
        merged = (y.detach().cpu().numpy()[0] > 0.5)
        orig = (original.detach().cpu().numpy()[0] > 0.5)
        kept = connected_components_touching_original(merged, orig)
        y = torch.from_numpy(kept.astype(np.float32)[None]).to(x.device)
    return y.float()


def make_corruption(x: torch.Tensor, probe_type: str, level: float, rng: random.Random) -> Tuple[torch.Tensor, Dict]:
    meta: Dict = {"requested_level": level, "applied_level": level, "adapted": False}
    if probe_type == "clean":
        return x.clone(), meta
    if probe_type == "shifted":
        y, shift = random_shift_fixed(x, int(level), rng)
        meta["shift_zyx"] = json.dumps(list(shift))
        return y, meta
    if probe_type == "eroded":
        y, applied, adapted = sitk_binary_erode_adaptive(x, int(level), kernel_radius=1)
        meta["applied_level"] = applied
        meta["adapted"] = adapted
        return y, meta
    if probe_type == "dilated":
        return sitk_binary_dilate(x, int(level), kernel_radius=1), meta
    if probe_type == "cut_lower_half":
        return cut_by_ratio(x, axis=0, side="high", ratio=float(level)), meta
    if probe_type == "cut_upper_half":
        return cut_by_ratio(x, axis=0, side="low", ratio=float(level)), meta
    if probe_type == "cut_posterior_half":
        return cut_by_ratio(x, axis=1, side="high", ratio=float(level)), meta
    if probe_type == "big_bite":
        return local_box_bite_or_blob(x, frac=float(level), rng=rng, mode="bite"), meta
    if probe_type == "big_blob":
        return local_box_bite_or_blob(x, frac=float(level), rng=rng, mode="blob"), meta
    raise ValueError(f"Unknown probe type: {probe_type}")


# -----------------------------
# Plotting
# -----------------------------

PREFERRED_PROBE_ORDER = [
    "clean", "shifted", "eroded", "dilated", "cut_lower_half", "cut_upper_half", "cut_posterior_half", "big_bite", "big_blob",
]


def plot_box_by_level(df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str, drop_inf: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    if drop_inf:
        work = work[np.isfinite(work[metric])].copy()

    for probe in PREFERRED_PROBE_ORDER:
        sub = work[work["probe_type"] == probe].copy()
        if sub.empty:
            continue
        levels = sorted(sub["level"].dropna().unique(), key=lambda v: float(v))
        data = [sub[sub["level"] == lev][metric].dropna().values for lev in levels]
        if not any(len(d) for d in data):
            continue
        labels = [str(int(l)) if float(l).is_integer() else f"{float(l):.2f}" for l in levels]
        plt.figure(figsize=(max(7.5, 0.75 * len(levels)), 5.5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xlabel("Corruption level", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{probe}: {ylabel} by corruption level", fontsize=14)
        if "Dice" in ylabel:
            plt.ylim(0.0, 1.0)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{probe}_{metric}_box_by_level.png", dpi=300)
        plt.close()


def plot_summary_trends(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in [
        ("dice_input_vs_target_mean", "Input-Target Dice"),
        ("dice_input_vs_recon_mean", "Input-Recon Dice"),
        ("dice_recon_vs_target_mean", "Recon-Target Dice"),
        ("hd95_input_vs_target_mean", "Input-Target HD95"),
        ("hd95_input_vs_recon_mean", "Input-Recon HD95"),
        ("hd95_recon_vs_target_mean", "Recon-Target HD95"),
    ]:
        if metric not in summary.columns:
            continue
        plt.figure(figsize=(9, 5.5))
        for probe in PREFERRED_PROBE_ORDER:
            sub = summary[summary["probe_type"] == probe].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("level")
            plt.plot(sub["level"], sub[metric], marker="o", linewidth=1.8, label=probe)
        plt.xlabel("Corruption level", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"Severity sweep: {ylabel}", fontsize=14)
        if "Dice" in ylabel:
            plt.ylim(0.0, 1.0)
        plt.grid(axis="y", alpha=0.3)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"severity_trend_{metric}.png", dpi=300)
        plt.close()


# -----------------------------
# Main evaluation
# -----------------------------

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def select_case_indices(n: int, ratio: float, seed: int) -> set:
    if ratio >= 1.0:
        return set(range(n))
    rng = random.Random(seed)
    keep = max(1, min(n, int(round(n * ratio))))
    return set(rng.sample(list(range(n)), keep))


def save_volume_like(mask: torch.Tensor, out_path: Path, spacing: Sequence[float]) -> None:
    arr = (mask.detach().cpu().numpy()[0] > 0.5).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    # SimpleITK spacing is x,y,z; config spacing is treated as z,y,x by surface_distance code historically.
    sp = tuple(float(v) for v in spacing)
    if len(sp) == 3:
        img.SetSpacing((sp[2], sp[1], sp[0]))
    sitk.WriteImage(img, str(out_path), True)


def _mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    # Accept [1,D,H,W] or [1,1,D,H,W]; return [D,H,W] uint8.
    arr = mask.detach().cpu().numpy()
    arr = np.squeeze(arr)
    return (arr > 0.5).astype(np.uint8)


def _best_slice_indices(arrs: Sequence[np.ndarray]) -> Tuple[int, int, int]:
    """Pick axial/coronal/sagittal slices with most foreground across all masks."""
    union = np.zeros_like(arrs[0], dtype=bool)
    for a in arrs:
        union |= (a > 0)
    if not union.any():
        z, y, x = np.array(union.shape) // 2
        return int(z), int(y), int(x)
    z_scores = union.sum(axis=(1, 2))
    y_scores = union.sum(axis=(0, 2))
    x_scores = union.sum(axis=(0, 1))
    return int(np.argmax(z_scores)), int(np.argmax(y_scores)), int(np.argmax(x_scores))


def save_triplet_png(clean: torch.Tensor, corrupted: torch.Tensor, recon: torch.Tensor, out_path: Path, title: str = "") -> None:
    """Save a compact 3-plane visualization of input/recon/target masks.

    Rows are axial/coronal/sagittal. Columns are input-corrupted, recon, target.
    The selected slices are chosen from the union foreground, not merely image center.
    """
    target_np = _mask_to_numpy(clean)
    input_np = _mask_to_numpy(corrupted)
    recon_np = _mask_to_numpy(recon)
    z, y, x = _best_slice_indices([target_np, input_np, recon_np])

    panels = [
        ("Input corrupted", input_np),
        ("Recon", recon_np),
        ("Target clean", target_np),
    ]
    row_specs = [
        ("Axial", lambda a: a[z, :, :]),
        ("Coronal", lambda a: a[:, y, :]),
        ("Sagittal", lambda a: a[:, :, x]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for r, (row_name, slicer) in enumerate(row_specs):
        for c, (col_name, arr) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(slicer(arr), cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_name, fontsize=11)
            if c == 0:
                ax.set_ylabel(row_name, fontsize=11)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_visual_example(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    recon: torch.Tensor,
    vis_dir: Path,
    spacing: Sequence[float],
    metadata: Dict,
) -> None:
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_volume_like(clean, vis_dir / "target_clean.nii.gz", spacing)
    save_volume_like(corrupted, vis_dir / "input_corrupted.nii.gz", spacing)
    save_volume_like(recon, vis_dir / "recon.nii.gz", spacing)
    with open(vis_dir / "metrics_and_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    title = f"{metadata.get('patient_id', '')} | {metadata.get('probe_type', '')} | level={metadata.get('level', '')}"
    save_triplet_png(clean, corrupted, recon, vis_dir / "input_recon_target_3plane.png", title=title)


def main() -> None:
    p = argparse.ArgumentParser(description="Severity-sweep probe evaluation for vanilla AE shape models")
    p.add_argument("-c", "--config", required=True, help="VariantVAE YAML config")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"], help="Dataset split to evaluate")
    p.add_argument("--out_subdir", default="probe_eval_sweep", help="Output subdir under experiment folder")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sample_ratio", type=float, default=1.0, help="Evaluate a random subset of cases")
    p.add_argument("--seed", type=int, default=88)

    p.add_argument("--cut_levels", default="0,0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--local_frac_levels", default="0,0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--voxel_levels", default="0,1,2,3,4,5")
    p.add_argument("--probe_types", nargs="*", default=PREFERRED_PROBE_ORDER)
    p.add_argument("--save_vis", action="store_true", help="Save visual examples under probe_eval_sweep/visual_examples")
    p.add_argument("--vis_max_cases", type=int, default=3, help="Legacy cap for old first-N-case visualization behavior")
    p.add_argument("--vis_max_cases_per_probe_level", type=int, default=1,
                   help="Number of example cases to save for each probe type and severity level")
    p.add_argument("--vis_save_legacy_first_cases", action="store_true",
                   help="Also save the old first-N-case visualization layout under probe_eval_sweep/vis")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--drop_inf_in_plots", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cf = resolve_run_paths(load_config(args.config))
    if "ori_size" not in cf:
        cf["ori_size"] = cf["proj_size"][::-1]
    device = torch.device(cf.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    cf["device"] = device

    out_dir = Path(cf["output_folder"]) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"> Output: {out_dir}")

    model = make_model(cf, device)
    loader = get_eval_loader(cf, split=args.split)
    chosen = select_case_indices(len(loader.dataset), args.sample_ratio, args.seed)

    cut_levels = parse_float_list(args.cut_levels)
    local_levels = parse_float_list(args.local_frac_levels)
    voxel_levels = parse_int_list(args.voxel_levels)

    level_map = {
        "clean": [0.0],
        "cut_lower_half": cut_levels,
        "cut_upper_half": cut_levels,
        "cut_posterior_half": cut_levels,
        "big_bite": local_levels,
        "big_blob": local_levels,
        "shifted": voxel_levels,
        "eroded": voxel_levels,
        "dilated": voxel_levels,
    }

    spacing = tuple(float(x) for x in cf.get("proj_spacing", [1.0, 1.0, 1.0]))
    fov_name = infer_fov_name(cf)
    rows: List[Dict] = []
    global_case_index = -1
    vis_counts: Dict[Tuple[str, float], int] = {}

    for batch in loader:
        x_target_raw = batch.get("target_image", batch["proj_image"]).to(device)
        names = batch["proj_name"]
        x_target = preprocess_clean_target(x_target_raw, cf)
        B = x_target.shape[0]

        for b in range(B):
            global_case_index += 1
            if global_case_index not in chosen:
                continue
            clean = x_target[b : b + 1]
            patient_id = extract_patient_id(names[b]) if isinstance(names, (list, tuple)) else str(global_case_index)

            for probe_type in args.probe_types:
                if probe_type not in level_map:
                    raise ValueError(f"Unknown probe_type {probe_type}; known {sorted(level_map)}")
                for level in level_map[probe_type]:
                    rng = random.Random(args.seed + 100000 * global_case_index + hash((probe_type, float(level))) % 100000)
                    corrupted_1, meta = make_corruption(clean[0], probe_type, level, rng)
                    corrupted = corrupted_1.unsqueeze(0)  # [1,1,D,H,W]

                    recon = reconstruct_binary(model, corrupted, cf, threshold=args.threshold)

                    d_it = dice_score_binary(corrupted, clean)[0].item()
                    d_ir = dice_score_binary(corrupted, recon)[0].item()
                    d_rt = dice_score_binary(recon, clean)[0].item()
                    h_it = hd95_binary(corrupted, clean, spacing)[0].item()
                    h_ir = hd95_binary(corrupted, recon, spacing)[0].item()
                    h_rt = hd95_binary(recon, clean, spacing)[0].item()
                    a_it = assd_binary(corrupted, clean, spacing)[0].item()
                    a_ir = assd_binary(corrupted, recon, spacing)[0].item()
                    a_rt = assd_binary(recon, clean, spacing)[0].item()

                    row = {
                        "config": args.config,
                        "exp_name": cf["exp_name"],
                        "fov": fov_name,
                        "split": args.split,
                        "case_index": global_case_index,
                        "patient_id": patient_id,
                        "probe_type": probe_type,
                        "level": float(level),
                        "requested_level": meta.get("requested_level", level),
                        "applied_level": meta.get("applied_level", level),
                        "adapted": bool(meta.get("adapted", False)),
                        "input_voxels": float(volume_vox(corrupted)[0].item()),
                        "target_voxels": float(volume_vox(clean)[0].item()),
                        "recon_voxels": float(volume_vox(recon)[0].item()),
                        "dice_input_vs_target": d_it,
                        "dice_input_vs_recon": d_ir,
                        "dice_recon_vs_target": d_rt,
                        "hd95_input_vs_target": h_it,
                        "hd95_input_vs_recon": h_ir,
                        "hd95_recon_vs_target": h_rt,
                        "assd_input_vs_target": a_it,
                        "assd_input_vs_recon": a_ir,
                        "assd_recon_vs_target": a_rt,
                    }
                    row.update({k: v for k, v in meta.items() if k not in row})
                    rows.append(row)

                    if args.save_vis:
                        # New behavior: save at least N examples for every probe type and severity level.
                        vis_key = (probe_type, float(level))
                        n_saved = vis_counts.get(vis_key, 0)
                        if n_saved < args.vis_max_cases_per_probe_level:
                            level_name = str(level).replace(".", "p")
                            vis_dir = (
                                out_dir / "visual_examples" / probe_type / f"level_{level_name}"
                                / f"case_{global_case_index:04d}_{patient_id}"
                            )
                            save_visual_example(
                                clean=clean[0],
                                corrupted=corrupted[0],
                                recon=recon[0],
                                vis_dir=vis_dir,
                                spacing=spacing,
                                metadata=row,
                            )
                            vis_counts[vis_key] = n_saved + 1

                        # Optional legacy behavior: old first-N-case folder layout.
                        if args.vis_save_legacy_first_cases and global_case_index < args.vis_max_cases:
                            vis_dir = out_dir / "vis" / f"case_{global_case_index:04d}_{patient_id}" / probe_type / str(level)
                            save_visual_example(
                                clean=clean[0],
                                corrupted=corrupted[0],
                                recon=recon[0],
                                vis_dir=vis_dir,
                                spacing=spacing,
                                metadata=row,
                            )

    detailed = pd.DataFrame(rows)
    detailed_csv = out_dir / "probe_eval_sweep_detailed.csv"
    detailed.to_csv(detailed_csv, index=False)

    # Summary: one row per probe type / severity level.
    metric_cols = [
        "dice_input_vs_target", "dice_input_vs_recon", "dice_recon_vs_target",
        "hd95_input_vs_target", "hd95_input_vs_recon", "hd95_recon_vs_target",
        "assd_input_vs_target", "assd_input_vs_recon", "assd_recon_vs_target",
        "input_voxels", "target_voxels", "recon_voxels",
    ]
    grouped = detailed.copy()
    for c in metric_cols:
        grouped[c] = pd.to_numeric(grouped[c], errors="coerce")
    summary = grouped.groupby(["fov", "probe_type", "level"], dropna=False)[metric_cols].agg(["mean", "median", "std", "min", "max", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary_csv = out_dir / "probe_eval_sweep_summary.csv"
    summary.to_csv(summary_csv, index=False)

    if args.save_vis:
        vis_index_rows = [
            {"probe_type": k[0], "level": k[1], "n_visual_examples_saved": v}
            for k, v in sorted(vis_counts.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        ]
        pd.DataFrame(vis_index_rows).to_csv(out_dir / "visual_examples_index.csv", index=False)

    if not args.no_plots:
        plot_dir = out_dir / "plots_by_probe_level"
        # Input-Target: severity of the synthetic corruption itself.
        plot_box_by_level(detailed, plot_dir, "dice_input_vs_target", "Input-Target Dice", drop_inf=False)
        plot_box_by_level(detailed, plot_dir, "hd95_input_vs_target", "Input-Target HD95", drop_inf=args.drop_inf_in_plots)
        plot_box_by_level(detailed, plot_dir, "assd_input_vs_target", "Input-Target ASSD", drop_inf=args.drop_inf_in_plots)

        # Input-Recon: how close AE output stays to the corrupted input.
        plot_box_by_level(detailed, plot_dir, "dice_input_vs_recon", "Input-Recon Dice", drop_inf=False)
        plot_box_by_level(detailed, plot_dir, "hd95_input_vs_recon", "Input-Recon HD95", drop_inf=args.drop_inf_in_plots)
        plot_box_by_level(detailed, plot_dir, "assd_input_vs_recon", "Input-Recon ASSD", drop_inf=args.drop_inf_in_plots)

        # Recon-Target: true repair quality relative to clean target.
        plot_box_by_level(detailed, plot_dir, "dice_recon_vs_target", "Recon-Target Dice", drop_inf=False)
        plot_box_by_level(detailed, plot_dir, "hd95_recon_vs_target", "Recon-Target HD95", drop_inf=args.drop_inf_in_plots)
        plot_box_by_level(detailed, plot_dir, "assd_recon_vs_target", "Recon-Target ASSD", drop_inf=args.drop_inf_in_plots)

        plot_summary_trends(summary, plot_dir)

    print(f"Wrote detailed CSV: {detailed_csv}")
    print(f"Wrote summary CSV:  {summary_csv}")
    if not args.no_plots:
        print(f"Wrote plots:        {out_dir / 'plots_by_probe_level'}")


if __name__ == "__main__":
    main()
