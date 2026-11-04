#!/usr/bin/env python3
"""Torch/CUDA implementation of the benchmark-overlap contour corruptions.

The benchmark corruption distribution is unchanged.  This module replaces the
legacy SimpleITK/SciPy full-volume morphology path so training can keep each
mask on the same torch device as the model.
"""
from __future__ import annotations

import json
import random
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _sitk_ball_radius1_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """3-D radius-1 ITK/SimpleITK ball footprint (19 voxels)."""
    kernel = torch.ones((3, 3, 3), device=device, dtype=dtype)
    for z in (0, 2):
        for y in (0, 2):
            for x in (0, 2):
                kernel[z, y, x] = 0
    return kernel.view(1, 1, 3, 3, 3)


def _binary_morph_once(x: torch.Tensor, *, dilate: bool) -> torch.Tensor:
    """One radius-1 ball dilation/erosion step on a [C,D,H,W] mask."""
    xb = (x > 0.5).float()
    if xb.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W], got {tuple(xb.shape)}")
    channels = int(xb.shape[0])
    kernel = _sitk_ball_radius1_kernel(xb.device, xb.dtype).repeat(channels, 1, 1, 1, 1)
    counts = F.conv3d(xb.unsqueeze(0), kernel, padding=1, groups=channels).squeeze(0)
    if dilate:
        return counts > 0
    # Legacy code used BoundaryToForeground(False), so the zero-padding outside
    # the FOV is background and all 19 footprint voxels must be foreground.
    return counts >= 18.5


@torch.no_grad()
def binary_erode_adaptive(x: torch.Tensor, iterations: int) -> Tuple[torch.Tensor, int, bool]:
    """Return the largest non-empty erosion level up to ``iterations``."""
    x0 = x > 0.5
    if iterations <= 0 or not bool(x0.any().item()):
        return x0.float(), 0, False

    current = x0
    last_nonempty = x0
    applied = 0
    for it in range(1, int(iterations) + 1):
        current = _binary_morph_once(current.float(), dilate=False)
        if not bool(current.any().item()):
            return last_nonempty.float(), applied, True
        last_nonempty = current
        applied = it
    return last_nonempty.float(), applied, False


@torch.no_grad()
def binary_dilate(x: torch.Tensor, iterations: int) -> torch.Tensor:
    current = x > 0.5
    if iterations <= 0 or not bool(current.any().item()):
        return current.float()
    for _ in range(int(iterations)):
        current = _binary_morph_once(current.float(), dilate=True)
    return current.float()


def foreground_bbox(x: torch.Tensor):
    fg = (x[0] > 0.5).nonzero(as_tuple=False)
    if fg.numel() == 0:
        return None, None, None
    mins = fg.min(dim=0).values
    maxs = fg.max(dim=0).values
    center = fg.float().mean(dim=0)
    return center, mins, maxs


def cut_by_ratio(x: torch.Tensor, axis: int, side: str, ratio: float) -> torch.Tensor:
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
    x = (x > 0.5).float()
    if voxels <= 0:
        return x, (0, 0, 0)
    choices = [-1, 0, 1]
    dz = dy = dx = 0
    while dz == dy == dx == 0:
        dz, dy, dx = (rng.choice(choices), rng.choice(choices), rng.choice(choices))
    dz, dy, dx = dz * voxels, dy * voxels, dx * voxels

    out = torch.zeros_like(x)
    _, depth, height, width = x.shape
    z_src0, z_src1 = max(0, -dz), min(depth, depth - dz) if dz >= 0 else depth
    y_src0, y_src1 = max(0, -dy), min(height, height - dy) if dy >= 0 else height
    x_src0, x_src1 = max(0, -dx), min(width, width - dx) if dx >= 0 else width
    z_dst0, y_dst0, x_dst0 = max(0, dz), max(0, dy), max(0, dx)
    z_dst1 = z_dst0 + (z_src1 - z_src0)
    y_dst1 = y_dst0 + (y_src1 - y_src0)
    x_dst1 = x_dst0 + (x_src1 - x_src0)
    if z_src1 > z_src0 and y_src1 > y_src0 and x_src1 > x_src0:
        out[:, z_dst0:z_dst1, y_dst0:y_dst1, x_dst0:x_dst1] = x[:, z_src0:z_src1, y_src0:y_src1, x_src0:x_src1]
    return out, (dz, dy, dx)


def surface_voxels(x: torch.Tensor) -> torch.Tensor | None:
    xb = x > 0.5
    if not bool(xb.any().item()):
        return None
    eroded = _binary_morph_once(xb.float(), dilate=False)
    surface = xb & ~eroded
    coords = surface[0].nonzero(as_tuple=False)
    if coords.numel() == 0:
        coords = xb[0].nonzero(as_tuple=False)
    return coords


def choose_surface_anchor(x: torch.Tensor, rng: random.Random) -> Tuple[int, int, int] | None:
    coords = surface_voxels(x)
    if coords is None or coords.numel() == 0:
        return None
    idx = rng.randrange(coords.shape[0])
    return tuple(int(v) for v in coords[idx].tolist())


def _box_touches_original_6conn(
    original: torch.Tensor,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> bool:
    """Equivalent connectivity check for one added solid cuboid."""
    _, depth, height, width = original.shape
    touch = original[:, z0:z1, y0:y1, x0:x1].any()
    if z0 > 0:
        touch = touch | original[:, z0 - 1 : z0, y0:y1, x0:x1].any()
    if z1 < depth:
        touch = touch | original[:, z1 : z1 + 1, y0:y1, x0:x1].any()
    if y0 > 0:
        touch = touch | original[:, z0:z1, y0 - 1 : y0, x0:x1].any()
    if y1 < height:
        touch = touch | original[:, z0:z1, y1 : y1 + 1, x0:x1].any()
    if x0 > 0:
        touch = touch | original[:, z0:z1, y0:y1, x0 - 1 : x0].any()
    if x1 < width:
        touch = touch | original[:, z0:z1, y0:y1, x1 : x1 + 1].any()
    return bool(touch.item())


def local_box_bite_or_blob(
    x: torch.Tensor,
    frac: float,
    rng: random.Random,
    mode: str,
    inward_fraction: float = 0.35,
    keep_connected: bool = True,
) -> torch.Tensor:
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
        shift = torch.stack([
            inward_fraction * dz * direction[0],
            inward_fraction * dy * direction[1],
            inward_fraction * dx * direction[2],
        ])
        box_center = anchor_t + shift
    elif mode == "bite":
        box_center = anchor_t
    else:
        raise ValueError(mode)

    zc, yc, xc = [int(round(v)) for v in box_center.tolist()]
    z0, y0, x0 = max(0, zc - dz // 2), max(0, yc - dy // 2), max(0, xc - dx // 2)
    z1, y1, x1 = min(y.shape[1], z0 + dz), min(y.shape[2], y0 + dy), min(y.shape[3], x0 + dx)

    if mode == "bite":
        y[:, z0:z1, y0:y1, x0:x1] = 0
        return y

    original = x > 0.5
    if keep_connected and not _box_touches_original_6conn(original, z0, z1, y0, y1, x0, x1):
        return original.float()
    y[:, z0:z1, y0:y1, x0:x1] = 1
    return y.float()


@torch.no_grad()
def make_corruption_gpu(
    x: torch.Tensor,
    probe_type: str,
    level: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, Dict]:
    """Apply one benchmark corruption without leaving the torch device."""
    meta: Dict = {"requested_level": level, "applied_level": level, "adapted": False}
    if probe_type == "clean":
        return x.clone(), meta
    if probe_type == "shifted":
        y, shift = random_shift_fixed(x, int(level), rng)
        meta["shift_zyx"] = json.dumps(list(shift))
        return y, meta
    if probe_type == "eroded":
        y, applied, adapted = binary_erode_adaptive(x, int(level))
        meta["applied_level"] = applied
        meta["adapted"] = adapted
        return y, meta
    if probe_type == "dilated":
        return binary_dilate(x, int(level)), meta

    # Anatomically explicit cut aliases for LPS-canonical [C,D,H,W] masks.
    # D is NumPy/SimpleITK z: low-z is inferior, high-z is superior.
    # H is NumPy/SimpleITK y: high-y is posterior.  Keep the historical
    # cut_upper_half/cut_lower_half behavior below unchanged for reproducibility.
    if probe_type == "cut_inferior_portion":
        return cut_by_ratio(x, axis=0, side="low", ratio=float(level)), meta
    if probe_type == "cut_superior_portion":
        return cut_by_ratio(x, axis=0, side="high", ratio=float(level)), meta
    if probe_type == "cut_posterior_portion":
        return cut_by_ratio(x, axis=1, side="high", ratio=float(level)), meta

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
