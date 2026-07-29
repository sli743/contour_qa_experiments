#!/usr/bin/env python3
"""Simple online random morphological corruptions for contour denoising.

These training corruptions intentionally do not reproduce the fixed evaluation
probe families or discrete benchmark severity levels. Each non-clean sample can
receive independent random morphology, translation, and one local ball-shaped
addition/removal.
"""
from __future__ import annotations

import random
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage


def _translate_no_wrap(mask: np.ndarray, shift_zyx: Sequence[int]) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    src, dst = [], []
    for size, delta in zip(mask.shape, shift_zyx):
        delta = int(delta)
        if delta >= 0:
            src.append(slice(0, max(0, size - delta)))
            dst.append(slice(delta, size))
        else:
            src.append(slice(-delta, size))
            dst.append(slice(0, size + delta))
    out[tuple(dst)] = mask[tuple(src)]
    return out


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return np.argwhere(mask & ~eroded)


def _ball(shape: Sequence[int], center: Sequence[int], radius: int) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = [int(v) for v in center]
    return (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= int(radius) ** 2


def random_morphology_corruption(
    clean: torch.Tensor,
    cfg: Dict,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Corrupt one [1,D,H,W] binary contour and return input/change mask/metadata."""
    if clean.ndim != 4 or clean.shape[0] != 1:
        raise ValueError(f"Expected [1,D,H,W], got {tuple(clean.shape)}")

    spec = cfg["random_morphology"]
    arr = clean.detach().cpu().numpy()[0] > 0.5
    original = arr.copy()
    applied = []

    if rng.random() < float(spec.get("clean_probability", 0.20)):
        result = original
        applied.append("clean")
    else:
        result = original.copy()

        morph = spec.get("morphology", {})
        if rng.random() < float(morph.get("probability", 0.80)):
            lo, hi = morph.get("iterations", [1, 4])
            iterations = rng.randint(int(lo), int(hi))
            if rng.random() < float(morph.get("erosion_probability", 0.50)):
                candidate = ndimage.binary_erosion(result, iterations=iterations, border_value=0)
                if candidate.any():
                    result = candidate
                    applied.append(f"erosion_{iterations}")
            else:
                result = ndimage.binary_dilation(result, iterations=iterations)
                applied.append(f"dilation_{iterations}")

        trans = spec.get("translation", {})
        if rng.random() < float(trans.get("probability", 0.50)):
            lo, hi = trans.get("voxels", [-3, 3])
            shift = [rng.randint(int(lo), int(hi)) for _ in range(3)]
            if any(shift):
                candidate = _translate_no_wrap(result, shift)
                if candidate.any():
                    result = candidate
                    applied.append("translation_" + "_".join(map(str, shift)))

        local = spec.get("local_ball", {})
        if rng.random() < float(local.get("probability", 0.40)) and result.any():
            surface = _surface_voxels(result)
            if len(surface):
                center = surface[rng.randrange(len(surface))]
                rlo, rhi = local.get("radius_voxels", [3, 10])
                radius = rng.randint(int(rlo), int(rhi))
                ball = _ball(result.shape, center, radius)
                if rng.random() < float(local.get("bite_probability", 0.50)):
                    candidate = result & ~ball
                    if candidate.any():
                        result = candidate
                        applied.append(f"bite_r{radius}")
                else:
                    result = result | ball
                    applied.append(f"blob_r{radius}")

        # Avoid accidental identity samples outside the explicit clean branch.
        if np.array_equal(result, original):
            fallback = ndimage.binary_dilation(original, iterations=1)
            if not np.array_equal(fallback, original):
                result = fallback
                applied.append("fallback_dilation_1")

    out = torch.from_numpy(result.astype(np.float32))[None].to(clean.device)
    changed = (out != clean).float()
    metadata = {
        "operations": applied,
        "changed_fraction": float(changed.mean().item()),
    }
    return out, changed, metadata
