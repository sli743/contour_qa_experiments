#!/usr/bin/env python3
"""Shared mask/geometry metrics for SpinalCord NSR-QA diagnostics.

The module intentionally computes HD95/HD99/max-HD directly from masks so the
reported definitions are explicit and do not depend on the ambiguous external
clinical metric label ``haus``.

All endpoint calculations are performed after reorienting to LPS. NumPy arrays
therefore use z,y,x order with z increasing from inferior to superior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def resample_binary_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if not same_geometry(mask, reference):
        mask = sitk.Resample(
            mask,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    arr = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference)
    return sitk.Cast(out, sitk.sitkUInt8)


def read_binary_lps(path: str | Path) -> Tuple[np.ndarray, sitk.Image, Tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    img = sitk.DICOMOrient(img, "LPS")
    arr = sitk.GetArrayFromImage(img) > 0
    spacing_xyz = tuple(float(v) for v in img.GetSpacing())
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return arr, img, spacing_zyx


def surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=_STRUCTURE_6, border_value=0)
    return mask & ~eroded


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    den = int(a.sum()) + int(b.sum())
    if den == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / den)


def surface_metrics(a: np.ndarray, b: np.ndarray, spacing_zyx: Sequence[float]) -> Dict[str, float]:
    """Symmetric voxel-surface metrics in physical millimetres.

    HD95 and HD99 are percentiles of the concatenated bidirectional surface
    distances; max-HD is the maximum of that same distribution.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if not a.any() and not b.any():
        return {"assd_mm": 0.0, "hd95_mm": 0.0, "hd99_mm": 0.0, "max_hd_mm": 0.0}
    if not a.any() or not b.any():
        return {"assd_mm": np.inf, "hd95_mm": np.inf, "hd99_mm": np.inf, "max_hd_mm": np.inf}
    sa = surface(a)
    sb = surface(b)
    d_to_a = ndimage.distance_transform_edt(~sa, sampling=spacing_zyx)
    d_to_b = ndimage.distance_transform_edt(~sb, sampling=spacing_zyx)
    vals = np.concatenate([d_to_b[sa], d_to_a[sb]]).astype(float, copy=False)
    return {
        "assd_mm": float(np.mean(vals)),
        "hd95_mm": float(np.percentile(vals, 95.0)),
        "hd99_mm": float(np.percentile(vals, 99.0)),
        "max_hd_mm": float(np.max(vals)),
    }


def _z_bounds(mask: np.ndarray) -> Tuple[int, int]:
    z = np.where(np.any(np.asarray(mask, dtype=bool), axis=(1, 2)))[0]
    if len(z) == 0:
        return -1, -1
    return int(z.min()), int(z.max())


def endpoint_metrics(reference: np.ndarray, candidate: np.ndarray, spacing_z_mm: float) -> Dict[str, float]:
    """Candidate endpoint errors relative to reference in LPS-oriented arrays."""
    r0, r1 = _z_bounds(reference)
    c0, c1 = _z_bounds(candidate)
    if min(r0, r1, c0, c1) < 0:
        return {
            "inferior_missing_mm": np.nan,
            "inferior_excess_mm": np.nan,
            "superior_missing_mm": np.nan,
            "superior_excess_mm": np.nan,
            "inferior_signed_error_mm": np.nan,
            "superior_signed_error_mm": np.nan,
            "max_abs_endpoint_error_mm": np.nan,
            "si_span_reference_mm": np.nan,
            "si_span_candidate_mm": np.nan,
            "si_span_difference_signed_mm": np.nan,
        }
    inf_signed = float((c0 - r0) * spacing_z_mm)  # + = missing, - = excess
    sup_signed = float((c1 - r1) * spacing_z_mm)  # + = excess, - = missing
    rspan = float((r1 - r0 + 1) * spacing_z_mm)
    cspan = float((c1 - c0 + 1) * spacing_z_mm)
    return {
        "inferior_missing_mm": max(inf_signed, 0.0),
        "inferior_excess_mm": max(-inf_signed, 0.0),
        "superior_missing_mm": max(-sup_signed, 0.0),
        "superior_excess_mm": max(sup_signed, 0.0),
        "inferior_signed_error_mm": inf_signed,
        "superior_signed_error_mm": sup_signed,
        "max_abs_endpoint_error_mm": max(abs(inf_signed), abs(sup_signed)),
        "si_span_reference_mm": rspan,
        "si_span_candidate_mm": cspan,
        "si_span_difference_signed_mm": cspan - rspan,
    }


def pair_metrics(reference: np.ndarray, candidate: np.ndarray, spacing_zyx: Sequence[float]) -> Dict[str, float]:
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    d = dice(reference, candidate)
    sm = surface_metrics(reference, candidate, spacing_zyx)
    ref_vox = int(reference.sum())
    cand_vox = int(candidate.sum())
    rel = abs(cand_vox - ref_vox) / ref_vox if ref_vox else np.nan
    signed_rel = (cand_vox - ref_vox) / ref_vox if ref_vox else np.nan
    ep = endpoint_metrics(reference, candidate, float(spacing_zyx[0]))
    return {
        "dice": d,
        "one_minus_dice": 1.0 - d,
        "relative_volume_difference": float(rel),
        "relative_volume_difference_signed": float(signed_rel),
        "reference_voxels": ref_vox,
        "candidate_voxels": cand_vox,
        **sm,
        **ep,
    }


def _safe_fraction(num: int | float, den: int | float) -> float:
    return float(num / den) if den else np.nan


def correction_decomposition(approved: np.ndarray, autoseg: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    """Decompose reconstruction changes into beneficial and harmful actions."""
    g = np.asarray(approved, dtype=bool)
    a = np.asarray(autoseg, dtype=bool)
    r = np.asarray(recon, dtype=bool)

    true_missing = g & ~a
    true_excess = a & ~g
    added = r & ~a
    removed = a & ~r

    beneficial_add = added & true_missing
    harmful_add = added & ~g
    beneficial_remove = removed & true_excess
    harmful_remove = removed & g

    true_error = true_missing | true_excess
    residual_error = np.logical_xor(g, r)
    beneficial = beneficial_add | beneficial_remove
    harmful = harmful_add | harmful_remove
    changed = added | removed

    return {
        "true_missing_voxels": int(true_missing.sum()),
        "true_excess_voxels": int(true_excess.sum()),
        "network_added_voxels": int(added.sum()),
        "network_removed_voxels": int(removed.sum()),
        "beneficial_added_voxels": int(beneficial_add.sum()),
        "harmful_added_voxels": int(harmful_add.sum()),
        "beneficial_removed_voxels": int(beneficial_remove.sum()),
        "harmful_removed_voxels": int(harmful_remove.sum()),
        "missing_correction_recall": _safe_fraction(beneficial_add.sum(), true_missing.sum()),
        "addition_precision": _safe_fraction(beneficial_add.sum(), added.sum()),
        "excess_correction_recall": _safe_fraction(beneficial_remove.sum(), true_excess.sum()),
        "removal_precision": _safe_fraction(beneficial_remove.sum(), removed.sum()),
        "overall_error_correction_recall": _safe_fraction(beneficial.sum(), true_error.sum()),
        "change_precision": _safe_fraction(beneficial.sum(), changed.sum()),
        "overcorrection_fraction_of_change": _safe_fraction(harmful.sum(), changed.sum()),
        "voxel_error_before": int(true_error.sum()),
        "voxel_error_after": int(residual_error.sum()),
        "voxel_error_reduction_fraction": _safe_fraction(true_error.sum() - residual_error.sum(), true_error.sum()),
    }


def prefixed(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}
