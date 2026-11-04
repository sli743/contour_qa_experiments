#!/usr/bin/env python3
"""Geometry helpers for full-volume SpinalCord autosegmentation-error analysis.

All arrays use NumPy z,y,x order. Physical distances are reported in millimetres.
The module deliberately does not depend on model outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


@dataclass
class MaskPair:
    patient_id: int
    approved_img: sitk.Image
    autoseg_img: sitk.Image
    approved: np.ndarray
    autoseg: np.ndarray
    spacing_zyx: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def resample_binary_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if same_geometry(mask, reference):
        out = mask
    else:
        out = sitk.Resample(
            mask,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    arr = (sitk.GetArrayFromImage(out) > 0).astype(np.uint8)
    binary = sitk.GetImageFromArray(arr)
    binary.CopyInformation(reference)
    return sitk.Cast(binary, sitk.sitkUInt8)


def orient_lps(img: sitk.Image) -> sitk.Image:
    """Permute/flip orthogonal medical-image axes into LPS without interpolation."""
    return sitk.DICOMOrient(img, "LPS")


def load_raw_pair(raw_data_root: Path, patient_id: int, roi: str = "SpinalCord") -> MaskPair:
    raw_data_root = Path(raw_data_root)
    ct_path = raw_data_root / "images" / f"patient{patient_id}_image.nii.gz"
    approved_path = raw_data_root / "labels-approved" / f"patient{patient_id}_{roi}_approved.nii.gz"
    autoseg_path = raw_data_root / "labels-autoseg" / f"patient{patient_id}_{roi}_autoseg.nii.gz"
    missing = [str(p) for p in (ct_path, approved_path, autoseg_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(";".join(missing))

    ct = sitk.ReadImage(str(ct_path))
    approved = resample_binary_to_reference(sitk.ReadImage(str(approved_path)), ct)
    autoseg = resample_binary_to_reference(sitk.ReadImage(str(autoseg_path)), ct)
    approved = orient_lps(approved)
    autoseg = orient_lps(autoseg)
    if not same_geometry(approved, autoseg):
        raise RuntimeError(f"LPS geometry mismatch for patient {patient_id}")

    app_arr = sitk.GetArrayFromImage(approved) > 0
    auto_arr = sitk.GetArrayFromImage(autoseg) > 0
    if not app_arr.any():
        raise ValueError(f"Approved {roi} is empty for patient {patient_id}")
    if not auto_arr.any():
        raise ValueError(f"Autoseg {roi} is empty for patient {patient_id}")

    spacing_xyz = tuple(float(v) for v in approved.GetSpacing())
    return MaskPair(
        patient_id=int(patient_id),
        approved_img=approved,
        autoseg_img=autoseg,
        approved=app_arr,
        autoseg=auto_arr,
        spacing_zyx=(spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]),
        origin_xyz=tuple(float(v) for v in approved.GetOrigin()),
        direction=tuple(float(v) for v in approved.GetDirection()),
    )


def crop_pair_to_union(
    pair: MaskPair,
    margin_xy_mm: float = 30.0,
    margin_z_mm: float = 110.0,
) -> MaskPair:
    union = pair.approved | pair.autoseg
    pts = np.argwhere(union)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    spacing = np.asarray(pair.spacing_zyx, dtype=float)
    margins = np.ceil(np.asarray([margin_z_mm, margin_xy_mm, margin_xy_mm]) / spacing).astype(int)
    lo = np.maximum(0, lo - margins)
    hi = np.minimum(np.asarray(union.shape) - 1, hi + margins)
    size_zyx = hi - lo + 1
    index_xyz = [int(lo[2]), int(lo[1]), int(lo[0])]
    size_xyz = [int(size_zyx[2]), int(size_zyx[1]), int(size_zyx[0])]
    app_img = sitk.RegionOfInterest(pair.approved_img, size_xyz, index_xyz)
    auto_img = sitk.RegionOfInterest(pair.autoseg_img, size_xyz, index_xyz)
    return MaskPair(
        patient_id=pair.patient_id,
        approved_img=app_img,
        autoseg_img=auto_img,
        approved=sitk.GetArrayFromImage(app_img) > 0,
        autoseg=sitk.GetArrayFromImage(auto_img) > 0,
        spacing_zyx=pair.spacing_zyx,
        origin_xyz=tuple(float(v) for v in app_img.GetOrigin()),
        direction=tuple(float(v) for v in app_img.GetDirection()),
    )


def crop_arrays_to_union(
    a: np.ndarray,
    b: np.ndarray,
    spacing_zyx: Sequence[float],
    margin_mm: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tight computational crop around the complete union of two binary masks."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = a | b
    if not union.any():
        return a, b
    pts = np.argwhere(union)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    spacing = np.asarray(spacing_zyx, dtype=float)
    margin = np.ceil(float(margin_mm) / spacing).astype(int)
    lo = np.maximum(0, lo - margin)
    hi = np.minimum(np.asarray(union.shape) - 1, hi + margin)
    sl = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
    return a[sl], b[sl]


def surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=_STRUCTURE_6, border_value=0)
    return mask & ~eroded


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def error_mask_similarity(real: np.ndarray, synthetic: np.ndarray) -> Tuple[float, float]:
    if not real.any() and not synthetic.any():
        return 1.0, 1.0
    if not real.any() or not synthetic.any():
        return 0.0, 0.0
    return dice(real, synthetic), jaccard(real, synthetic)


def surface_distances(a: np.ndarray, b: np.ndarray, spacing_zyx: Sequence[float]) -> Dict[str, float]:
    sa = surface(a)
    sb = surface(b)
    if not sa.any() or not sb.any():
        return {"assd_mm": np.nan, "hd95_mm": np.nan, "surface_max_mm": np.nan}
    da = ndimage.distance_transform_edt(~sa, sampling=spacing_zyx)
    db = ndimage.distance_transform_edt(~sb, sampling=spacing_zyx)
    a_to_b = db[sa]
    b_to_a = da[sb]
    both = np.concatenate([a_to_b, b_to_a]).astype(float, copy=False)
    return {
        "assd_mm": float(np.mean(both)),
        "hd95_mm": float(np.percentile(both, 95.0)),
        "surface_max_mm": float(np.max(both)),
    }


def signed_approved_surface_distance(
    approved: np.ndarray,
    autoseg: np.ndarray,
    spacing_zyx: Sequence[float],
) -> np.ndarray:
    """Signed distance from approved surface voxels to the autoseg surface.

    Positive means the approved surface voxel lies inside the autoseg mask
    (local outward/excess tendency); negative means it lies outside autoseg
    (local inward/missing tendency). Exact shared surface voxels have distance 0.
    """
    sg = surface(approved)
    sa = surface(autoseg)
    if not sg.any() or not sa.any():
        return np.asarray([], dtype=float)
    d = ndimage.distance_transform_edt(~sa, sampling=spacing_zyx)
    vals = d[sg].astype(float)
    sign = np.where(autoseg[sg], 1.0, -1.0)
    vals *= sign
    vals[d[sg] == 0] = 0.0
    return vals


def _centroid_index(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.asarray([np.nan, np.nan, np.nan])
    return pts.mean(axis=0)


def _bbox(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.full(3, np.nan), np.full(3, np.nan)
    return pts.min(axis=0).astype(float), pts.max(axis=0).astype(float)


def _component_rows(
    mask: np.ndarray,
    kind: str,
    spacing_zyx: Sequence[float],
    gt_voxels: int,
    gt_z_min: int,
    gt_z_max: int,
) -> List[Dict[str, float]]:
    labels, n = ndimage.label(mask, structure=_STRUCTURE_6)
    rows: List[Dict[str, float]] = []
    total = max(int(mask.sum()), 1)
    spacing = np.asarray(spacing_zyx, dtype=float)
    for label in range(1, n + 1):
        comp = labels == label
        vox = int(comp.sum())
        pts = np.argwhere(comp)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        ctr = pts.mean(axis=0)
        denom = max(gt_z_max - gt_z_min, 1)
        rows.append({
            "component_kind": kind,
            "component_index": int(label),
            "component_voxels": vox,
            "component_volume_mm3": float(vox * np.prod(spacing)),
            "fraction_of_same_error_type": float(vox / total),
            "fraction_of_gt_volume": float(vox / max(gt_voxels, 1)),
            "extent_si_mm": float((hi[0] - lo[0] + 1) * spacing[0]),
            "extent_ap_mm": float((hi[1] - lo[1] + 1) * spacing[1]),
            "extent_lr_mm": float((hi[2] - lo[2] + 1) * spacing[2]),
            "centroid_si_normalized": float((ctr[0] - gt_z_min) / denom),
            "centroid_z_index": float(ctr[0]),
            "centroid_y_index": float(ctr[1]),
            "centroid_x_index": float(ctr[2]),
            "bbox_z_min": int(lo[0]),
            "bbox_y_min": int(lo[1]),
            "bbox_x_min": int(lo[2]),
            "bbox_z_max": int(hi[0]),
            "bbox_y_max": int(hi[1]),
            "bbox_x_max": int(hi[2]),
        })
    rows.sort(key=lambda r: r["component_voxels"], reverse=True)
    return rows


def extract_pair_features(
    approved: np.ndarray,
    autoseg: np.ndarray,
    spacing_zyx: Sequence[float],
    patient_id: Optional[int] = None,
    cohort: Optional[str] = None,
    include_components: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    approved = np.asarray(approved, dtype=bool)
    autoseg = np.asarray(autoseg, dtype=bool)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if not approved.any() or not autoseg.any():
        raise ValueError("Both approved and autoseg masks must be non-empty")

    missing = approved & ~autoseg
    excess = autoseg & ~approved
    changed = missing | excess
    gt_vox = int(approved.sum())
    auto_vox = int(autoseg.sum())
    voxel_volume = float(np.prod(spacing))

    gt_lo, gt_hi = _bbox(approved)
    au_lo, au_hi = _bbox(autoseg)
    gt_ctr = _centroid_index(approved)
    au_ctr = _centroid_index(autoseg)
    delta_mm = (au_ctr - gt_ctr) * spacing

    gt_z_min, gt_z_max = int(gt_lo[0]), int(gt_hi[0])
    au_z_min, au_z_max = int(au_lo[0]), int(au_hi[0])
    inf_delta = float((au_z_min - gt_z_min) * spacing[0])
    sup_delta = float((au_z_max - gt_z_max) * spacing[0])
    gt_length = float((gt_z_max - gt_z_min + 1) * spacing[0])
    auto_length = float((au_z_max - au_z_min + 1) * spacing[0])

    sd = surface_distances(approved, autoseg, spacing)
    signed = signed_approved_surface_distance(approved, autoseg, spacing)
    signed_q = {q: (float(np.percentile(signed, q)) if len(signed) else np.nan) for q in (5, 25, 50, 75, 95)}

    component_rows: List[Dict[str, float]] = []
    if include_components:
        miss_rows = _component_rows(missing, "missing", spacing, gt_vox, gt_z_min, gt_z_max)
        exc_rows = _component_rows(excess, "excess", spacing, gt_vox, gt_z_min, gt_z_max)
        component_rows = miss_rows + exc_rows
    else:
        miss_rows = _component_rows(missing, "missing", spacing, gt_vox, gt_z_min, gt_z_max)
        exc_rows = _component_rows(excess, "excess", spacing, gt_vox, gt_z_min, gt_z_max)

    def largest(rows: List[Dict[str, float]], field: str, default: float = 0.0) -> float:
        return float(rows[0][field]) if rows else float(default)

    features: Dict[str, float] = {
        "patient_id": int(patient_id) if patient_id is not None else -1,
        "cohort": cohort or "",
        "spacing_z_mm": float(spacing[0]),
        "spacing_y_mm": float(spacing[1]),
        "spacing_x_mm": float(spacing[2]),
        "approved_volume_mm3": float(gt_vox * voxel_volume),
        "autoseg_volume_mm3": float(auto_vox * voxel_volume),
        "relative_volume_change_signed": float((auto_vox - gt_vox) / max(gt_vox, 1)),
        "relative_volume_change_abs": float(abs(auto_vox - gt_vox) / max(gt_vox, 1)),
        "missing_fraction_gt": float(missing.sum() / max(gt_vox, 1)),
        "excess_fraction_gt": float(excess.sum() / max(gt_vox, 1)),
        "changed_fraction_gt": float(changed.sum() / max(gt_vox, 1)),
        "dice": dice(approved, autoseg),
        "jaccard": jaccard(approved, autoseg),
        **sd,
        "signed_surface_p05_mm": signed_q[5],
        "signed_surface_p25_mm": signed_q[25],
        "signed_surface_p50_mm": signed_q[50],
        "signed_surface_p75_mm": signed_q[75],
        "signed_surface_p95_mm": signed_q[95],
        "surface_fraction_below_minus_1p5mm": float(np.mean(signed < -1.5)) if len(signed) else np.nan,
        "surface_fraction_below_minus_3mm": float(np.mean(signed < -3.0)) if len(signed) else np.nan,
        "surface_fraction_below_minus_5mm": float(np.mean(signed < -5.0)) if len(signed) else np.nan,
        "surface_fraction_above_plus_1p5mm": float(np.mean(signed > 1.5)) if len(signed) else np.nan,
        "surface_fraction_above_plus_3mm": float(np.mean(signed > 3.0)) if len(signed) else np.nan,
        "surface_fraction_above_plus_5mm": float(np.mean(signed > 5.0)) if len(signed) else np.nan,
        "centroid_shift_si_mm": float(delta_mm[0]),
        "centroid_shift_ap_mm": float(delta_mm[1]),
        "centroid_shift_lr_mm": float(delta_mm[2]),
        "centroid_shift_mm": float(np.linalg.norm(delta_mm)),
        "gt_length_si_mm": gt_length,
        "autoseg_length_si_mm": auto_length,
        "length_difference_signed_mm": float(auto_length - gt_length),
        "length_difference_abs_mm": float(abs(auto_length - gt_length)),
        "length_ratio": float(auto_length / max(gt_length, 1e-6)),
        "inferior_missing_mm": max(inf_delta, 0.0),
        "inferior_excess_mm": max(-inf_delta, 0.0),
        "superior_missing_mm": max(-sup_delta, 0.0),
        "superior_excess_mm": max(sup_delta, 0.0),
        "bbox_extent_si_gt_mm": float((gt_hi[0] - gt_lo[0] + 1) * spacing[0]),
        "bbox_extent_ap_gt_mm": float((gt_hi[1] - gt_lo[1] + 1) * spacing[1]),
        "bbox_extent_lr_gt_mm": float((gt_hi[2] - gt_lo[2] + 1) * spacing[2]),
        "bbox_extent_si_auto_mm": float((au_hi[0] - au_lo[0] + 1) * spacing[0]),
        "bbox_extent_ap_auto_mm": float((au_hi[1] - au_lo[1] + 1) * spacing[1]),
        "bbox_extent_lr_auto_mm": float((au_hi[2] - au_lo[2] + 1) * spacing[2]),
        "n_missing_components": int(len(miss_rows)),
        "n_excess_components": int(len(exc_rows)),
        "largest_missing_component_fraction": largest(miss_rows, "fraction_of_same_error_type"),
        "largest_excess_component_fraction": largest(exc_rows, "fraction_of_same_error_type"),
        "largest_missing_component_fraction_gt": largest(miss_rows, "fraction_of_gt_volume"),
        "largest_excess_component_fraction_gt": largest(exc_rows, "fraction_of_gt_volume"),
        "largest_missing_extent_si_mm": largest(miss_rows, "extent_si_mm"),
        "largest_missing_extent_ap_mm": largest(miss_rows, "extent_ap_mm"),
        "largest_missing_extent_lr_mm": largest(miss_rows, "extent_lr_mm"),
        "largest_excess_extent_si_mm": largest(exc_rows, "extent_si_mm"),
        "largest_excess_extent_ap_mm": largest(exc_rows, "extent_ap_mm"),
        "largest_excess_extent_lr_mm": largest(exc_rows, "extent_lr_mm"),
        "largest_missing_centroid_si_normalized": largest(miss_rows, "centroid_si_normalized", np.nan),
        "largest_excess_centroid_si_normalized": largest(exc_rows, "centroid_si_normalized", np.nan),
    }
    if patient_id is not None:
        for r in component_rows:
            r["patient_id"] = int(patient_id)
            r["cohort"] = cohort or ""
    return features, component_rows


def longitudinal_slice_profile(
    approved: np.ndarray,
    autoseg: np.ndarray,
    spacing_zyx: Sequence[float],
    patient_id: int,
    cohort: str,
) -> List[Dict[str, float]]:
    approved = np.asarray(approved, dtype=bool)
    autoseg = np.asarray(autoseg, dtype=bool)
    spacing = np.asarray(spacing_zyx, dtype=float)
    union = approved | autoseg
    z_any = np.where(np.any(union, axis=(1, 2)))[0]
    gt_z = np.where(np.any(approved, axis=(1, 2)))[0]
    gt_min, gt_max = int(gt_z.min()), int(gt_z.max())
    denom = max(gt_max - gt_min, 1)
    rows: List[Dict[str, float]] = []
    for z in range(int(z_any.min()), int(z_any.max()) + 1):
        g = approved[z]
        a = autoseg[z]
        gv = int(g.sum())
        av = int(a.sum())
        miss = int((g & ~a).sum())
        exc = int((a & ~g).sum())
        gc = np.argwhere(g).mean(axis=0) if gv else np.asarray([np.nan, np.nan])
        ac = np.argwhere(a).mean(axis=0) if av else np.asarray([np.nan, np.nan])
        if gv and av:
            cdelta = (ac - gc) * spacing[1:]
            cdist = float(np.linalg.norm(cdelta))
        else:
            cdist = np.nan
        rows.append({
            "patient_id": int(patient_id),
            "cohort": cohort,
            "slice_z_index": int(z),
            "position_si_mm_from_gt_inferior": float((z - gt_min) * spacing[0]),
            "normalized_gt_position": float((z - gt_min) / denom),
            "approved_area_mm2": float(gv * spacing[1] * spacing[2]),
            "autoseg_area_mm2": float(av * spacing[1] * spacing[2]),
            "area_ratio": float(av / gv) if gv else np.nan,
            "missing_area_fraction_of_gt_slice": float(miss / gv) if gv else np.nan,
            "excess_area_fraction_of_gt_slice": float(exc / gv) if gv else np.nan,
            "slice_dice": dice(g, a) if (gv or av) else np.nan,
            "centroid_displacement_mm": cdist,
        })
    return rows


def signed_maurer_array(mask: np.ndarray, spacing_zyx: Sequence[float]) -> np.ndarray:
    img = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
    img.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
    sdf = sitk.SignedMaurerDistanceMap(
        img,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return sitk.GetArrayFromImage(sdf).astype(np.float32, copy=False)


def boundary_offset(mask: np.ndarray, spacing_zyx: Sequence[float], offset_mm: float) -> np.ndarray:
    sdf = signed_maurer_array(mask, spacing_zyx)
    # Positive offset expands; negative offset contracts.
    return (sdf + float(offset_mm)) >= 0.0


def translate_no_wrap(mask: np.ndarray, shift_zyx: Sequence[int]) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    src: List[slice] = []
    dst: List[slice] = []
    for size, delta in zip(mask.shape, shift_zyx):
        d = int(delta)
        if d >= 0:
            src.append(slice(0, max(0, size - d)))
            dst.append(slice(d, size))
        else:
            src.append(slice(-d, size))
            dst.append(slice(0, size + d))
    out[tuple(dst)] = mask[tuple(src)]
    return out


def endpoint_cut(mask: np.ndarray, spacing_z_mm: float, side: str, length_mm: float) -> np.ndarray:
    out = mask.copy()
    z = np.where(np.any(mask, axis=(1, 2)))[0]
    if len(z) == 0:
        return out
    n = max(1, int(round(float(length_mm) / float(spacing_z_mm))))
    if side == "inferior":
        cutoff = min(int(z.min()) + n, int(z.max()) + 1)
        out[:cutoff] = False
    elif side == "superior":
        cutoff = max(int(z.max()) - n + 1, int(z.min()))
        out[cutoff:] = False
    else:
        raise ValueError(side)
    return out


def endpoint_extension(mask: np.ndarray, spacing_z_mm: float, side: str, length_mm: float) -> np.ndarray:
    """Simple diagnostic extension by repeating the terminal cross-section."""
    out = mask.copy()
    z = np.where(np.any(mask, axis=(1, 2)))[0]
    if len(z) == 0:
        return out
    n = max(1, int(round(float(length_mm) / float(spacing_z_mm))))
    if side == "inferior":
        source = int(z.min())
        lo = max(0, source - n)
        for zi in range(lo, source):
            out[zi] |= mask[source]
    elif side == "superior":
        source = int(z.max())
        hi = min(mask.shape[0], source + n + 1)
        for zi in range(source + 1, hi):
            out[zi] |= mask[source]
    else:
        raise ValueError(side)
    return out


def ellipsoid_mask(shape: Sequence[int], center_zyx: Sequence[float], radii_zyx: Sequence[float]) -> np.ndarray:
    radii = np.maximum(np.asarray(radii_zyx, dtype=float), 0.5)
    center = np.asarray(center_zyx, dtype=float)
    lo = np.maximum(0, np.floor(center - radii - 1).astype(int))
    hi = np.minimum(np.asarray(shape), np.ceil(center + radii + 2).astype(int))
    out = np.zeros(tuple(int(v) for v in shape), dtype=bool)
    zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    val = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = val
    return out


def component_matched_local_candidate(
    approved: np.ndarray,
    component_row: Mapping[str, float],
    kind: str,
    scale: float = 1.0,
) -> np.ndarray:
    center = [
        float(component_row["centroid_z_index"]),
        float(component_row["centroid_y_index"]),
        float(component_row["centroid_x_index"]),
    ]
    radii = [
        0.5 * (float(component_row["bbox_z_max"]) - float(component_row["bbox_z_min"]) + 1.0) * scale,
        0.5 * (float(component_row["bbox_y_max"]) - float(component_row["bbox_y_min"]) + 1.0) * scale,
        0.5 * (float(component_row["bbox_x_max"]) - float(component_row["bbox_x_min"]) + 1.0) * scale,
    ]
    local = ellipsoid_mask(approved.shape, center, radii)
    if kind == "missing":
        candidate = approved & ~local
        return candidate if candidate.any() else approved.copy()
    if kind == "excess":
        return approved | local
    raise ValueError(kind)


def robust_feature_scaler(rows: Sequence[Mapping[str, float]], features: Sequence[str]) -> Dict[str, Dict[str, object]]:
    # Imported lazily to keep this geometry module usable by the original
    # preprocessing scripts without introducing a circular import.
    from analysis.spinalcord_corruption_support import fit_mixed_scaler

    return fit_mixed_scaler(rows, features)


def grouped_feature_distance(
    a: Mapping[str, float],
    b: Mapping[str, float],
    scaler: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
) -> Tuple[float, Dict[str, float]]:
    from analysis.spinalcord_corruption_support import mixed_grouped_distance

    return mixed_grouped_distance(a, b, scaler, groups)


def weighted_error_spatial_scores(
    approved: np.ndarray,
    real_autoseg: np.ndarray,
    synthetic: np.ndarray,
) -> Dict[str, float]:
    real_missing = approved & ~real_autoseg
    real_excess = real_autoseg & ~approved
    syn_missing = approved & ~synthetic
    syn_excess = synthetic & ~approved
    md, mj = error_mask_similarity(real_missing, syn_missing)
    ed, ej = error_mask_similarity(real_excess, syn_excess)
    mr = int(real_missing.sum())
    er = int(real_excess.sum())
    total = mr + er
    if total == 0:
        synthetic_total = int(syn_missing.sum()) + int(syn_excess.sum())
        wd = 1.0 if synthetic_total == 0 else 0.0
        wj = 1.0 if synthetic_total == 0 else 0.0
    else:
        wd = float((mr * md + er * ed) / total)
        wj = float((mr * mj + er * ej) / total)
    return {
        "missing_error_dice": md,
        "missing_error_jaccard": mj,
        "excess_error_dice": ed,
        "excess_error_jaccard": ej,
        "weighted_error_dice": wd,
        "weighted_error_jaccard": wj,
    }
