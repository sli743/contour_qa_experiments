#!/usr/bin/env python3
"""Audit whether the fixed NSR-QA SpinalCord FOV hides S/I extent errors.

This inspection reproduces the two geometries that matter for the original
SpinalCord NSR-QA experiment:

1. ``training_approved_reference``: the already-materialized approved-derived
   training/validation/testing crop used by the trained model.
2. ``autoseg_centered``: the deployment/real-autoseg crop created by centering
   the same fixed template geometry on the raw autosegmented SpinalCord.

For every patient in the NSR-QA config, the script compares the *raw* approved
and autosegmentation masks against those FOVs in physical space.  It reports:

* raw approved/autoseg S-I (crop-z-axis) endpoint extents;
* fraction of raw foreground voxel centers lying inside each FOV;
* whether the approved contour reaches the z=0 / z=max crop boundaries after
  nearest-neighbor resampling;
* full-volume approved-beyond-autoseg endpoint error (under-contour extent);
* the same endpoint error after the autoseg-centered FOV is applied;
* how many millimetres of the endpoint error are therefore hidden by the FOV.

The foreground-center fraction is geometric and does not compare voxel counts
across different spacings.  This is important because the original preprocessing
may resample the raw CT grid to 1.5-mm isotropic voxels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from evaluation.actual_autoseg.preprocess_actual_autoseg import (
    binary_on_ct,
    extract_patient_id,
    load_yaml,
    make_reference_from_template_center,
    raw_label,
    resample_to_reference,
    split_label_dir,
    union_center_physical,
)

ROI = "SpinalCord"
GROUP = "spine"


def _float_or_nan(x: Optional[float]) -> float:
    return float(x) if x is not None else math.nan


def _foreground_physical_points(mask: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(mask) > 0
    coords_zyx = np.column_stack(np.where(arr))
    if coords_zyx.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    idx_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float64, copy=False)
    spacing = np.asarray(mask.GetSpacing(), dtype=np.float64)
    origin = np.asarray(mask.GetOrigin(), dtype=np.float64)
    direction = np.asarray(mask.GetDirection(), dtype=np.float64).reshape(3, 3)
    return origin[None, :] + (idx_xyz * spacing[None, :]) @ direction.T


def _physical_to_continuous_index(points_xyz: np.ndarray, ref: sitk.Image) -> np.ndarray:
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    spacing = np.asarray(ref.GetSpacing(), dtype=np.float64)
    origin = np.asarray(ref.GetOrigin(), dtype=np.float64)
    direction = np.asarray(ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    inv_direction = np.linalg.inv(direction)
    local_mm = (points_xyz - origin[None, :]) @ inv_direction.T
    return local_mm / spacing[None, :]


def _axis_from_reference(ref: sitk.Image) -> np.ndarray:
    direction = np.asarray(ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    axis = direction[:, 2]
    norm = np.linalg.norm(axis)
    if norm <= 0:
        raise ValueError("Invalid reference direction: zero z-axis vector")
    return axis / norm


def _axis_side_labels(axis: np.ndarray) -> Tuple[str, str]:
    """Return labels for low/high coordinate along crop z-axis.

    SimpleITK physical coordinates are interpreted in the usual LPS convention.
    For an axial CT, +physical-Z is superior.  If the crop z-axis is strongly
    aligned with physical Z, name the sides inferior/superior; otherwise retain
    neutral axis-low/axis-high labels.
    """
    z_component = float(axis[2])
    if z_component >= 0.7:
        return "inferior", "superior"
    if z_component <= -0.7:
        return "superior", "inferior"
    return "axis_low", "axis_high"


def _extent_along_axis(mask: sitk.Image, axis: np.ndarray) -> Tuple[float, float, float, int]:
    points = _foreground_physical_points(mask)
    if points.shape[0] == 0:
        return math.nan, math.nan, math.nan, 0
    proj = points @ axis
    low = float(np.min(proj))
    high = float(np.max(proj))
    return low, high, high - low, int(points.shape[0])


def _fov_geometry_stats(mask: sitk.Image, ref: sitk.Image, axis: np.ndarray) -> Dict[str, float]:
    points = _foreground_physical_points(mask)
    n = int(points.shape[0])
    if n == 0:
        return {
            "foreground_voxels_raw": 0,
            "foreground_centers_inside_fov": 0,
            "foreground_centers_outside_fov": 0,
            "foreground_center_fraction_inside_fov": math.nan,
            "outside_z_low_centers": 0,
            "outside_z_high_centers": 0,
            "outside_xy_centers": 0,
            "extent_low_axis_mm": math.nan,
            "extent_high_axis_mm": math.nan,
            "extent_span_axis_mm": math.nan,
            "extent_beyond_fov_low_mm": math.nan,
            "extent_beyond_fov_high_mm": math.nan,
        }

    ci = _physical_to_continuous_index(points, ref)
    size = np.asarray(ref.GetSize(), dtype=np.float64)

    # Physical FOV boundaries lie half a voxel outside the first/last centers.
    lower = np.full(3, -0.5, dtype=np.float64)
    upper = size - 0.5
    inside_dims = (ci >= lower[None, :]) & (ci <= upper[None, :])
    inside = np.all(inside_dims, axis=1)

    outside_z_low = ci[:, 2] < -0.5
    outside_z_high = ci[:, 2] > (size[2] - 0.5)
    outside_xy = (~inside_dims[:, 0]) | (~inside_dims[:, 1])

    proj = points @ axis
    extent_low = float(np.min(proj))
    extent_high = float(np.max(proj))

    origin = np.asarray(ref.GetOrigin(), dtype=np.float64)
    spacing_z = float(ref.GetSpacing()[2])
    ref_center_low = float(origin @ axis)
    ref_center_high = ref_center_low + (float(ref.GetSize()[2]) - 1.0) * spacing_z
    fov_edge_low = ref_center_low - 0.5 * spacing_z
    fov_edge_high = ref_center_high + 0.5 * spacing_z

    return {
        "foreground_voxels_raw": n,
        "foreground_centers_inside_fov": int(np.sum(inside)),
        "foreground_centers_outside_fov": int(np.sum(~inside)),
        "foreground_center_fraction_inside_fov": float(np.mean(inside)),
        "outside_z_low_centers": int(np.sum(outside_z_low)),
        "outside_z_high_centers": int(np.sum(outside_z_high)),
        "outside_xy_centers": int(np.sum(outside_xy)),
        "extent_low_axis_mm": extent_low,
        "extent_high_axis_mm": extent_high,
        "extent_span_axis_mm": extent_high - extent_low,
        "extent_beyond_fov_low_mm": max(0.0, fov_edge_low - extent_low),
        "extent_beyond_fov_high_mm": max(0.0, extent_high - fov_edge_high),
    }


def _resampled_boundary_stats(mask: sitk.Image, ref: sitk.Image) -> Dict[str, object]:
    out = resample_to_reference(mask, ref, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(out) > 0
    if not np.any(arr):
        return {
            "resampled_voxels": 0,
            "bbox_min_zyx": "",
            "bbox_max_zyx": "",
            "touches_z_low": False,
            "touches_z_high": False,
            "boundary_voxels": 0,
        }

    coords = np.column_stack(np.where(arr))
    mins = coords.min(axis=0).astype(int)
    maxs = coords.max(axis=0).astype(int)
    boundary = np.zeros_like(arr, dtype=bool)
    boundary[0, :, :] |= arr[0, :, :]
    boundary[-1, :, :] |= arr[-1, :, :]
    boundary[:, 0, :] |= arr[:, 0, :]
    boundary[:, -1, :] |= arr[:, -1, :]
    boundary[:, :, 0] |= arr[:, :, 0]
    boundary[:, :, -1] |= arr[:, :, -1]
    return {
        "resampled_voxels": int(arr.sum()),
        "bbox_min_zyx": json.dumps(mins.tolist()),
        "bbox_max_zyx": json.dumps(maxs.tolist()),
        "touches_z_low": bool(mins[0] == 0),
        "touches_z_high": bool(maxs[0] == arr.shape[0] - 1),
        "boundary_voxels": int(boundary.sum()),
    }


def _approved_beyond_autoseg(
    approved_extent: Tuple[float, float, float, int],
    autoseg_extent: Tuple[float, float, float, int],
) -> Tuple[float, float, float, float]:
    app_low, app_high, _, app_n = approved_extent
    auto_low, auto_high, _, auto_n = autoseg_extent
    if app_n == 0 or auto_n == 0:
        return math.nan, math.nan, math.nan, math.nan
    under_low = max(0.0, auto_low - app_low)
    under_high = max(0.0, app_high - auto_high)
    over_low = max(0.0, app_low - auto_low)
    over_high = max(0.0, auto_high - app_high)
    return under_low, under_high, over_low, over_high


def _clip_mask(mask: sitk.Image, ref: sitk.Image) -> sitk.Image:
    return resample_to_reference(mask, ref, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def _prefixed(row: Dict[str, object], prefix: str, values: Dict[str, object]) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def _map_si_values(row: Dict[str, object], low_label: str, high_label: str) -> None:
    for stem in ["raw_under", "raw_over", "visible_under", "visible_over", "hidden_under"]:
        low_key = f"{stem}_low_mm"
        high_key = f"{stem}_high_mm"
        low_val = row.get(low_key, math.nan)
        high_val = row.get(high_key, math.nan)
        if low_label == "inferior":
            row[f"{stem}_inferior_mm"] = low_val
            row[f"{stem}_superior_mm"] = high_val
        elif high_label == "inferior":
            row[f"{stem}_inferior_mm"] = high_val
            row[f"{stem}_superior_mm"] = low_val
        else:
            row[f"{stem}_inferior_mm"] = math.nan
            row[f"{stem}_superior_mm"] = math.nan


def inspect_case(
    *,
    pid: str,
    split: str,
    approved_template_path: Path,
    raw_root: Path,
    hidden_mm_threshold: float,
    major_hidden_mm_threshold: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "patient_id": pid,
        "split": split,
        "roi": ROI,
        "status": "ok",
        "reason": "",
        "approved_template": str(approved_template_path),
    }

    raw_ct_path = raw_root / "images" / f"patient{pid}_image.nii.gz"
    raw_app_path = raw_label(raw_root, pid, ROI, "approved")
    raw_auto_path = raw_label(raw_root, pid, ROI, "autoseg")
    row.update({
        "raw_ct": str(raw_ct_path),
        "raw_approved": str(raw_app_path),
        "raw_autoseg": str(raw_auto_path),
    })
    missing = [str(p) for p in [raw_ct_path, raw_app_path, raw_auto_path, approved_template_path] if not p.is_file()]
    if missing:
        row["status"] = "skipped_missing_file"
        row["reason"] = ";".join(missing)
        return row

    ct = sitk.Cast(sitk.ReadImage(str(raw_ct_path)), sitk.sitkFloat32)
    approved = binary_on_ct(raw_app_path, ct)
    autoseg = binary_on_ct(raw_auto_path, ct)
    training_ref = sitk.ReadImage(str(approved_template_path))
    axis = _axis_from_reference(training_ref)
    low_label, high_label = _axis_side_labels(axis)

    row["axis_low_label"] = low_label
    row["axis_high_label"] = high_label
    row["crop_z_axis_lps_z_component"] = float(axis[2])
    row["training_fov_size_xyz"] = json.dumps(list(training_ref.GetSize()))
    row["training_fov_spacing_xyz"] = json.dumps(list(training_ref.GetSpacing()))
    row["training_fov_origin_xyz"] = json.dumps(list(training_ref.GetOrigin()))
    row["training_fov_z_center_span_mm"] = (training_ref.GetSize()[2] - 1) * training_ref.GetSpacing()[2]
    row["training_fov_z_edge_span_mm"] = training_ref.GetSize()[2] * training_ref.GetSpacing()[2]

    app_extent = _extent_along_axis(approved, axis)
    auto_extent = _extent_along_axis(autoseg, axis)
    row["raw_approved_extent_low_axis_mm"] = app_extent[0]
    row["raw_approved_extent_high_axis_mm"] = app_extent[1]
    row["raw_approved_extent_span_axis_mm"] = app_extent[2]
    row["raw_autoseg_extent_low_axis_mm"] = auto_extent[0]
    row["raw_autoseg_extent_high_axis_mm"] = auto_extent[1]
    row["raw_autoseg_extent_span_axis_mm"] = auto_extent[2]

    raw_under_low, raw_under_high, raw_over_low, raw_over_high = _approved_beyond_autoseg(app_extent, auto_extent)
    row.update({
        "raw_under_low_mm": raw_under_low,
        "raw_under_high_mm": raw_under_high,
        "raw_over_low_mm": raw_over_low,
        "raw_over_high_mm": raw_over_high,
    })

    # 1) Exact geometry used by the original approved-derived training input.
    _prefixed(row, "training_approved", _fov_geometry_stats(approved, training_ref, axis))
    _prefixed(row, "training_approved", _resampled_boundary_stats(approved, training_ref))
    _prefixed(row, "training_autoseg", _fov_geometry_stats(autoseg, training_ref, axis))
    row["training_fov_clips_approved"] = bool(row["training_approved_foreground_centers_outside_fov"] > 0)

    # 2) Exact candidate/autoseg-centered geometry used by the primary real-error QA pipeline.
    center = union_center_physical([autoseg])
    if center is None:
        row["status"] = "empty_autoseg_no_center"
        row["reason"] = "raw SpinalCord autosegmentation is empty"
        return row
    auto_ref = make_reference_from_template_center(training_ref, center)
    row["autoseg_center_phys_xyz"] = json.dumps([float(x) for x in center])
    row["autoseg_fov_origin_xyz"] = json.dumps(list(auto_ref.GetOrigin()))

    _prefixed(row, "autoseg_fov_approved", _fov_geometry_stats(approved, auto_ref, axis))
    _prefixed(row, "autoseg_fov_autoseg", _fov_geometry_stats(autoseg, auto_ref, axis))
    _prefixed(row, "autoseg_fov_approved", _resampled_boundary_stats(approved, auto_ref))
    _prefixed(row, "autoseg_fov_autoseg", _resampled_boundary_stats(autoseg, auto_ref))

    app_visible = _clip_mask(approved, auto_ref)
    auto_visible = _clip_mask(autoseg, auto_ref)
    app_visible_extent = _extent_along_axis(app_visible, axis)
    auto_visible_extent = _extent_along_axis(auto_visible, axis)
    vis_under_low, vis_under_high, vis_over_low, vis_over_high = _approved_beyond_autoseg(
        app_visible_extent, auto_visible_extent
    )
    row.update({
        "visible_under_low_mm": vis_under_low,
        "visible_under_high_mm": vis_under_high,
        "visible_over_low_mm": vis_over_low,
        "visible_over_high_mm": vis_over_high,
        "hidden_under_low_mm": max(0.0, raw_under_low - vis_under_low) if np.isfinite(raw_under_low) and np.isfinite(vis_under_low) else math.nan,
        "hidden_under_high_mm": max(0.0, raw_under_high - vis_under_high) if np.isfinite(raw_under_high) and np.isfinite(vis_under_high) else math.nan,
    })
    _map_si_values(row, low_label, high_label)

    hidden_vals = [row["hidden_under_low_mm"], row["hidden_under_high_mm"]]
    finite_hidden = [float(v) for v in hidden_vals if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
    max_hidden = max(finite_hidden) if finite_hidden else math.nan
    row["max_hidden_undercontour_mm"] = max_hidden
    row["qa_unobservable_risk"] = bool(np.isfinite(max_hidden) and max_hidden >= hidden_mm_threshold)
    row["qa_unobservable_major_risk"] = bool(np.isfinite(max_hidden) and max_hidden >= major_hidden_mm_threshold)
    row["autoseg_fov_clips_approved"] = bool(row["autoseg_fov_approved_foreground_centers_outside_fov"] > 0)
    row["autoseg_fov_clips_autoseg"] = bool(row["autoseg_fov_autoseg_foreground_centers_outside_fov"] > 0)
    return row


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "patient_id", "split", "status", "reason", "axis_low_label", "axis_high_label",
        "training_fov_size_xyz", "training_fov_spacing_xyz", "training_fov_z_edge_span_mm",
        "raw_approved_extent_span_axis_mm", "raw_autoseg_extent_span_axis_mm",
        "raw_under_inferior_mm", "raw_under_superior_mm",
        "visible_under_inferior_mm", "visible_under_superior_mm",
        "hidden_under_inferior_mm", "hidden_under_superior_mm",
        "max_hidden_undercontour_mm", "training_fov_clips_approved",
        "autoseg_fov_clips_approved", "autoseg_fov_clips_autoseg",
        "qa_unobservable_risk", "qa_unobservable_major_risk",
        "training_approved_foreground_center_fraction_inside_fov",
        "autoseg_fov_approved_foreground_center_fraction_inside_fov",
        "autoseg_fov_autoseg_foreground_center_fraction_inside_fov",
        "training_approved_touches_z_low", "training_approved_touches_z_high",
        "autoseg_fov_approved_touches_z_low", "autoseg_fov_approved_touches_z_high",
    ]
    ordered = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        w.writerows(rows)


def _fmt(x: object, digits: int = 1) -> str:
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "NA"
        return f"{xf:.{digits}f}"
    except Exception:
        return str(x)


def _write_report(path: Path, rows: List[Dict[str, object]], hidden_mm_threshold: float, major_threshold: float) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    training_clip = [r for r in ok if r.get("training_fov_clips_approved")]
    auto_clip = [r for r in ok if r.get("autoseg_fov_clips_approved")]
    risk = [r for r in ok if r.get("qa_unobservable_risk")]
    major = [r for r in ok if r.get("qa_unobservable_major_risk")]
    ranked = sorted(
        risk,
        key=lambda r: float(r.get("max_hidden_undercontour_mm", -math.inf)),
        reverse=True,
    )

    lines = [
        "# SpinalCord NSR-QA FOV visibility audit",
        "",
        "## Purpose",
        "",
        "This audit asks whether the fixed SpinalCord FOV can crop away clinically relevant superior/inferior extent errors before NSR-QA sees them.",
        "",
        "The audit uses physical-space foreground voxel centers; it does **not** interpret raw/output voxel-count ratios as retained volume when spacing differs.",
        "",
        "## Summary",
        "",
        f"- Successfully inspected cases: **{len(ok)}**",
        f"- Approved contours geometrically outside the original approved-derived training FOV: **{len(training_clip)}**",
        f"- Approved contours geometrically outside the primary autoseg-centered FOV: **{len(auto_clip)}**",
        f"- Cases with at least {hidden_mm_threshold:g} mm of approved-beyond-autoseg endpoint error hidden by the autoseg-centered FOV: **{len(risk)}**",
        f"- Cases with at least {major_threshold:g} mm hidden: **{len(major)}**",
        "",
        "`qa_unobservable_risk` means an approved-beyond-autoseg endpoint discrepancy exists in the raw masks but is reduced after applying the primary autoseg-centered FOV. It does not by itself determine whether the clinical contour or autosegmentation is the correct S/I extent convention.",
        "",
        "## Highest hidden endpoint discrepancies",
        "",
        "| patient | split | raw inferior under (mm) | visible inferior under (mm) | hidden inferior (mm) | raw superior under (mm) | hidden superior (mm) | training FOV clips approved | autoseg FOV clips approved |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in ranked[:30]:
        lines.append(
            "| {pid} | {split} | {ri} | {vi} | {hi} | {rs} | {hs} | {tc} | {ac} |".format(
                pid=r.get("patient_id", ""),
                split=r.get("split", ""),
                ri=_fmt(r.get("raw_under_inferior_mm")),
                vi=_fmt(r.get("visible_under_inferior_mm")),
                hi=_fmt(r.get("hidden_under_inferior_mm")),
                rs=_fmt(r.get("raw_under_superior_mm")),
                hs=_fmt(r.get("hidden_under_superior_mm")),
                tc=r.get("training_fov_clips_approved", ""),
                ac=r.get("autoseg_fov_clips_approved", ""),
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- `training_fov_clips_approved=True`: some raw approved SpinalCord foreground voxel centers lie outside the original approved-derived fixed FOV used to materialize NSR-QA data.",
        "- `autoseg_fov_clips_approved=True`: the candidate-centered deployment FOV excludes part of the raw approved contour.",
        "- `raw_under_*_mm`: endpoint extent by which approved extends beyond autoseg in the full raw CT geometry.",
        "- `visible_under_*_mm`: the corresponding discrepancy after both masks are resampled into the primary autoseg-centered NSR-QA FOV.",
        "- `hidden_under_*_mm = raw - visible`: the portion of that endpoint discrepancy lost because of the FOV.",
        "",
        "The S/I labels are used only when the crop z-axis is strongly aligned with physical LPS Z; otherwise neutral axis-low/axis-high columns remain authoritative.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Generated SpinalCord NSR-QA YAML")
    p.add_argument("--raw-data-root", default="/home/shaohan/radformation-data/data")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--splits", nargs="+", choices=["training1", "validation1", "testing1"],
        default=["training1", "validation1", "testing1"],
    )
    p.add_argument("--hidden-mm-threshold", type=float, default=5.0)
    p.add_argument("--major-hidden-mm-threshold", type=float, default=50.0)
    args = p.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cf = load_yaml(config_path)
    configured_roi = str(cf.get("data", {}).get("roi_name", ""))
    if configured_roi != ROI:
        raise ValueError(f"Expected SpinalCord config, got roi_name={configured_roi!r}: {config_path}")

    raw_root = Path(args.raw_data_root).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for split in args.splits:
        label_dir = split_label_dir(cf, split)
        files = sorted(label_dir.glob("*.nii.gz"))
        if not files:
            raise RuntimeError(f"No preprocessed SpinalCord labels found for {split}: {label_dir}")
        for template_path in files:
            pid = extract_patient_id(template_path)
            row = inspect_case(
                pid=pid,
                split=split,
                approved_template_path=template_path,
                raw_root=raw_root,
                hidden_mm_threshold=args.hidden_mm_threshold,
                major_hidden_mm_threshold=args.major_hidden_mm_threshold,
            )
            rows.append(row)
            print(
                f"[{split}] patient={pid} status={row.get('status')} "
                f"training_clip={row.get('training_fov_clips_approved', 'NA')} "
                f"autoseg_clip={row.get('autoseg_fov_clips_approved', 'NA')} "
                f"hidden_mm={_fmt(row.get('max_hidden_undercontour_mm'))}"
            )

    csv_path = outdir / "spinalcord_fov_visibility_cases.csv"
    report_path = outdir / "spinalcord_fov_visibility_report.md"
    _write_csv(csv_path, rows)
    _write_report(report_path, rows, args.hidden_mm_threshold, args.major_hidden_mm_threshold)

    summary = {
        "source_config": str(config_path),
        "raw_data_root": str(raw_root),
        "splits": args.splits,
        "hidden_mm_threshold": args.hidden_mm_threshold,
        "major_hidden_mm_threshold": args.major_hidden_mm_threshold,
        "n_rows": len(rows),
        "n_ok": sum(r.get("status") == "ok" for r in rows),
        "n_training_fov_clips_approved": sum(bool(r.get("training_fov_clips_approved")) for r in rows if r.get("status") == "ok"),
        "n_autoseg_fov_clips_approved": sum(bool(r.get("autoseg_fov_clips_approved")) for r in rows if r.get("status") == "ok"),
        "n_qa_unobservable_risk": sum(bool(r.get("qa_unobservable_risk")) for r in rows if r.get("status") == "ok"),
        "n_qa_unobservable_major_risk": sum(bool(r.get("qa_unobservable_major_risk")) for r in rows if r.get("status") == "ok"),
    }
    (outdir / "spinalcord_fov_visibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
