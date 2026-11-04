#!/usr/bin/env python3
"""Preprocess real autosegmentation contours for NSR-QA regression experiments.

Two localization modes are supported:

``autoseg_centered`` (primary)
    Build the fixed-size ROI/group crop around the available autosegmentation
    center ROIs. The approved contour is accessed only *after* the reference
    grid has been fixed, and is resampled onto that grid for evaluation.

``autoseg_xy_ct_z`` (deployable reference-frame diagnostic)
    Derive the in-plane (X/Y) center from the available autosegmentation, as in
    ``autoseg_centered``, but replace the longitudinal/slice-axis component with
    the raw CT acquisition midpoint.  This preserves a candidate-derived lateral
    localization while preventing a shortened/overextended candidate from
    normalizing its own S/I position by recentering the crop.

``approved_reference`` (oracle-localized sensitivity analysis)
    Reuse the exact geometry of the existing preprocessed approved contour and
    resample the raw autosegmentation onto that grid.

The crop size/spacing/direction come from the already-materialized NSR-QA data
configuration so this script remains compatible with the trained decoder. In
``autoseg_centered`` mode, the approved contour does not determine the crop
origin/center.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk
import yaml


CENTER_ROIS_BY_GROUP: Dict[str, Sequence[str]] = {
    "airway": ("Bronchus", "Carina", "Trachea"),
    "heart_core": ("Heart", "A_Aorta", "A_Pulmonary"),
    "mediastinum_tubes": ("Esophagus", "V_VenaCava_S"),
    "lungs_chest": ("Lung_L", "Lung_R"),
    "brachial": ("BrachPlx_L", "BrachPlx_R"),
    "spine": ("SpinalCord",),
    "upper_abdomen": ("Liver", "Stomach"),
}


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_patient_id(path: Path) -> str:
    m = re.search(r"patient_?(\d+)", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse patient id from {path}")
    return m.group(1)


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    default_value: float = 0.0,
    pixel_id: Optional[int] = None,
) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference)
    r.SetTransform(sitk.Transform())
    r.SetInterpolator(interpolator)
    r.SetDefaultPixelValue(default_value)
    if pixel_id is not None:
        r.SetOutputPixelType(pixel_id)
    return r.Execute(moving)


def binary_on_ct(mask_path: Path, ct_img: sitk.Image) -> sitk.Image:
    mask = sitk.ReadImage(str(mask_path))
    if not same_geometry(mask, ct_img):
        mask = resample_to_reference(mask, ct_img, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    arr = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ct_img)
    return sitk.Cast(out, sitk.sitkUInt8)


def union_center_physical(masks: Iterable[sitk.Image]) -> Optional[np.ndarray]:
    """Foreground-centroid center (legacy/default behavior)."""
    ref = None
    union = None
    for m in masks:
        ref = m
        arr = sitk.GetArrayFromImage(m) > 0
        union = arr.copy() if union is None else (union | arr)
    if ref is None or union is None or not np.any(union):
        return None
    zyx = np.column_stack(np.where(union))
    center_zyx = zyx.mean(axis=0)
    idx_xyz = [float(center_zyx[2]), float(center_zyx[1]), float(center_zyx[0])]
    return np.asarray(ref.TransformContinuousIndexToPhysicalPoint(idx_xyz), dtype=float)


def union_bbox_midpoint_physical(masks: Iterable[sitk.Image]) -> Optional[np.ndarray]:
    """Center on the midpoint of the union foreground bounding box.

    This is useful for long structures such as SpinalCord: unlike the foreground
    centroid, it guarantees symmetric geometric headroom around the observed
    candidate extent for a fixed-size crop.
    """
    ref = None
    union = None
    for m in masks:
        ref = m
        arr = sitk.GetArrayFromImage(m) > 0
        union = arr.copy() if union is None else (union | arr)
    if ref is None or union is None or not np.any(union):
        return None
    zyx = np.column_stack(np.where(union))
    lo = zyx.min(axis=0).astype(float)
    hi = zyx.max(axis=0).astype(float)
    midpoint_zyx = 0.5 * (lo + hi)
    idx_xyz = [float(midpoint_zyx[2]), float(midpoint_zyx[1]), float(midpoint_zyx[0])]
    return np.asarray(ref.TransformContinuousIndexToPhysicalPoint(idx_xyz), dtype=float)


def image_center_physical(img: sitk.Image) -> np.ndarray:
    """Physical center of an image grid using the center-to-center convention."""
    size = np.asarray(img.GetSize(), dtype=float)
    idx = 0.5 * (size - 1.0)
    return np.asarray(
        img.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in idx)),
        dtype=float,
    )


def candidate_inplane_ct_slice_center(
    candidate_center_xyz: Sequence[float],
    ct_img: sitk.Image,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Keep candidate in-plane localization but anchor slice-axis position to CT.

    This is deliberately defined in the CT image basis rather than assuming that
    the third array axis is exactly physical +Z.  Let ``s`` be the CT slice-axis
    direction vector (the third direction-cosine column).  Starting from the
    candidate-derived center, replace only the coordinate along ``s`` with the
    raw CT grid midpoint.  The two orthogonal/in-plane components therefore
    remain candidate-derived.
    """
    candidate = np.asarray(candidate_center_xyz, dtype=float)
    ct_center = image_center_physical(ct_img)
    direction = np.asarray(ct_img.GetDirection(), dtype=float).reshape(3, 3)
    slice_axis = direction[:, 2].astype(float)
    norm = float(np.linalg.norm(slice_axis))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid CT slice-axis direction vector")
    slice_axis /= norm
    delta_along_slice = float(np.dot(ct_center - candidate, slice_axis))
    center = candidate + delta_along_slice * slice_axis
    meta = {
        "ct_center_x_mm": float(ct_center[0]),
        "ct_center_y_mm": float(ct_center[1]),
        "ct_center_z_mm": float(ct_center[2]),
        "slice_axis_x": float(slice_axis[0]),
        "slice_axis_y": float(slice_axis[1]),
        "slice_axis_z": float(slice_axis[2]),
        "candidate_to_ct_center_slice_shift_mm": delta_along_slice,
        "slice_axis_alignment_with_physical_z_abs": float(abs(slice_axis[2])),
    }
    return center, meta


def make_reference_from_template_center(template: sitk.Image, center_xyz: Sequence[float]) -> sitk.Image:
    size = np.asarray(template.GetSize(), dtype=float)
    spacing = np.asarray(template.GetSpacing(), dtype=float)
    direction = np.asarray(template.GetDirection(), dtype=float).reshape(3, 3)
    center = np.asarray(center_xyz, dtype=float)
    half_extent = 0.5 * (size - 1.0) * spacing
    origin = center - direction @ half_extent
    ref = sitk.Image(tuple(int(x) for x in template.GetSize()), sitk.sitkFloat32)
    ref.SetSpacing(tuple(float(x) for x in template.GetSpacing()))
    ref.SetDirection(tuple(float(x) for x in template.GetDirection()))
    ref.SetOrigin(tuple(float(x) for x in origin))
    return ref


def write_binary(img: sitk.Image, path: Path) -> None:
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(out, sitk.sitkUInt8), str(path), True)


def split_label_dir(cf: Dict, split: str) -> Path:
    key = {
        "training1": "train_label_dir",
        "validation1": "validation_label_dir",
        "testing1": "test_label_dir",
    }[split]
    return Path(cf["data"][key]).expanduser().resolve()


def raw_label(raw_root: Path, pid: str, roi: str, kind: str) -> Path:
    suffix = "approved" if kind == "approved" else "autoseg"
    folder = "labels-approved" if kind == "approved" else "labels-autoseg"
    return raw_root / folder / f"patient{pid}_{roi}_{suffix}.nii.gz"


def process_case(
    *,
    pid: str,
    split: str,
    roi: str,
    group: str,
    approved_preprocessed: Path,
    raw_root: Path,
    output_root: Path,
    mode: str,
    center_method: str = "foreground_centroid",
) -> Dict:
    case_dir = output_root / mode / roi / split / f"patient_{pid}"
    case_dir.mkdir(parents=True, exist_ok=True)

    raw_ct_path = raw_root / "images" / f"patient{pid}_image.nii.gz"
    raw_autoseg_path = raw_label(raw_root, pid, roi, "autoseg")
    raw_approved_path = raw_label(raw_root, pid, roi, "approved")
    required = [raw_ct_path, raw_autoseg_path, raw_approved_path, approved_preprocessed]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return {
            "patient_id": pid,
            "split": split,
            "roi": roi,
            "group": group,
            "localization_mode": mode,
            "status": "skipped_missing_file",
            "reason": ";".join(missing),
        }

    ct = sitk.Cast(sitk.ReadImage(str(raw_ct_path)), sitk.sitkFloat32)
    approved_template = sitk.ReadImage(str(approved_preprocessed))
    autoseg_ct = binary_on_ct(raw_autoseg_path, ct)
    approved_ct = binary_on_ct(raw_approved_path, ct)

    autoseg_vox_raw = int((sitk.GetArrayFromImage(autoseg_ct) > 0).sum())
    if autoseg_vox_raw == 0 and mode in {"autoseg_centered", "autoseg_xy_ct_z"}:
        return {
            "patient_id": pid,
            "split": split,
            "roi": roi,
            "group": group,
            "localization_mode": mode,
            "status": "skipped_empty_autoseg_no_center",
            "reason": "ROI autosegmentation is empty; candidate-derived crop center is undefined",
        }

    center_rois_used: List[str] = []
    center_xyz: Optional[np.ndarray] = None
    candidate_center_xyz: Optional[np.ndarray] = None
    hybrid_meta: Dict[str, float] = {}
    if mode == "approved_reference":
        ref = approved_template
    else:
        center_masks: List[sitk.Image] = []
        for center_roi in CENTER_ROIS_BY_GROUP.get(group, (roi,)):
            p = raw_label(raw_root, pid, center_roi, "autoseg")
            if not p.is_file():
                continue
            m = binary_on_ct(p, ct)
            if np.any(sitk.GetArrayFromImage(m) > 0):
                center_masks.append(m)
                center_rois_used.append(center_roi)
        if not center_masks:
            # Deployment-safe fallback: use the candidate ROI itself, never approved.
            center_masks = [autoseg_ct]
            center_rois_used = [roi]
        if center_method == "bbox_midpoint":
            candidate_center_xyz = union_bbox_midpoint_physical(center_masks)
        else:
            candidate_center_xyz = union_center_physical(center_masks)
        if candidate_center_xyz is None:
            return {
                "patient_id": pid,
                "split": split,
                "roi": roi,
                "group": group,
                "localization_mode": mode,
                "status": "skipped_empty_center_union",
                "reason": "All available candidate center masks are empty",
            }
        if mode == "autoseg_xy_ct_z":
            center_xyz, hybrid_meta = candidate_inplane_ct_slice_center(candidate_center_xyz, ct)
        else:
            center_xyz = candidate_center_xyz
        ref = make_reference_from_template_center(approved_template, center_xyz)

    ct_out = resample_to_reference(ct, ref, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    autoseg_out = resample_to_reference(autoseg_ct, ref, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    if mode == "approved_reference":
        approved_out = approved_template
    else:
        approved_out = resample_to_reference(approved_ct, ref, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    sitk.WriteImage(ct_out, str(case_dir / "ct.nii.gz"), True)
    write_binary(approved_out, case_dir / "approved.nii.gz")
    write_binary(autoseg_out, case_dir / "autoseg_input.nii.gz")

    # Geometry must be exactly common after export.
    saved_ct = sitk.ReadImage(str(case_dir / "ct.nii.gz"))
    saved_app = sitk.ReadImage(str(case_dir / "approved.nii.gz"))
    saved_auto = sitk.ReadImage(str(case_dir / "autoseg_input.nii.gz"))
    geometry_ok = same_geometry(saved_ct, saved_app) and same_geometry(saved_ct, saved_auto)
    if not geometry_ok:
        raise RuntimeError(f"Output geometry mismatch for {roi} patient {pid} mode {mode}")

    return {
        "patient_id": pid,
        "split": split,
        "roi": roi,
        "group": group,
        "localization_mode": mode,
        "status": "ok",
        "reason": "",
        "center_rois_used": ",".join(center_rois_used),
        "center_method": (
            center_method
            if mode == "autoseg_centered"
            else (f"{center_method}_xy_plus_ct_slice_midpoint" if mode == "autoseg_xy_ct_z" else "approved_reference")
        ),
        "center_phys_xyz": json.dumps([float(x) for x in center_xyz]) if center_xyz is not None else "approved_reference",
        "candidate_center_phys_xyz": json.dumps([float(x) for x in candidate_center_xyz]) if candidate_center_xyz is not None else "approved_reference",
        **hybrid_meta,
        "raw_ct": str(raw_ct_path),
        "raw_autoseg": str(raw_autoseg_path),
        "raw_approved": str(raw_approved_path),
        "approved_template": str(approved_preprocessed),
        "case_dir": str(case_dir),
        "output_size_xyz": json.dumps(list(saved_ct.GetSize())),
        "output_spacing_xyz": json.dumps(list(saved_ct.GetSpacing())),
        "output_origin_xyz": json.dumps(list(saved_ct.GetOrigin())),
        "output_direction": json.dumps(list(saved_ct.GetDirection())),
        "raw_autoseg_voxels": autoseg_vox_raw,
        "output_autoseg_voxels": int((sitk.GetArrayFromImage(saved_auto) > 0).sum()),
        "output_approved_voxels": int((sitk.GetArrayFromImage(saved_app) > 0).sum()),
        "geometry_verified": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Generated NSR-QA YAML for one ROI")
    p.add_argument("--roi", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--raw-data-root", default="/home/shaohan/radformation-data/data")
    p.add_argument("--output-root", required=True)
    p.add_argument(
        "--mode",
        choices=["autoseg_centered", "autoseg_xy_ct_z", "approved_reference", "both", "all"],
        default="both",
        help=(
            "Localization geometry. 'both' preserves the historical two-mode behavior; "
            "'all' additionally materializes the candidate-X/Y + CT-slice-axis diagnostic mode."
        ),
    )
    p.add_argument("--center-method", choices=["foreground_centroid", "bbox_midpoint"], default="foreground_centroid", help="Localization center for autoseg_centered mode; default preserves the original study")
    p.add_argument("--splits", nargs="+", choices=["training1", "validation1", "testing1"], default=["training1", "validation1", "testing1"])
    args = p.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cf = load_yaml(config_path)
    raw_root = Path(args.raw_data_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    if args.mode == "both":
        modes = ["autoseg_centered", "approved_reference"]
    elif args.mode == "all":
        modes = ["autoseg_centered", "autoseg_xy_ct_z", "approved_reference"]
    else:
        modes = [args.mode]

    rows: List[Dict] = []
    for split in args.splits:
        label_dir = split_label_dir(cf, split)
        files = sorted(label_dir.glob("*.nii.gz"))
        if not files:
            raise RuntimeError(f"No approved preprocessed labels found for {args.roi} {split}: {label_dir}")
        for approved_preprocessed in files:
            pid = extract_patient_id(approved_preprocessed)
            for mode in modes:
                row = process_case(
                    pid=pid,
                    split=split,
                    roi=args.roi,
                    group=args.group,
                    approved_preprocessed=approved_preprocessed,
                    raw_root=raw_root,
                    output_root=out_root,
                    mode=mode,
                    center_method=args.center_method,
                )
                rows.append(row)
                print(f"[{args.roi}] {split} patient={pid} mode={mode}: {row['status']}")

    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.roi}_preprocess_manifest.csv"
    keys = sorted({k for row in rows for k in row})
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    metadata = {
        "source_config": str(config_path),
        "roi": args.roi,
        "group": args.group,
        "raw_data_root": str(raw_root),
        "output_root": str(out_root),
        "modes": modes,
        "splits": args.splits,
        "primary_localization_mode": "autoseg_centered",
        "approved_reference_role": "oracle_localization_sensitivity_analysis",
        "autoseg_xy_ct_z_role": "deployable_reference_frame_diagnostic_candidate_inplane_plus_raw_ct_slice_midpoint",
        "center_rois": list(CENTER_ROIS_BY_GROUP.get(args.group, (args.roi,))),
        "center_method": args.center_method,
        "note": "In autoseg_centered mode, approved contours are accessed only after the reference grid is fixed.",
    }
    (manifest_dir / f"{args.roi}_preprocess_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
