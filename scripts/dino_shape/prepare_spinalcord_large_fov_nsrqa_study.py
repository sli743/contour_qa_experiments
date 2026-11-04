#!/usr/bin/env python3
"""Prepare a SpinalCord-only NSR-QA study with a coverage-safe larger FOV.

The original study used 96x96x192 voxels at 1.5 mm for SpinalCord and centered
that crop on the foreground centroid.  FOV auditing showed that long/curved
cords can be clipped even after increasing Z alone, because centroid centering
is not guaranteed to center the structure's spatial extent.

This revision:
  * centers each approved SpinalCord crop on its physical bounding-box midpoint;
  * optionally resolves one global crop size from development (training/validation)
    approved contours plus a configurable physical margin;
  * audits per-axis/per-face clipping on all splits before any training config is
    generated; and
  * keeps all model/corruption/training settings unchanged.

By default, patient membership is copied from the original all-ROI preprocessed
label folders. With --split-config, the same established month-temporal policy as
prepare_month_temporal_nsrqa_study.py is used before the larger-FOV preprocessing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import SimpleITK as sitk
import yaml

SPLITS = ("training1", "validation1", "testing1")
ROI = "SpinalCord"
GROUP = "spine"
GLOBAL_LABEL_ID = 13
DEFAULT_STUDY_ROOT = Path("/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study")
DEFAULT_SOURCE_DATA_ROOT = Path("/home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom")
DEFAULT_RAW_ROOT = Path("/home/shaohan/radformation-data/data")
DEFAULT_REGISTRY = Path("configs/dino_shape/spinalcord_large_fov_nsrqa_registry.yaml")
# Interpreted as minimum sizes when --auto-size-from-approved is enabled.
DEFAULT_SIZE_XYZ = (96, 96, 384)
DEFAULT_SPACING_MM = 1.5
DEFAULT_MARGIN_MM_XYZ = (15.0, 15.0, 15.0)
DEFAULT_ROUND_TO = 16
DEFAULT_SIZING_SPLITS = ("training1", "validation1")
def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unique_ints(values: Iterable[int]) -> list[int]:
    return sorted({int(x) for x in values})


def choose_temporal_validation(
    month_ids: Sequence[int],
    held_out: set[int],
    fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    dev = np.array(sorted(set(map(int, month_ids)) - held_out), dtype=int)
    if len(dev) < 2:
        raise ValueError(f"Need at least 2 development patients; got {len(dev)}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(dev)
    n_val = int(round(len(dev) * fraction))
    n_val = max(1, min(len(dev) - 1, n_val))
    val = sorted(int(x) for x in shuffled[:n_val])
    train = sorted(int(x) for x in shuffled[n_val:])
    return train, val


def make_month_temporal_split(cfg: dict) -> tuple[dict[str, list[int]], list[dict]]:
    """Match prepare_month_temporal_nsrqa_study.py exactly.

    Month3 is always held out for testing. Month1 and Month2 independently
    contribute training/validation patients using the configured validation
    fraction and deterministic seed offsets. Month3 wins any cross-period
    duplicate to prevent temporal leakage.
    """
    month1 = unique_ints(cfg["month1"])
    month2 = unique_ints(cfg["month2"])
    month3 = unique_ints(cfg["month3"])
    test = set(month3)

    overlaps = []
    for month_name, ids in (("month1", month1), ("month2", month2)):
        for pid in sorted(set(ids) & test):
            overlaps.append({
                "patient_id": pid,
                "earlier_month": month_name,
                "later_month": "month3",
                "resolution": "testing1_month3_priority",
            })

    fraction = float(cfg.get("validation_fraction", 0.20))
    seed = int(cfg.get("seed", 88))
    tr1, va1 = choose_temporal_validation(month1, test, fraction, seed + 101)
    tr2, va2 = choose_temporal_validation(month2, test, fraction, seed + 202)
    split = {
        "training1": sorted(tr1 + tr2),
        "validation1": sorted(va1 + va2),
        "testing1": sorted(test),
    }

    train, val, test_ids = map(set, (split["training1"], split["validation1"], split["testing1"]))
    if train & val or train & test_ids or val & test_ids:
        raise AssertionError("Patient leakage remains after temporal split construction")
    return split, overlaps


def source_month(pid: int, cfg: dict) -> str:
    memberships = [m for m in ("month1", "month2", "month3") if pid in set(map(int, cfg[m]))]
    if "month3" in memberships:
        return "month3"
    return "+".join(memberships) if memberships else "unknown"


def sha256_json(obj: dict) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_tsv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def all_source_patient_ids(source_root: Path) -> set[int]:
    ids: set[int] = set()
    for split in SPLITS:
        ids.update(split_patient_ids(source_root, split))
    return ids


def resolve_temporal_split_for_spinalcord(source_root: Path, cfg: dict) -> tuple[dict[str, list[int]], dict]:
    """Build the established temporal split, then restrict to available SpinalCord cases.

    The assignment is made before ROI availability filtering, matching the generic
    all-ROI temporal study. Thus a missing SpinalCord case is audited as missing
    rather than causing a different validation draw for this ROI.
    """
    planned, overlaps = make_month_temporal_split(cfg)
    available = all_source_patient_ids(source_root)
    actual = {split: [pid for pid in ids if pid in available] for split, ids in planned.items()}
    missing = [
        {
            "patient_id": pid,
            "source_month": source_month(pid, cfg),
            "target_split": split,
            "reason": "missing_from_source_spinalcord_preprocessed_tree",
        }
        for split, ids in planned.items()
        for pid in ids
        if pid not in available
    ]
    for split in SPLITS:
        if not actual[split]:
            raise RuntimeError(f"Temporal split {split} has no available SpinalCord cases after source-data filtering")
    sets = {k: set(v) for k, v in actual.items()}
    if sets["training1"] & sets["validation1"] or sets["training1"] & sets["testing1"] or sets["validation1"] & sets["testing1"]:
        raise AssertionError("SpinalCord temporal split has patient leakage after availability filtering")
    meta = {
        "planned_split": planned,
        "actual_spinalcord_split": actual,
        "overlap_resolutions": overlaps,
        "missing_spinalcord_cases": missing,
        "available_source_patient_count": len(available),
    }
    return actual, meta


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
    default_value: float,
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


def binary_on_ct(mask_path: Path, ct: sitk.Image) -> sitk.Image:
    mask = sitk.ReadImage(str(mask_path))
    if not same_geometry(mask, ct):
        mask = resample_to_reference(mask, ct, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
    arr = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ct)
    return sitk.Cast(out, sitk.sitkUInt8)


def foreground_bbox_stats(mask: sitk.Image) -> dict:
    """Return foreground bbox in mask XYZ index coordinates and physical span.

    The output crop direction is copied from the CT/mask direction, so bbox
    extents in mask XYZ axes map directly to the output crop axes.
    """
    arr = sitk.GetArrayFromImage(mask) > 0
    coords_zyx = np.column_stack(np.where(arr))
    if coords_zyx.size == 0:
        raise ValueError("SpinalCord approved mask is empty")
    coords_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float64)
    min_xyz = coords_xyz.min(axis=0)
    max_xyz = coords_xyz.max(axis=0)
    midpoint_idx_xyz = 0.5 * (min_xyz + max_xyz)
    spacing_xyz = np.asarray(mask.GetSpacing(), dtype=np.float64)
    span_center_xyz_mm = (max_xyz - min_xyz) * spacing_xyz
    midpoint_phys_xyz = np.asarray(
        mask.TransformContinuousIndexToPhysicalPoint([float(v) for v in midpoint_idx_xyz]),
        dtype=np.float64,
    )
    return {
        "bbox_min_index_xyz": min_xyz,
        "bbox_max_index_xyz": max_xyz,
        "bbox_midpoint_index_xyz": midpoint_idx_xyz,
        "bbox_midpoint_phys_xyz": midpoint_phys_xyz,
        "bbox_center_span_xyz_mm": span_center_xyz_mm,
        "n_foreground_voxels": int(coords_xyz.shape[0]),
    }


def make_reference(
    template: sitk.Image,
    center_xyz: Sequence[float],
    size_xyz: Sequence[int],
    spacing_mm: float,
) -> sitk.Image:
    size = np.asarray(size_xyz, dtype=float)
    spacing = np.asarray([spacing_mm] * 3, dtype=float)
    direction = np.asarray(template.GetDirection(), dtype=float).reshape(3, 3)
    center = np.asarray(center_xyz, dtype=float)
    half_extent = 0.5 * (size - 1.0) * spacing
    origin = center - direction @ half_extent
    ref = sitk.Image(tuple(int(x) for x in size_xyz), sitk.sitkFloat32)
    ref.SetSpacing(tuple(float(x) for x in spacing))
    ref.SetDirection(tuple(float(x) for x in template.GetDirection()))
    ref.SetOrigin(tuple(float(x) for x in origin))
    return ref


def normalize_ct_to_uint8(ct: sitk.Image, clip_min: float = -1024.0, clip_max: float = 3324.0) -> sitk.Image:
    x = sitk.Cast(ct, sitk.sitkFloat32)
    x = sitk.Clamp(x, lowerBound=clip_min, upperBound=clip_max)
    x = (x - clip_min) / float(clip_max - clip_min) * 255.0
    return sitk.Cast(x, sitk.sitkUInt8)


def extract_patient_id(path: Path) -> int:
    m = re.search(r"patient_?(\d+)", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse patient id from {path}")
    return int(m.group(1))


def source_split_label_dir(source_root: Path, split: str) -> Path:
    return source_root / GROUP / ROI / split / GROUP / ROI / "labels"


def split_patient_ids(source_root: Path, split: str) -> list[int]:
    d = source_split_label_dir(source_root, split)
    files = sorted(d.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No source split labels found: {d}")
    ids = sorted({extract_patient_id(p) for p in files})
    return ids


def raw_paths(raw_root: Path, pid: int) -> tuple[Path, Path]:
    ct = raw_root / "images" / f"patient{pid}_image.nii.gz"
    approved = raw_root / "labels-approved" / f"patient{pid}_{ROI}_approved.nii.gz"
    return ct, approved


def raw_case_bbox(raw_root: Path, split: str, pid: int) -> dict:
    ct_path, approved_path = raw_paths(raw_root, pid)
    missing = [str(p) for p in (ct_path, approved_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw file(s): " + "; ".join(missing))
    ct = sitk.Cast(sitk.ReadImage(str(ct_path)), sitk.sitkFloat32)
    approved_ct = binary_on_ct(approved_path, ct)
    stats = foreground_bbox_stats(approved_ct)
    return {
        "patient_id": pid,
        "split": split,
        "raw_ct": str(ct_path),
        "raw_approved": str(approved_path),
        "bbox_min_index_xyz": json.dumps([float(v) for v in stats["bbox_min_index_xyz"]]),
        "bbox_max_index_xyz": json.dumps([float(v) for v in stats["bbox_max_index_xyz"]]),
        "bbox_midpoint_index_xyz": json.dumps([float(v) for v in stats["bbox_midpoint_index_xyz"]]),
        "bbox_midpoint_phys_xyz": json.dumps([float(v) for v in stats["bbox_midpoint_phys_xyz"]]),
        "bbox_center_span_x_mm": float(stats["bbox_center_span_xyz_mm"][0]),
        "bbox_center_span_y_mm": float(stats["bbox_center_span_xyz_mm"][1]),
        "bbox_center_span_z_mm": float(stats["bbox_center_span_xyz_mm"][2]),
        "n_foreground_voxels": int(stats["n_foreground_voxels"]),
    }


def round_up_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"round-to must be >0, got {multiple}")
    return int(math.ceil(value / multiple) * multiple)


def required_size_xyz_for_bbox(
    span_xyz_mm: Sequence[float],
    spacing_mm: float,
    margin_mm_xyz: Sequence[float],
    round_to: int,
) -> tuple[int, int, int]:
    out = []
    for span, margin in zip(span_xyz_mm, margin_mm_xyz):
        # Output voxel-center span is (N-1)*spacing. Require margin on both sides.
        n = int(math.ceil((float(span) + 2.0 * float(margin)) / float(spacing_mm))) + 1
        out.append(round_up_multiple(max(n, 1), round_to))
    return tuple(out)


def foreground_inside_stats(raw_mask: sitk.Image, ref: sitk.Image) -> dict:
    arr = sitk.GetArrayFromImage(raw_mask) > 0
    zyx = np.column_stack(np.where(arr))
    if zyx.size == 0:
        return {
            "fraction_all": float("nan"),
            "fraction_x": float("nan"),
            "fraction_y": float("nan"),
            "fraction_z": float("nan"),
            "outside_x_low": 0,
            "outside_x_high": 0,
            "outside_y_low": 0,
            "outside_y_high": 0,
            "outside_z_low": 0,
            "outside_z_high": 0,
        }
    xyz = zyx[:, [2, 1, 0]].astype(np.float64)

    src_spacing = np.asarray(raw_mask.GetSpacing(), dtype=np.float64)
    src_origin = np.asarray(raw_mask.GetOrigin(), dtype=np.float64)
    src_direction = np.asarray(raw_mask.GetDirection(), dtype=np.float64).reshape(3, 3)
    phys = src_origin[None, :] + (xyz * src_spacing[None, :]) @ src_direction.T

    ref_origin = np.asarray(ref.GetOrigin(), dtype=np.float64)
    ref_direction = np.asarray(ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    ref_spacing = np.asarray(ref.GetSpacing(), dtype=np.float64)
    ref_xyz = ((phys - ref_origin[None, :]) @ np.linalg.inv(ref_direction).T) / ref_spacing[None, :]
    ref_size = np.asarray(ref.GetSize(), dtype=np.float64)

    low = -0.5
    high = ref_size - 0.5
    inside_axis = (ref_xyz >= low) & (ref_xyz <= high[None, :])
    inside_all = np.all(inside_axis, axis=1)

    return {
        "fraction_all": float(inside_all.mean()),
        "fraction_x": float(inside_axis[:, 0].mean()),
        "fraction_y": float(inside_axis[:, 1].mean()),
        "fraction_z": float(inside_axis[:, 2].mean()),
        "outside_x_low": int((ref_xyz[:, 0] < low).sum()),
        "outside_x_high": int((ref_xyz[:, 0] > high[0]).sum()),
        "outside_y_low": int((ref_xyz[:, 1] < low).sum()),
        "outside_y_high": int((ref_xyz[:, 1] > high[1]).sum()),
        "outside_z_low": int((ref_xyz[:, 2] < low).sum()),
        "outside_z_high": int((ref_xyz[:, 2] > high[2]).sum()),
    }


def boundary_flags(mask_out: sitk.Image) -> dict:
    arr = sitk.GetArrayFromImage(mask_out) > 0  # z,y,x
    if not np.any(arr):
        return {k: False for k in ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high")} | {"n_boundary_voxels": 0}
    flags = {
        "x_low": bool(np.any(arr[:, :, 0])),
        "x_high": bool(np.any(arr[:, :, -1])),
        "y_low": bool(np.any(arr[:, 0, :])),
        "y_high": bool(np.any(arr[:, -1, :])),
        "z_low": bool(np.any(arr[0, :, :])),
        "z_high": bool(np.any(arr[-1, :, :])),
    }
    boundary = np.zeros_like(arr, dtype=bool)
    boundary[:, :, 0] |= arr[:, :, 0]
    boundary[:, :, -1] |= arr[:, :, -1]
    boundary[:, 0, :] |= arr[:, 0, :]
    boundary[:, -1, :] |= arr[:, -1, :]
    boundary[0, :, :] |= arr[0, :, :]
    boundary[-1, :, :] |= arr[-1, :, :]
    flags["n_boundary_voxels"] = int(boundary.sum())
    return flags


def output_paths(data_root: Path, split: str, pid: int) -> tuple[Path, Path]:
    base = data_root / GROUP / ROI / split / GROUP / ROI
    return base / "ct" / f"ct_patient_{pid}.nii.gz", base / "labels" / f"labels_patient_{pid}.nii.gz"


def write_label(mask: sitk.Image, path: Path) -> None:
    arr = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint16) * GLOBAL_LABEL_ID
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(out, sitk.sitkUInt16), str(path), True)


def preprocess_case(
    raw_root: Path,
    data_root: Path,
    split: str,
    pid: int,
    size_xyz: Sequence[int],
    spacing_mm: float,
) -> dict:
    ct_path, approved_path = raw_paths(raw_root, pid)
    missing = [str(p) for p in (ct_path, approved_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw file(s): " + "; ".join(missing))

    ct = sitk.Cast(sitk.ReadImage(str(ct_path)), sitk.sitkFloat32)
    approved_ct = binary_on_ct(approved_path, ct)
    bbox = foreground_bbox_stats(approved_ct)
    center = bbox["bbox_midpoint_phys_xyz"]
    ref = make_reference(ct, center, size_xyz, spacing_mm)

    inside = foreground_inside_stats(approved_ct, ref)
    approved_out = resample_to_reference(approved_ct, ref, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
    ct_out = resample_to_reference(ct, ref, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    ct_out = normalize_ct_to_uint8(ct_out)
    touches = boundary_flags(approved_out)

    out_ct, out_label = output_paths(data_root, split, pid)
    out_ct.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct_out, str(out_ct), True)
    write_label(approved_out, out_label)

    in_arr = sitk.GetArrayFromImage(approved_ct) > 0
    out_arr = sitk.GetArrayFromImage(approved_out) > 0
    raw_voxel_volume = float(np.prod(approved_ct.GetSpacing()))
    out_voxel_volume = float(np.prod(approved_out.GetSpacing()))
    raw_volume_mm3 = float(in_arr.sum()) * raw_voxel_volume
    out_volume_mm3 = float(out_arr.sum()) * out_voxel_volume
    span = bbox["bbox_center_span_xyz_mm"]

    return {
        "patient_id": pid,
        "split": split,
        "crop_size_xyz": json.dumps([int(v) for v in size_xyz]),
        "spacing_xyz": json.dumps([float(spacing_mm)] * 3),
        "physical_fov_xyz_mm": json.dumps([float(v) * float(spacing_mm) for v in size_xyz]),
        "center_phys_xyz": json.dumps([float(v) for v in center]),
        "center_definition": "approved_spinalcord_bbox_midpoint",
        "raw_approved_bbox_center_span_x_mm": float(span[0]),
        "raw_approved_bbox_center_span_y_mm": float(span[1]),
        "raw_approved_bbox_center_span_z_mm": float(span[2]),
        "approved_foreground_fraction_inside_fov": inside["fraction_all"],
        "approved_foreground_fraction_inside_x": inside["fraction_x"],
        "approved_foreground_fraction_inside_y": inside["fraction_y"],
        "approved_foreground_fraction_inside_z": inside["fraction_z"],
        "approved_outside_x_low_voxels": inside["outside_x_low"],
        "approved_outside_x_high_voxels": inside["outside_x_high"],
        "approved_outside_y_low_voxels": inside["outside_y_low"],
        "approved_outside_y_high_voxels": inside["outside_y_high"],
        "approved_outside_z_low_voxels": inside["outside_z_low"],
        "approved_outside_z_high_voxels": inside["outside_z_high"],
        "approved_volume_fraction_retained": (out_volume_mm3 / raw_volume_mm3) if raw_volume_mm3 > 0 else float("nan"),
        "approved_touches_x_low": touches["x_low"],
        "approved_touches_x_high": touches["x_high"],
        "approved_touches_y_low": touches["y_low"],
        "approved_touches_y_high": touches["y_high"],
        "approved_touches_z_low": touches["z_low"],
        "approved_touches_z_high": touches["z_high"],
        "approved_boundary_voxels": touches["n_boundary_voxels"],
        "raw_ct": str(ct_path),
        "raw_approved": str(approved_path),
        "output_ct": str(out_ct),
        "output_label": str(out_label),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def annotate_generated_configs(
    study_root: Path,
    size_xyz: Sequence[int],
    spacing_mm: float,
    source_root: Path,
    auto_size: bool,
    sizing_splits: Sequence[str],
    margin_mm_xyz: Sequence[float],
    round_to: int,
    split_metadata: Optional[dict] = None,
) -> None:
    cfg_dir = study_root / "configs" / ROI
    for p in sorted(cfg_dir.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        meta = dict(cfg.get("study_metadata") or {})
        meta.update({
            "fov_revision": "spinalcord_coverage_safe_fov_v2",
            "fov_size_xyz": [int(v) for v in size_xyz],
            "fov_spacing_xyz_mm": [float(spacing_mm)] * 3,
            "fov_physical_size_xyz_mm": [float(v) * float(spacing_mm) for v in size_xyz],
            "fov_center_definition": "approved_spinalcord_bbox_midpoint",
            "fov_auto_size_from_approved": bool(auto_size),
            "fov_sizing_splits": list(sizing_splits),
            "fov_margin_mm_xyz_per_side": [float(v) for v in margin_mm_xyz],
            "fov_round_to_voxels": int(round_to),
            "original_comparator_fov_size_xyz": [96, 96, 192],
            "source_patient_split_root": str(source_root),
            "patient_split_mode": (split_metadata or {}).get("split_mode", "source_existing"),
            "temporal_split_config": (split_metadata or {}).get("split_config", ""),
            "testing1_semantics": (split_metadata or {}).get("testing1_semantics", "source testing1"),
            "development_semantics": (split_metadata or {}).get("development_semantics", "source training1 + validation1"),
            "revision_reason": "avoid 3D clipping of long/curved SpinalCord contours and hidden endpoint errors",
        })
        cfg["study_metadata"] = meta
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", default=str(DEFAULT_STUDY_ROOT))
    ap.add_argument("--source-data-root", default=str(DEFAULT_SOURCE_DATA_ROOT), help="Original 96x96x192 preprocessed tree; used only to copy patient split membership")
    ap.add_argument("--split-config", default=None, help="Optional month-temporal split YAML. When supplied, use the established Month1+Month2 development / Month3 test policy instead of copying source split membership")
    ap.add_argument("--raw-data-root", default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--crop-size-xyz", type=int, nargs=3, default=list(DEFAULT_SIZE_XYZ), help="Fixed crop size, or per-axis minimum when --auto-size-from-approved is enabled")
    ap.add_argument("--spacing-mm", type=float, default=DEFAULT_SPACING_MM)
    ap.add_argument("--auto-size-from-approved", action="store_true", help="Resolve one global crop size from approved SpinalCord bboxes in --sizing-splits")
    ap.add_argument("--sizing-splits", nargs="+", choices=list(SPLITS), default=list(DEFAULT_SIZING_SPLITS), help="Development splits allowed to determine the global FOV")
    ap.add_argument("--margin-mm-xyz", type=float, nargs=3, default=list(DEFAULT_MARGIN_MM_XYZ), help="Physical margin per side added to approved bbox during auto sizing")
    ap.add_argument("--round-to", type=int, default=DEFAULT_ROUND_TO, help="Round each auto-sized voxel dimension up to this multiple")
    ap.add_argument("--overwrite-data", action="store_true")
    ap.add_argument("--allow-approved-clipping", action="store_true", help="Do not fail if any raw approved SpinalCord voxel centers remain outside the revised FOV")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    study_root = Path(args.study_root).expanduser().resolve()
    source_root = Path(args.source_data_root).expanduser().resolve()
    raw_root = Path(args.raw_data_root).expanduser().resolve()
    registry = (project_root / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry).resolve()
    min_size_xyz = tuple(int(v) for v in args.crop_size_xyz)
    spacing_mm = float(args.spacing_mm)
    margin_mm_xyz = tuple(float(v) for v in args.margin_mm_xyz)
    sizing_splits = tuple(args.sizing_splits)
    if any(v <= 0 for v in min_size_xyz):
        raise ValueError(f"Invalid crop size: {min_size_xyz}")
    if any(v < 0 for v in margin_mm_xyz):
        raise ValueError(f"Margins must be >=0: {margin_mm_xyz}")
    if args.round_to <= 0:
        raise ValueError(f"--round-to must be >0, got {args.round_to}")

    manifest_dir = study_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    split_metadata: dict = {
        "split_mode": "source_existing",
        "testing1_semantics": "source testing1",
        "development_semantics": "source training1 + validation1",
    }
    temporal_cfg = None
    if args.split_config:
        split_cfg_path = Path(args.split_config)
        if not split_cfg_path.is_absolute():
            split_cfg_path = project_root / split_cfg_path
        split_cfg_path = split_cfg_path.expanduser().resolve()
        temporal_cfg = load_yaml(split_cfg_path)
        split_ids, temporal_meta = resolve_temporal_split_for_spinalcord(source_root, temporal_cfg)
        split_metadata.update({
            "split_mode": "month_temporal",
            "split_config": str(split_cfg_path),
            "testing1_semantics": "month3 temporal holdout",
            "development_semantics": "month1 + month2; validation sampled separately within each month",
            "validation_fraction": float(temporal_cfg.get("validation_fraction", 0.20)),
            "validation_strategy": temporal_cfg.get("validation_strategy", "stratified_by_month"),
            "overlap_policy": temporal_cfg.get("overlap_policy", "month3_test_priority"),
            "seed": int(temporal_cfg.get("seed", 88)),
            **temporal_meta,
        })
    else:
        split_ids = {split: split_patient_ids(source_root, split) for split in SPLITS}
        sets = {k: set(v) for k, v in split_ids.items()}
        overlaps = []
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                both = sorted(sets[a] & sets[b])
                if both:
                    overlaps.append({"split_a": a, "split_b": b, "patient_ids": ",".join(map(str, both))})
        if overlaps:
            raise RuntimeError(f"Source SpinalCord split contains overlap(s): {overlaps}")

    if temporal_cfg is not None:
        patient_rows = []
        for target_split, ids in split_metadata["planned_split"].items():
            actual = set(split_ids[target_split])
            for pid in ids:
                patient_rows.append({
                    "patient_id": pid,
                    "source_month": source_month(pid, temporal_cfg),
                    "target_split": target_split,
                    "held_out_test": target_split == "testing1",
                    "spinalcord_available": pid in actual,
                })
        patient_rows.sort(key=lambda r: (r["target_split"], int(r["patient_id"])))
        write_tsv(
            manifest_dir / "temporal_split_patients.tsv",
            patient_rows,
            ["patient_id", "source_month", "target_split", "held_out_test", "spinalcord_available"],
        )
        write_tsv(
            manifest_dir / "temporal_split_overlap_resolution.tsv",
            split_metadata["overlap_resolutions"],
            ["patient_id", "earlier_month", "later_month", "resolution"],
        )
        write_tsv(
            manifest_dir / "temporal_split_missing_spinalcord.tsv",
            split_metadata["missing_spinalcord_cases"],
            ["patient_id", "source_month", "target_split", "reason"],
        )
        split_payload = {
            "split_config": split_metadata["split_config"],
            "source_data_root": str(source_root),
            "study_root": str(study_root),
            "seed": split_metadata["seed"],
            "validation_fraction": split_metadata["validation_fraction"],
            "validation_strategy": split_metadata["validation_strategy"],
            "overlap_policy": split_metadata["overlap_policy"],
            "planned_split": split_metadata["planned_split"],
            "actual_spinalcord_split": split_ids,
            "overlap_resolutions": split_metadata["overlap_resolutions"],
            "missing_spinalcord_cases": split_metadata["missing_spinalcord_cases"],
        }
        split_payload["split_sha256"] = sha256_json(split_payload)
        (manifest_dir / "temporal_split_summary.json").write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")
        md = [
            "# SpinalCord large-FOV month-temporal split",
            "",
            "This uses the same temporal policy as `prepare_month_temporal_nsrqa_study.py`.",
            "",
            f"- training1: **{len(split_ids['training1'])}** available SpinalCord patients from Month1+Month2",
            f"- validation1: **{len(split_ids['validation1'])}** available SpinalCord patients from Month1+Month2",
            f"- testing1: **{len(split_ids['testing1'])}** available SpinalCord patients = Month3 temporal holdout",
            f"- validation fraction: **{split_metadata['validation_fraction']:.2f}**, sampled independently within Month1 and Month2",
            f"- overlap policy: **{split_metadata['overlap_policy']}**",
            "",
            "Month3 is never used to select the FOV when the default sizing splits are training1+validation1.",
        ]
        if split_metadata["overlap_resolutions"]:
            md += ["", "## Cross-period duplicate IDs", "", "| patient | earlier period | resolution |", "|---:|---|---|"]
            md += [f"| {r['patient_id']} | {r['earlier_month']} | {r['resolution']} |" for r in split_metadata["overlap_resolutions"]]
        if split_metadata["missing_spinalcord_cases"]:
            md += ["", "## Planned temporal patients unavailable for SpinalCord", "", f"Count: **{len(split_metadata['missing_spinalcord_cases'])}**. See `temporal_split_missing_spinalcord.tsv`."]
        (manifest_dir / "temporal_split_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # First pass: characterize raw approved bbox extent for every case.  Auto-size
    # uses development splits only; testing1 remains a coverage audit, not a sizing input.
    sizing_rows = []
    for split in SPLITS:
        for pid in split_ids[split]:
            row = raw_case_bbox(raw_root, split, pid)
            span = (
                row["bbox_center_span_x_mm"],
                row["bbox_center_span_y_mm"],
                row["bbox_center_span_z_mm"],
            )
            req = required_size_xyz_for_bbox(span, spacing_mm, margin_mm_xyz, args.round_to)
            row.update({
                "required_crop_x_vox_with_margin": req[0],
                "required_crop_y_vox_with_margin": req[1],
                "required_crop_z_vox_with_margin": req[2],
                "used_for_auto_sizing": split in sizing_splits,
            })
            sizing_rows.append(row)

    sizing_path = manifest_dir / "spinalcord_fov_sizing_audit.csv"
    write_csv(sizing_path, sizing_rows)

    size_xyz = min_size_xyz
    if args.auto_size_from_approved:
        dev_rows = [r for r in sizing_rows if r["split"] in sizing_splits]
        if not dev_rows:
            raise RuntimeError(f"No cases in sizing splits: {sizing_splits}")
        required_dev = tuple(
            max(int(r[f"required_crop_{axis}_vox_with_margin"]) for r in dev_rows)
            for axis in ("x", "y", "z")
        )
        size_xyz = tuple(max(min_size_xyz[i], required_dev[i]) for i in range(3))

    data_root = study_root / "data" / f"preprocessed_spinalcord_{spacing_mm:g}mm_fov{size_xyz[0]}x{size_xyz[1]}x{size_xyz[2]}"
    if args.overwrite_data and data_root.exists():
        shutil.rmtree(data_root)

    print(f"Resolved crop XYZ: {size_xyz} vox @ {spacing_mm} mm")
    print(f"Physical FOV XYZ: {[float(v) * spacing_mm for v in size_xyz]} mm")
    print(f"Center definition: approved SpinalCord bbox midpoint")
    if args.auto_size_from_approved:
        print(f"Auto-sized from: {', '.join(sizing_splits)}; margin/side XYZ={margin_mm_xyz} mm; round_to={args.round_to}")

    audit_rows = []
    for split in SPLITS:
        (manifest_dir / f"{split}_patients.txt").write_text("\n".join(map(str, split_ids[split])) + "\n", encoding="utf-8")
        print(f"{split}: {len(split_ids[split])} patients")
        for pid in split_ids[split]:
            row = preprocess_case(raw_root, data_root, split, pid, size_xyz, spacing_mm)
            audit_rows.append(row)
            print(
                f"  patient {pid}: bbox_span_xyz="
                f"({row['raw_approved_bbox_center_span_x_mm']:.1f},"
                f" {row['raw_approved_bbox_center_span_y_mm']:.1f},"
                f" {row['raw_approved_bbox_center_span_z_mm']:.1f}) mm, "
                f"inside={row['approved_foreground_fraction_inside_fov']:.6f}, "
                f"inside_xyz=({row['approved_foreground_fraction_inside_x']:.6f},"
                f"{row['approved_foreground_fraction_inside_y']:.6f},"
                f"{row['approved_foreground_fraction_inside_z']:.6f})"
            )

    audit_path = manifest_dir / "spinalcord_large_fov_coverage_audit.csv"
    write_csv(audit_path, audit_rows)
    clipped = [r for r in audit_rows if float(r["approved_foreground_fraction_inside_fov"]) < 1.0 - 1e-12]
    edge_touch = [
        r for r in audit_rows
        if any(bool(r[f"approved_touches_{axis}_{side}"]) for axis in ("x", "y", "z") for side in ("low", "high"))
    ]

    all_required = tuple(
        max(int(r[f"required_crop_{axis}_vox_with_margin"]) for r in sizing_rows)
        for axis in ("x", "y", "z")
    )
    fov_spec = {
        "roi": ROI,
        "group": GROUP,
        "crop_size_xyz": list(size_xyz),
        "minimum_requested_crop_size_xyz": list(min_size_xyz),
        "spacing_xyz_mm": [spacing_mm] * 3,
        "physical_fov_xyz_mm": [float(v) * spacing_mm for v in size_xyz],
        "center_to_center_span_xyz_mm": [(float(v) - 1.0) * spacing_mm for v in size_xyz],
        "center_definition": "approved_spinalcord_bbox_midpoint",
        "auto_size_from_approved": bool(args.auto_size_from_approved),
        "sizing_splits": list(sizing_splits),
        "margin_mm_xyz_per_side": list(margin_mm_xyz),
        "round_to_voxels": int(args.round_to),
        "required_crop_xyz_if_all_splits_were_used": list(all_required),
        "comparator_crop_size_xyz": [96, 96, 192],
        "n_cases": len(audit_rows),
        "n_approved_clipped_any_axis": len(clipped),
        "n_approved_touching_any_output_face": len(edge_touch),
        "source_split_root": str(source_root),
        "patient_split_mode": split_metadata.get("split_mode", "source_existing"),
        "temporal_split_config": split_metadata.get("split_config", ""),
        "testing1_semantics": split_metadata.get("testing1_semantics", "source testing1"),
        "development_semantics": split_metadata.get("development_semantics", "source training1 + validation1"),
        "raw_data_root": str(raw_root),
        "data_root": str(data_root),
        "sizing_audit": str(sizing_path),
    }
    (manifest_dir / "spinalcord_large_fov_spec.json").write_text(json.dumps(fov_spec, indent=2), encoding="utf-8")

    if clipped and not args.allow_approved_clipping:
        ids = ", ".join(str(r["patient_id"]) for r in clipped[:20])
        raise RuntimeError(
            f"Coverage-safe FOV still excludes raw approved SpinalCord voxel centers for {len(clipped)} case(s): {ids}. "
            f"Resolved size={size_xyz}; all-split bbox+margin requirement would be {all_required}. "
            "Inspect per-axis outside counts in the coverage audit. Do not bypass this check for the primary study. "
            f"Audit: {audit_path}"
        )

    prep_cmd = [
        sys.executable,
        str(project_root / "scripts/dino_shape/prepare_all_roi_nsrqa_study.py"),
        "--registry", str(registry),
        "--study-root", str(study_root),
        "--data-root", str(data_root),
    ]
    subprocess.run(prep_cmd, cwd=project_root, check=True)
    annotate_generated_configs(
        study_root,
        size_xyz,
        spacing_mm,
        source_root,
        args.auto_size_from_approved,
        sizing_splits,
        margin_mm_xyz,
        args.round_to,
        split_metadata,
    )

    print("\nPrepared SpinalCord coverage-safe large-FOV NSR-QA study")
    print(f"  study root: {study_root}")
    print(f"  data root:  {data_root}")
    print(f"  FOV XYZ:    {size_xyz} vox @ {spacing_mm} mm = {[v * spacing_mm for v in size_xyz]} mm")
    print(f"  split mode: {split_metadata.get('split_mode', 'source_existing')}")
    if split_metadata.get("split_mode") == "month_temporal":
        print(f"  temporal:   train={len(split_ids['training1'])} val={len(split_ids['validation1'])} test={len(split_ids['testing1'])}")
        print(f"  split audit:{manifest_dir / 'temporal_split_summary.json'}")
    print(f"  sizing:     {sizing_path}")
    print(f"  audit:      {audit_path}")
    print(f"  configs:    {study_root / 'configs' / ROI}")
    print("Inspect the audits and standardized corruption examples before training.")


if __name__ == "__main__":
    main()
