#!/usr/bin/env python3
"""Audit whether the fixed lungs_chest crop clips Lung_L (or another lung ROI).

``radformation_preprocess_roiwise.py`` crops every lungs_chest ROI with one
fixed box (192x192x160 voxels at 1.5 mm = 288x288x240 mm) centered on the
foreground centroid of the approved Lung_L + Lung_R union.  A contour that is
partly outside that box is still written; only fully empty crops are skipped.
``preprocess_actual_autoseg.py`` then reuses the same box size, centered on
the autoseg Lung_L + Lung_R centroid, for the real-autoseg test crops.

For every patient in an NSR-QA config this script compares the *raw* masks with
both geometries in physical space:

1. ``training``: the stored approved-derived label crop (the model's training,
   validation and test target).
2. ``autoseg_fov``: the autoseg-centered crop used for real-autoseg QA.

Per geometry and mask it reports the raw volume, the volume outside the box
(mL and %), which faces clip the contour, how many millimetres extend beyond
each face, and which faces the saved nearest-neighbour crop touches.  Volumes
come from voxel counts times the voxel volume of each grid, so the raw CT grid
and the 1.5-mm crop grid are never compared voxel-for-voxel.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from analysis.inspect_spinalcord_fov_visibility import (
    _fmt,
    _foreground_physical_points,
    _physical_to_continuous_index,
    _prefixed,
)
from evaluation.actual_autoseg.preprocess_actual_autoseg import (
    CENTER_ROIS_BY_GROUP,
    binary_on_ct,
    extract_patient_id,
    image_center_physical,
    load_yaml,
    make_reference_from_template_center,
    raw_label,
    resample_to_reference,
    split_label_dir,
    union_center_physical,
)

# SimpleITK physical space is LPS: +x left, +y posterior, +z superior.
LPS_AXIS_NAMES = (("right", "left"), ("anterior", "posterior"), ("inferior", "superior"))
GEOMETRIES = ("training_approved", "autoseg_fov_approved", "autoseg_fov_autoseg")


def _face_names(ref: sitk.Image) -> List[Tuple[str, str]]:
    """Anatomical (low, high) face names for each crop index axis x, y, z."""
    direction = np.asarray(ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    names = []
    for k in range(3):
        col = direction[:, k]
        j = int(np.argmax(np.abs(col)))
        low, high = LPS_AXIS_NAMES[j]
        if col[j] < 0:
            low, high = high, low
        names.append((low, high))
    return names


def _clip_stats(mask: sitk.Image, ref: sitk.Image, faces: Sequence[Tuple[str, str]]) -> Dict[str, object]:
    raw_voxel_ml = float(np.prod(mask.GetSpacing())) / 1000.0
    points = _foreground_physical_points(mask)
    n = int(points.shape[0])
    out: Dict[str, object] = {"raw_volume_ml": n * raw_voxel_ml}
    if n == 0:
        out.update({"volume_outside_ml": math.nan, "lost_pct": math.nan, "clipped_faces": "", "touched_faces": ""})
        return out

    ci = _physical_to_continuous_index(points, ref)
    size = np.asarray(ref.GetSize(), dtype=np.float64)
    spacing = np.asarray(ref.GetSpacing(), dtype=np.float64)
    # Physical FOV boundaries lie half a voxel outside the first/last centers.
    below = ci < -0.5
    above = ci > (size - 0.5)[None, :]
    outside = np.any(below | above, axis=1)

    out["volume_outside_ml"] = float(outside.sum()) * raw_voxel_ml
    out["lost_pct"] = 100.0 * float(outside.mean())
    clipped: List[str] = []
    for k, (low, high) in enumerate(faces):
        out[f"outside_{low}_ml"] = float(below[:, k].sum()) * raw_voxel_ml
        out[f"outside_{high}_ml"] = float(above[:, k].sum()) * raw_voxel_ml
        out[f"beyond_{low}_mm"] = max(0.0, -0.5 - float(ci[:, k].min())) * float(spacing[k])
        out[f"beyond_{high}_mm"] = max(0.0, float(ci[:, k].max()) - (size[k] - 0.5)) * float(spacing[k])
        if below[:, k].any():
            clipped.append(low)
        if above[:, k].any():
            clipped.append(high)
    out["clipped_faces"] = ",".join(clipped)

    # What the saved crop actually keeps after nearest-neighbour resampling.
    crop = sitk.GetArrayFromImage(resample_to_reference(mask, ref, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
    out["crop_volume_ml"] = float(crop.sum()) * float(np.prod(spacing)) / 1000.0
    touched: List[str] = []
    for k, (low, high) in enumerate(faces):
        axis = 2 - k  # array order is zyx
        if np.take(crop, 0, axis=axis).any():
            touched.append(low)
        if np.take(crop, -1, axis=axis).any():
            touched.append(high)
    out["touched_faces"] = ",".join(touched)
    return out


def _severity(lost_pct: object, minor_pct: float, major_pct: float) -> str:
    try:
        x = float(lost_pct)
    except (TypeError, ValueError):
        return "not_measured"
    if not np.isfinite(x):
        return "not_measured"
    if x >= major_pct:
        return "major"
    if x >= minor_pct:
        return "minor"
    return "trace" if x > 0 else "none"


def inspect_case(
    *,
    pid: str,
    split: str,
    roi: str,
    center_rois: Sequence[str],
    template_path: Path,
    raw_root: Path,
    minor_pct: float,
    major_pct: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "patient_id": pid,
        "split": split,
        "roi": roi,
        "status": "ok",
        "reason": "",
        "approved_template": str(template_path),
    }
    raw_ct_path = raw_root / "images" / f"patient{pid}_image.nii.gz"
    raw_app_path = raw_label(raw_root, pid, roi, "approved")
    missing = [str(p) for p in (raw_ct_path, raw_app_path, template_path) if not p.is_file()]
    if missing:
        row["status"] = "skipped_missing_file"
        row["reason"] = ";".join(missing)
        return row

    ct = sitk.ReadImage(str(raw_ct_path))
    template = sitk.ReadImage(str(template_path))
    faces = _face_names(template)
    row["crop_size_xyz"] = json.dumps(list(template.GetSize()))
    row["crop_spacing_xyz"] = json.dumps(list(template.GetSpacing()))
    row["crop_edge_span_xyz_mm"] = json.dumps([float(s * d) for s, d in zip(template.GetSize(), template.GetSpacing())])

    approved = {roi: binary_on_ct(raw_app_path, ct)}
    autoseg: Dict[str, sitk.Image] = {}
    for c in center_rois:
        app_p = raw_label(raw_root, pid, c, "approved")
        if c not in approved and app_p.is_file():
            approved[c] = binary_on_ct(app_p, ct)
        auto_p = raw_label(raw_root, pid, c, "autoseg")
        if auto_p.is_file():
            autoseg[c] = binary_on_ct(auto_p, ct)
    auto_p = raw_label(raw_root, pid, roi, "autoseg")
    if roi not in autoseg and auto_p.is_file():
        autoseg[roi] = binary_on_ct(auto_p, ct)

    _prefixed(row, "training_approved", _clip_stats(approved[roi], template, faces))

    # The generator centered the box on the approved centre-ROI centroid; a large
    # offset means the stored crop was not built the way this audit assumes.
    app_centre_used = [c for c in center_rois if c in approved] or [roi]
    expected = union_center_physical([approved[c] for c in app_centre_used])
    row["training_center_rois"] = ",".join(app_centre_used)
    row["training_center_offset_mm"] = (
        float(np.linalg.norm(image_center_physical(template) - expected)) if expected is not None else math.nan
    )

    auto_centre_used = [c for c in center_rois if c in autoseg]
    center = union_center_physical([autoseg[c] for c in auto_centre_used]) if auto_centre_used else None
    if center is None:
        row["reason"] = "no raw autoseg for centre ROIs; autoseg-centered FOV not audited"
    else:
        auto_ref = make_reference_from_template_center(template, center)
        row["autoseg_fov_center_rois"] = ",".join(auto_centre_used)
        row["autoseg_fov_center_phys_xyz"] = json.dumps([float(x) for x in center])
        _prefixed(row, "autoseg_fov_approved", _clip_stats(approved[roi], auto_ref, faces))
        if roi in autoseg:
            _prefixed(row, "autoseg_fov_autoseg", _clip_stats(autoseg[roi], auto_ref, faces))

    for g in GEOMETRIES:
        row[f"{g}_severity"] = _severity(row.get(f"{g}_lost_pct"), minor_pct, major_pct)
    return row


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "patient_id", "split", "roi", "status", "reason", "crop_size_xyz", "crop_edge_span_xyz_mm",
        "training_approved_severity", "training_approved_lost_pct", "training_approved_raw_volume_ml",
        "training_approved_volume_outside_ml", "training_approved_clipped_faces", "training_approved_touched_faces",
        "autoseg_fov_approved_severity", "autoseg_fov_approved_lost_pct", "autoseg_fov_approved_clipped_faces",
        "autoseg_fov_autoseg_severity", "autoseg_fov_autoseg_lost_pct", "autoseg_fov_autoseg_clipped_faces",
        "training_center_offset_mm",
    ]
    ordered = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        w.writerows(rows)


def _max_beyond(row: Dict[str, object], prefix: str) -> float:
    vals = [float(v) for k, v in row.items() if k.startswith(f"{prefix}_beyond_") and isinstance(v, (int, float))]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else math.nan


def _write_report(path: Path, rows: List[Dict[str, object]], roi: str, minor_pct: float, major_pct: float, top_n: int) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    splits = sorted({str(r["split"]) for r in ok})
    labels = {
        "training_approved": "Stored training crop, approved contour",
        "autoseg_fov_approved": "Autoseg-centered crop, approved contour",
        "autoseg_fov_autoseg": "Autoseg-centered crop, autoseg contour",
    }
    lines = [
        f"# {roi} FOV clipping audit",
        "",
        f"Patients audited: **{len(ok)}** (skipped or failed: {len(rows) - len(ok)}).",
        f"Severity uses the share of raw foreground volume outside the crop: "
        f"major >= {major_pct:g}%, minor >= {minor_pct:g}%, trace > 0%.",
        "",
        "## Clipped patients by split",
        "",
        "| Geometry | Split | n | major | minor | trace | none |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in GEOMETRIES:
        for s in splits:
            sev = Counter(r.get(f"{g}_severity") for r in ok if r["split"] == s)
            n = sum(v for k, v in sev.items() if k != "not_measured")
            if n == 0:
                continue
            lines.append(f"| {labels[g]} | {s} | {n} | {sev['major']} | {sev['minor']} | {sev['trace']} | {sev['none']} |")

    lines += ["", "## Faces that clip (patients with any loss)", "", "| Geometry | Face counts |", "|---|---|"]
    for g in GEOMETRIES:
        faces = Counter(f for r in ok for f in str(r.get(f"{g}_clipped_faces", "")).split(",") if f)
        if faces:
            lines.append(f"| {labels[g]} | " + ", ".join(f"{k}: {v}" for k, v in faces.most_common()) + " |")

    def lost(r: Dict[str, object], g: str) -> float:
        try:
            x = float(r.get(f"{g}_lost_pct", math.nan))
        except (TypeError, ValueError):
            return math.nan
        return x if np.isfinite(x) else math.nan

    worst = sorted(ok, key=lambda r: -np.nan_to_num(lost(r, "training_approved"), nan=-1.0))[:top_n]
    lines += [
        "",
        f"## Worst {len(worst)} patients (stored training crop)",
        "",
        "| Patient | Split | Raw mL | Lost % | Lost mL | Clipped faces | Max beyond (mm) | Autoseg-FOV approved lost % | Autoseg-FOV autoseg lost % |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in worst:
        lines.append(
            f"| {r['patient_id']} | {r['split']} | {_fmt(r.get('training_approved_raw_volume_ml'), 0)} | "
            f"{_fmt(lost(r, 'training_approved'))} | {_fmt(r.get('training_approved_volume_outside_ml'), 0)} | "
            f"{r.get('training_approved_clipped_faces') or '-'} | {_fmt(_max_beyond(r, 'training_approved'))} | "
            f"{_fmt(lost(r, 'autoseg_fov_approved'))} | {_fmt(lost(r, 'autoseg_fov_autoseg'))} |"
        )

    offsets = [float(r["training_center_offset_mm"]) for r in ok
               if isinstance(r.get("training_center_offset_mm"), float) and np.isfinite(r["training_center_offset_mm"])]
    if offsets:
        lines += [
            "",
            "## Sanity check",
            "",
            f"Distance between the stored crop center and the recomputed approved centre-ROI centroid: "
            f"median {np.median(offsets):.2f} mm, max {max(offsets):.2f} mm. Values above a voxel (1.5 mm) mean "
            "the stored crop was not centered the way this audit assumes.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="NSR-QA YAML whose data.*_label_dir entries hold the stored crops")
    p.add_argument("--roi", default="Lung_L")
    p.add_argument("--group", default="lungs_chest", help="Selects the centre ROIs used by both crop geometries")
    p.add_argument("--raw-data-root", default="/home/shaohan/radformation-data/data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["training1", "validation1", "testing1"])
    p.add_argument("--patients", nargs="+", default=None, help="Optional patient IDs, e.g. 62")
    p.add_argument("--minor-loss-pct", type=float, default=1.0)
    p.add_argument("--major-loss-pct", type=float, default=5.0)
    p.add_argument("--top-n", type=int, default=25)
    args = p.parse_args()

    cf = load_yaml(Path(args.config))
    raw_root = Path(args.raw_data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    center_rois = list(CENTER_ROIS_BY_GROUP.get(args.group, (args.roi,)))
    keep = {str(int(x)) for x in args.patients} if args.patients else None

    rows: List[Dict[str, object]] = []
    for split in args.splits:
        label_dir = split_label_dir(cf, split)
        paths = sorted(label_dir.glob("*.nii.gz"))
        print(f"[{split}] {len(paths)} stored crops in {label_dir}", flush=True)
        for path in paths:
            pid = extract_patient_id(path)
            if keep is not None and str(int(pid)) not in keep:
                continue
            try:
                row = inspect_case(
                    pid=pid, split=split, roi=args.roi, center_rois=center_rois, template_path=path,
                    raw_root=raw_root, minor_pct=args.minor_loss_pct, major_pct=args.major_loss_pct,
                )
            except Exception as e:  # keep auditing the rest of the cohort
                row = {"patient_id": pid, "split": split, "roi": args.roi, "status": "error", "reason": repr(e)}
            rows.append(row)
            print(
                f"  patient {pid}: training lost {_fmt(row.get('training_approved_lost_pct'), 2)}% "
                f"[{row.get('training_approved_clipped_faces') or '-'}], autoseg-FOV approved lost "
                f"{_fmt(row.get('autoseg_fov_approved_lost_pct'), 2)}% ({row['status']})"
                + (f" {row['reason']}" if row["status"] == "error" else ""),
                flush=True,
            )

    csv_path = out_dir / f"{args.roi}_fov_clipping_audit.csv"
    report_path = out_dir / f"{args.roi}_fov_clipping_audit.md"
    _write_csv(csv_path, rows)
    _write_report(report_path, rows, args.roi, args.minor_loss_pct, args.major_loss_pct, args.top_n)
    (out_dir / f"{args.roi}_fov_clipping_audit_metadata.json").write_text(json.dumps({
        "config": str(args.config),
        "roi": args.roi,
        "group": args.group,
        "center_rois": center_rois,
        "raw_data_root": str(raw_root),
        "splits": args.splits,
        "patients_filter": sorted(keep) if keep else None,
        "minor_loss_pct": args.minor_loss_pct,
        "major_loss_pct": args.major_loss_pct,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"CSV:    {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
