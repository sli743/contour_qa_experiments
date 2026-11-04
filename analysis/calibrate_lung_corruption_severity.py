#!/usr/bin/env python3
"""Calibrate legacy Lung_L synthetic-corruption severity from real training errors.

The corruption vocabulary is intentionally fixed.  The script answers:

1. Which old corruption family is geometrically compatible with each real
   approved-vs-autoseg error?
2. What severity range (native parameter and realized physical extent) best
   matches the training data for each family?
3. Does severity depend on Lung_L size strongly enough to justify conditioning?
4. Are historical cut names anatomically misleading for the source images?

Month3/testing1 is never used for calibration.  New cut probes use explicit
anatomical names on LPS-canonical arrays.  Historical ``cut_upper_half`` and
``cut_lower_half`` semantics are preserved only for audit/reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from scipy import ndimage, stats

from analysis.spinalcord_diagnostic_utils import pair_metrics
from training.corruptions.benchmark_gpu import make_corruption_gpu


STRUCT6 = ndimage.generate_binary_structure(3, 1)

FAMILY_FEATURES: Dict[str, Sequence[str]] = {
    "shifted": (
        "one_minus_dice", "missing_fraction_gt", "excess_fraction_gt",
        "centroid_shift_si_mm", "centroid_shift_ap_mm", "centroid_shift_lr_mm",
        "relative_volume_change_signed",
    ),
    "eroded": (
        "one_minus_dice", "missing_fraction_gt", "excess_fraction_gt",
        "relative_volume_change_signed", "bbox_extent_si_change_mm",
        "bbox_extent_ap_change_mm", "bbox_extent_lr_change_mm",
    ),
    "dilated": (
        "one_minus_dice", "missing_fraction_gt", "excess_fraction_gt",
        "relative_volume_change_signed", "bbox_extent_si_change_mm",
        "bbox_extent_ap_change_mm", "bbox_extent_lr_change_mm",
    ),
    "cut_inferior_portion": (
        "one_minus_dice", "missing_fraction_gt", "inferior_missing_mm",
        "bbox_extent_si_change_mm", "centroid_shift_si_mm",
    ),
    "cut_superior_portion": (
        "one_minus_dice", "missing_fraction_gt", "superior_missing_mm",
        "bbox_extent_si_change_mm", "centroid_shift_si_mm",
    ),
    "cut_posterior_portion": (
        "one_minus_dice", "missing_fraction_gt", "posterior_missing_mm",
        "bbox_extent_ap_change_mm", "centroid_shift_ap_mm",
    ),
    "big_bite": (
        "one_minus_dice", "missing_fraction_gt", "excess_fraction_gt",
        "largest_missing_component_fraction_gt", "largest_missing_extent_si_mm",
        "largest_missing_extent_ap_mm", "largest_missing_extent_lr_mm",
    ),
    "big_blob": (
        "one_minus_dice", "missing_fraction_gt", "excess_fraction_gt",
        "largest_excess_component_fraction_gt", "largest_excess_extent_si_mm",
        "largest_excess_extent_ap_mm", "largest_excess_extent_lr_mm",
    ),
}

BANK_FEATURES = (
    "one_minus_dice", "missing_fraction_gt", "excess_fraction_gt",
    "relative_volume_change_signed", "centroid_shift_mm",
    "bbox_extent_si_change_mm", "bbox_extent_ap_change_mm", "bbox_extent_lr_change_mm",
    "inferior_missing_mm", "superior_missing_mm", "posterior_missing_mm",
    "largest_missing_component_fraction_gt", "largest_excess_component_fraction_gt",
)

STOCHASTIC_FAMILIES = {"shifted", "big_bite", "big_blob"}
INTEGER_LEVEL_FAMILIES = {"shifted", "eroded", "dilated"}


@dataclass
class CaseData:
    patient_id: int
    cohort: str
    ct: sitk.Image
    reference: sitk.Image
    approved: np.ndarray
    autoseg: np.ndarray
    spacing_zyx: Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def stable_seed(*parts: object, base: int = 88) -> int:
    s = "|".join(str(x) for x in parts)
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int(base + int.from_bytes(h[:4], "little"))


def unique_ints(values: Iterable[int]) -> List[int]:
    return sorted({int(x) for x in values})


def choose_validation(month_ids: Sequence[int], held_out: set[int], fraction: float, seed: int):
    dev = np.array(sorted(set(map(int, month_ids)) - held_out), dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(dev)
    n_val = max(1, min(len(dev) - 1, int(round(len(dev) * fraction))))
    return sorted(map(int, shuffled[n_val:])), sorted(map(int, shuffled[:n_val]))


def build_temporal_split(split_yaml: Path) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    cfg = yaml.safe_load(split_yaml.read_text())
    month1 = unique_ints(cfg["month1"])
    month2 = unique_ints(cfg["month2"])
    month3 = unique_ints(cfg["month3"])
    held = set(month3)
    frac = float(cfg.get("validation_fraction", 0.20))
    seed = int(cfg.get("seed", 88))
    tr1, va1 = choose_validation(month1, held, frac, seed + 101)
    tr2, va2 = choose_validation(month2, held, frac, seed + 202)
    split = {
        "training1": sorted(tr1 + tr2),
        "validation1": sorted(va1 + va2),
        "testing1": sorted(held),
    }
    cohort: Dict[int, str] = {}
    for pid in month1:
        cohort.setdefault(int(pid), "month1")
    for pid in month2:
        cohort.setdefault(int(pid), "month2")
    for pid in month3:
        cohort[int(pid)] = "month3"
    return split, cohort


def resolve_case_paths(root: Path, pid: int, roi: str) -> Tuple[Path, Path, Path]:
    exact = (
        root / "images" / f"patient{pid}_image.nii.gz",
        root / "labels-approved" / f"patient{pid}_{roi}_approved.nii.gz",
        root / "labels-autoseg" / f"patient{pid}_{roi}_autoseg.nii.gz",
    )
    if all(p.is_file() for p in exact):
        return exact

    def one(directory: Path, patterns: Sequence[str]) -> Path:
        hits: List[Path] = []
        for pat in patterns:
            hits.extend(directory.glob(pat))
        hits = sorted(set(hits))
        if len(hits) != 1:
            raise FileNotFoundError(f"Expected one file under {directory} for {patterns}; found {hits}")
        return hits[0]

    return (
        one(root / "images", [f"*patient*{pid}*image*.nii.gz", f"*patient_{pid}*.nii.gz"]),
        one(root / "labels-approved", [f"*patient*{pid}*{roi}*approved*.nii.gz"]),
        one(root / "labels-autoseg", [f"*patient*{pid}*{roi}*autoseg*.nii.gz"]),
    )


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def resample_to_reference(img: sitk.Image, ref: sitk.Image, interp: int, pixel_type: int) -> sitk.Image:
    if same_geometry(img, ref):
        return sitk.Cast(img, pixel_type)
    return sitk.Resample(img, ref, sitk.Transform(), interp, 0, pixel_type)


def isotropic_reference(img: sitk.Image, spacing_mm: float) -> sitk.Image:
    old_size = np.asarray(img.GetSize(), dtype=float)
    old_spacing = np.asarray(img.GetSpacing(), dtype=float)
    new_spacing = np.repeat(float(spacing_mm), 3)
    new_size = np.maximum(1, np.round((old_size - 1.0) * old_spacing / new_spacing).astype(int) + 1)
    ref = sitk.Image([int(v) for v in new_size], sitk.sitkFloat32)
    ref.SetSpacing(tuple(float(v) for v in new_spacing))
    ref.SetOrigin(img.GetOrigin())
    ref.SetDirection(img.GetDirection())
    return ref


def crop_to_union(ct: sitk.Image, approved: sitk.Image, autoseg: sitk.Image, margin_mm: float):
    a = sitk.GetArrayFromImage(approved) > 0
    b = sitk.GetArrayFromImage(autoseg) > 0
    union = a | b
    pts = np.argwhere(union)
    if not len(pts):
        raise ValueError("Empty approved/autoseg union")
    spacing_zyx = np.asarray(approved.GetSpacing()[::-1], dtype=float)
    margin = np.ceil(float(margin_mm) / spacing_zyx).astype(int)
    lo = np.maximum(0, pts.min(axis=0) - margin)
    hi = np.minimum(np.asarray(union.shape) - 1, pts.max(axis=0) + margin)
    index_xyz = [int(lo[2]), int(lo[1]), int(lo[0])]
    size_xyz = [int(hi[2] - lo[2] + 1), int(hi[1] - lo[1] + 1), int(hi[0] - lo[0] + 1)]
    return tuple(sitk.RegionOfInterest(x, size_xyz, index_xyz) for x in (ct, approved, autoseg))


def load_case(raw_root: Path, pid: int, roi: str, cohort: str, spacing_mm: float, margin_mm: float) -> CaseData:
    ct_path, app_path, auto_path = resolve_case_paths(raw_root, pid, roi)
    ct = sitk.DICOMOrient(sitk.ReadImage(str(ct_path), sitk.sitkFloat32), "LPS")
    approved = sitk.DICOMOrient(sitk.ReadImage(str(app_path)), "LPS")
    autoseg = sitk.DICOMOrient(sitk.ReadImage(str(auto_path)), "LPS")
    approved = resample_to_reference(approved, ct, sitk.sitkNearestNeighbor, sitk.sitkUInt8)
    autoseg = resample_to_reference(autoseg, ct, sitk.sitkNearestNeighbor, sitk.sitkUInt8)

    iso_ref = isotropic_reference(ct, spacing_mm)
    ct = resample_to_reference(ct, iso_ref, sitk.sitkLinear, sitk.sitkFloat32)
    approved = resample_to_reference(approved, iso_ref, sitk.sitkNearestNeighbor, sitk.sitkUInt8)
    autoseg = resample_to_reference(autoseg, iso_ref, sitk.sitkNearestNeighbor, sitk.sitkUInt8)
    ct, approved, autoseg = crop_to_union(ct, approved, autoseg, margin_mm)

    a = sitk.GetArrayFromImage(approved) > 0
    b = sitk.GetArrayFromImage(autoseg) > 0
    if not a.any() or not b.any():
        raise ValueError(f"Empty {roi} after preprocessing for patient {pid}")
    spacing_zyx = tuple(float(v) for v in approved.GetSpacing()[::-1])
    return CaseData(pid, cohort, ct, approved, a, b, spacing_zyx)


def bbox(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    if not len(pts):
        return np.full(3, np.nan), np.full(3, np.nan)
    return pts.min(axis=0).astype(float), pts.max(axis=0).astype(float)


def centroid(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    return pts.mean(axis=0) if len(pts) else np.full(3, np.nan)


def largest_component_features(mask: np.ndarray, spacing: np.ndarray, gt_vox: int, prefix: str) -> Dict[str, float]:
    labels, n = ndimage.label(mask, structure=STRUCT6)
    if n == 0:
        return {
            f"largest_{prefix}_component_fraction_gt": 0.0,
            f"largest_{prefix}_extent_si_mm": 0.0,
            f"largest_{prefix}_extent_ap_mm": 0.0,
            f"largest_{prefix}_extent_lr_mm": 0.0,
        }
    counts = ndimage.sum(mask, labels, range(1, n + 1))
    lab = int(np.argmax(counts)) + 1
    comp = labels == lab
    pts = np.argwhere(comp)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return {
        f"largest_{prefix}_component_fraction_gt": float(comp.sum() / max(gt_vox, 1)),
        f"largest_{prefix}_extent_si_mm": float((hi[0] - lo[0] + 1) * spacing[0]),
        f"largest_{prefix}_extent_ap_mm": float((hi[1] - lo[1] + 1) * spacing[1]),
        f"largest_{prefix}_extent_lr_mm": float((hi[2] - lo[2] + 1) * spacing[2]),
    }


def cheap_features(approved: np.ndarray, candidate: np.ndarray, spacing_zyx: Sequence[float], components: bool = True) -> Dict[str, float]:
    g = np.asarray(approved, bool)
    a = np.asarray(candidate, bool)
    spacing = np.asarray(spacing_zyx, dtype=float)
    gv, av = int(g.sum()), int(a.sum())
    inter = int((g & a).sum())
    d = 2.0 * inter / max(gv + av, 1)
    missing, excess = g & ~a, a & ~g
    glo, ghi = bbox(g)
    alo, ahi = bbox(a)
    gc, ac = centroid(g), centroid(a)
    delta = (ac - gc) * spacing
    gext = (ghi - glo + 1) * spacing
    aext = (ahi - alo + 1) * spacing

    inf_missing = max(float((alo[0] - glo[0]) * spacing[0]), 0.0)
    sup_missing = max(float((ghi[0] - ahi[0]) * spacing[0]), 0.0)
    # LPS NumPy y increases anterior->posterior; high-y removal is posterior truncation.
    posterior_missing = max(float((ghi[1] - ahi[1]) * spacing[1]), 0.0)
    out = {
        "dice": float(d),
        "one_minus_dice": float(1.0 - d),
        "missing_fraction_gt": float(missing.sum() / max(gv, 1)),
        "excess_fraction_gt": float(excess.sum() / max(gv, 1)),
        "changed_fraction_gt": float((missing | excess).sum() / max(gv, 1)),
        "relative_volume_change_signed": float((av - gv) / max(gv, 1)),
        "centroid_shift_si_mm": float(delta[0]),
        "centroid_shift_ap_mm": float(delta[1]),
        "centroid_shift_lr_mm": float(delta[2]),
        "centroid_shift_mm": float(np.linalg.norm(delta)),
        "bbox_extent_si_gt_mm": float(gext[0]),
        "bbox_extent_ap_gt_mm": float(gext[1]),
        "bbox_extent_lr_gt_mm": float(gext[2]),
        "bbox_extent_si_change_mm": float(aext[0] - gext[0]),
        "bbox_extent_ap_change_mm": float(aext[1] - gext[1]),
        "bbox_extent_lr_change_mm": float(aext[2] - gext[2]),
        "inferior_missing_mm": inf_missing,
        "superior_missing_mm": sup_missing,
        "posterior_missing_mm": posterior_missing,
        "approved_volume_mm3": float(gv * np.prod(spacing)),
    }
    if components:
        out.update(largest_component_features(missing, spacing, gv, "missing"))
        out.update(largest_component_features(excess, spacing, gv, "excess"))
    else:
        for prefix in ("missing", "excess"):
            out.update({
                f"largest_{prefix}_component_fraction_gt": np.nan,
                f"largest_{prefix}_extent_si_mm": np.nan,
                f"largest_{prefix}_extent_ap_mm": np.nan,
                f"largest_{prefix}_extent_lr_mm": np.nan,
            })
    return out


def error_overlap(approved: np.ndarray, real: np.ndarray, synthetic: np.ndarray) -> Dict[str, float]:
    rm, re = approved & ~real, real & ~approved
    sm, se = approved & ~synthetic, synthetic & ~approved

    def dice_mask(x, y):
        den = int(x.sum()) + int(y.sum())
        if den == 0:
            return 1.0
        return float(2 * (x & y).sum() / den)

    md, ed = dice_mask(rm, sm), dice_mask(re, se)
    mr, er = int(rm.sum()), int(re.sum())
    total = mr + er
    if total == 0:
        wd = 1.0 if int(sm.sum()) + int(se.sum()) == 0 else 0.0
    else:
        wd = float((mr * md + er * ed) / total)
    return {"missing_error_dice": md, "excess_error_dice": ed, "weighted_error_dice": wd}


def fit_scaler(real_rows: Sequence[Mapping[str, float]]) -> Dict[str, Tuple[float, float]]:
    scaler = {}
    for feature in sorted(set(BANK_FEATURES) | {x for vals in FAMILY_FEATURES.values() for x in vals}):
        vals = np.asarray([float(r.get(feature, np.nan)) for r in real_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            scaler[feature] = (0.0, 1.0)
            continue
        med = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        scale = float(q75 - q25)
        if scale < 1e-6:
            mad = float(np.median(np.abs(vals - med))) * 1.4826
            scale = mad if mad >= 1e-6 else max(float(np.std(vals)), 1e-3)
        scaler[feature] = (med, scale)
    return scaler


def feature_distance(real: Mapping[str, float], syn: Mapping[str, float], features: Sequence[str], scaler) -> float:
    diffs = []
    for f in features:
        rv, sv = float(real.get(f, np.nan)), float(syn.get(f, np.nan))
        if not (np.isfinite(rv) and np.isfinite(sv)):
            continue
        _, scale = scaler[f]
        diffs.append((sv - rv) / scale)
    return float(np.sqrt(np.mean(np.square(diffs)))) if diffs else np.inf


def candidate_from_family(approved: np.ndarray, family: str, level: float, seed: int) -> np.ndarray:
    x = torch.from_numpy(np.asarray(approved, np.float32))[None]
    rng = random.Random(int(seed))

    # The calibration/report uses anatomically explicit names, but older
    # benchmark_gpu.py revisions only know the historical cut names. Preserve
    # the historical dispatcher semantics at this call boundary:
    #   cut_upper_half -> removes low NumPy-z -> inferior truncation in LPS
    #   cut_lower_half -> removes high NumPy-z -> superior truncation in LPS
    # This keeps calibration runnable even if the central anatomical-alias
    # compatibility patch has not been applied to benchmark_gpu.py.
    dispatcher_family = {
        "cut_inferior_portion": "cut_upper_half",
        "cut_superior_portion": "cut_lower_half",
        "cut_posterior_portion": "cut_posterior_half",
    }.get(family, family)

    y, _ = make_corruption_gpu(x, dispatcher_family, float(level), rng)
    return y[0].cpu().numpy() > 0.5


def raw_orientation_audit(img: sitk.Image, pid: int) -> Dict[str, object]:
    direction = tuple(float(v) for v in img.GetDirection())
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(direction)
    size = img.GetSize()
    center = [int((s - 1) // 2) for s in size]

    def physical_for_axis(itk_axis: int, side: str):
        idx = center.copy()
        idx[itk_axis] = 0 if side == "low" else int(size[itk_axis] - 1)
        return np.asarray(img.TransformIndexToPhysicalPoint(tuple(idx)), dtype=float)

    zlo, zhi = physical_for_axis(2, "low"), physical_for_axis(2, "high")
    ylo, yhi = physical_for_axis(1, "low"), physical_for_axis(1, "high")
    low_z_endpoint = "inferior" if zlo[2] < zhi[2] else "superior"
    high_z_endpoint = "superior" if zhi[2] > zlo[2] else "inferior"
    high_y_endpoint = "posterior" if yhi[1] > ylo[1] else "anterior"
    return {
        "patient_id": int(pid),
        "raw_orientation_code": orientation,
        "raw_direction": json.dumps(direction),
        "raw_low_numpy_z_endpoint": low_z_endpoint,
        "raw_high_numpy_z_endpoint": high_z_endpoint,
        "raw_high_numpy_y_endpoint": high_y_endpoint,
        "legacy_cut_upper_half_action_raw": f"{low_z_endpoint}_portion_truncation",
        "legacy_cut_lower_half_action_raw": f"{high_z_endpoint}_portion_truncation",
        "legacy_cut_posterior_half_action_raw": f"{high_y_endpoint}_portion_truncation",
        "legacy_upper_lower_names_reversed": bool(low_z_endpoint != "superior" or high_z_endpoint != "inferior"),
        "new_calibration_orientation": "LPS",
        "new_cut_inferior_portion_semantics": "remove low NumPy-z foreground extent",
        "new_cut_superior_portion_semantics": "remove high NumPy-z foreground extent",
        "new_cut_posterior_portion_semantics": "remove high NumPy-y foreground extent",
    }


def load_domains(path: Path | None, roi: str, site: str) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame(columns=["patient_id", "domain"])
    try:
        from analysis.analyze_nsrqa_domain_performance import load_domain_assignments
        assignments, conflicts, _ = load_domain_assignments(str(path), site)
        assignments = assignments[(assignments["roi"] == roi) & (assignments["domain"] != "CONFLICT")].copy()
        return assignments[["patient_id", "domain"]]
    except Exception as exc:
        print(f"WARNING: domain table could not be loaded ({exc}); continuing without domains")
        return pd.DataFrame(columns=["patient_id", "domain"])


def compatibility(row: Mapping[str, float], cfg: Mapping) -> str:
    d = float(row["match_distance"])
    e = float(row["weighted_error_dice"])
    if d <= float(cfg["strong_max_distance"]) and e >= float(cfg["strong_min_error_dice"]):
        return "strong"
    if d <= float(cfg["plausible_max_distance"]) and e >= float(cfg["plausible_min_error_dice"]):
        return "plausible"
    return "weak"


def quantile_summary(vals: np.ndarray) -> Dict[str, float]:
    return {f"p{q:02d}": float(np.percentile(vals, q)) for q in (10, 25, 50, 75, 90, 95)}


def triangular_sample(lo: float, mode: float, hi: float, rng: random.Random, integer: bool) -> float:
    if not np.isfinite(lo + mode + hi):
        return mode
    if hi < lo:
        lo, hi = hi, lo
    mode = min(max(mode, lo), hi)
    x = rng.triangular(lo, hi, mode) if hi > lo else lo
    return float(int(round(x))) if integer else float(x)


def image_from_array(mask: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(np.asarray(mask, np.uint8))
    out.CopyInformation(ref)
    return out


def save_case_inspection(case: CaseData, family_rows: pd.DataFrame, cfg: Mapping, root: Path, seed: int):
    pdir = root / f"patient_{case.patient_id}"
    pdir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(case.ct, str(pdir / "ct.nii.gz"), True)
    sitk.WriteImage(image_from_array(case.approved, case.reference), str(pdir / "approved.nii.gz"), True)
    sitk.WriteImage(image_from_array(case.autoseg, case.reference), str(pdir / "autoseg.nii.gz"), True)
    sitk.WriteImage(image_from_array(case.approved & ~case.autoseg, case.reference), str(pdir / "real_missing.nii.gz"), True)
    sitk.WriteImage(image_from_array(case.autoseg & ~case.approved, case.reference), str(pdir / "real_excess.nii.gz"), True)

    grids = {k: list(map(float, v["levels"])) for k, v in cfg["families"].items()}
    for _, row in family_rows.iterrows():
        family = str(row["family"])
        fdir = pdir / family
        fdir.mkdir(exist_ok=True)
        best_level = float(row["level"])
        best_repeat = int(row["repeat"])
        levels = grids[family]
        idx = min(range(len(levels)), key=lambda i: abs(levels[i] - best_level))
        variants = [("best", levels[idx])]
        if bool(cfg.get("inspection", {}).get("neighbor_levels", True)):
            if idx > 0:
                variants.append(("lower", levels[idx - 1]))
            if idx + 1 < len(levels):
                variants.append(("upper", levels[idx + 1]))
        for variant, level in variants:
            s = stable_seed(case.patient_id, family, level, best_repeat, base=seed)
            syn = candidate_from_family(case.approved, family, level, s)
            sitk.WriteImage(image_from_array(syn, case.reference), str(fdir / f"{variant}.nii.gz"), True)
            sitk.WriteImage(image_from_array(case.approved & ~syn, case.reference), str(fdir / f"{variant}_missing.nii.gz"), True)
            sitk.WriteImage(image_from_array(syn & ~case.approved, case.reference), str(fdir / f"{variant}_excess.nii.gz"), True)
            detail = {
                "patient_id": case.patient_id,
                "family": family,
                "variant": variant,
                "level": level,
                "real": pair_metrics(case.approved, case.autoseg, case.spacing_zyx),
                "synthetic": pair_metrics(case.approved, syn, case.spacing_zyx),
                "error_overlap": error_overlap(case.approved, case.autoseg, syn),
            }
            (fdir / f"{variant}_metrics.json").write_text(json.dumps(detail, indent=2, allow_nan=True))


def plot_family_levels(matches: pd.DataFrame, outdir: Path):
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    for family, g in matches.groupby("family"):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        vals = g.loc[g["compatibility"].isin(["strong", "plausible"]), "level"].dropna().to_numpy(float)
        if len(vals):
            ax.hist(vals, bins=min(12, max(4, len(np.unique(vals)))))
        ax.set_title(f"{family}: best-fit severity")
        ax.set_xlabel("Native corruption level")
        ax.set_ylabel("Training cases")
        fig.tight_layout()
        fig.savefig(figdir / f"severity__{family}.png", dpi=160)
        plt.close(fig)


def plot_summary(real: pd.DataFrame, matches: pd.DataFrame, bank: pd.DataFrame, size_df: pd.DataFrame, outdir: Path):
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    overall = matches[matches["is_overall_best"]].copy()
    counts = overall["overall_best_family"].value_counts().sort_values(ascending=False)
    if len(counts):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("Training cases")
        ax.set_title("Best-matching legacy corruption family")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(figdir / "best_family_counts.png", dpi=160)
        plt.close(fig)

    for family, g in overall.groupby("overall_best_family"):
        if len(g) < 3:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(g["approved_volume_mm3"] / 1000.0, g["overall_best_level"])
        ax.set_xlabel("Approved Lung_L volume (cm³)")
        ax.set_ylabel("Best native severity level")
        ax.set_title(f"{family}: severity vs Lung_L volume")
        fig.tight_layout()
        fig.savefig(figdir / f"size_dependence__{family}.png", dpi=160)
        plt.close(fig)

    if len(bank):
        for feature in ("one_minus_dice", "missing_fraction_gt", "excess_fraction_gt", "centroid_shift_mm"):
            rv = real[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            sv = bank[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            if not len(rv) or not len(sv):
                continue
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.boxplot([rv, sv], labels=["Real", "Synthetic bank"], showfliers=False)
            ax.set_ylabel(feature)
            ax.set_title(f"Real vs calibrated synthetic: {feature}")
            fig.tight_layout()
            fig.savefig(figdir / f"real_vs_synthetic__{feature}.png", dpi=160)
            plt.close(fig)


def write_report(outdir: Path, cfg: Mapping, real: pd.DataFrame, matches: pd.DataFrame,
                 rec: pd.DataFrame, size_df: pd.DataFrame, audit: pd.DataFrame,
                 nn_df: pd.DataFrame, clf_df: pd.DataFrame):
    lines = [
        "# Lung_L synthetic-corruption severity calibration",
        "",
        "This is a **training-set-only** calibration. Month3/testing1 is not used to select corruption ranges.",
        "The historical corruption vocabulary is kept fixed; only severity/extent is calibrated.",
        "",
        "## Cut-direction audit",
        "",
    ]
    if len(audit):
        counts = audit.groupby([
            "legacy_cut_upper_half_action_raw", "legacy_cut_lower_half_action_raw",
            "legacy_cut_posterior_half_action_raw"
        ]).size().reset_index(name="n")
        lines += ["| legacy upper action | legacy lower action | legacy posterior action | n |", "|---|---|---|---:|"]
        for _, r in counts.iterrows():
            lines.append(f"| {r.iloc[0]} | {r.iloc[1]} | {r.iloc[2]} | {int(r['n'])} |")
        reversed_n = int(audit["legacy_upper_lower_names_reversed"].sum())
        lines += [
            "",
            f"`cut_upper_half`/`cut_lower_half` are anatomically misleading in **{reversed_n}/{len(audit)}** audited source cases. ",
            "The new calibration therefore uses explicit LPS-canonical names: `cut_inferior_portion`, `cut_superior_portion`, and `cut_posterior_portion`.",
            "Historical names are retained only so old results remain reproducible.",
        ]

    lines += ["", "## Real training errors", "", f"Cases analyzed: **{len(real)}**."]
    if "domain" in real.columns:
        dc = real["domain"].fillna("Unassigned").value_counts()
        lines += ["", "| Domain | n |", "|---|---:|"] + [f"| {k} | {int(v)} |" for k, v in dc.items()]

    lines += ["", "## Recommended severity ranges", ""]
    if len(rec):
        cols = ["family", "n_credible", "level_p10", "level_p50", "level_p90", "realized_changed_fraction_p50", "recommendation_status"]
        lines += ["| Family | n credible | P10 level | P50 | P90 | median changed/GT | status |", "|---|---:|---:|---:|---:|---:|---|"]
        for _, r in rec[cols].iterrows():
            lines.append(
                f"| {r['family']} | {int(r['n_credible'])} | {r['level_p10']:.4g} | {r['level_p50']:.4g} | "
                f"{r['level_p90']:.4g} | {r['realized_changed_fraction_p50']:.3f} | {r['recommendation_status']} |"
            )
    else:
        lines.append("No family had enough credible matches to recommend a range.")

    lines += ["", "## Size dependence", "", "| Family | size variable | n | Spearman rho | p | suggest conditioning |", "|---|---|---:|---:|---:|---|"]
    for _, r in size_df.iterrows():
        lines.append(f"| {r['family']} | {r['size_variable']} | {int(r['n'])} | {r['spearman_rho']:.3f} | {r['p_value']:.3g} | {r['suggest_size_conditioning']} |")

    lines += ["", "## Synthetic-bank coverage", ""]
    if len(nn_df):
        lines.append(
            f"Same-patient nearest-neighbor standardized distance: median **{nn_df['nn_distance'].median():.3f}**, "
            f"P90 **{np.percentile(nn_df['nn_distance'],90):.3f}**."
        )
    if len(clf_df):
        row = clf_df.iloc[0]
        lines.append(f"Group-CV real-vs-synthetic logistic AUC: **{row['auc_mean']:.3f} ± {row['auc_std']:.3f}**.")

    lines += [
        "", "## Inspection", "",
        "Open `slicer/LungCorruptionCalibrationViewer.py` in 3D Slicer and select the analysis `inspection/` directory.",
        "Each patient stores CT/approved/autoseg once; each corruption family stores the best numerical match plus adjacent severity levels.",
        "",
        "## Interpretation guardrails", "",
        "- A low numerical distance does not prove that the synthetic error is clinically realistic; use the Slicer inspection before freezing the training distribution.",
        "- Recorded H-domain cases are excluded from primary severity recommendations by default because the approved contour may itself be questionable.",
        "- Domain labels stratify the report; they do not directly set corruption probabilities.",
        "- The generated YAML is a recommendation artifact, not automatically a training configuration.",
    ]
    (outdir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    project_root = Path.cwd()
    split_yaml = Path(cfg["split_yaml"])
    if not split_yaml.is_absolute():
        split_yaml = project_root / split_yaml
    raw_root = Path(cfg["raw_data_root"])
    domain_path = Path(cfg["domain_csv"]) if cfg.get("domain_csv") else None
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    split, cohorts = build_temporal_split(split_yaml)
    calibration_split = str(cfg.get("calibration_split", "training1"))
    if calibration_split == "testing1":
        raise ValueError("Refusing to calibrate corruption severity on held-out testing1/Month3")
    patient_ids = split[calibration_split]
    roi = str(cfg.get("roi", "Lung_L"))
    spacing_mm = float(cfg["geometry"].get("isotropic_spacing_mm", 1.5))
    margin_mm = float(cfg["geometry"].get("crop_margin_mm", 60.0))
    seed = int(cfg.get("seed", 88))

    domains = load_domains(domain_path, roi, str(cfg.get("site", "thorax")))
    domain_map = {int(r.patient_id): r.domain for r in domains.itertuples() if str(r.patient_id).isdigit()}

    cases: Dict[int, CaseData] = {}
    real_rows: List[Dict] = []
    audit_rows: List[Dict] = []
    skipped: List[Dict] = []
    print(f"Loading {len(patient_ids)} {roi} {calibration_split} cases ...")
    for pid in patient_ids:
        try:
            ct_path, app_path, _ = resolve_case_paths(raw_root, pid, roi)
            raw_app = sitk.ReadImage(str(app_path))
            audit_rows.append(raw_orientation_audit(raw_app, pid))
            case = load_case(raw_root, pid, roi, cohorts.get(pid, ""), spacing_mm, margin_mm)
            cases[pid] = case
            cheap = cheap_features(case.approved, case.autoseg, case.spacing_zyx, components=True)
            full = pair_metrics(case.approved, case.autoseg, case.spacing_zyx)
            row = {"patient_id": pid, "cohort": case.cohort, "domain": domain_map.get(pid, "Unassigned"), **cheap}
            row.update({f"detail_{k}": v for k, v in full.items()})
            real_rows.append(row)
        except Exception as exc:
            skipped.append({"patient_id": pid, "reason": repr(exc)})
            print(f"WARNING patient {pid}: {exc}")

    real_df = pd.DataFrame(real_rows).sort_values("patient_id")
    audit_df = pd.DataFrame(audit_rows).sort_values("patient_id")
    real_df.to_csv(outdir / "real_error_features.csv", index=False)
    audit_df.to_csv(outdir / "cut_direction_orientation_audit.csv", index=False)
    pd.DataFrame(skipped).to_csv(outdir / "skipped_cases.csv", index=False)
    scaler = fit_scaler(real_rows)
    (outdir / "feature_scaler.json").write_text(json.dumps({k: {"median": v[0], "scale": v[1]} for k, v in scaler.items()}, indent=2))

    sweep_rows: List[Dict] = []
    match_cfg = cfg["matching"]
    spatial_weight = float(match_cfg.get("spatial_error_weight", 0.75))
    for n_case, (pid, case) in enumerate(cases.items(), start=1):
        real = next(r for r in real_rows if int(r["patient_id"]) == pid)
        print(f"[{n_case}/{len(cases)}] severity sweep patient {pid}")
        for family, spec in cfg["families"].items():
            levels = [float(x) for x in spec["levels"]]
            repeats = int(spec.get("repeats", 1))
            for level in levels:
                for rep in range(repeats):
                    s = stable_seed(pid, family, level, rep, base=seed)
                    syn = candidate_from_family(case.approved, family, level, s)
                    need_components = family in {"big_bite", "big_blob"}
                    sf = cheap_features(case.approved, syn, case.spacing_zyx, components=need_components)
                    ov = error_overlap(case.approved, case.autoseg, syn)
                    fd = feature_distance(real, sf, FAMILY_FEATURES[family], scaler)
                    total = fd + spatial_weight * (1.0 - ov["weighted_error_dice"])
                    sweep_rows.append({
                        "patient_id": pid, "cohort": case.cohort, "domain": domain_map.get(pid, "Unassigned"),
                        "family": family, "level": level, "repeat": rep, "seed": s,
                        "feature_distance": fd, "match_distance": total, **ov, **sf,
                    })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(outdir / "severity_sweep.csv", index=False)
    idx = sweep_df.groupby(["patient_id", "family"])["match_distance"].idxmin()
    family_best = sweep_df.loc[idx].sort_values(["patient_id", "family"]).copy()
    family_best["compatibility"] = [compatibility(r, match_cfg) for r in family_best.to_dict("records")]
    overall_idx = family_best.groupby("patient_id")["match_distance"].idxmin()
    overall = family_best.loc[overall_idx, ["patient_id", "family", "level", "repeat", "match_distance", "compatibility"]].copy()
    overall = overall.rename(columns={
        "family": "overall_best_family", "level": "overall_best_level", "repeat": "overall_best_repeat",
        "match_distance": "overall_best_distance", "compatibility": "overall_best_compatibility",
    })
    family_best = family_best.merge(overall, on="patient_id", how="left")
    family_best["is_overall_best"] = family_best["family"] == family_best["overall_best_family"]
    family_best.to_csv(outdir / "case_family_matches.csv", index=False)

    excluded_domains = set(map(str, match_cfg.get("exclude_recorded_domains_from_recommendation", ["H"])))
    eligible = family_best[
        family_best["is_overall_best"]
        & family_best["compatibility"].isin(["strong", "plausible"])
        & ~family_best["domain"].astype(str).isin(excluded_domains)
    ].copy()
    min_n = int(match_cfg.get("min_recommended_cases", 5))
    rec_rows = []
    recommendation_yaml = {"roi": roi, "source_split": calibration_split, "families": {}}
    for family in cfg["families"]:
        g = eligible[eligible["family"] == family]
        if len(g) == 0:
            rec_rows.append({
                "family": family, "n_credible": 0,
                "level_p10": np.nan, "level_p25": np.nan, "level_p50": np.nan,
                "level_p75": np.nan, "level_p90": np.nan, "level_p95": np.nan,
                "realized_changed_fraction_p10": np.nan,
                "realized_changed_fraction_p50": np.nan,
                "realized_changed_fraction_p90": np.nan,
                "recommendation_status": "no_credible_overall_matches",
            })
            continue
        q = quantile_summary(g["level"].to_numpy(float))
        cq = quantile_summary(g["changed_fraction_gt"].to_numpy(float))
        status = "recommended" if len(g) >= min_n else "insufficient_n_review_manually"
        rec_rows.append({
            "family": family, "n_credible": len(g),
            **{f"level_{k}": v for k, v in q.items()},
            "realized_changed_fraction_p10": cq["p10"],
            "realized_changed_fraction_p50": cq["p50"],
            "realized_changed_fraction_p90": cq["p90"],
            "recommendation_status": status,
        })
        recommendation_yaml["families"][family] = {
            "n_credible_overall_matches": int(len(g)),
            "native_level_triangular": [float(q["p10"]), float(q["p50"]), float(q["p90"])],
            "realized_changed_fraction_p10_p50_p90": [float(cq["p10"]), float(cq["p50"]), float(cq["p90"])],
            "status": status,
        }
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(outdir / "family_severity_recommendations.csv", index=False)
    (outdir / "recommended_corruption_ranges.yaml").write_text(yaml.safe_dump(recommendation_yaml, sort_keys=False))

    size_rows = []
    size_vars = ["approved_volume_mm3", "bbox_extent_si_gt_mm", "bbox_extent_ap_gt_mm", "bbox_extent_lr_gt_mm"]
    for family, g in eligible.groupby("family"):
        for var in size_vars:
            x, y = g[var].to_numpy(float), g["level"].to_numpy(float)
            good = np.isfinite(x) & np.isfinite(y)
            if good.sum() >= 3:
                rho, pval = stats.spearmanr(x[good], y[good])
            else:
                rho, pval = np.nan, np.nan
            suggest = bool(good.sum() >= 8 and np.isfinite(rho) and abs(rho) >= 0.5 and pval < 0.05)
            size_rows.append({
                "family": family, "size_variable": var, "n": int(good.sum()),
                "spearman_rho": rho, "p_value": pval, "suggest_size_conditioning": suggest,
            })
    size_df = pd.DataFrame(size_rows)
    size_df.to_csv(outdir / "size_conditioning_analysis.csv", index=False)

    # Build a validation bank from the empirical overall-best family mixture and
    # the recommended P10/P50/P90 levels. This is diagnostic only; it does not
    # write a training config or use Month3.
    bank_rows = []
    if len(eligible):
        weights = eligible["family"].value_counts(normalize=True)
        families = list(weights.index)
        probs = list(weights.values)
        rec_map = {r["family"]: r for r in rec_rows if r.get("n_credible", 0) > 0}
        n_bank = int(cfg.get("synthetic_bank", {}).get("samples_per_patient", 32))
        for pid, case in cases.items():
            rng = random.Random(stable_seed("bank", pid, base=seed))
            for j in range(n_bank):
                family = rng.choices(families, weights=probs, k=1)[0]
                rr = rec_map[family]
                level = triangular_sample(
                    float(rr["level_p10"]), float(rr["level_p50"]), float(rr["level_p90"]), rng,
                    family in INTEGER_LEVEL_FAMILIES,
                )
                s = stable_seed("bank", pid, j, family, level, base=seed)
                syn = candidate_from_family(case.approved, family, level, s)
                sf = cheap_features(case.approved, syn, case.spacing_zyx, components=family in {"big_bite", "big_blob"})
                bank_rows.append({"patient_id": pid, "sample": j, "family": family, "level": level, **sf})
    bank_df = pd.DataFrame(bank_rows)
    bank_df.to_csv(outdir / "synthetic_validation_bank_features.csv", index=False)

    nn_rows = []
    if len(bank_df):
        for _, rr in real_df.iterrows():
            pid = int(rr["patient_id"])
            bg = bank_df[bank_df["patient_id"] == pid]
            ds = []
            for _, sr in bg.iterrows():
                ds.append(feature_distance(rr, sr, BANK_FEATURES, scaler))
            nn_rows.append({"patient_id": pid, "nn_distance": float(np.min(ds)) if ds else np.nan})
    nn_df = pd.DataFrame(nn_rows)
    nn_df.to_csv(outdir / "same_patient_nn_coverage.csv", index=False)

    clf_rows = []
    if len(bank_df):
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import GroupKFold
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            real_clf = real_df[["patient_id", *BANK_FEATURES]].copy()
            real_clf["label"] = 1
            syn_clf = bank_df[["patient_id", *BANK_FEATURES]].copy()
            syn_clf["label"] = 0
            allc = pd.concat([real_clf, syn_clf], ignore_index=True)
            X = allc[list(BANK_FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
            y = allc["label"].to_numpy(int)
            groups = allc["patient_id"].to_numpy(int)
            nfold = min(int(cfg.get("synthetic_bank", {}).get("classifier_folds", 5)), len(np.unique(groups)))
            aucs = []
            for train_i, test_i in GroupKFold(n_splits=nfold).split(X, y, groups):
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
                model.fit(X[train_i], y[train_i])
                aucs.append(roc_auc_score(y[test_i], model.predict_proba(X[test_i])[:, 1]))
            clf_rows.append({"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "folds": nfold})
        except Exception as exc:
            clf_rows.append({"auc_mean": np.nan, "auc_std": np.nan, "folds": 0, "error": repr(exc)})
    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(outdir / "real_vs_synthetic_classifier.csv", index=False)

    if bool(cfg.get("inspection", {}).get("export", True)):
        inspection = outdir / "inspection"
        inspection.mkdir(exist_ok=True)
        maxp = int(cfg.get("inspection", {}).get("max_patients", 0))
        export_ids = sorted(cases)[:maxp] if maxp > 0 else sorted(cases)
        for pid in export_ids:
            save_case_inspection(cases[pid], family_best[family_best["patient_id"] == pid], cfg, inspection, seed)

    plot_family_levels(family_best, outdir)
    plot_summary(real_df, family_best, bank_df, size_df, outdir)
    write_report(outdir, cfg, real_df, family_best, rec_df, size_df, audit_df, nn_df, clf_df)
    print(f"Wrote calibration results to {outdir}")


if __name__ == "__main__":
    main()
