#!/usr/bin/env python3
"""Directly characterize real lung (Lung_L or Lung_R, config ``roi``) autosegmentation error fields.

This analysis intentionally avoids fitting each patient to a synthetic
corruption family.  For every development patient it measures the observed
approved-vs-autoseg error directly:

* missing and excess connected components,
* signed displacement of the approved surface to the autoseg surface,
* smooth/local displacement components at configurable physical scales,
* global centroid/bounding-box/endpoint errors,
* patch size, depth, footprint and normalized anatomical location.

Patients are independent and are processed in a ProcessPoolExecutor.  The
analysis then ranks complementary cases for manual inspection and exports only
that shortlist (by default) as Slicer-compatible NIfTI files.

Month3/testing1 is never used when the default temporal split is selected.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from analysis.spinalcord_diagnostic_utils import pair_metrics, surface


STRUCT6 = ndimage.generate_binary_structure(3, 1)
EPS = 1e-8


@dataclass
class LoadedCase:
    patient_id: int
    cohort: str
    domain: str
    ct: sitk.Image
    reference: sitk.Image
    approved: np.ndarray
    autoseg: np.ndarray
    spacing_zyx: Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=None,
                   help="Override config parallel.workers (0/negative = auto from Slurm CPUs).")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


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
    """Resolve CT/approved/autoseg paths without substring patient-ID collisions.

    ``*patient*53*`` also matches ``patient153``.  Resolve each file
    independently so an unrelated CT naming fallback never forces otherwise
    exact label paths through a broad glob, and require the fallback basename
    to contain the requested patient ID as a complete numeric token.
    """
    pid = int(pid)
    patient_token = re.compile(rf"patient[_-]?{pid}(?!\d)", re.IGNORECASE)

    def one(directory: Path, exact_name: str, patterns: Sequence[str]) -> Path:
        exact = directory / exact_name
        if exact.is_file():
            return exact

        hits: List[Path] = []
        for pat in patterns:
            hits.extend(directory.glob(pat))
        hits = sorted({p for p in hits if patient_token.search(p.name)})
        if len(hits) != 1:
            raise FileNotFoundError(
                f"Expected one file for patient {pid} under {directory}; "
                f"exact={exact}; patterns={patterns}; token-filtered hits={hits}"
            )
        return hits[0]

    return (
        one(
            root / "images",
            f"patient{pid}_image.nii.gz",
            [f"*patient*{pid}*image*.nii.gz", f"*patient_{pid}*.nii.gz"],
        ),
        one(
            root / "labels-approved",
            f"patient{pid}_{roi}_approved.nii.gz",
            [f"*patient*{pid}*{roi}*approved*.nii.gz"],
        ),
        one(
            root / "labels-autoseg",
            f"patient{pid}_{roi}_autoseg.nii.gz",
            [f"*patient*{pid}*{roi}*autoseg*.nii.gz"],
        ),
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


def load_case(raw_root: Path, pid: int, roi: str, cohort: str, domain: str,
              spacing_mm: float, margin_mm: float) -> LoadedCase:
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

    g = sitk.GetArrayFromImage(approved) > 0
    a = sitk.GetArrayFromImage(autoseg) > 0
    if not g.any() or not a.any():
        raise ValueError(f"Empty {roi} after preprocessing for patient {pid}")
    spacing_zyx = tuple(float(v) for v in approved.GetSpacing()[::-1])
    return LoadedCase(pid, cohort, domain, ct, approved, g, a, spacing_zyx)


def load_domains(path: Path | None, roi: str, site: str) -> Dict[int, str]:
    if path is None or not path.is_file():
        return {}
    try:
        from analysis.analyze_nsrqa_domain_performance import load_domain_assignments
        assignments, _conflicts, _ = load_domain_assignments(str(path), site)
        x = assignments[(assignments["roi"] == roi) & (assignments["domain"] != "CONFLICT")].copy()
        return {int(r.patient_id): str(r.domain) for r in x.itertuples()}
    except Exception as exc:
        print(f"WARNING: domain table could not be loaded ({exc}); continuing without domains")
        return {}


def bbox(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    if not len(pts):
        return np.full(3, np.nan), np.full(3, np.nan)
    return pts.min(axis=0).astype(float), pts.max(axis=0).astype(float)


def centroid(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    return pts.mean(axis=0) if len(pts) else np.full(3, np.nan)


def signed_surface_displacement_mm(approved: np.ndarray, autoseg: np.ndarray,
                                   spacing_zyx: Sequence[float]):
    """Nearest-surface displacement on the approved surface.

    Positive means the approved surface point is inside the autoseg and the
    autoseg boundary is therefore outward from that point. Negative means the
    approved surface point is outside the autoseg (inward/missing error).

    This nearest-surface quantity is an approximation to normal displacement,
    but unlike a voxel signed-distance transform it is exactly zero when the
    two binary surfaces are identical.
    """
    sg = surface(approved)
    sa = surface(autoseg)
    gp = np.argwhere(sg)
    ap = np.argwhere(sa)
    if not len(gp) or not len(ap):
        return sg, np.empty((0, 3)), np.empty(0)
    spacing = np.asarray(spacing_zyx, dtype=float)
    gp_mm = gp.astype(float) * spacing
    ap_mm = ap.astype(float) * spacing
    tree = cKDTree(ap_mm)
    dist, _ = tree.query(gp_mm, k=1, workers=1)
    inside = autoseg[tuple(gp.T)]
    disp = np.where(dist <= 1e-7, 0.0, np.where(inside, dist, -dist)).astype(np.float32)
    return sg, gp_mm, disp


def deterministic_surface_subsample(points_mm: np.ndarray, values: np.ndarray,
                                    grid_mm: float, max_points: int, seed: int):
    if len(points_mm) == 0:
        return points_mm, values, np.empty(0, dtype=int)
    grid_mm = max(float(grid_mm), 1e-3)
    bins = np.floor(points_mm / grid_mm).astype(np.int64)
    _, idx = np.unique(bins, axis=0, return_index=True)
    idx = np.sort(idx)
    if max_points > 0 and len(idx) > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return points_mm[idx], values[idx], idx


def surface_multiscale_summary(points_mm: np.ndarray, displacement_mm: np.ndarray,
                               cfg: Mapping, seed: int):
    sigmas = [float(x) for x in cfg.get("smooth_sigmas_mm", [10.0, 20.0, 40.0])]
    primary_sigma = float(cfg.get("primary_smooth_sigma_mm", 20.0))
    grid_mm = float(cfg.get("surface_sample_grid_mm", 3.0))
    max_points = int(cfg.get("max_surface_samples", 30000))
    k = int(cfg.get("smoothing_neighbors", 96))

    sp, sv, original_idx = deterministic_surface_subsample(
        points_mm, displacement_mm, grid_mm, max_points, seed
    )
    if len(sp) == 0:
        return [], {}, (sp, sv, sv, original_idx)
    if len(sp) == 1:
        rows = []
        for sigma in sigmas:
            rows.append({
                "smooth_sigma_mm": sigma,
                "sampled_surface_points": 1,
                "smooth_rms_mm": abs(float(sv[0])),
                "local_rms_mm": 0.0,
                "smooth_energy_fraction": 1.0,
            })
        return rows, {primary_sigma: (sv.copy(), np.zeros_like(sv))}, (sp, sv, sv, original_idx)

    tree = cKDTree(sp)
    kk = max(2, min(k, len(sp)))
    dists, inds = tree.query(sp, k=kk, workers=1)
    if dists.ndim == 1:
        dists = dists[:, None]
        inds = inds[:, None]

    rows = []
    fields = {}
    for sigma in sigmas:
        sigma = max(float(sigma), 1e-3)
        w = np.exp(-0.5 * np.square(dists / sigma))
        wsum = np.maximum(w.sum(axis=1), EPS)
        smooth = (w * sv[inds]).sum(axis=1) / wsum
        local = sv - smooth
        smooth_rms = float(np.sqrt(np.mean(np.square(smooth))))
        local_rms = float(np.sqrt(np.mean(np.square(local))))
        denom = smooth_rms * smooth_rms + local_rms * local_rms
        rows.append({
            "smooth_sigma_mm": sigma,
            "sampled_surface_points": int(len(sp)),
            "smooth_rms_mm": smooth_rms,
            "local_rms_mm": local_rms,
            "smooth_energy_fraction": float(smooth_rms * smooth_rms / denom) if denom > EPS else 0.0,
            "smooth_abs_p95_mm": float(np.percentile(np.abs(smooth), 95)),
            "local_abs_p95_mm": float(np.percentile(np.abs(local), 95)),
        })
        fields[sigma] = (smooth.astype(np.float32), local.astype(np.float32))

    if primary_sigma not in fields:
        nearest = min(fields, key=lambda x: abs(x - primary_sigma))
        fields[primary_sigma] = fields[nearest]
    return rows, fields, (sp, sv, fields[primary_sigma][0], original_idx)


def displacement_correlation_length(points_mm: np.ndarray, values: np.ndarray,
                                    cfg: Mapping, seed: int) -> Tuple[float, pd.DataFrame]:
    max_points = int(cfg.get("correlation_max_points", 2500))
    n_pairs = int(cfg.get("correlation_pairs", 40000))
    bin_mm = float(cfg.get("correlation_bin_mm", 5.0))
    max_dist = float(cfg.get("correlation_max_distance_mm", 100.0))
    if len(points_mm) < 10 or np.std(values) < 1e-5:
        return np.nan, pd.DataFrame(columns=["distance_mm", "correlation", "n_pairs"])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(points_mm))
    if len(idx) > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    p = points_mm[idx]
    v = values[idx].astype(float)
    v = (v - np.mean(v)) / max(np.std(v), 1e-8)
    i = rng.integers(0, len(p), size=n_pairs)
    j = rng.integers(0, len(p), size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    dist = np.linalg.norm(p[i] - p[j], axis=1)
    prod = v[i] * v[j]
    edges = np.arange(0.0, max_dist + bin_mm, bin_mm)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if int(m.sum()) < 30:
            continue
        rows.append({
            "distance_mm": float((lo + hi) / 2.0),
            "correlation": float(np.mean(prod[m])),
            "n_pairs": int(m.sum()),
        })
    df = pd.DataFrame(rows)
    corr_len = np.nan
    if len(df):
        hit = df[df["correlation"] <= math.exp(-1.0)]
        if len(hit):
            corr_len = float(hit.iloc[0]["distance_mm"])
    return corr_len, df


def component_rows(mask: np.ndarray, kind: str, approved: np.ndarray,
                   approved_surface: np.ndarray, distance_to_approved_surface: np.ndarray,
                   spacing_zyx: Sequence[float], min_volume_mm3: float,
                   patient_id: int, cohort: str, domain: str, lateral_is_low_x: bool = False):
    spacing = np.asarray(spacing_zyx, dtype=float)
    voxel_volume = float(np.prod(spacing))
    gt_vox = int(approved.sum())
    gt_surface_vox = max(int(approved_surface.sum()), 1)
    glo, ghi = bbox(approved)
    labels, n = ndimage.label(mask, structure=STRUCT6)
    rows = []
    raw_count = int(n)
    for lab in range(1, n + 1):
        comp = labels == lab
        vox = int(comp.sum())
        vol = vox * voxel_volume
        if vol < min_volume_mm3:
            continue
        pts = np.argwhere(comp)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        cen = pts.mean(axis=0)
        span = np.maximum(ghi - glo, 1.0)
        norm = np.clip((cen - glo) / span, 0.0, 1.0)
        depth = distance_to_approved_surface[comp]
        contact = ndimage.binary_dilation(comp, structure=STRUCT6) & approved_surface
        rows.append({
            "patient_id": int(patient_id),
            "cohort": cohort,
            "domain": domain,
            "component_type": kind,
            "component_label": int(lab),
            "volume_mm3": float(vol),
            "volume_cc": float(vol / 1000.0),
            "fraction_approved_volume": float(vox / max(gt_vox, 1)),
            "extent_si_mm": float((hi[0] - lo[0] + 1) * spacing[0]),
            "extent_ap_mm": float((hi[1] - lo[1] + 1) * spacing[1]),
            "extent_lr_mm": float((hi[2] - lo[2] + 1) * spacing[2]),
            "max_depth_from_approved_surface_mm": float(np.max(depth)) if len(depth) else 0.0,
            "median_depth_from_approved_surface_mm": float(np.median(depth)) if len(depth) else 0.0,
            "surface_footprint_fraction": float(contact.sum() / gt_surface_vox),
            # Canonical LPS: low z=inferior, low y=anterior, low x=patient right.
            # For Lung_L high x is lateral; for Lung_R low x is lateral, so flip.
            "centroid_norm_si_inferior_to_superior": float(norm[0]),
            "centroid_norm_ap_anterior_to_posterior": float(norm[1]),
            "centroid_norm_lr_medial_to_lateral": float(1.0 - norm[2] if lateral_is_low_x else norm[2]),
            "touches_inferior_extent": bool(lo[0] <= glo[0] + 1),
            "touches_superior_extent": bool(hi[0] >= ghi[0] - 1),
            "touches_anterior_extent": bool(lo[1] <= glo[1] + 1),
            "touches_posterior_extent": bool(hi[1] >= ghi[1] - 1),
        })
    rows.sort(key=lambda r: r["volume_mm3"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["component_rank_by_volume"] = rank
    return rows, raw_count


def image_from_array(arr: np.ndarray, ref: sitk.Image, pixel_type=sitk.sitkUInt8) -> sitk.Image:
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return sitk.Cast(out, pixel_type)


def case_characterization(pid: int, cohort: str, domain: str, cfg: Mapping,
                          need_export_arrays: bool = False):
    # Avoid nested ITK/OpenMP thread pools inside the patient-level process pool.
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(cfg["parallel"].get("sitk_threads_per_worker", 1)))
    except Exception:
        pass

    geom = cfg["geometry"]
    case = load_case(
        Path(cfg["raw_data_root"]), pid, str(cfg["roi"]), cohort, domain,
        float(geom["isotropic_spacing_mm"]), float(geom["crop_margin_mm"]),
    )
    g, a = case.approved, case.autoseg
    spacing = np.asarray(case.spacing_zyx, dtype=float)
    real = pair_metrics(g, a, spacing)
    missing = g & ~a
    excess = a & ~g
    gv = int(g.sum())
    voxel_volume = float(np.prod(spacing))
    real.update({
        "patient_id": int(pid), "cohort": cohort, "domain": domain,
        "missing_fraction_gt": float(missing.sum() / max(gv, 1)),
        "excess_fraction_gt": float(excess.sum() / max(gv, 1)),
        "changed_fraction_gt": float((missing | excess).sum() / max(gv, 1)),
        "approved_volume_mm3": float(gv * voxel_volume),
        "approved_volume_cc": float(gv * voxel_volume / 1000.0),
    })

    gc, ac = centroid(g), centroid(a)
    delta = (ac - gc) * spacing
    glo, ghi = bbox(g)
    alo, ahi = bbox(a)
    gext = (ghi - glo + 1) * spacing
    aext = (ahi - alo + 1) * spacing
    real.update({
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
    })

    sg, surface_points_mm, displacement = signed_surface_displacement_mm(g, a, spacing)
    noise_floor = float(cfg["error_field"].get("direction_noise_floor_mm", 0.5))
    if len(displacement):
        real.update({
            "surface_displacement_mean_mm": float(np.mean(displacement)),
            "surface_displacement_median_mm": float(np.median(displacement)),
            "surface_displacement_rms_mm": float(np.sqrt(np.mean(np.square(displacement)))),
            "surface_displacement_abs_p50_mm": float(np.percentile(np.abs(displacement), 50)),
            "surface_displacement_abs_p90_mm": float(np.percentile(np.abs(displacement), 90)),
            "surface_displacement_abs_p95_mm": float(np.percentile(np.abs(displacement), 95)),
            "surface_displacement_abs_max_mm": float(np.max(np.abs(displacement))),
            "surface_outward_fraction": float(np.mean(displacement > noise_floor)),
            "surface_inward_fraction": float(np.mean(displacement < -noise_floor)),
            "surface_near_zero_fraction": float(np.mean(np.abs(displacement) <= noise_floor)),
        })
    else:
        for k in (
            "surface_displacement_mean_mm", "surface_displacement_median_mm", "surface_displacement_rms_mm",
            "surface_displacement_abs_p50_mm", "surface_displacement_abs_p90_mm",
            "surface_displacement_abs_p95_mm", "surface_displacement_abs_max_mm",
            "surface_outward_fraction", "surface_inward_fraction", "surface_near_zero_fraction",
        ):
            real[k] = np.nan

    scale_rows, smooth_fields, sample_tuple = surface_multiscale_summary(
        surface_points_mm, displacement, cfg["error_field"], seed=int(cfg["seed"]) + int(pid)
    )
    primary_sigma = float(cfg["error_field"].get("primary_smooth_sigma_mm", 20.0))
    primary_row = min(scale_rows, key=lambda r: abs(float(r["smooth_sigma_mm"]) - primary_sigma)) if scale_rows else None
    if primary_row:
        real.update({
            "primary_smooth_sigma_mm": float(primary_row["smooth_sigma_mm"]),
            "smooth_rms_mm": float(primary_row["smooth_rms_mm"]),
            "local_rms_mm": float(primary_row["local_rms_mm"]),
            "smooth_energy_fraction": float(primary_row["smooth_energy_fraction"]),
            "smooth_abs_p95_mm": float(primary_row["smooth_abs_p95_mm"]),
            "local_abs_p95_mm": float(primary_row["local_abs_p95_mm"]),
        })
    else:
        real.update({
            "primary_smooth_sigma_mm": primary_sigma, "smooth_rms_mm": np.nan,
            "local_rms_mm": np.nan, "smooth_energy_fraction": np.nan,
            "smooth_abs_p95_mm": np.nan, "local_abs_p95_mm": np.nan,
        })

    corr_len, corr_df = displacement_correlation_length(
        sample_tuple[0], sample_tuple[1], cfg["error_field"], seed=int(cfg["seed"]) + 100000 + int(pid)
    )
    real["surface_displacement_correlation_length_mm"] = corr_len

    # One distance-to-approved-surface map supports physical depth estimates for
    # both missing and excess connected components.
    d_to_gt_surface = ndimage.distance_transform_edt(~sg, sampling=spacing)
    min_vol = float(cfg["components"].get("min_component_volume_mm3", 100.0))
    lateral_low_x = str(cfg["roi"]).upper().endswith("_R")
    missing_rows, missing_raw = component_rows(
        missing, "missing", g, sg, d_to_gt_surface, spacing, min_vol, pid, cohort, domain,
        lateral_is_low_x=lateral_low_x,
    )
    excess_rows, excess_raw = component_rows(
        excess, "excess", g, sg, d_to_gt_surface, spacing, min_vol, pid, cohort, domain,
        lateral_is_low_x=lateral_low_x,
    )
    real.update({
        "missing_component_count_raw": missing_raw,
        "excess_component_count_raw": excess_raw,
        "missing_component_count_significant": int(len(missing_rows)),
        "excess_component_count_significant": int(len(excess_rows)),
        "total_component_count_significant": int(len(missing_rows) + len(excess_rows)),
        "largest_missing_component_fraction_gt": float(missing_rows[0]["fraction_approved_volume"]) if missing_rows else 0.0,
        "largest_excess_component_fraction_gt": float(excess_rows[0]["fraction_approved_volume"]) if excess_rows else 0.0,
        "largest_missing_depth_mm": float(missing_rows[0]["max_depth_from_approved_surface_mm"]) if missing_rows else 0.0,
        "largest_excess_depth_mm": float(excess_rows[0]["max_depth_from_approved_surface_mm"]) if excess_rows else 0.0,
    })

    scale_out = []
    for row in scale_rows:
        scale_out.append({"patient_id": int(pid), "cohort": cohort, "domain": domain, **row})
    corr_out = []
    for row in corr_df.to_dict("records"):
        corr_out.append({"patient_id": int(pid), "cohort": cohort, "domain": domain, **row})

    extra = None
    if need_export_arrays:
        extra = {
            "case": case,
            "missing": missing,
            "excess": excess,
            "approved_surface": sg,
            "surface_points_mm": surface_points_mm,
            "surface_displacement": displacement,
            "sample_points_mm": sample_tuple[0],
            "sample_displacement": sample_tuple[1],
            "smooth_fields": smooth_fields,
            "missing_rows": missing_rows,
            "excess_rows": excess_rows,
        }
    return real, missing_rows + excess_rows, scale_out, corr_out, extra


def worker_task(args):
    pid, cohort, domain, cfg = args
    try:
        summary, comps, scales, corr, _ = case_characterization(pid, cohort, domain, cfg, False)
        return {"ok": True, "patient_id": pid, "summary": summary, "components": comps, "scales": scales, "corr": corr}
    except Exception as exc:
        return {"ok": False, "patient_id": pid, "error": repr(exc)}


def robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    med = float(np.nanmedian(x))
    q25, q75 = np.nanpercentile(x, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-8:
        mad = float(np.nanmedian(np.abs(x - med))) * 1.4826
        scale = mad if np.isfinite(mad) and mad >= 1e-8 else max(float(np.nanstd(x)), 1e-3)
    return (x - med) / scale


def make_inspection_recommendations(df: pd.DataFrame, cfg: Mapping):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    n_each = int(cfg["inspection"].get("patients_per_category", 2))
    work = df.copy()
    miss = work["missing_fraction_gt"].fillna(0.0)
    exc = work["excess_fraction_gt"].fillna(0.0)
    changed = miss + exc
    balance = 1.0 - np.abs(miss - exc) / np.maximum(changed, 1e-8)
    work["missing_minus_excess"] = miss - exc
    work["bidirectional_balance"] = balance
    work["bidirectional_strength"] = np.minimum(miss, exc) * balance
    work["local_dominance"] = work["local_rms_mm"].fillna(0.0) / np.maximum(work["smooth_rms_mm"].fillna(0.0) + work["local_rms_mm"].fillna(0.0), 1e-8)

    typical_features = [
        "one_minus_dice", "missing_fraction_gt", "excess_fraction_gt",
        "surface_displacement_rms_mm", "smooth_rms_mm", "local_rms_mm",
        "max_abs_endpoint_error_mm", "total_component_count_significant",
    ]
    z2 = np.zeros(len(work), dtype=float)
    used = 0
    for f in typical_features:
        if f in work:
            z = robust_z(work[f]).fillna(0.0).to_numpy(float)
            z2 += z * z
            used += 1
    work["typical_distance"] = np.sqrt(z2 / max(used, 1))

    categories = [
        ("typical", True, "typical_distance", "multivariate feature profile closest to the training-set median"),
        ("low_error", True, "one_minus_dice", "one of the cleanest real autosegmentations"),
        ("high_error", False, "one_minus_dice", "large overall approved-vs-autoseg disagreement"),
        ("missing_dominant", False, "missing_minus_excess", "missing anatomy dominates excess anatomy"),
        ("excess_dominant", True, "missing_minus_excess", "excess anatomy dominates missing anatomy"),
        ("balanced_bidirectional", False, "bidirectional_strength", "substantial missing and excess anatomy in the same case"),
        ("smooth_dominant", False, "smooth_energy_fraction", "surface error is dominated by broad smooth displacement"),
        ("local_patch_dominant", False, "local_dominance", "surface error is dominated by local/high-frequency residual"),
        ("endpoint_dominant", False, "max_abs_endpoint_error_mm", "large S/I endpoint error"),
        ("multi_patch_complex", False, "total_component_count_significant", "many significant missing/excess connected patches"),
    ]

    rows = []
    for cat, ascending, score_col, reason in categories:
        x = work[np.isfinite(pd.to_numeric(work[score_col], errors="coerce"))].sort_values(score_col, ascending=ascending)
        for rank, r in enumerate(x.head(n_each).itertuples(), start=1):
            rows.append({
                "inspection_category": cat,
                "category_rank": rank,
                "patient_id": int(r.patient_id),
                "domain": str(r.domain),
                "score": float(getattr(r, score_col)),
                "reason": reason,
                "one_minus_dice": float(r.one_minus_dice),
                "assd_mm": float(r.assd_mm),
                "hd99_mm": float(r.hd99_mm),
                "missing_fraction_gt": float(r.missing_fraction_gt),
                "excess_fraction_gt": float(r.excess_fraction_gt),
                "surface_displacement_rms_mm": float(r.surface_displacement_rms_mm),
                "smooth_rms_mm": float(r.smooth_rms_mm),
                "local_rms_mm": float(r.local_rms_mm),
                "max_abs_endpoint_error_mm": float(r.max_abs_endpoint_error_mm),
                "total_component_count_significant": int(r.total_component_count_significant),
            })

    # Add one representative case per recorded domain, including Unassigned
    # (the normal/default group in the domain table workflow).
    if bool(cfg["inspection"].get("include_domain_representatives", True)):
        for domain, g in work.groupby("domain", dropna=False):
            g = g.sort_values("typical_distance")
            if len(g):
                r = g.iloc[0]
                rows.append({
                    "inspection_category": f"domain_{domain}",
                    "category_rank": 1,
                    "patient_id": int(r.patient_id),
                    "domain": str(domain),
                    "score": float(r.typical_distance),
                    "reason": f"representative case for domain {domain}",
                    "one_minus_dice": float(r.one_minus_dice),
                    "assd_mm": float(r.assd_mm),
                    "hd99_mm": float(r.hd99_mm),
                    "missing_fraction_gt": float(r.missing_fraction_gt),
                    "excess_fraction_gt": float(r.excess_fraction_gt),
                    "surface_displacement_rms_mm": float(r.surface_displacement_rms_mm),
                    "smooth_rms_mm": float(r.smooth_rms_mm),
                    "local_rms_mm": float(r.local_rms_mm),
                    "max_abs_endpoint_error_mm": float(r.max_abs_endpoint_error_mm),
                    "total_component_count_significant": int(r.total_component_count_significant),
                })

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked, ranked
    shortlist_rows = []
    for pid, g in ranked.groupby("patient_id", sort=False):
        r = g.iloc[0]
        shortlist_rows.append({
            "patient_id": int(pid),
            "domain": str(r.domain),
            "inspection_categories": ";".join(g["inspection_category"].astype(str)),
            "inspection_reasons": "; ".join(g["reason"].astype(str)),
            "n_reasons": int(len(g)),
            "one_minus_dice": float(r.one_minus_dice),
            "assd_mm": float(r.assd_mm),
            "hd99_mm": float(r.hd99_mm),
            "missing_fraction_gt": float(r.missing_fraction_gt),
            "excess_fraction_gt": float(r.excess_fraction_gt),
            "surface_displacement_rms_mm": float(r.surface_displacement_rms_mm),
            "smooth_rms_mm": float(r.smooth_rms_mm),
            "local_rms_mm": float(r.local_rms_mm),
            "max_abs_endpoint_error_mm": float(r.max_abs_endpoint_error_mm),
            "total_component_count_significant": int(r.total_component_count_significant),
        })
    shortlist = pd.DataFrame(shortlist_rows).sort_values(["n_reasons", "one_minus_dice"], ascending=[False, False])
    max_cases = int(cfg["inspection"].get("max_shortlist_patients", 16))
    if max_cases > 0:
        shortlist = shortlist.head(max_cases).copy()
    return ranked, shortlist


def write_case_export(pid: int, cohort: str, domain: str, cfg: Mapping,
                      outdir: Path, shortlist_row: Mapping):
    summary, components, scales, _corr, extra = case_characterization(pid, cohort, domain, cfg, True)
    assert extra is not None
    case: LoadedCase = extra["case"]
    pdir = outdir / f"patient_{pid}"
    pdir.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(case.ct, str(pdir / "ct.nii.gz"), True)
    sitk.WriteImage(image_from_array(case.approved.astype(np.uint8), case.reference), str(pdir / "approved.nii.gz"), True)
    sitk.WriteImage(image_from_array(case.autoseg.astype(np.uint8), case.reference), str(pdir / "autoseg.nii.gz"), True)
    sitk.WriteImage(image_from_array(extra["missing"].astype(np.uint8), case.reference), str(pdir / "real_missing.nii.gz"), True)
    sitk.WriteImage(image_from_array(extra["excess"].astype(np.uint8), case.reference), str(pdir / "real_excess.nii.gz"), True)
    sitk.WriteImage(image_from_array(extra["approved_surface"].astype(np.uint8), case.reference), str(pdir / "approved_surface.nii.gz"), True)

    surf_idx = np.argwhere(extra["approved_surface"])
    disp = np.asarray(extra["surface_displacement"], dtype=np.float32)
    disp_vol = np.zeros(case.approved.shape, dtype=np.float32)
    if len(surf_idx) == len(disp):
        disp_vol[tuple(surf_idx.T)] = disp
    sitk.WriteImage(image_from_array(disp_vol, case.reference, sitk.sitkFloat32), str(pdir / "surface_displacement_mm.nii.gz"), True)

    primary_sigma = float(cfg["error_field"].get("primary_smooth_sigma_mm", 20.0))
    sp = np.asarray(extra["sample_points_mm"])
    fields = extra["smooth_fields"]
    smooth_sample, local_sample = fields.get(primary_sigma, (np.empty(0), np.empty(0)))
    smooth_full = np.zeros_like(disp, dtype=np.float32)
    local_full = np.zeros_like(disp, dtype=np.float32)
    if len(sp) and len(smooth_sample) and len(surf_idx):
        full_points_mm = surf_idx.astype(float) * np.asarray(case.spacing_zyx, dtype=float)
        tree = cKDTree(sp)
        _d, nn = tree.query(full_points_mm, k=1, workers=1)
        smooth_full = np.asarray(smooth_sample, dtype=np.float32)[nn]
        local_full = np.asarray(local_sample, dtype=np.float32)[nn]
    smooth_vol = np.zeros(case.approved.shape, dtype=np.float32)
    local_vol = np.zeros(case.approved.shape, dtype=np.float32)
    if len(surf_idx):
        smooth_vol[tuple(surf_idx.T)] = smooth_full
        local_vol[tuple(surf_idx.T)] = local_full
    sitk.WriteImage(image_from_array(smooth_vol, case.reference, sitk.sitkFloat32), str(pdir / "smooth_surface_displacement_mm.nii.gz"), True)
    sitk.WriteImage(image_from_array(local_vol, case.reference, sitk.sitkFloat32), str(pdir / "local_surface_residual_mm.nii.gz"), True)

    thr = float(cfg["inspection"].get("surface_display_threshold_mm", 2.0))
    binary_maps = {
        "surface_inward_ge_threshold.nii.gz": disp_vol <= -thr,
        "surface_outward_ge_threshold.nii.gz": disp_vol >= thr,
        "smooth_inward_ge_threshold.nii.gz": smooth_vol <= -thr,
        "smooth_outward_ge_threshold.nii.gz": smooth_vol >= thr,
        "local_inward_ge_threshold.nii.gz": local_vol <= -thr,
        "local_outward_ge_threshold.nii.gz": local_vol >= thr,
    }
    for name, arr in binary_maps.items():
        sitk.WriteImage(image_from_array(arr.astype(np.uint8), case.reference), str(pdir / name), True)

    # Export largest meaningful missing/excess components as separate masks.
    for kind, mask in (("missing", extra["missing"]), ("excess", extra["excess"])):
        labels, _ = ndimage.label(mask, structure=STRUCT6)
        rows = extra[f"{kind}_rows"]
        if rows:
            lab = int(rows[0]["component_label"])
            arr = labels == lab
            sitk.WriteImage(image_from_array(arr.astype(np.uint8), case.reference), str(pdir / f"largest_{kind}_component.nii.gz"), True)

    pd.DataFrame(components).to_csv(pdir / "components.csv", index=False)
    pd.DataFrame(scales).to_csv(pdir / "multiscale_surface_summary.csv", index=False)
    payload = {
        "patient_id": int(pid), "cohort": cohort, "domain": domain,
        "inspection_categories": str(shortlist_row.get("inspection_categories", "")),
        "inspection_reasons": str(shortlist_row.get("inspection_reasons", "")),
        "surface_display_threshold_mm": thr,
        "summary": summary,
    }
    (pdir / "metrics.json").write_text(json.dumps(payload, indent=2))
    return int(pid)


def export_worker_task(args):
    pid, cohort, domain, cfg, outdir, shortlist_row = args
    try:
        out = write_case_export(pid, cohort, domain, cfg, Path(outdir), shortlist_row)
        return {"ok": True, "patient_id": out}
    except Exception as exc:
        return {"ok": False, "patient_id": pid, "error": repr(exc)}


def plot_outputs(case_df: pd.DataFrame, component_df: pd.DataFrame, scale_df: pd.DataFrame, outdir: Path,
                 roi: str = "Lung_L"):
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(case_df["missing_fraction_gt"], case_df["excess_fraction_gt"], alpha=0.8)
    lim = max(float(case_df["missing_fraction_gt"].max()), float(case_df["excess_fraction_gt"].max()), 0.01)
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    ax.set_xlabel(f"Missing fraction of approved {roi}")
    ax.set_ylabel(f"Excess fraction of approved {roi}")
    ax.set_title(f"Real {roi} missing vs excess error")
    fig.tight_layout(); fig.savefig(figdir / "missing_vs_excess.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(case_df["smooth_rms_mm"], case_df["local_rms_mm"], alpha=0.8)
    ax.set_xlabel("Smooth displacement RMS (mm)")
    ax.set_ylabel("Local residual RMS (mm)")
    ax.set_title("Broad vs local surface error")
    fig.tight_layout(); fig.savefig(figdir / "smooth_vs_local_rms.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    vals = case_df["surface_displacement_abs_p95_mm"].dropna().to_numpy(float)
    ax.hist(vals, bins=min(18, max(6, int(math.sqrt(max(len(vals), 1))))))
    ax.set_xlabel("P95 |surface displacement| (mm)")
    ax.set_ylabel("Patients")
    ax.set_title(f"{roi} real surface-error severity")
    fig.tight_layout(); fig.savefig(figdir / "surface_displacement_p95_hist.png", dpi=170); plt.close(fig)

    if len(component_df):
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        groups = [
            component_df.loc[component_df.component_type == "missing", "volume_cc"].to_numpy(float),
            component_df.loc[component_df.component_type == "excess", "volume_cc"].to_numpy(float),
        ]
        ax.boxplot(groups, labels=["Missing", "Excess"], showfliers=True)
        ax.set_ylabel("Connected-component volume (cc)")
        ax.set_title("Significant real error-patch volumes")
        fig.tight_layout(); fig.savefig(figdir / "component_volume_boxplot.png", dpi=170); plt.close(fig)

    if len(scale_df):
        med = scale_df.groupby("smooth_sigma_mm")[["smooth_rms_mm", "local_rms_mm"]].median().reset_index()
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.plot(med["smooth_sigma_mm"], med["smooth_rms_mm"], marker="o", label="Smooth RMS")
        ax.plot(med["smooth_sigma_mm"], med["local_rms_mm"], marker="o", label="Local residual RMS")
        ax.set_xlabel("Smoothing scale (mm)")
        ax.set_ylabel("Median RMS (mm)")
        ax.set_title("Multi-scale decomposition of real surface error")
        ax.legend()
        fig.tight_layout(); fig.savefig(figdir / "multiscale_surface_error.png", dpi=170); plt.close(fig)


def fmt_med_p90(df: pd.DataFrame, col: str, digits: int = 3) -> str:
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
    if not len(x):
        return "NA"
    return f"{np.median(x):.{digits}f} / {np.percentile(x, 90):.{digits}f}"


def write_report(outdir: Path, cfg: Mapping, case_df: pd.DataFrame, component_df: pd.DataFrame,
                 scale_df: pd.DataFrame, shortlist: pd.DataFrame, failures: pd.DataFrame):
    lines = [
        f"# {cfg['roi']} direct real-error field characterization",
        "",
        "This report characterizes approved↔autoseg error directly. It does **not** fit patients to synthetic corruption families and it does not modify a training corruption distribution.",
        "",
        f"- ROI: `{cfg['roi']}`",
        f"- Calibration split: `{cfg['calibration_split']}`",
        f"- Patients analyzed successfully: **{len(case_df)}**",
        f"- Failed patients: **{len(failures)}**",
        f"- Isotropic analysis spacing: **{cfg['geometry']['isotropic_spacing_mm']} mm**",
        f"- Primary smooth/local decomposition scale: **{cfg['error_field']['primary_smooth_sigma_mm']} mm**",
        f"- Significant component minimum volume: **{cfg['components']['min_component_volume_mm3']} mm³**",
        "",
        "## Overall error geometry",
        "",
        "Values below are median / P90 across patients.",
        "",
        "| Quantity | Median / P90 |",
        "|---|---:|",
        f"| 1−Dice | {fmt_med_p90(case_df, 'one_minus_dice', 4)} |",
        f"| ASSD (mm) | {fmt_med_p90(case_df, 'assd_mm')} |",
        f"| HD99 (mm) | {fmt_med_p90(case_df, 'hd99_mm')} |",
        f"| Missing fraction | {fmt_med_p90(case_df, 'missing_fraction_gt', 4)} |",
        f"| Excess fraction | {fmt_med_p90(case_df, 'excess_fraction_gt', 4)} |",
        f"| Surface displacement RMS (mm) | {fmt_med_p90(case_df, 'surface_displacement_rms_mm')} |",
        f"| P95 |surface displacement| (mm) | {fmt_med_p90(case_df, 'surface_displacement_abs_p95_mm')} |",
        f"| Smooth RMS (mm) | {fmt_med_p90(case_df, 'smooth_rms_mm')} |",
        f"| Local residual RMS (mm) | {fmt_med_p90(case_df, 'local_rms_mm')} |",
        f"| Max S/I endpoint error (mm) | {fmt_med_p90(case_df, 'max_abs_endpoint_error_mm')} |",
        f"| Significant error-patch count | {fmt_med_p90(case_df, 'total_component_count_significant', 1)} |",
        "",
        "## Missing/excess component summary",
        "",
    ]
    if len(component_df):
        comp_sum = component_df.groupby("component_type").agg(
            components=("component_type", "size"),
            median_volume_cc=("volume_cc", "median"),
            p90_volume_cc=("volume_cc", lambda x: np.percentile(x, 90)),
            median_max_depth_mm=("max_depth_from_approved_surface_mm", "median"),
            p90_max_depth_mm=("max_depth_from_approved_surface_mm", lambda x: np.percentile(x, 90)),
            median_surface_footprint=("surface_footprint_fraction", "median"),
        ).reset_index()
        lines.extend([
            "| Type | Components | Median volume (cc) | P90 volume (cc) | Median max depth (mm) | P90 max depth (mm) | Median surface footprint |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for r in comp_sum.itertuples():
            lines.append(
                f"| {r.component_type} | {r.components} | {r.median_volume_cc:.3f} | {r.p90_volume_cc:.3f} | "
                f"{r.median_max_depth_mm:.2f} | {r.p90_max_depth_mm:.2f} | {r.median_surface_footprint:.4f} |"
            )
    else:
        lines.append("No components passed the configured minimum-volume threshold.")

    lines.extend(["", "## Domain-stratified summary", ""])
    domain = case_df.groupby("domain", dropna=False).agg(
        n=("patient_id", "size"),
        median_1_minus_dice=("one_minus_dice", "median"),
        median_missing=("missing_fraction_gt", "median"),
        median_excess=("excess_fraction_gt", "median"),
        median_surface_rms_mm=("surface_displacement_rms_mm", "median"),
        median_smooth_fraction=("smooth_energy_fraction", "median"),
    ).reset_index()
    lines.extend([
        "| Domain | n | Median 1−Dice | Median missing | Median excess | Median surface RMS (mm) | Median smooth energy fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in domain.itertuples():
        lines.append(
            f"| {r.domain} | {r.n} | {r.median_1_minus_dice:.4f} | {r.median_missing:.4f} | "
            f"{r.median_excess:.4f} | {r.median_surface_rms_mm:.3f} | {r.median_smooth_fraction:.3f} |"
        )

    lines.extend([
        "",
        "## Recommended patients to inspect",
        "",
        "The shortlist intentionally spans different real-error phenotypes rather than simply choosing the highest-error patients.",
        "",
        "| Patient | Domain | Why inspect | 1−Dice | Missing | Excess | Surface RMS (mm) | Smooth/local RMS (mm) | Endpoint max (mm) | Patches |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in shortlist.itertuples():
        lines.append(
            f"| {r.patient_id} | {r.domain} | {str(r.inspection_categories).replace('|','/')} | {r.one_minus_dice:.4f} | "
            f"{r.missing_fraction_gt:.4f} | {r.excess_fraction_gt:.4f} | {r.surface_displacement_rms_mm:.2f} | "
            f"{r.smooth_rms_mm:.2f}/{r.local_rms_mm:.2f} | {r.max_abs_endpoint_error_mm:.1f} | {r.total_component_count_significant} |"
        )

    lines.extend([
        "",
        "### How to inspect the shortlist",
        "",
        "Use `slicer/LungErrorFieldViewer.py` and point it to the report's `inspection/` directory. The export contains approved/autoseg, real missing/excess masks, signed surface-displacement volumes, broad smooth/local residual volumes, and binary ≥threshold inward/outward surface masks.",
        "",
        "## Interpretation",
        "",
        "- `surface_displacement_mm` is positive for outward/excess displacement and negative for inward/missing displacement of the approved surface toward the nearest autoseg surface.",
        "- Smooth/local decomposition is descriptive; it is not yet a proposed synthetic corruption generator.",
        "- Connected components are filtered only for reporting by the configured physical-volume threshold; raw counts are retained in the case table.",
        "- Normalized component locations are broad empirical descriptors. With this dataset size they should not be interpreted as a detailed anatomical probability map.",
        "- Month3/testing1 is not used by this calibration when the default configuration is used.",
        "",
        "## Files",
        "",
        "- `case_error_field_summary.csv`: one row per patient",
        "- `error_components.csv`: one row per significant missing/excess patch",
        "- `surface_multiscale_summary.csv`: smooth/local decomposition by scale",
        "- `surface_correlation_curves.csv`: sampled displacement autocorrelation by physical separation",
        "- `inspection_recommendations.csv`: category-specific patient rankings",
        "- `inspection_shortlist.csv`: deduplicated patients exported for Slicer inspection",
        "- `inspection/`: Slicer-compatible selected cases",
    ])
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    split, cohorts = build_temporal_split(Path(cfg["split_yaml"]))
    calibration_split = str(cfg.get("calibration_split", "training1"))
    if calibration_split == "testing1":
        raise ValueError("Refusing to calibrate error fields on testing1/Month3")
    patient_ids = [int(x) for x in split[calibration_split]]
    domain_map = load_domains(Path(cfg["domain_csv"]) if cfg.get("domain_csv") else None,
                              str(cfg["roi"]), str(cfg.get("site", "thorax")))

    requested_workers = args.workers if args.workers is not None else int(cfg["parallel"].get("workers", 0))
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = requested_workers if requested_workers and requested_workers > 0 else slurm_cpus
    workers = max(1, min(int(workers), len(patient_ids)))
    print(f"Characterizing {len(patient_ids)} {cfg['roi']} cases with {workers} patient-level workers")

    tasks = [
        (pid, cohorts.get(pid, "unknown"), domain_map.get(pid, "Unassigned"), cfg)
        for pid in patient_ids
    ]
    summaries, components, scales, corrs, failures = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(worker_task, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(future_map):
            result = fut.result()
            done += 1
            if result["ok"]:
                summaries.append(result["summary"])
                components.extend(result["components"])
                scales.extend(result["scales"])
                corrs.extend(result["corr"])
                print(f"[{done}/{len(tasks)}] patient {result['patient_id']} complete")
            else:
                failures.append({"patient_id": result["patient_id"], "error": result["error"]})
                print(f"[{done}/{len(tasks)}] patient {result['patient_id']} FAILED: {result['error']}")

    case_df = pd.DataFrame(summaries).sort_values("patient_id") if summaries else pd.DataFrame()
    component_df = pd.DataFrame(components)
    scale_df = pd.DataFrame(scales)
    corr_df = pd.DataFrame(corrs)
    failure_df = pd.DataFrame(failures)
    case_df.to_csv(outdir / "case_error_field_summary.csv", index=False)
    component_df.to_csv(outdir / "error_components.csv", index=False)
    scale_df.to_csv(outdir / "surface_multiscale_summary.csv", index=False)
    corr_df.to_csv(outdir / "surface_correlation_curves.csv", index=False)
    failure_df.to_csv(outdir / "failures.csv", index=False)

    ranked, shortlist = make_inspection_recommendations(case_df, cfg)
    ranked.to_csv(outdir / "inspection_recommendations.csv", index=False)
    shortlist.to_csv(outdir / "inspection_shortlist.csv", index=False)

    if bool(cfg["inspection"].get("export_shortlist", True)) and len(shortlist):
        inspect_root = outdir / "inspection"
        inspect_root.mkdir(exist_ok=True)
        export_tasks = []
        for r in shortlist.to_dict("records"):
            pid = int(r["patient_id"])
            export_tasks.append((
                pid, cohorts.get(pid, "unknown"), domain_map.get(pid, "Unassigned"),
                cfg, str(inspect_root), r,
            ))
        export_workers = max(1, min(workers, int(cfg["inspection"].get("export_workers", workers)), len(export_tasks)))
        print(f"Exporting {len(export_tasks)} shortlisted patients with {export_workers} workers")
        export_failures = []
        with ProcessPoolExecutor(max_workers=export_workers) as ex:
            future_map = {ex.submit(export_worker_task, t): t[0] for t in export_tasks}
            for fut in as_completed(future_map):
                result = fut.result()
                if result["ok"]:
                    print(f"  exported patient {result['patient_id']}")
                else:
                    export_failures.append(result)
                    print(f"  export FAILED patient {result['patient_id']}: {result['error']}")
        if export_failures:
            pd.DataFrame(export_failures).to_csv(outdir / "inspection_export_failures.csv", index=False)

    plot_outputs(case_df, component_df, scale_df, outdir, roi=str(cfg["roi"]))
    write_report(outdir, cfg, case_df, component_df, scale_df, shortlist, failure_df)
    print(f"Wrote direct {cfg['roi']} error-field characterization to {outdir}")


if __name__ == "__main__":
    main()
