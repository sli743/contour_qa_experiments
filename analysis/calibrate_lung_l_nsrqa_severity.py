#!/usr/bin/env python3
"""Calibrate lung NSR-QA corruption severities from measured real errors.

Originally written for Lung_L (lung_l_nsrqa_severity_calibration_v1, the
severity config behind lung_l_nsrqa_corruption_ablation_month_temporal_v1).
The config key ``roi`` (default Lung_L) selects the ROI, so the identical
method can be applied to Lung_R from a Lung_R error-field characterization.

This is intentionally a calibration / visual-audit stage.  It does not modify
training code and it does not fit each clinical case to a synthetic corruption
family.  Instead it maps directly measured Lung_L error-field statistics to
severity parameters of the existing stochastic-region-dropout NSR-QA sampler:

* smooth_warp: real smooth signed-surface displacement magnitude,
* distance_noise: real local residual displacement magnitude,
* stochastic_region_dropout: largest real missing-patch volume/depth,
* global_boundary_offset: broad, predominantly one-sided displacement.

The corruption vocabulary, clean probability, branch probabilities, spatial
correlation ranges, and sampler implementation are copied unchanged from the
specified previous NSR-QA Lung_L config.  Only severity ranges are proposed.

Outputs include a recommended full training config, a random synthetic audit
bank (metrics only), and a cropped NIfTI example bank compatible with the
existing Slicer TrainingCorruptionExampleViewer.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

from analysis.spinalcord_diagnostic_utils import pair_metrics
from training.corruptions.deformation_noise import signed_distance_noise
from training.corruptions.stochastic_region_dropout import (
    global_signed_distance_offset,
    smooth_random_warp_fast,
    stochastic_region_dropout,
    stochastic_region_dropout_corruption,
)


BRANCHES = [
    "smooth_warp",
    "distance_noise",
    "combined",
    "stochastic_region_dropout",
    "global_boundary_offset",
]

PATIENT_TOKEN_RE = re.compile(r"patient[_-]?(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dino_shape/lung_l_nsrqa_severity_calibration.yaml")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def exact_patient_id(path: Path) -> int | None:
    m = PATIENT_TOKEN_RE.search(path.name)
    return int(m.group(1)) if m else None


def resolve_data_dir(root: Path) -> Path:
    """Resolve configured data dirs and tolerate the legacy ``images``/``ct`` name.

    The Lung_L preprocessed training tree stores CTs under ``ct/``.  The first
    calibration config accidentally used ``images/``.  Prefer the configured
    path when it exists; otherwise try the sibling directory name before
    raising so older configs remain runnable.
    """
    root = Path(root).expanduser()
    if root.is_dir():
        return root.resolve()
    sibling_names = {"images": "ct", "ct": "images"}
    alt_name = sibling_names.get(root.name.lower())
    if alt_name is not None:
        alt = root.with_name(alt_name)
        if alt.is_dir():
            print(f"Configured data directory does not exist: {root}")
            print(f"Using existing sibling directory instead: {alt}")
            return alt.resolve()
    raise FileNotFoundError(root)


def build_patient_index(root: Path) -> Dict[int, Path]:
    root = resolve_data_dir(root)
    out: Dict[int, Path] = {}
    collisions: Dict[int, List[Path]] = {}
    for path in root.rglob("*.nii.gz"):
        pid = exact_patient_id(path)
        if pid is None:
            continue
        if pid in out and out[pid] != path:
            collisions.setdefault(pid, [out[pid]]).append(path)
        else:
            out[pid] = path
    if collisions:
        details = "; ".join(f"{pid}: {[str(x) for x in paths]}" for pid, paths in sorted(collisions.items()))
        raise RuntimeError(f"Multiple files resolve to the same exact patient ID under {root}: {details}")
    return out


def finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    x = pd.to_numeric(df[column], errors="coerce")
    return x[np.isfinite(x)].astype(float)


def quantiles(values: Iterable[float], probs: Sequence[float]) -> Dict[float, float]:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return {float(p): float("nan") for p in probs}
    return {float(p): float(np.quantile(x, p)) for p in probs}


def clipped_triplet(values: Iterable[float], floor: float, ceiling: float | None = None,
                    probs: Sequence[float] = (0.10, 0.50, 0.90)) -> Dict[str, float]:
    q = quantiles(values, probs)
    vals = [q[float(p)] for p in probs]
    if not all(np.isfinite(vals)):
        raise ValueError("Cannot derive severity quantiles from an empty/non-finite distribution")
    vals = [max(float(floor), float(v)) for v in vals]
    if ceiling is not None:
        vals = [min(float(ceiling), v) for v in vals]
    vals = sorted(vals)
    # Keep a nonzero interval for the existing uniform sampler.
    if vals[2] <= vals[0] + 1e-6:
        vals[2] = vals[0] + max(0.25, 0.25 * vals[0])
        if ceiling is not None:
            vals[2] = min(float(ceiling), vals[2])
    return {"p10": vals[0], "p50": vals[1], "p90": vals[2], "n": int(len(list(values))) if not isinstance(values, pd.Series) else int(values.notna().sum())}


def calibrate_severity(case_df: pd.DataFrame, component_df: pd.DataFrame, cfg: Mapping) -> Tuple[Dict, pd.DataFrame]:
    cal = cfg["calibration"]
    exclusions = {int(x) for x in cal.get("exclude_patient_ids", [])}
    x = case_df.copy()
    if exclusions:
        x = x[~x["patient_id"].astype(int).isin(exclusions)].copy()
    if len(x) < int(cal.get("minimum_cases", 12)):
        raise RuntimeError(f"Only {len(x)} calibration cases remain after exclusions")

    floor_mm = float(cal.get("minimum_effective_mm", 0.75))
    max_mm = float(cal.get("maximum_allowed_mm", 20.0))

    smooth = finite_series(x, "smooth_abs_p95_mm")
    smooth_q = clipped_triplet(smooth, floor_mm, max_mm)

    # signed_distance_noise adds amplitude_mm * standardized Gaussian noise.
    # For a standard normal, |Z| P95 ~= 1.96.  Using this conversion keeps the
    # initial proposal in the physical scale of the measured local residual;
    # the exact-sampler audit bank below checks the realized severity.
    local = finite_series(x, "local_abs_p95_mm") / float(cal.get("distance_noise_abs_p95_factor", 1.96))
    local_q = clipped_triplet(local, floor_mm, max_mm)

    # Whole-surface offsets should be calibrated only from broad, mostly one-
    # sided real errors.  This avoids treating localized bites/blobs as global
    # erosion/dilation severity.
    dominant = pd.concat([
        finite_series(x, "surface_outward_fraction").rename("outward"),
        finite_series(x, "surface_inward_fraction").rename("inward"),
    ], axis=1).max(axis=1)
    smooth_energy = pd.to_numeric(x.get("smooth_energy_fraction", np.nan), errors="coerce")
    med = pd.to_numeric(x.get("surface_displacement_median_mm", np.nan), errors="coerce").abs()
    broad = med[(dominant >= float(cal.get("global_offset_min_one_sided_fraction", 0.65))) &
                (smooth_energy >= float(cal.get("global_offset_min_smooth_energy", 0.50))) & np.isfinite(med)]
    min_global_n = int(cal.get("global_offset_min_cases", 5))
    if len(broad) < min_global_n:
        broad = med[np.isfinite(med)]
        global_source = "fallback_all_cases_abs_surface_median"
    else:
        global_source = "broad_one_sided_cases_abs_surface_median"
    global_q = clipped_triplet(broad, floor_mm, max_mm)

    comp = component_df.copy()
    if len(comp):
        comp = comp[(comp["component_type"].astype(str) == "missing")]
        if exclusions:
            comp = comp[~comp["patient_id"].astype(int).isin(exclusions)]
        if "component_rank_by_volume" in comp.columns:
            comp = comp[pd.to_numeric(comp["component_rank_by_volume"], errors="coerce") == 1]
    frac = finite_series(comp, "fraction_approved_volume")
    depth = finite_series(comp, "max_depth_from_approved_surface_mm")
    if len(frac) < int(cal.get("dropout_min_patch_cases", 5)):
        # Fall back to total per-case missing fraction; this is less specific,
        # therefore it is called out in the report/CSV.
        frac = finite_series(x, "missing_fraction_gt")
        dropout_source = "fallback_case_total_missing_fraction"
    else:
        dropout_source = "largest_significant_missing_component"
    frac_q = clipped_triplet(
        frac,
        float(cal.get("dropout_min_removed_fraction", 0.005)),
        float(cal.get("dropout_max_removed_fraction", 0.40)),
    )
    if len(depth) >= int(cal.get("dropout_min_patch_cases", 5)):
        depth_q = clipped_triplet(
            depth,
            float(cal.get("dropout_min_depth_floor_mm", 1.5)),
            float(cal.get("dropout_max_depth_ceiling_mm", 60.0)),
        )
    else:
        # Retain the previous-generation depth acceptance window when the
        # measured significant-patch sample is too sparse.
        base_dropout = cfg["_base_training_cfg"]["stochastic_region_dropout"]["dropout"]
        depth_q = {
            "p10": float(base_dropout.get("min_max_depth_mm", 6.0)),
            "p50": float(np.mean([base_dropout.get("min_max_depth_mm", 6.0), base_dropout.get("max_max_depth_mm", 55.0)])),
            "p90": float(base_dropout.get("max_max_depth_mm", 55.0)),
            "n": int(len(depth)),
        }

    rows = [
        {"family": "smooth_warp", "parameter": "max_displacement_mm", "source_measure": "smooth_abs_p95_mm", **smooth_q},
        {"family": "distance_noise", "parameter": "amplitude_mm", "source_measure": "local_abs_p95_mm / 1.96", **local_q},
        {"family": "global_boundary_offset", "parameter": "magnitude_mm", "source_measure": global_source, **global_q},
        {"family": "stochastic_region_dropout", "parameter": "removed_fraction", "source_measure": dropout_source, **frac_q},
        {"family": "stochastic_region_dropout", "parameter": "max_depth_mm", "source_measure": "largest missing patch max depth", **depth_q},
    ]
    return {
        "smooth_warp": smooth_q,
        "distance_noise": local_q,
        "global_boundary_offset": global_q,
        "dropout_fraction": frac_q,
        "dropout_depth": depth_q,
        "global_offset_source": global_source,
        "dropout_source": dropout_source,
        "n_cases_used": int(len(x)),
        "excluded_patient_ids": sorted(exclusions),
    }, pd.DataFrame(rows)


def cfg_roi(cfg: Mapping) -> str:
    return str(cfg.get("roi", "Lung_L"))


def make_recommended_training_config(base_cfg: Mapping, sev: Mapping, cfg: Mapping) -> Dict:
    out = copy.deepcopy(dict(base_cfg))
    roi = cfg_roi(cfg)
    out["experiment_name"] = str(cfg["output"].get("recommended_experiment_name", f"dino_shape_stochastic_region_dropout_{roi}_data_calibrated"))
    # When calibrating another ROI from a shared base config, make the data
    # block describe that ROI (downstream prepare scripts override data paths).
    # Lung_L (same ROI as the base config) keeps the original output unchanged.
    if (out.get("data") or {}).get("roi_name") != roi:
        out["data"] = {"roi_name": roi, **dict(cfg.get("data") or {})}
    spec = out["stochastic_region_dropout"]
    # Deliberately leave clean_probability, branch_probabilities and all
    # correlation/localization parameters untouched.
    spec["smooth_warp"]["max_displacement_mm"] = [float(sev["smooth_warp"]["p10"]), float(sev["smooth_warp"]["p90"])]
    spec["distance_noise"]["amplitude_mm"] = [float(sev["distance_noise"]["p10"]), float(sev["distance_noise"]["p90"])]
    spec["global_boundary_offset"]["magnitude_mm"] = [float(sev["global_boundary_offset"]["p10"]), float(sev["global_boundary_offset"]["p90"])]
    spec["dropout"]["removed_fraction"] = [float(sev["dropout_fraction"]["p10"]), float(sev["dropout_fraction"]["p90"])]
    spec["dropout"]["min_max_depth_mm"] = float(sev["dropout_depth"]["p10"])
    spec["dropout"]["max_max_depth_mm"] = float(sev["dropout_depth"]["p90"])
    out.setdefault("study_metadata", {})["severity_calibration"] = f"{roi} direct real-error field characterization, training1 only"
    out["study_metadata"]["severity_calibration_roi"] = roi
    out["study_metadata"]["severity_calibration_error_field_root"] = str(cfg.get("error_field_root"))
    out["study_metadata"]["severity_calibration_note"] = (
        "Corruption families/probabilities/correlation settings preserved from base config; only severity ranges changed."
    )
    return out


def image_array(path: Path) -> Tuple[sitk.Image, np.ndarray]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) > 0
    if not arr.any():
        raise ValueError(f"Empty mask: {path}")
    return img, arr


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def resample_to_reference(image: sitk.Image, ref: sitk.Image, nearest: bool) -> sitk.Image:
    if same_geometry(image, ref):
        return image
    return sitk.Resample(
        image, ref, sitk.Transform(),
        sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear,
        0, image.GetPixelID(),
    )


def to_image(arr: np.ndarray, ref: sitk.Image, pixel_type=sitk.sitkUInt8) -> sitk.Image:
    out = sitk.GetImageFromArray(arr.astype(np.uint8 if pixel_type == sitk.sitkUInt8 else np.float32))
    out.CopyInformation(ref)
    return sitk.Cast(out, pixel_type)


def pair_metrics_plus(clean: np.ndarray, corrupted: np.ndarray, spacing_zyx: Sequence[float]) -> Dict[str, float]:
    m = pair_metrics(clean, corrupted, spacing_zyx)
    gv = max(int(clean.sum()), 1)
    missing = clean & ~corrupted
    excess = corrupted & ~clean
    m.update({
        "missing_fraction_gt": float(missing.sum() / gv),
        "excess_fraction_gt": float(excess.sum() / gv),
        "changed_fraction_gt": float((missing | excess).sum() / gv),
    })
    return m


def forced_branch(clean: np.ndarray, spacing_zyx: Sequence[float], spec: Mapping, branch: str,
                  rng: random.Random, severity_values: Mapping[str, float] | None = None,
                  direction: str | None = None, cache_key: str | None = None) -> Tuple[np.ndarray, Dict]:
    sev = dict(severity_values or {})
    work = clean.astype(bool, copy=True)
    meta: Dict[str, object] = {"branch": branch}

    if branch == "smooth_warp":
        sub = copy.deepcopy(spec["smooth_warp"])
        if "smooth_warp_mm" in sev:
            sub["max_displacement_mm"] = [float(sev["smooth_warp_mm"])] * 2
        work, x = smooth_random_warp_fast(work, spacing_zyx, sub, rng); meta.update(x)
    elif branch == "distance_noise":
        sub = copy.deepcopy(spec["distance_noise"])
        if "distance_noise_mm" in sev:
            sub["amplitude_mm"] = [float(sev["distance_noise_mm"])] * 2
        work, x = signed_distance_noise(work, spacing_zyx, sub, rng); meta.update(x)
    elif branch == "combined":
        subw = copy.deepcopy(spec["smooth_warp"])
        subn = copy.deepcopy(spec["distance_noise"])
        if "smooth_warp_mm" in sev:
            subw["max_displacement_mm"] = [float(sev["smooth_warp_mm"])] * 2
        if "distance_noise_mm" in sev:
            subn["amplitude_mm"] = [float(sev["distance_noise_mm"])] * 2
        work, x = smooth_random_warp_fast(work, spacing_zyx, subw, rng); meta.update(x)
        work, x = signed_distance_noise(work, spacing_zyx, subn, rng); meta.update(x)
    elif branch == "stochastic_region_dropout":
        sub = copy.deepcopy(spec["dropout"])
        if "dropout_fraction" in sev:
            sub["removed_fraction"] = [float(sev["dropout_fraction"])] * 2
        work, x = stochastic_region_dropout(work, spacing_zyx, sub, rng, cache_key=cache_key); meta.update(x)
    elif branch == "global_boundary_offset":
        sub = copy.deepcopy(spec["global_boundary_offset"])
        if "global_offset_mm" in sev:
            sub["magnitude_mm"] = [float(sev["global_offset_mm"])] * 2
        if direction == "inward":
            sub["outward_probability"] = 0.0
        elif direction == "outward":
            sub["outward_probability"] = 1.0
        work, x = global_signed_distance_offset(work, spacing_zyx, sub, rng); meta.update(x)
    else:
        raise ValueError(branch)
    return work.astype(bool), meta


def crop_union(images: Mapping[str, sitk.Image], mask_arrays: Sequence[np.ndarray], margin_mm: float) -> Dict[str, sitk.Image]:
    union = np.zeros(mask_arrays[0].shape, dtype=bool)
    for arr in mask_arrays:
        union |= arr.astype(bool)
    pts = np.argwhere(union)
    if not len(pts):
        return dict(images)
    ref = next(iter(images.values()))
    spacing_zyx = np.asarray(ref.GetSpacing()[::-1], dtype=float)
    margin = np.ceil(float(margin_mm) / spacing_zyx).astype(int)
    lo = np.maximum(0, pts.min(axis=0) - margin)
    hi = np.minimum(np.asarray(union.shape) - 1, pts.max(axis=0) + margin)
    index_xyz = [int(lo[2]), int(lo[1]), int(lo[0])]
    size_xyz = [int(hi[2] - lo[2] + 1), int(hi[1] - lo[1] + 1), int(hi[0] - lo[0] + 1)]
    return {k: sitk.RegionOfInterest(v, size_xyz, index_xyz) for k, v in images.items()}


def audit_worker(task: Tuple) -> Dict:
    sample_id, pid, label_path, proposed_cfg, seed = task
    try:
        try: sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        except Exception: pass
        ref, clean = image_array(Path(label_path))
        spacing = tuple(float(v) for v in ref.GetSpacing()[::-1])
        t = torch.from_numpy(clean.astype(np.float32, copy=False))[None]
        corrupted_t, _changed_t, meta = stochastic_region_dropout_corruption(
            t, proposed_cfg, random.Random(int(seed)), cache_key=f"audit-{pid}"
        )
        corrupted = corrupted_t.numpy()[0] > 0.5
        metrics = pair_metrics_plus(clean, corrupted, spacing)
        return {"ok": True, "sample_id": int(sample_id), "patient_id": int(pid), **meta, **metrics}
    except Exception as exc:
        return {"ok": False, "sample_id": int(sample_id), "patient_id": int(pid), "error": repr(exc)}


def select_example_patients(case_df: pd.DataFrame, available: Sequence[int], cfg: Mapping) -> List[int]:
    explicit = [int(x) for x in cfg["example_bank"].get("patient_ids", [])]
    avail = {int(x) for x in available}
    if explicit:
        chosen = [x for x in explicit if x in avail]
        if not chosen:
            raise ValueError("None of example_bank.patient_ids are available in the training labels")
        return chosen
    n = int(cfg["example_bank"].get("n_patients", 5))
    x = case_df[case_df["patient_id"].astype(int).isin(avail)].copy()
    if not len(x):
        return sorted(avail)[:n]
    vol_col = "approved_volume_cc" if "approved_volume_cc" in x.columns else None
    if not vol_col:
        return x["patient_id"].astype(int).tolist()[:n]
    x = x[np.isfinite(pd.to_numeric(x[vol_col], errors="coerce"))].copy()
    if not len(x):
        return sorted(avail)[:n]
    probs = np.linspace(0.05, 0.95, min(n, len(x)))
    chosen: List[int] = []
    vals = pd.to_numeric(x[vol_col], errors="coerce")
    for q in probs:
        target = float(np.quantile(vals, q))
        order = np.argsort(np.abs(vals.to_numpy(dtype=float) - target))
        for idx in order:
            pid = int(x.iloc[int(idx)]["patient_id"])
            if pid not in chosen:
                chosen.append(pid); break
    return chosen[:n]


def severity_for_stratum(sev: Mapping, stratum: str) -> Dict[str, float]:
    key = {"low": "p10", "typical": "p50", "high": "p90"}[stratum]
    return {
        "smooth_warp_mm": float(sev["smooth_warp"][key]),
        "distance_noise_mm": float(sev["distance_noise"][key]),
        "global_offset_mm": float(sev["global_boundary_offset"][key]),
        "dropout_fraction": float(sev["dropout_fraction"][key]),
    }


def export_example_case(pid: int, label_path: Path, ct_path: Path, outdir: Path, proposed_cfg: Mapping,
                        sev: Mapping, stratum: str, seed: int, margin_mm: float) -> List[Dict]:
    ref, clean = image_array(label_path)
    ct = resample_to_reference(sitk.ReadImage(str(ct_path), sitk.sitkFloat32), ref, nearest=False)
    spacing = tuple(float(v) for v in ref.GetSpacing()[::-1])
    svals = severity_for_stratum(sev, stratum)
    outputs: Dict[str, np.ndarray] = {"clean": clean}
    rows: List[Dict] = []

    branch_specs = [
        ("smooth_warp", "smooth_warp", None),
        ("distance_noise", "distance_noise", None),
        ("combined", "combined", None),
        ("stochastic_region_dropout", "stochastic_region_dropout", None),
        ("global_boundary_offset_inward", "global_boundary_offset", "inward"),
        ("global_boundary_offset_outward", "global_boundary_offset", "outward"),
    ]
    for j, (tag, branch, direction) in enumerate(branch_specs):
        corr, meta = forced_branch(
            clean, spacing, proposed_cfg["stochastic_region_dropout"], branch,
            random.Random(int(seed) + j * 100003), svals, direction=direction,
            cache_key=f"example-{stratum}-{pid}-{tag}",
        )
        outputs[tag] = corr
        metrics = pair_metrics_plus(clean, corr, spacing)
        rows.append({
            "patient_id": pid, "severity_stratum": stratum, "tag": tag,
            **svals, **meta, **metrics,
        })

    image_map: Dict[str, sitk.Image] = {"ct": ct, "clean": to_image(clean, ref)}
    for tag, arr in outputs.items():
        if tag == "clean": continue
        image_map[tag] = to_image(arr, ref)
        image_map[f"changed_{tag}"] = to_image(np.logical_xor(clean, arr), ref)
        image_map[f"missing_{tag}"] = to_image(clean & ~arr, ref)
        image_map[f"excess_{tag}"] = to_image(arr & ~clean, ref)
    image_map = crop_union(image_map, list(outputs.values()), margin_mm)

    outdir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image_map["ct"], str(outdir / f"original_ct_patient_{pid}.nii.gz"), True)
    sitk.WriteImage(image_map["clean"], str(outdir / f"clean_target_patient_{pid}.nii.gz"), True)
    for tag, _branch, _direction in branch_specs:
        sitk.WriteImage(image_map[tag], str(outdir / f"corrupted_input_{tag}_patient_{pid}.nii.gz"), True)
        sitk.WriteImage(image_map[f"changed_{tag}"], str(outdir / f"changed_region_{tag}_patient_{pid}.nii.gz"), True)
        sitk.WriteImage(image_map[f"missing_{tag}"], str(outdir / f"missing_region_{tag}_patient_{pid}.nii.gz"), True)
        sitk.WriteImage(image_map[f"excess_{tag}"], str(outdir / f"excess_region_{tag}_patient_{pid}.nii.gz"), True)
    (outdir / "metadata.json").write_text(json.dumps({"patient_id": pid, "severity_stratum": stratum, "severity_values": svals, "rows": rows}, indent=2))
    return rows


def write_plots(real_df: pd.DataFrame, audit_df: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"; figdir.mkdir(exist_ok=True)
    metrics = ["one_minus_dice", "assd_mm", "hd95_mm", "hd99_mm", "missing_fraction_gt", "excess_fraction_gt"]
    for metric in metrics:
        if metric not in real_df.columns or metric not in audit_df.columns: continue
        rv = pd.to_numeric(real_df[metric], errors="coerce"); rv = rv[np.isfinite(rv)]
        sv = pd.to_numeric(audit_df[metric], errors="coerce"); sv = sv[np.isfinite(sv)]
        if not len(rv) or not len(sv): continue
        plt.figure(figsize=(7, 4.5))
        bins = 30
        plt.hist(rv, bins=bins, density=True, alpha=0.55, label="real approved-vs-autoseg")
        plt.hist(sv, bins=bins, density=True, alpha=0.55, label="proposed synthetic training")
        plt.xlabel(metric); plt.ylabel("density"); plt.legend(); plt.tight_layout()
        plt.savefig(figdir / f"real_vs_synthetic_{metric}.png", dpi=180); plt.close()


def quantile_comparison(real_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["one_minus_dice", "assd_mm", "hd95_mm", "hd99_mm", "missing_fraction_gt", "excess_fraction_gt", "changed_fraction_gt"]:
        if metric not in audit_df.columns: continue
        real = finite_series(real_df, metric)
        syn = finite_series(audit_df, metric)
        for source, vals in [("real", real), ("synthetic", syn)]:
            if not len(vals): continue
            q = quantiles(vals, [0.10, 0.50, 0.90, 0.95])
            rows.append({"metric": metric, "source": source, "n": len(vals), "p10": q[0.10], "p50": q[0.50], "p90": q[0.90], "p95": q[0.95]})
    return pd.DataFrame(rows)


def write_report(outdir: Path, cfg: Mapping, sev: Mapping, sev_df: pd.DataFrame,
                 audit_df: pd.DataFrame, audit_fail: pd.DataFrame, examples: Sequence[int]) -> None:
    base = cfg["_base_training_cfg"]["stochastic_region_dropout"]
    roi = cfg_roi(cfg)
    lines = [
        f"# {roi} NSR-QA corruption severity calibration",
        "",
        f"This stage changes **severity only**. The existing corruption families, clean probability, branch probabilities, and spatial correlation/localization settings are copied unchanged from the base NSR-QA config (`{cfg.get('base_training_config')}`). Severities come from the {roi} error-field characterization (`{cfg.get('error_field_root')}`).",
        "",
        f"Calibration cases: {sev['n_cases_used']} (training1 only). Excluded IDs: {sev['excluded_patient_ids'] or 'none'}.",
        "Month3/testing1 is not used.",
        "",
        "## Preserved mixture",
        "",
        f"- clean_probability: {base.get('clean_probability')}",
        f"- branch_probabilities: `{json.dumps(base.get('branch_probabilities', {}), sort_keys=True)}`",
        "",
        "## Proposed severity",
        "",
        sev_df.to_markdown(index=False),
        "",
        "Notes:",
        "- Smooth warp is anchored to the measured smooth surface-displacement P95 distribution.",
        "- Distance-noise amplitude uses measured local-residual P95 divided by 1.96 because the sampler multiplies a standardized Gaussian field by amplitude_mm. The exact-sampler audit below is the check on this approximation.",
        f"- Global offset source: {sev['global_offset_source']}.",
        f"- Dropout source: {sev['dropout_source']}.",
        "- Combined corruption intentionally retains the previous sampler semantics: it applies the calibrated smooth and local components sequentially. If its realized tail is too severe, inspect the bank before changing training.",
        "",
        "## Exact-sampler audit",
        "",
        f"Successful random samples: {len(audit_df)}; failures: {len(audit_fail)}.",
    ]
    if len(audit_df) and "branch" in audit_df:
        counts = audit_df["branch"].value_counts().rename_axis("branch").reset_index(name="n")
        lines += ["", counts.to_markdown(index=False)]
    lines += [
        "",
        "## Visual inspection bank",
        "",
        f"Patients selected across {roi} volume: {', '.join(map(str, examples))}",
        "",
        "Open `example_bank/` with the existing `TrainingCorruptionExampleViewer` Slicer module. The viewer will show three pseudo-ROI groups:",
        f"- `{roi}_low`: P10 physical severity",
        f"- `{roi}_typical`: P50 physical severity",
        f"- `{roi}_high`: P90 physical severity",
        "",
        "For each patient/group it exports clean target, smooth warp, distance noise, combined, stochastic dropout, inward global offset, outward global offset, and changed/missing/excess diagnostic masks.",
        "",
        "## Files",
        "",
        "- `severity_calibration.csv`: real-data P10/P50/P90 calibration table",
        "- `recommended_training_config.yaml`: previous NSR-QA config with severity ranges replaced",
        "- `random_training_distribution_audit.csv`: exact current sampler draws using the proposed config",
        "- `real_vs_synthetic_quantiles.csv`: metric distribution audit",
        "- `example_bank_manifest.csv`: exported visual examples and realized metrics",
        "- `example_bank/`: Slicer-compatible NIfTI bank",
        "- `figures/`: real-vs-synthetic distribution plots",
    ]
    (outdir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    base_path = Path(cfg["base_training_config"])
    if not base_path.is_absolute(): base_path = (Path.cwd() / base_path).resolve()
    base_cfg = yaml.safe_load(base_path.read_text())
    cfg["_base_training_cfg"] = base_cfg

    outdir = Path(cfg["output"]["directory"]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ef = Path(cfg["error_field_root"]).expanduser().resolve()
    case_path = ef / "case_error_field_summary.csv"
    comp_path = ef / "error_components.csv"
    if not case_path.is_file() or not comp_path.is_file():
        raise FileNotFoundError(
            f"Direct error-field outputs are required first: {case_path} and {comp_path}"
        )
    case_df = pd.read_csv(case_path)
    component_df = pd.read_csv(comp_path)

    sev, sev_df = calibrate_severity(case_df, component_df, cfg)
    sev_df.to_csv(outdir / "severity_calibration.csv", index=False)
    proposed_cfg = make_recommended_training_config(base_cfg, sev, cfg)
    (outdir / "recommended_training_config.yaml").write_text(yaml.safe_dump(proposed_cfg, sort_keys=False))
    (outdir / "calibration_details.json").write_text(json.dumps(sev, indent=2))

    label_root = Path(cfg["data"]["train_label_dir"]).expanduser().resolve()
    ct_root = Path(cfg["data"]["train_ct_dir"]).expanduser().resolve()
    labels = build_patient_index(label_root)
    cts = build_patient_index(ct_root)
    common = sorted(set(labels) & set(cts))
    if not common: raise RuntimeError("No exact patient IDs shared by training labels and CTs")

    # Random exact-sampler distribution audit.  This validates the actual
    # realized metric range without saving hundreds of large NIfTIs.
    audit_cfg = cfg["audit_bank"]
    n_samples = int(audit_cfg.get("n_samples", 240))
    seed0 = int(cfg.get("seed", 91))
    rng = np.random.default_rng(seed0 + 321)
    tasks = []
    for i in range(n_samples):
        pid = int(rng.choice(common))
        tasks.append((i, pid, str(labels[pid]), proposed_cfg, seed0 + 1000003 * (i + 1)))
    requested = args.workers if args.workers is not None else int(cfg.get("parallel", {}).get("workers", 0))
    slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = requested if requested and requested > 0 else slurm
    workers = max(1, min(int(workers), int(cfg.get("parallel", {}).get("max_workers", 8)), len(tasks)))
    print(f"Generating {len(tasks)} exact-sampler audit draws with {workers} workers")
    rows, fails = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fmap = {ex.submit(audit_worker, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(fmap):
            r = fut.result(); done += 1
            if r["ok"]: rows.append(r)
            else: fails.append(r)
            if done % 20 == 0 or done == len(tasks): print(f"  audit {done}/{len(tasks)}")
    audit_df = pd.DataFrame(rows).sort_values("sample_id") if rows else pd.DataFrame()
    fail_df = pd.DataFrame(fails)
    audit_df.to_csv(outdir / "random_training_distribution_audit.csv", index=False)
    fail_df.to_csv(outdir / "random_training_distribution_failures.csv", index=False)

    qcmp = quantile_comparison(case_df, audit_df)
    qcmp.to_csv(outdir / "real_vs_synthetic_quantiles.csv", index=False)
    write_plots(case_df, audit_df, outdir)

    # Small, deliberately interpretable NIfTI bank for Slicer.
    examples = select_example_patients(case_df, common, cfg)
    bank_root = outdir / "example_bank"
    manifest_rows: List[Dict] = []
    strata = [str(x) for x in cfg["example_bank"].get("severity_strata", ["low", "typical", "high"])]
    for si, stratum in enumerate(strata):
        if stratum not in {"low", "typical", "high"}: raise ValueError(stratum)
        for pi, pid in enumerate(examples):
            case_dir = bank_root / f"{cfg_roi(cfg)}_{stratum}" / f"patient_{pid}"
            rows = export_example_case(
                pid, labels[pid], cts[pid], case_dir, proposed_cfg, sev, stratum,
                seed0 + si * 10000019 + pi * 100003,
                float(cfg["example_bank"].get("crop_margin_mm", 20.0)),
            )
            manifest_rows.extend(rows)
            print(f"  exported {stratum} patient {pid}")
    pd.DataFrame(manifest_rows).to_csv(outdir / "example_bank_manifest.csv", index=False)

    write_report(outdir, cfg, sev, sev_df, audit_df, fail_df, examples)
    print(f"Wrote {cfg_roi(cfg)} NSR-QA severity calibration to {outdir}")
    print(f"Recommended training config: {outdir / 'recommended_training_config.yaml'}")
    print(f"Slicer example bank: {bank_root}")


if __name__ == "__main__":
    main()
