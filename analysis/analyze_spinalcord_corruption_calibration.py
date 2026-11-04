#!/usr/bin/env python3
"""Calibrate SpinalCord synthetic corruptions against real autosegmentation errors.

Design principles
-----------------
* Real clinical error geometry is measured on the original full-volume masks.
* Month1+Month2 are the development cohort used to characterize corruption
  families/ranges. Month3 is a frozen temporal audit.
* Controlled corruption families are expressed in physical millimetres where
  possible and are used for interpretation, not model training.
* A separate training-bank audit calls the exact current NSR-QA corruption
  function on the prepared 1.5-mm development masks to quantify support of the
  distribution the model actually sees.
* No NSR-QA reconstruction score is used to design the corruption distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from scipy import stats
from scipy.spatial.distance import cdist

from analysis.spinalcord_corruption_support import mixed_support_coverage
from analysis.spinalcord_error_geometry import (
    component_matched_local_candidate,
    crop_arrays_to_union,
    crop_pair_to_union,
    endpoint_cut,
    endpoint_extension,
    extract_pair_features,
    grouped_feature_distance,
    load_raw_pair,
    longitudinal_slice_profile,
    robust_feature_scaler,
    signed_maurer_array,
    translate_no_wrap,
    weighted_error_spatial_scores,
)
from training.corruptions.stochastic_region_dropout import (
    smooth_random_warp_fast,
    stochastic_region_dropout_corruption,
)


def load_yaml(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_seed(base: int, *parts: object) -> int:
    text = "|".join([str(base), *[str(p) for p in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def float_grid(spec: Mapping[str, float]) -> List[float]:
    lo = float(spec["min"])
    hi = float(spec["max"])
    step = float(spec["step"])
    if step <= 0:
        raise ValueError("Grid step must be > 0")
    n = int(math.floor((hi - lo) / step + 0.5))
    vals = [lo + i * step for i in range(n + 1)]
    return [float(round(v, 6)) for v in vals if v <= hi + 1e-8]


def unique_ints(xs: Iterable[object]) -> List[int]:
    return sorted({int(x) for x in xs})


def temporal_cohorts(split_cfg: Mapping[str, object]) -> Tuple[List[int], List[int], List[Dict[str, object]]]:
    m1 = set(unique_ints(split_cfg["month1"]))
    m2 = set(unique_ints(split_cfg["month2"]))
    m3 = set(unique_ints(split_cfg["month3"]))
    dev = sorted((m1 | m2) - m3)
    test = sorted(m3)
    overlap_rows: List[Dict[str, object]] = []
    for pid in sorted((m1 | m2) & m3):
        earlier = []
        if pid in m1:
            earlier.append("month1")
        if pid in m2:
            earlier.append("month2")
        overlap_rows.append({
            "patient_id": pid,
            "earlier_periods": "+".join(earlier),
            "resolution": "month3_temporal_audit_priority",
        })
    return dev, test, overlap_rows


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No rows._"
    x = df.head(max_rows).copy()
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in x.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("NA" if not np.isfinite(v) else f"{float(v):.3f}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def nearest_feature_embedding(
    rows: Sequence[Mapping[str, object]],
    scaler: Mapping[str, Mapping[str, float]],
    groups: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Equal-group standardized embedding used only for support/NN distances."""
    group_names = list(groups)
    cols: List[np.ndarray] = []
    n_rows = len(rows)
    for group in group_names:
        names = list(groups[group])
        if not names:
            continue
        weight = 1.0 / math.sqrt(max(len(group_names), 1) * len(names))
        for name in names:
            med = float(scaler[name]["median"])
            scale = max(float(scaler[name]["scale"]), 1e-6)
            arr = np.asarray([float(r.get(name, np.nan)) for r in rows], dtype=float)
            arr = np.where(np.isfinite(arr), arr, med)
            cols.append(((arr - med) / scale * weight).reshape(n_rows, 1))
    return np.concatenate(cols, axis=1) if cols else np.zeros((n_rows, 0), dtype=float)


def source_month(pid: int, split_cfg: Mapping[str, object]) -> str:
    m = []
    for name in ("month1", "month2", "month3"):
        if int(pid) in set(unique_ints(split_cfg[name])):
            m.append(name)
    return "+".join(m) if m else "unknown"


def characterize_real_cases(
    patient_ids: Sequence[int],
    cohort: str,
    raw_root: Path,
    roi: str,
    margin_xy_mm: float,
    margin_z_mm: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    features: List[Dict[str, object]] = []
    components: List[Dict[str, object]] = []
    profiles: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    for i, pid in enumerate(patient_ids, start=1):
        print(f"[real:{cohort}] {i}/{len(patient_ids)} patient {pid}", flush=True)
        try:
            pair = crop_pair_to_union(load_raw_pair(raw_root, pid, roi), margin_xy_mm, margin_z_mm)
            f, comp = extract_pair_features(pair.approved, pair.autoseg, pair.spacing_zyx, pid, cohort, True)
            prof = longitudinal_slice_profile(pair.approved, pair.autoseg, pair.spacing_zyx, pid, cohort)
            features.append(f)
            components.extend(comp)
            profiles.extend(prof)
            manifest.append({"patient_id": pid, "cohort": cohort, "status": "ok", "reason": ""})
        except Exception as exc:
            manifest.append({"patient_id": pid, "cohort": cohort, "status": "skipped", "reason": str(exc)})
            print(f"  WARNING: skipped patient {pid}: {exc}", flush=True)
    return features, components, profiles, manifest


def candidate_row(
    *,
    patient_id: int,
    cohort: str,
    family: str,
    parameters: Mapping[str, object],
    approved: np.ndarray,
    real_autoseg: np.ndarray,
    candidate: np.ndarray,
    spacing_zyx: Sequence[float],
    real_features: Mapping[str, object],
    scaler: Mapping[str, Mapping[str, float]],
    groups: Mapping[str, Sequence[str]],
) -> Optional[Dict[str, object]]:
    if not candidate.any():
        return None
    metric_gt, metric_candidate = crop_arrays_to_union(approved, candidate, spacing_zyx, margin_mm=20.0)
    f, _ = extract_pair_features(metric_gt, metric_candidate, spacing_zyx, patient_id, cohort, False)
    distance, group_dist = grouped_feature_distance(real_features, f, scaler, groups)
    spatial = weighted_error_spatial_scores(approved, real_autoseg, candidate)
    row: Dict[str, object] = {
        "patient_id": int(patient_id),
        "cohort": cohort,
        "family": family,
        "phenotype_distance": float(distance),
        "phenotype_similarity_exp": float(math.exp(-distance)) if np.isfinite(distance) else np.nan,
        **spatial,
    }
    for k, v in parameters.items():
        row[f"parameter_{k}"] = v
    for group_name, value in group_dist.items():
        row[f"distance_{group_name}"] = value
    # Keep a compact candidate phenotype audit in the fit CSV.
    for name in (
        "missing_fraction_gt", "excess_fraction_gt", "relative_volume_change_signed",
        "assd_mm", "hd95_mm", "centroid_shift_mm", "inferior_missing_mm",
        "superior_missing_mm", "inferior_excess_mm", "superior_excess_mm",
        "largest_missing_component_fraction", "largest_excess_component_fraction",
    ):
        row[f"candidate_{name}"] = f.get(name, np.nan)
    return row


def controlled_candidates_for_case(
    pid: int,
    cohort: str,
    pair,
    real_features: Mapping[str, object],
    case_components: Sequence[Mapping[str, object]],
    cfg: Mapping[str, object],
    scaler: Mapping[str, Mapping[str, float]],
    groups: Mapping[str, Sequence[str]],
    seed: int,
) -> List[Dict[str, object]]:
    approved = pair.approved
    autoseg = pair.autoseg
    spacing = pair.spacing_zyx
    rows: List[Dict[str, object]] = []

    offset_grid = float_grid(cfg["controlled_families"]["global_boundary_offset_mm"])
    approved_sdf = signed_maurer_array(approved, spacing)
    for magnitude in offset_grid:
        for direction, sign in (("inward", -1.0), ("outward", 1.0)):
            candidate = (approved_sdf + sign * magnitude) >= 0.0
            row = candidate_row(
                patient_id=pid, cohort=cohort, family=f"global_{direction}_offset",
                parameters={"magnitude_mm": magnitude, "signed_offset_mm": sign * magnitude},
                approved=approved, real_autoseg=autoseg, candidate=candidate,
                spacing_zyx=spacing, real_features=real_features, scaler=scaler, groups=groups,
            )
            if row is not None:
                rows.append(row)

    endpoint_grid = float_grid(cfg["controlled_families"]["endpoint_length_mm"])
    for length in endpoint_grid:
        for side in ("inferior", "superior"):
            candidate = endpoint_cut(approved, spacing[0], side, length)
            row = candidate_row(
                patient_id=pid, cohort=cohort, family=f"{side}_cut",
                parameters={"side": side, "length_mm": length},
                approved=approved, real_autoseg=autoseg, candidate=candidate,
                spacing_zyx=spacing, real_features=real_features, scaler=scaler, groups=groups,
            )
            if row is not None:
                rows.append(row)
            candidate = endpoint_extension(approved, spacing[0], side, length)
            row = candidate_row(
                patient_id=pid, cohort=cohort, family=f"{side}_extension",
                parameters={"side": side, "length_mm": length},
                approved=approved, real_autoseg=autoseg, candidate=candidate,
                spacing_zyx=spacing, real_features=real_features, scaler=scaler, groups=groups,
            )
            if row is not None:
                rows.append(row)

    # Pure translation: use the observed centroid displacement as the parameter
    # estimate, rounded to the native analysis grid. This asks whether a rigid
    # displacement alone explains the real error pattern.
    delta_mm = np.asarray([
        float(real_features["centroid_shift_si_mm"]),
        float(real_features["centroid_shift_ap_mm"]),
        float(real_features["centroid_shift_lr_mm"]),
    ])
    shift_zyx = np.rint(delta_mm / np.asarray(spacing)).astype(int)
    if np.any(shift_zyx):
        candidate = translate_no_wrap(approved, shift_zyx)
        row = candidate_row(
            patient_id=pid, cohort=cohort, family="centroid_translation",
            parameters={
                "shift_z_vox": int(shift_zyx[0]), "shift_y_vox": int(shift_zyx[1]), "shift_x_vox": int(shift_zyx[2]),
                "magnitude_mm": float(np.linalg.norm(shift_zyx * np.asarray(spacing))),
            },
            approved=approved, real_autoseg=autoseg, candidate=candidate,
            spacing_zyx=spacing, real_features=real_features, scaler=scaler, groups=groups,
        )
        if row is not None:
            rows.append(row)

    by_kind: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for r in case_components:
        by_kind[str(r["component_kind"])].append(r)
    scales = [float(x) for x in cfg["controlled_families"].get("local_component_scale", [0.75, 1.0, 1.25])]
    for kind, family in (("missing", "local_bite"), ("excess", "local_blob")):
        if by_kind[kind]:
            largest = sorted(by_kind[kind], key=lambda x: float(x["component_voxels"]), reverse=True)[0]
            for scale in scales:
                candidate = component_matched_local_candidate(approved, largest, kind, scale)
                row = candidate_row(
                    patient_id=pid, cohort=cohort, family=family,
                    parameters={
                        "scale": scale,
                        "observed_extent_si_mm": float(largest["extent_si_mm"]),
                        "observed_extent_ap_mm": float(largest["extent_ap_mm"]),
                        "observed_extent_lr_mm": float(largest["extent_lr_mm"]),
                        "observed_fraction_gt": float(largest["fraction_of_gt_volume"]),
                    },
                    approved=approved, real_autoseg=autoseg, candidate=candidate,
                    spacing_zyx=spacing, real_features=real_features, scaler=scaler, groups=groups,
                )
                if row is not None:
                    rows.append(row)

    warp = cfg["controlled_families"].get("smooth_warp", {})
    corr_mm = float(warp.get("correlation_mm", 18.0))
    mean_spacing = float(np.mean(spacing))
    sigma_vox = max(corr_mm / max(mean_spacing, 1e-6), 0.5)
    n_rep = int(warp.get("candidates_per_amplitude", 2))
    for amplitude in [float(x) for x in warp.get("amplitudes_mm", [2.0, 4.0, 6.0])]:
        for rep in range(n_rep):
            rng = random.Random(stable_seed(seed, "controlled_warp", pid, amplitude, rep))
            spec = {
                "control_grid_points_zyx": {"min": [5, 5, 5], "max": [5, 5, 5]},
                "gaussian_sigma_vox": [sigma_vox, sigma_vox],
                "max_displacement_mm": [amplitude, amplitude],
                "upsample_order": 1,
            }
            candidate, _ = smooth_random_warp_fast(approved, spacing, spec, rng)
            row = candidate_row(
                patient_id=pid, cohort=cohort, family="smooth_warp",
                parameters={"max_displacement_mm": amplitude, "correlation_mm": corr_mm, "replicate": rep},
                approved=approved, real_autoseg=autoseg, candidate=candidate,
                spacing_zyx=spacing, real_features=real_features, scaler=scaler, groups=groups,
            )
            if row is not None:
                rows.append(row)
    return rows


def summarize_best_families(fit_df: pd.DataFrame, development_ids: set[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pid, g in fit_df.groupby("patient_id", sort=True):
        fam_best = (
            g.sort_values(["phenotype_distance", "weighted_error_dice"], ascending=[True, False])
            .groupby("family", as_index=False)
            .first()
            .sort_values("phenotype_distance")
        )
        if fam_best.empty:
            continue
        first = fam_best.iloc[0]
        second = fam_best.iloc[1] if len(fam_best) > 1 else None
        out = dict(first)
        out["best_family"] = first["family"]
        out["best_family_distance"] = float(first["phenotype_distance"])
        out["best_family_similarity"] = float(first["phenotype_similarity_exp"])
        out["second_family"] = str(second["family"]) if second is not None else ""
        out["second_family_distance"] = float(second["phenotype_distance"]) if second is not None else np.nan
        out["fit_margin_distance"] = (
            float(second["phenotype_distance"] - first["phenotype_distance"]) if second is not None else np.nan
        )
        rows.append(out)
    best = pd.DataFrame(rows)
    if best.empty:
        return best
    dev = best[best["patient_id"].astype(int).isin(development_ids)]
    if not dev.empty:
        weak_thr = float(np.percentile(dev["best_family_distance"], 90))
        finite_margin = pd.to_numeric(dev["fit_margin_distance"], errors="coerce").dropna()
        ambiguous_thr = float(np.percentile(finite_margin, 10)) if len(finite_margin) else np.nan
        best["development_p90_best_distance"] = weak_thr
        best["development_p10_fit_margin"] = ambiguous_thr
        best["low_controlled_fit"] = best["best_family_distance"] >= weak_thr
        best["ambiguous_family"] = best["fit_margin_distance"] <= ambiguous_thr if np.isfinite(ambiguous_thr) else False
    return best


def extract_pid(path: Path) -> Optional[int]:
    m = re.search(r"patient[_]?(\d+)", path.name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_training_label_files(nsrqa_cfg: Mapping[str, object]) -> List[Tuple[str, Path]]:
    data = nsrqa_cfg["data"]
    out: List[Tuple[str, Path]] = []
    for split, key in (("training1", "train_label_dir"), ("validation1", "validation_label_dir")):
        root = Path(data[key]).expanduser()
        for p in sorted(root.glob("*.nii.gz")):
            if extract_pid(p) is not None:
                out.append((split, p))
    return out


def generate_training_bank(
    nsrqa_cfg: Mapping[str, object],
    samples_per_case: int,
    seed: int,
) -> List[Dict[str, object]]:
    files = load_training_label_files(nsrqa_cfg)
    if not files:
        raise FileNotFoundError("No training/validation label NIfTI files found from NSR-QA config")
    rows: List[Dict[str, object]] = []
    for case_idx, (split, path) in enumerate(files, start=1):
        pid = extract_pid(path)
        assert pid is not None
        print(f"[training-bank] {case_idx}/{len(files)} patient {pid} ({split})", flush=True)
        img = sitk.ReadImage(str(path))
        approved = sitk.GetArrayFromImage(img) > 0
        if not approved.any():
            print(f"  WARNING: empty training mask {path}", flush=True)
            continue
        spacing_xyz = img.GetSpacing()
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
        clean = torch.from_numpy(approved.astype(np.float32, copy=False))[None]
        for sample_idx in range(int(samples_per_case)):
            rng = random.Random(stable_seed(seed, "training_bank", pid, split, sample_idx))
            corrupted, changed, meta = stochastic_region_dropout_corruption(
                clean, dict(nsrqa_cfg), rng, cache_key=f"calibration_{pid}"
            )
            candidate = corrupted.detach().cpu().numpy()[0] > 0.5
            if not candidate.any():
                continue
            metric_gt, metric_candidate = crop_arrays_to_union(approved, candidate, spacing_zyx, margin_mm=20.0)
            f, _ = extract_pair_features(metric_gt, metric_candidate, spacing_zyx, pid, "synthetic_training", False)
            row: Dict[str, object] = {
                "patient_id": int(pid),
                "source_split": split,
                "sample_index": int(sample_idx),
                "branch": str(meta.get("branch", "unknown")),
                "requested_branch": str(meta.get("requested_branch", "")),
                "changed_voxel_fraction": float(meta.get("changed_voxel_fraction", float(changed.mean().item()))),
                "input_target_volume_change_pct": float(meta.get("input_target_volume_change_pct", np.nan)),
            }
            for k, v in meta.items():
                if k in row:
                    continue
                if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                    row[f"generator_{k}"] = v
            row.update(f)
            rows.append(row)
    return rows


def support_coverage(
    real_rows: Sequence[Mapping[str, object]],
    bank_rows: Sequence[Mapping[str, object]],
    scaler: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
    percentile: float,
    exclude_clean: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    # v2 uses a bounded, group-balanced mixed distance. Sparse endpoint
    # variables are compared as presence + conditional positive magnitude, so
    # zero-IQR endpoint distributions cannot create arbitrarily large distances.
    return mixed_support_coverage(
        real_rows, bank_rows, scaler, groups, percentile, exclude_clean
    )


def feature_distribution_comparison(
    real_dev: pd.DataFrame,
    bank: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    rows = []
    nonclean = bank[(bank["branch"] != "clean") & (bank["branch"] != "rejected_to_clean")].copy()
    for name in features:
        rv = pd.to_numeric(real_dev[name], errors="coerce").dropna().to_numpy(float)
        sv = pd.to_numeric(nonclean[name], errors="coerce").dropna().to_numpy(float)
        if len(rv) == 0 or len(sv) == 0:
            continue
        qreal = np.percentile(rv, [5, 25, 50, 75, 90, 95])
        qsyn = np.percentile(sv, [5, 25, 50, 75, 90, 95])
        iqr = max(float(np.percentile(rv, 75) - np.percentile(rv, 25)), 1e-6)
        rows.append({
            "feature": name,
            **{f"real_p{q:02d}": v for q, v in zip((5, 25, 50, 75, 90, 95), qreal)},
            **{f"synthetic_p{q:02d}": v for q, v in zip((5, 25, 50, 75, 90, 95), qsyn)},
            "wasserstein": float(stats.wasserstein_distance(rv, sv)),
            "wasserstein_over_real_iqr": float(stats.wasserstein_distance(rv, sv) / iqr),
            "fraction_real_outside_synthetic_p05_p95": float(np.mean((rv < qsyn[0]) | (rv > qsyn[-1]))),
        })
    return pd.DataFrame(rows)


def temporal_shift_summary(dev: pd.DataFrame, test: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for name in features:
        a = pd.to_numeric(dev[name], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(test[name], errors="coerce").dropna().to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            continue
        iqr = max(float(np.percentile(a, 75) - np.percentile(a, 25)), 1e-6)
        p95 = float(np.percentile(a, 95))
        p05 = float(np.percentile(a, 5))
        rows.append({
            "feature": name,
            "development_median": float(np.median(a)),
            "month3_median": float(np.median(b)),
            "development_p05": p05,
            "development_p95": p95,
            "month3_p05": float(np.percentile(b, 5)),
            "month3_p95": float(np.percentile(b, 95)),
            "wasserstein": float(stats.wasserstein_distance(a, b)),
            "wasserstein_over_development_iqr": float(stats.wasserstein_distance(a, b) / iqr),
            "fraction_month3_outside_development_p05_p95": float(np.mean((b < p05) | (b > p95))),
        })
    return pd.DataFrame(rows).sort_values("wasserstein_over_development_iqr", ascending=False)


def family_prevalence(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    counts = best.groupby(["cohort", "best_family"]).size().rename("n").reset_index()
    totals = best.groupby("cohort").size().rename("total").reset_index()
    out = counts.merge(totals, on="cohort", how="left")
    out["fraction"] = out["n"] / out["total"]
    return out.sort_values(["cohort", "fraction"], ascending=[True, False])


def quantile_dict(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"p{q:02d}": np.nan for q in (5, 25, 50, 75, 95)}
    vals = np.percentile(x, [5, 25, 50, 75, 95])
    return {f"p{q:02d}": float(v) for q, v in zip((5, 25, 50, 75, 95), vals)}


def suggested_support(best_dev: pd.DataFrame, real_dev: pd.DataFrame, nsrqa_cfg: Mapping[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not best_dev.empty:
        for family, g in best_dev.groupby("best_family"):
            prevalence = len(g) / max(len(best_dev), 1)
            if family in {"global_inward_offset", "global_outward_offset", "centroid_translation"}:
                values = pd.to_numeric(g.get("parameter_magnitude_mm"), errors="coerce").dropna()
                if len(values):
                    rows.append({"family": family, "parameter": "magnitude_mm", "observed_prevalence": prevalence, **quantile_dict(values)})
            elif family in {"inferior_cut", "superior_cut", "inferior_extension", "superior_extension"}:
                values = pd.to_numeric(g.get("parameter_length_mm"), errors="coerce").dropna()
                if len(values):
                    rows.append({"family": family, "parameter": "length_mm", "observed_prevalence": prevalence, **quantile_dict(values)})
            elif family == "smooth_warp":
                values = pd.to_numeric(g.get("parameter_max_displacement_mm"), errors="coerce").dropna()
                if len(values):
                    rows.append({"family": family, "parameter": "max_displacement_mm", "observed_prevalence": prevalence, **quantile_dict(values)})

    # Local component severity comes directly from the real error geometry rather
    # than the diagnostic ellipsoid scale used for fitting.
    for family, prefix in (("local_bite", "missing"), ("local_blob", "excess")):
        fam_fraction = float(np.mean(best_dev["best_family"] == family)) if len(best_dev) else np.nan
        selected_ids = set(best_dev.loc[best_dev["best_family"] == family, "patient_id"].astype(int))
        selected_real = real_dev[real_dev["patient_id"].astype(int).isin(selected_ids)]
        for param, col in (
            ("extent_si_mm", f"largest_{prefix}_extent_si_mm"),
            ("component_fraction_gt", f"largest_{prefix}_component_fraction_gt"),
        ):
            values = pd.to_numeric(selected_real[col], errors="coerce").dropna()
            values = values[values > 0]
            if len(values):
                rows.append({"family": family, "parameter": param, "observed_prevalence": fam_fraction, **quantile_dict(values)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Add current configured support where a direct mapping exists.
    spec = nsrqa_cfg.get("stochastic_region_dropout", {})
    current = {
        ("global_inward_offset", "magnitude_mm"): spec.get("global_boundary_offset", {}).get("magnitude_mm"),
        ("global_outward_offset", "magnitude_mm"): spec.get("global_boundary_offset", {}).get("magnitude_mm"),
        ("smooth_warp", "max_displacement_mm"): spec.get("smooth_warp", {}).get("max_displacement_mm"),
        ("local_bite", "component_fraction_gt"): spec.get("dropout", {}).get("removed_fraction"),
    }
    out["current_config_support"] = [json.dumps(current.get((r.family, r.parameter), None)) for r in out.itertuples()]
    out["suggested_empirical_support"] = out.apply(
        lambda r: json.dumps([float(r["p05"]), float(r["p95"])]) if np.isfinite(r["p05"]) and np.isfinite(r["p95"]) else "null",
        axis=1,
    )
    return out.sort_values(["family", "parameter"])


def development_severity_quantiles(dev: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for name in features:
        if name not in dev.columns:
            continue
        vals = pd.to_numeric(dev[name], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        rows.append({"feature": name, "n": len(vals), **quantile_dict(vals)})
    return pd.DataFrame(rows)


def make_figures(
    fig_dir: Path,
    real_df: pd.DataFrame,
    best_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    coverage_threshold: float,
    profiles_df: pd.DataFrame,
    bank_df: pd.DataFrame,
) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []

    # Family prevalence by cohort.
    prev = family_prevalence(best_df)
    if not prev.empty:
        pivot = prev.pivot(index="best_family", columns="cohort", values="fraction").fillna(0)
        pivot = pivot.sort_values("development" if "development" in pivot.columns else pivot.columns[0], ascending=True)
        ax = pivot.plot(kind="barh", figsize=(9, max(4.5, 0.42 * len(pivot))))
        ax.set_xlabel("Fraction of cases")
        ax.set_ylabel("Best controlled corruption family")
        ax.set_title("Controlled-family fit prevalence")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        path = fig_dir / "best_corruption_family_counts.png"
        plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
        made.append(path.name)

    # Endpoint error distributions.
    cols = ["inferior_missing_mm", "superior_missing_mm", "inferior_excess_mm", "superior_excess_mm"]
    data = [pd.to_numeric(real_df[real_df.cohort == "development"][c], errors="coerce").dropna().to_numpy() for c in cols]
    if any(len(x) for x in data):
        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=["Inf missing", "Sup missing", "Inf excess", "Sup excess"], showfliers=True)
        plt.ylabel("Endpoint discrepancy (mm)")
        plt.title("Development SpinalCord endpoint errors")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        path = fig_dir / "endpoint_error_distribution.png"
        plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
        made.append(path.name)

    # Coverage distance.
    if not coverage_df.empty:
        plt.figure(figsize=(8, 5))
        for cohort, g in coverage_df.groupby("cohort"):
            vals = pd.to_numeric(g["nearest_pooled_training_distance"], errors="coerce").dropna()
            if len(vals):
                plt.hist(vals, bins=min(20, max(5, int(math.sqrt(len(vals))))), alpha=0.55, label=cohort)
        plt.axvline(coverage_threshold, linestyle="--", linewidth=1.5, label=f"training P95 NN = {coverage_threshold:.2f}")
        plt.xlabel("Nearest current-training-bank distance")
        plt.ylabel("Cases")
        plt.title("Real-error coverage by current NSR-QA corruption distribution")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "training_support_coverage_distance.png"
        plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
        made.append(path.name)

    # Real vs current training bank for core metrics.
    metric_pairs = [
        ("missing_fraction_gt", "Missing fraction"),
        ("excess_fraction_gt", "Excess fraction"),
        ("hd95_mm", "HD95 (mm)"),
        ("assd_mm", "ASSD (mm)"),
    ]
    dev = real_df[real_df.cohort == "development"]
    nonclean = bank_df[(bank_df.branch != "clean") & (bank_df.branch != "rejected_to_clean")]
    for col, label in metric_pairs:
        rv = pd.to_numeric(dev[col], errors="coerce").dropna().to_numpy()
        sv = pd.to_numeric(nonclean[col], errors="coerce").dropna().to_numpy()
        if len(rv) and len(sv):
            plt.figure(figsize=(6.4, 4.8))
            plt.boxplot([rv, sv], labels=["Real dev", "Current synthetic"], showfliers=False)
            plt.ylabel(label)
            plt.title(f"Real vs current synthetic: {label}")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = fig_dir / f"real_vs_training_{col}.png"
            plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
            made.append(path.name)

    # Development longitudinal missing-error heatmap, interpolated to normalized GT position.
    dev_prof = profiles_df[(profiles_df.cohort == "development") & profiles_df.normalized_gt_position.between(0, 1)]
    matrices = []
    labels = []
    grid = np.linspace(0, 1, 100)
    for pid, g in dev_prof.groupby("patient_id"):
        g = g.sort_values("normalized_gt_position")
        x = g["normalized_gt_position"].to_numpy(float)
        y = pd.to_numeric(g["missing_area_fraction_of_gt_slice"], errors="coerce").fillna(0).to_numpy(float)
        if len(x) >= 2:
            matrices.append(np.interp(grid, x, y))
            labels.append(int(pid))
    if matrices:
        M = np.asarray(matrices)
        order = np.argsort(M.mean(axis=1))
        M = M[order]
        labels = [labels[i] for i in order]
        plt.figure(figsize=(10, max(5, min(12, 0.12 * len(labels) + 3))))
        im = plt.imshow(M, aspect="auto", origin="lower", extent=[0, 1, 0, len(labels)], vmin=0, vmax=1)
        plt.colorbar(im, label="Missing fraction of GT slice")
        plt.xlabel("Normalized inferior → superior GT position")
        plt.ylabel("Development cases (sorted by mean missing fraction)")
        plt.title("Longitudinal SpinalCord under-contouring profiles")
        plt.tight_layout()
        path = fig_dir / "longitudinal_error_heatmap.png"
        plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
        made.append(path.name)
    return made


def build_case_review_queue(
    best: pd.DataFrame,
    real_df: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    dev = best[best.cohort == "development"]
    for family, g in dev.groupby("best_family"):
        representative = g.sort_values("best_family_distance").iloc[0]
        rows.append({"patient_id": int(representative.patient_id), "reason": f"best_representative_{family}", "family": family})
        severe = g.merge(real_df[["patient_id", "hd95_mm"]], on="patient_id", how="left").sort_values("hd95_mm", ascending=False).iloc[0]
        rows.append({"patient_id": int(severe.patient_id), "reason": f"most_severe_{family}", "family": family})
    for _, r in dev.sort_values("best_family_distance", ascending=False).head(10).iterrows():
        rows.append({"patient_id": int(r.patient_id), "reason": "worst_controlled_family_fit", "family": r.best_family})
    cdev = coverage[coverage.cohort == "development"].sort_values("nearest_pooled_training_distance", ascending=False).head(10)
    for _, r in cdev.iterrows():
        rows.append({"patient_id": int(r.patient_id), "reason": "largest_current_training_coverage_gap", "family": ""})
    out = pd.DataFrame(rows).drop_duplicates(["patient_id", "reason"]).sort_values(["reason", "patient_id"])
    return out


def write_report(
    out_dir: Path,
    real_df: pd.DataFrame,
    best_df: pd.DataFrame,
    prevalence_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    coverage_threshold: float,
    dist_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    support_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    figures: Sequence[str],
    args,
) -> Path:
    dev = real_df[real_df.cohort == "development"]
    test = real_df[real_df.cohort == "month3"]
    cov_dev = coverage_df[coverage_df.cohort == "development"]
    cov_test = coverage_df[coverage_df.cohort == "month3"]
    lines: List[str] = []
    lines += [
        "# SpinalCord real-error / synthetic-corruption calibration",
        "",
        "## Analysis role",
        "",
        "This is a corruption-design calibration study, not a model-performance analysis. Real error geometry is measured from the full clinical approved/autoseg masks. Month1+Month2 define the development distribution; Month3 is a frozen temporal audit and is not used to change the current experiment.",
        "",
        f"- Development cases analyzed: **{len(dev)}**",
        f"- Month3 cases analyzed: **{len(test)}**",
        f"- Skipped/missing cases: **{int((manifest_df.status != 'ok').sum())}**",
        f"- Current-training support threshold: **{coverage_threshold:.3f}** (P{args.coverage_percentile:g} nearest-neighbor distance among non-clean synthetic training samples)",
        "",
        "## Primary question: does the current NSR-QA training generator cover real errors?",
        "",
        f"- Development covered: **{int(cov_dev.covered_by_current_training_support.sum())}/{len(cov_dev)} ({cov_dev.covered_by_current_training_support.mean():.1%})**" if len(cov_dev) else "- Development coverage unavailable.",
        f"- Month3 covered: **{int(cov_test.covered_by_current_training_support.sum())}/{len(cov_test)} ({cov_test.covered_by_current_training_support.mean():.1%})**" if len(cov_test) else "- Month3 coverage unavailable.",
        "",
        "Coverage uses a bounded, equal-group mixed distance (global size, surface error, longitudinal extent, displacement, and localization). Sparse endpoint variables are compared as presence plus conditional positive magnitude. The P95 support radius is learned from nearest *distinct* non-clean synthetic phenotypes only; it is not tuned on Month3.",
        "",
        "## Controlled corruption-family fits",
        "",
        "The controlled library is diagnostic. Global boundary offsets and endpoint lengths are in physical millimetres; local bite/blob fits use ellipsoids matched to the largest real connected error component; smooth warps are low-frequency diagnostic samples. Best-family assignment is based on phenotype distance. Spatial missing/excess error-mask Dice is retained separately and should be inspected before treating a family label as a mechanistic explanation.",
        "",
    ]
    if not prevalence_df.empty:
        lines.append(markdown_table(prevalence_df[["cohort", "best_family", "n", "fraction"]], 60))
    lines += ["", "## Empirical severity support from development cohort", ""]
    if not support_df.empty:
        lines.append(markdown_table(support_df[["family", "parameter", "observed_prevalence", "p05", "p25", "p50", "p75", "p95", "current_config_support", "suggested_empirical_support"]], 80))
    else:
        lines.append("_No support rows._")
    lines += [
        "",
        "The empirical P05–P95 interval is a suggested *support range*, not a direct recommendation for sampling probability. Rare severe errors may still warrant deliberate oversampling.",
        "",
        "## Real development vs current synthetic distribution",
        "",
    ]
    if not dist_df.empty:
        show = dist_df.sort_values("wasserstein_over_real_iqr", ascending=False).head(20)
        lines.append(markdown_table(show[["feature", "real_p50", "real_p95", "synthetic_p50", "synthetic_p95", "wasserstein_over_real_iqr", "fraction_real_outside_synthetic_p05_p95"]], 20))
    lines += ["", "## Month3 temporal shift audit", ""]
    if not temporal_df.empty:
        lines.append(markdown_table(temporal_df[["feature", "development_median", "month3_median", "wasserstein_over_development_iqr", "fraction_month3_outside_development_p05_p95"]], 30))
    if not overlap_df.empty:
        lines += ["", "## Cross-period IDs", "", markdown_table(overlap_df, 20)]
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Do not tune corruption ranges or family probabilities using Month3 if Month3 remains the temporal test cohort.",
        "- A low controlled-family phenotype distance does not prove the clinical mechanism; review its missing/excess spatial-overlap score and selected cases.",
        "- The local bite/blob diagnostic intentionally uses the observed largest component geometry to test whether a localized family can explain the error; those exact patient-specific masks are never proposed as training samples.",
        "- The current-training bank is generated with the exact prepared NSR-QA corruption function/config on the 1.5-mm development masks. It reproducibly samples that distribution; it does not attempt to reconstruct the historical DataLoader RNG stream from the completed training run. Its nearest-neighbor audit is therefore the primary training-support result.",
        "- The analysis does not use NSR-QA reconstruction outputs or QA scores.",
        "",
        "## Figures",
        "",
    ]
    for name in figures:
        lines += [f"![{name}](figures/{name})", ""]
    lines += [
        "## Key output files",
        "",
        "- `real/case_features.csv`: one row per real approved/autoseg pair.",
        "- `real/longitudinal_slice_profiles.csv`: per-slice area, Dice, missing/excess, and centroid profiles.",
        "- `real/component_features.csv`: connected missing/excess components.",
        "- `fits/controlled_family_fits.csv`: every controlled candidate fit.",
        "- `fits/best_family_per_case.csv`: best/second family and fit margin.",
        "- `training_coverage/synthetic_bank_features.csv`: samples from the exact current NSR-QA generator.",
        "- `training_coverage/real_to_synthetic_nearest_neighbor.csv`: primary real-error coverage result.",
        "- `recommendations/suggested_corruption_support.csv`: development-only empirical severity support.",
        "- `recommendations/case_review_queue.csv`: targeted cases for visual review.",
        "- `temporal_audit/temporal_shift_summary.csv`: Month3 distribution-shift audit.",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dino_shape/spinalcord_corruption_calibration.yaml")
    ap.add_argument("--raw-data-root", default="/home/shaohan/radformation-data/data")
    ap.add_argument("--split-config", default="configs/dino_shape/month_temporal_nsrqa_split.yaml")
    ap.add_argument("--study-root", default="/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study")
    ap.add_argument("--nsrqa-config", default=None, help="Defaults to <study-root>/configs/SpinalCord/nsrqa.yaml")
    ap.add_argument("--output-dir", default=None, help="Defaults to <study-root>/spinalcord_corruption_calibration")
    ap.add_argument("--training-bank-samples-per-case", type=int, default=None)
    ap.add_argument("--skip-controlled-fits", action="store_true")
    ap.add_argument("--skip-training-bank", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_yaml(cfg_path)
    split_path = Path(args.split_config)
    if not split_path.is_absolute():
        split_path = project_root / split_path
    split_cfg = load_yaml(split_path)
    raw_root = Path(args.raw_data_root).expanduser().resolve()
    study_root = Path(args.study_root).expanduser().resolve()
    nsrqa_path = Path(args.nsrqa_config).expanduser().resolve() if args.nsrqa_config else study_root / "configs" / "SpinalCord" / "nsrqa.yaml"
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else study_root / "spinalcord_corruption_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("real", "fits", "training_coverage", "recommendations", "temporal_audit", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    roi = str(cfg.get("roi", "SpinalCord"))
    seed = int(cfg.get("seed", 88))
    endpoint_max = float(cfg["controlled_families"]["endpoint_length_mm"]["max"])
    margin_xy = float(cfg["analysis"].get("margin_xy_mm", 30.0))
    real_margin_z = float(cfg["analysis"].get("margin_z_mm", 30.0))
    fit_margin_z = max(real_margin_z, endpoint_max + 5.0)
    groups = {k: list(v) for k, v in cfg["phenotype_groups"].items()}
    feature_names = [x for names in groups.values() for x in names]

    dev_ids, month3_ids, overlap_rows = temporal_cohorts(split_cfg)
    pd.DataFrame(overlap_rows).to_csv(out_dir / "temporal_audit" / "cross_period_overlap_resolution.csv", index=False)

    dev_feat, dev_comp, dev_prof, dev_manifest = characterize_real_cases(dev_ids, "development", raw_root, roi, margin_xy, real_margin_z)
    test_feat, test_comp, test_prof, test_manifest = characterize_real_cases(month3_ids, "month3", raw_root, roi, margin_xy, real_margin_z)
    all_feat = dev_feat + test_feat
    all_comp = dev_comp + test_comp
    all_prof = dev_prof + test_prof
    manifest = pd.DataFrame(dev_manifest + test_manifest)
    real_df = pd.DataFrame(all_feat)
    comp_df = pd.DataFrame(all_comp)
    prof_df = pd.DataFrame(all_prof)
    if len(dev_feat) < 5:
        raise RuntimeError(f"Only {len(dev_feat)} development cases were analyzable; refusing to calibrate corruption support")
    scaler = robust_feature_scaler(dev_feat, feature_names)
    (out_dir / "analysis_config_resolved.json").write_text(json.dumps({
        "analysis_config": str(cfg_path.resolve()),
        "split_config": str(split_path.resolve()),
        "raw_data_root": str(raw_root),
        "study_root": str(study_root),
        "nsrqa_config": str(nsrqa_path),
        "roi": roi,
        "seed": seed,
        "development_patient_ids": dev_ids,
        "month3_patient_ids": month3_ids,
        "phenotype_scaler": scaler,
    }, indent=2) + "\n", encoding="utf-8")
    manifest.to_csv(out_dir / "cohort_manifest.csv", index=False)
    real_df.to_csv(out_dir / "real" / "case_features.csv", index=False)
    surface_cols = [c for c in [
        "patient_id", "cohort", "assd_mm", "hd95_mm", "surface_max_mm",
        "signed_surface_p05_mm", "signed_surface_p25_mm", "signed_surface_p50_mm",
        "signed_surface_p75_mm", "signed_surface_p95_mm",
        "surface_fraction_below_minus_1p5mm", "surface_fraction_below_minus_3mm",
        "surface_fraction_below_minus_5mm", "surface_fraction_above_plus_1p5mm",
        "surface_fraction_above_plus_3mm", "surface_fraction_above_plus_5mm",
    ] if c in real_df.columns]
    real_df[surface_cols].to_csv(out_dir / "real" / "surface_distance_quantiles.csv", index=False)
    real_df[real_df.cohort == "month3"].to_csv(out_dir / "temporal_audit" / "month3_case_features.csv", index=False)
    comp_df.to_csv(out_dir / "real" / "component_features.csv", index=False)
    prof_df.to_csv(out_dir / "real" / "longitudinal_slice_profiles.csv", index=False)

    # Controlled-family fitting on both cohorts. Month3 uses the already-frozen
    # family definitions and development-derived feature scaler.
    fit_rows: List[Dict[str, object]] = []
    if not args.skip_controlled_fits:
        feature_by_pid = {int(r["patient_id"]): r for r in all_feat}
        for cohort, ids in (("development", dev_ids), ("month3", month3_ids)):
            ok_ids = [pid for pid in ids if pid in feature_by_pid]
            for i, pid in enumerate(ok_ids, start=1):
                print(f"[controlled:{cohort}] {i}/{len(ok_ids)} patient {pid}", flush=True)
                pair = crop_pair_to_union(load_raw_pair(raw_root, pid, roi), margin_xy, fit_margin_z)
                _, fit_components = extract_pair_features(
                    pair.approved, pair.autoseg, pair.spacing_zyx, pid, cohort, True
                )
                fit_rows.extend(controlled_candidates_for_case(
                    pid, cohort, pair, feature_by_pid[pid], fit_components, cfg, scaler, groups, seed
                ))
        fit_df = pd.DataFrame(fit_rows)
        fit_df.to_csv(out_dir / "fits" / "controlled_family_fits.csv", index=False)
        if not fit_df.empty:
            morph = fit_df[fit_df.family.isin(["global_inward_offset", "global_outward_offset"])].copy()
            morph = morph.sort_values("phenotype_distance").groupby(["patient_id", "cohort", "family"], as_index=False).first()
            morph.to_csv(out_dir / "fits" / "morphology_equivalent_fit.csv", index=False)
            endpoint = fit_df[fit_df.family.isin(["inferior_cut", "superior_cut", "inferior_extension", "superior_extension"])].copy()
            endpoint = endpoint.sort_values("phenotype_distance").groupby(["patient_id", "cohort", "family"], as_index=False).first()
            endpoint.to_csv(out_dir / "fits" / "endpoint_fit.csv", index=False)
            fit_df[fit_df.cohort == "month3"].to_csv(out_dir / "temporal_audit" / "month3_family_fits.csv", index=False)
        best_df = summarize_best_families(fit_df, set(dev_ids))
        best_df.to_csv(out_dir / "fits" / "best_family_per_case.csv", index=False)
    else:
        fit_df = pd.DataFrame()
        best_df = pd.DataFrame()

    # Exact current NSR-QA training-generator support audit.
    bank_rows: List[Dict[str, object]] = []
    coverage_df = pd.DataFrame()
    bank_nn_df = pd.DataFrame()
    coverage_threshold = np.nan
    dist_df = pd.DataFrame()
    if not args.skip_training_bank:
        if not nsrqa_path.is_file():
            raise FileNotFoundError(
                f"Prepared NSR-QA config not found: {nsrqa_path}. Run the temporal large-FOV study preparer first."
            )
        nsrqa_cfg = load_yaml(nsrqa_path)
        samples_per_case = int(
            args.training_bank_samples_per_case
            if args.training_bank_samples_per_case is not None
            else cfg["training_bank"].get("samples_per_development_case", 20)
        )
        bank_rows = generate_training_bank(nsrqa_cfg, samples_per_case, seed)
        bank_df = pd.DataFrame(bank_rows)
        bank_df.to_csv(out_dir / "training_coverage" / "synthetic_bank_features.csv", index=False)
        coverage_percentile = float(cfg["training_bank"].get("nearest_neighbor_percentile", 95.0))
        exclude_clean = bool(cfg["training_bank"].get("coverage_reference_excludes_clean", True))
        coverage_df, bank_nn_df, coverage_threshold = support_coverage(
            all_feat, bank_rows, scaler, groups, coverage_percentile, exclude_clean
        )
        coverage_df.to_csv(out_dir / "training_coverage" / "real_to_synthetic_nearest_neighbor.csv", index=False)
        bank_nn_df.to_csv(out_dir / "training_coverage" / "synthetic_bank_nearest_neighbor_reference.csv", index=False)
        dist_df = feature_distribution_comparison(pd.DataFrame(dev_feat), bank_df, feature_names)
        dist_df.to_csv(out_dir / "training_coverage" / "feature_distribution_comparison.csv", index=False)
        branch_summary = bank_df.groupby("branch").size().rename("n").reset_index()
        branch_summary["fraction"] = branch_summary.n / branch_summary.n.sum()
        branch_summary.to_csv(out_dir / "training_coverage" / "synthetic_bank_branch_summary.csv", index=False)
    else:
        nsrqa_cfg = {}
        bank_df = pd.DataFrame()
        coverage_percentile = float(cfg["training_bank"].get("nearest_neighbor_percentile", 95.0))

    prevalence_df = family_prevalence(best_df) if not best_df.empty else pd.DataFrame()
    prevalence_df.to_csv(out_dir / "fits" / "corruption_family_summary.csv", index=False)

    temporal_features = list(cfg.get("report", {}).get("temporal_shift_features", feature_names))
    temporal_df = temporal_shift_summary(
        real_df[real_df.cohort == "development"],
        real_df[real_df.cohort == "month3"],
        temporal_features,
    ) if len(real_df[real_df.cohort == "month3"]) else pd.DataFrame()
    temporal_df.to_csv(out_dir / "temporal_audit" / "temporal_shift_summary.csv", index=False)

    dev_severity_df = development_severity_quantiles(real_df[real_df.cohort == "development"], feature_names)
    dev_severity_df.to_csv(out_dir / "recommendations" / "development_severity_quantiles.csv", index=False)
    if not best_df.empty:
        dev_best = best_df[best_df.cohort == "development"]
        support_df = suggested_support(dev_best, real_df[real_df.cohort == "development"], nsrqa_cfg)
    else:
        support_df = pd.DataFrame()
    support_df.to_csv(out_dir / "recommendations" / "suggested_corruption_support.csv", index=False)

    if not best_df.empty and not coverage_df.empty:
        review_df = build_case_review_queue(best_df, real_df, coverage_df)
    elif not best_df.empty:
        review_df = best_df[best_df.cohort == "development"].sort_values("best_family_distance", ascending=False).head(20)[["patient_id", "best_family"]].rename(columns={"best_family": "family"})
        review_df["reason"] = "worst_controlled_family_fit"
    else:
        review_df = pd.DataFrame(columns=["patient_id", "reason", "family"])
    review_df.to_csv(out_dir / "recommendations" / "case_review_queue.csv", index=False)

    if not coverage_df.empty:
        cov_summary = coverage_df.groupby("cohort").agg(
            n=("patient_id", "size"),
            covered=("covered_by_current_training_support", "sum"),
            median_nearest_distance=("nearest_pooled_training_distance", "median"),
            p95_nearest_distance=("nearest_pooled_training_distance", lambda s: float(np.percentile(s, 95))),
        ).reset_index()
        cov_summary["coverage_fraction"] = cov_summary.covered / cov_summary.n
        cov_summary["coverage_distance_threshold"] = coverage_threshold
    else:
        cov_summary = pd.DataFrame()
    cov_summary.to_csv(out_dir / "training_coverage" / "coverage_summary.csv", index=False)

    figures = make_figures(
        out_dir / "figures", real_df, best_df, coverage_df, coverage_threshold,
        prof_df, bank_df,
    ) if not bank_df.empty else []

    args.coverage_percentile = coverage_percentile
    report = write_report(
        out_dir, real_df, best_df, prevalence_df, coverage_df, coverage_threshold,
        dist_df, temporal_df, support_df, manifest, pd.DataFrame(overlap_rows), figures, args,
    )
    print("\nSpinalCord corruption calibration complete")
    print(f"  output: {out_dir}")
    print(f"  report: {report}")
    print(f"  development real cases: {len(real_df[real_df.cohort == 'development'])}")
    print(f"  month3 real cases: {len(real_df[real_df.cohort == 'month3'])}")
    if not coverage_df.empty:
        for cohort, g in coverage_df.groupby("cohort"):
            print(f"  {cohort} current-training coverage: {g.covered_by_current_training_support.mean():.1%} ({int(g.covered_by_current_training_support.sum())}/{len(g)})")


if __name__ == "__main__":
    main()
