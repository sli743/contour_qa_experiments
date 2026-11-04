#!/usr/bin/env python3
"""Support-distance utilities for SpinalCord corruption calibration.

The primary distance is deliberately bounded and group-balanced. Sparse endpoint
variables use a two-part comparison (presence + positive magnitude) so a zero-IQR
feature cannot dominate the geometry with arbitrarily large standardized values.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

SPARSE_ENDPOINT_FEATURES = (
    "inferior_missing_mm",
    "inferior_excess_mm",
    "superior_missing_mm",
    "superior_excess_mm",
)


def _records(rows: Sequence[Mapping[str, object]] | pd.DataFrame) -> List[Mapping[str, object]]:
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict("records")
    return list(rows)


def _numeric(rows: Sequence[Mapping[str, object]], name: str) -> np.ndarray:
    out = []
    for row in rows:
        try:
            out.append(float(row.get(name, np.nan)))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _continuous_floor(name: str, z_resolution_mm: float) -> float:
    """Conservative minimum scale used only when development spread is tiny."""
    if name in SPARSE_ENDPOINT_FEATURES:
        return max(z_resolution_mm, 1.0)
    lower = name.lower()
    if "fraction" in lower or "relative_volume" in lower or lower in {"dice", "jaccard", "length_ratio"}:
        return 0.05
    if lower.startswith("n_") or lower.endswith("_count"):
        return 1.0
    if lower.endswith("_mm") or "extent_" in lower:
        if "length_difference" in lower or "extent_si" in lower:
            return max(2.0 * z_resolution_mm, 4.0)
        return max(z_resolution_mm, 1.5)
    return 1e-3


def fit_mixed_scaler(
    development_rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    features: Sequence[str],
) -> Dict[str, Dict[str, float | str]]:
    """Fit development-only scales for bounded mixed support distances.

    Endpoint features are zero-inflated by construction. They therefore store a
    presence threshold and a positive-only magnitude scale instead of an IQR
    around a median of zero.
    """
    rows = _records(development_rows)
    if not rows:
        raise ValueError("Need development rows to fit support scaler")
    z = _numeric(rows, "spacing_z_mm")
    z = z[np.isfinite(z) & (z > 0)]
    z_resolution_mm = float(np.median(z)) if len(z) else 1.5
    presence_threshold = max(0.5 * z_resolution_mm, 0.5)

    scaler: Dict[str, Dict[str, float | str]] = {}
    for name in features:
        vals = _numeric(rows, name)
        vals = vals[np.isfinite(vals)]
        if name in SPARSE_ENDPOINT_FEATURES:
            pos = vals[vals > presence_threshold]
            if len(pos) >= 2:
                q25, q75 = np.percentile(pos, [25, 75])
                iqr = float(q75 - q25)
                mad = float(np.median(np.abs(pos - np.median(pos)))) * 1.4826
                positive_scale = max(iqr, mad, z_resolution_mm)
            elif len(pos) == 1:
                # With a single positive development observation, do not infer a
                # narrow severity distribution from one case. Presence carries
                # the main evidence; magnitude is deliberately downweighted.
                positive_scale = max(float(abs(pos[0])), 2.0 * z_resolution_mm)
            else:
                positive_scale = max(2.0 * z_resolution_mm, 3.0)
            scaler[name] = {
                "kind": "sparse_endpoint",
                "presence_threshold": float(presence_threshold),
                "positive_scale": float(max(positive_scale, 1e-6)),
                "n_positive_development": int(len(pos)),
                "z_resolution_mm": float(z_resolution_mm),
            }
            continue

        if len(vals) == 0:
            med, scale = 0.0, _continuous_floor(name, z_resolution_mm)
        else:
            med = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = float(q75 - q25)
            mad = float(np.median(np.abs(vals - med))) * 1.4826
            scale = max(iqr, mad, _continuous_floor(name, z_resolution_mm))
        scaler[name] = {
            "kind": "continuous",
            "median": med,
            "scale": float(max(scale, 1e-6)),
            "z_resolution_mm": float(z_resolution_mm),
        }
    return scaler


def feature_pair_distance(a: float, b: float, spec: Mapping[str, object]) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    if str(spec.get("kind", "continuous")) == "sparse_endpoint":
        threshold = float(spec["presence_threshold"])
        a_present = a > threshold
        b_present = b > threshold
        if a_present != b_present:
            return 1.0
        if not a_present:
            return 0.0
        scale = max(float(spec["positive_scale"]), 1e-6)
        return float(min(abs(a - b) / scale, 1.0))
    scale = max(float(spec.get("scale", 1.0)), 1e-6)
    return float(min(abs(a - b) / scale, 1.0))


def mixed_grouped_distance(
    a: Mapping[str, object],
    b: Mapping[str, object],
    scaler: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
) -> Tuple[float, Dict[str, float]]:
    group_dist: Dict[str, float] = {}
    for group, names in groups.items():
        ds = []
        for name in names:
            try:
                av = float(a.get(name, np.nan))
                bv = float(b.get(name, np.nan))
            except Exception:
                continue
            if name not in scaler:
                continue
            d = feature_pair_distance(av, bv, scaler[name])
            if np.isfinite(d):
                ds.append(d)
        if ds:
            group_dist[group] = float(np.sqrt(np.mean(np.square(ds))))
    overall = float(np.mean(list(group_dist.values()))) if group_dist else np.nan
    return overall, group_dist


def _pairwise_group_matrix(
    a_rows: Sequence[Mapping[str, object]],
    b_rows: Sequence[Mapping[str, object]],
    scaler: Mapping[str, Mapping[str, object]],
    names: Sequence[str],
) -> np.ndarray:
    na, nb = len(a_rows), len(b_rows)
    sumsq = np.zeros((na, nb), dtype=np.float32)
    counts = np.zeros((na, nb), dtype=np.float32)
    for name in names:
        if name not in scaler:
            continue
        av = _numeric(a_rows, name)[:, None]
        bv = _numeric(b_rows, name)[None, :]
        valid = np.isfinite(av) & np.isfinite(bv)
        spec = scaler[name]
        if str(spec.get("kind", "continuous")) == "sparse_endpoint":
            threshold = float(spec["presence_threshold"])
            pa, pb = av > threshold, bv > threshold
            mismatch = np.logical_xor(pa, pb)
            both = pa & pb
            d = np.zeros((na, nb), dtype=np.float32)
            d[mismatch] = 1.0
            if np.any(both):
                scale = max(float(spec["positive_scale"]), 1e-6)
                raw = np.minimum(np.abs(av - bv) / scale, 1.0)
                d[both] = raw[both].astype(np.float32)
        else:
            scale = max(float(spec.get("scale", 1.0)), 1e-6)
            d = np.minimum(np.abs(av - bv) / scale, 1.0).astype(np.float32)
        d[~valid] = 0.0
        sumsq += d * d
        counts += valid.astype(np.float32)
    out = np.full((na, nb), np.nan, dtype=np.float32)
    ok = counts > 0
    out[ok] = np.sqrt(sumsq[ok] / counts[ok])
    return out


def mixed_distance_matrices(
    a_rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    b_rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    scaler: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    a = _records(a_rows)
    b = _records(b_rows)
    mats: Dict[str, np.ndarray] = {}
    for group, names in groups.items():
        mat = _pairwise_group_matrix(a, b, scaler, names)
        if np.isfinite(mat).any():
            mats[group] = mat
    if not mats:
        return np.full((len(a), len(b)), np.nan, dtype=np.float32), mats
    stack = np.stack([m for m in mats.values()], axis=0)
    overall = np.nanmean(stack, axis=0)
    return overall.astype(np.float32), mats


def _nonclean_reference(
    bank_rows: Sequence[Mapping[str, object]], exclude_clean: bool
) -> List[Mapping[str, object]]:
    if not exclude_clean:
        return list(bank_rows)
    return [
        r for r in bank_rows
        if str(r.get("branch", "")) not in {"clean", "rejected_to_clean"}
        and float(r.get("changed_fraction_gt", 0.0) or 0.0) > 0
    ]




def _nearest_distinct_distance_matrix(d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Nearest positive distance for each row, ignoring exact duplicate phenotypes.

    Exact duplicate synthetic phenotypes are common (especially for endpoint-only
    representations). Using a zero-distance duplicate as the support radius would
    collapse the P95 threshold to zero even when the sampled support contains
    several distinct severities.
    """
    x = np.asarray(d, dtype=float).copy()
    x[~np.isfinite(x)] = np.inf
    x[x <= eps] = np.inf
    out = np.min(x, axis=1)
    # If a row has no distinct neighbor, fall back to zero rather than inf.
    out[~np.isfinite(out)] = 0.0
    return out

def mixed_support_coverage(
    real_rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    bank_rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    scaler: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
    percentile: float = 95.0,
    exclude_clean: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Calculate synthetic-support coverage with bounded mixed distances."""
    real = _records(real_rows)
    bank = _records(bank_rows)
    ref = _nonclean_reference(bank, exclude_clean)
    if len(ref) < 2:
        raise RuntimeError("Need at least two non-clean synthetic samples")

    dbb, _ = mixed_distance_matrices(ref, ref, scaler, groups)
    np.fill_diagonal(dbb, np.inf)
    bank_nn = _nearest_distinct_distance_matrix(dbb)
    threshold = float(np.nanpercentile(bank_nn, percentile))
    bank_df = pd.DataFrame({
        "patient_id": [int(r["patient_id"]) for r in ref],
        "branch": [str(r.get("branch", "")) for r in ref],
        "nearest_other_training_distance": bank_nn,
        "coverage_reference_percentile": float(percentile),
        "coverage_distance_threshold": threshold,
        "distance_method": "mixed_bounded_v2",
    })

    drb, group_mats = mixed_distance_matrices(real, ref, scaler, groups)
    ref_pids = np.asarray([int(r["patient_id"]) for r in ref], dtype=int)
    rows: List[Dict[str, object]] = []
    for i, r in enumerate(real):
        row_dist = drb[i]
        j = int(np.nanargmin(row_dist))
        pid = int(r["patient_id"])
        same = np.where(ref_pids == pid)[0]
        if len(same):
            local = same[int(np.nanargmin(row_dist[same]))]
            same_dist = float(row_dist[local])
            same_branch = str(ref[local].get("branch", ""))
        else:
            same_dist, same_branch = np.nan, ""
        out: Dict[str, object] = {
            "patient_id": pid,
            "cohort": str(r.get("cohort", "")),
            "nearest_pooled_training_distance": float(row_dist[j]),
            "nearest_training_patient_id": int(ref[j]["patient_id"]),
            "nearest_training_branch": str(ref[j].get("branch", "")),
            "nearest_same_patient_training_distance": same_dist,
            "nearest_same_patient_training_branch": same_branch,
            "coverage_distance_threshold": threshold,
            "covered_by_current_training_support": bool(float(row_dist[j]) <= threshold),
            "distance_method": "mixed_bounded_v2",
        }
        group_values = {}
        for group, mat in group_mats.items():
            value = float(mat[i, j])
            out[f"nearest_{group}_distance"] = value
            group_values[group] = value
        if group_values:
            out["dominant_gap_group"] = max(group_values, key=group_values.get)
            out["dominant_gap_distance"] = group_values[out["dominant_gap_group"]]
        rows.append(out)
    return pd.DataFrame(rows), bank_df, threshold


def group_support_summary(
    real_rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    bank_rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    scaler: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
    percentile: float = 95.0,
    exclude_clean: bool = True,
) -> pd.DataFrame:
    """Per-group support thresholds and real coverage rates."""
    real = _records(real_rows)
    bank = _records(bank_rows)
    ref = _nonclean_reference(bank, exclude_clean)
    out = []
    for group, names in groups.items():
        one = {group: list(names)}
        dbb, _ = mixed_distance_matrices(ref, ref, scaler, one)
        np.fill_diagonal(dbb, np.inf)
        bank_nn = _nearest_distinct_distance_matrix(dbb)
        thr = float(np.nanpercentile(bank_nn, percentile))
        drb, _ = mixed_distance_matrices(real, ref, scaler, one)
        real_nn = np.nanmin(drb, axis=1)
        for cohort in sorted({str(r.get("cohort", "")) for r in real}):
            idx = np.asarray([str(r.get("cohort", "")) == cohort for r in real], dtype=bool)
            vals = real_nn[idx]
            out.append({
                "cohort": cohort,
                "group": group,
                "n": int(idx.sum()),
                "synthetic_p95_nn_threshold": thr,
                "median_real_nn_distance": float(np.nanmedian(vals)),
                "p90_real_nn_distance": float(np.nanpercentile(vals, 90)),
                "covered": int(np.sum(vals <= thr)),
                "coverage_fraction": float(np.mean(vals <= thr)),
            })
    return pd.DataFrame(out)
