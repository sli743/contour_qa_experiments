#!/usr/bin/env python3
"""Domain analysis for thoracic auto-contour Dice and HD95.

The complete metrics table is the analysis universe. Domain assignments are
left-joined at patient-ROI level. Only month2/month3 patients are domain-review
eligible; unassigned pairs within that reviewed cohort are labeled ``Normal``.
Month1 patients and patients outside all three declared month periods are labeled
``Unknown``.

Outputs
-------
- analysis_dataset.csv
- domain_assignment_conflicts.csv
- domain_counts.csv
- descriptive_statistics.csv
- kruskal_wallis.csv
- domain_vs_normal.csv
- abnormal_domain_pairwise.csv
- pooled_severity_domain_vs_normal.csv
- figures/*.png
- report.md

Expected metrics input (preferred long format)
----------------------------------------------
patient_id, site, roi, metric, value, ...

``metric`` may use ``dice`` and any of ``haus``, ``hd95``,
``hausdorff95``, or ``hausdorff_95``. A limited wide format containing
``dice`` and ``haus``/``hd95`` columns is also accepted.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


MONTH1_PATIENT_IDS = {
    "14", "87", "151", "44", "74", "166", "43", "59", "10", "128",
    "122", "18", "23", "165", "89", "16", "37", "53", "167", "61",
    "51", "125", "82", "55", "52", "45", "27", "94", "134", "127",
    "71", "143", "17", "162", "79", "102", "120", "31", "54", "142",
}
MONTH2_PATIENT_IDS = {
    "92", "80", "38", "36", "41", "69", "88", "28", "2", "70", "58",
    "13", "85", "11", "159", "124", "104", "7", "108", "81", "57",
    "149", "56", "21", "46", "86", "65", "4", "84", "75", "66", "107",
    "141", "72", "32", "150", "60", "24", "161", "3", "111", "29",
    "130", "137",
}
MONTH3_PATIENT_IDS = {
    "95", "148", "154", "76", "77", "164", "109", "48", "30", "101",
    "135", "136", "39", "152", "115", "20", "90", "83", "145", "168",
    "99", "16", "106", "15", "113", "129", "114", "139", "33", "49",
    "42", "116", "47", "112", "158", "40", "118", "12", "63", "133",
    "1", "34", "78", "19", "93", "156", "110", "68", "144", "157",
    "62", "73", "131", "155", "26", "163", "9", "97", "91", "105",
    "153", "103", "35", "123", "146", "100", "117", "138", "160", "150",
}
ALL_MONTH_PATIENT_IDS = MONTH1_PATIENT_IDS | MONTH2_PATIENT_IDS | MONTH3_PATIENT_IDS
DOMAIN_REVIEWED_PATIENT_IDS = (MONTH2_PATIENT_IDS | MONTH3_PATIENT_IDS) - MONTH1_PATIENT_IDS
UNKNOWN_DOMAIN = "Unknown"
DOMAIN_ORDER = ["Normal", "Unknown", "I", "AV", "MI", "AC", "H", "CA", "NDI", "LDI"]
POOR_AGREEMENT_DOMAINS = {"I", "AV", "MI", "AC", "H"}
BETTER_AGREEMENT_DOMAINS = {"CA", "NDI", "LDI"}

METRIC_ALIASES = {
    "dice": "dice",
    "dsc": "dice",
    "dice_score": "dice",
    "haus": "hd95",
    "hd95": "hd95",
    "haus95": "hd95",
    "hausdorff": "hd95",
    "hausdorff95": "hd95",
    "hausdorff_95": "hd95",
    "hausdorff_95_mm": "hd95",
}

METRIC_LABELS = {
    "dice": "Dice similarity coefficient",
    "hd95": "HD95 (mm)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze thorax Dice/HD95 by OOD/error domain. Unassigned patient-ROI "
            "pairs are labeled Normal only within month2/month3; month1 and out-of-cohort patients are Unknown."
        )
    )
    p.add_argument("--metrics_csv", required=True, help="Complete thorax metrics CSV.")
    p.add_argument("--domain_csv", required=True, help="OOD domain-assignment CSV.")
    p.add_argument("--output_dir", required=True, help="Directory for tables, figures, and report.md.")
    p.add_argument("--site", default="thorax", help="Site to analyze (default: thorax).")
    p.add_argument(
        "--min_group_n",
        type=int,
        default=5,
        help="Minimum n per group for inferential comparisons (default: 5).",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Patient-cluster bootstrap replicates for median-difference CIs (default: 2000).",
    )
    p.add_argument("--seed", type=int, default=88)
    p.add_argument(
        "--conflict_policy",
        choices=["exclude", "error", "first"],
        default="exclude",
        help=(
            "How to handle patient-ROI pairs assigned >1 domain. 'exclude' writes them "
            "to a conflict table and omits them from primary analyses (default)."
        ),
    )
    p.add_argument(
        "--duplicate_policy",
        choices=["error", "median", "first"],
        default="error",
        help="How to handle duplicate contour-inference-metric rows after using patient+study-uid+ROI when available.",
    )
    p.add_argument(
        "--keep_gc_label",
        action="store_true",
        help="Keep source label GC instead of displaying it as CA. Raw source label is always retained.",
    )
    p.add_argument(
        "--no_log_hd95",
        action="store_true",
        help="Do not generate the optional log-scale HD95 distribution figure.",
    )
    return p.parse_args()


def normalize_patient_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def is_month1_patient(x) -> bool:
    return normalize_patient_id(x) in MONTH1_PATIENT_IDS


def is_outside_three_month_periods(x) -> bool:
    return normalize_patient_id(x) not in ALL_MONTH_PATIENT_IDS


def is_domain_reviewed_patient(x) -> bool:
    return normalize_patient_id(x) in DOMAIN_REVIEWED_PATIENT_IDS


def apply_domain_review_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Force non-reviewed patients to Unknown.

    Only month2/month3 patients are domain-review eligible. Month1 takes
    precedence for the duplicated patient 16, and patients outside all three
    declared month periods are also treated as Unknown.
    """
    out = df.copy()
    if "patient_id" not in out.columns:
        return out
    month1_mask = out["patient_id"].map(is_month1_patient)
    outside_mask = out["patient_id"].map(is_outside_three_month_periods)
    unknown_mask = ~out["patient_id"].map(is_domain_reviewed_patient)
    if "domain" in out.columns:
        out.loc[unknown_mask, "domain"] = UNKNOWN_DOMAIN
    if "domain_conflict" in out.columns:
        out.loc[unknown_mask, "domain_conflict"] = False
    out["month1_domain_unknown"] = month1_mask.astype(bool)
    out["outside_three_month_periods"] = outside_mask.astype(bool)
    out["domain_review_unknown"] = unknown_mask.astype(bool)
    return out


def canonical_metric(x: str) -> Optional[str]:
    key = str(x).strip().lower().replace(" ", "_").replace("-", "_")
    return METRIC_ALIASES.get(key)


def canonical_domain(x, keep_gc_label: bool = False) -> str:
    if pd.isna(x):
        return ""
    d = str(x).strip().upper()
    if d == "GC" and not keep_gc_label:
        return "CA"
    return d


def case_key_columns(df: pd.DataFrame, include_domain: bool = False) -> List[str]:
    """Return the contour-inference identity columns available in ``df``.

    ``patient_id`` alone is not guaranteed to identify one thorax inference.  The
    Radformation metrics/domain tables include ``study-uid`` for patients with
    multiple studies, so use it whenever available.  ``roi`` always completes
    the contour-level key.
    """
    cols = ["patient_id"]
    if "study-uid" in df.columns:
        cols.append("study-uid")
    cols.append("roi")
    if include_domain and "domain" in df.columns:
        cols.append("domain")
    return cols


def shared_case_key_columns(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    """Case key safe for joining two tables.

    Use ``study-uid`` only when both tables contain it; otherwise fall back to
    patient+ROI for backwards compatibility with older exports.
    """
    cols = ["patient_id"]
    if "study-uid" in left.columns and "study-uid" in right.columns:
        cols.append("study-uid")
    cols.append("roi")
    return cols


def load_metrics(path: str, site: str, duplicate_policy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required_base = {"patient_id", "roi"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV missing required columns: {sorted(missing)}")

    if "site" in df.columns:
        df = df[df["site"].astype(str).str.lower() == site.lower()].copy()

    # Preferred long format.
    if {"metric", "value"}.issubset(df.columns):
        out = df.copy()
        out["metric"] = out["metric"].map(canonical_metric)
        out = out[out["metric"].isin(["dice", "hd95"])].copy()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
    else:
        # Limited wide-format support.
        lower_to_original = {c.lower(): c for c in df.columns}
        dice_col = lower_to_original.get("dice") or lower_to_original.get("dsc")
        hd_col = None
        for candidate in ["hd95", "haus", "hausdorff95", "hausdorff_95"]:
            if candidate in lower_to_original:
                hd_col = lower_to_original[candidate]
                break
        if dice_col is None or hd_col is None:
            raise ValueError(
                "Metrics CSV must be long format (metric,value) or wide format containing Dice and HD95/Haus columns."
            )
        id_cols = [c for c in df.columns if c not in {dice_col, hd_col}]
        out = df[id_cols + [dice_col, hd_col]].melt(
            id_vars=id_cols,
            value_vars=[dice_col, hd_col],
            var_name="metric",
            value_name="value",
        )
        out["metric"] = out["metric"].map(canonical_metric)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out["patient_id"] = out["patient_id"].map(normalize_patient_id)
    out["roi"] = out["roi"].astype(str).str.strip()
    out = out[out["patient_id"].ne("") & out["roi"].ne("") & out["value"].notna()].copy()

    key = case_key_columns(out) + ["metric"]
    dup = out[out.duplicated(key, keep=False)].sort_values(key).copy()
    if not dup.empty:
        if duplicate_policy == "error":
            raise ValueError(
                f"Found {len(dup)} duplicate patient-ROI-metric rows. "
                "Use --duplicate_policy median/first after reviewing duplicates."
            )
        if duplicate_policy == "median":
            # Preserve a compact, reproducible analysis universe.
            keep_cols = [c for c in out.columns if c not in {"value"}]
            first = out.sort_values(key).groupby(key, as_index=False)[keep_cols].first()
            med = out.groupby(key, as_index=False)["value"].median()
            out = first.drop(columns=key, errors="ignore").merge(med, on=key, how="right")
        elif duplicate_policy == "first":
            out = out.sort_values(key).drop_duplicates(key, keep="first")

    return out.reset_index(drop=True), dup.reset_index(drop=True)


def load_pair_domains(
    path: str,
    site: str,
    keep_gc_label: bool,
    conflict_policy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(path)
    required = {"patient_id", "roi", "domain"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Domain CSV missing required columns: {sorted(missing)}")

    if "site" in d.columns:
        d = d[d["site"].astype(str).str.lower() == site.lower()].copy()

    d["patient_id"] = d["patient_id"].map(normalize_patient_id)
    d["roi"] = d["roi"].astype(str).str.strip()
    d["domain_raw"] = d["domain"].astype(str).str.strip().str.upper()
    d["domain"] = d["domain"].map(lambda x: canonical_domain(x, keep_gc_label=keep_gc_label))
    d["month1_domain_unknown"] = d["patient_id"].map(is_month1_patient)
    d["outside_three_month_periods"] = d["patient_id"].map(is_outside_three_month_periods)
    d["domain_review_unknown"] = ~d["patient_id"].map(is_domain_reviewed_patient)
    d.loc[d["domain_review_unknown"], "domain"] = UNKNOWN_DOMAIN
    d = d[d["patient_id"].ne("") & d["roi"].ne("") & d["domain"].ne("")].copy()

    keys = case_key_columns(d)
    rows = []
    conflicts = []
    for key_values, g in d.groupby(keys, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key_record = dict(zip(keys, key_values))
        domains = sorted(set(g["domain"].dropna().astype(str)))
        raw_domains = sorted(set(g["domain_raw"].dropna().astype(str)))
        if len(domains) == 1:
            rows.append({
                **key_record,
                "domain": domains[0],
                "domain_raw": ";".join(raw_domains),
                "domain_source_rows": len(g),
            })
        else:
            record = {
                **key_record,
                "domains": ";".join(domains),
                "raw_domains": ";".join(raw_domains),
                "n_source_rows": len(g),
            }
            conflicts.append(record)
            if conflict_policy == "error":
                pretty_key = ", ".join(f"{k}={key_record[k]}" for k in keys)
                raise ValueError(f"Conflicting domain assignment for {pretty_key}: {domains}")
            if conflict_policy == "first":
                first = g.iloc[0]
                rows.append({
                    **key_record,
                    "domain": first["domain"],
                    "domain_raw": first["domain_raw"],
                    "domain_source_rows": len(g),
                })
            # exclude => no pair-domain row; merged analysis marks it Conflict separately.

    pair_domains = pd.DataFrame(rows)
    conflict_df = pd.DataFrame(conflicts)
    source = d.copy()
    return pair_domains, conflict_df, source


def build_analysis_dataset(
    metrics: pd.DataFrame,
    pair_domains: pd.DataFrame,
    conflicts: pd.DataFrame,
    conflict_policy: str,
) -> pd.DataFrame:
    join_cols = shared_case_key_columns(metrics, pair_domains) if not pair_domains.empty else case_key_columns(metrics)
    pair_cols = join_cols + ["domain", "domain_raw", "domain_source_rows"]
    if pair_domains.empty:
        merged = metrics.copy()
        merged["domain"] = "Normal"
        merged["domain_raw"] = ""
        merged["domain_source_rows"] = 0
    else:
        merged = metrics.merge(pair_domains[pair_cols], on=join_cols, how="left")
        merged["domain"] = merged["domain"].fillna("Normal")
        merged["domain_raw"] = merged["domain_raw"].fillna("")
        merged["domain_source_rows"] = merged["domain_source_rows"].fillna(0).astype(int)

    merged["domain_conflict"] = False
    if not conflicts.empty and conflict_policy == "exclude":
        conflict_key_cols = shared_case_key_columns(merged, conflicts)
        conflict_keys = set(
            map(tuple, conflicts[conflict_key_cols].astype(str).itertuples(index=False, name=None))
        )
        merged_keys = merged[conflict_key_cols].astype(str)
        mask = [tuple(row) in conflict_keys for row in merged_keys.itertuples(index=False, name=None)]
        merged.loc[mask, "domain"] = "Conflict"
        merged.loc[mask, "domain_conflict"] = True

    # Unrecorded non-reviewed pairs would otherwise be labeled Normal. Override
    # month1 and all patients outside the three declared month periods after the
    # join, and clear conflict status for these non-reviewed patients.
    return apply_domain_review_scope(merged)


def add_normalized_severity(df: pd.DataFrame, min_normal_n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add ROI-specific robust severity score where larger = worse.

    Baseline is Normal within each ROI/metric. Scale priority:
      scaled MAD -> IQR/1.349 -> SD.
    """
    out = df.copy()
    out["severity_z"] = np.nan
    baseline_rows = []

    for (roi, metric), idx in out.groupby(["roi", "metric"]).groups.items():
        sub = out.loc[idx]
        normal = sub[(sub["domain"] == "Normal") & sub["value"].notna()]
        if len(normal) < min_normal_n:
            baseline_rows.append({
                "roi": roi, "metric": metric, "normal_n": len(normal),
                "center": np.nan, "scale": np.nan, "scale_method": "insufficient_normal_n",
            })
            continue

        vals = normal["value"].to_numpy(float)
        center = float(np.median(vals))
        mad = float(np.median(np.abs(vals - center)))
        scale = 1.4826 * mad
        method = "1.4826*MAD"

        if not np.isfinite(scale) or scale <= 0:
            q75, q25 = np.percentile(vals, [75, 25])
            scale = float((q75 - q25) / 1.349)
            method = "IQR/1.349"
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            method = "SD"
        if not np.isfinite(scale) or scale <= 0:
            baseline_rows.append({
                "roi": roi, "metric": metric, "normal_n": len(normal),
                "center": center, "scale": np.nan, "scale_method": "zero_scale",
            })
            continue

        x = out.loc[idx, "value"].astype(float)
        if metric == "dice":
            # Lower Dice = worse.
            z = (center - x) / scale
        else:
            # Larger HD95 = worse.
            z = (x - center) / scale
        out.loc[idx, "severity_z"] = z
        baseline_rows.append({
            "roi": roi, "metric": metric, "normal_n": len(normal),
            "center": center, "scale": scale, "scale_method": method,
        })

    return out, pd.DataFrame(baseline_rows)


def domain_order_present(values: Iterable[str], include_conflict: bool = False) -> List[str]:
    present = set(map(str, values))
    order = [d for d in DOMAIN_ORDER if d in present]
    extras = sorted(present - set(DOMAIN_ORDER) - {"Conflict"})
    order.extend(extras)
    if include_conflict and "Conflict" in present:
        order.append("Conflict")
    return order


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta, positive when values in x tend to be larger than y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # Dataset is small enough for direct broadcasting, but chunk to avoid surprises.
    gt = 0
    lt = 0
    chunk = 2048
    for i in range(0, len(x), chunk):
        comp = x[i:i + chunk, None] - y[None, :]
        gt += int((comp > 0).sum())
        lt += int((comp < 0).sum())
    return (gt - lt) / (len(x) * len(y))


def bh_fdr(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(m)
    tmp[order] = q
    out[np.where(valid)[0]] = tmp
    return out


def clustered_median_diff_ci(
    a: pd.DataFrame,
    b: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Median(a)-median(b), with patient-cluster bootstrap CI."""
    est = float(a["value"].median() - b["value"].median())
    if n_boot <= 0 or a.empty or b.empty:
        return est, np.nan, np.nan

    rng = np.random.default_rng(seed)
    a_groups = {k: g["value"].to_numpy(float) for k, g in a.groupby("patient_id")}
    b_groups = {k: g["value"].to_numpy(float) for k, g in b.groupby("patient_id")}
    a_ids = np.array(list(a_groups), dtype=object)
    b_ids = np.array(list(b_groups), dtype=object)
    if len(a_ids) == 0 or len(b_ids) == 0:
        return est, np.nan, np.nan

    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a_ids, size=len(a_ids), replace=True)
        sb = rng.choice(b_ids, size=len(b_ids), replace=True)
        va = np.concatenate([a_groups[k] for k in sa])
        vb = np.concatenate([b_groups[k] for k in sb])
        boot[i] = np.median(va) - np.median(vb)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, float(lo), float(hi)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = df[~df["domain_conflict"]].copy()
    for (roi, metric, domain), g in primary.groupby(["roi", "metric", "domain"], sort=True):
        vals = g["value"].dropna().astype(float)
        if vals.empty:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        rows.append({
            "roi": roi,
            "metric": metric,
            "domain": domain,
            "n": len(vals),
            "n_patients": g.loc[vals.index, "patient_id"].nunique(),
            "mean": vals.mean(),
            "std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "median": med,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": vals.min(),
            "max": vals.max(),
        })
    return pd.DataFrame(rows)


def run_inference(
    df: pd.DataFrame,
    min_group_n: int,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = df[(~df["domain_conflict"]) & (df["domain"] != UNKNOWN_DOMAIN)].copy()
    kw_rows = []
    normal_rows = []
    abnormal_rows = []

    for (roi, metric), sub in primary.groupby(["roi", "metric"], sort=True):
        counts = sub.groupby("domain")["value"].count()
        eligible = [d for d, n in counts.items() if n >= min_group_n]
        arrays = [sub.loc[sub["domain"] == d, "value"].dropna().to_numpy(float) for d in eligible]
        if len(arrays) >= 2:
            stat, p = kruskal(*arrays)
            kw_rows.append({
                "roi": roi,
                "metric": metric,
                "groups": ";".join(eligible),
                "n_groups": len(eligible),
                "H": stat,
                "p": p,
            })
        else:
            kw_rows.append({
                "roi": roi,
                "metric": metric,
                "groups": ";".join(eligible),
                "n_groups": len(eligible),
                "H": np.nan,
                "p": np.nan,
            })

        normal = sub[sub["domain"] == "Normal"].dropna(subset=["value"])
        for domain in [d for d in domain_order_present(sub["domain"]) if d != "Normal"]:
            g = sub[sub["domain"] == domain].dropna(subset=["value"])
            row = {
                "roi": roi,
                "metric": metric,
                "domain": domain,
                "reference": "Normal",
                "n_domain": len(g),
                "n_normal": len(normal),
            }
            if len(g) >= min_group_n and len(normal) >= min_group_n:
                u, p = mannwhitneyu(g["value"], normal["value"], alternative="two-sided")
                est, lo, hi = clustered_median_diff_ci(g, normal, n_boot, seed + len(normal_rows))
                row.update({
                    "U": u,
                    "p": p,
                    "median_difference_domain_minus_normal": est,
                    "median_difference_ci95_low": lo,
                    "median_difference_ci95_high": hi,
                    "cliffs_delta_domain_vs_normal": cliffs_delta(g["value"], normal["value"]),
                })
            else:
                row.update({
                    "U": np.nan,
                    "p": np.nan,
                    "median_difference_domain_minus_normal": np.nan,
                    "median_difference_ci95_low": np.nan,
                    "median_difference_ci95_high": np.nan,
                    "cliffs_delta_domain_vs_normal": np.nan,
                })
            normal_rows.append(row)

        abnormal_domains = [
            d for d in domain_order_present(sub["domain"])
            if d != "Normal" and counts.get(d, 0) >= min_group_n
        ]
        for i, d1 in enumerate(abnormal_domains):
            for d2 in abnormal_domains[i + 1:]:
                g1 = sub[sub["domain"] == d1].dropna(subset=["value"])
                g2 = sub[sub["domain"] == d2].dropna(subset=["value"])
                u, p = mannwhitneyu(g1["value"], g2["value"], alternative="two-sided")
                est, lo, hi = clustered_median_diff_ci(g1, g2, n_boot, seed + len(abnormal_rows) + 10000)
                abnormal_rows.append({
                    "roi": roi,
                    "metric": metric,
                    "domain_a": d1,
                    "domain_b": d2,
                    "n_a": len(g1),
                    "n_b": len(g2),
                    "U": u,
                    "p": p,
                    "median_difference_a_minus_b": est,
                    "median_difference_ci95_low": lo,
                    "median_difference_ci95_high": hi,
                    "cliffs_delta_a_vs_b": cliffs_delta(g1["value"], g2["value"]),
                })

    kw = pd.DataFrame(kw_rows)
    nv = pd.DataFrame(normal_rows)
    ab = pd.DataFrame(abnormal_rows)
    for table in [kw, nv, ab]:
        if not table.empty and "p" in table.columns:
            table["q_bh"] = bh_fdr(table["p"].to_numpy())
    return kw, nv, ab


def run_pooled_severity_inference(
    df: pd.DataFrame,
    min_group_n: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    primary = df[(~df["domain_conflict"]) & (df["domain"] != UNKNOWN_DOMAIN) & df["severity_z"].notna()].copy()
    for metric, sub in primary.groupby("metric", sort=True):
        normal = sub[sub["domain"] == "Normal"].copy()
        normal["raw_value"] = normal["value"]
        normal["value"] = normal["severity_z"]
        for domain in [d for d in domain_order_present(sub["domain"]) if d != "Normal"]:
            g = sub[sub["domain"] == domain].copy()
            g["raw_value"] = g["value"]
            g["value"] = g["severity_z"]
            row = {
                "metric": metric,
                "domain": domain,
                "reference": "Normal",
                "n_domain_rows": len(g),
                "n_domain_patients": g["patient_id"].nunique(),
                "n_normal_rows": len(normal),
                "n_normal_patients": normal["patient_id"].nunique(),
            }
            if len(g) >= min_group_n and len(normal) >= min_group_n:
                u, p = mannwhitneyu(g["value"], normal["value"], alternative="two-sided")
                est, lo, hi = clustered_median_diff_ci(g, normal, n_boot, seed + len(rows) + 20000)
                row.update({
                    "U": u,
                    "p": p,
                    "median_severity_difference": est,
                    "median_difference_ci95_low": lo,
                    "median_difference_ci95_high": hi,
                    "cliffs_delta": cliffs_delta(g["value"], normal["value"]),
                })
            else:
                row.update({
                    "U": np.nan, "p": np.nan,
                    "median_severity_difference": np.nan,
                    "median_difference_ci95_low": np.nan,
                    "median_difference_ci95_high": np.nan,
                    "cliffs_delta": np.nan,
                })
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh"] = bh_fdr(out["p"].to_numpy())
    return out


def save_domain_count_heatmap(df: pd.DataFrame, outpath: Path) -> None:
    primary0 = df[~df["domain_conflict"]].copy()
    primary = primary0.drop_duplicates(case_key_columns(primary0, include_domain=True))
    ct = pd.crosstab(primary["roi"], primary["domain"])
    order = domain_order_present(ct.columns)
    ct = ct.reindex(columns=order, fill_value=0)
    if ct.empty:
        return

    fig_w = max(8, 0.9 * len(ct.columns) + 4)
    fig_h = max(5, 0.45 * len(ct.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(ct.values, aspect="auto")
    ax.set_xticks(range(len(ct.columns)), labels=ct.columns)
    ax.set_yticks(range(len(ct.index)), labels=ct.index)
    ax.set_xlabel("Domain")
    ax.set_ylabel("ROI")
    ax.set_title("Patient–ROI counts by domain")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, str(int(ct.iat[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _boxplot_with_points(ax, sub: pd.DataFrame, metric: str, title: str, log_y: bool, seed: int) -> None:
    order = domain_order_present(sub["domain"])
    data = [sub.loc[sub["domain"] == d, "value"].dropna().to_numpy(float) for d in order]
    if not any(len(x) for x in data):
        ax.axis("off")
        return
    positions = np.arange(1, len(order) + 1)
    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)
    rng = np.random.default_rng(seed)
    ymax_candidates = []
    for pos, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=13, alpha=0.65)
        ymax_candidates.extend(vals[np.isfinite(vals)].tolist())
        # n annotation uses axes coordinates to remain readable under log scale.
        ax.text(pos, 0.98, f"n={len(vals)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.set_xticks(positions, order, rotation=35, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if log_y:
        positive = sub.loc[sub["value"] > 0, "value"]
        if not positive.empty:
            ax.set_yscale("log")


def save_roi_distribution_figure(
    df: pd.DataFrame,
    metric: str,
    outpath: Path,
    log_y: bool = False,
    seed: int = 88,
) -> None:
    primary = df[(~df["domain_conflict"]) & (df["metric"] == metric)].copy()
    rois = sorted(primary["roi"].unique())
    if not rois:
        return
    ncols = min(4, max(1, int(math.ceil(math.sqrt(len(rois))))))
    nrows = int(math.ceil(len(rois) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)
    for k, roi in enumerate(rois):
        ax = axes.flat[k]
        _boxplot_with_points(
            ax,
            primary[primary["roi"] == roi],
            metric,
            roi,
            log_y,
            seed + k,
        )
    for k in range(len(rois), nrows * ncols):
        axes.flat[k].axis("off")
    scale_note = " (log y-axis)" if log_y else ""
    fig.suptitle(f"{METRIC_LABELS[metric]} by domain and ROI{scale_note}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def save_pooled_severity_figure(df: pd.DataFrame, outpath: Path, seed: int = 88) -> None:
    primary = df[(~df["domain_conflict"]) & (df["domain"] != UNKNOWN_DOMAIN) & df["severity_z"].notna()].copy()
    metrics = [m for m in ["dice", "hd95"] if m in set(primary["metric"])]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
    rng = np.random.default_rng(seed)
    for j, metric in enumerate(metrics):
        ax = axes[0, j]
        sub = primary[primary["metric"] == metric]
        order = domain_order_present(sub["domain"])
        data = [sub.loc[sub["domain"] == d, "severity_z"].to_numpy(float) for d in order]
        pos = np.arange(1, len(order) + 1)
        ax.boxplot(data, positions=pos, widths=0.6, showfliers=False)
        for p, vals in zip(pos, data):
            if len(vals):
                jitter = rng.uniform(-0.16, 0.16, len(vals))
                ax.scatter(np.full(len(vals), p) + jitter, vals, s=11, alpha=0.55)
                ax.text(p, 0.98, f"n={len(vals)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
        ax.axhline(0, linewidth=1)
        ax.set_xticks(pos, order, rotation=35, ha="right")
        ax.set_ylabel("ROI-normalized error severity (robust z)")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Pooled thorax domain severity after ROI-specific normalization", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def save_dice_hd95_scatter(df: pd.DataFrame, outpath: Path) -> None:
    primary = df[~df["domain_conflict"]].copy()
    scatter_index = case_key_columns(primary, include_domain=True)
    wide = primary.pivot_table(
        index=scatter_index, columns="metric", values="value", aggfunc="first"
    ).reset_index()
    if not {"dice", "hd95"}.issubset(wide.columns):
        return
    wide = wide.dropna(subset=["dice", "hd95"])
    rois = sorted(wide["roi"].unique())
    if not rois:
        return
    ncols = min(4, max(1, int(math.ceil(math.sqrt(len(rois))))))
    nrows = int(math.ceil(len(rois) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.0 * nrows), squeeze=False)
    domains = domain_order_present(wide["domain"])
    # Use a stable matplotlib categorical map without hard-coding publication colors.
    cmap = plt.get_cmap("tab10")
    domain_to_color = {d: cmap(i % 10) for i, d in enumerate(domains)}
    for k, roi in enumerate(rois):
        ax = axes.flat[k]
        sub = wide[wide["roi"] == roi]
        for d in domains:
            g = sub[sub["domain"] == d]
            if not g.empty:
                ax.scatter(g["dice"], g["hd95"], s=22, alpha=0.7, label=d, color=domain_to_color[d])
        ax.set_xlabel("Dice")
        ax.set_ylabel("HD95 (mm)")
        ax.set_title(roi)
        ax.grid(alpha=0.2)
    for k in range(len(rois), nrows * ncols):
        axes.flat[k].axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", label=d, color=domain_to_color[d])
        for d in domains
    ]
    fig.legend(handles=handles, labels=domains, loc="upper center", ncol=min(len(domains), 9), frameon=False)
    fig.suptitle("Dice versus HD95 by domain", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def save_effect_size_forest(domain_vs_normal: pd.DataFrame, outpath: Path) -> None:
    if domain_vs_normal.empty:
        return
    d = domain_vs_normal.dropna(subset=["cliffs_delta_domain_vs_normal"]).copy()
    if d.empty:
        return
    d["label"] = d["roi"].astype(str) + " | " + d["metric"].astype(str) + " | " + d["domain"].astype(str)
    d = d.sort_values("cliffs_delta_domain_vs_normal")
    # Avoid unreadably tall images in very large analyses; all rows remain in CSV.
    if len(d) > 80:
        d = pd.concat([d.head(40), d.tail(40)]).sort_values("cliffs_delta_domain_vs_normal")
    fig_h = max(6, 0.22 * len(d) + 2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(d))
    ax.scatter(d["cliffs_delta_domain_vs_normal"], y, s=24)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y, d["label"], fontsize=7)
    ax.set_xlabel("Cliff's delta: domain vs Normal (raw metric)")
    ax.set_title("Domain-vs-Normal effect sizes")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No eligible comparisons._\n"
    x = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]):
            x[c] = x[c].map(lambda v: fmt(v, 3))
    header = "| " + " | ".join(map(str, x.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(x.columns)) + " |"
    rows = ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in x.to_numpy()]
    suffix = "\n\n" + (f"_Showing first {max_rows} rows; see CSV for the complete table._\n" if len(df) > max_rows else "")
    return "\n".join([header, sep] + rows) + suffix


def write_report(
    outdir: Path,
    analysis: pd.DataFrame,
    source_domains: pd.DataFrame,
    conflicts: pd.DataFrame,
    baseline: pd.DataFrame,
    desc: pd.DataFrame,
    kw: pd.DataFrame,
    nv: pd.DataFrame,
    ab: pd.DataFrame,
    pooled: pd.DataFrame,
    min_group_n: int,
    duplicate_rows: pd.DataFrame,
    keep_gc_label: bool,
) -> None:
    primary = analysis[~analysis["domain_conflict"]].copy()
    pair_table = primary.drop_duplicates(case_key_columns(primary, include_domain=True))
    counts = pair_table["domain"].value_counts().reindex(domain_order_present(pair_table["domain"]), fill_value=0)
    n_patients = primary["patient_id"].nunique()
    n_pairs = primary[case_key_columns(primary)].drop_duplicates().shape[0]
    n_rois = primary["roi"].nunique()
    n_metric_rows = len(primary)

    significant_nv = nv[(nv.get("q_bh", pd.Series(index=nv.index, dtype=float)) < 0.05)].copy() if not nv.empty else pd.DataFrame()
    if not significant_nv.empty:
        significant_nv["abs_delta"] = significant_nv["cliffs_delta_domain_vs_normal"].abs()
        significant_nv = significant_nv.sort_values(["abs_delta"], ascending=False)

    significant_ab = ab[(ab.get("q_bh", pd.Series(index=ab.index, dtype=float)) < 0.05)].copy() if not ab.empty else pd.DataFrame()
    if not significant_ab.empty:
        significant_ab["abs_delta"] = significant_ab["cliffs_delta_a_vs_b"].abs()
        significant_ab = significant_ab.sort_values(["abs_delta"], ascending=False)

    sparse = desc[desc["n"] < min_group_n].sort_values(["roi", "metric", "domain"]) if not desc.empty else pd.DataFrame()
    gc_note = (
        "The source domain label `GC` is preserved as `GC` because `--keep_gc_label` was used."
        if keep_gc_label
        else "Source label `GC` is displayed as `CA` for consistency with the study terminology; `domain_raw` preserves the original value."
    )

    report = []
    report.append("# Thorax domain analysis: Dice and HD95\n")
    report.append("## Analysis definition\n")
    report.append(
        "The complete metrics CSV defines the analysis universe. Domain assignments are joined at the "
        "**contour-inference level** using patient + study UID + ROI when `study-uid` is available "
        "(otherwise patient + ROI). An inference absent from the assignment CSV is labeled **Normal**. "
        "A recorded domain is propagated to both Dice and HD95 for that inference. Inferences with "
        "multiple incompatible recorded domains are excluded from primary inference and listed separately.\n"
    )
    report.append(f"{gc_note}\n")
    report.append(
        "Domain groups: **I** imaging, **AV** anatomical variation, **MI** metallic implants, "
        "**AC** auto-contour error, **H** human/clinical-contour issue, **CA** clinically acceptable, "
        "**NDI** minor edits/no dosimetric impact, and **LDI** major edits/likely dosimetric impact.\n"
    )

    report.append("## Cohort and domain composition\n")
    report.append(
        f"Primary dataset: **{n_patients} patients**, **{n_pairs} contour inferences**, "
        f"**{n_rois} ROIs**, and **{n_metric_rows} Dice/HD95 observations**.\n"
    )
    count_df = counts.rename_axis("domain").reset_index(name="patient_roi_pairs")
    report.append(markdown_table(count_df, ["domain", "patient_roi_pairs"], max_rows=20))
    if not conflicts.empty:
        report.append(
            f"There are **{len(conflicts)} conflicting contour-inference domain assignments**. They are excluded "
            "from the primary analysis under the default policy; review `domain_assignment_conflicts.csv`.\n"
        )
    if not duplicate_rows.empty:
        report.append(
            f"The complete metrics input contained **{len(duplicate_rows)} duplicate inference-metric rows** after using the available case key. "
            "See `duplicate_metric_rows.csv` and the selected duplicate policy.\n"
        )

    report.append("![Domain counts](figures/domain_counts_heatmap.png)\n")

    report.append("## Descriptive distributions\n")
    report.append(
        "The ROI-faceted plots are the primary descriptive view. Points are individual observations; boxes "
        "show the median and interquartile range. Sample sizes are printed above each domain. Comparisons with "
        f"fewer than **{min_group_n} observations per group** are considered descriptive only.\n"
    )
    report.append("![Dice distributions](figures/dice_by_domain_roi.png)\n")
    report.append("![HD95 distributions](figures/hd95_by_domain_roi.png)\n")
    if (outdir / "figures" / "hd95_by_domain_roi_log.png").exists():
        report.append(
            "The log-scale HD95 figure is provided only to make long-tailed distributions visible; statistical "
            "tests are performed on the original HD95 values.\n"
        )
        report.append("![HD95 log distributions](figures/hd95_by_domain_roi_log.png)\n")

    report.append("## ROI-normalized pooled analysis\n")
    report.append(
        "To compare domains across heterogeneous ROIs, each metric is converted to an ROI-specific robust "
        "severity score using the **Normal** group as the reference. Higher severity is always worse: Dice is "
        "sign-reversed so lower Dice produces higher severity, while higher HD95 directly produces higher severity. "
        "The primary scale is 1.4826×MAD, with IQR/1.349 and SD fallbacks if needed. This pooled view should "
        "complement—not replace—the ROI-specific results.\n"
    )
    report.append("![Pooled normalized severity](figures/pooled_normalized_severity.png)\n")
    report.append(
        f"ROI/metric strata without at least {min_group_n} Normal observations cannot be normalized and are "
        "listed in `normalization_baselines.csv`.\n"
    )

    report.append("## Joint Dice–HD95 failure patterns\n")
    report.append(
        "The Dice-versus-HD95 scatter is useful for identifying domain-specific error signatures. A case with "
        "moderate Dice but very high HD95 may represent a localized extent/boundary failure, whereas simultaneous "
        "Dice degradation and HD95 elevation indicates broader disagreement.\n"
    )
    report.append("![Dice versus HD95](figures/dice_vs_hd95_by_domain_roi.png)\n")

    report.append("## Statistical comparisons\n")
    report.append(
        "Per ROI and metric, the analysis uses a Kruskal–Wallis omnibus test followed by two-sided Mann–Whitney "
        "comparisons. Benjamini–Hochberg FDR correction is applied within each output table. Effect magnitude is "
        "reported with Cliff's delta, and median differences use a patient-cluster bootstrap 95% CI.\n"
    )
    report.append("### Domain versus Normal\n")
    if significant_nv.empty:
        report.append("No domain-versus-Normal comparison met FDR q<0.05 among eligible comparisons.\n")
    else:
        report.append("Largest FDR-significant domain-versus-Normal effects:\n")
        report.append(markdown_table(
            significant_nv,
            ["roi", "metric", "domain", "n_domain", "n_normal", "median_difference_domain_minus_normal", "cliffs_delta_domain_vs_normal", "p", "q_bh"],
            max_rows=20,
        ))
    report.append("![Effect sizes](figures/domain_vs_normal_effect_sizes.png)\n")

    report.append("### Differences among recorded domains\n")
    if significant_ab.empty:
        report.append("No abnormal/recorded-domain pairwise comparison met FDR q<0.05 among eligible comparisons.\n")
    else:
        report.append("Largest FDR-significant recorded-domain differences:\n")
        report.append(markdown_table(
            significant_ab,
            ["roi", "metric", "domain_a", "domain_b", "n_a", "n_b", "median_difference_a_minus_b", "cliffs_delta_a_vs_b", "p", "q_bh"],
            max_rows=20,
        ))

    report.append("## Sparse groups\n")
    report.append(
        f"Groups with n<{min_group_n} are plotted and summarized but not used for primary pairwise inference. "
        "This is especially important for rare domains such as metallic implants or clinically categorized edit groups.\n"
    )
    if not sparse.empty:
        report.append(markdown_table(sparse, ["roi", "metric", "domain", "n", "median", "q1", "q3"], max_rows=30))

    report.append("## Interpretation guidelines\n")
    report.append(
        "1. **Start with ROI-specific distributions.** Raw Dice and HD95 have very different baselines across structures; "
        "a pooled raw-metric comparison can be confounded by ROI composition.\n"
        "2. **Use effect sizes and confidence intervals with p/q-values.** A small p-value does not imply a clinically "
        "important separation, particularly with large Normal groups.\n"
        "3. **Treat Normal-versus-OOD differences as partly selection-conditioned.** The poorer-agreement domains were "
        "assigned after cases were flagged using Dice/HD95 control limits. Therefore, large Normal-versus-I/AV/MI/AC/H "
        "differences are expected and are not an independent validation that those domains cause low Dice or high HD95.\n"
        "4. **Differences among abnormal domains are less circular and often more informative.** They can reveal whether, "
        "for example, extent errors, imaging problems, anatomical variation, or human-reference issues produce distinct "
        "metric severity profiles.\n"
        "5. **Interpret CA/NDI/LDI separately from the poorer-agreement domain taxonomy.** These labels describe clinical "
        "acceptability/edit burden among better-agreement cases rather than the same causal mechanism categories.\n"
        "6. **Do not overinterpret sparse cells.** Rare domain×ROI combinations should be treated as case-series evidence "
        "until additional observations are available.\n"
        "7. **Dice and HD95 are complementary.** Dice is overlap/volume sensitive; HD95 is boundary/extreme-error sensitive. "
        "A domain may affect one much more strongly than the other.\n"
        "8. **Avoid causal wording.** This analysis establishes association between assigned domain and contour agreement; "
        "it does not establish that the domain itself caused the segmentation error.\n"
        "9. **Only month2/month3 are domain-review eligible.** Month1 and patients outside all three declared month periods are "
        "shown descriptively as `Unknown` and are excluded from Normal-vs-domain and abnormal-domain inferential comparisons.\n"
        "10. **Review conflict and duplicate tables before publication.** Domain-assignment disagreements should be adjudicated "
        "rather than silently resolved when the final manuscript dataset is frozen.\n"
    )

    report.append("## Output files\n")
    report.append(
        "- `analysis_dataset.csv`: complete metrics with propagated domain and normalized severity.\n"
        "- `domain_assignment_conflicts.csv`: patient–ROI pairs with incompatible recorded domains.\n"
        "- `domain_assignment_source_thorax.csv`: filtered source assignments with raw and analysis labels.\n"
        "- `descriptive_statistics.csv`: ROI×metric×domain summaries.\n"
        "- `kruskal_wallis.csv`: omnibus tests.\n"
        "- `domain_vs_normal.csv`: primary pairwise comparisons.\n"
        "- `abnormal_domain_pairwise.csv`: comparisons among recorded domains.\n"
        "- `pooled_severity_domain_vs_normal.csv`: ROI-normalized pooled comparisons.\n"
        "- `normalization_baselines.csv`: ROI/metric Normal baselines and robust scales.\n"
        "- `figures/`: publication-ready diagnostic plots.\n"
    )

    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    metrics, duplicate_rows = load_metrics(args.metrics_csv, args.site, args.duplicate_policy)
    pair_domains, conflicts, source_domains = load_pair_domains(
        args.domain_csv,
        args.site,
        args.keep_gc_label,
        args.conflict_policy,
    )
    analysis = build_analysis_dataset(metrics, pair_domains, conflicts, args.conflict_policy)
    analysis, baseline = add_normalized_severity(analysis, args.min_group_n)

    desc = descriptive_statistics(analysis)
    kw, nv, ab = run_inference(
        analysis,
        min_group_n=args.min_group_n,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    pooled = run_pooled_severity_inference(
        analysis,
        min_group_n=args.min_group_n,
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    # Tables.
    analysis.to_csv(outdir / "analysis_dataset.csv", index=False)
    source_domains.to_csv(outdir / "domain_assignment_source_thorax.csv", index=False)
    conflicts.to_csv(outdir / "domain_assignment_conflicts.csv", index=False)
    duplicate_rows.to_csv(outdir / "duplicate_metric_rows.csv", index=False)
    baseline.to_csv(outdir / "normalization_baselines.csv", index=False)
    desc.to_csv(outdir / "descriptive_statistics.csv", index=False)
    kw.to_csv(outdir / "kruskal_wallis.csv", index=False)
    nv.to_csv(outdir / "domain_vs_normal.csv", index=False)
    ab.to_csv(outdir / "abnormal_domain_pairwise.csv", index=False)
    pooled.to_csv(outdir / "pooled_severity_domain_vs_normal.csv", index=False)

    count_source = analysis[~analysis["domain_conflict"]].copy()
    pair_counts = (
        count_source
        .drop_duplicates(case_key_columns(count_source, include_domain=True))
        .groupby(["roi", "domain"], as_index=False)
        .size()
        .rename(columns={"size": "patient_roi_pairs"})
    )
    pair_counts.to_csv(outdir / "domain_counts.csv", index=False)

    # Figures.
    save_domain_count_heatmap(analysis, figdir / "domain_counts_heatmap.png")
    save_roi_distribution_figure(analysis, "dice", figdir / "dice_by_domain_roi.png", seed=args.seed)
    save_roi_distribution_figure(analysis, "hd95", figdir / "hd95_by_domain_roi.png", seed=args.seed)
    if not args.no_log_hd95:
        save_roi_distribution_figure(
            analysis, "hd95", figdir / "hd95_by_domain_roi_log.png", log_y=True, seed=args.seed
        )
    save_pooled_severity_figure(analysis, figdir / "pooled_normalized_severity.png", seed=args.seed)
    save_dice_hd95_scatter(analysis, figdir / "dice_vs_hd95_by_domain_roi.png")
    save_effect_size_forest(nv, figdir / "domain_vs_normal_effect_sizes.png")

    write_report(
        outdir=outdir,
        analysis=analysis,
        source_domains=source_domains,
        conflicts=conflicts,
        baseline=baseline,
        desc=desc,
        kw=kw,
        nv=nv,
        ab=ab,
        pooled=pooled,
        min_group_n=args.min_group_n,
        duplicate_rows=duplicate_rows,
        keep_gc_label=args.keep_gc_label,
    )

    print("Domain analysis complete.")
    print(f"  Metrics rows: {len(metrics)}")
    print(f"  Patients: {analysis['patient_id'].nunique()}")
    print(f"  Contour inferences: {analysis[case_key_columns(analysis)].drop_duplicates().shape[0]}")
    print(f"  Domain conflicts: {len(conflicts)}")
    print(f"  Report: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
