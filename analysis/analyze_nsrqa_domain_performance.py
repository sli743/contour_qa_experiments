#!/usr/bin/env python3
"""Domain-stratified evaluation of NSR-QA on real thorax auto-contours.

This script analyzes already-generated NSR-QA case tables. By default it uses
held-out/out-of-fold ranking/classification cases. An explicit exploratory
``--analysis_scope whole`` mode instead uses one fold's complete
training+validation+testing cohort from the upstream NSR-QA metrics CSVs. It
does not retrain the shape model and does not use domain labels during model
fitting.

Primary questions
-----------------
1. At a fixed QA operating point, which domains are detected or missed?
2. Within each domain, does the NSR-QA score track true contour-error severity?
3. After accounting for true error severity and ROI, does the NSR response differ
   by domain (domain-specific blind spots / excess response)?

Input
-----
--ranking_root should contain one or more ``classification_cases.csv`` files
from ``analysis/analyze_actual_autoseg_ranking_classification.py``. In the
default ``oof`` scope these are the analyzed cases. In ``whole`` scope they are
used only to recover the already-frozen development QA thresholds when
available; the case rows instead come from
``actual_autoseg_regression/metrics/<ROI>/<localization>_metrics.csv``.
``--metrics_root`` may point directly to that metrics directory; otherwise the
script tries to discover it near the study root.

The domain CSV is the OOD/domain assignment table. Only month2/month3 patients
are domain-review eligible; unrecorded pairs within that reviewed cohort are
assigned to ``Normal``. Month1 and patients outside all three declared month
periods are assigned to ``Unknown``. ``GC`` is presented as ``CA`` while the source label
is preserved in audit output. Clinical Dice and Hausdorff/HD95 truth values are
read from ``--clinical_metrics_csv`` (the original Radformation ``metrics.csv``)
rather than recomputed from the resampled NSR-QA geometry. The NSR-QA score still
comes from the ranking/upstream NSR-QA result tables.

Important limitations
---------------------
The ``whole`` scope is deliberately descriptive/sensitivity analysis, not an
unbiased generalization estimate. Training and validation cases were exposed to
model fitting and/or model selection and can therefore look artificially good.
Use the default held-out/OOF analysis as the primary performance estimate.

Current NSR-QA case tables identify cases by patient_id + ROI + outer fold but do
not carry study-uid. The domain table can carry study-uid. Therefore the primary
join is patient_id + ROI. Any patient-ROI having conflicting non-null domain
labels is excluded from primary domain comparisons and written to an audit CSV.
Patient-ROI pairs associated with multiple study UIDs are also listed for review.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = {
    "1_minus_dice": {
        "qa_col": "qa_1_minus_dice",
        "true_col": "true_1_minus_dice",
        "label": "1 - Dice",
        "transform": "identity",
    },
    "hd95": {
        "qa_col": "qa_hd95_mm",
        "true_col": "true_hd95_mm",
        "label": "HD95 (mm)",
        "transform": "log1p",
    },
}

# Original clinical metrics table naming. In this project ``haus`` is the
# Hausdorff/HD95 quantity used for the domain analysis.
CLINICAL_METRIC_MAP = {
    "1_minus_dice": "dice",
    "hd95": "haus",
}
ABNORMAL_DOMAINS = ["I", "AV", "MI", "AC", "H"]

# Domain review was performed only for the second and third month periods.
# Month1 is explicitly unreviewed, and patients outside all three month periods
# must also remain Unknown. Patient 16 appears in month1 and month3; the
# conservative month1 rule takes precedence, so it is treated as Unknown.
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

FAILURE_DOMAINS = ["Normal", "I", "AV", "MI", "AC", "H"]
CLINICAL_DOMAINS = ["Normal", "CA", "NDI", "LDI"]
DOMAIN_ORDER = ["Normal", "Unknown", "I", "AV", "MI", "AC", "H", "CA", "NDI", "LDI"]
DOMAIN_DESCRIPTIONS = {
    "Normal": "Unrecorded patient-ROI inference within the domain-reviewed month2/month3 cohort; treated as the reference domain.",
    "Unknown": "Patient outside the domain-reviewed cohort (month1 or outside all three month periods).",
    "I": "Imaging: image quality, positioning, contrast, or related imaging factors.",
    "AV": "Anatomical variation: unusual anatomy, surgery, calcification, etc.",
    "MI": "Metallic implants: pacemaker, hip implant, or related metal effects.",
    "AC": "Auto-contour error: extent, over-contouring, under-contouring, etc.",
    "H": "Human: suboptimal clinical contour or contour not following guidelines.",
    "CA": "Clinically acceptable contour (source label GC is mapped to CA for presentation).",
    "NDI": "Minor edits needed; no expected dosimetric impact.",
    "LDI": "Major edits needed; likely dosimetric impact.",
}



def _norm_patient_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.0+)?", s):
        try:
            return str(int(float(s)))
        except Exception:
            pass
    return s


def _normalize_domain(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"NAN", "NONE"}:
        return None
    if s == "GC":
        return "CA"
    return s


def _is_month1_patient(x) -> bool:
    return _norm_patient_id(x) in MONTH1_PATIENT_IDS


def _is_outside_three_month_periods(x) -> bool:
    return _norm_patient_id(x) not in ALL_MONTH_PATIENT_IDS


def _is_domain_reviewed_patient(x) -> bool:
    return _norm_patient_id(x) in DOMAIN_REVIEWED_PATIENT_IDS


def _apply_domain_review_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Force every non-reviewed patient to Unknown, regardless of recorded domain.

    Only month2/month3 patients are domain-review eligible. Month1 takes
    precedence over the duplicated month3 entry for patient 16, and patients
    outside all three declared month periods are also treated as Unknown.
    """
    out = df.copy()
    if "patient_id" not in out.columns:
        return out
    month1_mask = out["patient_id"].map(_is_month1_patient)
    outside_mask = out["patient_id"].map(_is_outside_three_month_periods)
    unknown_mask = ~out["patient_id"].map(_is_domain_reviewed_patient)
    if "domain" in out.columns:
        out.loc[unknown_mask, "domain"] = UNKNOWN_DOMAIN
    if "domain_conflict" in out.columns:
        out.loc[unknown_mask, "domain_conflict"] = False
    out["month1_domain_unknown"] = month1_mask.astype(bool)
    out["outside_three_month_periods"] = outside_mask.astype(bool)
    out["domain_review_unknown"] = unknown_mask.astype(bool)
    return out


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out[valid] = q[inv]
    return out


def bootstrap_rate_cluster(
    df: pd.DataFrame,
    value_col: str = "qa_alert",
    cluster_col: str = "patient_id",
    n_boot: int = 1000,
    seed: int = 88,
) -> Tuple[float, float, int]:
    d = df[[cluster_col, value_col]].dropna().copy()
    if d.empty:
        return np.nan, np.nan, 0
    grouped = d.groupby(cluster_col)[value_col].agg(["sum", "count"])
    if len(grouped) < 2:
        return np.nan, np.nan, 0
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    ncl = len(grouped)
    for b in range(n_boot):
        idx = rng.integers(0, ncl, size=ncl)
        denom = counts[idx].sum()
        vals[b] = sums[idx].sum() / denom if denom > 0 else np.nan
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi), int(len(vals))


def bootstrap_spearman_cluster(
    df: pd.DataFrame,
    x_col: str = "qa_score",
    y_col: str = "true_error",
    cluster_col: str = "patient_id",
    n_boot: int = 1000,
    seed: int = 88,
) -> Tuple[float, float, int]:
    d = df[[cluster_col, x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    clusters = d[cluster_col].astype(str).unique()
    if len(clusters) < 2:
        return np.nan, np.nan, 0
    pieces = []
    for c in clusters:
        g = d[d[cluster_col].astype(str) == c]
        pieces.append((g[x_col].to_numpy(float), g[y_col].to_numpy(float)))
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    ncl = len(pieces)
    for _ in range(n_boot):
        idx = rng.integers(0, ncl, size=ncl)
        x = np.concatenate([pieces[i][0] for i in idx])
        y = np.concatenate([pieces[i][1] for i in idx])
        if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        r = stats.spearmanr(x, y).statistic
        if np.isfinite(r):
            vals.append(float(r))
    if not vals:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi), len(vals)


def safe_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan, np.nan
    r = stats.spearmanr(d["x"], d["y"])
    return float(r.statistic), float(r.pvalue)


def empirical_specificity(values: pd.Series, threshold: float) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0 or not np.isfinite(threshold):
        return np.nan
    return float(np.mean(x < threshold))


def specificity_threshold(values: pd.Series, target_specificity: float) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan
    # Higher NSR score means more abnormal. "higher" is conservative for the
    # requested false-positive rate in a finite calibration sample.
    try:
        q = float(np.quantile(x, target_specificity, method="higher"))
    except TypeError:  # old NumPy
        q = float(np.quantile(x, target_specificity, interpolation="higher"))
    # Alerts are defined as score >= threshold. Move infinitesimally above the
    # selected calibration value so ties at a quantized score (common for
    # HD95) count as non-alerts and empirical specificity is at least the
    # requested finite-sample quantile whenever possible.
    return float(np.nextafter(q, np.inf))


def load_domain_assignments(path: str, site: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(path)
    required = {"patient_id", "roi", "domain"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Domain CSV missing required columns: {missing}")
    if "site" in d.columns:
        d = d[d["site"].astype(str).str.lower() == site.lower()].copy()
    d["patient_id"] = d["patient_id"].map(_norm_patient_id)
    d["roi"] = d["roi"].astype(str).str.strip()
    d["domain_source"] = d["domain"]
    d["domain"] = d["domain"].map(_normalize_domain)
    d["month1_domain_unknown"] = d["patient_id"].map(_is_month1_patient)
    d["outside_three_month_periods"] = d["patient_id"].map(_is_outside_three_month_periods)
    d["domain_review_unknown"] = ~d["patient_id"].map(_is_domain_reviewed_patient)
    d.loc[d["domain_review_unknown"], "domain"] = UNKNOWN_DOMAIN

    # Only a non-null recorded domain constitutes an assignment. Rows with a
    # blank domain do not turn an otherwise unrecorded inference into a domain.
    assigned = d[d["domain"].notna()].copy()

    key = ["patient_id", "roi"]
    agg_rows = []
    conflict_rows = []
    for (pid, roi), g in assigned.groupby(key, dropna=False):
        domains = sorted(set(g["domain"].dropna().astype(str)))
        study_uids = sorted(set(g["study-uid"].dropna().astype(str))) if "study-uid" in g.columns else []
        row = {
            "patient_id": pid,
            "roi": roi,
            "domain": domains[0] if len(domains) == 1 else "CONFLICT",
            "n_recorded_rows": len(g),
            "n_unique_domains": len(domains),
            "domains_recorded": "|".join(domains),
            "n_unique_study_uid": len(study_uids),
            "study_uids_recorded": "|".join(study_uids),
        }
        agg_rows.append(row)
        if len(domains) > 1:
            conflict_rows.append(row)

    assignments = pd.DataFrame(agg_rows)
    if assignments.empty:
        assignments = pd.DataFrame(columns=key + [
            "domain", "n_recorded_rows", "n_unique_domains", "domains_recorded",
            "n_unique_study_uid", "study_uids_recorded"
        ])
    conflicts = pd.DataFrame(conflict_rows)

    # Audit patient-ROI assignments that span >1 study UID because NSR case
    # tables currently lack study-uid and cannot disambiguate those studies.
    multi_study = assignments[assignments["n_unique_study_uid"] > 1].copy()
    return assignments, conflicts, multi_study


def load_clinical_metrics(
    path: str,
    site: str,
    duplicate_policy: str = "exclude",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load original clinical geometry metrics from Radformation ``metrics.csv``.

    Returns
    -------
    case_metric : DataFrame
        One row per patient/ROI/metric for joining to NSR-QA case tables, which
        currently do not carry study UID. Ambiguous multi-study values are
        handled according to ``duplicate_policy``.
    exact_metric : DataFrame
        One row per patient/study/ROI/metric when study UID is available. This
        is used for the SpinalCord abnormal-case summary so values are not
        unnecessarily mixed across studies.
    audit : DataFrame
        Patient/ROI/metric groups containing multiple source rows/studies.
    """
    d = pd.read_csv(path)
    required = {"patient_id", "roi", "metric", "value"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Clinical metrics CSV missing required columns: {missing}")
    if "site" in d.columns:
        d = d[d["site"].astype(str).str.lower() == site.lower()].copy()
    d["patient_id"] = d["patient_id"].map(_norm_patient_id)
    d["roi"] = d["roi"].astype(str).str.strip()
    d["metric"] = d["metric"].astype(str).str.strip().str.lower()
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d[np.isfinite(d["value"])].copy()
    if "study-uid" in d.columns:
        d["study-uid"] = d["study-uid"].fillna("").astype(str).str.strip()
    else:
        d["study-uid"] = ""

    # First collapse exact duplicate rows within a single study. If distinct
    # values remain for the same exact case, use the requested policy and audit.
    exact_rows = []
    exact_audit = []
    exact_key = ["patient_id", "study-uid", "roi", "metric"]
    for key, g in d.groupby(exact_key, dropna=False, sort=True):
        vals = pd.to_numeric(g["value"], errors="coerce").dropna().to_numpy(float)
        uniq = np.unique(vals)
        row = dict(zip(exact_key, key))
        row["n_source_rows"] = int(len(g))
        row["n_unique_values"] = int(len(uniq))
        if len(uniq) <= 1:
            value = float(uniq[0]) if len(uniq) else np.nan
            status = "unique" if len(g) == 1 else "duplicate_identical"
        elif duplicate_policy == "median":
            value = float(np.median(vals))
            status = "duplicate_median"
        elif duplicate_policy == "first":
            value = float(vals[0])
            status = "duplicate_first"
        elif duplicate_policy == "exclude":
            value = np.nan
            status = "duplicate_excluded"
        else:
            raise ValueError(
                "Conflicting clinical metric values for exact patient/study/ROI/metric "
                f"{key}: {uniq[:10].tolist()}"
            )
        row["clinical_metric_value"] = value
        row["clinical_metric_status"] = status
        exact_rows.append(row)
        if len(g) > 1 or len(uniq) > 1:
            exact_audit.append({**row, "source_values": "|".join(map(str, vals.tolist()))})

    exact = pd.DataFrame(exact_rows)

    # NSR-QA case tables currently lack study UID. Collapse across study UIDs
    # only for the join needed by the severity analysis. Ambiguous groups are
    # excluded by default rather than silently mixing separate studies.
    case_rows = []
    case_audit = []
    for (pid, roi, metric), g in exact.groupby(["patient_id", "roi", "metric"], sort=True):
        vals = pd.to_numeric(g["clinical_metric_value"], errors="coerce").dropna().to_numpy(float)
        studies = sorted(x for x in g["study-uid"].astype(str).unique() if x)
        uniq = np.unique(vals)
        row = {
            "patient_id": pid,
            "roi": roi,
            "clinical_metric_name": metric,
            "clinical_metric_n_studies": int(len(studies)),
            "clinical_metric_study_uids": "|".join(studies),
            "clinical_metric_n_values": int(len(vals)),
        }
        if len(uniq) <= 1:
            value = float(uniq[0]) if len(uniq) else np.nan
            status = "unique" if len(g) == 1 else "multi_study_identical"
        elif duplicate_policy == "median":
            value = float(np.median(vals))
            status = "multi_study_median"
        elif duplicate_policy == "first":
            value = float(vals[0])
            status = "multi_study_first"
        elif duplicate_policy == "exclude":
            value = np.nan
            status = "multi_study_excluded"
        else:
            raise ValueError(
                "Multiple clinical metric values across studies for patient/ROI/metric "
                f"{(pid, roi, metric)}: {uniq[:10].tolist()}"
            )
        row["clinical_metric_value"] = value
        row["clinical_metric_status"] = status
        case_rows.append(row)
        if len(g) > 1 or len(uniq) > 1:
            case_audit.append({
                **row,
                "source_values": "|".join(map(str, vals.tolist())),
            })

    case_metric = pd.DataFrame(case_rows)
    audit = pd.concat(
        [pd.DataFrame(exact_audit).assign(audit_level="exact_study"),
         pd.DataFrame(case_audit).assign(audit_level="patient_roi")],
        ignore_index=True,
        sort=False,
    )
    return case_metric, exact, audit


def replace_true_error_from_clinical_metrics(
    cases: pd.DataFrame,
    clinical_case_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Use original ``metrics.csv`` values as the clinical geometric truth.

    NSR-QA QA scores are untouched. The pipeline-derived true error is retained
    as ``pipeline_true_error`` for audit/reproducibility.
    """
    out = cases.copy()
    out["pipeline_true_error"] = pd.to_numeric(out["true_error"], errors="coerce")
    out["clinical_metric_name"] = out["metric"].map(CLINICAL_METRIC_MAP)

    cm = clinical_case_metrics.rename(columns={"clinical_metric_name": "clinical_metric_name_join"})
    out = out.merge(
        cm,
        left_on=["patient_id", "roi", "clinical_metric_name"],
        right_on=["patient_id", "roi", "clinical_metric_name_join"],
        how="left",
        validate="many_to_one",
    )
    out = out.drop(columns=["clinical_metric_name_join"], errors="ignore")

    raw = pd.to_numeric(out["clinical_metric_value"], errors="coerce")
    out["true_error"] = np.where(
        out["metric"].eq("1_minus_dice"),
        1.0 - raw,
        raw,
    )
    out["true_error_source"] = "metrics.csv"
    out.loc[~np.isfinite(pd.to_numeric(out["true_error"], errors="coerce")), "true_error_source"] = (
        "metrics.csv_unavailable_or_ambiguous"
    )
    return out


def build_spinalcord_abnormal_summary(
    domain_csv: str,
    clinical_exact_metrics: pd.DataFrame,
    nsr_cases: pd.DataFrame,
    site: str,
    roi: str = "SpinalCord",
    max_patient_id_exclusive: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize abnormal SpinalCord cases using original ``metrics.csv`` values.

    ``ABNORMAL_DOMAINS`` are the poorer-agreement mechanism domains I/AV/MI/AC/H.
    The wide table has one row per patient/study and one column per available
    clinical metric. The long table preserves metric/value pairs directly.
    """
    d = pd.read_csv(domain_csv)
    required = {"patient_id", "roi", "domain"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Domain CSV missing required columns for SpinalCord summary: {missing}")
    if "site" in d.columns:
        d = d[d["site"].astype(str).str.lower() == site.lower()].copy()
    d["patient_id"] = d["patient_id"].map(_norm_patient_id)
    d["roi"] = d["roi"].astype(str).str.strip()
    d["domain_norm"] = d["domain"].map(_normalize_domain)
    domain_review_unknown = ~d["patient_id"].map(_is_domain_reviewed_patient)
    d.loc[domain_review_unknown, "domain_norm"] = UNKNOWN_DOMAIN
    if "study-uid" not in d.columns:
        d["study-uid"] = ""
    else:
        d["study-uid"] = d["study-uid"].fillna("").astype(str).str.strip()
    d = d[(d["roi"] == roi) & d["domain_norm"].isin(ABNORMAL_DOMAINS)].copy()

    # Keep only real clinical patient IDs requested for this export. Patient IDs
    # that are non-numeric or >= max_patient_id_exclusive are excluded rather
    # than silently coerced into the clinical table.
    d["patient_id_numeric"] = pd.to_numeric(d["patient_id"], errors="coerce")
    d = d[d["patient_id_numeric"].notna() &
          (d["patient_id_numeric"] < int(max_patient_id_exclusive))].copy()
    d = d.drop(columns=["patient_id_numeric"])

    group_key = ["patient_id", "study-uid", "roi"]
    case_rows = []
    for key, g in d.groupby(group_key, dropna=False, sort=True):
        domains = sorted(set(g["domain_norm"].dropna().astype(str)))
        row = dict(zip(group_key, key))
        row["domain"] = domains[0] if len(domains) == 1 else "CONFLICT"
        row["domains_recorded"] = "|".join(domains)
        row["domain_conflict"] = len(domains) > 1
        if "metric" in g.columns:
            row["domain_source_metrics"] = "|".join(sorted(set(g["metric"].dropna().astype(str))))
        if "month" in g.columns:
            months = sorted(set(pd.to_numeric(g["month"], errors="coerce").dropna().astype(int).tolist()))
            row["month"] = "|".join(map(str, months))
        if "comment" in g.columns:
            comments = [str(x).strip() for x in g["comment"].dropna() if str(x).strip()]
            row["comment"] = " | ".join(dict.fromkeys(comments))
        case_rows.append(row)
    cases = pd.DataFrame(case_rows)
    if cases.empty:
        return cases, pd.DataFrame()

    metric = clinical_exact_metrics[clinical_exact_metrics["roi"].astype(str) == roi].copy()
    join_keys = ["patient_id", "roi"]
    if (cases["study-uid"].astype(str).str.len() > 0).any() and \
            (metric["study-uid"].astype(str).str.len() > 0).any():
        join_keys.insert(1, "study-uid")

    long = cases.merge(metric, on=join_keys, how="left")
    keep = [c for c in [
        "patient_id", "study-uid", "roi", "domain", "domains_recorded",
        "domain_conflict", "domain_source_metrics", "month", "comment",
        "metric", "clinical_metric_value", "clinical_metric_status",
    ] if c in long.columns]
    long = long[keep].sort_values(["patient_id", "metric"], kind="stable").reset_index(drop=True)

    metric_index = [c for c in ["patient_id", "study-uid"] if c in long.columns]
    metric_values = long.dropna(subset=["metric"]).pivot(
        index=metric_index, columns="metric", values="clinical_metric_value"
    ).reset_index() if long["metric"].notna().any() else pd.DataFrame(columns=metric_index)
    metric_values.columns.name = None
    wide_metric = cases.merge(metric_values, on=metric_index, how="left", validate="one_to_one")

    # Add the two primary NSR-QA signals. Current NSR case tables do not carry
    # study UID, so QA scores are summarized at patient+ROI+metric level. If a
    # patient has more than one distinct QA score for the same metric (for
    # example because multiple incompatible folds/results were mixed), mark the
    # score ambiguous instead of choosing one silently.
    qa = nsr_cases.copy()
    if not qa.empty:
        qa["patient_id"] = qa["patient_id"].map(_norm_patient_id)
        qa["roi"] = qa["roi"].astype(str).str.strip()
        qa["patient_id_numeric"] = pd.to_numeric(qa["patient_id"], errors="coerce")
        qa = qa[(qa["roi"] == roi) & qa["patient_id_numeric"].notna() &
                (qa["patient_id_numeric"] < int(max_patient_id_exclusive))].copy()

    qa_rows = []
    for (pid, metric_name), g in qa.groupby(["patient_id", "metric"], sort=True):
        vals = pd.to_numeric(g["qa_score"], errors="coerce")
        vals = vals[np.isfinite(vals)].to_numpy(float)
        unique_vals = np.unique(vals)
        qa_rows.append({
            "patient_id": pid,
            "metric": metric_name,
            "qa_score_value": float(unique_vals[0]) if len(unique_vals) == 1 else np.nan,
            "qa_score_status": "ok" if len(unique_vals) == 1 else ("missing" if len(unique_vals) == 0 else "ambiguous"),
            "qa_score_n_unique": int(len(unique_vals)),
        })
    qa_summary = pd.DataFrame(qa_rows)

    if not qa_summary.empty:
        qa_values = qa_summary.pivot(index="patient_id", columns="metric", values="qa_score_value").reset_index()
        qa_values.columns.name = None
        qa_values = qa_values.rename(columns={
            "1_minus_dice": "qa_1_minus_dice",
            "hd95": "qa_hd95_mm",
        })
        if "qa_1_minus_dice" in qa_values.columns:
            qa_values["qa_input_recon_dice"] = 1.0 - pd.to_numeric(qa_values["qa_1_minus_dice"], errors="coerce")

        qa_status = qa_summary.pivot(index="patient_id", columns="metric", values="qa_score_status").reset_index()
        qa_status.columns.name = None
        qa_status = qa_status.rename(columns={
            "1_minus_dice": "qa_1_minus_dice_status",
            "hd95": "qa_hd95_mm_status",
        })
        wide_metric = wide_metric.merge(qa_values, on="patient_id", how="left", validate="many_to_one")
        wide_metric = wide_metric.merge(qa_status, on="patient_id", how="left", validate="many_to_one")

        # Add QA values to the long table as audit columns too. They repeat
        # across clinical metric rows by design.
        qa_long_cols = [c for c in [
            "patient_id", "qa_1_minus_dice", "qa_input_recon_dice", "qa_hd95_mm",
            "qa_1_minus_dice_status", "qa_hd95_mm_status",
        ] if c in wide_metric.columns]
        long = long.merge(
            wide_metric[qa_long_cols].drop_duplicates(subset=["patient_id"]),
            on="patient_id", how="left", validate="many_to_one"
        )

    return wide_metric, long


def _classification_case_files(root: str) -> List[str]:
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        return []
    return sorted(glob.glob(os.path.join(root, "**", "classification_cases.csv"), recursive=True))


def _candidate_result_roots(files: Sequence[str]) -> List[str]:
    """Summarize likely ranking-classification roots for diagnostics.

    A normal per-ROI layout is:
      <root>/<ROI>/classification_cases.csv
    so the likely result root is two directories above the CSV.
    """
    roots = []
    for f in files:
        roots.append(os.path.dirname(os.path.dirname(os.path.abspath(f))))
    return sorted(set(roots))


def resolve_ranking_root(ranking_root: str) -> Tuple[str, List[str]]:
    """Resolve a ranking result root with a conservative parent fallback.

    The submit command historically used a conventional subdirectory name such
    as ``.../all_roi_NSRQA_study/ranking_classification``.  Some all-ROI/five-fold
    workflows instead place the per-ROI ranking outputs one level deeper or use
    fold-specific directory names.  If the requested root contains no
    ``classification_cases.csv`` files, search only its immediate parent study
    directory.  Existing duplicate/conflict checks still guard against mixing
    incompatible result copies.
    """
    requested = os.path.abspath(os.path.expanduser(ranking_root))
    files = _classification_case_files(requested)
    if files:
        return requested, files

    parent = os.path.dirname(requested.rstrip(os.sep))
    parent_files = _classification_case_files(parent)
    if parent_files:
        candidate_roots = _candidate_result_roots(parent_files)
        warnings.warn(
            "No classification_cases.csv found under the requested --ranking_root "
            f"{requested!r}; using its immediate parent study directory {parent!r} "
            f"instead. Discovered {len(parent_files)} case files across "
            f"{len(candidate_roots)} candidate result roots. "
            "The normal duplicate/conflict checks will remain active."
        )
        return parent, parent_files

    diagnostic_base = parent if os.path.isdir(parent) else os.path.dirname(parent)
    cmd = f"find {diagnostic_base!r} -type f -name classification_cases.csv -print | head -100"
    raise FileNotFoundError(
        "No classification_cases.csv found under the requested ranking root or "
        "its immediate parent.\n"
        f"Requested: {requested}\n"
        f"Checked parent: {parent}\n"
        "Locate the real ranking output with:\n"
        f"  {cmd}\n"
        "Then pass either the common parent of those files or the corresponding "
        "ranking-classification result directory as --ranking_root."
    )


def load_classification_cases(
    ranking_root: str,
    localization_mode: str,
    true_error_percentile: float,
) -> pd.DataFrame:
    resolved_root, files = resolve_ranking_root(ranking_root)
    print(f"Resolved ranking root: {resolved_root}")
    print(f"Found {len(files)} classification_cases.csv files")

    chunks = []
    for f in files:
        d = pd.read_csv(f)
        needed = {"patient_id", "roi", "localization_mode", "qa_metric", "true_metric"}
        if not needed.issubset(d.columns):
            warnings.warn(f"Skipping incompatible {f}: missing {sorted(needed-set(d.columns))}")
            continue
        d = d[d["localization_mode"].astype(str) == localization_mode].copy()
        if "true_error_percentile" in d.columns:
            d = d[np.isclose(pd.to_numeric(d["true_error_percentile"], errors="coerce"), true_error_percentile)].copy()
        d["source_file"] = f
        chunks.append(d)

    if not chunks:
        raise ValueError("No compatible classification cases remained after filtering.")

    all_rows = pd.concat(chunks, ignore_index=True)
    all_rows["patient_id"] = all_rows["patient_id"].map(_norm_patient_id)
    all_rows["roi"] = all_rows["roi"].astype(str).str.strip()
    if "shape_model_split" not in all_rows.columns:
        all_rows["shape_model_split"] = "unknown"
    all_rows["outer_fold"] = all_rows["shape_model_split"].astype(str)

    primary = []
    for metric, spec in PRIMARY_METRICS.items():
        q = all_rows[(all_rows["qa_metric"].astype(str) == metric) &
                     (all_rows["true_metric"].astype(str) == metric)].copy()
        if q.empty:
            continue
        for c in [spec["qa_col"], spec["true_col"]]:
            if c not in q.columns:
                raise ValueError(f"Expected column {c!r} for corresponding {metric} rows.")
        q["metric"] = metric
        q["qa_score"] = pd.to_numeric(q[spec["qa_col"]], errors="coerce")
        q["true_error"] = pd.to_numeric(q[spec["true_col"]], errors="coerce")
        if "qa_threshold_from_development" in q.columns:
            q["existing_dev_threshold"] = pd.to_numeric(q["qa_threshold_from_development"], errors="coerce")
        else:
            q["existing_dev_threshold"] = np.nan
        primary.append(q)

    if not primary:
        raise ValueError("No corresponding Dice→Dice or HD95→HD95 rows were found.")

    cases = pd.concat(primary, ignore_index=True)
    cases = cases[np.isfinite(cases["qa_score"]) & np.isfinite(cases["true_error"])].copy()

    # Recursive roots can contain repeated copies of the same analysis output.
    # Exact duplicates are safe to drop; conflicting duplicates are fatal.
    key = ["patient_id", "roi", "outer_fold", "metric"]
    dup = cases[cases.duplicated(key, keep=False)].copy()
    if not dup.empty:
        consistency = dup.groupby(key).agg(
            n_qa=("qa_score", "nunique"),
            n_true=("true_error", "nunique"),
            n_thr=("existing_dev_threshold", "nunique"),
        ).reset_index()
        bad = consistency[(consistency["n_qa"] > 1) | (consistency["n_true"] > 1) | (consistency["n_thr"] > 1)]
        if not bad.empty:
            raise ValueError(
                "Conflicting duplicate NSR-QA case rows were found for patient/ROI/fold/metric. "
                "Avoid pointing --ranking_root at overlapping result copies. First conflicts:\n" +
                bad.head(20).to_string(index=False)
            )
        cases = cases.sort_values("source_file").drop_duplicates(key, keep="first")

    cases["analysis_scope"] = "oof"
    cases["data_split"] = cases["shape_model_split"].astype(str)
    return cases.reset_index(drop=True)


def _fold_number_from_split(x) -> Optional[int]:
    m = re.search(r"(\d+)$", str(x).strip())
    return int(m.group(1)) if m else None


def _metrics_case_files(root: str, localization_mode: str) -> List[str]:
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        return []
    target = f"{localization_mode}_metrics.csv"
    return sorted(glob.glob(os.path.join(root, "**", target), recursive=True))


def resolve_metrics_root(metrics_root: Optional[str], ranking_root: str, localization_mode: str) -> Tuple[str, List[str]]:
    """Resolve upstream NSR-QA metrics used by exploratory whole-data analysis."""
    candidates: List[str] = []
    if metrics_root:
        candidates.append(os.path.abspath(os.path.expanduser(metrics_root)))

    requested_rank = os.path.abspath(os.path.expanduser(ranking_root))
    probes = [requested_rank, os.path.dirname(requested_rank)]
    # Common study layouts place ranking outputs and actual_autoseg_regression
    # under the same study root. Search only a few nearby deterministic paths.
    for base in list(probes):
        candidates.extend([
            os.path.join(base, "actual_autoseg_regression", "metrics"),
            os.path.join(base, "metrics"),
        ])
        parent = os.path.dirname(base.rstrip(os.sep))
        candidates.extend([
            os.path.join(parent, "actual_autoseg_regression", "metrics"),
            os.path.join(parent, "metrics"),
        ])

    seen = set()
    for c in candidates:
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        files = _metrics_case_files(c, localization_mode)
        if files:
            return c, files

    checked = "\n  ".join(sorted(seen))
    raise FileNotFoundError(
        f"No {localization_mode}_metrics.csv files found for --analysis_scope whole. "
        "Pass --metrics_root explicitly. Checked:\n  " + checked
    )


def _existing_threshold_map(
    ranking_root: str,
    localization_mode: str,
    true_error_percentile: float,
) -> pd.DataFrame:
    """Return one existing development threshold per ROI/metric/fold when available."""
    try:
        c = load_classification_cases(ranking_root, localization_mode, true_error_percentile)
    except Exception as exc:
        warnings.warn(f"Could not load existing ranking thresholds for whole-data mode: {exc}")
        return pd.DataFrame(columns=["roi", "metric", "fold_number", "existing_dev_threshold"])
    c = c.copy()
    c["fold_number"] = c["shape_model_split"].map(_fold_number_from_split)
    rows = []
    for (roi, metric, fold), g in c.groupby(["roi", "metric", "fold_number"], dropna=False):
        vals = pd.to_numeric(g["existing_dev_threshold"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append({
            "roi": roi, "metric": metric, "fold_number": fold,
            "existing_dev_threshold": float(vals.median()),
        })
    return pd.DataFrame(rows)


def load_whole_dataset_cases(
    metrics_root: Optional[str],
    ranking_root: str,
    localization_mode: str,
    true_error_percentile: float,
    whole_fold: int,
) -> pd.DataFrame:
    """Load one fold's training+validation+testing cohort exactly once.

    The upstream ``*_metrics.csv`` files already contain the true contour error
    and NSR-QA disagreement for all three splits.  Restricting to one fold avoids
    counting the same patient multiple times across outer-fold training sets.
    """
    resolved_root, files = resolve_metrics_root(metrics_root, ranking_root, localization_mode)
    print(f"Resolved whole-data metrics root: {resolved_root}")
    print(f"Found {len(files)} {localization_mode}_metrics.csv files")

    threshold_map = _existing_threshold_map(ranking_root, localization_mode, true_error_percentile)
    chunks = []
    required = {"patient_id", "roi", "shape_model_split"}
    metric_cols = {spec["qa_col"] for spec in PRIMARY_METRICS.values()} | {spec["true_col"] for spec in PRIMARY_METRICS.values()}
    for f in files:
        d = pd.read_csv(f)
        if not required.issubset(d.columns) or not metric_cols.issubset(d.columns):
            warnings.warn(f"Skipping incompatible whole-data metrics file {f}")
            continue
        if "localization_mode" in d.columns:
            d = d[d["localization_mode"].astype(str) == localization_mode].copy()
        d["fold_number"] = d["shape_model_split"].map(_fold_number_from_split)
        d = d[d["fold_number"] == int(whole_fold)].copy()
        if d.empty:
            continue
        d["source_file"] = f
        chunks.append(d)

    if not chunks:
        raise ValueError(
            f"No training/validation/testing rows for fold {whole_fold} remained under {resolved_root}."
        )
    raw = pd.concat(chunks, ignore_index=True)
    raw["patient_id"] = raw["patient_id"].map(_norm_patient_id)
    raw["roi"] = raw["roi"].astype(str).str.strip()
    raw["data_split"] = raw["shape_model_split"].astype(str)
    raw["outer_fold"] = f"fold{int(whole_fold)}"

    expanded = []
    for metric, spec in PRIMARY_METRICS.items():
        q = raw.copy()
        q["metric"] = metric
        q["qa_score"] = pd.to_numeric(q[spec["qa_col"]], errors="coerce")
        q["true_error"] = pd.to_numeric(q[spec["true_col"]], errors="coerce")
        q["existing_dev_threshold"] = np.nan
        if not threshold_map.empty:
            tm = threshold_map[(threshold_map["metric"] == metric) &
                               (threshold_map["fold_number"] == int(whole_fold))][["roi", "existing_dev_threshold"]]
            if not tm.empty:
                q = q.drop(columns=["existing_dev_threshold"]).merge(tm, on="roi", how="left", validate="many_to_one")
        expanded.append(q)

    cases = pd.concat(expanded, ignore_index=True)
    cases = cases[np.isfinite(cases["qa_score"]) & np.isfinite(cases["true_error"])].copy()
    cases["analysis_scope"] = "whole"

    # A patient must occur only once per ROI/split/metric in the selected fold.
    # Recursive roots can contain copied outputs; exact copies are collapsed,
    # conflicting copies stop the analysis rather than silently averaging.
    key = ["patient_id", "roi", "data_split", "metric"]
    dup = cases[cases.duplicated(key, keep=False)].copy()
    if not dup.empty:
        consistency = dup.groupby(key).agg(
            n_qa=("qa_score", "nunique"),
            n_true=("true_error", "nunique"),
        ).reset_index()
        bad = consistency[(consistency["n_qa"] > 1) | (consistency["n_true"] > 1)]
        if not bad.empty:
            raise ValueError(
                "Conflicting duplicate whole-data NSR-QA rows were found. Point --metrics_root at one "
                "specific study/fold result tree. First conflicts:\n" + bad.head(20).to_string(index=False)
            )
        cases = cases.sort_values("source_file").drop_duplicates(key, keep="first")

    return cases.reset_index(drop=True)


def attach_domains(cases: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    m = cases.merge(assignments, on=["patient_id", "roi"], how="left", validate="many_to_one")
    m["domain"] = m["domain"].fillna("Normal")
    m["domain_conflict"] = m["domain"].eq("CONFLICT")
    # Only month2/month3 were domain-reviewed. Override both recorded labels
    # and the usual unrecorded=>Normal fallback for month1 and for patients
    # outside all three declared month periods.
    return _apply_domain_review_scope(m)


def assign_operating_thresholds(
    cases: pd.DataFrame,
    target_specificity: float,
    min_calibration_n: int,
    analysis_scope: str = "oof",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign QA operating thresholds with leakage provenance recorded.

    OOF scope:
      For fold k, use Normal cases from every other outer fold for the same
      ROI/metric. If this is impossible or sparse, use the existing development-
      derived threshold already frozen by the ranking/classification analysis.

    Whole scope:
      Prefer the same existing frozen development threshold. If it is unavailable,
      calibrate on Normal training+validation cases from the selected fold. This
      fallback is explicitly leakage-exposed and is labeled as such in the audit.
    """
    out = cases.copy()
    out["operating_threshold"] = np.nan
    out["threshold_source"] = ""
    out["threshold_calibration_n"] = 0
    out["threshold_calibration_specificity"] = np.nan

    audit = []
    folds_global = sorted(out["outer_fold"].dropna().astype(str).unique())
    for (roi, metric, fold), idx in out.groupby(["roi", "metric", "outer_fold"]).groups.items():
        rowset = out.loc[idx]
        threshold = np.nan
        source = ""
        cal_n = 0
        cal_spec = np.nan

        if analysis_scope == "oof":
            cal = out[(out["roi"] == roi) &
                      (out["metric"] == metric) &
                      (out["outer_fold"].astype(str) != str(fold)) &
                      (out["domain"] == "Normal") &
                      (~out["domain_conflict"])].copy()
            cal_n = len(cal)
            if len(folds_global) >= 2 and cal_n >= min_calibration_n:
                threshold = specificity_threshold(cal["qa_score"], target_specificity)
                cal_spec = empirical_specificity(cal["qa_score"], threshold)
                source = "crossfit_other_folds_normal"
            else:
                vals = pd.to_numeric(rowset["existing_dev_threshold"], errors="coerce").dropna()
                if len(vals):
                    threshold = float(vals.median())
                    source = "existing_development_threshold_fallback"
                else:
                    source = "unavailable"
        else:
            # Whole-data analysis is exploratory. Keep the already-frozen
            # development threshold whenever the ranking analysis provides it.
            vals = pd.to_numeric(rowset["existing_dev_threshold"], errors="coerce").dropna()
            if len(vals):
                threshold = float(vals.median())
                source = "existing_development_threshold_whole_data"
            else:
                split = rowset.get("data_split", pd.Series("", index=rowset.index)).astype(str)
                is_dev = split.str.match(r"^(training|validation)\d+$")
                cal = rowset[is_dev &
                             (rowset["domain"] == "Normal") &
                             (~rowset["domain_conflict"])].copy()
                cal_n = len(cal)
                if cal_n >= min_calibration_n:
                    threshold = specificity_threshold(cal["qa_score"], target_specificity)
                    cal_spec = empirical_specificity(cal["qa_score"], threshold)
                    source = "whole_data_normal_development_fallback_LEAKAGE_EXPOSED"
                else:
                    source = "unavailable"

        out.loc[idx, "operating_threshold"] = threshold
        out.loc[idx, "threshold_source"] = source
        out.loc[idx, "threshold_calibration_n"] = cal_n
        out.loc[idx, "threshold_calibration_specificity"] = cal_spec
        audit.append({
            "analysis_scope": analysis_scope,
            "roi": roi,
            "metric": metric,
            "outer_fold": fold,
            "threshold": threshold,
            "threshold_source": source,
            "calibration_normal_n": cal_n,
            "calibration_empirical_specificity": cal_spec,
            "target_specificity": target_specificity,
        })

    finite_thr = np.isfinite(out["operating_threshold"].to_numpy(float))
    out["qa_alert"] = np.where(
        finite_thr,
        (out["qa_score"].to_numpy(float) >= out["operating_threshold"].to_numpy(float)).astype(float),
        np.nan,
    )
    return out, pd.DataFrame(audit)

def domain_detection_table(cases: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    use = cases[~cases["domain_conflict"] & cases["qa_alert"].notna()].copy()
    for (roi, metric, domain), g in use.groupby(["roi", "metric", "domain"]):
        rate = float(g["qa_alert"].astype(float).mean())
        lo, hi, nb = bootstrap_rate_cluster(
            g, value_col="qa_alert", n_boot=n_boot, seed=seed
        )
        rows.append({
            "roi": roi,
            "metric": metric,
            "domain": domain,
            "n": len(g),
            "n_patients": g["patient_id"].nunique(),
            "n_alert": int(g["qa_alert"].astype(bool).sum()),
            "alert_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "bootstrap_valid": nb,
        })
    return pd.DataFrame(rows)


def macro_detection_table(det: pd.DataFrame, min_cell_n: int) -> pd.DataFrame:
    rows = []
    if det.empty:
        return pd.DataFrame()
    for (metric, domain), g in det.groupby(["metric", "domain"]):
        eligible = g[g["n"] >= min_cell_n].copy()
        if eligible.empty:
            macro = np.nan
            n_roi = 0
        else:
            macro = float(eligible["alert_rate"].mean())
            n_roi = len(eligible)
        rows.append({
            "metric": metric,
            "domain": domain,
            "macro_alert_rate": macro,
            "n_roi_in_macro": n_roi,
            "total_n": int(g["n"].sum()),
            "n_roi_observed": len(g),
        })
    return pd.DataFrame(rows)


def domain_correlations(cases: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    use = cases[~cases["domain_conflict"]].copy()
    for (roi, metric, domain), g in use.groupby(["roi", "metric", "domain"]):
        rho, p = safe_spearman(g["qa_score"], g["true_error"])
        lo, hi, nb = bootstrap_spearman_cluster(
            g, x_col="qa_score", y_col="true_error", n_boot=n_boot, seed=seed
        )
        rows.append({
            "roi": roi,
            "metric": metric,
            "domain": domain,
            "n": len(g),
            "n_patients": g["patient_id"].nunique(),
            "spearman_rho": rho,
            "spearman_p": p,
            "ci95_low": lo,
            "ci95_high": hi,
            "bootstrap_valid": nb,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["spearman_q_fdr"] = bh_fdr(out["spearman_p"].to_numpy())
    return out


def _domain_sort_key(x: str) -> int:
    try:
        return DOMAIN_ORDER.index(x)
    except ValueError:
        return len(DOMAIN_ORDER)

def summarize_roi_support(
    cases: pd.DataFrame,
    min_abnormal_n: int,
    min_domain_n: int,
    min_supported_domains: int,
    secondary_single_domain_n: int,
) -> pd.DataFrame:
    """Classify ROI support for domain-specific analysis."""
    d = cases[~cases["domain_conflict"]].drop_duplicates(
        ["patient_id", "roi", "outer_fold"]
    ).copy()
    rows = []
    for roi, g in d.groupby("roi"):
        counts = g["domain"].value_counts().to_dict()
        failure_counts = {
            dom: int(counts.get(dom, 0))
            for dom in FAILURE_DOMAINS if dom != "Normal"
        }
        abnormal_total = int(sum(failure_counts.values()))
        supported = [dom for dom, n in failure_counts.items() if n >= min_domain_n]
        largest_domain = max(failure_counts, key=failure_counts.get) if failure_counts else ""
        largest_n = failure_counts.get(largest_domain, 0) if largest_domain else 0

        if abnormal_total >= min_abnormal_n and len(supported) >= min_supported_domains:
            tier = "primary_cross_domain"
            reason = f"{abnormal_total} abnormal cases; {len(supported)} domains have n>={min_domain_n}"
        elif abnormal_total >= min_abnormal_n and largest_n >= secondary_single_domain_n:
            tier = "secondary_single_domain"
            reason = (
                f"{abnormal_total} abnormal cases but only {len(supported)} domains have n>={min_domain_n}; "
                f"strongest={largest_domain} (n={largest_n})"
            )
        else:
            tier = "descriptive_only"
            reason = f"{abnormal_total} abnormal cases; {len(supported)} domains have n>={min_domain_n}"

        row = {
            "roi": roi,
            "tier": tier,
            "reason": reason,
            "n_total": int(len(g)),
            "n_normal": int(counts.get("Normal", 0)),
            "n_abnormal_mechanism": abnormal_total,
            "n_supported_abnormal_domains": int(len(supported)),
            "supported_abnormal_domains": "|".join(supported),
            "largest_abnormal_domain": largest_domain,
            "largest_abnormal_domain_n": int(largest_n),
        }
        for dom in ["I", "AV", "MI", "AC", "H", "CA", "NDI", "LDI"]:
            row[f"n_{dom}"] = int(counts.get(dom, 0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["tier", "n_abnormal_mechanism", "roi"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def add_roi_model_values(cases: pd.DataFrame) -> pd.DataFrame:
    """Add per-ROI model values without cross-ROI normalization."""
    out = cases.copy()
    out["qa_model_value"] = np.nan
    out["true_model_value"] = np.nan
    for metric, spec in PRIMARY_METRICS.items():
        idx = out["metric"] == metric
        if spec["transform"] == "log1p":
            out.loc[idx, "qa_model_value"] = np.log1p(
                np.clip(pd.to_numeric(out.loc[idx, "qa_score"], errors="coerce"), 0, None)
            )
            out.loc[idx, "true_model_value"] = np.log1p(
                np.clip(pd.to_numeric(out.loc[idx, "true_error"], errors="coerce"), 0, None)
            )
        else:
            out.loc[idx, "qa_model_value"] = pd.to_numeric(out.loc[idx, "qa_score"], errors="coerce")
            out.loc[idx, "true_model_value"] = pd.to_numeric(out.loc[idx, "true_error"], errors="coerce")
    return out


def _fit_roi_cluster_robust_ols(
    d: pd.DataFrame,
    eligible_domains: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fit one ROI: QA response ~ true error * domain, clustered by patient."""
    eligible_domains = ["Normal"] + [x for x in eligible_domains if x != "Normal"]
    cols: List[np.ndarray] = [
        np.ones(len(d), dtype=float),
        d["true_model_value"].to_numpy(float),
    ]
    names = ["Intercept", "true_error"]
    domain_s = d["domain"].astype(str)
    true_x = d["true_model_value"].to_numpy(float)
    for dom in eligible_domains:
        if dom == "Normal":
            continue
        ind = (domain_s == dom).to_numpy(float)
        cols.append(ind)
        names.append(f"domain[T.{dom}]")
        cols.append(ind * true_x)
        names.append(f"true_error:domain[T.{dom}]")

    X = np.column_stack(cols)
    y = d["qa_model_value"].to_numpy(float)
    n, k = X.shape
    rank = int(np.linalg.matrix_rank(X))
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    cluster = d["patient_id"].astype(str).to_numpy()
    clusters = np.unique(cluster)
    g = len(clusters)
    if g < 2:
        raise ValueError("Need at least two patient clusters.")

    meat = np.zeros((k, k), dtype=float)
    for c in clusters:
        idx = cluster == c
        score = X[idx].T @ resid[idx]
        meat += np.outer(score, score)
    correction = (g / (g - 1.0)) * ((n - 1.0) / max(n - k, 1.0))
    cov = correction * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = beta / se
    df = max(g - 1, 1)
    pvals = 2.0 * stats.t.sf(np.abs(tval), df=df)
    crit = float(stats.t.ppf(0.975, df=df))
    lo = beta - crit * se
    hi = beta + crit * se
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return pd.DataFrame({
        "term": names,
        "coef": beta,
        "std_err_cluster": se,
        "t_cluster": tval,
        "p_value": pvals,
        "ci95_low": lo,
        "ci95_high": hi,
    }), {
        "n": n,
        "n_patients": g,
        "n_parameters": k,
        "design_rank": rank,
        "cluster_df": df,
        "r2": r2,
    }


def fit_roi_domain_interaction_models(
    cases: pd.DataFrame,
    selected_rois: Sequence[str],
    min_domain_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit severity-adjusted failure-domain models separately by ROI."""
    coef_tables = []
    audits = []
    for roi in selected_rois:
        for metric in PRIMARY_METRICS:
            d = cases[
                (cases["roi"] == roi)
                & (cases["metric"] == metric)
                & (~cases["domain_conflict"])
                & (cases["domain"].isin(FAILURE_DOMAINS))
            ].copy()
            d = d[
                np.isfinite(pd.to_numeric(d["qa_model_value"], errors="coerce"))
                & np.isfinite(pd.to_numeric(d["true_model_value"], errors="coerce"))
            ].copy()
            counts = d["domain"].value_counts()
            eligible = ["Normal"] + [
                dom for dom in FAILURE_DOMAINS
                if dom != "Normal" and counts.get(dom, 0) >= min_domain_n
            ]
            eligible = list(dict.fromkeys(eligible))
            d = d[d["domain"].isin(eligible)].copy()
            if counts.get("Normal", 0) < min_domain_n or len(eligible) < 2:
                audits.append({
                    "roi": roi, "metric": metric, "status": "skipped",
                    "reason": "insufficient Normal or no abnormal domain meeting minimum n",
                    "domains_in_model": "|".join(eligible),
                })
                continue
            try:
                tab, meta = _fit_roi_cluster_robust_ols(d, eligible)
            except Exception as exc:
                audits.append({
                    "roi": roi, "metric": metric, "status": "failed",
                    "reason": str(exc), "domains_in_model": "|".join(eligible),
                })
                continue
            tab["roi"] = roi
            tab["metric"] = metric
            tab["domains_in_model"] = "|".join(eligible)
            tab["n"] = int(meta["n"])
            tab["n_patients"] = int(meta["n_patients"])
            coef_tables.append(tab)
            audits.append({
                "roi": roi, "metric": metric, "status": "ok", "reason": "",
                "domains_in_model": "|".join(eligible),
                "n": int(meta["n"]), "n_patients": int(meta["n_patients"]),
                "r2": float(meta["r2"]), "n_parameters": int(meta["n_parameters"]),
                "design_rank": int(meta["design_rank"]),
            })
    coef = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
    if not coef.empty:
        interaction = coef["term"].str.startswith("true_error:domain[T.")
        coef["q_fdr_interactions"] = np.nan
        if interaction.any():
            coef.loc[interaction, "q_fdr_interactions"] = bh_fdr(
                coef.loc[interaction, "p_value"].to_numpy()
            )
    return coef, pd.DataFrame(audits)


def plot_roi_true_vs_qa(cases: pd.DataFrame, roi: str, metric: str, path: str) -> None:
    d = cases[(cases["roi"] == roi) & (cases["metric"] == metric) & (~cases["domain_conflict"])].copy()
    d = d[np.isfinite(d["qa_score"]) & np.isfinite(d["true_error"])]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for dom in [x for x in DOMAIN_ORDER if x in set(d["domain"])]:
        g = d[d["domain"] == dom]
        ax.scatter(
            g["true_error"], g["qa_score"],
            s=16 if dom == "Normal" else 28,
            alpha=0.28 if dom == "Normal" else 0.75,
            label=f"{dom} (n={len(g)})",
        )
    ax.set_xlabel(f"True contour error: {PRIMARY_METRICS[metric]['label']}")
    ax.set_ylabel(f"NSR-QA score: {PRIMARY_METRICS[metric]['label']}")
    ax.set_title(f"{roi}: true error vs NSR-QA response")
    if metric == "hd95":
        vals = np.concatenate([d["true_error"].to_numpy(float), d["qa_score"].to_numpy(float)])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) and vals.max() / max(vals.min(), 1e-6) > 20:
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roi_domain_box(cases: pd.DataFrame, roi: str, metric: str, path: str) -> None:
    d = cases[(cases["roi"] == roi) & (cases["metric"] == metric) & (~cases["domain_conflict"])].copy()
    domains = [x for x in DOMAIN_ORDER if x in set(d["domain"])]
    if not domains:
        return
    vals = [pd.to_numeric(d.loc[d["domain"] == dom, "qa_score"], errors="coerce").dropna().to_numpy(float) for dom in domains]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.boxplot(vals, positions=np.arange(1, len(domains) + 1), showfliers=False)
    rng = np.random.default_rng(88)
    for i, arr in enumerate(vals, start=1):
        if len(arr):
            ax.scatter(rng.normal(i, 0.045, len(arr)), arr, s=12, alpha=0.42)
    ax.set_xticks(np.arange(1, len(domains) + 1), [f"{dom}\n(n={len(arr)})" for dom, arr in zip(domains, vals)])
    ax.set_ylabel(f"NSR-QA score: {PRIMARY_METRICS[metric]['label']}")
    ax.set_title(f"{roi}: NSR-QA response by domain")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roi_detection(det: pd.DataFrame, roi: str, metric: str, path: str) -> None:
    d = det[(det["roi"] == roi) & (det["metric"] == metric)].copy()
    if d.empty:
        return
    domains = [x for x in DOMAIN_ORDER if x in set(d["domain"])]
    rows = [d[d["domain"] == dom].iloc[0] for dom in domains]
    y = np.array([float(r["alert_rate"]) for r in rows])
    lo = np.array([float(r["ci95_low"]) for r in rows])
    hi = np.array([float(r["ci95_high"]) for r in rows])
    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.bar(x, y, yerr=np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)]), capsize=3)
    ax.set_xticks(x, [f"{dom}\n(n={int(r['n'])})" for dom, r in zip(domains, rows)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("NSR-QA alert rate")
    ax.set_title(f"{roi}: alert/detection rate — {PRIMARY_METRICS[metric]['label']}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)



def plot_domain_counts(cases: pd.DataFrame, path: str) -> None:
    d = cases[~cases["domain_conflict"]].drop_duplicates(["patient_id", "roi", "outer_fold"]).copy()
    tab = pd.crosstab(d["roi"], d["domain"])
    cols = sorted(tab.columns, key=_domain_sort_key)
    tab = tab.reindex(columns=cols, fill_value=0)
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cols) + 3), max(6, 0.38 * len(tab) + 2)))
    im = ax.imshow(tab.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(tab.index)), tab.index)
    ax.set_title("Available NSR-QA cases by ROI and domain")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, str(int(tab.iat[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Case count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_float(x, digits=3) -> str:
    return "NA" if pd.isna(x) or not np.isfinite(float(x)) else f"{float(x):.{digits}f}"


def write_report(
    path: str,
    cases: pd.DataFrame,
    det: pd.DataFrame,
    corr: pd.DataFrame,
    threshold_audit: pd.DataFrame,
    roi_support: pd.DataFrame,
    selected_rois: Sequence[str],
    roi_coef: pd.DataFrame,
    roi_audit: pd.DataFrame,
    conflicts: pd.DataFrame,
    multi_study: pd.DataFrame,
    spinal_wide: pd.DataFrame,
    clinical_metrics_csv: str,
    clinical_metrics_duplicate_policy: str,
    min_cell_n: int,
    target_specificity: float,
    analysis_scope: str,
    whole_fold: int,
) -> None:
    lines: List[str] = []
    title_suffix = "exploratory whole-data" if analysis_scope == "whole" else "held-out / OOF"
    lines.append(f"# ROI-specific NSR-QA domain performance ({title_suffix})")
    lines.append("")
    if analysis_scope == "whole":
        lines.append(
            "**Exploratory/leakage warning.** This report uses one fold's complete training+validation+testing cohort. "
            "Training and validation cases were exposed to model fitting and/or model selection, so their performance can be optimistic. "
            "Use the OOF/test report for unbiased performance claims."
        )
        lines.append("")

    lines.append("## Clinical metric provenance")
    lines.append("")
    lines.append(
        f"Clinical geometric truth is read from the original Radformation `metrics.csv`: `{clinical_metrics_csv}`. "
        "The NSR-QA score still comes from the NSR-QA ranking/upstream result tables. "
        "For Dice, true error is `1 - dice`; for HD95, the `haus` value in `metrics.csv` is used directly. "
        "The former resampled/localized pipeline value is retained as `pipeline_true_error` in `cases_with_domains.csv` for audit."
    )
    lines.append("")
    lines.append(
        f"Because current NSR-QA case tables do not carry study UID, ambiguous patient/ROI metrics spanning multiple studies use policy "
        f"**{clinical_metrics_duplicate_policy}**. Ambiguities are listed in `clinical_metrics_join_audit.csv`."
    )
    lines.append("")
    lines.append(
        "**Domain-review cohort rule.** Only month2 and month3 are domain-review eligible. Month1 patients and patients outside "
        "all three declared month periods are labeled `Unknown` regardless of whether a domain label appears in the source CSV. "
        "`Unknown` is retained in descriptive tables/plots but is excluded from Normal calibration and abnormal-domain inferential models."
    )
    lines.append("")

    lines.append("## Why ROI-specific analysis")
    lines.append("")
    lines.append(
        "This report does **not** pool or ROI-normalize NSR-QA scores. Dice- and HD95-based QA behavior can differ substantially by anatomy, "
        "and the abnormal-domain sample sizes are highly uneven across ROIs. Each selected ROI is therefore analyzed on its native scale."
    )
    lines.append("")
    lines.append(
        "The default automatic selection requires enough abnormal cases overall and at least two failure-mechanism domains with adequate sample size. "
        "ROIs dominated by a single abnormal domain are retained in the support table but are not treated as strong cross-domain evidence."
    )
    lines.append("")

    lines.append("## ROI support and selection")
    lines.append("")
    if roi_support.empty:
        lines.append("No ROI support table was available.")
    else:
        cols = [
            "roi", "tier", "n_total", "n_normal", "n_abnormal_mechanism",
            "n_I", "n_AV", "n_MI", "n_AC", "n_H", "n_CA", "n_NDI", "n_LDI",
            "supported_abnormal_domains", "reason",
        ]
        cols = [c for c in cols if c in roi_support.columns]
        lines.append(roi_support[cols].to_markdown(index=False))
    lines.append("")
    lines.append("**Main ROI-specific report set:** " + (", ".join(selected_rois) if selected_rois else "none"))
    lines.append("")
    lines.append(
        "Interpret `H` separately from algorithm/domain-shift mechanisms: `H` means the clinical/reference contour itself was considered suboptimal or off-guideline, "
        "so disagreement there is not equivalent to an auto-contour failure."
    )
    lines.append("")

    lines.append("## Domain definitions")
    lines.append("")
    for dom in DOMAIN_ORDER:
        lines.append(f"- **{dom}:** {DOMAIN_DESCRIPTIONS[dom]}")
    lines.append("")

    lines.append("## Overall coverage")
    lines.append("")
    lines.append("![Domain counts](figures/domain_counts_heatmap.png)")
    lines.append("")
    lines.append(
        "**Description.** Counts are one inference per patient/ROI. Use this figure before interpreting any ROI/domain performance result. "
        f"Cells with n < {min_cell_n} are descriptive only."
    )
    lines.append("")

    for roi in selected_rois:
        lines.append(f"# ROI: {roi}")
        lines.append("")
        support_row = roi_support[roi_support["roi"] == roi] if not roi_support.empty else pd.DataFrame()
        if not support_row.empty:
            r = support_row.iloc[0]
            lines.append(
                f"Support: **{int(r['n_abnormal_mechanism'])} abnormal mechanism cases**; "
                f"supported abnormal domains (n≥configured minimum): **{r['supported_abnormal_domains'] or 'none'}**."
            )
            lines.append("")

        for metric in PRIMARY_METRICS:
            label = PRIMARY_METRICS[metric]["label"]
            lines.append(f"## {roi}: {label}")
            lines.append("")
            lines.append(f"![Scatter](figures/{roi}_true_vs_qa_{metric}.png)")
            lines.append("")
            lines.append(
                "**Scatter description.** Each point is one contour inference. The x-axis is true auto-contour error relative to the approved contour; "
                "the y-axis is the NSR-QA input-versus-reconstruction discrepancy using the same metric. No cross-ROI normalization is applied. "
                "For HD95, a symmetric-log display may be used only when the dynamic range is large; values remain in mm."
            )
            lines.append("")
            lines.append(f"![Domain response](figures/{roi}_qa_by_domain_{metric}.png)")
            lines.append("")
            lines.append(
                "**Distribution description.** Raw NSR-QA score by domain for this ROI. Individual observations are shown because small-domain sample sizes make smooth density summaries misleading."
            )
            lines.append("")
            lines.append(f"![Detection](figures/{roi}_detection_{metric}.png)")
            lines.append("")
            lines.append(
                f"**Detection description.** Alert/detection rate at the fixed ROI-specific operating threshold (target Normal specificity {target_specificity:.0%}), "
                "with patient-cluster bootstrap 95% confidence intervals. For failure domains, larger is better detection; for Normal/CA/NDI it is review burden."
            )
            lines.append("")

            c = corr[(corr["roi"] == roi) & (corr["metric"] == metric)].copy()
            if not c.empty:
                c = c.sort_values("domain", key=lambda s: s.map(_domain_sort_key))
                show_cols = ["domain", "n", "spearman_rho", "ci95_low", "ci95_high", "spearman_q_fdr"]
                lines.append("**Within-domain severity tracking:**")
                lines.append("")
                lines.append(c[show_cols].to_markdown(index=False, floatfmt=".3f"))
                lines.append("")

        audit = roi_audit[roi_audit["roi"] == roi] if not roi_audit.empty else pd.DataFrame()
        coef = roi_coef[roi_coef["roi"] == roi] if not roi_coef.empty else pd.DataFrame()
        lines.append(f"## {roi}: severity-adjusted domain response")
        lines.append("")
        lines.append(
            "For each metric separately, the model is `NSR_QA_response ~ true_error × domain`, using patient-cluster robust standard errors. "
            "Dice uses 1-Dice directly. HD95 uses log1p(HD95) in the regression for numerical stability, but plots remain on the ROI's native units. "
            "Normal is the reference domain. A negative true-error×domain interaction indicates relative under-response as true error increases."
        )
        lines.append("")
        if not audit.empty:
            lines.append("Model audit:")
            lines.append("")
            lines.append(audit.to_markdown(index=False, floatfmt=".3f"))
            lines.append("")
        if coef.empty:
            lines.append("No domain interaction was estimable for this ROI under the sample-size rule.")
            lines.append("")
        else:
            interactions = coef[coef["term"].str.startswith("true_error:domain[T.")].copy()
            if interactions.empty:
                lines.append("No interaction terms were estimable.")
            else:
                cols = ["metric", "term", "coef", "ci95_low", "ci95_high", "p_value", "q_fdr_interactions", "n", "n_patients"]
                lines.append(interactions[cols].to_markdown(index=False, floatfmt=".3f"))
            lines.append("")

    lines.append("# SpinalCord abnormal-patient clinical metrics")
    lines.append("")
    lines.append(
        "This table includes poorer-agreement SpinalCord domain cases (`I`, `AV`, `MI`, `AC`, `H`) with patient ID < 1000. "
        "Clinical metric values come from the original `metrics.csv`, not from resampled NSR-QA geometry. "
        "The QA columns come from the NSR-QA case table: `qa_1_minus_dice` is the anomaly score, "
        "`qa_input_recon_dice = 1 - qa_1_minus_dice`, and `qa_hd95_mm` is the QA Haus/HD95 score."
    )
    lines.append("")
    if spinal_wide.empty:
        lines.append("No abnormal SpinalCord domain assignments were found.")
    else:
        show_cols = [c for c in [
            "patient_id", "domain", "month", "dice", "haus",
            "qa_1_minus_dice", "qa_input_recon_dice", "qa_hd95_mm",
            "jacc", "mean_ssd", "max_ssd", "comment"
        ] if c in spinal_wide.columns]
        lines.append(spinal_wide[show_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("# Interpretation guide")
    lines.append("")
    lines.append(
        f"1. **Do not over-interpret sparse domains.** n < {min_cell_n} is descriptive only. A point estimate from one to four cases is useful for case discovery, not a stable domain-performance estimate."
    )
    lines.append(
        "2. **Prefer within-ROI comparisons.** The absolute scale and difficulty of Dice/HD95 differ by anatomy, so this report deliberately avoids pooling normalized scores across ROIs."
    )
    lines.append(
        "3. **Detection and severity tracking are different endpoints.** A domain can have a high alert rate but poor within-domain correlation if the detector recognizes large failures without ranking their severity well."
    )
    lines.append(
        "4. **Use interaction terms for the blind-spot question.** A domain-vs-Normal score difference alone can simply reflect more severe true contour errors. The true-error×domain interaction asks whether NSR-QA responds differently at comparable severity."
    )
    lines.append(
        "5. **Treat H as a reference-quality sensitivity analysis.** A high NSR-QA response in H cases may reflect disagreement with a questionable clinical reference rather than a genuine auto-contour defect."
    )
    lines.append(
        "6. **AC-heavy ROIs can still support AC-vs-Normal analysis, but not a general cross-domain claim.** Those ROIs belong in a secondary analysis unless another abnormal domain also has adequate n."
    )
    lines.append(
        "7. **Domain assignment and geometric error are not independent.** Because DSC/HD95 control-limit behavior informed case review, Normal-vs-OOD geometric differences are partly circular. The NSR-QA response conditional on true error is the more informative endpoint."
    )
    if analysis_scope == "whole":
        lines.append(
            "8. **Whole-data results are exploratory.** The larger sample size helps characterize rare domains, but train/validation performance is leakage-exposed. Confirm important patterns in OOF/test results."
        )
    lines.append("")

    lines.append("# Audit and outputs")
    lines.append("")
    lines.append(f"- Conflicting patient-ROI domain assignments excluded: **{len(conflicts)}**")
    lines.append(f"- Patient-ROI assignments spanning >1 study UID: **{len(multi_study)}**")
    lines.append("- `roi_domain_support.csv`: ROI/domain counts and automatic selection tier.")
    lines.append("- `domain_detection_by_roi.csv`: ROI/domain alert rates and bootstrap CIs.")
    lines.append("- `domain_correlations_by_roi.csv`: ROI/domain Spearman severity associations.")
    lines.append("- `roi_severity_adjusted_domain_model.csv`: per-ROI interaction coefficients.")
    lines.append("- `roi_severity_adjusted_domain_model_audit.csv`: model eligibility/status.")
    lines.append("- `threshold_audit.csv`: operating-threshold provenance.")
    lines.append("- `clinical_metrics_join_audit.csv`: duplicate/multi-study `metrics.csv` join audit.")
    lines.append("- `spinalcord_abnormal_patient_metrics.csv`: one row per abnormal SpinalCord patient/study (patient ID < 1000) with original clinical metrics plus Dice/HD95 NSR-QA scores.")
    lines.append("- `spinalcord_abnormal_patient_metrics_long.csv`: long-form patient/domain/metric/value table.")
    lines.append("- `spinalcord_abnormal_patient_metrics.md`: copyable SpinalCord abnormal-case summary table.")
    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ranking_root", required=True, help="Root containing classification_cases.csv files (searched recursively).")
    p.add_argument("--analysis_scope", choices=["oof", "whole"], default="oof",
                   help="oof=held-out classification cases; whole=one fold's training+validation+testing upstream metrics.")
    p.add_argument("--metrics_root", default=None,
                   help="Optional root containing <localization>_metrics.csv files for --analysis_scope whole. Auto-discovered near the study root if omitted.")
    p.add_argument("--whole_fold", type=int, default=1,
                   help="Fold number used for whole-data analysis; only trainingN+validationN+testingN are included so each patient is counted once.")
    p.add_argument("--domain_csv", required=True, help="OOD/domain assignment CSV.")
    p.add_argument("--clinical_metrics_csv", default="../radformation-data/data/metrics.csv",
                   help="Original Radformation metrics.csv. Its dice/haus values are authoritative for clinical geometric truth.")
    p.add_argument("--clinical_metrics_duplicate_policy", choices=["exclude", "median", "first", "error"], default="exclude",
                   help="How to handle multiple distinct metrics.csv values when NSR case tables lack study UID. 'exclude' is safest and is the default.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--site", default="thorax")
    p.add_argument("--localization_mode", default="autoseg_centered")
    p.add_argument("--true_error_percentile", type=float, default=75.0,
                   help="Select the matching classification rows; does not redefine the domain labels.")
    p.add_argument("--target_specificity", type=float, default=0.95)
    p.add_argument("--min_calibration_n", type=int, default=10,
                   help="Minimum Normal cases from other folds for cross-fitted threshold calibration.")
    p.add_argument("--min_cell_n", type=int, default=5,
                   help="Minimum domain/ROI n for inferential emphasis.")
    p.add_argument("--min_domain_model_n", type=int, default=5,
                   help="Minimum abnormal-domain n within one ROI to include in its severity-adjusted interaction model.")
    p.add_argument("--roi_selection", choices=["auto", "all", "manual"], default="auto",
                   help="auto selects only ROIs with adequate multi-domain support; manual uses --rois.")
    p.add_argument("--rois", default="",
                   help="Comma-separated ROI list used when --roi_selection manual.")
    p.add_argument("--min_roi_abnormal_n", type=int, default=15,
                   help="Minimum total failure-mechanism cases for automatic primary/secondary ROI selection.")
    p.add_argument("--min_roi_supported_domains", type=int, default=2,
                   help="Minimum number of abnormal domains with n>=--min_cell_n for primary cross-domain selection.")
    p.add_argument("--secondary_single_domain_n", type=int, default=10,
                   help="Minimum largest-domain n for an ROI to qualify as secondary single-domain evidence.")
    p.add_argument("--include_secondary_rois", action="store_true",
                   help="Also generate ROI-specific plots for secondary single-domain ROIs in auto mode.")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=88)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    assignments, conflicts, multi_study = load_domain_assignments(args.domain_csv, args.site)
    clinical_case_metrics, clinical_exact_metrics, clinical_metric_audit = load_clinical_metrics(
        args.clinical_metrics_csv, args.site, args.clinical_metrics_duplicate_policy
    )
    if args.analysis_scope == "whole":
        cases = load_whole_dataset_cases(
            args.metrics_root, args.ranking_root, args.localization_mode,
            args.true_error_percentile, args.whole_fold,
        )
    else:
        cases = load_classification_cases(args.ranking_root, args.localization_mode, args.true_error_percentile)
    cases = replace_true_error_from_clinical_metrics(cases, clinical_case_metrics)
    cases = attach_domains(cases, assignments)
    cases, threshold_audit = assign_operating_thresholds(
        cases, args.target_specificity, args.min_calibration_n, args.analysis_scope
    )

    det = domain_detection_table(cases, args.n_boot, args.seed)
    corr = domain_correlations(cases, args.n_boot, args.seed)
    cases_model = add_roi_model_values(cases)

    roi_support = summarize_roi_support(
        cases_model,
        min_abnormal_n=args.min_roi_abnormal_n,
        min_domain_n=args.min_cell_n,
        min_supported_domains=args.min_roi_supported_domains,
        secondary_single_domain_n=args.secondary_single_domain_n,
    )

    if args.roi_selection == "manual":
        selected_rois = [x.strip() for x in args.rois.split(",") if x.strip()]
        unknown = sorted(set(selected_rois) - set(cases_model["roi"].astype(str).unique()))
        if unknown:
            raise ValueError(f"Requested --rois not present in analysis cases: {unknown}")
    elif args.roi_selection == "all":
        selected_rois = sorted(cases_model["roi"].astype(str).unique())
    else:
        selected_rois = roi_support.loc[
            roi_support["tier"] == "primary_cross_domain", "roi"
        ].tolist()
        if args.include_secondary_rois:
            selected_rois += roi_support.loc[
                roi_support["tier"] == "secondary_single_domain", "roi"
            ].tolist()
        selected_rois = list(dict.fromkeys(selected_rois))

    if not selected_rois:
        warnings.warn(
            "No ROI met the automatic cross-domain support rule. "
            "Review roi_domain_support.csv or use --roi_selection manual/all."
        )

    roi_coef, roi_audit = fit_roi_domain_interaction_models(
        cases_model, selected_rois, args.min_domain_model_n
    )

    spinal_wide, spinal_long = build_spinalcord_abnormal_summary(
        args.domain_csv, clinical_exact_metrics, cases_model, args.site,
        roi="SpinalCord", max_patient_id_exclusive=1000,
    )

    cases_model.to_csv(outdir / "cases_with_domains.csv", index=False)
    clinical_metric_audit.to_csv(outdir / "clinical_metrics_join_audit.csv", index=False)
    spinal_wide.to_csv(outdir / "spinalcord_abnormal_patient_metrics.csv", index=False)
    spinal_long.to_csv(outdir / "spinalcord_abnormal_patient_metrics_long.csv", index=False)
    spinal_md_cols = [c for c in [
        "patient_id", "domain", "month", "dice", "haus",
        "qa_1_minus_dice", "qa_input_recon_dice", "qa_hd95_mm",
        "jacc", "mean_ssd", "max_ssd", "comment"
    ] if c in spinal_wide.columns]
    spinal_md = [
        "# SpinalCord abnormal-patient clinical metrics",
        "",
        "Failure-mechanism domains included: I, AV, MI, AC, H. Only patient IDs < 1000 are included.",
        "Clinical metric values come from the original Radformation `metrics.csv`; QA scores come from the NSR-QA case table.",
        "`qa_1_minus_dice` is the NSR anomaly score; `qa_input_recon_dice = 1 - qa_1_minus_dice`; `qa_hd95_mm` is the NSR QA HD95 score.",
        "",
    ]
    spinal_md.append(
        spinal_wide[spinal_md_cols].to_markdown(index=False, floatfmt=".4f")
        if not spinal_wide.empty else "No abnormal SpinalCord domain assignments were found."
    )
    (outdir / "spinalcord_abnormal_patient_metrics.md").write_text("\n".join(spinal_md) + "\n", encoding="utf-8")
    threshold_audit.to_csv(outdir / "threshold_audit.csv", index=False)
    det.to_csv(outdir / "domain_detection_by_roi.csv", index=False)
    corr.to_csv(outdir / "domain_correlations_by_roi.csv", index=False)
    roi_support.to_csv(outdir / "roi_domain_support.csv", index=False)
    conflicts.to_csv(outdir / "domain_assignment_conflicts.csv", index=False)
    multi_study.to_csv(outdir / "multi_study_domain_assignments.csv", index=False)
    roi_coef.to_csv(outdir / "roi_severity_adjusted_domain_model.csv", index=False)
    roi_audit.to_csv(outdir / "roi_severity_adjusted_domain_model_audit.csv", index=False)

    count_cases = cases_model[~cases_model["domain_conflict"]].drop_duplicates(
        ["patient_id", "roi", "outer_fold"]
    )
    pd.crosstab(count_cases["roi"], count_cases["domain"]).to_csv(outdir / "domain_counts.csv")

    plot_domain_counts(cases_model, str(figdir / "domain_counts_heatmap.png"))
    for roi in selected_rois:
        for metric in PRIMARY_METRICS:
            plot_roi_true_vs_qa(
                cases_model, roi, metric,
                str(figdir / f"{roi}_true_vs_qa_{metric}.png"),
            )
            plot_roi_domain_box(
                cases_model, roi, metric,
                str(figdir / f"{roi}_qa_by_domain_{metric}.png"),
            )
            plot_roi_detection(
                det, roi, metric,
                str(figdir / f"{roi}_detection_{metric}.png"),
            )

    write_report(
        str(outdir / "report.md"), cases_model, det, corr, threshold_audit,
        roi_support, selected_rois, roi_coef, roi_audit,
        conflicts, multi_study, spinal_wide, args.clinical_metrics_csv,
        args.clinical_metrics_duplicate_policy, args.min_cell_n, args.target_specificity,
        args.analysis_scope, args.whole_fold,
    )

    run_meta = vars(args).copy()
    run_meta.update({
        "n_case_metric_rows": len(cases_model),
        "n_unique_inferences": int(count_cases.shape[0]),
        "n_unique_patients": int(count_cases["patient_id"].nunique()),
        "n_rois": int(count_cases["roi"].nunique()),
        "selected_rois": selected_rois,
        "outer_folds": sorted(count_cases["outer_fold"].astype(str).unique().tolist()),
        "data_splits": sorted(count_cases["data_split"].astype(str).unique().tolist()) if "data_split" in count_cases.columns else [],
        "domain_join_key": ["patient_id", "roi"],
        "clinical_true_error_source": args.clinical_metrics_csv,
        "clinical_metrics_duplicate_policy": args.clinical_metrics_duplicate_policy,
        "clinical_metric_mapping": CLINICAL_METRIC_MAP,
        "spinalcord_abnormal_domains": ABNORMAL_DOMAINS,
        "domain_review_rule": "month2/month3 only; month1 and outside-three-month patients are Unknown",
        "month1_patient_ids": sorted(MONTH1_PATIENT_IDS, key=int),
        "month2_patient_ids": sorted(MONTH2_PATIENT_IDS, key=int),
        "month3_patient_ids": sorted(MONTH3_PATIENT_IDS, key=int),
        "domain_reviewed_patient_ids": sorted(DOMAIN_REVIEWED_PATIENT_IDS, key=int),
        "whole_data_is_leakage_exposed": bool(args.analysis_scope == "whole"),
        "study_uid_available_in_domain_csv_but_not_current_nsr_case_tables": True,
    })
    (outdir / "analysis_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Wrote NSR-QA domain analysis to: {outdir}")
    print(f"Report: {outdir / 'report.md'}")
    print(f"Threshold sources: {threshold_audit['threshold_source'].value_counts().to_dict()}")
    if len(conflicts):
        print(f"WARNING: excluded {len(conflicts)} conflicting patient-ROI domain assignments; see domain_assignment_conflicts.csv")
    if len(multi_study):
        print(f"WARNING: {len(multi_study)} patient-ROI assignments span >1 study UID; see multi_study_domain_assignments.csv")


if __name__ == "__main__":
    main()
