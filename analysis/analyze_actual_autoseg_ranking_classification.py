#!/usr/bin/env python3
"""Ranking and high-error classification analysis for real autosegmentations.

Primary design
--------------
* QA model: frozen NSR-QA only (metrics are read from the existing real-autoseg CSVs).
* Primary localization: autoseg_centered.
* Primary ranking set: testing1 only; no downstream model is fitted.
* Primary QA score: qa_assd_mm. Other QA scores are reported as secondary.
* Primary true-error target for the corresponding-metric analysis: true_assd_mm.
* High-error classification label: true error > the 75th percentile estimated from
  training1 + validation1. The resulting true-error cutoff is frozen and applied
  to testing1. A 50th-percentile sensitivity endpoint is also reported.
* ROC/PR AUC on testing1 use the raw QA disagreement score and do not fit a classifier.
* A deployment-style QA operating threshold is selected on development cases to
  achieve >=95% specificity when possible, then frozen and applied to testing1.

This script intentionally does not call NSR-QA inference. It consumes the CSVs
already produced by evaluation.actual_autoseg.eval_actual_autoseg_nsrqa.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.analyze_nsrqa_domain_performance import load_clinical_metrics


METRICS = {
    "1_minus_dice": ("qa_1_minus_dice", "true_1_minus_dice", "1 - Dice"),
    "hd95": ("qa_hd95_mm", "true_hd95_mm", "HD95 (mm)"),
    "assd": ("qa_assd_mm", "true_assd_mm", "ASSD (mm)"),
    "relative_volume_difference": (
        "qa_relative_volume_difference",
        "true_relative_volume_difference",
        "Relative volume difference",
    ),
}

PRIMARY_QA_METRIC = "assd"
PRIMARY_TRUE_METRIC = "assd"


CLINICAL_TRUE_METRICS = {
    "1_minus_dice": {"clinical_metric": "dice", "true_col": "true_1_minus_dice"},
    "hd95": {"clinical_metric": "haus", "true_col": "true_hd95_mm"},
}


def _norm_patient_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def overlay_full_volume_clinical_truth(
    df: pd.DataFrame,
    roi: str,
    clinical_case_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replace Dice/HD95 truth with original full-volume ``metrics.csv`` values.

    QA scores are left untouched. The crop-derived truth columns are preserved as
    ``pipeline_<true_col>`` for audit. Ambiguous/missing clinical rows become NaN
    under the selected duplicate policy and are therefore excluded only from the
    affected endpoint by the existing finite-value filtering.
    """
    out = df.copy()
    out["patient_id"] = out["patient_id"].map(_norm_patient_id)
    if "roi" not in out.columns:
        out["roi"] = roi
    out["roi"] = out["roi"].astype(str).str.strip()

    cm = clinical_case_metrics.copy()
    cm["patient_id"] = cm["patient_id"].map(_norm_patient_id)
    cm["roi"] = cm["roi"].astype(str).str.strip()
    cm = cm[cm["clinical_metric_name"].isin(["dice", "haus"])].copy()

    value_wide = cm.pivot(index=["patient_id", "roi"], columns="clinical_metric_name", values="clinical_metric_value")
    status_wide = cm.pivot(index=["patient_id", "roi"], columns="clinical_metric_name", values="clinical_metric_status")
    value_wide = value_wide.rename(columns=lambda c: f"clinical_{c}_value").reset_index()
    status_wide = status_wide.rename(columns=lambda c: f"clinical_{c}_status").reset_index()
    join = value_wide.merge(status_wide, on=["patient_id", "roi"], how="outer", validate="one_to_one")
    out = out.merge(join, on=["patient_id", "roi"], how="left", validate="many_to_one")

    for name, spec in CLINICAL_TRUE_METRICS.items():
        true_col = spec["true_col"]
        clinical_metric = spec["clinical_metric"]
        if true_col not in out.columns:
            raise KeyError(f"Metrics table missing expected truth column: {true_col}")
        pipeline_col = f"pipeline_{true_col}"
        if pipeline_col not in out.columns:
            out[pipeline_col] = pd.to_numeric(out[true_col], errors="coerce")
        raw = pd.to_numeric(out[f"clinical_{clinical_metric}_value"], errors="coerce")
        clinical_true = 1.0 - raw if name == "1_minus_dice" else raw
        out[true_col] = clinical_true
        source_col = f"{true_col}_source"
        out[source_col] = np.where(
            np.isfinite(clinical_true),
            "metrics.csv",
            "metrics.csv_unavailable_or_ambiguous",
        )

    audit_cols = [c for c in [
        "patient_id", "roi", "localization_mode", "shape_model_split",
        "pipeline_true_1_minus_dice", "true_1_minus_dice", "true_1_minus_dice_source",
        "clinical_dice_status",
        "pipeline_true_hd95_mm", "true_hd95_mm", "true_hd95_mm_source",
        "clinical_haus_status",
    ] if c in out.columns]
    return out, out[audit_cols].copy()


def finite_xy(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    out = df.copy()
    out[xcol] = pd.to_numeric(out[xcol], errors="coerce")
    out[ycol] = pd.to_numeric(out[ycol], errors="coerce")
    return out[np.isfinite(out[xcol]) & np.isfinite(out[ycol])].copy()


def percentile_higher(values: Sequence[float], q: float) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan
    try:
        return float(np.percentile(a, q, method="higher"))
    except TypeError:  # numpy <1.22
        return float(np.percentile(a, q, interpolation="higher"))


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(score, method="average")
    u = float(ranks[pos].sum() - n1 * (n1 + 1) / 2.0)
    return u / float(n1 * n0)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    npos = int((y == 1).sum())
    if npos == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision[ys == 1]) / npos)


def roc_curve_points(y: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not len(y) or not np.any(y == 1) or not np.any(y == 0):
        return np.array([]), np.array([])
    thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]
    fpr, tpr = [], []
    p, n = (y == 1).sum(), (y == 0).sum()
    for t in thresholds:
        pred = score >= t
        fpr.append(float(np.sum(pred & (y == 0)) / n))
        tpr.append(float(np.sum(pred & (y == 1)) / p))
    return np.asarray(fpr), np.asarray(tpr)


def choose_score_threshold_at_specificity(y: np.ndarray, score: np.ndarray, target_spec: float) -> Tuple[float, float, float]:
    """Choose development threshold maximizing sensitivity subject to specificity >= target."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not np.any(y == 1) or not np.any(y == 0):
        return np.nan, np.nan, np.nan
    candidates = np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]
    best = None
    for t in candidates:
        pred = score >= t
        tn = int(np.sum((~pred) & (y == 0)))
        fp = int(np.sum(pred & (y == 0)))
        tp = int(np.sum(pred & (y == 1)))
        fn = int(np.sum((~pred) & (y == 1)))
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        if np.isfinite(spec) and spec + 1e-12 >= target_spec:
            cand = (sens, -float(t), float(t), spec)
            if best is None or cand > best:
                best = cand
    if best is None:
        return np.nan, np.nan, np.nan
    return best[2], best[0], best[3]


def confusion_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not np.isfinite(threshold):
        return {k: np.nan for k in ("tp", "fp", "tn", "fn", "sensitivity", "specificity", "ppv", "npv")}
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum((~pred) & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
    }


def concordance_index(true_error: np.ndarray, qa_score: np.ndarray) -> float:
    true_error = np.asarray(true_error, dtype=float)
    qa_score = np.asarray(qa_score, dtype=float)
    concordant = 0.0
    comparable = 0
    n = len(true_error)
    for i in range(n):
        for j in range(i + 1, n):
            dy = true_error[i] - true_error[j]
            if dy == 0:
                continue
            ds = qa_score[i] - qa_score[j]
            comparable += 1
            if ds == 0:
                concordant += 0.5
            elif dy * ds > 0:
                concordant += 1.0
    return concordant / comparable if comparable else np.nan


def bootstrap_ranking_ci(d: pd.DataFrame, xcol: str, ycol: str, n_boot: int, seed: int) -> Dict[str, float]:
    d = finite_xy(d, xcol, ycol).reset_index(drop=True)
    if len(d) < 4:
        return {"spearman_low": np.nan, "spearman_high": np.nan, "kendall_low": np.nan, "kendall_high": np.nan, "cindex_low": np.nan, "cindex_high": np.nan, "valid": 0}
    rng = np.random.default_rng(seed)
    vals = {"spearman": [], "kendall": [], "cindex": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        b = d.iloc[idx]
        if b[xcol].nunique() < 2 or b[ycol].nunique() < 2:
            continue
        sr = stats.spearmanr(b[xcol], b[ycol]).statistic
        kt = stats.kendalltau(b[xcol], b[ycol]).statistic
        ci = concordance_index(b[ycol].to_numpy(float), b[xcol].to_numpy(float))
        if np.isfinite(sr) and np.isfinite(kt) and np.isfinite(ci):
            vals["spearman"].append(float(sr)); vals["kendall"].append(float(kt)); vals["cindex"].append(float(ci))
    out: Dict[str, float] = {"valid": len(vals["spearman"])}
    for k in ("spearman", "kendall", "cindex"):
        if vals[k]:
            lo, hi = np.percentile(vals[k], [2.5, 97.5])
            out[f"{k}_low"] = float(lo); out[f"{k}_high"] = float(hi)
        else:
            out[f"{k}_low"] = np.nan; out[f"{k}_high"] = np.nan
    return out


def bootstrap_classification_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    if len(y) < 4 or not np.any(y == 1) or not np.any(y == 0):
        return {"auc_low": np.nan, "auc_high": np.nan, "ap_low": np.nan, "ap_high": np.nan, "valid": 0}
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yy, ss = y[idx], score[idx]
        if not np.any(yy == 1) or not np.any(yy == 0):
            continue
        aucs.append(auc_rank(yy, ss)); aps.append(average_precision(yy, ss))
    if not aucs:
        return {"auc_low": np.nan, "auc_high": np.nan, "ap_low": np.nan, "ap_high": np.nan, "valid": 0}
    alo, ahi = np.percentile(aucs, [2.5, 97.5]); plo, phi = np.percentile(aps, [2.5, 97.5])
    return {"auc_low": float(alo), "auc_high": float(ahi), "ap_low": float(plo), "ap_high": float(phi), "valid": len(aucs)}


def top_fraction_retrieval(d: pd.DataFrame, xcol: str, ycol: str, fraction: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    d = finite_xy(d, xcol, ycol).copy()
    n = len(d)
    if n == 0:
        return {}, pd.DataFrame()
    k = max(1, int(math.ceil(fraction * n)))
    d["true_rank"] = d[ycol].rank(method="min", ascending=False).astype(int)
    d["qa_rank"] = d[xcol].rank(method="min", ascending=False).astype(int)
    d["rank_error_abs"] = (d["true_rank"] - d["qa_rank"]).abs()
    true_top_ids = set(d.nlargest(k, ycol).index)
    qa_top_ids = set(d.nlargest(k, xcol).index)
    overlap = len(true_top_ids & qa_top_ids)
    recall = overlap / k
    enrichment = recall / (k / n) if k else np.nan
    d["true_top_fraction"] = d.index.isin(true_top_ids)
    d["qa_top_fraction"] = d.index.isin(qa_top_ids)
    return {
        "n": n, "k": k, "fraction": fraction, "overlap": overlap,
        "precision_at_k": recall, "recall_at_k": recall,
        "enrichment_over_random": enrichment,
        "mean_absolute_rank_error": float(d["rank_error_abs"].mean()),
    }, d


def save_rank_plot(d: pd.DataFrame, xcol: str, ycol: str, label_x: str, label_y: str, title: str, path: Path) -> None:
    d = finite_xy(d, xcol, ycol).copy()
    if not len(d):
        return
    d["true_rank"] = d[ycol].rank(method="average", ascending=False)
    d["qa_rank"] = d[xcol].rank(method="average", ascending=False)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(d["true_rank"], d["qa_rank"], s=34, alpha=0.75)
    lim = max(float(d["true_rank"].max()), float(d["qa_rank"].max()), 1.0)
    ax.plot([1, lim], [1, lim], linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"True-error rank (1 = worst): {label_y}")
    ax.set_ylabel(f"NSR-QA rank (1 = highest): {label_x}")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def save_roc_plot(y: np.ndarray, score: np.ndarray, title: str, path: Path) -> None:
    fpr, tpr = roc_curve_points(y, score)
    if not len(fpr):
        return
    auc = auc_rank(y, score)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def ranking_rows(df: pd.DataFrame, roi: str, mode: str, n_boot: int, seed: int, out_dir: Path, top_fraction: float) -> Tuple[List[Dict], List[pd.DataFrame]]:
    test = df[df["shape_model_split"] == "testing1"].copy()
    rows: List[Dict] = []; cases: List[pd.DataFrame] = []
    counter = 0
    for qa_name, (xcol, _same_y, qlabel) in METRICS.items():
        for true_name, (_same_x, ycol, tlabel) in METRICS.items():
            d = finite_xy(test, xcol, ycol)
            if len(d) >= 3 and d[xcol].nunique() >= 2 and d[ycol].nunique() >= 2:
                sr = float(stats.spearmanr(d[xcol], d[ycol]).statistic)
                kt = float(stats.kendalltau(d[xcol], d[ycol]).statistic)
                ci = float(concordance_index(d[ycol].to_numpy(float), d[xcol].to_numpy(float)))
            else:
                sr = kt = ci = np.nan
            ci_boot = bootstrap_ranking_ci(d, xcol, ycol, n_boot, seed + counter * 37); counter += 1
            retr, case_df = top_fraction_retrieval(d, xcol, ycol, top_fraction)
            row = {
                "roi": roi, "localization_mode": mode, "subset": "testing1",
                "qa_metric": qa_name, "true_metric": true_name, "qa_column": xcol, "true_column": ycol,
                "n": int(len(d)), "spearman_rho": sr,
                "spearman_ci95_low": ci_boot.get("spearman_low"), "spearman_ci95_high": ci_boot.get("spearman_high"),
                "kendall_tau": kt, "kendall_ci95_low": ci_boot.get("kendall_low"), "kendall_ci95_high": ci_boot.get("kendall_high"),
                "concordance_index": ci, "cindex_ci95_low": ci_boot.get("cindex_low"), "cindex_ci95_high": ci_boot.get("cindex_high"),
                "bootstrap_valid_replicates": ci_boot.get("valid", 0),
                **{f"top_{int(round(top_fraction*100))}pct_{k}": v for k, v in retr.items()},
                "is_primary_pair": bool(qa_name == PRIMARY_QA_METRIC and true_name == PRIMARY_TRUE_METRIC),
            }
            rows.append(row)
            if len(case_df):
                keep = [c for c in ("patient_id", "roi", "localization_mode", "shape_model_split", "case_dir", xcol, ycol, "true_rank", "qa_rank", "rank_error_abs", "true_top_fraction", "qa_top_fraction") if c in case_df.columns]
                cc = case_df[keep].copy(); cc.insert(0, "qa_metric", qa_name); cc.insert(1, "true_metric", true_name); cases.append(cc)
            if qa_name == true_name:
                save_rank_plot(d, xcol, ycol, qlabel, tlabel, f"{roi} | {mode} | testing1 ranking | {qlabel}", out_dir / "ranking_plots" / f"{qa_name}_corresponding_rank.png")
    return rows, cases


def classification_rows(df: pd.DataFrame, roi: str, mode: str, percentiles: Sequence[float], target_spec: float, min_class_n: int, n_boot: int, seed: int, out_dir: Path) -> Tuple[List[Dict], List[pd.DataFrame]]:
    development = df[df["shape_model_split"].isin(["training1", "validation1"])].copy()
    test = df[df["shape_model_split"] == "testing1"].copy()
    rows: List[Dict] = []; cases: List[pd.DataFrame] = []
    counter = 0
    for true_name, (_same_x, ycol, tlabel) in METRICS.items():
        dev_y = pd.to_numeric(development[ycol], errors="coerce")
        for pct in percentiles:
            cutoff = percentile_higher(dev_y, pct)
            for qa_name, (xcol, _same_y, qlabel) in METRICS.items():
                dev = finite_xy(development, xcol, ycol)
                te = finite_xy(test, xcol, ycol)
                if not np.isfinite(cutoff):
                    continue
                dev_label = (dev[ycol].to_numpy(float) > cutoff).astype(int)
                test_label = (te[ycol].to_numpy(float) > cutoff).astype(int)
                test_score = te[xcol].to_numpy(float)
                n_bad = int(test_label.sum()); n_good = int((test_label == 0).sum())
                auc = auc_rank(test_label, test_score)
                ap = average_precision(test_label, test_score)
                reliable = bool(n_bad >= min_class_n and n_good >= min_class_n)
                boot = bootstrap_classification_ci(test_label, test_score, n_boot, seed + counter * 53); counter += 1
                score_thr, dev_sens, dev_spec = choose_score_threshold_at_specificity(dev_label, dev[xcol].to_numpy(float), target_spec)
                cm = confusion_at_threshold(test_label, test_score, score_thr)
                row = {
                    "roi": roi, "localization_mode": mode,
                    "true_metric": true_name, "qa_metric": qa_name,
                    "true_error_percentile_from_development": pct,
                    "true_error_cutoff": cutoff,
                    "development_n": int(len(dev)), "development_bad": int(dev_label.sum()), "development_good": int((dev_label == 0).sum()),
                    "test_n": int(len(te)), "test_bad": n_bad, "test_good": n_good,
                    "roc_auc_testing1": auc, "roc_auc_ci95_low": boot["auc_low"], "roc_auc_ci95_high": boot["auc_high"],
                    "pr_auc_testing1": ap, "pr_auc_ci95_low": boot["ap_low"], "pr_auc_ci95_high": boot["ap_high"],
                    "bootstrap_valid_replicates": boot["valid"],
                    "reliable_roc_min_class_n": reliable, "min_class_n_required": min_class_n,
                    "development_qa_threshold_target_specificity": target_spec,
                    "development_selected_qa_threshold": score_thr,
                    "development_sensitivity_at_selected_threshold": dev_sens,
                    "development_specificity_at_selected_threshold": dev_spec,
                    **{f"testing1_{k}_fixed_dev_threshold": v for k, v in cm.items()},
                    "is_primary_endpoint": bool(pct == 75 and qa_name == PRIMARY_QA_METRIC and true_name == PRIMARY_TRUE_METRIC),
                }
                rows.append(row)
                if len(te):
                    cc = te[[c for c in ("patient_id", "roi", "localization_mode", "shape_model_split", "case_dir", xcol, ycol) if c in te.columns]].copy()
                    cc.insert(0, "qa_metric", qa_name); cc.insert(1, "true_metric", true_name); cc.insert(2, "true_error_percentile", pct)
                    cc["true_error_cutoff"] = cutoff; cc["true_bad"] = test_label; cc["qa_threshold_from_development"] = score_thr
                    if np.isfinite(score_thr):
                        cc["qa_predicted_bad"] = test_score >= score_thr
                        cc["confusion"] = np.where((cc["true_bad"] == 1) & cc["qa_predicted_bad"], "TP", np.where((cc["true_bad"] == 0) & cc["qa_predicted_bad"], "FP", np.where((cc["true_bad"] == 0) & (~cc["qa_predicted_bad"]), "TN", "FN")))
                    cases.append(cc)
                if pct == 75 and qa_name == true_name:
                    save_roc_plot(test_label, test_score, f"{roi} | {mode} | testing1 high-error {tlabel} | dev P75", out_dir / "roc_plots" / f"{qa_name}_corresponding_p75.png")
    return rows, cases


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roi", required=True)
    p.add_argument("--metrics", action="append", required=True, help="mode=metrics.csv; repeat for localization modes")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument("--bad-percentiles", nargs="+", type=float, default=[75.0, 50.0])
    p.add_argument("--target-specificity", type=float, default=0.95)
    p.add_argument("--top-fraction", type=float, default=0.25)
    p.add_argument("--min-class-n", type=int, default=4)
    p.add_argument("--clinical-metrics-csv", default=None,
                   help="Original full-volume Radformation metrics.csv. When supplied, its dice/haus values replace crop-derived Dice/HD95 truth.")
    p.add_argument("--clinical-site", default="thorax")
    p.add_argument("--clinical-metrics-duplicate-policy", choices=["exclude", "median", "first", "error"], default="exclude",
                   help="How to handle conflicting patient/ROI/metric rows because QA CSVs do not carry study UID.")
    args = p.parse_args()

    out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    ranking_all: List[pd.DataFrame] = []; class_all: List[pd.DataFrame] = []
    rank_cases_all: List[pd.DataFrame] = []; class_cases_all: List[pd.DataFrame] = []

    clinical_case_metrics = None
    clinical_metric_audit = pd.DataFrame()
    clinical_join_audits: List[pd.DataFrame] = []
    if args.clinical_metrics_csv:
        clinical_case_metrics, _clinical_exact_metrics, clinical_metric_audit = load_clinical_metrics(
            args.clinical_metrics_csv, args.clinical_site, args.clinical_metrics_duplicate_policy
        )
        clinical_metric_audit.to_csv(out / "clinical_metrics_source_audit.csv", index=False)

    parsed = []
    for item in args.metrics:
        if "=" not in item:
            raise ValueError(f"Expected mode=csv, got: {item}")
        mode, raw = item.split("=", 1); parsed.append((mode, Path(raw).expanduser().resolve()))

    for mode, csv_path in parsed:
        df = pd.read_csv(csv_path, dtype={"patient_id": str})
        if clinical_case_metrics is not None:
            df, join_audit = overlay_full_volume_clinical_truth(df, args.roi, clinical_case_metrics)
            join_audit.insert(0, "metrics_csv", str(csv_path))
            clinical_join_audits.append(join_audit)
        required = {"shape_model_split"} | {c for pair in METRICS.values() for c in pair[:2]}
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"{csv_path} missing columns: {missing}")
        mode_dir = out / mode
        rr, rc = ranking_rows(df, args.roi, mode, args.n_boot, args.seed, mode_dir, args.top_fraction)
        cr, cc = classification_rows(df, args.roi, mode, args.bad_percentiles, args.target_specificity, args.min_class_n, args.n_boot, args.seed + 20000, mode_dir)
        ranking_all.append(pd.DataFrame(rr)); class_all.append(pd.DataFrame(cr)); rank_cases_all.extend(rc); class_cases_all.extend(cc)

    ranking = pd.concat(ranking_all, ignore_index=True) if ranking_all else pd.DataFrame()
    classification = pd.concat(class_all, ignore_index=True) if class_all else pd.DataFrame()
    ranking.to_csv(out / "ranking_summary.csv", index=False)
    classification.to_csv(out / "classification_summary.csv", index=False)
    if rank_cases_all:
        pd.concat(rank_cases_all, ignore_index=True).to_csv(out / "ranking_cases.csv", index=False)
    if class_cases_all:
        pd.concat(class_cases_all, ignore_index=True).to_csv(out / "classification_cases.csv", index=False)

    if clinical_join_audits:
        pd.concat(clinical_join_audits, ignore_index=True).to_csv(out / "clinical_truth_join_audit.csv", index=False)

    # Direct primary-vs-oracle comparison on the locked testing1 set.
    if len(ranking):
        key = ["qa_metric", "true_metric"]
        a = ranking[ranking["localization_mode"] == "autoseg_centered"]
        b = ranking[ranking["localization_mode"] == "approved_reference"]
        if len(a) and len(b):
            keep = key + ["n", "spearman_rho", "kendall_tau", "concordance_index"]
            comp = a[keep].merge(b[keep], on=key, suffixes=("_autoseg_centered", "_approved_reference"))
            for c in ("spearman_rho", "kendall_tau", "concordance_index"):
                comp[f"delta_{c}_autoseg_minus_approved"] = comp[f"{c}_autoseg_centered"] - comp[f"{c}_approved_reference"]
            comp.to_csv(out / "localization_ranking_comparison.csv", index=False)
    if len(classification):
        key = ["qa_metric", "true_metric", "true_error_percentile_from_development"]
        a = classification[classification["localization_mode"] == "autoseg_centered"]
        b = classification[classification["localization_mode"] == "approved_reference"]
        if len(a) and len(b):
            keep = key + ["test_n", "test_bad", "test_good", "roc_auc_testing1", "pr_auc_testing1"]
            comp = a[keep].merge(b[keep], on=key, suffixes=("_autoseg_centered", "_approved_reference"))
            for c in ("roc_auc_testing1", "pr_auc_testing1"):
                comp[f"delta_{c}_autoseg_minus_approved"] = comp[f"{c}_autoseg_centered"] - comp[f"{c}_approved_reference"]
            comp.to_csv(out / "localization_classification_comparison.csv", index=False)

    metadata = {
        "roi": args.roi,
        "qa_model": "NSR-QA only; no downstream classifier is fitted for ROC/PR AUC",
        "primary_localization": "autoseg_centered",
        "oracle_localization_sensitivity": "approved_reference",
        "primary_ranking_subset": "testing1",
        "primary_qa_metric": PRIMARY_QA_METRIC,
        "primary_true_error_metric": PRIMARY_TRUE_METRIC,
        "primary_high_error_endpoint": "true ASSD > development(training1+validation1) 75th percentile",
        "secondary_high_error_percentile": 50,
        "qa_operating_threshold": f"selected on training1+validation1 for >= {args.target_specificity:.2f} specificity, then frozen for testing1",
        "top_fraction_retrieval": args.top_fraction,
        "min_positive_and_negative_for_reliable_roc_flag": args.min_class_n,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "clinical_thresholds": "not encoded; percentile-defined endpoints are high-error discrimination, not clinical failure criteria",
        "clinical_true_error_source": args.clinical_metrics_csv if args.clinical_metrics_csv else "pipeline-derived actual-autoseg metrics CSV",
        "clinical_full_volume_metrics_applied_to": ["1_minus_dice", "hd95"] if args.clinical_metrics_csv else [],
        "clinical_metrics_duplicate_policy": args.clinical_metrics_duplicate_policy if args.clinical_metrics_csv else None,
        "clinical_site": args.clinical_site if args.clinical_metrics_csv else None,
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote ranking/classification analysis to {out}")


if __name__ == "__main__":
    main()
