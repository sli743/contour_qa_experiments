#!/usr/bin/env python3
"""Paired actual-autoseg evaluation: previous-generation vs v4 SpinalCord NSR-QA.

Primary endpoints are measured on held-out testing1 (Month3):
  * raw corresponding-metric QA ranking (Spearman rho), and
  * P75 high-error classification (AUROC/AUPRC) using true-error cutoffs learned
    from training1+validation1 and frozen before testing1.

P50 classification, fixed development-fit regression, approved-reference
localization, shared synthetic probes, and domain-stratified analyses are
secondary/sensitivity analyses. All old-vs-v4 uncertainty is paired by patient.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.analyze_actual_autoseg_ranking_classification import (
    auc_rank,
    average_precision,
    overlay_full_volume_clinical_truth,
    percentile_higher,
    roc_curve_points,
)
from analysis.analyze_nsrqa_domain_performance import (
    attach_domains,
    load_clinical_metrics,
    load_domain_assignments,
)

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
PRIMARY_METRICS = {"1_minus_dice", "hd95"}


def norm_pid(x) -> str:
    s = str(x).strip()
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def load_metric_table(path: str, model_key: str, clinical_case_metrics: pd.DataFrame | None) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    d = pd.read_csv(p, dtype={"patient_id": str})
    d["patient_id"] = d["patient_id"].map(norm_pid)
    if "roi" not in d.columns:
        d["roi"] = "SpinalCord"
    if clinical_case_metrics is not None:
        d, _ = overlay_full_volume_clinical_truth(d, "SpinalCord", clinical_case_metrics)
    d["model_key"] = model_key
    d["source_metrics_csv"] = str(p)
    return d


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 2 or len(np.unique(x)) < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def fixed_regression_metrics(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> Dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) == 0 or not np.isfinite(slope) or not np.isfinite(intercept):
        return {"n": int(len(x)), "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    pred = intercept + slope * x
    resid = y - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(x)),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
    }


def paired_model_table(previous: pd.DataFrame, v4: pd.DataFrame) -> pd.DataFrame:
    keys = ["patient_id", "roi", "localization_mode", "shape_model_split"]
    keep = keys + [c for pair in METRICS.values() for c in pair[:2]]
    for extra in ("case_dir", "model_name", "corruption_label"):
        if extra in previous.columns or extra in v4.columns:
            keep.append(extra)
    keep = list(dict.fromkeys(c for c in keep if c in previous.columns and c in v4.columns or c in keys))
    p = previous[[c for c in keep if c in previous.columns]].copy()
    q = v4[[c for c in keep if c in v4.columns]].copy()
    m = p.merge(q, on=keys, how="inner", suffixes=("_previous", "_v4"), validate="one_to_one")
    if len(m) != len(previous) or len(m) != len(v4):
        missing_prev = sorted(set(v4["patient_id"]) - set(previous["patient_id"]))
        missing_v4 = sorted(set(previous["patient_id"]) - set(v4["patient_id"]))
        raise RuntimeError(
            f"Model case tables are not identical: previous={len(previous)}, v4={len(v4)}, paired={len(m)}, "
            f"missing_previous={missing_prev[:10]}, missing_v4={missing_v4[:10]}"
        )
    for metric, (_q, t, _label) in METRICS.items():
        a = pd.to_numeric(m[f"{t}_previous"], errors="coerce")
        b = pd.to_numeric(m[f"{t}_v4"], errors="coerce")
        both = np.isfinite(a) & np.isfinite(b)
        if both.any() and not np.allclose(a[both], b[both], atol=1e-9, rtol=1e-9):
            raise RuntimeError(f"True-error values differ between model arms for {metric}")
        m[f"true_{metric}_shared"] = a.where(np.isfinite(a), b)
    return m


def paired_bootstrap(
    test: pd.DataFrame,
    qa_prev_col: str,
    qa_v4_col: str,
    truth_col: str,
    cutoff: float | None,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    cols = [qa_prev_col, qa_v4_col, truth_col]
    d = test[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    delta_rho: List[float] = []
    delta_auc: List[float] = []
    delta_ap: List[float] = []
    for _ in range(n_boot):
        if len(d) < 3:
            break
        idx = rng.integers(0, len(d), size=len(d))
        b = d.iloc[idx]
        r0 = safe_spearman(b[qa_prev_col], b[truth_col])
        r1 = safe_spearman(b[qa_v4_col], b[truth_col])
        if np.isfinite(r0) and np.isfinite(r1):
            delta_rho.append(r1 - r0)
        if cutoff is not None and np.isfinite(cutoff):
            y = (b[truth_col].to_numpy(float) > float(cutoff)).astype(int)
            if y.min() != y.max():
                a0 = auc_rank(y, b[qa_prev_col].to_numpy(float))
                a1 = auc_rank(y, b[qa_v4_col].to_numpy(float))
                p0 = average_precision(y, b[qa_prev_col].to_numpy(float))
                p1 = average_precision(y, b[qa_v4_col].to_numpy(float))
                if np.isfinite(a0) and np.isfinite(a1): delta_auc.append(a1 - a0)
                if np.isfinite(p0) and np.isfinite(p1): delta_ap.append(p1 - p0)

    def summarize(vals: List[float], prefix: str) -> Dict[str, float]:
        if not vals:
            return {f"{prefix}_ci95_low": np.nan, f"{prefix}_ci95_high": np.nan, f"{prefix}_p_gt_0": np.nan, f"{prefix}_valid_boot": 0}
        a = np.asarray(vals, dtype=float)
        lo, hi = np.percentile(a, [2.5, 97.5])
        return {
            f"{prefix}_ci95_low": float(lo),
            f"{prefix}_ci95_high": float(hi),
            f"{prefix}_p_gt_0": float(np.mean(a > 0)),
            f"{prefix}_valid_boot": int(len(a)),
        }

    return {**summarize(delta_rho, "delta_spearman"), **summarize(delta_auc, "delta_auc"), **summarize(delta_ap, "delta_ap")}


def evaluate_mode(paired: pd.DataFrame, mode: str, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = paired[paired["localization_mode"] == mode].copy()
    dev = d[d["shape_model_split"].isin(["training1", "validation1"])].copy()
    test = d[d["shape_model_split"] == "testing1"].copy()
    metric_rows, class_rows, regression_rows = [], [], []

    for mi, (metric, (qa_col, _true_col, label)) in enumerate(METRICS.items()):
        q0 = f"{qa_col}_previous"; q1 = f"{qa_col}_v4"; truth = f"true_{metric}_shared"
        rho0 = safe_spearman(test[q0], test[truth]); rho1 = safe_spearman(test[q1], test[truth])
        boot = paired_bootstrap(test, q0, q1, truth, None, n_boot, seed + 1000 * mi)
        metric_rows.append({
            "localization_mode": mode, "metric": metric, "label": label,
            "test_n": int(test[[q0, q1, truth]].dropna().shape[0]),
            "spearman_previous": rho0, "spearman_v4": rho1,
            "delta_spearman_v4_minus_previous": rho1 - rho0 if np.isfinite(rho0) and np.isfinite(rho1) else np.nan,
            "primary_endpoint": bool(mode == "autoseg_centered" and metric in PRIMARY_METRICS),
            **{k: v for k, v in boot.items() if k.startswith("delta_spearman")},
        })

        x0d = pd.to_numeric(dev[q0], errors="coerce").to_numpy(float)
        x1d = pd.to_numeric(dev[q1], errors="coerce").to_numpy(float)
        yd = pd.to_numeric(dev[truth], errors="coerce").to_numpy(float)
        slope0, intercept0 = fit_line(x0d, yd); slope1, intercept1 = fit_line(x1d, yd)
        ytest = pd.to_numeric(test[truth], errors="coerce").to_numpy(float)
        reg0 = fixed_regression_metrics(pd.to_numeric(test[q0], errors="coerce").to_numpy(float), ytest, slope0, intercept0)
        reg1 = fixed_regression_metrics(pd.to_numeric(test[q1], errors="coerce").to_numpy(float), ytest, slope1, intercept1)
        regression_rows.append({
            "localization_mode": mode, "metric": metric,
            "development_slope_previous": slope0, "development_intercept_previous": intercept0,
            "development_slope_v4": slope1, "development_intercept_v4": intercept1,
            "test_n": min(reg0["n"], reg1["n"]),
            "mae_previous": reg0["mae"], "mae_v4": reg1["mae"], "delta_mae_v4_minus_previous": reg1["mae"] - reg0["mae"],
            "rmse_previous": reg0["rmse"], "rmse_v4": reg1["rmse"], "delta_rmse_v4_minus_previous": reg1["rmse"] - reg0["rmse"],
            "r2_previous": reg0["r2"], "r2_v4": reg1["r2"], "delta_r2_v4_minus_previous": reg1["r2"] - reg0["r2"],
        })

        dev_truth = pd.to_numeric(dev[truth], errors="coerce")
        for pct in (75.0, 50.0):
            cutoff = percentile_higher(dev_truth, pct)
            te = test[[q0, q1, truth]].apply(pd.to_numeric, errors="coerce").dropna()
            y = (te[truth].to_numpy(float) > cutoff).astype(int)
            auc0 = auc_rank(y, te[q0].to_numpy(float)); auc1 = auc_rank(y, te[q1].to_numpy(float))
            ap0 = average_precision(y, te[q0].to_numpy(float)); ap1 = average_precision(y, te[q1].to_numpy(float))
            bootc = paired_bootstrap(te, q0, q1, truth, cutoff, n_boot, seed + 1000 * mi + int(pct))
            class_rows.append({
                "localization_mode": mode, "metric": metric, "true_error_percentile": pct,
                "true_error_cutoff": cutoff, "test_n": int(len(te)), "test_bad": int(y.sum()), "test_good": int((y == 0).sum()),
                "roc_auc_previous": auc0, "roc_auc_v4": auc1,
                "delta_auc_v4_minus_previous": auc1 - auc0 if np.isfinite(auc0) and np.isfinite(auc1) else np.nan,
                "pr_auc_previous": ap0, "pr_auc_v4": ap1,
                "delta_ap_v4_minus_previous": ap1 - ap0 if np.isfinite(ap0) and np.isfinite(ap1) else np.nan,
                "primary_endpoint": bool(mode == "autoseg_centered" and pct == 75.0 and metric in PRIMARY_METRICS),
                **{k: v for k, v in bootc.items() if k.startswith("delta_auc") or k.startswith("delta_ap")},
            })

    return pd.DataFrame(metric_rows), pd.DataFrame(class_rows), pd.DataFrame(regression_rows)


def plot_scatter(test: pd.DataFrame, metric: str, out: Path) -> None:
    qa, _t, label = METRICS[metric]
    truth = f"true_{metric}_shared"; q0 = f"{qa}_previous"; q1 = f"{qa}_v4"
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(test[q0], test[truth], marker="o", alpha=0.65, label="Previous-generation")
    ax.scatter(test[q1], test[truth], marker="x", alpha=0.8, label="V4 corruption")
    ax.set_xlabel(f"QA disagreement: {label}"); ax.set_ylabel(f"True autoseg error: {label}")
    ax.set_title(f"Month3 actual autosegs: {label}"); ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=180); plt.close(fig)


def plot_roc(test: pd.DataFrame, metric: str, cutoff: float, out: Path) -> None:
    qa, _t, label = METRICS[metric]; truth = f"true_{metric}_shared"
    d = test[[f"{qa}_previous", f"{qa}_v4", truth]].apply(pd.to_numeric, errors="coerce").dropna()
    y = (d[truth].to_numpy(float) > cutoff).astype(int)
    if len(np.unique(y)) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for col, name in ((f"{qa}_previous", "Previous-generation"), (f"{qa}_v4", "V4 corruption")):
        fpr, tpr = roc_curve_points(y, d[col].to_numpy(float))
        auc = auc_rank(y, d[col].to_numpy(float))
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Month3 P75 high-error ROC: {label}", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=180); plt.close(fig)


def domain_analysis(
    paired: pd.DataFrame,
    domain_csv: str,
    n_boot: int,
    seed: int,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments, conflicts, multi_study = load_domain_assignments(domain_csv, "thorax")
    test = paired[(paired["localization_mode"] == "autoseg_centered") & (paired["shape_model_split"] == "testing1")].copy()
    test = attach_domains(test, assignments)
    test = test[~test["domain_conflict"]].copy()
    rows, sep_rows, kw_rows = [], [], []
    rng = np.random.default_rng(seed)

    for metric, (qa, _t, _label) in METRICS.items():
        truth = f"true_{metric}_shared"; q0 = f"{qa}_previous"; q1 = f"{qa}_v4"
        dev = paired[(paired["localization_mode"] == "autoseg_centered") & paired["shape_model_split"].isin(["training1", "validation1"])]
        cutoff = percentile_higher(pd.to_numeric(dev[truth], errors="coerce"), 75.0)
        for domain, g in test.groupby("domain", sort=True):
            gg = g[[q0, q1, truth]].apply(pd.to_numeric, errors="coerce").dropna()
            y = (gg[truth].to_numpy(float) > cutoff).astype(int) if len(gg) else np.asarray([], dtype=int)
            a0 = auc_rank(y, gg[q0].to_numpy(float)) if len(np.unique(y)) == 2 else np.nan
            a1 = auc_rank(y, gg[q1].to_numpy(float)) if len(np.unique(y)) == 2 else np.nan
            rows.append({
                "metric": metric, "domain": domain, "n": int(len(gg)),
                "true_error_median": float(gg[truth].median()) if len(gg) else np.nan,
                "qa_median_previous": float(gg[q0].median()) if len(gg) else np.nan,
                "qa_median_v4": float(gg[q1].median()) if len(gg) else np.nan,
                "spearman_previous": safe_spearman(gg[q0], gg[truth]),
                "spearman_v4": safe_spearman(gg[q1], gg[truth]),
                "p75_bad_n": int(y.sum()) if len(y) else 0,
                "p75_good_n": int((y == 0).sum()) if len(y) else 0,
                "p75_auc_previous": a0, "p75_auc_v4": a1,
            })

        # Exploratory: can one scalar QA signal distinguish this domain from the
        # remaining Month3 cases? Report direction-free separability as max(AUC,1-AUC).
        for domain in sorted(test["domain"].dropna().unique()):
            ydom = (test["domain"].astype(str) == str(domain)).astype(int).to_numpy()
            if ydom.min() == ydom.max():
                continue
            for model, col in (("previous_generation", q0), ("v4_corruption", q1)):
                score = pd.to_numeric(test[col], errors="coerce").to_numpy(float)
                ok = np.isfinite(score)
                if ok.sum() < 4 or len(np.unique(ydom[ok])) < 2:
                    auc = np.nan
                else:
                    auc = auc_rank(ydom[ok], score[ok])
                sep = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
                sep_rows.append({
                    "metric": metric, "domain_vs_rest": domain, "model": model,
                    "n": int(ok.sum()), "domain_n": int(ydom[ok].sum()),
                    "roc_auc_signed": auc, "separability_auc": sep,
                    "higher_qa_in_domain": bool(auc >= 0.5) if np.isfinite(auc) else None,
                })

        for model, col in (("previous_generation", q0), ("v4_corruption", q1)):
            groups = [pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float) for _, g in test.groupby("domain")]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) >= 2:
                kw = stats.kruskal(*groups)
                kw_rows.append({"metric": metric, "model": model, "kruskal_h": float(kw.statistic), "kruskal_p": float(kw.pvalue), "n_domains": len(groups)})

        # Boxplot for the two primary metrics only.
        if metric in PRIMARY_METRICS:
            domains = sorted(test["domain"].dropna().astype(str).unique())
            if domains:
                fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(domains)), 5.4))
                positions, data, labels = [], [], []
                pos = 1.0
                for dom in domains:
                    gd = test[test["domain"].astype(str) == dom]
                    for col, suffix in ((q0, "prev"), (q1, "v4")):
                        vals = pd.to_numeric(gd[col], errors="coerce").dropna().to_numpy(float)
                        if len(vals):
                            positions.append(pos); data.append(vals); labels.append(f"{dom}\n{suffix}")
                        pos += 1.0
                    pos += 0.35
                if data:
                    ax.boxplot(data, positions=positions, widths=0.65, showfliers=False)
                    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.set_ylabel(f"QA disagreement: {METRICS[metric][2]}")
                    ax.set_title(f"Month3 QA signal by OOD/error domain: {METRICS[metric][2]}")
                    ax.grid(axis="y", alpha=0.2); fig.tight_layout()
                    fig.savefig(out_dir / f"domain_boxplot_{metric}.png", dpi=180); plt.close(fig)

    test.to_csv(out_dir / "domain_cases_testing1.csv", index=False)
    conflicts.to_csv(out_dir / "domain_assignment_conflicts.csv", index=False)
    multi_study.to_csv(out_dir / "domain_multi_study_audit.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(sep_rows), pd.DataFrame(kw_rows), conflicts


def markdown_table(df: pd.DataFrame, columns: Sequence[str], digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    x = df[list(columns)].copy()
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].map(lambda v: "NA" if pd.isna(v) else f"{float(v):.{digits}f}")
    headers = list(x.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in x.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--previous-primary", required=True)
    p.add_argument("--v4-primary", required=True)
    p.add_argument("--previous-oracle", default=None)
    p.add_argument("--v4-oracle", default=None)
    p.add_argument("--clinical-metrics-csv", default=None)
    p.add_argument("--clinical-metrics-duplicate-policy", choices=["exclude", "median", "first", "error"], default="exclude")
    p.add_argument("--domain-csv", default=None)
    p.add_argument("--previous-probe-csv", default=None)
    p.add_argument("--v4-probe-csv", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=88)
    args = p.parse_args()

    out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    clinical = None
    clinical_audit = pd.DataFrame()
    if args.clinical_metrics_csv:
        clinical, _exact, clinical_audit = load_clinical_metrics(
            args.clinical_metrics_csv, "thorax", args.clinical_metrics_duplicate_policy
        )
        clinical_audit.to_csv(out / "clinical_metrics_duplicate_audit.csv", index=False)

    mode_pairs = [("autoseg_centered", args.previous_primary, args.v4_primary)]
    if args.previous_oracle and args.v4_oracle:
        mode_pairs.append(("approved_reference", args.previous_oracle, args.v4_oracle))

    paired_parts = []
    for mode, p0, p1 in mode_pairs:
        d0 = load_metric_table(p0, "previous_generation", clinical)
        d1 = load_metric_table(p1, "v4_corruption", clinical)
        d0 = d0[d0["localization_mode"] == mode].copy(); d1 = d1[d1["localization_mode"] == mode].copy()
        paired_parts.append(paired_model_table(d0, d1))
    paired = pd.concat(paired_parts, ignore_index=True)
    paired.to_csv(out / "paired_actual_autoseg_cases.csv", index=False)

    rank_parts, cls_parts, reg_parts = [], [], []
    for i, (mode, _p0, _p1) in enumerate(mode_pairs):
        r, c, g = evaluate_mode(paired, mode, args.n_boot, args.seed + i * 100000)
        rank_parts.append(r); cls_parts.append(c); reg_parts.append(g)
    ranking = pd.concat(rank_parts, ignore_index=True); classification = pd.concat(cls_parts, ignore_index=True); regression = pd.concat(reg_parts, ignore_index=True)
    ranking.to_csv(out / "paired_ranking_summary.csv", index=False)
    classification.to_csv(out / "paired_classification_summary.csv", index=False)
    regression.to_csv(out / "paired_regression_summary.csv", index=False)

    primary_test = paired[(paired["localization_mode"] == "autoseg_centered") & (paired["shape_model_split"] == "testing1")]
    primary_dev = paired[(paired["localization_mode"] == "autoseg_centered") & paired["shape_model_split"].isin(["training1", "validation1"])]
    for metric in PRIMARY_METRICS:
        plot_scatter(primary_test, metric, out / f"scatter_testing1_{metric}.png")
        cutoff = percentile_higher(pd.to_numeric(primary_dev[f"true_{metric}_shared"], errors="coerce"), 75.0)
        plot_roc(primary_test, metric, cutoff, out / f"roc_testing1_p75_{metric}.png")

    domain_perf = domain_sep = domain_kw = pd.DataFrame()
    if args.domain_csv:
        domain_dir = out / "domain_analysis"; domain_dir.mkdir(parents=True, exist_ok=True)
        domain_perf, domain_sep, domain_kw, _ = domain_analysis(paired, args.domain_csv, args.n_boot, args.seed + 700000, domain_dir)
        domain_perf.to_csv(domain_dir / "domain_performance.csv", index=False)
        domain_sep.to_csv(domain_dir / "domain_separability.csv", index=False)
        domain_kw.to_csv(domain_dir / "domain_kruskal_wallis.csv", index=False)

    probe_summary = pd.DataFrame()
    if args.previous_probe_csv and args.v4_probe_csv:
        p0 = pd.read_csv(args.previous_probe_csv); p1 = pd.read_csv(args.v4_probe_csv)
        common_keys = [c for c in ("patient_id", "probe_type", "severity", "sample_id", "case_id") if c in p0.columns and c in p1.columns]
        # Keep this deliberately descriptive because exact probe schemas vary across historical exports.
        rows = []
        for model, d in (("previous_generation", p0), ("v4_corruption", p1)):
            row = {"model": model, "n_rows": len(d)}
            for c in ("recon_target_dice", "input_recon_dice", "correction_gain", "qa_assd_mm", "qa_hd95_mm"):
                if c in d.columns:
                    row[f"{c}_mean"] = float(pd.to_numeric(d[c], errors="coerce").mean())
            rows.append(row)
        probe_summary = pd.DataFrame(rows); probe_summary.to_csv(out / "shared_probe_descriptive_summary.csv", index=False)

    primary_rank = ranking[ranking["primary_endpoint"]].copy()
    primary_cls = classification[classification["primary_endpoint"]].copy()
    report = [
        "# SpinalCord NSR-QA: previous-generation vs v4 corruption",
        "",
        "## Study design",
        "",
        "The two NSR-QA arms use the same month-temporal SpinalCord split, coverage-safe FOV, frozen 3DINO encoder, decoder architecture, symmetric reconstruction loss, optimizer, and evaluation procedures. The intended scientific intervention is the training corruption distribution only. Month3/testing1 is held out for the primary actual-autoseg evaluation.",
        "",
        "Primary QA performance is measured on actual autosegmentations rather than on synthetic probes. Corresponding-metric raw QA signals are evaluated directly, without fitting a downstream classifier on Month3.",
        "",
        "## Primary Month3 endpoints",
        "",
        "### Ranking / continuous QA association",
        "",
        markdown_table(primary_rank, ["metric", "test_n", "spearman_previous", "spearman_v4", "delta_spearman_v4_minus_previous", "delta_spearman_ci95_low", "delta_spearman_ci95_high"]),
        "",
        "### P75 bad-contour classification",
        "",
        markdown_table(primary_cls, ["metric", "true_error_cutoff", "test_bad", "test_good", "roc_auc_previous", "roc_auc_v4", "delta_auc_v4_minus_previous", "delta_auc_ci95_low", "delta_auc_ci95_high", "pr_auc_previous", "pr_auc_v4"]),
        "",
        "## Secondary fixed-fit regression",
        "",
        markdown_table(regression[regression["localization_mode"] == "autoseg_centered"], ["metric", "test_n", "mae_previous", "mae_v4", "r2_previous", "r2_v4"]),
        "",
        "## Localization sensitivity",
        "",
    ]
    oracle_rank = ranking[ranking["localization_mode"] == "approved_reference"]
    report.append(markdown_table(oracle_rank, ["metric", "test_n", "spearman_previous", "spearman_v4", "delta_spearman_v4_minus_previous"]))
    if args.domain_csv:
        report += [
            "",
            "## OOD/error-domain robustness (secondary/exploratory)",
            "",
            "Domain labels are not used for model training. Unrecorded reviewed Month3 patient-ROI pairs are treated as Normal by the existing OOD-domain convention; conflicting assignments are excluded. Within-domain correlations/AUCs can be unstable when n is small.",
            "",
            markdown_table(domain_perf[domain_perf["metric"].isin(PRIMARY_METRICS)], ["metric", "domain", "n", "true_error_median", "qa_median_previous", "qa_median_v4", "spearman_previous", "spearman_v4", "p75_auc_previous", "p75_auc_v4"]),
            "",
            "The `domain_analysis/domain_separability.csv` file additionally reports exploratory one-vs-rest domain separability of each scalar QA signal. This is mechanistic analysis, not the definition of QA success.",
        ]
    report += [
        "",
        "## Interpretation rules",
        "",
        "1. The main question is whether v4 improves actual Month3 QA relative to the previous-generation model, especially P75 AUROC/AUPRC and Spearman association for 1-Dice and HD95.",
        "2. Paired bootstrap deltas resample the same Month3 patients for both models. Positive delta-AUC/delta-Spearman favors v4.",
        "3. Development data determine P75/P50 true-error cutoffs and any fixed regression calibration; Month3 outcomes are never used to fit those mappings.",
        "4. Synthetic-probe results are secondary mechanism checks and should not override actual-autoseg performance.",
        "5. Domain-stratified findings are secondary/exploratory because domain sample sizes can be small and labels originate from the OOD-review process.",
        "",
        "## Viewer outputs",
        "",
        "Each model's case directory is self-contained and uses the existing ActualAutosegNSRQAViewer format: `ct.nii.gz`, `approved.nii.gz`, `autoseg_input.nii.gz`, `nsrqa_reconstruction.nii.gz`, four missing/excess diagnostic masks, and `metrics.json`.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    metadata = {
        "primary_localization": "autoseg_centered",
        "primary_test_split": "testing1 / Month3",
        "primary_metrics": sorted(PRIMARY_METRICS),
        "primary_classification_percentile": 75,
        "classification_cutoff_source": "training1 + validation1 true error",
        "paired_bootstrap_replicates": args.n_boot,
        "seed": args.seed,
        "clinical_truth_overlay": bool(args.clinical_metrics_csv),
        "domain_analysis": bool(args.domain_csv),
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote paired comparison report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
