#!/usr/bin/env python3
"""
ROC analysis for Lung_L with corrupted samples restricted to volume change > threshold.

Default task:
  Negative class: clean Lung_L samples
  Positive class: corrupted Lung_L samples with input-vs-target volume change > 20%

Outputs:
  tables/roc_summary_by_metric.csv
  tables/threshold_operating_points.csv
  tables/bootstrap_auc_ci.csv
  tables/roc_curve_points.csv
  tables/analysis_dataset.csv
  figures/roc_curves_all_metrics.png
  figures/score_distribution_clean_vs_corrupt_gt20.png
  figures/sensitivity_specificity_thresholds.png
  figures/metric_ranking_barplot.png
  report/Lung_L_ROC_report.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
    "abs_input_recon_volume_diff_frac",
]

DISPLAY_NAMES = {
    "one_minus_dice_input_vs_recon": "1-Dice(input,recon)",
    "dice_input_vs_recon": "Dice(input,recon)",
    "hd95_input_vs_recon": "HD95(input,recon)",
    "assd_input_vs_recon": "ASSD(input,recon)",
    "abs_input_recon_volume_diff_frac": "Volume diff(input,recon)",
}


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "value"


def load_csvs(csvs: List[str]) -> pd.DataFrame:
    frames = []
    for raw in csvs:
        p = Path(raw)
        if not p.exists():
            print(f"WARNING: missing CSV, skip: {p}")
            continue
        df = pd.read_csv(p)
        if df.empty:
            print(f"WARNING: empty CSV, skip: {p}")
            continue
        df["source_csv"] = str(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid input CSVs loaded.")
    return pd.concat(frames, ignore_index=True)


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "roi" not in out.columns:
        out["roi"] = "unknown_roi"
    if "probe_type" not in out.columns:
        raise ValueError("Input CSV must contain probe_type.")

    for c in ["input_voxels", "target_voxels", "recon_voxels"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "requested_level_numeric" in out.columns:
        out["requested_level_numeric"] = pd.to_numeric(out["requested_level_numeric"], errors="coerce").fillna(0)
    elif "requested_level" in out.columns:
        out["requested_level_numeric"] = pd.to_numeric(out["requested_level"], errors="coerce").fillna(0)
    elif "level" in out.columns:
        out["requested_level_numeric"] = pd.to_numeric(out["level"], errors="coerce").fillna(0)
    else:
        out["requested_level_numeric"] = 0.0

    out["is_clean"] = (out["probe_type"].astype(str) == "clean") | (out["requested_level_numeric"] == 0)

    if "one_minus_dice_input_vs_recon" not in out.columns and "dice_input_vs_recon" in out.columns:
        out["one_minus_dice_input_vs_recon"] = 1.0 - pd.to_numeric(out["dice_input_vs_recon"], errors="coerce")

    if "abs_input_recon_volume_diff_frac" not in out.columns and {"input_voxels", "recon_voxels"}.issubset(out.columns):
        denom = out["input_voxels"].replace(0, np.nan)
        out["abs_input_recon_volume_diff_frac"] = ((out["input_voxels"] - out["recon_voxels"]) / denom).abs()

    if "actual_volume_change_pct" in out.columns:
        out["actual_volume_change_pct"] = pd.to_numeric(out["actual_volume_change_pct"], errors="coerce")
    elif "abs_input_volume_change_pct" in out.columns:
        out["actual_volume_change_pct"] = pd.to_numeric(out["abs_input_volume_change_pct"], errors="coerce")
    elif "abs_input_volume_change_frac" in out.columns:
        out["actual_volume_change_pct"] = 100.0 * pd.to_numeric(out["abs_input_volume_change_frac"], errors="coerce")
    elif {"input_voxels", "target_voxels"}.issubset(out.columns):
        denom = out["target_voxels"].replace(0, np.nan)
        out["actual_volume_change_pct"] = 100.0 * ((out["input_voxels"] - out["target_voxels"]) / denom).abs()
    else:
        raise ValueError("Need volume-change columns or input_voxels/target_voxels.")

    out.loc[out["is_clean"], "actual_volume_change_pct"] = 0.0
    return out


def choose_metrics(df: pd.DataFrame, metrics: Optional[List[str]]) -> List[str]:
    if metrics:
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise ValueError(f"Requested metrics missing: {missing}")
        return metrics
    return [m for m in DEFAULT_METRICS if m in df.columns]


def roc_curve_manual(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores)
    y = y_true[order]
    s = scores[order]
    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    distinct = np.r_[True, s[1:] != s[:-1]]
    idx = np.where(distinct)[0]
    thresholds = s[idx]
    tps = np.array([np.sum(y[:i + 1] == 1) for i in idx])
    fps = np.array([np.sum(y[:i + 1] == 0) for i in idx])
    tpr = tps / P
    fpr = fps / N
    return np.r_[0.0, fpr, 1.0], np.r_[0.0, tpr, 1.0], np.r_[np.inf, thresholds, -np.inf]


def auc_pairwise(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    comp = pos[:, None] - neg[None, :]
    return float((np.sum(comp > 0) + 0.5 * np.sum(comp == 0)) / comp.size)


def partial_auc_standardized(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float = 0.05) -> float:
    order = np.argsort(fpr)
    f = fpr[order]
    t = tpr[order]
    if max_fpr not in f:
        t_at = np.interp(max_fpr, f, t)
        keep = f < max_fpr
        f = np.r_[f[keep], max_fpr]
        t = np.r_[t[keep], t_at]
    else:
        keep = f <= max_fpr
        f, t = f[keep], t[keep]
    raw = float(np.trapz(t, f))
    min_area = 0.5 * max_fpr * max_fpr
    max_area = max_fpr
    return 0.5 + 0.5 * (raw - min_area) / (max_area - min_area)


def operating_points(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, fixed_specificities: List[float]) -> Dict[str, float]:
    specificity = 1.0 - fpr
    out = {}
    youden = tpr - fpr
    j = int(np.nanargmax(youden))
    out.update({
        "youden_index": float(youden[j]),
        "youden_threshold": float(thresholds[j]),
        "youden_sensitivity": float(tpr[j]),
        "youden_specificity": float(specificity[j]),
    })
    for sp in fixed_specificities:
        feasible = np.where(specificity >= sp)[0]
        k = int(round(sp * 100))
        if len(feasible) == 0:
            out[f"sens_at_{k}spec"] = np.nan
            out[f"threshold_at_{k}spec"] = np.nan
            out[f"actual_spec_at_{k}spec"] = np.nan
        else:
            best = feasible[np.nanargmax(tpr[feasible])]
            out[f"sens_at_{k}spec"] = float(tpr[best])
            out[f"threshold_at_{k}spec"] = float(thresholds[best])
            out[f"actual_spec_at_{k}spec"] = float(specificity[best])
    return out


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(auc_pairwise(yy, ss))
    if not vals:
        return np.nan, np.nan, np.nan
    vals = np.asarray(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def build_analysis_dataset(df, roi, volume_threshold_pct, exclude_shift, probe_types):
    g = df[df["roi"].astype(str) == roi].copy()
    if g.empty:
        g = df[df["roi"].astype(str).str.lower() == roi.lower()].copy()
    if g.empty:
        raise ValueError(f"No rows found for ROI={roi}.")
    if exclude_shift:
        g = g[g["probe_type"].astype(str) != "shifted"].copy()
    if probe_types:
        keep = set(probe_types) | {"clean"}
        g = g[g["probe_type"].astype(str).isin(keep)].copy()
    clean = g[g["is_clean"]].copy()
    corrupt = g[(~g["is_clean"]) & (g["actual_volume_change_pct"] > volume_threshold_pct)].copy()
    clean["class_label"] = "clean"
    corrupt["class_label"] = f"corrupt_vol_gt_{volume_threshold_pct:g}"
    clean["y_true"] = 0
    corrupt["y_true"] = 1
    return pd.concat([clean, corrupt], ignore_index=True)


def run_roc(dataset, metrics, fixed_specificities, n_boot, seed, pauc_max_fpr):
    summary_rows, op_rows, ci_rows, curve_rows = [], [], [], []
    for metric in metrics:
        sub = dataset[["y_true", metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", metric])
        if sub["y_true"].nunique() < 2:
            print(f"WARNING: skip {metric}, only one class after dropping NaN.")
            continue
        y = sub["y_true"].astype(int).to_numpy()
        s = sub[metric].astype(float).to_numpy()
        fpr, tpr, thresholds = roc_curve_manual(y, s)
        auc = auc_pairwise(y, s)
        pauc = partial_auc_standardized(fpr, tpr, max_fpr=pauc_max_fpr)
        ops = operating_points(fpr, tpr, thresholds, fixed_specificities)
        boot_mean, ci_low, ci_high = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed)
        row = {
            "metric": metric,
            "metric_display": DISPLAY_NAMES.get(metric, metric),
            "n_clean": int(np.sum(y == 0)),
            "n_corrupt": int(np.sum(y == 1)),
            "auc": auc,
            "auc_boot_mean": boot_mean,
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            f"standardized_pauc_fpr_0_{pauc_max_fpr:g}": pauc,
            **ops,
        }
        summary_rows.append(row)
        op_rows.append({"metric": metric, "metric_display": DISPLAY_NAMES.get(metric, metric), **ops})
        ci_rows.append({"metric": metric, "metric_display": DISPLAY_NAMES.get(metric, metric), "auc": auc, "bootstrap_mean_auc": boot_mean, "auc_ci_low": ci_low, "auc_ci_high": ci_high, "n_boot": n_boot})
        for ff, tt, th in zip(fpr, tpr, thresholds):
            curve_rows.append({"metric": metric, "metric_display": DISPLAY_NAMES.get(metric, metric), "fpr": float(ff), "specificity": float(1 - ff), "tpr_sensitivity": float(tt), "threshold": float(th) if np.isfinite(th) else th})
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        sort_cols = ["auc"] + (["sens_at_95spec"] if "sens_at_95spec" in summary.columns else [])
        summary = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        summary["rank_by_auc"] = np.arange(1, len(summary) + 1)
    return summary, pd.DataFrame(op_rows), pd.DataFrame(ci_rows), pd.DataFrame(curve_rows)


def plot_roc_curves(curve_df, summary, out_path, title):
    plt.figure(figsize=(6.4, 6.0))
    for metric, g in curve_df.groupby("metric", dropna=False):
        disp = DISPLAY_NAMES.get(metric, metric)
        auc = summary.loc[summary["metric"] == metric, "auc"]
        label = f"{disp} (AUC={float(auc.iloc[0]):.3f})" if len(auc) else disp
        gg = g.sort_values("fpr")
        plt.plot(gg["fpr"], gg["tpr_sensitivity"], linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_score_distributions(dataset, metrics, out_path, volume_threshold_pct):
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(7.8, max(3.2, 2.8 * n)), squeeze=False)
    axes = axes[:, 0]
    for ax, metric in zip(axes, metrics):
        clean = pd.to_numeric(dataset[dataset["y_true"] == 0][metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        corrupt = pd.to_numeric(dataset[dataset["y_true"] == 1][metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(clean, bins=35, density=True, alpha=0.45, label=f"clean (n={len(clean)})")
        ax.hist(corrupt, bins=35, density=True, alpha=0.45, label=f"corrupt >{volume_threshold_pct:g}% (n={len(corrupt)})")
        try:
            clean.plot(kind="kde", ax=ax, linewidth=2, label="clean KDE")
            corrupt.plot(kind="kde", ax=ax, linewidth=2, label="corrupt KDE")
        except Exception:
            pass
        ax.set_xlabel(DISPLAY_NAMES.get(metric, metric))
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Clean vs corrupted score distributions, volume change > {volume_threshold_pct:g}%", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_threshold_operating_points(curve_df, out_path):
    plt.figure(figsize=(8.0, 5.6))
    for metric, g in curve_df.groupby("metric", dropna=False):
        gg = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["threshold"])
        if gg.empty:
            continue
        gg = gg.sort_values("threshold")
        x = np.arange(len(gg))
        plt.plot(x, gg["tpr_sensitivity"], linewidth=1.8, label=f"{DISPLAY_NAMES.get(metric, metric)} sensitivity")
        plt.plot(x, gg["specificity"], linestyle="--", linewidth=1.4, label=f"{DISPLAY_NAMES.get(metric, metric)} specificity")
    plt.xlabel("Threshold index, sorted low to high")
    plt.ylabel("Rate")
    plt.title("Sensitivity and specificity across thresholds")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metric_ranking(summary, out_path):
    if summary.empty:
        return
    df = summary.sort_values("auc", ascending=True)
    labels = df["metric_display"].to_list()
    y = np.arange(len(df))
    plt.figure(figsize=(7.8, max(3.8, 0.55 * len(df))))
    plt.barh(y, df["auc"])
    plt.yticks(y, labels)
    plt.xlabel("ROC AUC")
    plt.title("Metric ranking by AUC")
    plt.xlim(0, 1.0)
    for i, (_, row) in enumerate(df.iterrows()):
        ci = ""
        if np.isfinite(row.get("auc_ci_low", np.nan)) and np.isfinite(row.get("auc_ci_high", np.nan)):
            ci = f" [{row['auc_ci_low']:.2f}, {row['auc_ci_high']:.2f}]"
        plt.text(min(row["auc"] + 0.01, 0.98), i, f"{row['auc']:.3f}{ci}", va="center", fontsize=8)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_report(out_path, roi, volume_threshold_pct, summary, dataset, figures_rel):
    best = summary.iloc[0] if not summary.empty else None
    n_clean = int((dataset["y_true"] == 0).sum())
    n_corrupt = int((dataset["y_true"] == 1).sum())
    probe_counts = dataset[dataset["y_true"] == 1]["probe_type"].astype(str).value_counts().to_dict()
    lines = []
    lines.append(f"# ROC analysis report: {roi}\n")
    lines.append("## Task definition\n")
    lines.append(f"- Negative class: clean `{roi}` contours.")
    lines.append(f"- Positive class: corrupted `{roi}` contours with input-vs-target volume change > `{volume_threshold_pct:g}%`.")
    lines.append("- Detector scores: reconstruction discrepancy metrics comparing input contour with AE reconstruction.")
    lines.append("- Larger detector scores are interpreted as more suspicious.\n")
    lines.append("## Dataset used\n")
    lines.append(f"- Clean samples: `{n_clean}`")
    lines.append(f"- Corrupted samples with volume change > {volume_threshold_pct:g}%: `{n_corrupt}`")
    if probe_counts:
        lines.append("- Corrupted samples by probe type:")
        for k, v in probe_counts.items():
            lines.append(f"  - `{k}`: {v}")
    lines.append("")
    lines.append("## Main ROC comparison\n")
    if best is not None:
        ci_low = best.get("auc_ci_low", np.nan)
        ci_high = best.get("auc_ci_high", np.nan)
        lines.append(f"The best metric by ROC AUC was **{best['metric_display']}** (AUC = {best['auc']:.3f}, 95% bootstrap CI = [{ci_low:.3f}, {ci_high:.3f}]).")
        if "sens_at_95spec" in best:
            lines.append(f"At approximately 95% specificity, this metric achieved sensitivity = {best['sens_at_95spec']:.3f} with threshold = {best['threshold_at_95spec']:.4g}.")
    lines.append("")
    lines.append("### Metric summary\n")
    if not summary.empty:
        cols = ["metric_display", "auc", "auc_ci_low", "auc_ci_high", "sens_at_95spec", "threshold_at_95spec", "youden_threshold", "youden_sensitivity", "youden_specificity"]
        avail = [c for c in cols if c in summary.columns]
        lines.append(summary[avail].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Figures\n")
    for name, rel in figures_rel.items():
        lines.append(f"### {name}\n")
        lines.append(f"![{name}]({rel})\n")
    lines.append("## Suggested interpretation paragraph\n")
    if best is not None:
        lines.append(f"For `{roi}`, ROC analysis was performed by comparing clean contours with synthetic corruptions that produced more than `{volume_threshold_pct:g}%` input-vs-target volume change. Among the tested reconstruction discrepancy metrics, `{best['metric_display']}` provided the strongest discrimination, with AUC `{best['auc']:.3f}`. At a high-specificity operating point, this metric achieved sensitivity `{best.get('sens_at_95spec', np.nan):.3f}` at approximately 95% specificity. This suggests that, for volume-changing corruptions above the selected severity threshold, `{best['metric_display']}` is the preferred detector score for `{roi}`.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Standard ROC analysis for Lung_L with volume-change threshold.")
    p.add_argument("--csvs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--roi", default="Lung_L")
    p.add_argument("--volume_threshold_pct", type=float, default=20.0)
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--include_shift", action="store_true")
    p.add_argument("--probe_types", nargs="+", default=None)
    p.add_argument("--fixed_specificities", nargs="+", type=float, default=[0.90, 0.95])
    p.add_argument("--pauc_max_fpr", type=float, default=0.05)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    report_dir = out_dir / "report"
    for d in [tables_dir, figs_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = add_columns(load_csvs(args.csvs))
    metrics = choose_metrics(df, args.metrics)
    dataset = build_analysis_dataset(df, args.roi, args.volume_threshold_pct, exclude_shift=not args.include_shift, probe_types=args.probe_types)
    if dataset["y_true"].nunique() < 2:
        raise ValueError(f"Need both classes. Counts: {dataset['y_true'].value_counts().to_dict()}")
    dataset.to_csv(tables_dir / "analysis_dataset.csv", index=False)

    summary, ops, ci, curves = run_roc(dataset, metrics, args.fixed_specificities, args.n_boot, args.seed, args.pauc_max_fpr)
    summary.to_csv(tables_dir / "roc_summary_by_metric.csv", index=False)
    ops.to_csv(tables_dir / "threshold_operating_points.csv", index=False)
    ci.to_csv(tables_dir / "bootstrap_auc_ci.csv", index=False)
    curves.to_csv(tables_dir / "roc_curve_points.csv", index=False)

    plot_roc_curves(curves, summary, figs_dir / "roc_curves_all_metrics.png", title=f"{args.roi}: ROC curves, corruptions > {args.volume_threshold_pct:g}% volume change")
    plot_score_distributions(dataset, metrics, figs_dir / "score_distribution_clean_vs_corrupt_gt20.png", volume_threshold_pct=args.volume_threshold_pct)
    plot_threshold_operating_points(curves, figs_dir / "sensitivity_specificity_thresholds.png")
    plot_metric_ranking(summary, figs_dir / "metric_ranking_barplot.png")

    write_report(
        report_dir / f"{sanitize(args.roi)}_ROC_report.md",
        roi=args.roi,
        volume_threshold_pct=args.volume_threshold_pct,
        summary=summary,
        dataset=dataset,
        figures_rel={
            "ROC curves": "../figures/roc_curves_all_metrics.png",
            "Score distributions": "../figures/score_distribution_clean_vs_corrupt_gt20.png",
            "Sensitivity/specificity thresholds": "../figures/sensitivity_specificity_thresholds.png",
            "Metric ranking": "../figures/metric_ranking_barplot.png",
        },
    )
    print(f"Metrics: {metrics}")
    print(f"Wrote: {out_dir}")
    print(f"Main summary: {tables_dir / 'roc_summary_by_metric.csv'}")
    print(f"Report: {report_dir / f'{sanitize(args.roi)}_ROC_report.md'}")


if __name__ == "__main__":
    main()
