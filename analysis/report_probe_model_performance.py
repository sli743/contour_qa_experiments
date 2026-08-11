#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METRICS = {
    "one_minus_dice": ("1 - Dice(input,recon)", lambda d: 1.0 - pd.to_numeric(d["dice_input_vs_recon"], errors="coerce")),
    "hd95": ("HD95(input,recon)", lambda d: pd.to_numeric(d["hd95_input_vs_recon"], errors="coerce")),
    "assd": ("ASSD(input,recon)", lambda d: pd.to_numeric(d["assd_input_vs_recon"], errors="coerce")),
    "rel_volume": ("Relative volume difference(input,recon)", lambda d: (pd.to_numeric(d["input_voxels"], errors="coerce") - pd.to_numeric(d["recon_voxels"], errors="coerce")).abs() / pd.to_numeric(d["input_voxels"], errors="coerce").replace(0, np.nan)),
}


def is_shift(s: pd.Series) -> pd.Series:
    return s.astype(str).str.contains("shift|translation", case=False, regex=True, na=False)


def is_clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("clean")


def labels_from_volume(df: pd.DataFrame, threshold_pct: float) -> pd.Series:
    iv = pd.to_numeric(df["input_voxels"], errors="coerce")
    tv = pd.to_numeric(df["target_voxels"], errors="coerce").replace(0, np.nan)
    pct = 100.0 * (iv - tv).abs() / tv
    y = (pct > threshold_pct).astype(float)
    y.loc[is_clean(df["probe_type"])] = 0.0
    return y


def op_at_specificity(y: np.ndarray, score: np.ndarray, target_spec: float = 0.95):
    fpr, tpr, thr = roc_curve(y, score)
    spec = 1.0 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return np.nan, np.nan, np.nan
    j = idx[np.argmax(tpr[idx])]
    return float(tpr[j]), float(spec[j]), float(thr[j])


def bootstrap_auc(y: np.ndarray, score: np.ndarray, seed: int = 88, n_boot: int = 1000):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(roc_auc_score(yy, score[idx]))
    if not vals:
        return np.nan, np.nan
    return tuple(np.quantile(vals, [0.025, 0.975]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--roi", required=True)
    p.add_argument("--volume-threshold-pct", type=float, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--exclude-shifts", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    roc_df = df.copy()
    if args.exclude_shifts:
        roc_df = roc_df.loc[~is_shift(roc_df["probe_type"])].copy()
    # deduplicate the repeated level-0 clean rows produced by the probe sweep
    clean = roc_df.loc[is_clean(roc_df["probe_type"])].drop_duplicates("patient_id")
    nonclean = roc_df.loc[~is_clean(roc_df["probe_type"])]
    roc_df = pd.concat([clean, nonclean], ignore_index=True)
    y_series = labels_from_volume(roc_df, args.volume_threshold_pct)

    perf_rows = []
    for key, (label, fn) in METRICS.items():
        score = fn(roc_df)
        valid = y_series.notna() & score.notna() & np.isfinite(score)
        y = y_series.loc[valid].astype(int).to_numpy()
        s = score.loc[valid].to_numpy(float)
        if np.unique(y).size < 2:
            continue
        auc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
        lo, hi = bootstrap_auc(y, s)
        sens95, spec95, threshold95 = op_at_specificity(y, s, 0.95)
        fpr, tpr, _ = roc_curve(y, s)
        j = int(np.argmax(tpr - fpr))
        perf_rows.append({"metric": key, "label": label, "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi, "average_precision": ap, "sensitivity_at_95_specificity": sens95, "achieved_specificity": spec95, "threshold_at_95_specificity": threshold95, "youden_sensitivity": float(tpr[j]), "youden_specificity": float(1-fpr[j]), "n_positive": int(y.sum()), "n_negative": int((y==0).sum())})
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.plot([0,1],[0,1], "--", linewidth=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{args.roi}: {label}")
        ax.legend(loc="lower right"); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(plot_dir / f"roc_{key}.png", dpi=250); fig.savefig(plot_dir / f"roc_{key}.pdf"); plt.close(fig)

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(args.output_dir / "metric_performance.csv", index=False)

    clean0 = df.loc[is_clean(df["probe_type"])].drop_duplicates("patient_id").copy()
    clean_summary = pd.DataFrame([{
        "n": len(clean0),
        "dice_recon_target_mean": pd.to_numeric(clean0["dice_recon_vs_target"], errors="coerce").mean(),
        "dice_recon_target_median": pd.to_numeric(clean0["dice_recon_vs_target"], errors="coerce").median(),
        "hd95_recon_target_median": pd.to_numeric(clean0["hd95_recon_vs_target"], errors="coerce").replace([np.inf,-np.inf], np.nan).median(),
        "assd_recon_target_median": pd.to_numeric(clean0["assd_recon_vs_target"], errors="coerce").replace([np.inf,-np.inf], np.nan).median(),
    }])
    clean_summary.to_csv(args.output_dir / "clean_preservation.csv", index=False)

    repair = df.loc[~is_clean(df["probe_type"])].copy()
    if args.exclude_shifts:
        repair = repair.loc[~is_shift(repair["probe_type"])].copy()
    repair_summary = repair.groupby(["probe_type", "level"], dropna=False).agg(
        n=("patient_id", "size"),
        dice_input_target_median=("dice_input_vs_target", "median"),
        dice_recon_target_median=("dice_recon_vs_target", "median"),
        dice_gain_median=("dice_correction_gain", "median"),
        dice_gain_mean=("dice_correction_gain", "mean"),
        hd95_improvement_median=("hd95_improvement", "median"),
        assd_improvement_median=("assd_improvement", "median"),
    ).reset_index()
    repair_summary.to_csv(args.output_dir / "probe_level_repair_summary.csv", index=False)

    by_probe = repair.groupby("probe_type")["dice_correction_gain"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, .45*len(by_probe)+1)))
    ax.barh(by_probe.index.astype(str), by_probe.values)
    ax.axvline(0, linewidth=1); ax.set_xlabel("Median Dice correction gain"); ax.set_title(f"{args.roi}: repair by probe family"); fig.tight_layout()
    fig.savefig(plot_dir / "repair_by_probe.png", dpi=250); fig.savefig(plot_dir / "repair_by_probe.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for probe, sub in repair_summary.groupby("probe_type"):
        ax.plot(sub["level"], sub["dice_gain_median"], marker="o", label=str(probe))
    ax.axhline(0, linewidth=1); ax.set(xlabel="Probe level", ylabel="Median Dice correction gain", title=f"{args.roi}: correction versus severity")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(plot_dir / "correction_vs_severity.png", dpi=250); fig.savefig(plot_dir / "correction_vs_severity.pdf"); plt.close(fig)

    best = perf.sort_values("auc", ascending=False).iloc[0] if len(perf) else None
    md = [f"# {args.roi} stochastic-region-dropout probe evaluation", "", f"Failure definition: absolute input-vs-target volume change > {args.volume_threshold_pct:g}%" + ("; shift probes excluded." if args.exclude_shifts else "."), "", "## Detection performance", ""]
    if len(perf):
        md += [perf[["label","auc","auc_ci_low","auc_ci_high","average_precision","sensitivity_at_95_specificity"]].to_markdown(index=False, floatfmt=".3f"), ""]
        md += [f"Best detector by AUROC: **{best['label']}**, AUC={best['auc']:.3f} (95% bootstrap CI {best['auc_ci_low']:.3f}-{best['auc_ci_high']:.3f}).", ""]
    md += ["## Clean-contour preservation", "", clean_summary.to_markdown(index=False, floatfmt=".3f"), "", "## Probe repair", "", "Positive Dice gain / HD95 improvement / ASSD improvement means reconstruction moved toward the clean target.", "", repair_summary.to_markdown(index=False, floatfmt=".3f"), "", "## Figures", ""]
    for key, (label, _) in METRICS.items():
        if (plot_dir/f"roc_{key}.png").exists(): md.append(f"### ROC: {label}\n\n![](plots/roc_{key}.png)\n")
    md += ["### Repair by probe family", "", "![](plots/repair_by_probe.png)", "", "### Correction versus severity", "", "![](plots/correction_vs_severity.png)", ""]
    (args.output_dir / "report.md").write_text("\n".join(md))
    print(f"Wrote report: {args.output_dir / 'report.md'}")

if __name__ == "__main__":
    main()
