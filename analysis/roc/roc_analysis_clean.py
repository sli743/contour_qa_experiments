#!/usr/bin/env python3
"""
Clean ROC analysis for Lung_L or any ROI.

Default task
------------
Negative class:
  clean ROI contours

Positive class:
  corrupted ROI contours with input-vs-target volume change > 20%

Default ROI:
  Lung_L

Default detector scores:
  one_minus_dice_input_vs_recon
  hd95_input_vs_recon
  assd_input_vs_recon
  abs_input_recon_volume_diff_frac

Outputs
-------
out_dir/
  tables/
    analysis_dataset.csv
    roc_summary_by_metric.csv
    roc_curve_points.csv
    bootstrap_auc_ci.csv
    threshold_operating_points.csv

  figures/
    roc_curves_all_metrics.png
    metric_ranking_barplot.png
    score_distribution_clean_vs_corrupt.png
    operating_point_tradeoff.png

  report/
    <ROI>_ROC_report.md

Notes
-----
All detector scores are assumed to be oriented so that larger = more abnormal.
For Dice, use one_minus_dice_input_vs_recon rather than dice_input_vs_recon.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    "hd95_input_vs_recon": "HD95(input,recon)",
    "assd_input_vs_recon": "ASSD(input,recon)",
    "abs_input_recon_volume_diff_frac": "Volume diff(input,recon)",
}

PUB_COLS = {
    "metric_display": "Metric",
    "auc": "AUC",
    "auc_ci_low": "AUC CI low",
    "auc_ci_high": "AUC CI high",
    "pauc_fpr_le_0.05": "pAUC FPR≤0.05",
    "sens_at_95spec": "Sensitivity @95% specificity",
    "threshold_at_95spec": "Threshold @95% specificity",
    "sens_at_90spec": "Sensitivity @90% specificity",
    "threshold_at_90spec": "Threshold @90% specificity",
    "spec_at_95sens": "Specificity @95% sensitivity",
    "threshold_at_95sens": "Threshold @95% sensitivity",
    "spec_at_90sens": "Specificity @90% sensitivity",
    "threshold_at_90sens": "Threshold @90% sensitivity",
    "youden_threshold": "Youden threshold",
    "youden_sensitivity": "Youden sensitivity",
    "youden_specificity": "Youden specificity",
}


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "value"


def load_csvs(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"WARNING: missing CSV, skipped: {p}")
            continue
        df = pd.read_csv(p)
        if df.empty:
            print(f"WARNING: empty CSV, skipped: {p}")
            continue
        df["source_csv"] = str(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid CSV files loaded.")
    return pd.concat(frames, ignore_index=True)


def add_required_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError(
            "Need actual_volume_change_pct, abs_input_volume_change_pct, "
            "abs_input_volume_change_frac, or input_voxels/target_voxels."
        )

    out.loc[out["is_clean"], "actual_volume_change_pct"] = 0.0
    return out


def choose_metrics(df: pd.DataFrame, requested: Optional[List[str]]) -> List[str]:
    if requested:
        missing = [m for m in requested if m not in df.columns]
        if missing:
            raise ValueError(f"Requested metrics missing from CSV: {missing}")
        return requested
    return [m for m in DEFAULT_METRICS if m in df.columns]


def build_dataset(
    df: pd.DataFrame,
    roi: str,
    volume_threshold_pct: float,
    exclude_shift: bool,
    probe_types: Optional[List[str]],
) -> pd.DataFrame:
    g = df[df["roi"].astype(str) == roi].copy()
    if g.empty:
        g = df[df["roi"].astype(str).str.lower() == roi.lower()].copy()
    if g.empty:
        raise ValueError(f"No rows found for ROI={roi}")

    if exclude_shift:
        g = g[g["probe_type"].astype(str) != "shifted"].copy()

    if probe_types:
        keep = set(probe_types) | {"clean"}
        g = g[g["probe_type"].astype(str).isin(keep)].copy()

    clean = g[g["is_clean"]].copy()
    corrupt = g[(~g["is_clean"]) & (g["actual_volume_change_pct"] > volume_threshold_pct)].copy()

    clean["y_true"] = 0
    corrupt["y_true"] = 1
    clean["class_label"] = "clean"
    corrupt["class_label"] = f"corrupt_vol_gt_{volume_threshold_pct:g}"

    dataset = pd.concat([clean, corrupt], ignore_index=True)
    if dataset["y_true"].nunique() < 2:
        raise ValueError(f"Need both classes. Counts: {dataset['y_true'].value_counts().to_dict()}")

    return dataset


def roc_curve_manual(y_true: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)

    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    distinct = np.r_[True, s[1:] != s[:-1]]
    idx = np.where(distinct)[0]

    thresholds = s[idx]
    tp = np.array([(y[:i + 1] == 1).sum() for i in idx], dtype=float)
    fp = np.array([(y[:i + 1] == 0).sum() for i in idx], dtype=float)

    tpr = tp / n_pos
    fpr = fp / n_neg

    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    thresholds = np.r_[np.inf, thresholds, -np.inf]
    return fpr, tpr, thresholds


def auc_pairwise(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]

    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    comp = pos[:, None] - neg[None, :]
    return float((np.sum(comp > 0) + 0.5 * np.sum(comp == 0)) / comp.size)


def standardized_partial_auc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    if len(fpr) == 0:
        return np.nan
    order = np.argsort(fpr)
    f = fpr[order]
    t = tpr[order]

    if max_fpr <= 0 or max_fpr > 1:
        return np.nan

    t_at_max = np.interp(max_fpr, f, t)
    mask = f < max_fpr
    f_clip = np.r_[f[mask], max_fpr]
    t_clip = np.r_[t[mask], t_at_max]

    raw = float(np.trapezoid(t_clip, f_clip))
    min_area = 0.5 * max_fpr * max_fpr
    max_area = max_fpr
    return float(0.5 + 0.5 * (raw - min_area) / (max_area - min_area))


def operating_points(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    fixed_specificities: List[float],
    fixed_sensitivities: List[float],
) -> Dict[str, float]:
    spec = 1.0 - fpr
    out: Dict[str, float] = {}

    youden = tpr - fpr
    j = int(np.nanargmax(youden))
    out["youden_index"] = float(youden[j])
    out["youden_threshold"] = float(thresholds[j])
    out["youden_sensitivity"] = float(tpr[j])
    out["youden_specificity"] = float(spec[j])

    for target_spec in fixed_specificities:
        suffix = int(round(100 * target_spec))
        feasible = np.where(spec >= target_spec)[0]
        if len(feasible) == 0:
            out[f"sens_at_{suffix}spec"] = np.nan
            out[f"threshold_at_{suffix}spec"] = np.nan
            out[f"actual_spec_at_{suffix}spec"] = np.nan
            continue
        best = feasible[np.nanargmax(tpr[feasible])]
        out[f"sens_at_{suffix}spec"] = float(tpr[best])
        out[f"threshold_at_{suffix}spec"] = float(thresholds[best])
        out[f"actual_spec_at_{suffix}spec"] = float(spec[best])

    for target_sens in fixed_sensitivities:
        suffix = int(round(100 * target_sens))
        feasible = np.where(tpr >= target_sens)[0]
        if len(feasible) == 0:
            out[f"spec_at_{suffix}sens"] = np.nan
            out[f"threshold_at_{suffix}sens"] = np.nan
            out[f"actual_sens_at_{suffix}sens"] = np.nan
            continue
        best = feasible[np.nanargmax(spec[feasible])]
        out[f"spec_at_{suffix}sens"] = float(spec[best])
        out[f"threshold_at_{suffix}sens"] = float(thresholds[best])
        out[f"actual_sens_at_{suffix}sens"] = float(tpr[best])

    return out


def bootstrap_auc(y_true: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]

    vals = []
    n = len(y)
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


def analyze_metric(
    dataset: pd.DataFrame,
    metric: str,
    fixed_specificities: List[float],
    fixed_sensitivities: List[float],
    pauc_max_fpr: float,
    n_boot: int,
    seed: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    sub = dataset[["y_true", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if sub["y_true"].nunique() < 2:
        raise ValueError(f"{metric}: only one class after dropping NaNs")

    y = sub["y_true"].astype(int).to_numpy()
    s = sub[metric].astype(float).to_numpy()

    fpr, tpr, thr = roc_curve_manual(y, s)
    auc = auc_pairwise(y, s)
    bmean, blo, bhi = bootstrap_auc(y, s, n_boot=n_boot, seed=seed)
    ops = operating_points(fpr, tpr, thr, fixed_specificities, fixed_sensitivities)
    pauc = standardized_partial_auc(fpr, tpr, max_fpr=pauc_max_fpr)

    row = {
        "metric": metric,
        "metric_display": DISPLAY_NAMES.get(metric, metric),
        "n_clean": int((y == 0).sum()),
        "n_corrupt": int((y == 1).sum()),
        "auc": auc,
        "auc_boot_mean": bmean,
        "auc_ci_low": blo,
        "auc_ci_high": bhi,
        f"pauc_fpr_le_{pauc_max_fpr:g}": pauc,
        "pauc_fpr_le_0.05": pauc if abs(pauc_max_fpr - 0.05) < 1e-12 else np.nan,
        **ops,
    }

    curve = pd.DataFrame({
        "metric": metric,
        "metric_display": DISPLAY_NAMES.get(metric, metric),
        "fpr": fpr,
        "specificity": 1.0 - fpr,
        "tpr_sensitivity": tpr,
        "threshold": thr,
    })

    return row, curve


def run_roc_analysis(
    dataset: pd.DataFrame,
    metrics: List[str],
    fixed_specificities: List[float],
    fixed_sensitivities: List[float],
    pauc_max_fpr: float,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = []
    for metric in metrics:
        try:
            row, curve = analyze_metric(
                dataset=dataset,
                metric=metric,
                fixed_specificities=fixed_specificities,
                fixed_sensitivities=fixed_sensitivities,
                pauc_max_fpr=pauc_max_fpr,
                n_boot=n_boot,
                seed=seed,
            )
            rows.append(row)
            curves.append(curve)
        except Exception as e:
            print(f"WARNING: skipped {metric}: {e}")

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No metrics could be analyzed.")
    summary = summary.sort_values(["auc", "sens_at_95spec"], ascending=[False, False]).reset_index(drop=True)
    summary["rank_by_auc"] = np.arange(1, len(summary) + 1)

    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return summary, curve_df


def plot_roc(curves: pd.DataFrame, summary: pd.DataFrame, out_path: Path, roi: str, volume_threshold_pct: float):
    plt.figure(figsize=(6.4, 6.0))
    for metric, g in curves.groupby("metric"):
        disp = DISPLAY_NAMES.get(metric, metric)
        auc_vals = summary.loc[summary["metric"] == metric, "auc"]
        label = f"{disp} (AUC={float(auc_vals.iloc[0]):.3f})" if len(auc_vals) else disp
        gg = g.sort_values("fpr")
        plt.plot(gg["fpr"], gg["tpr_sensitivity"], linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("Sensitivity")
    plt.title(f"{roi}: ROC curves, volume change > {volume_threshold_pct:g}%")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metric_ranking(summary: pd.DataFrame, out_path: Path):
    df = summary.sort_values("auc", ascending=True)
    y = np.arange(len(df))
    plt.figure(figsize=(7.8, max(3.8, 0.55 * len(df))))
    plt.barh(y, df["auc"])
    plt.yticks(y, df["metric_display"])
    plt.xlabel("ROC AUC")
    plt.title("Metric ranking by AUC")
    plt.xlim(0, 1.0)
    for i, (_, row) in enumerate(df.iterrows()):
        label = f"{row['auc']:.3f}"
        if np.isfinite(row.get("auc_ci_low", np.nan)):
            label += f" [{row['auc_ci_low']:.2f}, {row['auc_ci_high']:.2f}]"
        plt.text(min(row["auc"] + 0.01, 0.98), i, label, va="center", fontsize=8)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_score_distributions(dataset: pd.DataFrame, metrics: List[str], out_path: Path, volume_threshold_pct: float):
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
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.suptitle(f"Clean vs corrupted score distributions, volume change > {volume_threshold_pct:g}%", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_operating_point_tradeoff(summary: pd.DataFrame, out_path: Path):
    cols = [
        "sens_at_95spec",
        "sens_at_90spec",
        "spec_at_95sens",
        "spec_at_90sens",
    ]
    present = [c for c in cols if c in summary.columns]
    if not present:
        return

    x = np.arange(len(summary))
    width = 0.8 / len(present)

    plt.figure(figsize=(9.0, 5.2))
    for i, col in enumerate(present):
        vals = summary[col].astype(float).to_numpy()
        plt.bar(x + (i - (len(present) - 1) / 2) * width, vals, width=width, label=PUB_COLS.get(col, col))

    plt.xticks(x, summary["metric_display"], rotation=25, ha="right")
    plt.ylabel("Rate")
    plt.ylim(0, 1.02)
    plt.title("High-specificity and high-sensitivity operating points")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def markdown_table(summary: pd.DataFrame) -> str:
    cols = [
        "metric_display", "auc", "auc_ci_low", "auc_ci_high",
        "sens_at_95spec", "threshold_at_95spec",
        "sens_at_90spec", "threshold_at_90spec",
        "spec_at_95sens", "threshold_at_95sens",
        "spec_at_90sens", "threshold_at_90sens",
        "youden_threshold", "youden_sensitivity", "youden_specificity",
    ]
    avail = [c for c in cols if c in summary.columns]
    table = summary[avail].copy().rename(columns=PUB_COLS)
    return table.to_markdown(index=False, floatfmt=".3f")


def write_report(
    out_path: Path,
    roi: str,
    volume_threshold_pct: float,
    dataset: pd.DataFrame,
    summary: pd.DataFrame,
    figure_paths: Dict[str, str],
):
    best = summary.iloc[0]
    n_clean = int((dataset["y_true"] == 0).sum())
    n_corrupt = int((dataset["y_true"] == 1).sum())
    probe_counts = dataset.loc[dataset["y_true"] == 1, "probe_type"].astype(str).value_counts().to_dict()

    lines = []
    lines.append(f"# ROC analysis report: {roi}\n")
    lines.append("## Task definition\n")
    lines.append(f"- Negative class: clean `{roi}` contours.")
    lines.append(f"- Positive class: corrupted `{roi}` contours with input-vs-target volume change > `{volume_threshold_pct:g}%`.")
    lines.append("- Detector scores are reconstruction discrepancy metrics; larger values indicate more suspicious contours.\n")

    lines.append("## Dataset\n")
    lines.append(f"- Clean samples: `{n_clean}`")
    lines.append(f"- Corrupted samples: `{n_corrupt}`")
    if probe_counts:
        lines.append("- Corrupted samples by probe type:")
        for k, v in probe_counts.items():
            lines.append(f"  - `{k}`: {v}")
    lines.append("")

    lines.append("## Main ROC comparison\n")
    lines.append(
        f"The best metric by ROC AUC was **{best['metric_display']}** "
        f"(AUC = `{best['auc']:.3f}`, 95% bootstrap CI = "
        f"[`{best['auc_ci_low']:.3f}`, `{best['auc_ci_high']:.3f}`])."
    )
    lines.append(
        f"At high-specificity operating points, this metric achieved sensitivity "
        f"`{best.get('sens_at_95spec', np.nan):.3f}` at 95% specificity and "
        f"`{best.get('sens_at_90spec', np.nan):.3f}` at 90% specificity. "
        f"Conversely, when high sensitivity is required, the achievable specificity was "
        f"`{best.get('spec_at_95sens', np.nan):.3f}` at 95% sensitivity and "
        f"`{best.get('spec_at_90sens', np.nan):.3f}` at 90% sensitivity."
    )
    lines.append("")

    lines.append("## Metric summary\n")
    lines.append(markdown_table(summary))
    lines.append("")

    lines.append("## Figures\n")
    for title, rel in figure_paths.items():
        lines.append(f"### {title}\n")
        lines.append(f"![{title}]({rel})\n")

    lines.append("## Interpretation\n")
    lines.append(
        f"This ROC analysis evaluates clinically larger synthetic errors by restricting corrupted samples "
        f"to cases with >{volume_threshold_pct:g}% input-vs-target volume change. AUC provides the threshold-independent "
        f"ranking of detector scores. Sensitivity at fixed specificity estimates how many corrupted contours are detected "
        f"when false QA alarms among clean contours are constrained. Specificity at fixed sensitivity estimates the clean-contour "
        f"pass rate required to catch a high fraction of corrupted contours."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Clean ROC analysis for Lung_L or another ROI.")
    p.add_argument("--csvs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--roi", default="Lung_L")
    p.add_argument("--volume_threshold_pct", type=float, default=20.0)
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--probe_types", nargs="+", default=None)
    p.add_argument("--include_shift", action="store_true")
    p.add_argument("--fixed_specificities", nargs="+", type=float, default=[0.90, 0.95])
    p.add_argument("--fixed_sensitivities", nargs="+", type=float, default=[0.90, 0.95])
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

    df = add_required_columns(load_csvs(args.csvs))
    metrics = choose_metrics(df, args.metrics)

    dataset = build_dataset(
        df=df,
        roi=args.roi,
        volume_threshold_pct=args.volume_threshold_pct,
        exclude_shift=not args.include_shift,
        probe_types=args.probe_types,
    )
    dataset.to_csv(tables_dir / "analysis_dataset.csv", index=False)

    summary, curves = run_roc_analysis(
        dataset=dataset,
        metrics=metrics,
        fixed_specificities=args.fixed_specificities,
        fixed_sensitivities=args.fixed_sensitivities,
        pauc_max_fpr=args.pauc_max_fpr,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    summary.to_csv(tables_dir / "roc_summary_by_metric.csv", index=False)
    curves.to_csv(tables_dir / "roc_curve_points.csv", index=False)

    # Convenience tables.
    op_cols = [c for c in summary.columns if c.startswith(("sens_at_", "spec_at_", "threshold_at_", "actual_", "youden_"))]
    summary[["metric", "metric_display"] + op_cols].to_csv(tables_dir / "threshold_operating_points.csv", index=False)
    summary[["metric", "metric_display", "auc", "auc_boot_mean", "auc_ci_low", "auc_ci_high"]].to_csv(tables_dir / "bootstrap_auc_ci.csv", index=False)

    plot_roc(curves, summary, figs_dir / "roc_curves_all_metrics.png", args.roi, args.volume_threshold_pct)
    plot_metric_ranking(summary, figs_dir / "metric_ranking_barplot.png")
    plot_score_distributions(dataset, metrics, figs_dir / "score_distribution_clean_vs_corrupt.png", args.volume_threshold_pct)
    plot_operating_point_tradeoff(summary, figs_dir / "operating_point_tradeoff.png")

    figure_paths = {
        "ROC curves": "../figures/roc_curves_all_metrics.png",
        "Metric ranking": "../figures/metric_ranking_barplot.png",
        "Score distributions": "../figures/score_distribution_clean_vs_corrupt.png",
        "Operating point tradeoff": "../figures/operating_point_tradeoff.png",
    }
    write_report(
        out_path=report_dir / f"{sanitize(args.roi)}_ROC_report.md",
        roi=args.roi,
        volume_threshold_pct=args.volume_threshold_pct,
        dataset=dataset,
        summary=summary,
        figure_paths=figure_paths,
    )

    print(f"Metrics analyzed: {metrics}")
    print(f"Wrote: {out_dir}")
    print(f"Summary: {tables_dir / 'roc_summary_by_metric.csv'}")
    print(f"Report: {report_dir / f'{sanitize(args.roi)}_ROC_report.md'}")


if __name__ == "__main__":
    main()
