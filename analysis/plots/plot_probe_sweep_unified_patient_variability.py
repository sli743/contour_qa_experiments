#!/usr/bin/env python3
"""
Unified probe-sweep plots with patient-variability-aware volume-change views.

Input: probe_eval_sweep_detailed.csv from eval_post_dae_probe_sitk_morph_sweep_v2.py.

Outputs:
  old_level_boxplots/                 box plots by requested corruption level
  old_level_trends/                   severity trends with error bands
  volume_change_binned_boxplots/      binned actual-volume-change boxplots
  volume_change_scatter_lowess/       actual-volume-change scatter + LOWESS
  input_target_vs_input_recon/        x=input-target metric, y=input-recon metric
  repair_gain_plots/                  repair/detector-gain curves
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_METRICS = [
    "dice_input_vs_target",
    "dice_input_vs_recon",
    "dice_recon_vs_target",
    "hd95_input_vs_target",
    "hd95_input_vs_recon",
    "hd95_recon_vs_target",
]

PREFERRED_PROBE_ORDER = [
    "clean",
    "shifted",
    "eroded",
    "dilated",
    "cut_lower",
    "cut_upper",
    "cut_posterior",
    "cut_anterior",
    "cut_left",
    "cut_right",
    "cut_lower_half",
    "cut_upper_half",
    "cut_posterior_half",
    "big_bite",
    "big_blob",
]


def sanitize(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s)
    return s.strip("_") or "value"


def ordered_probe_types(values: Sequence[str]) -> List[str]:
    vals = [str(v) for v in pd.unique(pd.Series(values).dropna())]
    ordered = [p for p in PREFERRED_PROBE_ORDER if p in vals]
    ordered.extend([p for p in vals if p not in ordered])
    return ordered


def errorbar_arg(mode: str):
    if mode == "sd":
        return "sd"
    if mode == "se":
        return "se"
    if mode == "ci95":
        return ("ci", 95)
    return None


def maybe_lineplot(data, x, y, errorbar, marker="o", hue=None):
    kwargs = dict(data=data, x=x, y=y, marker=marker)
    if hue is not None and hue in data.columns and data[hue].nunique(dropna=True) > 1:
        kwargs["hue"] = hue
    try:
        sns.lineplot(**kwargs, errorbar=errorbar)
    except Exception:
        if errorbar == "sd":
            sns.lineplot(**kwargs, ci="sd")
        elif isinstance(errorbar, tuple) and errorbar[0] == "ci":
            sns.lineplot(**kwargs, ci=errorbar[1])
        else:
            sns.lineplot(**kwargs, ci=None)


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["input_voxels", "target_voxels", "recon_voxels"]:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")

    denom = out["target_voxels"].replace(0, np.nan)
    out["signed_input_volume_change_frac"] = (out["input_voxels"] - out["target_voxels"]) / denom
    out["abs_input_volume_change_frac"] = out["signed_input_volume_change_frac"].abs()
    out["signed_input_volume_change_pct"] = 100.0 * out["signed_input_volume_change_frac"]
    out["abs_input_volume_change_pct"] = 100.0 * out["abs_input_volume_change_frac"]

    out["signed_recon_volume_change_frac"] = (out["recon_voxels"] - out["target_voxels"]) / denom
    out["abs_recon_volume_change_frac"] = out["signed_recon_volume_change_frac"].abs()
    out["signed_recon_volume_change_pct"] = 100.0 * out["signed_recon_volume_change_frac"]
    out["abs_recon_volume_change_pct"] = 100.0 * out["abs_recon_volume_change_frac"]

    out["requested_level_numeric"] = pd.to_numeric(out.get("requested_level", np.nan), errors="coerce").fillna(0)
    out["applied_level_numeric"] = pd.to_numeric(out.get("applied_level", np.nan), errors="coerce")
    out["level_numeric"] = pd.to_numeric(out.get("level", np.nan), errors="coerce")

    for c in DEFAULT_METRICS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if {"dice_recon_vs_target", "dice_input_vs_target"}.issubset(out.columns):
        out["dice_repair_gain"] = out["dice_recon_vs_target"] - out["dice_input_vs_target"]
    if {"hd95_input_vs_target", "hd95_recon_vs_target"}.issubset(out.columns):
        out["hd95_repair_gain"] = out["hd95_input_vs_target"] - out["hd95_recon_vs_target"]
    if {"dice_input_vs_recon", "dice_input_vs_target"}.issubset(out.columns):
        out["dice_detector_gap"] = out["dice_input_vs_recon"] - out["dice_input_vs_target"]
    if {"hd95_input_vs_recon", "hd95_input_vs_target"}.issubset(out.columns):
        out["hd95_detector_gap"] = out["hd95_input_vs_recon"] - out["hd95_input_vs_target"]
    return out


def finite(df: pd.DataFrame, cols: List[str], drop_inf: bool) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=cols)
    if drop_inf:
        mask = np.ones(len(out), dtype=bool)
        for c in cols:
            mask &= np.isfinite(out[c].to_numpy(dtype=float))
        out = out[mask].copy()
    return out


def add_volume_bins(df: pd.DataFrame, x_col: str, bin_width: float) -> pd.DataFrame:
    out = df.copy()
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out = out.dropna(subset=[x_col]).copy()
    if bin_width <= 0:
        out["volume_change_bin_center"] = out[x_col]
    else:
        out["volume_change_bin_center"] = (np.round(out[x_col] / bin_width) * bin_width).astype(float)
    out["volume_change_bin_label"] = out["volume_change_bin_center"].map(lambda x: f"{x:g}")
    return out


def plot_old_level_boxplots(df, out_dir: Path, metrics: List[str], drop_inf: bool, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for probe in ordered_probe_types(df["probe_type"]):
        sub_probe = df[df["probe_type"].astype(str) == probe].copy()
        for metric in metrics:
            if metric not in sub_probe.columns:
                continue
            sub = finite(sub_probe, ["requested_level_numeric", metric], drop_inf)
            if sub.empty:
                continue
            plt.figure(figsize=(9.2, 5.6))
            sns.boxplot(data=sub, x="requested_level_numeric", y=metric, showfliers=False)
            sns.stripplot(data=sub, x="requested_level_numeric", y=metric, color="black", alpha=0.18, size=2, jitter=0.15)
            plt.xlabel("Requested corruption level")
            plt.ylabel(metric.replace("_", " "))
            plt.title(f"{metric.replace('_', ' ')} by requested level | {probe}")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / f"{sanitize(probe)}__{sanitize(metric)}__box_by_requested_level.png", dpi=dpi)
            plt.close()


def plot_old_level_trends(df, out_dir: Path, metrics: List[str], drop_inf: bool, errorbar, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for probe in ordered_probe_types(df["probe_type"]):
        sub_probe = df[df["probe_type"].astype(str) == probe].copy()
        for metric in metrics:
            if metric not in sub_probe.columns:
                continue
            sub = finite(sub_probe, ["requested_level_numeric", metric], drop_inf)
            if sub.empty or sub["requested_level_numeric"].nunique() < 2:
                continue
            summ = sub.groupby(["probe_type", "requested_level_numeric"])[metric].agg(["mean", "std", "count", "median", "min", "max"]).reset_index()
            summ["metric"] = metric
            summaries.append(summ)
            plt.figure(figsize=(9.2, 5.6))
            maybe_lineplot(sub, "requested_level_numeric", metric, errorbar)
            plt.xlabel("Requested corruption level")
            plt.ylabel(metric.replace("_", " "))
            plt.title(f"{metric.replace('_', ' ')} trend by requested level | {probe}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"{sanitize(probe)}__{sanitize(metric)}__trend_by_requested_level.png", dpi=dpi)
            plt.close()
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out_dir / "old_level_trend_summary.csv", index=False)


def plot_volume_binned_boxplots(df, out_dir: Path, metrics: List[str], x_col: str, bin_width: float, drop_inf: bool, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    work = add_volume_bins(df, x_col, bin_width)
    summaries = []
    for probe in ordered_probe_types(work["probe_type"]):
        sub_probe = work[work["probe_type"].astype(str) == probe].copy()
        if sub_probe.empty:
            continue
        bin_order = sorted(pd.unique(sub_probe["volume_change_bin_center"].dropna()))
        labels = [f"{x:g}" for x in bin_order]
        for metric in metrics:
            if metric not in sub_probe.columns:
                continue
            sub = finite(sub_probe, ["volume_change_bin_center", metric], drop_inf)
            if sub.empty or sub["volume_change_bin_center"].nunique() < 2:
                continue
            sub["volume_change_bin_label"] = pd.Categorical(
                sub["volume_change_bin_center"].map(lambda x: f"{x:g}"), categories=labels, ordered=True
            )
            summ = sub.groupby(["probe_type", "volume_change_bin_center"])[metric].agg(["mean", "std", "count", "median", "min", "max"]).reset_index()
            summ["metric"] = metric
            summ["x_col"] = x_col
            summ["bin_width"] = bin_width
            summaries.append(summ)
            plt.figure(figsize=(max(9.5, 0.55 * len(labels)), 5.8))
            sns.boxplot(data=sub, x="volume_change_bin_label", y=metric, showfliers=False)
            sns.stripplot(data=sub, x="volume_change_bin_label", y=metric, color="black", alpha=0.18, size=2, jitter=0.15)
            plt.xlabel(f"Absolute input volume change bin (%) | width={bin_width:g}")
            plt.ylabel(metric.replace("_", " "))
            plt.title(f"{metric.replace('_', ' ')} by actual volume-change bin | {probe}")
            plt.xticks(rotation=35, ha="right")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / f"{sanitize(probe)}__{sanitize(metric)}__box_by_volume_change_bin.png", dpi=dpi)
            plt.close()
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out_dir / f"volume_binned_boxplot_summary__{sanitize(x_col)}.csv", index=False)


def plot_volume_scatter_lowess(df, out_dir: Path, metrics: List[str], x_col: str, drop_inf: bool, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for probe in ordered_probe_types(df["probe_type"]):
        sub_probe = df[df["probe_type"].astype(str) == probe].copy()
        for metric in metrics:
            if metric not in sub_probe.columns:
                continue
            sub = finite(sub_probe, [x_col, metric], drop_inf)
            if sub.empty or sub[x_col].nunique() < 3:
                continue
            plt.figure(figsize=(9.2, 5.8))
            sns.scatterplot(data=sub, x=x_col, y=metric, alpha=0.42, s=35, edgecolor=None)
            try:
                sns.regplot(data=sub, x=x_col, y=metric, scatter=False, lowess=True, line_kws={"linewidth": 2.2})
            except Exception:
                pass
            plt.xlabel("Absolute input volume change (%)")
            plt.ylabel(metric.replace("_", " "))
            plt.title(f"{metric.replace('_', ' ')} vs actual volume change | {probe}\nScatter + LOWESS trend")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"{sanitize(probe)}__{sanitize(metric)}__scatter_lowess_vs_volume_change.png", dpi=dpi)
            plt.close()


def plot_input_target_vs_input_recon(df, out_dir: Path, drop_inf: bool, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("dice_input_vs_target", "dice_input_vs_recon", "Dice(Input, Target)", "Dice(Input, Recon)", "dice_input_target_vs_input_recon"),
        ("hd95_input_vs_target", "hd95_input_vs_recon", "HD95(Input, Target)", "HD95(Input, Recon)", "hd95_input_target_vs_input_recon"),
    ]
    rows = []
    for probe in ordered_probe_types(df["probe_type"]):
        sub_probe = df[df["probe_type"].astype(str) == probe].copy()
        for x, y, xl, yl, stem in pairs:
            if x not in sub_probe.columns or y not in sub_probe.columns:
                continue
            sub = finite(sub_probe, [x, y], drop_inf)
            if len(sub) < 3:
                continue
            pearson = sub[[x, y]].corr(method="pearson").iloc[0, 1] if sub[x].nunique() > 1 and sub[y].nunique() > 1 else np.nan
            spearman = sub[[x, y]].corr(method="spearman").iloc[0, 1] if sub[x].nunique() > 1 and sub[y].nunique() > 1 else np.nan
            rows.append({"probe_type": probe, "x": x, "y": y, "n": len(sub), "pearson": pearson, "spearman": spearman})
            plt.figure(figsize=(7.4, 6.4))
            sns.scatterplot(data=sub, x=x, y=y, hue="requested_level_numeric", palette="viridis", alpha=0.75, s=55, edgecolor=None)
            lo = np.nanmin([sub[x].min(), sub[y].min()])
            hi = np.nanmax([sub[x].max(), sub[y].max()])
            if np.isfinite(lo) and np.isfinite(hi):
                pad = 0.03 * (hi - lo + 1e-6)
                plt.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="black", alpha=0.75)
                plt.xlim(lo-pad, hi+pad)
                plt.ylim(lo-pad, hi+pad)
            plt.xlabel(xl)
            plt.ylabel(yl)
            plt.title(f"{yl} vs {xl} | {probe}\nDashed line = identity/copying behavior")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"{sanitize(probe)}__{stem}.png", dpi=dpi)
            plt.close()
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "input_target_vs_input_recon_correlation_summary.csv", index=False)


def plot_repair_gain(df, out_dir: Path, x_col: str, bin_width: float, drop_inf: bool, errorbar, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    work = add_volume_bins(df, x_col, bin_width)
    gains = [
        ("dice_repair_gain", "Dice repair gain = Dice(Recon,Target) - Dice(Input,Target)"),
        ("hd95_repair_gain", "HD95 repair gain = HD95(Input,Target) - HD95(Recon,Target)"),
        ("dice_detector_gap", "Dice detector gap = Dice(Input,Recon) - Dice(Input,Target)"),
        ("hd95_detector_gap", "HD95 detector gap = HD95(Input,Recon) - HD95(Input,Target)"),
    ]
    summaries = []
    for probe in ordered_probe_types(work["probe_type"]):
        sub_probe = work[work["probe_type"].astype(str) == probe].copy()
        for y, ylabel in gains:
            if y not in sub_probe.columns:
                continue
            for x, xl, stem in [
                ("requested_level_numeric", "Requested corruption level", "by_requested_level"),
                ("volume_change_bin_center", "Absolute input volume change bin (%)", "by_volume_change_bin"),
            ]:
                sub = finite(sub_probe, [x, y], drop_inf)
                if sub.empty or sub[x].nunique() < 2:
                    continue
                summ = sub.groupby(["probe_type", x])[y].agg(["mean", "std", "count", "median", "min", "max"]).reset_index()
                summ["gain_metric"] = y
                summ["x_axis"] = x
                summaries.append(summ)
                plt.figure(figsize=(9.2, 5.7))
                maybe_lineplot(sub, x, y, errorbar)
                plt.axhline(0, linestyle="--", linewidth=1.4, color="black", alpha=0.75)
                plt.xlabel(xl)
                plt.ylabel(ylabel)
                plt.title(f"{ylabel} | {probe}")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / f"{sanitize(probe)}__{sanitize(y)}__{stem}.png", dpi=dpi)
                plt.close()
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out_dir / "repair_gain_summary.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--x_col", default="abs_input_volume_change_pct")
    p.add_argument("--volume_bin_width", type=float, default=2.5)
    p.add_argument("--error_band", choices=["sd", "se", "ci95", "none"], default="sd")
    p.add_argument("--drop_inf", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    sns.set_theme(style="whitegrid", context="talk")
    csv = Path(args.csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    df = add_columns(df)
    df.to_csv(out / "normalized_probe_sweep_with_volume_change_and_gains.csv", index=False)
    eb = errorbar_arg(args.error_band)

    plot_old_level_boxplots(df, out / "old_level_boxplots", args.metrics, args.drop_inf, args.dpi)
    plot_old_level_trends(df, out / "old_level_trends", args.metrics, args.drop_inf, eb, args.dpi)
    plot_volume_binned_boxplots(df, out / "volume_change_binned_boxplots", args.metrics, args.x_col, args.volume_bin_width, args.drop_inf, args.dpi)
    plot_volume_scatter_lowess(df, out / "volume_change_scatter_lowess", args.metrics, args.x_col, args.drop_inf, args.dpi)
    plot_input_target_vs_input_recon(df, out / "input_target_vs_input_recon", args.drop_inf, args.dpi)
    plot_repair_gain(df, out / "repair_gain_plots", args.x_col, args.volume_bin_width, args.drop_inf, eb, args.dpi)

    (out / "README.md").write_text(
        "# Unified probe sweep plots\n\n"
        f"Input CSV: `{csv}`\n\n"
        "Plot groups:\n"
        "- old_level_boxplots: requested corruption level boxplots.\n"
        "- old_level_trends: requested corruption level trends with error bands.\n"
        "- volume_change_binned_boxplots: actual volume-change bin boxplots.\n"
        "- volume_change_scatter_lowess: actual volume-change scatter + LOWESS trend.\n"
        "- input_target_vs_input_recon: input corruption metric vs AE response metric.\n"
        "- repair_gain_plots: repair/detector gain curves.\n\n"
        f"Volume bin width: {args.volume_bin_width}%\n"
        f"Error band for line plots: {args.error_band}\n",
        encoding="utf-8",
    )
    print(f"Read: {csv}")
    print(f"Wrote plots to: {out}")


if __name__ == "__main__":
    main()
