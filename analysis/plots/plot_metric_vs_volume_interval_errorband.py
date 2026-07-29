#!/usr/bin/env python3
"""
Plot detector metric vs actual volume change using interval-aggregated x-axis
and error bands.

Input can be:
  - probe_eval_sweep_detailed.csv
  - probe_eval_sweep_with_detector_scores.csv
  - input_with_volume_and_scores.csv

Outputs:
  metric_vs_volume_interval_input_rows.csv
  metric_vs_volume_interval_summary.csv
  plots/single_roi_probe_metric/
  plots/by_probe/
  plots/by_metric/
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_METRICS = [
    "one_minus_dice_input_vs_recon",
    "dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
    "abs_input_recon_volume_diff_frac",
    "dice_input_vs_target",
    "dice_recon_vs_target",
    "hd95_input_vs_target",
    "hd95_recon_vs_target",
    "assd_input_vs_target",
    "assd_recon_vs_target",
]

PROBE_ORDER = [
    "shifted", "eroded", "dilated",
    "cut_lower", "cut_upper", "cut_posterior", "cut_anterior", "cut_left", "cut_right",
    "cut_lower_half", "cut_upper_half", "cut_posterior_half",
    "big_bite", "big_blob",
]


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "value"


def order_existing(values, preferred):
    values = list(values)
    ordered = [v for v in preferred if v in set(values)]
    ordered.extend([v for v in values if v not in set(ordered)])
    return ordered


def load_csvs(csvs: List[str]) -> pd.DataFrame:
    frames = []
    for raw in csvs:
        p = Path(raw)
        if not p.exists():
            print(f"WARNING: missing CSV, skipping: {p}")
            continue
        df = pd.read_csv(p)
        if df.empty:
            print(f"WARNING: empty CSV, skipping: {p}")
            continue
        df["source_csv"] = str(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid CSV files loaded.")
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame, max_volume_pct: Optional[float], min_volume_pct: float, exclude_shift: bool) -> pd.DataFrame:
    out = df.copy()
    if "roi" not in out.columns:
        out["roi"] = "unknown_roi"
    if "probe_type" not in out.columns:
        raise ValueError("Input CSV must contain probe_type column.")

    for c in ["input_voxels", "target_voxels", "recon_voxels"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "one_minus_dice_input_vs_recon" not in out.columns and "dice_input_vs_recon" in out.columns:
        out["one_minus_dice_input_vs_recon"] = 1.0 - pd.to_numeric(out["dice_input_vs_recon"], errors="coerce")

    if "abs_input_recon_volume_diff_frac" not in out.columns and {"input_voxels", "recon_voxels"}.issubset(out.columns):
        denom = out["input_voxels"].replace(0, np.nan)
        out["abs_input_recon_volume_diff_frac"] = ((out["input_voxels"] - out["recon_voxels"]) / denom).abs()

    if "abs_input_volume_change_pct" in out.columns:
        out["actual_volume_change_pct"] = pd.to_numeric(out["abs_input_volume_change_pct"], errors="coerce")
    elif "abs_input_volume_change_frac" in out.columns:
        out["actual_volume_change_pct"] = 100.0 * pd.to_numeric(out["abs_input_volume_change_frac"], errors="coerce")
    elif {"input_voxels", "target_voxels"}.issubset(out.columns):
        denom = out["target_voxels"].replace(0, np.nan)
        out["actual_volume_change_pct"] = 100.0 * ((out["input_voxels"] - out["target_voxels"]) / denom).abs()
    else:
        raise ValueError("Need abs_input_volume_change_pct, abs_input_volume_change_frac, or input_voxels/target_voxels.")

    out["actual_volume_change_pct"] = pd.to_numeric(out["actual_volume_change_pct"], errors="coerce")
    out = out[np.isfinite(out["actual_volume_change_pct"])].copy()

    if exclude_shift:
        out = out[out["probe_type"].astype(str) != "shifted"].copy()
    out = out[out["actual_volume_change_pct"] >= min_volume_pct].copy()
    if max_volume_pct is not None:
        out = out[out["actual_volume_change_pct"] <= max_volume_pct].copy()
    return out


def choose_metrics(df: pd.DataFrame, metrics: Optional[List[str]]) -> List[str]:
    if metrics:
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise ValueError(f"Requested metrics missing from input: {missing}")
        return metrics
    return [m for m in DEFAULT_METRICS if m in df.columns]


def add_interval_bins(df: pd.DataFrame, bin_width_pct: float) -> pd.DataFrame:
    if bin_width_pct <= 0:
        raise ValueError("--bin_width_pct must be positive.")
    out = df.copy()
    left = np.floor(out["actual_volume_change_pct"] / bin_width_pct) * bin_width_pct
    out["volume_bin_left_pct"] = left
    out["volume_bin_right_pct"] = left + bin_width_pct
    out["volume_bin_mid_pct"] = left + 0.5 * bin_width_pct
    out["volume_bin_label"] = out.apply(lambda r: f"{r['volume_bin_left_pct']:.1f}-{r['volume_bin_right_pct']:.1f}%", axis=1)
    return out


def summarize(df: pd.DataFrame, metrics: List[str], y_stat: str, error_band: str, min_n_per_bin: int) -> pd.DataFrame:
    rows = []
    group_cols = ["roi", "probe_type", "volume_bin_left_pct", "volume_bin_right_pct", "volume_bin_mid_pct", "volume_bin_label"]
    for keys, g in df.groupby(group_cols, dropna=False):
        base = {
            "roi": keys[0],
            "probe_type": keys[1],
            "volume_bin_left_pct": float(keys[2]),
            "volume_bin_right_pct": float(keys[3]),
            "volume_bin_mid_pct": float(keys[4]),
            "volume_bin_label": keys[5],
            "actual_volume_pct_mean": float(g["actual_volume_change_pct"].mean()),
            "actual_volume_pct_median": float(g["actual_volume_change_pct"].median()),
            "n_rows_in_bin": int(len(g)),
        }
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            n = len(vals)
            if n < min_n_per_bin:
                continue
            center = float(np.mean(vals)) if y_stat == "mean" else float(np.median(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(n) if n else np.nan
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            if error_band == "sd":
                lo, hi = center - sd, center + sd
            elif error_band == "sem":
                lo, hi = center - sem, center + sem
            elif error_band == "ci95":
                lo, hi = center - 1.96 * sem, center + 1.96 * sem
            elif error_band == "iqr":
                lo, hi = q25, q75
            elif error_band == "none":
                lo, hi = center, center
            else:
                raise ValueError(error_band)
            rows.append({
                **base,
                "metric": metric,
                "n": int(n),
                "metric_center": center,
                "metric_mean": float(np.mean(vals)),
                "metric_median": float(np.median(vals)),
                "metric_sd": sd,
                "metric_sem": sem,
                "metric_q25": q25,
                "metric_q75": q75,
                "metric_band_low": float(lo),
                "metric_band_high": float(hi),
            })
    return pd.DataFrame(rows)


def apply_ylim(metric: str, user_ylim):
    if user_ylim is not None:
        plt.ylim(*user_ylim)
        return
    m = metric.lower()
    if "dice" in m:
        plt.ylim(-0.02, 1.02)
    elif "volume_diff" in m or "vol" in m or "hd95" in m or "assd" in m:
        plt.ylim(bottom=-0.02)


def plot_single(summary: pd.DataFrame, out_dir: Path, user_ylim, title_suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (roi, probe, metric), g in summary.groupby(["roi", "probe_type", "metric"], dropna=False):
        gg = g.sort_values("volume_bin_mid_pct")
        x = gg["volume_bin_mid_pct"].to_numpy(float)
        y = gg["metric_center"].to_numpy(float)
        lo = gg["metric_band_low"].to_numpy(float)
        hi = gg["metric_band_high"].to_numpy(float)
        plt.figure(figsize=(7.5, 5.2))
        plt.plot(x, y, marker="o", linewidth=2)
        if np.any(hi > lo):
            plt.fill_between(x, lo, hi, alpha=0.22)
        plt.xlabel("Actual volume-change interval midpoint (%)")
        plt.ylabel(metric)
        plt.title(f"{roi} | {probe}\n{metric}{title_suffix}")
        apply_ylim(metric, user_ylim)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"metric_vs_volume_interval__{sanitize(roi)}__{sanitize(probe)}__{sanitize(metric)}.png", dpi=300)
        plt.close()


def plot_by_probe(summary: pd.DataFrame, out_dir: Path, user_ylim, title_suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (roi, metric), g in summary.groupby(["roi", "metric"], dropna=False):
        probes = order_existing(g["probe_type"].unique(), PROBE_ORDER)
        plt.figure(figsize=(8.8, 5.8))
        for probe in probes:
            gg = g[g["probe_type"] == probe].sort_values("volume_bin_mid_pct")
            if gg.empty:
                continue
            x = gg["volume_bin_mid_pct"].to_numpy(float)
            y = gg["metric_center"].to_numpy(float)
            lo = gg["metric_band_low"].to_numpy(float)
            hi = gg["metric_band_high"].to_numpy(float)
            plt.plot(x, y, marker="o", linewidth=1.8, label=probe)
            if np.any(hi > lo):
                plt.fill_between(x, lo, hi, alpha=0.10)
        plt.xlabel("Actual volume-change interval midpoint (%)")
        plt.ylabel(metric)
        plt.title(f"{roi}: {metric} vs actual volume change{title_suffix}")
        apply_ylim(metric, user_ylim)
        plt.grid(alpha=0.3)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"metric_vs_volume_interval_by_probe__{sanitize(roi)}__{sanitize(metric)}.png", dpi=300)
        plt.close()


def plot_by_metric(summary: pd.DataFrame, out_dir: Path, user_ylim, title_suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (roi, probe), g in summary.groupby(["roi", "probe_type"], dropna=False):
        metrics = list(g["metric"].dropna().unique())
        plt.figure(figsize=(8.8, 5.8))
        for metric in metrics:
            gg = g[g["metric"] == metric].sort_values("volume_bin_mid_pct")
            if gg.empty:
                continue
            x = gg["volume_bin_mid_pct"].to_numpy(float)
            y = gg["metric_center"].to_numpy(float)
            lo = gg["metric_band_low"].to_numpy(float)
            hi = gg["metric_band_high"].to_numpy(float)
            plt.plot(x, y, marker="o", linewidth=1.8, label=metric)
            if np.any(hi > lo):
                plt.fill_between(x, lo, hi, alpha=0.10)
        plt.xlabel("Actual volume-change interval midpoint (%)")
        plt.ylabel("Metric value")
        plt.title(f"{roi} | {probe}: metric trends{title_suffix}")
        if user_ylim is not None:
            plt.ylim(*user_ylim)
        plt.grid(alpha=0.3)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"metric_vs_volume_interval_by_metric__{sanitize(roi)}__{sanitize(probe)}.png", dpi=300)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--rois", nargs="+", default=None)
    p.add_argument("--probe_types", nargs="+", default=None)
    p.add_argument("--bin_width_pct", type=float, default=5.0)
    p.add_argument("--min_volume_pct", type=float, default=0.0)
    p.add_argument("--max_volume_pct", type=float, default=None)
    p.add_argument("--min_n_per_bin", type=int, default=1)
    p.add_argument("--y_stat", choices=["mean", "median"], default="mean")
    p.add_argument("--error_band", choices=["sd", "sem", "ci95", "iqr", "none"], default="sd")
    p.add_argument("--exclude_shift", action="store_true")
    p.add_argument("--ylim", default=None, help="Optional fixed y limits, e.g. 0,1")
    p.add_argument("--no_by_probe", action="store_true")
    p.add_argument("--no_by_metric", action="store_true")
    args = p.parse_args()

    user_ylim = None
    if args.ylim:
        vals = [float(x) for x in args.ylim.split(",")]
        if len(vals) != 2:
            raise ValueError("--ylim should be like 0,1")
        user_ylim = tuple(vals)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare(load_csvs(args.csvs), args.max_volume_pct, args.min_volume_pct, args.exclude_shift)
    if args.rois:
        df = df[df["roi"].isin(args.rois)].copy()
    if args.probe_types:
        df = df[df["probe_type"].isin(args.probe_types)].copy()

    metrics = choose_metrics(df, args.metrics)
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    df = add_interval_bins(df, args.bin_width_pct)
    df.to_csv(out_dir / "metric_vs_volume_interval_input_rows.csv", index=False)

    summary = summarize(df, metrics, args.y_stat, args.error_band, args.min_n_per_bin)
    summary.to_csv(out_dir / "metric_vs_volume_interval_summary.csv", index=False)

    suffix = f"\n{args.y_stat} ± {args.error_band}, bin width {args.bin_width_pct:g}%"
    plot_single(summary, out_dir / "plots" / "single_roi_probe_metric", user_ylim, suffix)
    if not args.no_by_probe:
        plot_by_probe(summary, out_dir / "plots" / "by_probe", user_ylim, suffix)
    if not args.no_by_metric:
        plot_by_metric(summary, out_dir / "plots" / "by_metric", user_ylim, suffix)

    (out_dir / "README_metric_vs_volume_interval_plots.md").write_text(
        "# Metric vs actual volume-change interval plots\n\n"
        f"Rows are grouped by actual volume-change intervals of width `{args.bin_width_pct}`%.\n"
        "The plotted x value is the interval midpoint.\n"
        f"The plotted y value is `{args.y_stat}` metric value.\n"
        f"Error band: `{args.error_band}` across cases/rows within the same interval.\n\n"
        "This avoids duplicate x values caused by patient-specific actual volume change.\n"
        "For shift probes, volume change is not a meaningful severity axis; use `--exclude_shift` "
        "or analyze shift separately by displacement in voxels/mm.\n",
        encoding="utf-8",
    )

    print(f"Metrics: {metrics}")
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
