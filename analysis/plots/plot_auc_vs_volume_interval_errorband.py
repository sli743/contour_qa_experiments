#!/usr/bin/env python3
"""
Plot AUC vs actual volume-change percentage using interval-aggregated x axis
and error bands.

This fixes the issue where multiple points share the same requested corruption
level. Instead of plotting against requested_level, this script bins by actual
volume change percentage and summarizes AUC within each bin.

Input
-----
volume_auc_clean_vs_corruption_binned_long.csv
or input_volume_auc_copy.csv

Expected columns:
  roi
  probe_type
  score
  actual_volume_pct_median
  auc

Output
------
out_dir/
  auc_vs_volume_interval_summary.csv
  plots/
    auc_vs_volume_interval__<roi>__<probe>__<score>.png
    auc_vs_volume_interval_by_probe__<roi>__<score>.png

Definitions
-----------
For each ROI/probe/score, rows are grouped into actual-volume-change intervals:
  [0, bin_width), [bin_width, 2*bin_width), ...

For each interval:
  x = interval midpoint
  y = mean/median AUC
  error band = SD, SEM, IQR, or 95% CI across rows in that interval
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_SCORES = [
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
    "abs_input_recon_volume_diff_frac",
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


def load_csvs(csvs):
    frames = []
    for c in csvs:
        p = Path(c)
        if not p.exists():
            print(f"WARNING: missing {p}; skip")
            continue
        df = pd.read_csv(p)
        if df.empty:
            print(f"WARNING: empty {p}; skip")
            continue
        df["source_csv"] = str(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid CSVs loaded.")
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame, max_volume_pct: float | None, min_volume_pct: float) -> pd.DataFrame:
    out = df.copy()
    required = ["roi", "probe_type", "score", "auc"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "actual_volume_pct_median" not in out.columns:
        if "abs_input_volume_change_pct" in out.columns:
            out["actual_volume_pct_median"] = out["abs_input_volume_change_pct"]
        else:
            raise ValueError("Need actual_volume_pct_median or abs_input_volume_change_pct.")

    out["auc"] = pd.to_numeric(out["auc"], errors="coerce")
    out["actual_volume_pct_median"] = pd.to_numeric(out["actual_volume_pct_median"], errors="coerce")

    out = out[np.isfinite(out["auc"]) & np.isfinite(out["actual_volume_pct_median"])].copy()
    out = out[out["actual_volume_pct_median"] >= min_volume_pct].copy()
    if max_volume_pct is not None:
        out = out[out["actual_volume_pct_median"] <= max_volume_pct].copy()

    return out


def add_interval_bins(df: pd.DataFrame, bin_width_pct: float) -> pd.DataFrame:
    out = df.copy()
    if bin_width_pct <= 0:
        raise ValueError("--bin_width_pct must be positive.")

    left = np.floor(out["actual_volume_pct_median"] / bin_width_pct) * bin_width_pct
    out["volume_bin_left_pct"] = left
    out["volume_bin_right_pct"] = left + bin_width_pct
    out["volume_bin_mid_pct"] = left + 0.5 * bin_width_pct
    out["volume_bin_label"] = out.apply(
        lambda r: f"{r['volume_bin_left_pct']:.1f}-{r['volume_bin_right_pct']:.1f}%",
        axis=1,
    )
    return out


def summarize(df: pd.DataFrame, y_stat: str, error_band: str, min_n_per_bin: int) -> pd.DataFrame:
    rows = []
    group_cols = ["roi", "probe_type", "score", "volume_bin_left_pct", "volume_bin_right_pct", "volume_bin_mid_pct", "volume_bin_label"]

    for keys, g in df.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(g["auc"], errors="coerce").dropna().to_numpy()
        n = len(vals)
        if n < min_n_per_bin:
            continue

        if y_stat == "mean":
            center = float(np.mean(vals))
        elif y_stat == "median":
            center = float(np.median(vals))
        else:
            raise ValueError(f"Unknown y_stat: {y_stat}")

        sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = sd / np.sqrt(n) if n > 0 else np.nan
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
            raise ValueError(f"Unknown error_band: {error_band}")

        rows.append({
            "roi": keys[0],
            "probe_type": keys[1],
            "score": keys[2],
            "volume_bin_left_pct": float(keys[3]),
            "volume_bin_right_pct": float(keys[4]),
            "volume_bin_mid_pct": float(keys[5]),
            "volume_bin_label": keys[6],
            "n": int(n),
            "auc_center": center,
            "auc_mean": float(np.mean(vals)),
            "auc_median": float(np.median(vals)),
            "auc_sd": sd,
            "auc_sem": sem,
            "auc_q25": q25,
            "auc_q75": q75,
            "auc_band_low": max(0.0, float(lo)),
            "auc_band_high": min(1.0, float(hi)),
            "actual_volume_pct_mean": float(g["actual_volume_pct_median"].mean()),
            "actual_volume_pct_median": float(g["actual_volume_pct_median"].median()),
        })
    return pd.DataFrame(rows)


def choose_scores(df: pd.DataFrame, scores):
    available = set(df["score"].astype(str))
    if scores:
        missing = [s for s in scores if s not in available]
        if missing:
            raise ValueError(f"Requested scores missing: {missing}")
        return scores
    return [s for s in DEFAULT_SCORES if s in available]


def plot_single(summary: pd.DataFrame, out_dir: Path, auc_thresholds, ylim):
    out_dir.mkdir(parents=True, exist_ok=True)

    for (roi, probe, score), g in summary.groupby(["roi", "probe_type", "score"], dropna=False):
        gg = g.sort_values("volume_bin_mid_pct")
        if gg.empty:
            continue

        x = gg["volume_bin_mid_pct"].to_numpy(dtype=float)
        y = gg["auc_center"].to_numpy(dtype=float)
        lo = gg["auc_band_low"].to_numpy(dtype=float)
        hi = gg["auc_band_high"].to_numpy(dtype=float)

        plt.figure(figsize=(7.5, 5.2))
        plt.plot(x, y, marker="o", linewidth=2)
        if np.any(hi > lo):
            plt.fill_between(x, lo, hi, alpha=0.22)
        for t in auc_thresholds:
            plt.axhline(t, linestyle="--", linewidth=1.1, alpha=0.7)

        plt.xlabel("Actual volume-change interval midpoint (%)")
        plt.ylabel("AUC(clean vs corruption)")
        plt.title(f"{roi} | {probe}\n{score}")
        plt.ylim(*ylim)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"auc_vs_volume_interval__{sanitize(roi)}__{sanitize(probe)}__{sanitize(score)}.png", dpi=300)
        plt.close()


def plot_by_probe(summary: pd.DataFrame, out_dir: Path, auc_thresholds, ylim):
    out_dir.mkdir(parents=True, exist_ok=True)

    for (roi, score), g in summary.groupby(["roi", "score"], dropna=False):
        probes = order_existing(g["probe_type"].unique(), PROBE_ORDER)
        plt.figure(figsize=(8.5, 5.8))

        for probe in probes:
            gg = g[g["probe_type"] == probe].sort_values("volume_bin_mid_pct")
            if gg.empty:
                continue
            x = gg["volume_bin_mid_pct"].to_numpy(dtype=float)
            y = gg["auc_center"].to_numpy(dtype=float)
            lo = gg["auc_band_low"].to_numpy(dtype=float)
            hi = gg["auc_band_high"].to_numpy(dtype=float)
            plt.plot(x, y, marker="o", linewidth=1.8, label=probe)
            if np.any(hi > lo):
                plt.fill_between(x, lo, hi, alpha=0.10)

        for t in auc_thresholds:
            plt.axhline(t, linestyle="--", linewidth=1.1, alpha=0.7)

        plt.xlabel("Actual volume-change interval midpoint (%)")
        plt.ylabel("AUC(clean vs corruption)")
        plt.title(f"{roi}: AUC vs actual volume change\n{score}")
        plt.ylim(*ylim)
        plt.grid(alpha=0.3)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"auc_vs_volume_interval_by_probe__{sanitize(roi)}__{sanitize(score)}.png", dpi=300)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+", required=True, help="volume_auc_clean_vs_corruption_binned_long.csv or input_volume_auc_copy.csv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scores", nargs="+", default=None)
    p.add_argument("--rois", nargs="+", default=None)
    p.add_argument("--probe_types", nargs="+", default=None)
    p.add_argument("--bin_width_pct", type=float, default=5.0)
    p.add_argument("--min_volume_pct", type=float, default=0.0)
    p.add_argument("--max_volume_pct", type=float, default=None)
    p.add_argument("--min_n_per_bin", type=int, default=1)
    p.add_argument("--y_stat", choices=["mean", "median"], default="mean")
    p.add_argument("--error_band", choices=["sd", "sem", "ci95", "iqr", "none"], default="sd")
    p.add_argument("--auc_thresholds", default="0.8")
    p.add_argument("--ylim", default="0.45,1.02")
    p.add_argument("--no_by_probe", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_thresholds = [float(x) for x in args.auc_thresholds.split(",") if x.strip()]
    ylim = tuple(float(x) for x in args.ylim.split(","))
    if len(ylim) != 2:
        raise ValueError("--ylim should be like 0.45,1.02")

    df = load_csvs(args.csvs)
    df = prepare(df, max_volume_pct=args.max_volume_pct, min_volume_pct=args.min_volume_pct)

    scores = choose_scores(df, args.scores)
    df = df[df["score"].isin(scores)].copy()

    if args.rois:
        df = df[df["roi"].isin(args.rois)].copy()
    if args.probe_types:
        df = df[df["probe_type"].isin(args.probe_types)].copy()

    df = add_interval_bins(df, args.bin_width_pct)
    df.to_csv(out_dir / "auc_vs_volume_interval_input_rows.csv", index=False)

    summ = summarize(df, y_stat=args.y_stat, error_band=args.error_band, min_n_per_bin=args.min_n_per_bin)
    summ.to_csv(out_dir / "auc_vs_volume_interval_summary.csv", index=False)

    plot_single(summ, out_dir / "plots" / "single_roi_probe_score", auc_thresholds, ylim)
    if not args.no_by_probe:
        plot_by_probe(summ, out_dir / "plots" / "by_probe", auc_thresholds, ylim)

    (out_dir / "README_auc_vs_volume_interval_plots.md").write_text(
        "# AUC vs actual volume-change interval plots\n\n"
        f"Input rows are grouped by actual volume-change intervals of width `{args.bin_width_pct}`%.\n"
        f"The plotted x value is the interval midpoint. The plotted y value is `{args.y_stat}` AUC.\n"
        f"Error band: `{args.error_band}` across AUC rows within the same interval.\n\n"
        "This removes duplicate x values caused by multiple actual-volume bins sharing the same requested corruption level.\n",
        encoding="utf-8",
    )

    print(f"Scores: {scores}")
    print(f"Wrote: {out_dir}")
    print(f"Summary CSV: {out_dir / 'auc_vs_volume_interval_summary.csv'}")


if __name__ == "__main__":
    main()
