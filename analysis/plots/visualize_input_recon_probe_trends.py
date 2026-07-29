#!/usr/bin/env python3
"""
Visualize clean-vs-corrupted score distributions and metric-vs-volume-change scatter plots.

Inputs:
  - probe_eval_sweep_detailed.csv
  - probe_eval_sweep_with_detector_scores.csv
  - input_with_volume_and_scores.csv

Outputs:
  score_distribution_summary.csv
  score_distribution_input_rows.csv
  paired_delta_summary.csv (if patient/file matching works)
  plots/hist_kde/
  plots/ecdf/
  plots/violin/
  plots/scatter_metric_vs_volume/
  plots/paired_delta/
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

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
    "clean", "shifted", "eroded", "dilated",
    "cut_lower", "cut_upper", "cut_posterior", "cut_anterior", "cut_left", "cut_right",
    "cut_lower_half", "cut_upper_half", "cut_posterior_half",
    "big_bite", "big_blob",
]


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
        raise FileNotFoundError("No valid CSVs loaded.")
    return pd.concat(frames, ignore_index=True)


def add_basic_columns(df: pd.DataFrame, exclude_shift: bool) -> pd.DataFrame:
    out = df.copy()
    if "roi" not in out.columns:
        out["roi"] = "unknown_roi"
    if "probe_type" not in out.columns:
        raise ValueError("CSV must contain probe_type.")

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
        out["actual_volume_change_pct"] = np.nan

    if "requested_level_numeric" in out.columns:
        level = pd.to_numeric(out["requested_level_numeric"], errors="coerce").fillna(0)
    elif "requested_level" in out.columns:
        level = pd.to_numeric(out["requested_level"], errors="coerce").fillna(0)
    elif "level" in out.columns:
        level = pd.to_numeric(out["level"], errors="coerce").fillna(0)
    else:
        level = pd.Series(0, index=out.index)

    out["requested_level_numeric"] = level
    out["is_clean"] = (out["probe_type"].astype(str) == "clean") | (out["requested_level_numeric"] == 0)
    out.loc[out["is_clean"], "actual_volume_change_pct"] = 0.0
    out["class_label"] = np.where(out["is_clean"], "clean", "corrupted")
    out["plot_probe_type"] = np.where(out["is_clean"], "clean", out["probe_type"].astype(str))

    if exclude_shift:
        out = out[(out["plot_probe_type"] != "shifted") | out["is_clean"]].copy()

    return out


def choose_metrics(df: pd.DataFrame, metrics: Optional[List[str]]) -> List[str]:
    if metrics:
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise ValueError(f"Missing requested metrics: {missing}")
        return metrics
    return [m for m in DEFAULT_METRICS if m in df.columns]


def clean_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def auc_pairwise(clean: np.ndarray, corrupt: np.ndarray) -> float:
    clean = clean[np.isfinite(clean)]
    corrupt = corrupt[np.isfinite(corrupt)]
    if len(clean) == 0 or len(corrupt) == 0:
        return np.nan
    comp = corrupt[:, None] - clean[None, :]
    return float((np.sum(comp > 0) + 0.5 * np.sum(comp == 0)) / comp.size)


def cohen_d(clean: np.ndarray, corrupt: np.ndarray) -> float:
    clean = clean[np.isfinite(clean)]
    corrupt = corrupt[np.isfinite(corrupt)]
    if len(clean) < 2 or len(corrupt) < 2:
        return np.nan
    pooled = ((len(clean)-1)*np.var(clean, ddof=1) + (len(corrupt)-1)*np.var(corrupt, ddof=1)) / (len(clean)+len(corrupt)-2)
    return float((np.mean(corrupt) - np.mean(clean)) / np.sqrt(pooled)) if pooled > 0 else np.nan


def overlap_coefficient_hist(clean: np.ndarray, corrupt: np.ndarray, bins: int = 40) -> float:
    clean = clean[np.isfinite(clean)]
    corrupt = corrupt[np.isfinite(corrupt)]
    if len(clean) == 0 or len(corrupt) == 0:
        return np.nan
    mn = min(float(np.min(clean)), float(np.min(corrupt)))
    mx = max(float(np.max(clean)), float(np.max(corrupt)))
    if mx <= mn:
        return 1.0
    h1, edges = np.histogram(clean, bins=bins, range=(mn, mx), density=True)
    h2, _ = np.histogram(corrupt, bins=edges, density=True)
    return float(np.sum(np.minimum(h1, h2)) * (edges[1] - edges[0]))


def distribution_summary(df: pd.DataFrame, metrics: List[str], hist_bins: int) -> pd.DataFrame:
    rows = []
    for roi, g_roi in df.groupby("roi", dropna=False):
        clean_base = g_roi[g_roi["is_clean"]].copy()
        corrupt_all = g_roi[~g_roi["is_clean"]].copy()
        for probe, g_probe in corrupt_all.groupby("plot_probe_type", dropna=False):
            for metric in metrics:
                clean = clean_numeric(clean_base[metric])
                corrupt = clean_numeric(g_probe[metric])
                if len(clean) == 0 or len(corrupt) == 0:
                    continue
                rows.append({
                    "roi": roi,
                    "probe_type": probe,
                    "metric": metric,
                    "n_clean": int(len(clean)),
                    "n_corrupt": int(len(corrupt)),
                    "clean_mean": float(np.mean(clean)),
                    "clean_median": float(np.median(clean)),
                    "clean_sd": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
                    "corrupt_mean": float(np.mean(corrupt)),
                    "corrupt_median": float(np.median(corrupt)),
                    "corrupt_sd": float(np.std(corrupt, ddof=1)) if len(corrupt) > 1 else 0.0,
                    "auc": auc_pairwise(clean, corrupt),
                    "cohen_d": cohen_d(clean, corrupt),
                    "overlap_coefficient": overlap_coefficient_hist(clean, corrupt, bins=hist_bins),
                })
    return pd.DataFrame(rows)


def ecdf(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.sort(vals[np.isfinite(vals)])
    if len(vals) == 0:
        return vals, vals
    return vals, np.arange(1, len(vals)+1) / len(vals)


def plot_hist_kde(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, bins: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in summary.iterrows():
        roi, probe, metric = row["roi"], row["probe_type"], row["metric"]
        g = df[(df["roi"] == roi) & ((df["plot_probe_type"] == probe) | df["is_clean"])].copy()
        clean = clean_numeric(g[g["is_clean"]][metric])
        corrupt = clean_numeric(g[(~g["is_clean"]) & (g["plot_probe_type"] == probe)][metric])
        if len(clean) == 0 or len(corrupt) == 0:
            continue

        plt.figure(figsize=(7.4, 5.2))
        plt.hist(clean, bins=bins, density=True, alpha=0.45, label=f"clean (n={len(clean)})")
        plt.hist(corrupt, bins=bins, density=True, alpha=0.45, label=f"{probe} (n={len(corrupt)})")
        try:
            pd.Series(clean).plot(kind="kde", linewidth=2)
            pd.Series(corrupt).plot(kind="kde", linewidth=2)
        except Exception:
            pass
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.title(f"{roi} | {probe} | {metric}\nAUC={row['auc']:.3f}, OVL={row['overlap_coefficient']:.3f}, d={row['cohen_d']:.2f}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_kde__{sanitize(roi)}__{sanitize(probe)}__{sanitize(metric)}.png", dpi=300)
        plt.close()



def plot_hist_kde_by_level(
    df: pd.DataFrame,
    metrics: List[str],
    out_dir: Path,
    bins: int,
    max_levels_per_plot: int,
):
    """
    For each ROI × corruption type × metric, plot clean distribution against
    separate corrupted distributions for each requested corruption level.

    This is different from plot_hist_kde(), which pools all corrupted levels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for roi, g_roi in df.groupby("roi", dropna=False):
        clean_df = g_roi[g_roi["is_clean"]].copy()
        corrupt_df = g_roi[~g_roi["is_clean"]].copy()
        if clean_df.empty or corrupt_df.empty:
            continue

        for probe, g_probe in corrupt_df.groupby("plot_probe_type", dropna=False):
            levels = sorted(
                [
                    x for x in pd.to_numeric(g_probe["requested_level_numeric"], errors="coerce").dropna().unique()
                    if x > 0
                ]
            )
            if max_levels_per_plot > 0:
                levels = levels[:max_levels_per_plot]
            if not levels:
                continue

            for metric in metrics:
                clean = clean_numeric(clean_df[metric])
                if len(clean) == 0:
                    continue

                plt.figure(figsize=(8.2, 5.6))

                # Clean reference distribution.
                plt.hist(
                    clean,
                    bins=bins,
                    density=True,
                    alpha=0.35,
                    label=f"clean (n={len(clean)})",
                )
                try:
                    pd.Series(clean).plot(kind="kde", linewidth=2.2, label="clean KDE")
                except Exception:
                    pass

                # One distribution per corruption level.
                level_summaries = []
                for lv in levels:
                    q = g_probe[np.isclose(pd.to_numeric(g_probe["requested_level_numeric"], errors="coerce"), lv)]
                    vals = clean_numeric(q[metric])
                    if len(vals) == 0:
                        continue

                    plt.hist(
                        vals,
                        bins=bins,
                        density=True,
                        alpha=0.18,
                        label=f"level={lv:g} (n={len(vals)})",
                    )
                    try:
                        pd.Series(vals).plot(kind="kde", linewidth=1.6, label=f"level={lv:g} KDE")
                    except Exception:
                        pass

                    level_summaries.append(
                        f"{lv:g}: AUC={auc_pairwise(clean, vals):.2f}, OVL={overlap_coefficient_hist(clean, vals, bins=bins):.2f}"
                    )

                plt.xlabel(metric)
                plt.ylabel("Density")
                subtitle = "; ".join(level_summaries[:4])
                if len(level_summaries) > 4:
                    subtitle += "; ..."
                plt.title(
                    f"{roi} | {probe} | {metric}\n"
                    f"Clean vs separate corruption levels\n{subtitle}"
                )
                plt.grid(alpha=0.3)
                plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
                plt.tight_layout()
                plt.savefig(
                    out_dir / f"hist_kde_by_level__{sanitize(roi)}__{sanitize(probe)}__{sanitize(metric)}.png",
                    dpi=300,
                )
                plt.close()

def plot_ecdf(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in summary.iterrows():
        roi, probe, metric = row["roi"], row["probe_type"], row["metric"]
        g = df[(df["roi"] == roi) & ((df["plot_probe_type"] == probe) | df["is_clean"])].copy()
        clean = clean_numeric(g[g["is_clean"]][metric])
        corrupt = clean_numeric(g[(~g["is_clean"]) & (g["plot_probe_type"] == probe)][metric])
        if len(clean) == 0 or len(corrupt) == 0:
            continue
        xc, yc = ecdf(clean)
        xq, yq = ecdf(corrupt)

        plt.figure(figsize=(7.4, 5.2))
        plt.step(xc, yc, where="post", linewidth=2, label="clean")
        plt.step(xq, yq, where="post", linewidth=2, label=probe)
        plt.xlabel(metric)
        plt.ylabel("ECDF")
        plt.title(f"{roi} | {probe} | {metric}\nAUC={row['auc']:.3f}, OVL={row['overlap_coefficient']:.3f}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"ecdf__{sanitize(roi)}__{sanitize(probe)}__{sanitize(metric)}.png", dpi=300)
        plt.close()


def plot_violin(df: pd.DataFrame, metrics: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for roi, g_roi in df.groupby("roi", dropna=False):
        groups = [p for p in PROBE_ORDER if p in set(g_roi["plot_probe_type"])]
        groups += [p for p in g_roi["plot_probe_type"].unique() if p not in groups]
        for metric in metrics:
            data, labels = [], []
            for group in groups:
                vals = clean_numeric(g_roi[g_roi["plot_probe_type"] == group][metric])
                if len(vals):
                    data.append(vals)
                    labels.append(group)
            if not data:
                continue
            plt.figure(figsize=(max(9, 0.55 * len(labels)), 5.6))
            plt.violinplot(data, showmeans=True, showmedians=True)
            plt.xticks(np.arange(1, len(labels)+1), labels, rotation=45, ha="right")
            plt.ylabel(metric)
            plt.title(f"{roi}: score distributions by corruption type\n{metric}")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"violin__{sanitize(roi)}__{sanitize(metric)}.png", dpi=300)
            plt.close()


def moving_trend(x: np.ndarray, y: np.ndarray, window: int, min_periods: int, band: str):
    """
    Moving-window trend after sorting by x.

    band:
      sd   = mean +/- rolling SD
      sem  = mean +/- rolling SEM
      ci95 = mean +/- 1.96 SEM
      iqr  = rolling 25th-75th percentile
      none = no band
    """
    order = np.argsort(x)
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]

    if len(xs) < min_periods:
        return None

    s = pd.Series(ys)
    rolling = s.rolling(window=window, center=True, min_periods=min_periods)

    y_mean = rolling.mean().to_numpy()
    y_sd = rolling.std().fillna(0).to_numpy()
    y_count = rolling.count().to_numpy()
    y_sem = np.divide(y_sd, np.sqrt(y_count), out=np.zeros_like(y_sd), where=y_count > 0)
    y_q25 = rolling.quantile(0.25).to_numpy()
    y_q75 = rolling.quantile(0.75).to_numpy()

    if band == "sd":
        lo, hi = y_mean - y_sd, y_mean + y_sd
    elif band == "sem":
        lo, hi = y_mean - y_sem, y_mean + y_sem
    elif band == "ci95":
        lo, hi = y_mean - 1.96 * y_sem, y_mean + 1.96 * y_sem
    elif band == "iqr":
        lo, hi = y_q25, y_q75
    elif band == "none":
        lo, hi = y_mean, y_mean
    else:
        raise ValueError(f"Unknown trend_band: {band}")

    valid = np.isfinite(xs) & np.isfinite(y_mean) & np.isfinite(lo) & np.isfinite(hi)
    if valid.sum() < min_periods:
        return None
    return xs[valid], y_mean[valid], lo[valid], hi[valid]


def moving_trend(x: np.ndarray, y: np.ndarray, window: int, min_periods: int, band: str):
    """
    Moving-window trend after sorting by x.

    band:
      sd   = mean +/- rolling SD
      sem  = mean +/- rolling SEM
      ci95 = mean +/- 1.96 SEM
      iqr  = rolling 25th-75th percentile
      none = no band
    """
    order = np.argsort(x)
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]

    if len(xs) < min_periods:
        return None

    s = pd.Series(ys)
    rolling = s.rolling(window=window, center=True, min_periods=min_periods)

    y_mean = rolling.mean().to_numpy()
    y_sd = rolling.std().fillna(0).to_numpy()
    y_count = rolling.count().to_numpy()
    y_sem = np.divide(y_sd, np.sqrt(y_count), out=np.zeros_like(y_sd), where=y_count > 0)
    y_q25 = rolling.quantile(0.25).to_numpy()
    y_q75 = rolling.quantile(0.75).to_numpy()

    if band == "sd":
        lo, hi = y_mean - y_sd, y_mean + y_sd
    elif band == "sem":
        lo, hi = y_mean - y_sem, y_mean + y_sem
    elif band == "ci95":
        lo, hi = y_mean - 1.96 * y_sem, y_mean + 1.96 * y_sem
    elif band == "iqr":
        lo, hi = y_q25, y_q75
    elif band == "none":
        lo, hi = y_mean, y_mean
    else:
        raise ValueError(f"Unknown trend_band: {band}")

    valid = np.isfinite(xs) & np.isfinite(y_mean) & np.isfinite(lo) & np.isfinite(hi)
    if valid.sum() < min_periods:
        return None
    return xs[valid], y_mean[valid], lo[valid], hi[valid]


def plot_scatter_metric_vs_volume(
    df: pd.DataFrame,
    metrics: List[str],
    out_dir: Path,
    marker_size: float,
    scatter_alpha: float,
    trend_window: int,
    trend_min_periods: int,
    trend_band: str,
    trend_by: str,
):
    """
    Scatter plot:
      x = input-vs-target volume change percentage
      y = selected input-vs-recon metric
      points = individual cases, color-coded by probe type
      trend/band = moving average and uncertainty band for each probe type

    Use trend_by='probe' for per-probe trend lines.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if trend_by not in {"class", "probe"}:
        raise ValueError("--trend_by must be 'class' or 'probe'")

    for roi, g_roi in df.groupby("roi", dropna=False):
        groups = [p for p in PROBE_ORDER if p in set(g_roi["plot_probe_type"])]
        groups += [p for p in g_roi["plot_probe_type"].unique() if p not in groups]

        for metric in metrics:
            plt.figure(figsize=(8.4, 5.9))

            # Scatter points colored by probe type. Use tiny markers.
            for group in groups:
                gg = g_roi[g_roi["plot_probe_type"] == group].copy()
                if gg.empty:
                    continue
                x = pd.to_numeric(gg["actual_volume_change_pct"], errors="coerce")
                y = pd.to_numeric(gg[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() == 0:
                    continue
                plt.scatter(
                    x[mask],
                    y[mask],
                    s=marker_size,
                    alpha=scatter_alpha if group != "clean" else min(1.0, scatter_alpha + 0.25),
                    label=group,
                    linewidths=0,
                )

            # Trend lines and error bands.
            if trend_by == "class":
                trend_groups = [
                    ("clean", g_roi[g_roi["is_clean"]]),
                    ("corrupted", g_roi[~g_roi["is_clean"]]),
                ]
            else:
                # Requested behavior: one trend per probe type.
                trend_groups = [(group, g_roi[g_roi["plot_probe_type"] == group]) for group in groups]

            for label, gg in trend_groups:
                if gg.empty:
                    continue
                x = pd.to_numeric(gg["actual_volume_change_pct"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(gg[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() == 0:
                    continue

                # Clean: all x=0, so a moving trend is not meaningful.
                # Show mean +/- SD as a reference error bar.
                if label == "clean" and np.nanmax(x[mask]) == 0:
                    y_clean = y[mask]
                    y0 = float(np.nanmean(y_clean))
                    sd = float(np.nanstd(y_clean, ddof=1)) if len(y_clean) > 1 else 0.0
                    plt.errorbar(
                        [0],
                        [y0],
                        yerr=[[sd], [sd]],
                        fmt="o",
                        markersize=max(2.0, marker_size + 1.0),
                        capsize=3,
                        linewidth=1.2,
                        label="clean mean ± SD",
                    )
                    continue

                tr = moving_trend(
                    x[mask],
                    y[mask],
                    window=trend_window,
                    min_periods=trend_min_periods,
                    band=trend_band,
                )
                if tr is None:
                    continue

                xs, ym, lo, hi = tr
                plt.plot(xs, ym, linewidth=2.0, label=f"{label} trend")
                if trend_band != "none":
                    plt.fill_between(xs, lo, hi, alpha=0.16)

            plt.xlabel("Input-vs-target volume change (%)")
            plt.ylabel(metric)
            plt.title(
                f"{roi}: {metric} vs input volume change\n"
                f"tiny scatter points; moving trend per {trend_by}; band={trend_band}"
            )
            if "dice" in metric.lower():
                plt.ylim(-0.02, 1.02)
            else:
                plt.ylim(bottom=-0.02)
            plt.grid(alpha=0.3)
            plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
            plt.tight_layout()
            plt.savefig(out_dir / f"scatter_metric_vs_volume_trend__{sanitize(roi)}__{sanitize(metric)}.png", dpi=300)
            plt.close()

def paired_delta_summary_and_plot(df: pd.DataFrame, metrics: List[str], out_dir: Path, id_cols: List[str]) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cols = [c for c in id_cols if c in df.columns]
    if not use_cols:
        return pd.DataFrame()
    rows = []
    clean_df = df[df["is_clean"]].copy()
    corrupt_df = df[~df["is_clean"]].copy()

    for roi in df["roi"].dropna().unique():
        c_roi = clean_df[clean_df["roi"] == roi]
        q_roi = corrupt_df[corrupt_df["roi"] == roi]
        for probe, q_probe in q_roi.groupby("plot_probe_type", dropna=False):
            for metric in metrics:
                c_small = c_roi[use_cols + [metric]].rename(columns={metric: "clean_score"})
                q_small = q_probe[use_cols + [metric]].rename(columns={metric: "corrupt_score"})
                merged = q_small.merge(c_small, on=use_cols, how="inner")
                if merged.empty:
                    continue
                vals = (pd.to_numeric(merged["corrupt_score"], errors="coerce") - pd.to_numeric(merged["clean_score"], errors="coerce")).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                if len(vals) == 0:
                    continue
                rows.append({
                    "roi": roi, "probe_type": probe, "metric": metric,
                    "n_pairs": int(len(vals)),
                    "mean_delta": float(np.mean(vals)),
                    "median_delta": float(np.median(vals)),
                    "frac_delta_gt_0": float(np.mean(vals > 0)),
                })
                plt.figure(figsize=(7.2, 5.0))
                plt.hist(vals, bins=30, alpha=0.75)
                plt.axvline(0, linestyle="--", linewidth=1.5)
                plt.xlabel(f"Delta {metric}: corrupted - clean")
                plt.ylabel("Count")
                plt.title(f"{roi} | {probe} | {metric}\nfrac(delta>0)={np.mean(vals>0):.2f}, median={np.median(vals):.3g}")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / f"paired_delta__{sanitize(roi)}__{sanitize(probe)}__{sanitize(metric)}.png", dpi=300)
                plt.close()
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--rois", nargs="+", default=None)
    p.add_argument("--probe_types", nargs="+", default=None)
    p.add_argument("--exclude_shift", action="store_true")
    p.add_argument("--hist_bins", type=int, default=40)
    p.add_argument("--max_plots_per_type", type=int, default=999999)
    p.add_argument("--paired_id_cols", nargs="+", default=["patient_id", "file"])

    # Scatter trend options
    p.add_argument("--scatter_marker_size", type=float, default=1.0)
    p.add_argument("--scatter_alpha", type=float, default=0.30)
    p.add_argument("--trend_window", type=int, default=25)
    p.add_argument("--trend_min_periods", type=int, default=5)
    p.add_argument("--trend_band", choices=["sd", "sem", "ci95", "iqr", "none"], default="sd")
    p.add_argument("--trend_by", choices=["class", "probe"], default="probe")

    # Histogram by corruption level
    p.add_argument("--hist_by_level", action="store_true")
    p.add_argument("--max_levels_per_hist", type=int, default=0,
                   help="Max positive levels to draw in each clean-vs-level histogram. 0 means all levels.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = add_basic_columns(load_csvs(args.csvs), exclude_shift=args.exclude_shift)
    if args.rois:
        df = df[df["roi"].isin(args.rois)].copy()
    if args.probe_types:
        keep = set(args.probe_types) | {"clean"}
        df = df[df["plot_probe_type"].isin(keep)].copy()

    metrics = choose_metrics(df, args.metrics)
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    df.to_csv(out_dir / "score_distribution_input_rows.csv", index=False)

    summary = distribution_summary(df, metrics, hist_bins=args.hist_bins)
    summary.to_csv(out_dir / "score_distribution_summary.csv", index=False)
    summary_for_plots = summary.head(args.max_plots_per_type).copy()

    plot_hist_kde(df, summary_for_plots, plot_dir / "hist_kde", bins=args.hist_bins)
    if args.hist_by_level:
        plot_hist_kde_by_level(
            df,
            metrics,
            plot_dir / "hist_kde_by_level",
            bins=args.hist_bins,
            max_levels_per_plot=args.max_levels_per_hist,
        )
    plot_ecdf(df, summary_for_plots, plot_dir / "ecdf")
    plot_violin(df, metrics, plot_dir / "violin")
    plot_scatter_metric_vs_volume(
        df,
        metrics,
        plot_dir / "scatter_metric_vs_volume",
        marker_size=args.scatter_marker_size,
        scatter_alpha=args.scatter_alpha,
        trend_window=args.trend_window,
        trend_min_periods=args.trend_min_periods,
        trend_band=args.trend_band,
        trend_by=args.trend_by,
    )

    delta = paired_delta_summary_and_plot(df, metrics, plot_dir / "paired_delta", id_cols=args.paired_id_cols)
    if not delta.empty:
        delta.to_csv(out_dir / "paired_delta_summary.csv", index=False)

    (out_dir / "README_score_distribution_analysis.md").write_text(
        "# Score distribution analysis\n\n"
        "Clean rows are defined as `probe_type == clean` or requested level equal to zero. "
        "Corrupted rows are non-clean rows with positive requested severity.\n\n"
        "The main summary reports AUC, Cohen's d, and a histogram-based overlap coefficient. "
        "If `--hist_by_level` is used, additional histograms compare clean scores with each corruption level separately for each ROI and corruption type. "
        "AUC is computed as `P(score_corrupt > score_clean) + 0.5 P(equal)`.\n\n"
        "Scatter plots show each metric versus input-vs-target volume change. Clean samples are assigned 0% volume change. "
        "Points are color-coded by corruption type. Trend lines are moving averages, with configurable error bands.\n\n"
        "For shift probes, volume change is not an appropriate severity axis; use `--exclude_shift` or analyze shift separately.\n",
        encoding="utf-8",
    )

    print(f"Metrics: {metrics}")
    print(f"Wrote: {out_dir}")
    print(f"Summary: {out_dir / 'score_distribution_summary.csv'}")


if __name__ == "__main__":
    main()
