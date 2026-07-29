#!/usr/bin/env python3
"""
Compute first detectable actual volume-change percentage.

This runs on top of existing probe_eval_sweep_with_detector_scores.csv files
produced by compute_probe_auc_sensitivity.py / compute_probe_auc_sensitivity_assd.py.

For each ROI / run / probe type / detector score:
  1. Define clean score distribution from clean rows.
  2. For corrupted rows, compute actual absolute input volume change:
       abs_input_volume_change_pct = 100 * |input_voxels - target_voxels| / target_voxels
  3. Bin corrupted rows by actual volume-change percentage.
  4. Compute AUC(clean vs corrupted-in-bin) for each bin.
  5. Report the first/lowest volume-change bin where AUC >= threshold.

This avoids interpreting different probe-level units as if they were comparable.

Outputs
-------
out_dir/
  volume_auc_clean_vs_corruption_binned_long.csv
  first_detectable_volume_change_by_probe.csv
  first_detectable_volume_change_summary_by_roi_score.csv
  best_detector_by_roi_volume_threshold.csv
  plots/
    auc_vs_volume_change__<roi>__<probe>__<score>.png
    heatmap_median_detectable_volume_pct_by_roi_score.png
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DEPLOYABLE_SCORES = [
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
    "abs_input_recon_volume_diff_frac",
]

ROI_ORDER = [
    "A_Aorta", "A_Pulmonary", "BrachPlx_L", "BrachPlx_R", "Bronchus", "Carina",
    "Esophagus", "Heart", "Liver", "Lung_L", "Lung_R", "Ribs", "SpinalCord",
    "Stomach", "Trachea", "V_VenaCava_S", "NS_Chestwall",
]

PROBE_ORDER = [
    "shifted", "eroded", "dilated",
    "cut_lower", "cut_upper", "cut_posterior", "cut_anterior", "cut_left", "cut_right",
    "cut_lower_half", "cut_upper_half", "cut_posterior_half",
    "big_bite", "big_blob",
]


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "value"


def infer_run_name(path: Path) -> str:
    if path.name == "probe_eval_sweep_with_detector_scores.csv":
        return path.parent.name
    return path.stem


def infer_roi(text: str) -> str:
    low = str(text).lower()
    for roi in sorted(ROI_ORDER, key=len, reverse=True):
        if roi.lower() in low:
            return roi
    return "unknown_roi"


def infer_fov(text: str) -> str:
    m = re.search(r"(\d+)x(\d+)x(\d+)", str(text))
    return f"{m.group(1)}x{m.group(2)}x{m.group(3)}" if m else "unknown_fov"


def auc_pairwise(clean: np.ndarray, corrupt: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=float)
    corrupt = np.asarray(corrupt, dtype=float)
    clean = clean[np.isfinite(clean)]
    corrupt = corrupt[np.isfinite(corrupt)]
    if len(clean) == 0 or len(corrupt) == 0:
        return np.nan
    comp = corrupt[:, None] - clean[None, :]
    return float((np.sum(comp > 0) + 0.5 * np.sum(comp == 0)) / comp.size)


def cohen_d(clean: np.ndarray, corrupt: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=float)
    corrupt = np.asarray(corrupt, dtype=float)
    clean = clean[np.isfinite(clean)]
    corrupt = corrupt[np.isfinite(corrupt)]
    if len(clean) < 2 or len(corrupt) < 2:
        return np.nan
    pooled = math.sqrt(((len(clean)-1)*np.var(clean, ddof=1) + (len(corrupt)-1)*np.var(corrupt, ddof=1)) / (len(clean)+len(corrupt)-2))
    return float((np.mean(corrupt) - np.mean(clean)) / pooled) if pooled > 0 else np.nan


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
        run = infer_run_name(p)
        if "source_csv" not in df.columns:
            df["source_csv"] = str(p)
        if "run_name" not in df.columns:
            df["run_name"] = run
        if "roi" not in df.columns:
            df["roi"] = infer_roi(run)
        else:
            df["roi"] = df["roi"].fillna(infer_roi(run))
        if "fov" not in df.columns:
            df["fov"] = infer_fov(run)
        else:
            df["fov"] = df["fov"].fillna(infer_fov(run))
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid input CSVs were loaded.")
    return pd.concat(frames, ignore_index=True)


def add_volume_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["input_voxels", "target_voxels", "recon_voxels"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "abs_input_volume_change_frac" not in out.columns:
        if {"input_voxels", "target_voxels"}.issubset(out.columns):
            denom = out["target_voxels"].replace(0, np.nan)
            out["abs_input_volume_change_frac"] = ((out["input_voxels"] - out["target_voxels"]) / denom).abs()
        else:
            raise ValueError("Need either abs_input_volume_change_frac or input_voxels/target_voxels columns.")

    out["abs_input_volume_change_frac"] = pd.to_numeric(out["abs_input_volume_change_frac"], errors="coerce")
    out["abs_input_volume_change_pct"] = 100.0 * out["abs_input_volume_change_frac"]

    # Ensure deployable scores exist when possible.
    if "one_minus_dice_input_vs_recon" not in out.columns and "dice_input_vs_recon" in out.columns:
        out["one_minus_dice_input_vs_recon"] = 1.0 - pd.to_numeric(out["dice_input_vs_recon"], errors="coerce")

    if "abs_input_recon_volume_diff_frac" not in out.columns:
        if {"input_voxels", "recon_voxels"}.issubset(out.columns):
            denom = out["input_voxels"].replace(0, np.nan)
            out["abs_input_recon_volume_diff_frac"] = ((out["input_voxels"] - out["recon_voxels"]) / denom).abs()

    # Numeric conversion for any likely score.
    for c in DEFAULT_DEPLOYABLE_SCORES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "requested_level" in out.columns:
        out["requested_level_numeric"] = pd.to_numeric(out["requested_level"], errors="coerce").fillna(0)
    elif "level" in out.columns:
        out["requested_level_numeric"] = pd.to_numeric(out["level"], errors="coerce").fillna(0)
    else:
        out["requested_level_numeric"] = 0.0

    return out


def choose_scores(df: pd.DataFrame, requested: Optional[List[str]]) -> List[str]:
    if requested:
        missing = [s for s in requested if s not in df.columns]
        if missing:
            raise ValueError(f"Requested scores missing from input: {missing}")
        return requested
    return [s for s in DEFAULT_DEPLOYABLE_SCORES if s in df.columns]


def make_bins(df: pd.DataFrame, bin_width_pct: float, max_pct: Optional[float]) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out["abs_input_volume_change_pct"], errors="coerce")
    out = out[np.isfinite(x)].copy()
    x = out["abs_input_volume_change_pct"]

    if max_pct is not None:
        out = out[out["abs_input_volume_change_pct"] <= max_pct].copy()

    if bin_width_pct <= 0:
        out["volume_bin_center_pct"] = out["abs_input_volume_change_pct"]
        out["volume_bin_left_pct"] = out["abs_input_volume_change_pct"]
        out["volume_bin_right_pct"] = out["abs_input_volume_change_pct"]
    else:
        left = np.floor(out["abs_input_volume_change_pct"] / bin_width_pct) * bin_width_pct
        out["volume_bin_left_pct"] = left
        out["volume_bin_right_pct"] = left + bin_width_pct
        out["volume_bin_center_pct"] = left + 0.5 * bin_width_pct

    out["volume_bin_label"] = out.apply(
        lambda r: f"{r['volume_bin_left_pct']:.1f}-{r['volume_bin_right_pct']:.1f}%",
        axis=1,
    )
    return out


def compute_volume_auc(
    df: pd.DataFrame,
    scores: List[str],
    bin_width_pct: float,
    max_pct: Optional[float],
    min_clean: int,
    min_corrupt: int,
    clean_probe_type: str = "clean",
) -> pd.DataFrame:
    rows = []
    group_cols = ["source_csv", "run_name", "roi", "fov"]

    for keys, g in df.groupby(group_cols, dropna=False):
        info = dict(zip(group_cols, keys))

        clean = g[g["probe_type"].astype(str) == clean_probe_type].copy()
        if clean.empty:
            clean = g[g["requested_level_numeric"] == 0].copy()

        if clean.empty:
            continue

        for probe_type, gp in g.groupby("probe_type", dropna=False):
            probe_type = str(probe_type)
            if probe_type == clean_probe_type:
                continue

            corrupted = gp.copy()
            # Exclude zero/no-op corruption from volume-threshold analysis.
            corrupted = corrupted[corrupted["abs_input_volume_change_pct"] > 0].copy()
            if corrupted.empty:
                continue

            corrupted = make_bins(corrupted, bin_width_pct=bin_width_pct, max_pct=max_pct)

            for bin_center, gb in corrupted.groupby("volume_bin_center_pct", dropna=False):
                for score in scores:
                    c = pd.to_numeric(clean[score], errors="coerce").to_numpy(dtype=float)
                    q = pd.to_numeric(gb[score], errors="coerce").to_numpy(dtype=float)
                    c = c[np.isfinite(c)]
                    q = q[np.isfinite(q)]

                    ok = len(c) >= min_clean and len(q) >= min_corrupt

                    rows.append({
                        **info,
                        "probe_type": probe_type,
                        "score": score,
                        "volume_bin_center_pct": float(bin_center) if pd.notna(bin_center) else np.nan,
                        "volume_bin_left_pct": float(gb["volume_bin_left_pct"].iloc[0]),
                        "volume_bin_right_pct": float(gb["volume_bin_right_pct"].iloc[0]),
                        "volume_bin_label": str(gb["volume_bin_label"].iloc[0]),
                        "actual_volume_pct_mean": float(gb["abs_input_volume_change_pct"].mean()),
                        "actual_volume_pct_median": float(gb["abs_input_volume_change_pct"].median()),
                        "actual_volume_pct_min": float(gb["abs_input_volume_change_pct"].min()),
                        "actual_volume_pct_max": float(gb["abs_input_volume_change_pct"].max()),
                        "requested_level_mean": float(gb["requested_level_numeric"].mean()),
                        "requested_level_median": float(gb["requested_level_numeric"].median()),
                        "n_clean": int(len(c)),
                        "n_corrupt": int(len(q)),
                        "clean_mean": float(np.mean(c)) if len(c) else np.nan,
                        "corrupt_mean": float(np.mean(q)) if len(q) else np.nan,
                        "auc": auc_pairwise(c, q) if ok else np.nan,
                        "cohen_d": cohen_d(c, q) if ok else np.nan,
                    })

    return pd.DataFrame(rows)


def first_detectable_by_probe(volume_auc: pd.DataFrame, auc_threshold: float) -> pd.DataFrame:
    rows = []
    id_cols = ["source_csv", "run_name", "roi", "fov", "probe_type", "score"]

    for keys, g in volume_auc.groupby(id_cols, dropna=False):
        info = dict(zip(id_cols, keys))
        gg = g.copy()
        gg["auc"] = pd.to_numeric(gg["auc"], errors="coerce")
        gg["actual_volume_pct_median"] = pd.to_numeric(gg["actual_volume_pct_median"], errors="coerce")
        gg = gg.dropna(subset=["auc", "actual_volume_pct_median"]).sort_values("actual_volume_pct_median")

        hit = gg[gg["auc"] >= auc_threshold]
        if hit.empty:
            rows.append({
                **info,
                "auc_threshold": auc_threshold,
                "detected": False,
                "first_detectable_volume_change_pct": np.nan,
                "first_detectable_volume_bin_center_pct": np.nan,
                "first_detectable_auc": np.nan,
                "n_bins_tested": int(len(gg)),
            })
        else:
            first = hit.iloc[0]
            rows.append({
                **info,
                "auc_threshold": auc_threshold,
                "detected": True,
                "first_detectable_volume_change_pct": float(first["actual_volume_pct_median"]),
                "first_detectable_volume_bin_center_pct": float(first["volume_bin_center_pct"]),
                "first_detectable_volume_bin_label": str(first["volume_bin_label"]),
                "first_detectable_auc": float(first["auc"]),
                "n_bins_tested": int(len(gg)),
            })

    return pd.DataFrame(rows)


def summarize_by_roi_score(first_df: pd.DataFrame, volume_auc: pd.DataFrame, auc_threshold: float) -> pd.DataFrame:
    rows = []
    for (roi, score), g in first_df.groupby(["roi", "score"], dropna=False):
        detected = g[g["detected"]].copy()
        n_probe = int(g["probe_type"].nunique()) if "probe_type" in g.columns else int(len(g))
        n_detected = int(detected["probe_type"].nunique()) if "probe_type" in detected.columns else int(len(detected))

        auc_g = volume_auc[(volume_auc["roi"] == roi) & (volume_auc["score"] == score)].copy()
        auc_vals = pd.to_numeric(auc_g["auc"], errors="coerce").dropna()

        rows.append({
            "roi": roi,
            "score": score,
            "auc_threshold": auc_threshold,
            "n_probe_types": n_probe,
            "n_detected_probe_types": n_detected,
            "detection_coverage": float(n_detected / n_probe) if n_probe else np.nan,
            "median_first_detectable_volume_change_pct": float(detected["first_detectable_volume_change_pct"].median()) if not detected.empty else np.inf,
            "mean_first_detectable_volume_change_pct": float(detected["first_detectable_volume_change_pct"].mean()) if not detected.empty else np.inf,
            "min_first_detectable_volume_change_pct": float(detected["first_detectable_volume_change_pct"].min()) if not detected.empty else np.inf,
            "mean_binned_auc": float(auc_vals.mean()) if len(auc_vals) else np.nan,
            "median_binned_auc": float(auc_vals.median()) if len(auc_vals) else np.nan,
            "n_auc_bins": int(len(auc_vals)),
        })

    return pd.DataFrame(rows)


def rank_best_by_roi(summary: pd.DataFrame, selection_rule: str) -> pd.DataFrame:
    df = summary.copy()
    finite = df["median_first_detectable_volume_change_pct"].replace([np.inf, -np.inf], np.nan)
    max_finite = finite.max()
    penalty = max_finite + 5.0 if np.isfinite(max_finite) else 999.0
    df["volume_threshold_for_rank"] = finite.fillna(penalty)

    parts = []
    for roi, g in df.groupby("roi", dropna=False):
        gg = g.copy()
        gg["rank_volume_threshold"] = gg["volume_threshold_for_rank"].rank(method="min", ascending=True)
        gg["rank_detection_coverage"] = gg["detection_coverage"].rank(method="min", ascending=False)
        gg["rank_mean_auc"] = gg["mean_binned_auc"].rank(method="min", ascending=False)
        gg["composite_rank_score"] = (
            gg["rank_volume_threshold"] + gg["rank_detection_coverage"] + gg["rank_mean_auc"]
        ) / 3.0

        if selection_rule == "volume_threshold":
            gg = gg.sort_values(["volume_threshold_for_rank", "detection_coverage", "mean_binned_auc"], ascending=[True, False, False])
            gg["final_rank"] = np.arange(1, len(gg) + 1)
        elif selection_rule == "mean_auc":
            gg["final_rank"] = gg["mean_binned_auc"].rank(method="first", ascending=False)
        elif selection_rule == "composite":
            gg["final_rank"] = gg["composite_rank_score"].rank(method="first", ascending=True)
        else:
            raise ValueError(f"Unknown selection rule: {selection_rule}")

        parts.append(gg)

    ranked = pd.concat(parts, ignore_index=True)
    return ranked.sort_values(["roi", "final_rank"])


def order_existing(values, preferred):
    values = list(values)
    ordered = [x for x in preferred if x in set(values)]
    ordered.extend([x for x in values if x not in set(ordered)])
    return ordered


def plot_auc_vs_volume(volume_auc: pd.DataFrame, out_dir: Path, auc_threshold: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    for (roi, probe, score), g in volume_auc.groupby(["roi", "probe_type", "score"], dropna=False):
        gg = g.copy()
        gg["auc"] = pd.to_numeric(gg["auc"], errors="coerce")
        gg["actual_volume_pct_median"] = pd.to_numeric(gg["actual_volume_pct_median"], errors="coerce")
        gg = gg.dropna(subset=["auc", "actual_volume_pct_median"]).sort_values("actual_volume_pct_median")
        if gg.empty:
            continue

        plt.figure(figsize=(7.4, 5.2))
        plt.plot(gg["actual_volume_pct_median"], gg["auc"], marker="o", linewidth=2)
        plt.axhline(auc_threshold, linestyle="--", linewidth=1.2, alpha=0.7)
        plt.ylim(0.45, 1.02)
        plt.xlabel("Actual input volume change (%)")
        plt.ylabel("AUC(clean vs corruption)")
        plt.title(f"{roi} | {probe}\n{score}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"auc_vs_volume__{sanitize(roi)}__{sanitize(probe)}__{sanitize(score)}.png", dpi=300)
        plt.close()


def plot_heatmap(summary: pd.DataFrame, out_path: Path):
    df = summary.copy()
    df["median_first_detectable_volume_change_pct"] = df["median_first_detectable_volume_change_pct"].replace([np.inf, -np.inf], np.nan)

    pivot = df.pivot_table(
        index="roi",
        columns="score",
        values="median_first_detectable_volume_change_pct",
        aggfunc="mean",
    )
    if pivot.empty:
        return

    rows = order_existing(pivot.index, ROI_ORDER)
    cols = list(pivot.columns)
    pivot = pivot.reindex(index=rows, columns=cols)
    arr = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(max(8, 1.35 * len(cols)), max(5, 0.38 * len(rows))))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap="magma_r")
    plt.colorbar(im, label="Median first detectable volume change (%)")
    plt.xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(np.arange(len(rows)), rows)
    plt.title("First detectable volume-change percentage by ROI and score")

    finite = arr[np.isfinite(arr)]
    mid = np.nanmean(finite) if finite.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                color = "white" if v > mid else "black"
                plt.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Compute first detectable actual volume-change percentage.")
    p.add_argument("--csvs", nargs="+", required=True, help="probe_eval_sweep_with_detector_scores.csv or detailed CSVs.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scores", nargs="+", default=None)
    p.add_argument("--auc_threshold", type=float, default=0.8)
    p.add_argument("--bin_width_pct", type=float, default=2.5)
    p.add_argument("--max_pct", type=float, default=None)
    p.add_argument("--min_clean", type=int, default=3)
    p.add_argument("--min_corrupt", type=int, default=3)
    p.add_argument("--selection_rule", choices=["volume_threshold", "mean_auc", "composite"], default="composite")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(args.csvs)
    df = add_volume_and_scores(df)
    scores = choose_scores(df, args.scores)

    if not scores:
        raise ValueError("No detector scores found. Expected Dice/HD95/ASSD/volume deployable score columns.")

    df.to_csv(out_dir / "input_with_volume_and_scores.csv", index=False)

    volume_auc = compute_volume_auc(
        df=df,
        scores=scores,
        bin_width_pct=args.bin_width_pct,
        max_pct=args.max_pct,
        min_clean=args.min_clean,
        min_corrupt=args.min_corrupt,
    )
    volume_auc.to_csv(out_dir / "volume_auc_clean_vs_corruption_binned_long.csv", index=False)

    first_df = first_detectable_by_probe(volume_auc, auc_threshold=args.auc_threshold)
    first_df.to_csv(out_dir / "first_detectable_volume_change_by_probe.csv", index=False)

    summary = summarize_by_roi_score(first_df, volume_auc, auc_threshold=args.auc_threshold)
    summary.to_csv(out_dir / "first_detectable_volume_change_summary_by_roi_score.csv", index=False)

    ranking = rank_best_by_roi(summary, selection_rule=args.selection_rule)
    ranking.to_csv(out_dir / "detector_ranking_by_roi_volume_threshold.csv", index=False)
    ranking[ranking["final_rank"] == 1].to_csv(out_dir / "best_detector_by_roi_volume_threshold.csv", index=False)

    if not args.no_plots:
        plot_dir = out_dir / "plots"
        plot_auc_vs_volume(volume_auc, plot_dir / "auc_vs_volume", auc_threshold=args.auc_threshold)
        plot_heatmap(summary, plot_dir / "heatmap_median_detectable_volume_pct_by_roi_score.png")

    (out_dir / "README_first_detectable_volume_change.md").write_text(
        "# First detectable volume-change percentage\n\n"
        "For each ROI/probe/score, corrupted samples are binned by actual absolute input volume change:\n\n"
        "`100 * |input_voxels - target_voxels| / target_voxels`\n\n"
        f"The first detectable volume-change percentage is the lowest volume-change bin whose AUC(clean vs corrupted) reaches `{args.auc_threshold}`.\n\n"
        "This replaces raw probe-level thresholds, whose units differ across cut/blob/bite/shift/erosion/dilation.\n",
        encoding="utf-8",
    )

    print(f"Scores used: {scores}")
    print(f"Wrote: {out_dir}")
    print(f"Main table: {out_dir / 'best_detector_by_roi_volume_threshold.csv'}")


if __name__ == "__main__":
    main()
