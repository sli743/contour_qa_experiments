#!/usr/bin/env python3
"""
Select best shape detector score for each ROI from AUC sensitivity outputs.

Use after compute_probe_auc_sensitivity.py.

Required inputs:
  auc_clean_vs_corruption_long.csv
  auc_detection_thresholds.csv
  detector_score_definitions.csv

Default uses only deployable scores:
  one_minus_dice_input_vs_recon
  hd95_input_vs_recon
  abs_input_recon_volume_diff_frac
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

DEPLOYABLE_DEFAULT = [
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "abs_input_recon_volume_diff_frac",
]


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "value"


def order_existing(values, preferred):
    values = list(values)
    ordered = [x for x in preferred if x in set(values)]
    ordered.extend([x for x in values if x not in set(ordered)])
    return ordered


def choose_scores(score_defs: pd.DataFrame, requested, include_synthetic: bool, include_control: bool):
    if requested:
        return requested
    if "category" not in score_defs.columns:
        return [s for s in DEPLOYABLE_DEFAULT if s in set(score_defs["score"])]
    scores = []
    for _, r in score_defs.iterrows():
        score = str(r["score"])
        cat = str(r.get("category", ""))
        if cat == "deployable":
            scores.append(score)
        elif cat == "synthetic_only" and include_synthetic:
            scores.append(score)
        elif cat == "corruption_control" and include_control:
            scores.append(score)
    return scores


def read_inputs(args):
    auc = pd.read_csv(args.auc_long)
    thr = pd.read_csv(args.thresholds)
    defs = pd.read_csv(args.score_defs)

    auc["auc"] = pd.to_numeric(auc["auc"], errors="coerce")
    auc["level"] = pd.to_numeric(auc["level"], errors="coerce")
    thr["auc_threshold"] = pd.to_numeric(thr["auc_threshold"], errors="coerce")
    thr["first_detectable_level"] = pd.to_numeric(thr["first_detectable_level"], errors="coerce")
    if "detected" in thr.columns:
        thr["detected"] = thr["detected"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        thr["detected"] = np.isfinite(thr["first_detectable_level"])
    return auc, thr, defs


def summarize_auc_by_roi_score(auc: pd.DataFrame, scores):
    df = auc[auc["score"].isin(scores)].copy()
    df = df.dropna(subset=["auc"])
    rows = []
    for (roi, score), g in df.groupby(["roi", "score"], dropna=False):
        rows.append({
            "roi": roi,
            "score": score,
            "mean_auc": float(g["auc"].mean()),
            "median_auc": float(g["auc"].median()),
            "std_auc": float(g["auc"].std(ddof=1)) if len(g) > 1 else np.nan,
            "min_auc": float(g["auc"].min()),
            "max_auc": float(g["auc"].max()),
            "n_auc_rows": int(len(g)),
            "n_probe_types": int(g["probe_type"].nunique()) if "probe_type" in g.columns else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_threshold_by_roi_score(thr: pd.DataFrame, scores, auc_threshold: float):
    df = thr[(thr["score"].isin(scores)) & (np.isclose(thr["auc_threshold"], auc_threshold))].copy()
    rows = []
    for (roi, score), g in df.groupby(["roi", "score"], dropna=False):
        detected = g[g["detected"]].copy()
        n_probe = int(g["probe_type"].nunique()) if "probe_type" in g.columns else int(len(g))
        n_detected = int(detected["probe_type"].nunique()) if "probe_type" in detected.columns else int(len(detected))
        rows.append({
            "roi": roi,
            "score": score,
            "auc_threshold": auc_threshold,
            "n_probe_types_threshold": n_probe,
            "n_detected_probe_types": n_detected,
            "detection_coverage": float(n_detected / n_probe) if n_probe else np.nan,
            "median_first_detectable_level": float(detected["first_detectable_level"].median()) if not detected.empty else np.inf,
            "mean_first_detectable_level": float(detected["first_detectable_level"].mean()) if not detected.empty else np.inf,
            "min_first_detectable_level": float(detected["first_detectable_level"].min()) if not detected.empty else np.inf,
        })
    return pd.DataFrame(rows)


def summarize_auc_by_roi_probe(auc: pd.DataFrame, scores):
    df = auc[auc["score"].isin(scores)].copy()
    df = df.dropna(subset=["auc"])
    rows = []
    for (roi, probe, score), g in df.groupby(["roi", "probe_type", "score"], dropna=False):
        rows.append({
            "roi": roi,
            "probe_type": probe,
            "score": score,
            "mean_auc": float(g["auc"].mean()),
            "median_auc": float(g["auc"].median()),
            "max_auc": float(g["auc"].max()),
            "n_levels": int(g["level"].nunique()) if "level" in g.columns else int(len(g)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank_within_roi_probe"] = out.groupby(["roi", "probe_type"])["mean_auc"].rank(method="min", ascending=False)
    return out


def rank_detectors(auc_summary, thr_summary, selection_rule: str):
    df = auc_summary.merge(thr_summary, on=["roi", "score"], how="left")
    finite = df["median_first_detectable_level"].replace([np.inf, -np.inf], np.nan)
    max_finite = finite.max()
    penalty = max_finite + 1.0 if np.isfinite(max_finite) else 999.0
    df["threshold_for_rank"] = finite.fillna(penalty)

    parts = []
    for roi, g in df.groupby("roi", dropna=False):
        gg = g.copy()
        gg["rank_mean_auc"] = gg["mean_auc"].rank(method="min", ascending=False)
        gg["rank_detection_coverage"] = gg["detection_coverage"].rank(method="min", ascending=False)
        gg["rank_threshold"] = gg["threshold_for_rank"].rank(method="min", ascending=True)
        gg["composite_rank_score"] = (
            gg["rank_mean_auc"] + gg["rank_detection_coverage"] + gg["rank_threshold"]
        ) / 3.0

        if selection_rule == "mean_auc":
            gg["final_rank"] = gg["mean_auc"].rank(method="first", ascending=False)
        elif selection_rule == "threshold":
            gg = gg.sort_values(
                ["detection_coverage", "threshold_for_rank", "mean_auc"],
                ascending=[False, True, False],
            )
            gg["final_rank"] = np.arange(1, len(gg) + 1)
        elif selection_rule == "composite":
            gg["final_rank"] = gg["composite_rank_score"].rank(method="first", ascending=True)
        else:
            raise ValueError(selection_rule)
        parts.append(gg)
    return pd.concat(parts, ignore_index=True).sort_values(["roi", "final_rank"])


def best_table(ranking):
    best = ranking[ranking["final_rank"] == 1].copy()
    cols = [
        "roi", "score", "mean_auc", "median_auc", "std_auc",
        "detection_coverage", "median_first_detectable_level",
        "n_probe_types", "n_detected_probe_types",
        "rank_mean_auc", "rank_detection_coverage", "rank_threshold",
        "composite_rank_score",
    ]
    cols = [c for c in cols if c in best.columns]
    return best[cols].sort_values("roi")


def plot_heatmap(df, value_col, out_path, title, vmin=None, vmax=None, cmap="viridis", fmt=".2f"):
    pivot = df.pivot_table(index="roi", columns="score", values=value_col, aggfunc="mean")
    if pivot.empty:
        return
    rows = order_existing(pivot.index, ROI_ORDER)
    cols = list(pivot.columns)
    pivot = pivot.reindex(index=rows, columns=cols)
    arr = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(max(8, 1.25 * len(cols)), max(5, 0.38 * len(rows))))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im, label=value_col)
    plt.xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(np.arange(len(rows)), rows)
    plt.title(title)

    finite_vals = arr[np.isfinite(arr)]
    mid = np.nanmean(finite_vals) if finite_vals.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                color = "white" if v > mid else "black"
                plt.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_best_counts(best, out_path):
    if best.empty:
        return
    counts = best["score"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(max(7, 1.1 * len(counts)), 4.8))
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of ROIs")
    plt.title("Best detector counts across ROIs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_best_by_roi_probe(roi_probe_summary, out_path):
    if roi_probe_summary.empty:
        return
    best = roi_probe_summary[roi_probe_summary["rank_within_roi_probe"] == 1].copy()
    if best.empty:
        return
    scores = list(best["score"].drop_duplicates())
    score_to_int = {s: i for i, s in enumerate(scores)}
    best["score_int"] = best["score"].map(score_to_int)
    pivot = best.pivot_table(index="roi", columns="probe_type", values="score_int", aggfunc="first")
    rows = order_existing(pivot.index, ROI_ORDER)
    cols = order_existing(pivot.columns, PROBE_ORDER)
    pivot = pivot.reindex(index=rows, columns=cols)

    plt.figure(figsize=(max(10, 0.75 * len(cols)), max(5, 0.38 * len(rows))))
    im = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ticks=list(score_to_int.values()))
    cbar.ax.set_yticklabels(list(score_to_int.keys()))
    plt.xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(np.arange(len(rows)), rows)
    plt.title("Best detector score by ROI and probe type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_markdown(best, out_path, selection_rule, auc_threshold):
    lines = [
        "# Best detector per ROI",
        "",
        f"Selection rule: `{selection_rule}`",
        f"AUC threshold for detection-level summaries: `{auc_threshold}`",
        "",
        "Only deployable detector scores are included by default.",
        "",
    ]
    if best.empty:
        lines.append("No results.")
    else:
        show = best.copy()
        for c in ["mean_auc", "median_auc", "detection_coverage", "median_first_detectable_level"]:
            if c in show.columns:
                show[c] = show[c].map(lambda x: "NA" if not np.isfinite(x) else f"{x:.3g}")
        try:
            lines.append(show.to_markdown(index=False))
        except Exception:
            lines.append(show.to_csv(index=False))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auc_long", required=True)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--score_defs", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--auc_threshold", type=float, default=0.8)
    p.add_argument("--scores", nargs="+", default=None)
    p.add_argument("--include_synthetic", action="store_true")
    p.add_argument("--include_control", action="store_true")
    p.add_argument("--selection_rule", choices=["mean_auc", "threshold", "composite"], default="composite")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    auc, thr, defs = read_inputs(args)
    scores = choose_scores(defs, args.scores, args.include_synthetic, args.include_control)
    if not scores:
        raise ValueError("No scores selected.")

    auc_summary = summarize_auc_by_roi_score(auc, scores)
    thr_summary = summarize_threshold_by_roi_score(thr, scores, args.auc_threshold)
    roi_probe_summary = summarize_auc_by_roi_probe(auc, scores)

    ranking = rank_detectors(auc_summary, thr_summary, args.selection_rule)
    best = best_table(ranking)

    auc_summary.to_csv(out_dir / "detector_score_roi_auc_summary.csv", index=False)
    thr_summary.to_csv(out_dir / "detector_score_roi_threshold_summary.csv", index=False)
    roi_probe_summary.to_csv(out_dir / "detector_score_roi_probe_auc_summary.csv", index=False)
    ranking.to_csv(out_dir / "detector_ranking_by_roi.csv", index=False)
    best.to_csv(out_dir / "best_detector_by_roi.csv", index=False)

    if not roi_probe_summary.empty:
        roi_probe_summary[roi_probe_summary["rank_within_roi_probe"] == 1].to_csv(
            out_dir / "best_detector_by_roi_by_probe.csv", index=False
        )

    if not args.no_plots:
        plot_heatmap(
            auc_summary, "mean_auc",
            plot_dir / "heatmap_mean_auc_by_roi_score.png",
            "Mean AUC by ROI and deployable detector score",
            vmin=0.5, vmax=1.0, cmap="viridis", fmt=".2f"
        )
        thr_plot = thr_summary.copy()
        thr_plot["median_first_detectable_level"] = thr_plot["median_first_detectable_level"].replace([np.inf, -np.inf], np.nan)
        plot_heatmap(
            thr_plot, "median_first_detectable_level",
            plot_dir / f"heatmap_median_threshold_by_roi_score_auc{str(args.auc_threshold).replace('.', 'p')}.png",
            f"Median first detectable level by ROI and score, AUC >= {args.auc_threshold:g}",
            cmap="magma_r", fmt=".2g"
        )
        plot_best_counts(best, plot_dir / "bar_best_detector_counts.png")
        plot_best_by_roi_probe(roi_probe_summary, plot_dir / "heatmap_best_score_by_roi_probe_auc.png")

    write_markdown(best, out_dir / "best_detector_summary.md", args.selection_rule, args.auc_threshold)

    print(f"Selected scores: {scores}")
    print(f"Wrote: {out_dir}")
    print(f"Best detector table: {out_dir / 'best_detector_by_roi.csv'}")


if __name__ == "__main__":
    main()
