#!/usr/bin/env python3
"""
Analyze FDVC by ROI × probe pair.

Inputs
------
1) first_detectable_volume_change_by_probe.csv
2) volume_auc_clean_vs_corruption_binned_long.csv

Both are produced by compute_first_detectable_volume_change.py.

Main outputs
------------
out_dir/
  roi_probe_fdvc_best_score.csv
  probe_family_fdvc_coverage_summary.csv
  roi_fdvc_coverage_summary.csv
  score_fdvc_coverage_summary.csv
  roi_probe_fdvc_matrix.csv
  roi_probe_fdvc_penalized_matrix.csv
  plots/
    heatmap_roi_probe_fdvc_best_score.png
    heatmap_roi_probe_detected_binary.png
    heatmap_roi_probe_best_score.png
    bar_probe_coverage.png
    bar_probe_fdvc_penalized.png
    bar_roi_coverage.png
    bar_roi_fdvc_penalized.png

Interpretation
--------------
- FDVC alone is conditional on detection. It can be low even when most pairs are not detected.
- Coverage = fraction of ROI/probe pairs that reached AUC >= threshold.
- Penalized FDVC assigns undetected pairs a penalty value, making low-coverage probes rank worse.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROI_ORDER = [
    "A_Aorta", "A_Pulmonary", "BrachPlx_L", "BrachPlx_R", "Bronchus", "Carina",
    "Esophagus", "Heart", "Liver", "Lung_L", "Lung_R", "Ribs", "SpinalCord",
    "Stomach", "Trachea", "V_VenaCava_S", "NS_Chestwall",
]

PROBE_ORDER = [
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

DEFAULT_SCORES = [
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
    "abs_input_recon_volume_diff_frac",
]


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "value"


def order_existing(values, preferred):
    values = list(values)
    ordered = [v for v in preferred if v in set(values)]
    ordered.extend([v for v in values if v not in set(ordered)])
    return ordered


def read_inputs(first_csv: str, auc_csv: Optional[str]) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    first = pd.read_csv(first_csv)
    first["detected"] = first["detected"].astype(str).str.lower().isin(["true", "1", "yes"])
    first["first_detectable_volume_change_pct"] = pd.to_numeric(
        first["first_detectable_volume_change_pct"], errors="coerce"
    )
    first["first_detectable_auc"] = pd.to_numeric(first["first_detectable_auc"], errors="coerce")
    first["auc_threshold"] = pd.to_numeric(first["auc_threshold"], errors="coerce")

    auc = None
    if auc_csv:
        auc = pd.read_csv(auc_csv)
        auc["auc"] = pd.to_numeric(auc["auc"], errors="coerce")
        auc["actual_volume_pct_median"] = pd.to_numeric(auc["actual_volume_pct_median"], errors="coerce")
    return first, auc


def choose_scores(first: pd.DataFrame, scores: Optional[List[str]]) -> List[str]:
    existing = set(first["score"].astype(str))
    if scores:
        missing = [s for s in scores if s not in existing]
        if missing:
            raise ValueError(f"Requested scores not found in first-detectable CSV: {missing}")
        return scores
    return [s for s in DEFAULT_SCORES if s in existing]


def select_best_score_per_roi_probe(
    first: pd.DataFrame,
    scores: List[str],
    penalty_fdvc_pct: float,
    selection_rule: str,
) -> pd.DataFrame:
    """
    Select best deployable score for each ROI × probe.

    Rules:
      coverage_first: detected first, then lower FDVC, then higher first-detectable AUC.
      penalized_fdvc: lower penalized FDVC, then higher first-detectable AUC.
    """
    df = first[first["score"].isin(scores)].copy()
    df["fdvc_penalized_pct"] = np.where(
        df["detected"],
        df["first_detectable_volume_change_pct"],
        penalty_fdvc_pct,
    )

    parts = []
    for (roi, probe), g in df.groupby(["roi", "probe_type"], dropna=False):
        gg = g.copy()
        if selection_rule == "coverage_first":
            gg = gg.sort_values(
                ["detected", "first_detectable_volume_change_pct", "first_detectable_auc"],
                ascending=[False, True, False],
                na_position="last",
            )
        elif selection_rule == "penalized_fdvc":
            gg = gg.sort_values(
                ["fdvc_penalized_pct", "first_detectable_auc"],
                ascending=[True, False],
                na_position="last",
            )
        else:
            raise ValueError(f"Unknown selection rule: {selection_rule}")

        best = gg.iloc[0].copy()
        best["selected_by"] = selection_rule
        best["n_scores_compared"] = len(gg)
        parts.append(best)

    out = pd.DataFrame(parts)
    out = out.sort_values(["roi", "probe_type"])
    return out


def summarize_probe_family(best: pd.DataFrame, penalty_fdvc_pct: float) -> pd.DataFrame:
    rows = []
    for probe, g in best.groupby("probe_type", dropna=False):
        detected = g[g["detected"]]
        rows.append({
            "probe_type": probe,
            "n_roi_probe_pairs": int(len(g)),
            "n_detected_pairs": int(len(detected)),
            "coverage": float(len(detected) / len(g)) if len(g) else np.nan,
            "median_fdvc_detected_pct": float(detected["first_detectable_volume_change_pct"].median()) if len(detected) else np.nan,
            "mean_fdvc_detected_pct": float(detected["first_detectable_volume_change_pct"].mean()) if len(detected) else np.nan,
            "median_fdvc_penalized_pct": float(g["fdvc_penalized_pct"].median()) if len(g) else np.nan,
            "mean_fdvc_penalized_pct": float(g["fdvc_penalized_pct"].mean()) if len(g) else np.nan,
            "penalty_fdvc_pct": penalty_fdvc_pct,
        })
    return pd.DataFrame(rows).sort_values(["coverage", "median_fdvc_penalized_pct"], ascending=[False, True])


def summarize_roi(best: pd.DataFrame, penalty_fdvc_pct: float) -> pd.DataFrame:
    rows = []
    for roi, g in best.groupby("roi", dropna=False):
        detected = g[g["detected"]]
        rows.append({
            "roi": roi,
            "n_probe_types": int(len(g)),
            "n_detected_probe_types": int(len(detected)),
            "coverage": float(len(detected) / len(g)) if len(g) else np.nan,
            "median_fdvc_detected_pct": float(detected["first_detectable_volume_change_pct"].median()) if len(detected) else np.nan,
            "mean_fdvc_detected_pct": float(detected["first_detectable_volume_change_pct"].mean()) if len(detected) else np.nan,
            "median_fdvc_penalized_pct": float(g["fdvc_penalized_pct"].median()) if len(g) else np.nan,
            "mean_fdvc_penalized_pct": float(g["fdvc_penalized_pct"].mean()) if len(g) else np.nan,
            "penalty_fdvc_pct": penalty_fdvc_pct,
        })
    return pd.DataFrame(rows).sort_values(["coverage", "median_fdvc_penalized_pct"], ascending=[False, True])


def summarize_score(first: pd.DataFrame, scores: List[str], penalty_fdvc_pct: float) -> pd.DataFrame:
    df = first[first["score"].isin(scores)].copy()
    df["fdvc_penalized_pct"] = np.where(df["detected"], df["first_detectable_volume_change_pct"], penalty_fdvc_pct)

    rows = []
    for score, g in df.groupby("score", dropna=False):
        detected = g[g["detected"]]
        rows.append({
            "score": score,
            "n_pairs": int(len(g)),
            "n_detected_pairs": int(len(detected)),
            "coverage": float(len(detected) / len(g)) if len(g) else np.nan,
            "median_fdvc_detected_pct": float(detected["first_detectable_volume_change_pct"].median()) if len(detected) else np.nan,
            "median_fdvc_penalized_pct": float(g["fdvc_penalized_pct"].median()) if len(g) else np.nan,
            "mean_fdvc_penalized_pct": float(g["fdvc_penalized_pct"].mean()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["coverage", "median_fdvc_penalized_pct"], ascending=[False, True])


def make_matrices(best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = order_existing(best["roi"].dropna().unique(), ROI_ORDER)
    cols = order_existing(best["probe_type"].dropna().unique(), PROBE_ORDER)

    fdvc = best.pivot_table(
        index="roi",
        columns="probe_type",
        values="first_detectable_volume_change_pct",
        aggfunc="first",
    ).reindex(index=rows, columns=cols)

    penalized = best.pivot_table(
        index="roi",
        columns="probe_type",
        values="fdvc_penalized_pct",
        aggfunc="first",
    ).reindex(index=rows, columns=cols)

    detected = best.pivot_table(
        index="roi",
        columns="probe_type",
        values="detected_int",
        aggfunc="first",
    ).reindex(index=rows, columns=cols)

    return fdvc, penalized, detected


def plot_heatmap(
    mat: pd.DataFrame,
    out_path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "magma_r",
    annotate: bool = True,
    fmt: str = ".1f",
    vmin=None,
    vmax=None,
    mask_nan: bool = False,
):
    arr = mat.to_numpy(dtype=float)
    if mask_nan:
        masked = np.ma.masked_invalid(arr)
    else:
        masked = arr

    plt.figure(figsize=(max(10, 0.8 * len(mat.columns)), max(5, 0.38 * len(mat.index))))
    im = plt.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=cbar_label)
    plt.xticks(np.arange(len(mat.columns)), mat.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(mat.index)), mat.index)
    plt.title(title)

    if annotate:
        finite = arr[np.isfinite(arr)]
        mid = np.nanmean(finite) if finite.size else 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    color = "white" if v > mid else "black"
                    plt.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=7, color=color)
                elif mask_nan:
                    plt.text(j, i, "ND", ha="center", va="center", fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_best_score_heatmap(best: pd.DataFrame, out_path: Path):
    rows = order_existing(best["roi"].dropna().unique(), ROI_ORDER)
    cols = order_existing(best["probe_type"].dropna().unique(), PROBE_ORDER)
    scores = list(best["score"].dropna().unique())
    score_to_int = {s: i for i, s in enumerate(scores)}

    temp = best.copy()
    temp["score_int"] = temp["score"].map(score_to_int)
    pivot = temp.pivot_table(index="roi", columns="probe_type", values="score_int", aggfunc="first").reindex(index=rows, columns=cols)
    arr = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(max(10, 0.8 * len(cols)), max(5, 0.38 * len(rows))))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ticks=list(score_to_int.values()))
    cbar.ax.set_yticklabels(list(score_to_int.keys()))
    plt.xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(np.arange(len(rows)), rows)
    plt.title("Best deployable detector by ROI × probe")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, ylabel: str, horizontal: bool = True):
    d = df.copy()
    if horizontal:
        d = d.sort_values(y, ascending=True)
        plt.figure(figsize=(8.5, max(4.5, 0.35 * len(d))))
        plt.barh(d[x].astype(str), d[y])
        plt.xlabel(ylabel)
        plt.title(title)
        plt.grid(axis="x", alpha=0.3)
    else:
        plt.figure(figsize=(max(8.5, 0.6 * len(d)), 5))
        plt.bar(d[x].astype(str), d[y])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_readme(out_dir: Path, penalty: float, auc_threshold: float, selection_rule: str):
    (out_dir / "README_roi_probe_fdvc_analysis.md").write_text(
        "# ROI × probe FDVC analysis\n\n"
        f"AUC threshold: `{auc_threshold}`\n\n"
        "FDVC is the first actual volume-change percentage bin whose AUC(clean vs corruption) reaches the threshold.\n\n"
        "Important: raw FDVC is conditional on detection. A probe can have low FDVC among detected pairs but poor coverage.\n"
        "Therefore this analysis reports both:\n\n"
        "- `coverage`: fraction of ROI × probe pairs detected.\n"
        "- `median_fdvc_detected_pct`: median FDVC among detected pairs only.\n"
        f"- `median_fdvc_penalized_pct`: undetected pairs assigned `{penalty}`%, so low-coverage probes are penalized.\n\n"
        f"Best score selection rule for each ROI × probe: `{selection_rule}`.\n",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser(description="Analyze FDVC by ROI × probe pair.")
    p.add_argument("--first_csv", required=True, help="first_detectable_volume_change_by_probe.csv")
    p.add_argument("--volume_auc_csv", default=None, help="volume_auc_clean_vs_corruption_binned_long.csv, optional")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scores", nargs="+", default=None)
    p.add_argument("--auc_threshold", type=float, default=0.8)
    p.add_argument("--penalty_fdvc_pct", type=float, default=105.0)
    p.add_argument("--selection_rule", choices=["coverage_first", "penalized_fdvc"], default="coverage_first")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    first, volume_auc = read_inputs(args.first_csv, args.volume_auc_csv)
    scores = choose_scores(first, args.scores)

    best = select_best_score_per_roi_probe(
        first=first,
        scores=scores,
        penalty_fdvc_pct=args.penalty_fdvc_pct,
        selection_rule=args.selection_rule,
    )
    best["detected_int"] = best["detected"].astype(int)

    probe_summary = summarize_probe_family(best, args.penalty_fdvc_pct)
    roi_summary = summarize_roi(best, args.penalty_fdvc_pct)
    score_summary = summarize_score(first, scores, args.penalty_fdvc_pct)

    fdvc_mat, penalized_mat, detected_mat = make_matrices(best)

    best.to_csv(out_dir / "roi_probe_fdvc_best_score.csv", index=False)
    probe_summary.to_csv(out_dir / "probe_family_fdvc_coverage_summary.csv", index=False)
    roi_summary.to_csv(out_dir / "roi_fdvc_coverage_summary.csv", index=False)
    score_summary.to_csv(out_dir / "score_fdvc_coverage_summary.csv", index=False)
    fdvc_mat.to_csv(out_dir / "roi_probe_fdvc_matrix.csv")
    penalized_mat.to_csv(out_dir / "roi_probe_fdvc_penalized_matrix.csv")
    detected_mat.to_csv(out_dir / "roi_probe_detected_matrix.csv")

    if volume_auc is not None:
        volume_auc.to_csv(out_dir / "input_volume_auc_copy.csv", index=False)

    if not args.no_plots:
        plot_heatmap(
            fdvc_mat,
            plot_dir / "heatmap_roi_probe_fdvc_best_score_detected_only.png",
            "FDVC by ROI × probe, detected pairs only",
            "FDVC (%)",
            cmap="magma_r",
            annotate=True,
            fmt=".1f",
            mask_nan=True,
        )
        plot_heatmap(
            penalized_mat,
            plot_dir / "heatmap_roi_probe_fdvc_penalized.png",
            f"FDVC by ROI × probe, undetected penalized as {args.penalty_fdvc_pct:g}%",
            "Penalized FDVC (%)",
            cmap="magma_r",
            annotate=True,
            fmt=".1f",
            vmin=0,
            vmax=args.penalty_fdvc_pct,
        )
        plot_heatmap(
            detected_mat,
            plot_dir / "heatmap_roi_probe_detected_binary.png",
            "Detection coverage by ROI × probe",
            "Detected = 1",
            cmap="Greens",
            annotate=True,
            fmt=".0f",
            vmin=0,
            vmax=1,
        )
        plot_best_score_heatmap(best, plot_dir / "heatmap_roi_probe_best_score.png")

        plot_bar(
            probe_summary,
            "probe_type",
            "coverage",
            plot_dir / "bar_probe_coverage.png",
            "Detection coverage by probe family",
            "Coverage",
            horizontal=True,
        )
        plot_bar(
            probe_summary,
            "probe_type",
            "median_fdvc_penalized_pct",
            plot_dir / "bar_probe_fdvc_penalized.png",
            "Penalized median FDVC by probe family",
            "Penalized median FDVC (%)",
            horizontal=True,
        )
        plot_bar(
            roi_summary,
            "roi",
            "coverage",
            plot_dir / "bar_roi_coverage.png",
            "Detection coverage by ROI",
            "Coverage",
            horizontal=True,
        )
        plot_bar(
            roi_summary,
            "roi",
            "median_fdvc_penalized_pct",
            plot_dir / "bar_roi_fdvc_penalized.png",
            "Penalized median FDVC by ROI",
            "Penalized median FDVC (%)",
            horizontal=True,
        )

    write_readme(out_dir, args.penalty_fdvc_pct, args.auc_threshold, args.selection_rule)

    print(f"Scores considered: {scores}")
    print(f"Wrote: {out_dir}")
    print("Main files:")
    print(f"  {out_dir / 'roi_probe_fdvc_best_score.csv'}")
    print(f"  {out_dir / 'probe_family_fdvc_coverage_summary.csv'}")
    print(f"  {out_dir / 'roi_fdvc_coverage_summary.csv'}")


if __name__ == "__main__":
    main()
