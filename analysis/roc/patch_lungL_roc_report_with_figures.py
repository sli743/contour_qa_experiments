#!/usr/bin/env python3
"""
Patch Lung_L ROC markdown report by embedding ROC figures and fetching previous
input-vs-recon plots.

This does not regenerate figures. It copies existing plots into:

  roc_analysis_volume_gt20/Lung_L/report/assets/

and writes:

  roc_analysis_volume_gt20/Lung_L/report/Lung_L_ROC_report_with_figures.md

Default locations assume:
  FDVC_ROOT=../radformation-data/probe_auc_sensitivity_assd/first_detectable_volume_custom/

Figures added
-------------
1. Current ROC analysis figures:
   - roc_curves_all_metrics.png
   - score_distribution_clean_vs_corrupt_gt20.png
   - sensitivity_specificity_thresholds.png
   - metric_ranking_barplot.png

2. Previous input-vs-recon plots, for each metric:
   - histogram+KDE by corruption level
   - scatter metric-vs-volume trend plot

The script searches common previous plot folders and copies the first match.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Optional


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


def find_first(patterns: Iterable[str], roots: Iterable[Path]) -> Optional[Path]:
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            matches = sorted(root.rglob(pat))
            if matches:
                return matches[0]
    return None


def copy_asset(src: Path, assets_dir: Path, prefix: str) -> str:
    assets_dir.mkdir(parents=True, exist_ok=True)
    dst_name = f"{prefix}__{src.name}"
    dst = assets_dir / dst_name
    shutil.copy2(src, dst)
    return f"assets/{dst_name}"


def image_md(title: str, rel_path: Optional[str], note: str = "") -> str:
    if rel_path:
        return f"### {title}\n\n![{title}]({rel_path})\n"
    else:
        extra = f"\n\n{note}" if note else ""
        return f"### {title}\n\n*Figure not found.*{extra}\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fdvc_root", default="../radformation-data/probe_auc_sensitivity_assd/first_detectable_volume_custom/")
    p.add_argument("--roc_dir", default=None, help="Default: FDVC_ROOT/roc_analysis_volume_gt20/Lung_L")
    p.add_argument("--previous_plot_root", default=None, help="Default: FDVC_ROOT/score_distribution_analysis_all/plots")
    p.add_argument("--roi", default="Lung_L")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--report_name", default=None)
    p.add_argument("--out_name", default=None)
    args = p.parse_args()

    fdvc_root = Path(args.fdvc_root)
    roc_dir = Path(args.roc_dir) if args.roc_dir else fdvc_root / "roc_analysis_volume_gt20" / args.roi
    prev_root = Path(args.previous_plot_root) if args.previous_plot_root else fdvc_root / "score_distribution_analysis_all" / "plots"

    report_dir = roc_dir / "report"
    figures_dir = roc_dir / "figures"
    assets_dir = report_dir / "assets"

    report_name = args.report_name or f"{args.roi}_ROC_report.md"
    in_report = report_dir / report_name
    out_name = args.out_name or f"{args.roi}_ROC_report_with_figures.md"
    out_report = report_dir / out_name

    if in_report.exists():
        base_text = in_report.read_text(encoding="utf-8")
    else:
        base_text = f"# ROC analysis report: {args.roi}\n\n"

    # Start with original report text, but append a robust figure appendix.
    lines = [base_text.rstrip(), "\n\n---\n", "# Figure appendix\n"]

    # Current ROC figures.
    current_figs = [
        ("ROC curves", figures_dir / "roc_curves_all_metrics.png"),
        ("Score distributions, clean vs corrupt >20%", figures_dir / "score_distribution_clean_vs_corrupt_gt20.png"),
        ("Sensitivity and specificity across thresholds", figures_dir / "sensitivity_specificity_thresholds.png"),
        ("Metric ranking by AUC", figures_dir / "metric_ranking_barplot.png"),
    ]

    lines.append("## Current ROC analysis figures\n")
    for title, path in current_figs:
        rel = copy_asset(path, assets_dir, "roc") if path.exists() else None
        lines.append(image_md(title, rel, note=f"Expected at `{path}`."))

    # Previous input-vs-recon plots.
    lines.append("\n## Previous input-vs-recon diagnostic plots\n")
    lines.append(
        "These figures are copied from previous score-distribution plotting folders. "
        "For each detector metric, the report includes the histogram/KDE plot with separate corruption levels "
        "and the metric-vs-volume scatter plot with per-probe trend lines when available.\n"
    )

    search_roots = [
        prev_root,
        fdvc_root / "score_distribution_analysis_all" / "plots",
        fdvc_root / "score_distribution_analysis_example" / "plots",
        fdvc_root / "metric_vs_volume_interval_errorband_plots" / "plots",
    ]

    for metric in args.metrics:
        disp = DISPLAY_NAMES.get(metric, metric)
        lines.append(f"## {disp}\n")

        # Histogram+KDE by level. Filename format:
        # hist_kde_by_level__Lung_L__big_bite__hd95_input_vs_recon.png
        # There may be one per probe type, so include a compact list.
        hist_root_candidates = [
            prev_root / "hist_kde_by_level",
            prev_root / "organized_by_metric" / "input_vs_recon" / "hist_kde_by_level",
            prev_root,
        ]
        hist_matches = []
        for root in hist_root_candidates:
            if root.exists():
                hist_matches.extend(sorted(root.rglob(f"hist_kde_by_level__{args.roi}__*__{metric}.png")))
        # Deduplicate
        seen = set()
        hist_matches_unique = []
        for m in hist_matches:
            if str(m) not in seen:
                hist_matches_unique.append(m)
                seen.add(str(m))

        if hist_matches_unique:
            lines.append("### Histogram + KDE by corruption level\n")
            for src in hist_matches_unique:
                # Extract probe from filename if possible.
                stem = src.stem
                parts = stem.split("__")
                probe = parts[2] if len(parts) >= 4 else src.stem
                rel = copy_asset(src, assets_dir, f"hist_by_level_{metric}_{probe}")
                lines.append(f"#### {probe}\n\n![{disp} histogram KDE by level: {probe}]({rel})\n")
        else:
            lines.append(
                image_md(
                    "Histogram + KDE by corruption level",
                    None,
                    note=(
                        "Searched for "
                        f"`hist_kde_by_level__{args.roi}__*__{metric}.png` under `{prev_root}`."
                    ),
                )
            )

        # Scatter metric-vs-volume trend. Filename format:
        # scatter_metric_vs_volume_trend__Lung_L__hd95_input_vs_recon.png
        scatter_patterns = [
            f"scatter_metric_vs_volume_trend__{args.roi}__{metric}.png",
            f"metric_vs_volume_interval_by_probe__{args.roi}__{metric}.png",
            f"metric_vs_volume_interval__{args.roi}__*__{metric}.png",
        ]
        scatter = find_first(scatter_patterns, [
            prev_root / "scatter_metric_vs_volume",
            prev_root / "organized_by_metric" / "input_vs_recon" / "scatter_metric_vs_volume",
            prev_root,
            *search_roots,
        ])

        if scatter:
            rel = copy_asset(scatter, assets_dir, f"scatter_trend_{metric}")
            lines.append(image_md("Scatter metric vs volume with trend", rel))
        else:
            lines.append(
                image_md(
                    "Scatter metric vs volume with trend",
                    None,
                    note=(
                        "Searched for scatter/trend plots matching "
                        f"`{args.roi}` and `{metric}` under `{prev_root}`."
                    ),
                )
            )

    out_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_report}")
    print(f"Assets: {assets_dir}")


if __name__ == "__main__":
    main()
