#!/usr/bin/env python3
"""
Pack Lung_L ROC report into a self-contained report folder and generate HTML.

What this script does
---------------------
1. Creates:
     report/assets/

2. Copies ROC figures from:
     <ROC_DIR>/figures/

   into:
     <ROC_DIR>/report/assets/

3. Copies previous input-vs-recon appendix figures from:
     <PREVIOUS_PLOT_ROOT>/

   into:
     <ROC_DIR>/report/assets/

   Specifically:
     - hist_kde_by_level__<ROI>__*__<metric>.png
     - scatter_metric_vs_volume_trend__<ROI>__<metric>.png

4. Writes:
     <ROI>_ROC_report_selfcontained.md

   with all image links pointing to:
     assets/<image>.png

5. Writes:
     report_style.css

   with larger HTML font and controlled image display size.

6. Optionally calls pandoc and writes:
     <ROI>_ROC_report_selfcontained.html

This script does not regenerate plots. It only copies and embeds existing figures.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
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


def copy_asset(src: Path, assets_dir: Path, dst_name: Optional[str] = None) -> Optional[str]:
    if src is None or not src.exists():
        return None
    assets_dir.mkdir(parents=True, exist_ok=True)
    dst = assets_dir / (dst_name or src.name)
    shutil.copy2(src, dst)
    return f"assets/{dst.name}"


def find_all(pattern: str, roots: Iterable[Path]) -> list[Path]:
    out = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob(pattern)):
            rp = str(p.resolve())
            if rp not in seen:
                out.append(p)
                seen.add(rp)
    return out


def find_first(patterns: Iterable[str], roots: Iterable[Path]) -> Optional[Path]:
    for pattern in patterns:
        found = find_all(pattern, roots)
        if found:
            return found[0]
    return None


def write_css(path: Path, image_width_px: int):
    css = f"""
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  font-size: 18px;
  line-height: 1.55;
  max-width: 1100px;
  margin: 32px auto;
  padding: 0 28px;
  color: #222;
}}

h1 {{
  font-size: 34px;
  margin-top: 0;
}}

h2 {{
  font-size: 27px;
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.25em;
  margin-top: 1.6em;
}}

h3 {{
  font-size: 22px;
  margin-top: 1.4em;
}}

h4 {{
  font-size: 19px;
  margin-top: 1.2em;
}}

table {{
  border-collapse: collapse;
  font-size: 15px;
  width: 100%;
  margin: 1em 0;
}}

th, td {{
  border: 1px solid #ddd;
  padding: 6px 8px;
}}

th {{
  background: #f4f4f4;
}}

img {{
  display: block;
  max-width: min(100%, {image_width_px}px);
  height: auto;
  margin: 16px auto 28px auto;
  border: 1px solid #ddd;
}}

code {{
  font-size: 0.95em;
}}

pre {{
  overflow-x: auto;
  background: #f7f7f7;
  padding: 12px;
}}

.figure-note {{
  color: #666;
  font-size: 15px;
}}
"""
    path.write_text(css.strip() + "\n", encoding="utf-8")


def image_block(title: str, rel: Optional[str], note: str = "") -> str:
    if rel:
        return f"### {title}\n\n![{title}]({rel})\n"
    return f"### {title}\n\n*Figure not found.* {note}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdvc_root", default="../radformation-data/probe_auc_sensitivity_assd/first_detectable_volume_custom/")
    ap.add_argument("--roc_dir", default=None, help="Default: FDVC_ROOT/roc_analysis_volume_gt20/Lung_L_clean")
    ap.add_argument("--previous_plot_root", default=None, help="Default: FDVC_ROOT/score_distribution_analysis_all/plots")
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--input_md", default=None, help="Default: ROC_DIR/report/<ROI>_ROC_report.md")
    ap.add_argument("--out_md", default=None, help="Default: ROC_DIR/report/<ROI>_ROC_report_selfcontained.md")
    ap.add_argument("--image_width_px", type=int, default=820)
    ap.add_argument("--no_pandoc", action="store_true")
    args = ap.parse_args()

    fdvc_root = Path(args.fdvc_root)
    roc_dir = Path(args.roc_dir) if args.roc_dir else fdvc_root / "roc_analysis_volume_gt20" / f"{args.roi}_clean"
    prev_root = Path(args.previous_plot_root) if args.previous_plot_root else fdvc_root / "score_distribution_analysis_all" / "plots"

    report_dir = roc_dir / "report"
    figures_dir = roc_dir / "figures"
    assets_dir = report_dir / "assets"
    report_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    in_md = Path(args.input_md) if args.input_md else report_dir / f"{args.roi}_ROC_report.md"
    out_md = Path(args.out_md) if args.out_md else report_dir / f"{args.roi}_ROC_report_selfcontained.md"
    out_html = out_md.with_suffix(".html")
    css_path = report_dir / "report_style.css"

    if in_md.exists():
        # Keep original textual report, but strip previous figure section only if desired? Simpler: use as base.
        base = in_md.read_text(encoding="utf-8").rstrip()
    else:
        base = f"# ROC analysis report: {args.roi}\n\n"

    write_css(css_path, image_width_px=args.image_width_px)

    lines = [base, "\n\n---\n", "# Self-contained figure appendix\n"]

    # Main ROC figures.
    lines.append("## Main ROC analysis figures\n")
    roc_figs = [
        ("ROC curves", figures_dir / "roc_curves_all_metrics.png", "roc_curves_all_metrics.png"),
        ("Metric ranking", figures_dir / "metric_ranking_barplot.png", "metric_ranking_barplot.png"),
        ("Score distributions", figures_dir / "score_distribution_clean_vs_corrupt.png", "score_distribution_clean_vs_corrupt.png"),
        ("Operating point tradeoff", figures_dir / "operating_point_tradeoff.png", "operating_point_tradeoff.png"),
        # Also tolerate old filename
        ("Score distributions", figures_dir / "score_distribution_clean_vs_corrupt_gt20.png", "score_distribution_clean_vs_corrupt_gt20.png"),
    ]

    copied_main = set()
    for title, src, dst in roc_figs:
        if src.exists() and src.name not in copied_main:
            rel = copy_asset(src, assets_dir, dst)
            copied_main.add(src.name)
            lines.append(image_block(title, rel, note=f"Expected `{src}`."))

    # Previous appendix plots.
    lines.append("\n## Input-vs-reconstruction diagnostic plots\n")
    lines.append(
        "These figures were copied from the previous score-distribution analysis. "
        "They are included here to visually support the ROC results.\n"
    )

    search_roots = [
        prev_root,
        prev_root / "hist_kde_by_level",
        prev_root / "scatter_metric_vs_volume",
        prev_root / "organized_by_metric" / "input_vs_recon",
        fdvc_root / "score_distribution_analysis_example" / "plots",
    ]

    for metric in args.metrics:
        disp = DISPLAY_NAMES.get(metric, metric)
        lines.append(f"## {disp}\n")

        # Histogram + KDE by separate corruption levels. Include all probe-specific plots.
        hist_matches = find_all(f"hist_kde_by_level__{args.roi}__*__{metric}.png", search_roots)
        if hist_matches:
            lines.append("### Histogram + KDE by corruption level\n")
            for src in hist_matches:
                parts = src.stem.split("__")
                probe = parts[2] if len(parts) >= 4 else "probe"
                safe_name = f"hist_kde_by_level__{args.roi}__{probe}__{metric}.png"
                rel = copy_asset(src, assets_dir, safe_name)
                lines.append(f"#### {probe}\n\n![{disp}: histogram KDE by level, {probe}]({rel})\n")
        else:
            lines.append(
                image_block(
                    "Histogram + KDE by corruption level",
                    None,
                    note=f"Searched for `hist_kde_by_level__{args.roi}__*__{metric}.png` under `{prev_root}`."
                )
            )

        # Scatter metric-vs-volume trend.
        scatter = find_first(
            [
                f"scatter_metric_vs_volume_trend__{args.roi}__{metric}.png",
                f"scatter_metric_vs_volume__{args.roi}__{metric}.png",
                f"metric_vs_volume_interval_by_probe__{args.roi}__{metric}.png",
            ],
            search_roots,
        )
        if scatter:
            rel = copy_asset(scatter, assets_dir, f"scatter_metric_vs_volume_trend__{args.roi}__{metric}.png")
            lines.append(image_block("Scatter metric vs volume with per-probe trend", rel))
        else:
            lines.append(
                image_block(
                    "Scatter metric vs volume with per-probe trend",
                    None,
                    note=f"Searched for scatter/trend plot for `{args.roi}` and `{metric}` under `{prev_root}`."
                )
            )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote markdown: {out_md}")
    print(f"Assets: {assets_dir}")
    print(f"CSS: {css_path}")

    if not args.no_pandoc:
        if shutil.which("pandoc") is None:
            print("WARNING: pandoc not found; HTML not generated.")
            return
        cmd = [
            "pandoc",
            out_md.name,
            "-s",
            "--toc",
            "--css",
            css_path.name,
            "--metadata",
            f"title={out_md.stem}",
            "-o",
            out_html.name,
        ]
        subprocess.run(cmd, cwd=report_dir, check=True)
        print(f"Wrote HTML: {out_html}")


if __name__ == "__main__":
    main()
