#!/usr/bin/env python3
"""Aggregate clean ROC outputs generated independently for multiple ROIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TABLES = [
    "roc_summary_by_metric.csv",
    "bootstrap_auc_ci.csv",
    "threshold_operating_points.csv",
    "roc_curve_points.csv",
]


def infer_roi(run_dir: Path) -> str:
    name = run_dir.name
    return name[:-6] if name.endswith("_clean") else name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.endswith("_clean") and (p / "tables").is_dir()
    )

    if not run_dirs:
        raise SystemExit(f"No per-ROI *_clean/tables directories found under {root}")

    combined = {}

    for table_name in TABLES:
        frames = []
        for run_dir in run_dirs:
            path = run_dir / "tables" / table_name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "roi" not in df.columns:
                df.insert(0, "roi", infer_roi(run_dir))
            df["run_dir"] = str(run_dir)
            frames.append(df)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            result.to_csv(out_dir / f"all_rois_{table_name}", index=False)
            combined[table_name] = result

    summary = combined.get("roc_summary_by_metric.csv")
    if summary is not None and not summary.empty:
        metric_col = "metric"
        if metric_col not in summary.columns:
            candidates = [c for c in summary.columns if c.lower() in {"metric_name", "detector_metric"}]
            if candidates:
                metric_col = candidates[0]

        if "auc" in summary.columns and metric_col in summary.columns:
            ranking = summary.copy()
            ranking["auc"] = pd.to_numeric(ranking["auc"], errors="coerce")
            ranking = ranking.sort_values(["roi", "auc"], ascending=[True, False])
            ranking["rank_within_roi"] = ranking.groupby("roi")["auc"].rank(
                method="min", ascending=False
            )
            ranking.to_csv(out_dir / "all_rois_metric_ranking.csv", index=False)

            best = ranking[ranking["rank_within_roi"] == 1].copy()
            best.to_csv(out_dir / "best_metric_per_roi.csv", index=False)

            auc_matrix = ranking.pivot_table(
                index="roi",
                columns=metric_col,
                values="auc",
                aggfunc="first",
            )
            auc_matrix.to_csv(out_dir / "auc_matrix_roi_by_metric.csv")

    status_rows = []
    for run_dir in run_dirs:
        status_rows.append(
            {
                "roi": infer_roi(run_dir),
                "run_dir": str(run_dir),
                "has_summary": (run_dir / "tables" / "roc_summary_by_metric.csv").exists(),
                "has_report": any((run_dir / "report").glob("*.md")),
                "has_figures": any((run_dir / "figures").glob("*.png")),
            }
        )
    pd.DataFrame(status_rows).to_csv(out_dir / "run_status.csv", index=False)

    print(f"Aggregated {len(run_dirs)} ROI runs into: {out_dir}")


if __name__ == "__main__":
    main()
