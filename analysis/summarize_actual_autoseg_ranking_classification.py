#!/usr/bin/env python3
"""Aggregate per-ROI ranking/classification results without patient-level pooling."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="actual_autoseg_regression root")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "ranking_classification_summary"
    out.mkdir(parents=True, exist_ok=True)

    rank_files = sorted((root / "ranking_classification").glob("*/ranking_summary.csv"))
    cls_files = sorted((root / "ranking_classification").glob("*/classification_summary.csv"))
    if not rank_files or not cls_files:
        raise RuntimeError(f"Missing per-ROI ranking/classification summaries under {root / 'ranking_classification'}")

    ranking = pd.concat([pd.read_csv(f) for f in rank_files], ignore_index=True)
    classification = pd.concat([pd.read_csv(f) for f in cls_files], ignore_index=True)
    ranking.to_csv(out / "all_roi_ranking_summary.csv", index=False)
    classification.to_csv(out / "all_roi_classification_summary.csv", index=False)

    primary_rank = ranking[(ranking["localization_mode"] == "autoseg_centered") & ranking["is_primary_pair"].astype(bool)].copy()
    primary_rank.to_csv(out / "primary_testing1_assd_ranking_by_roi.csv", index=False)

    primary_cls = classification[(classification["localization_mode"] == "autoseg_centered") & classification["is_primary_endpoint"].astype(bool)].copy()
    primary_cls.to_csv(out / "primary_testing1_assd_p75_classification_by_roi.csv", index=False)

    if len(primary_rank):
        cols = ["spearman_rho", "kendall_tau", "concordance_index", "top_25pct_recall_at_k", "top_25pct_enrichment_over_random"]
        med = primary_rank[[c for c in cols if c in primary_rank.columns]].median(numeric_only=True).to_frame("median_across_rois").reset_index(names="measure")
        med.to_csv(out / "primary_ranking_cross_roi_medians.csv", index=False)
    if len(primary_cls):
        reliable = primary_cls[primary_cls["reliable_roc_min_class_n"].astype(bool)].copy()
        reliable.to_csv(out / "primary_classification_reliable_only.csv", index=False)
        if len(reliable):
            cols = ["roc_auc_testing1", "pr_auc_testing1", "testing1_sensitivity_fixed_dev_threshold", "testing1_specificity_fixed_dev_threshold"]
            med = reliable[[c for c in cols if c in reliable.columns]].median(numeric_only=True).to_frame("median_across_rois").reset_index(names="measure")
            med.to_csv(out / "primary_classification_cross_roi_medians_reliable.csv", index=False)

    # Full 4x4 cross-ROI median matrices for testing1 ranking and P75 classification.
    rank_primary_mode = ranking[ranking["localization_mode"] == "autoseg_centered"].copy()
    rank_primary_mode.pivot_table(index="qa_metric", columns="true_metric", values="spearman_rho", aggfunc="median").to_csv(out / "testing1_spearman_median_matrix.csv")
    cls75 = classification[(classification["localization_mode"] == "autoseg_centered") & (classification["true_error_percentile_from_development"] == 75)].copy()
    cls75.pivot_table(index="qa_metric", columns="true_metric", values="roc_auc_testing1", aggfunc="median").to_csv(out / "testing1_p75_auc_median_matrix.csv")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
