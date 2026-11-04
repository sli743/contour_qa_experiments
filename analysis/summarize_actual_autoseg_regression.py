#!/usr/bin/env python3
"""Aggregate per-ROI actual-autoseg regression summaries without pooling patients."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="actual_autoseg_regression root")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "regression").glob("*/regression_summary.csv"))
    if not files:
        raise RuntimeError(f"No per-ROI regression summaries under {root / 'regression'}")
    dfs = [pd.read_csv(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out / "all_roi_regression_summary.csv", index=False)
    primary = all_df[
        (all_df["localization_mode"] == "autoseg_centered")
        & (all_df["fit_name"] == "training1_plus_validation1")
    ].copy()
    primary.to_csv(out / "primary_autoseg_centered_testing1_summary.csv", index=False)
    if len(primary):
        med = primary.groupby("metric")[[
            "test_r2_fixed_fit", "test_mae_fixed_fit", "test_rmse_fixed_fit",
            "test_pearson_r", "test_spearman_rho",
        ]].median(numeric_only=True).reset_index()
        med.to_csv(out / "primary_cross_roi_medians.csv", index=False)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
