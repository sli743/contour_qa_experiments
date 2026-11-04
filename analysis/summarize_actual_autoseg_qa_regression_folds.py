#!/usr/bin/env python3
"""Aggregate held-out QA regression results across outer folds for one ROI."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ROI QA regression root containing fold_*/analysis.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    files = sorted(root.glob("fold_*/analysis/qa_regression_test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No fold predictions under {root}")
    frames = [pd.read_csv(p, dtype={"patient_id": str}) for p in files]
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for (target, feature_set), d in df.groupby(["target", "feature_set"], sort=True):
        y = d["true"].to_numpy(float); p = d["predicted"].to_numpy(float)
        rows.append({
            "target": target,
            "feature_set": feature_set,
            "n_predictions": len(d),
            "n_unique_patients": d["patient_id"].nunique(),
            "mae": mean_absolute_error(y,p),
            "rmse": mean_squared_error(y,p)**0.5,
            "r2": r2_score(y,p) if len(y)>=2 and np.var(y)>0 else np.nan,
            "pearson_r": stats.pearsonr(y,p).statistic if len(y)>=3 and np.unique(y).size>1 and np.unique(p).size>1 else np.nan,
            "spearman_rho": stats.spearmanr(y,p).statistic if len(y)>=3 and np.unique(y).size>1 and np.unique(p).size>1 else np.nan,
        })
    output = Path(args.output).expanduser().resolve() if args.output else root / "qa_regression_crossfold_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    df.to_csv(root / "qa_regression_all_test_predictions.csv", index=False)
    print(output)

if __name__ == "__main__": main()
