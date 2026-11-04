#!/usr/bin/env python3
"""Fit direct ROI-specific QA regressors on deployment-feasible autoseg features.

Outer-fold protocol:
  1. fit candidate models on manifest split=train
  2. choose alpha/PCA dimensionality by validation MAE only
  3. refit the selected specification on train+validation
  4. evaluate exactly once on split=test

Feature sets compare simple geometry, frozen 3DINO features, optional supervised
correction-response summaries, and combinations. The approved contour appears
only in the target columns already stored by the feature extractor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {
    "1_minus_dice": "true_1_minus_dice",
    "assd": "true_assd_mm",
    "hd95": "true_hd95_mm",
}
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
PCA_COMPONENTS = (4, 8, 16, 32)


def numeric_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def feature_sets(df: pd.DataFrame):
    geom = numeric_columns(df, "geom_")
    legacy_dino = numeric_columns(df, "dino_mean_") + numeric_columns(df, "dino_std_")
    global_tails = numeric_columns(df, "dino_global_")
    boundary = numeric_columns(df, "dino_boundary_")
    corr = numeric_columns(df, "corr_")

    global_distribution = legacy_dino + global_tails
    boundary_distribution = boundary
    combined_dino = global_distribution + boundary_distribution

    sets = {
        # Exact old frozen-DINO baseline from the first QA experiment.
        "dino_global_mean_std": (legacy_dino, True),
        # Tests whether rare/extreme global token responses survive pooling.
        "dino_global_distribution": (global_distribution, True),
        # Tests the hypothesis that contour error information is concentrated near the boundary.
        "dino_boundary_distribution": (boundary_distribution, True),
        "dino_global_plus_boundary": (combined_dino, True),
        "geometry": (geom, False),
    }
    if corr:
        sets["correction_response"] = (corr, False)
        sets["boundary_plus_correction"] = (boundary_distribution + corr, True)
        sets["dino_plus_correction"] = (combined_dino + corr, True)
        sets["all_features"] = (geom + combined_dino + corr, True)
    else:
        sets["geometry_plus_boundary_dino"] = (geom + combined_dino, True)

    # Remove empty families and de-duplicate columns while preserving order.
    cleaned = {}
    for name, (cols, use_pca) in sets.items():
        cols = list(dict.fromkeys(cols))
        if cols:
            cleaned[name] = (cols, use_pca)
    return cleaned


def build_pipeline(alpha: float, use_pca: bool, n_components: int | None, n_train: int, n_features: int):
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    actual_components = None
    if use_pca:
        max_components = min(n_train - 1, n_features)
        if max_components >= 2:
            requested = int(n_components or min(16, max_components))
            actual_components = max(2, min(requested, max_components))
            steps.append(("pca", PCA(n_components=actual_components, random_state=88)))
    steps.append(("ridge", Ridge(alpha=float(alpha))))
    return Pipeline(steps), actual_components


def finite_target(df: pd.DataFrame, target: str):
    y = pd.to_numeric(df[target], errors="coerce")
    return df.loc[np.isfinite(y)].copy(), y.loc[np.isfinite(y)].to_numpy(float)


def evaluate(y, pred):
    if len(y) == 0:
        return {k: np.nan for k in ("mae", "rmse", "r2", "pearson_r", "spearman_rho")}
    out = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 and np.var(y) > 0 else np.nan,
    }
    if len(y) >= 3 and np.unique(y).size > 1 and np.unique(pred).size > 1:
        out["pearson_r"] = float(stats.pearsonr(y, pred).statistic)
        out["spearman_rho"] = float(stats.spearmanr(y, pred).statistic)
    else:
        out["pearson_r"] = np.nan
        out["spearman_rho"] = np.nan
    return out


def bootstrap_ci(y, pred, metric: str, n_boot: int, seed: int):
    if len(y) < 3 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yy, pp = y[idx], pred[idx]
        if metric == "mae":
            vals.append(mean_absolute_error(yy, pp))
        elif metric == "spearman":
            if np.unique(yy).size < 2 or np.unique(pp).size < 2:
                continue
            vals.append(stats.spearmanr(yy, pp).statistic)
    if not vals:
        return np.nan, np.nan
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))


def tune_model(train_df, val_df, cols, target, use_pca):
    tr, ytr = finite_target(train_df, target)
    va, yva = finite_target(val_df, target)
    if len(tr) < 5 or len(va) < 2:
        raise RuntimeError(f"Insufficient train/validation rows for {target}: {len(tr)}/{len(va)}")
    candidates = PCA_COMPONENTS if use_pca else (None,)
    rows = []
    best = None
    for ncomp in candidates:
        for alpha in ALPHAS:
            pipe, actual = build_pipeline(alpha, use_pca, ncomp, len(tr), len(cols))
            pipe.fit(tr[cols], ytr)
            pred = pipe.predict(va[cols])
            mae = float(mean_absolute_error(yva, pred))
            row = {"alpha": alpha, "requested_pca_components": ncomp, "actual_pca_components": actual, "validation_mae": mae}
            rows.append(row)
            key = (mae, float(alpha), int(actual or 0))
            if best is None or key < best[0]:
                best = (key, alpha, actual)
    return best[1], best[2], pd.DataFrame(rows)


def mean_baseline(train_val_df, test_df, target):
    dev, ydev = finite_target(train_val_df, target)
    te, yte = finite_target(test_df, target)
    mean = float(np.mean(ydev))
    pred = np.full(len(yte), mean, dtype=float)
    return te, yte, pred, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=88)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv, dtype={"patient_id": str})
    required = {"patient_id", "roi", "fold", "split"} | set(TARGETS.values())
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError(f"Expected nonempty train/validation/test; got {len(train)}/{len(val)}/{len(test)}")

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sets = feature_sets(df)
    summary = []
    predictions = []
    tuning_frames = []
    roi = str(df["roi"].iloc[0])
    fold = int(df["fold"].iloc[0])

    for target_name, target in TARGETS.items():
        dev = pd.concat([train, val], ignore_index=True)
        te, yte, pred_mean, mean_value = mean_baseline(dev, test, target)
        met = evaluate(yte, pred_mean)
        summary.append({
            "roi": roi, "fold": fold, "target": target_name, "feature_set": "mean_baseline",
            "n_features": 0, "alpha": np.nan, "pca_components": np.nan,
            "train_n": len(train), "validation_n": len(val), "test_n": len(yte), **met,
        })
        for pid, yy, pp in zip(te["patient_id"], yte, pred_mean):
            predictions.append({"patient_id": pid, "roi": roi, "fold": fold, "target": target_name, "feature_set": "mean_baseline", "true": yy, "predicted": pp})

        for feature_name, (cols, use_pca) in sets.items():
            alpha, ncomp, tune = tune_model(train, val, cols, target, use_pca)
            tune.insert(0, "feature_set", feature_name)
            tune.insert(0, "target", target_name)
            tuning_frames.append(tune)

            dev_f, ydev = finite_target(dev, target)
            test_f, ytest = finite_target(test, target)
            pipe, actual = build_pipeline(alpha, use_pca, ncomp, len(dev_f), len(cols))
            pipe.fit(dev_f[cols], ydev)
            pred = pipe.predict(test_f[cols])
            met = evaluate(ytest, pred)
            mae_lo, mae_hi = bootstrap_ci(ytest, pred, "mae", args.n_boot, args.seed + len(summary) * 13)
            sp_lo, sp_hi = bootstrap_ci(ytest, pred, "spearman", args.n_boot, args.seed + len(summary) * 17)
            summary.append({
                "roi": roi, "fold": fold, "target": target_name, "feature_set": feature_name,
                "n_features": len(cols), "alpha": alpha, "pca_components": actual,
                "train_n": len(train), "validation_n": len(val), "test_n": len(ytest), **met,
                "mae_ci95_low": mae_lo, "mae_ci95_high": mae_hi,
                "spearman_ci95_low": sp_lo, "spearman_ci95_high": sp_hi,
            })
            for pid, yy, pp in zip(test_f["patient_id"], ytest, pred):
                predictions.append({"patient_id": pid, "roi": roi, "fold": fold, "target": target_name, "feature_set": feature_name, "true": yy, "predicted": pp})

    pd.DataFrame(summary).to_csv(out / "qa_regression_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "qa_regression_test_predictions.csv", index=False)
    if tuning_frames:
        pd.concat(tuning_frames, ignore_index=True).to_csv(out / "qa_regression_validation_tuning.csv", index=False)
    (out / "qa_regression_protocol.json").write_text(json.dumps({
        "roi": roi,
        "fold": fold,
        "protocol": "tune on validation after fitting train; refit selected spec on train+validation; evaluate once on test",
        "feature_sets": {k: {"n_features": len(v[0]), "uses_pca": bool(v[1])} for k, v in sets.items()},
        "targets": TARGETS,
        "ridge_alphas": ALPHAS,
        "pca_candidates": PCA_COMPONENTS,
        "n_boot": args.n_boot,
    }, indent=2))
    print(out / "qa_regression_summary.csv")


if __name__ == "__main__":
    main()
