#!/usr/bin/env python3
"""Fit/evaluate per-ROI regressions from NSR-QA disagreement to true autoseg error.

Default predictive design:
  development = training1 + validation1
  final test  = testing1

A validation1-only -> testing1 fit is also produced as a sensitivity analysis.
The final-test outcomes are never used to fit coefficients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METRICS = {
    "1_minus_dice": ("qa_1_minus_dice", "true_1_minus_dice", "1 - Dice"),
    "hd95": ("qa_hd95_mm", "true_hd95_mm", "HD95 (mm)"),
    "assd": ("qa_assd_mm", "true_assd_mm", "ASSD (mm)"),
    "relative_volume_difference": (
        "qa_relative_volume_difference",
        "true_relative_volume_difference",
        "Relative volume difference",
    ),
}


def finite_xy(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    out = df.copy()
    out[xcol] = pd.to_numeric(out[xcol], errors="coerce")
    out[ycol] = pd.to_numeric(out[ycol], errors="coerce")
    return out[np.isfinite(out[xcol]) & np.isfinite(out[ycol])].copy()


def fit_line(df: pd.DataFrame, xcol: str, ycol: str) -> Tuple[float, float]:
    d = finite_xy(df, xcol, ycol)
    if len(d) < 3 or d[xcol].nunique() < 2:
        return np.nan, np.nan
    x = d[xcol].to_numpy(float)
    y = d[ycol].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def correlations(df: pd.DataFrame, xcol: str, ycol: str) -> Tuple[float, float, float, float]:
    d = finite_xy(df, xcol, ycol)
    if len(d) < 3 or d[xcol].nunique() < 2 or d[ycol].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    pr = stats.pearsonr(d[xcol], d[ycol])
    sr = stats.spearmanr(d[xcol], d[ycol])
    return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)


def predictive_metrics(df: pd.DataFrame, xcol: str, ycol: str, slope: float, intercept: float) -> Dict[str, float]:
    d = finite_xy(df, xcol, ycol)
    if len(d) == 0 or not np.isfinite(slope) or not np.isfinite(intercept):
        return {"n": int(len(d)), "r2": np.nan, "mae": np.nan, "rmse": np.nan}
    x = d[xcol].to_numpy(float)
    y = d[ycol].to_numpy(float)
    pred = intercept + slope * x
    resid = y - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return {
        "n": int(len(d)),
        "r2": float(r2),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
    }


def bootstrap_slope_ci(df: pd.DataFrame, xcol: str, ycol: str, n_boot: int, seed: int) -> Tuple[float, float, int]:
    d = finite_xy(df, xcol, ycol).reset_index(drop=True)
    if len(d) < 3:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        b = d.iloc[idx]
        if b[xcol].nunique() < 2:
            continue
        slope, _ = fit_line(b, xcol, ycol)
        if np.isfinite(slope):
            vals.append(slope)
    if not vals:
        return np.nan, np.nan, 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def fit_slope_pvalue(df: pd.DataFrame, xcol: str, ycol: str) -> float:
    d = finite_xy(df, xcol, ycol)
    if len(d) < 3 or d[xcol].nunique() < 2:
        return np.nan
    return float(stats.linregress(d[xcol], d[ycol]).pvalue)


def save_fixed_line_plot(
    train: pd.DataFrame,
    test: pd.DataFrame,
    xcol: str,
    ycol: str,
    label: str,
    slope: float,
    intercept: float,
    title: str,
    output: Path,
) -> None:
    tr = finite_xy(train, xcol, ycol)
    te = finite_xy(test, xcol, ycol)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if len(tr):
        ax.scatter(tr[xcol], tr[ycol], s=22, alpha=0.65, label="Regression development")
    if len(te):
        ax.scatter(te[xcol], te[ycol], s=34, marker="x", label="Held-out testing1")
    all_x = pd.concat([tr[xcol], te[xcol]], ignore_index=True) if len(tr) + len(te) else pd.Series(dtype=float)
    if len(all_x) and np.isfinite(slope) and np.isfinite(intercept):
        xs = np.linspace(float(all_x.min()), float(all_x.max()), 200)
        ax.plot(xs, intercept + slope * xs, linewidth=2, label="Fit from development only")
    ax.set_xlabel(f"NSR-QA disagreement: {label}")
    ax.set_ylabel(f"True autoseg error: {label}")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def regression_rows(df: pd.DataFrame, roi: str, mode: str, fit_name: str, train_splits: Sequence[str], n_boot: int, seed: int, out_dir: Path) -> List[Dict]:
    train = df[df["shape_model_split"].isin(train_splits)].copy()
    test = df[df["shape_model_split"] == "testing1"].copy()
    rows: List[Dict] = []
    for i, (metric, (xcol, ycol, label)) in enumerate(METRICS.items()):
        slope, intercept = fit_line(train, xcol, ycol)
        lo, hi, n_valid_boot = bootstrap_slope_ci(train, xcol, ycol, n_boot, seed + i * 101)
        p_slope = fit_slope_pvalue(train, xcol, ycol)
        train_pred = predictive_metrics(train, xcol, ycol, slope, intercept)
        test_pred = predictive_metrics(test, xcol, ycol, slope, intercept)
        tr_pr, tr_pp, tr_sr, tr_sp = correlations(train, xcol, ycol)
        te_pr, te_pp, te_sr, te_sp = correlations(test, xcol, ycol)
        rows.append({
            "roi": roi,
            "localization_mode": mode,
            "fit_name": fit_name,
            "train_splits": "+".join(train_splits),
            "test_split": "testing1",
            "metric": metric,
            "predictor": xcol,
            "target": ycol,
            "slope": slope,
            "intercept": intercept,
            "slope_ci95_low_bootstrap": lo,
            "slope_ci95_high_bootstrap": hi,
            "slope_bootstrap_valid_replicates": n_valid_boot,
            "slope_pvalue_development": p_slope,
            "development_n": train_pred["n"],
            "development_r2": train_pred["r2"],
            "development_mae": train_pred["mae"],
            "development_rmse": train_pred["rmse"],
            "development_pearson_r": tr_pr,
            "development_pearson_p": tr_pp,
            "development_spearman_rho": tr_sr,
            "development_spearman_p": tr_sp,
            "test_n": test_pred["n"],
            "test_r2_fixed_fit": test_pred["r2"],
            "test_mae_fixed_fit": test_pred["mae"],
            "test_rmse_fixed_fit": test_pred["rmse"],
            "test_pearson_r": te_pr,
            "test_pearson_p": te_pp,
            "test_spearman_rho": te_sr,
            "test_spearman_p": te_sp,
        })
        save_fixed_line_plot(
            train,
            test,
            xcol,
            ycol,
            label,
            slope,
            intercept,
            f"{roi} | {mode} | {fit_name} | {label}",
            out_dir / fit_name / metric / "development_fit_and_testing1.png",
        )
    return rows


def cross_metric_correlations(df: pd.DataFrame, roi: str, mode: str, subset_name: str, splits: Sequence[str]) -> pd.DataFrame:
    d = df[df["shape_model_split"].isin(splits)].copy()
    rows = []
    for qa_name, (xcol, _same_y, _label) in METRICS.items():
        for true_name, (_same_x, ycol, _true_label) in METRICS.items():
            pr, pp, sr, sp = correlations(d, xcol, ycol)
            n = len(finite_xy(d, xcol, ycol))
            rows.append({
                "roi": roi,
                "localization_mode": mode,
                "subset": subset_name,
                "qa_metric": qa_name,
                "true_metric": true_name,
                "n": n,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": sr,
                "spearman_p": sp,
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roi", required=True)
    p.add_argument("--metrics", action="append", required=True, help="mode=metrics.csv; repeat for each localization mode")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=88)
    args = p.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_summary: List[Dict] = []
    all_cross: List[pd.DataFrame] = []

    parsed = []
    for item in args.metrics:
        if "=" not in item:
            raise ValueError(f"Expected mode=csv, got: {item}")
        mode, raw = item.split("=", 1)
        parsed.append((mode, Path(raw).expanduser().resolve()))

    for mode, csv_path in parsed:
        df = pd.read_csv(csv_path, dtype={"patient_id": str})
        required = {"shape_model_split"} | {c for pair in METRICS.values() for c in pair[:2]}
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"{csv_path} missing columns: {missing}")

        mode_dir = out / mode
        all_summary += regression_rows(
            df, args.roi, mode, "training1_plus_validation1", ("training1", "validation1"), args.n_boot, args.seed, mode_dir
        )
        all_summary += regression_rows(
            df, args.roi, mode, "validation1_only_sensitivity", ("validation1",), args.n_boot, args.seed + 10000, mode_dir
        )
        all_cross.append(cross_metric_correlations(df, args.roi, mode, "development_training1_plus_validation1", ("training1", "validation1")))
        all_cross.append(cross_metric_correlations(df, args.roi, mode, "testing1_exploratory_association", ("testing1",)))

    summary = pd.DataFrame(all_summary)
    summary.to_csv(out / "regression_summary.csv", index=False)
    cross = pd.concat(all_cross, ignore_index=True) if all_cross else pd.DataFrame()
    cross.to_csv(out / "cross_metric_correlations.csv", index=False)

    # Directly compare the two localization assumptions on the locked testing1 set.
    primary = summary[(summary["localization_mode"] == "autoseg_centered") & (summary["fit_name"] == "training1_plus_validation1")]
    oracle = summary[(summary["localization_mode"] == "approved_reference") & (summary["fit_name"] == "training1_plus_validation1")]
    if len(primary) and len(oracle):
        cols = ["metric", "test_n", "test_r2_fixed_fit", "test_mae_fixed_fit", "test_rmse_fixed_fit", "test_pearson_r", "test_spearman_rho"]
        comp = primary[cols].merge(oracle[cols], on="metric", suffixes=("_autoseg_centered", "_approved_reference"))
        for metric_name in ("test_r2_fixed_fit", "test_pearson_r", "test_spearman_rho"):
            comp[f"delta_{metric_name}_autoseg_minus_approved"] = comp[f"{metric_name}_autoseg_centered"] - comp[f"{metric_name}_approved_reference"]
        comp.to_csv(out / "localization_mode_testing1_comparison.csv", index=False)

    metadata = {
        "roi": args.roi,
        "primary_predictive_fit": "training1 + validation1 -> testing1",
        "sensitivity_fit": "validation1 -> testing1",
        "primary_qa_model": "NSR-QA only",
        "primary_localization": "autoseg_centered",
        "oracle_localization_sensitivity": "approved_reference",
        "test_outcomes_used_for_fitting": False,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'regression_summary.csv'}")


if __name__ == "__main__":
    main()
