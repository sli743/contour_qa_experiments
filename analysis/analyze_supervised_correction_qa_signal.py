#!/usr/bin/env python3
"""Evaluate correction-model disagreement as a QA signal.

For each corresponding metric:
  QA score   = distance(input autoseg, model corrected output)
  True error = distance(input autoseg, approved contour)

The development set is train + validation.  No test outcome is used to choose
thresholds or fit the regression.

Binary QA:
  * Define high-error cases from a percentile of development true error
    (P75 primary; P50 sensitivity analysis by default).
  * Report raw test ROC/PR AUC from the QA score.
  * Select a QA-score operating threshold on development targeting >=95%
    specificity, then freeze it for test.

Regression QA:
  * Fit a univariate linear model true_error ~ qa_score on development.
  * Freeze slope/intercept and evaluate on test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats


METRICS = {
    "1_minus_dice": ("qa_1_minus_dice", "true_1_minus_dice"),
    "hd95": ("qa_hd95_mm", "true_hd95_mm"),
    "assd": ("qa_assd_mm", "true_assd_mm"),
}


def load_splits(evaluation_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "validation", "test"):
        path = evaluation_dir / f"{split}_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Re-run eval_supervised_correction_threeclass.py "
                "with the updated evaluator so test_metrics.csv is produced."
            )
        df = pd.read_csv(path, dtype={"patient_id": str})
        df["analysis_split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for qa_col, true_col in METRICS.values():
        out[qa_col] = pd.to_numeric(out[qa_col], errors="coerce")
        out[true_col] = pd.to_numeric(out[true_col], errors="coerce")
    return out


def finite_pair(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return df[np.isfinite(df[x]) & np.isfinite(df[y])].copy()


def percentile_higher(series: pd.Series, percentile: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return float("nan")
    try:
        return float(np.percentile(vals, percentile, method="higher"))
    except TypeError:  # numpy < 1.22
        return float(np.percentile(vals, percentile, interpolation="higher"))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.r_[pos, neg])
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float(precision[ys == 1].sum() / n_pos)


def confusion(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    if not np.isfinite(threshold):
        return {k: float("nan") for k in ("tp", "fp", "tn", "fn", "sensitivity", "specificity", "ppv", "npv")}
    pred = np.asarray(score) >= threshold
    y = np.asarray(y, int)
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & (~pred)).sum())
    fn = int(((y == 1) & (~pred)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
    }


def choose_threshold_at_specificity(y: np.ndarray, score: np.ndarray, target: float) -> Tuple[float, float, float]:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    candidates = np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]
    feasible = []
    for t in candidates:
        cm = confusion(y, score, float(t))
        if np.isfinite(cm["specificity"]) and cm["specificity"] >= target:
            feasible.append((cm["sensitivity"], cm["specificity"], float(t)))
    if not feasible:
        return float("nan"), float("nan"), float("nan")
    # Max sensitivity, then specificity, then the lower threshold (more sensitive).
    sens, spec, thr = max(feasible, key=lambda z: (z[0], z[1], -z[2]))
    return thr, sens, spec


def bootstrap_classification(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], score[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc(yb, sb))
        aps.append(average_precision(yb, sb))
    def ci(vals):
        if not vals:
            return (float("nan"), float("nan"))
        return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))
    auc_lo, auc_hi = ci(aucs)
    ap_lo, ap_hi = ci(aps)
    return {"auc_low": auc_lo, "auc_high": auc_hi, "ap_low": ap_lo, "ap_high": ap_hi, "valid": len(aucs)}


def fit_line(dev: pd.DataFrame, x: str, y: str) -> Tuple[float, float]:
    d = finite_pair(dev, x, y)
    if len(d) < 2 or d[x].nunique() < 2:
        return float("nan"), float("nan")
    lr = stats.linregress(d[x].to_numpy(float), d[y].to_numpy(float))
    return float(lr.slope), float(lr.intercept)


def regression_metrics(test: pd.DataFrame, x: str, y: str, slope: float, intercept: float) -> Dict[str, float]:
    d = finite_pair(test, x, y)
    if len(d) == 0 or not np.isfinite(slope) or not np.isfinite(intercept):
        return {"n": len(d), "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
    yt = d[y].to_numpy(float)
    yp = intercept + slope * d[x].to_numpy(float)
    resid = yt - yp
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    pearson = stats.pearsonr(yp, yt).statistic if len(d) >= 2 and np.std(yp) > 0 and np.std(yt) > 0 else np.nan
    spearman = stats.spearmanr(yp, yt).statistic if len(d) >= 2 else np.nan
    return {
        "n": int(len(d)),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": float(1 - sse / sst) if sst > 0 else np.nan,
        "pearson_r": float(pearson) if np.isfinite(pearson) else np.nan,
        "spearman_rho": float(spearman) if np.isfinite(spearman) else np.nan,
    }


def bootstrap_regression(test: pd.DataFrame, x: str, y: str, slope: float, intercept: float, n_boot: int, seed: int) -> Dict[str, float]:
    d = finite_pair(test, x, y).reset_index(drop=True)
    if len(d) < 2 or not np.isfinite(slope):
        return {"mae_low": np.nan, "mae_high": np.nan, "spearman_low": np.nan, "spearman_high": np.nan, "valid": 0}
    rng = np.random.default_rng(seed)
    maes, rhos = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        b = d.iloc[idx]
        m = regression_metrics(b, x, y, slope, intercept)
        if np.isfinite(m["mae"]):
            maes.append(m["mae"])
        if np.isfinite(m["spearman_rho"]):
            rhos.append(m["spearman_rho"])
    def ci(vals):
        if not vals:
            return (np.nan, np.nan)
        return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))
    ml, mh = ci(maes); sl, sh = ci(rhos)
    return {"mae_low": ml, "mae_high": mh, "spearman_low": sl, "spearman_high": sh, "valid": min(len(maes), len(rhos))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--bad-percentiles", nargs="+", type=float, default=[75.0, 50.0])
    ap.add_argument("--target-specificity", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=88)
    args = ap.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_splits(evaluation_dir)
    dev = df[df["analysis_split"].isin(["train", "validation"])].copy()
    test = df[df["analysis_split"] == "test"].copy()

    reg_rows = []
    class_rows = []
    pred_rows = []
    for i, (metric, (qa_col, true_col)) in enumerate(METRICS.items()):
        slope, intercept = fit_line(dev, qa_col, true_col)
        rm = regression_metrics(test, qa_col, true_col, slope, intercept)
        rb = bootstrap_regression(test, qa_col, true_col, slope, intercept, args.n_boot, args.seed + i * 101)
        reg_rows.append({
            "roi": args.roi,
            "variant": args.variant,
            "metric": metric,
            "qa_signal": qa_col,
            "true_target": true_col,
            "development_n": int(len(finite_pair(dev, qa_col, true_col))),
            "test_n": rm["n"],
            "slope_development": slope,
            "intercept_development": intercept,
            "test_mae": rm["mae"],
            "test_mae_ci95_low": rb["mae_low"],
            "test_mae_ci95_high": rb["mae_high"],
            "test_rmse": rm["rmse"],
            "test_r2": rm["r2"],
            "test_pearson_r": rm["pearson_r"],
            "test_spearman_rho": rm["spearman_rho"],
            "test_spearman_ci95_low": rb["spearman_low"],
            "test_spearman_ci95_high": rb["spearman_high"],
            "bootstrap_valid_replicates": rb["valid"],
        })
        td = finite_pair(test, qa_col, true_col).copy()
        td["metric"] = metric
        td["qa_regression_prediction"] = intercept + slope * td[qa_col] if np.isfinite(slope) else np.nan
        pred_rows.append(td[[c for c in ["patient_id", "analysis_split", "metric", qa_col, true_col, "qa_regression_prediction"] if c in td.columns]])

        for j, pct in enumerate(args.bad_percentiles):
            cutoff = percentile_higher(dev[true_col], pct)
            dd = finite_pair(dev, qa_col, true_col)
            tt = finite_pair(test, qa_col, true_col)
            ydev = (dd[true_col].to_numpy(float) > cutoff).astype(int)
            ytest = (tt[true_col].to_numpy(float) > cutoff).astype(int)
            sdev = dd[qa_col].to_numpy(float)
            stest = tt[qa_col].to_numpy(float)
            auc = roc_auc(ytest, stest)
            apv = average_precision(ytest, stest)
            boot = bootstrap_classification(ytest, stest, args.n_boot, args.seed + 1000 + i * 101 + j * 17)
            thr, dsens, dspec = choose_threshold_at_specificity(ydev, sdev, args.target_specificity)
            cm = confusion(ytest, stest, thr)
            class_rows.append({
                "roi": args.roi,
                "variant": args.variant,
                "metric": metric,
                "qa_signal": qa_col,
                "true_target": true_col,
                "true_error_percentile_from_development": pct,
                "true_error_cutoff": cutoff,
                "development_n": int(len(dd)),
                "development_bad": int(ydev.sum()),
                "development_good": int((ydev == 0).sum()),
                "test_n": int(len(tt)),
                "test_bad": int(ytest.sum()),
                "test_good": int((ytest == 0).sum()),
                "test_roc_auc": auc,
                "test_roc_auc_ci95_low": boot["auc_low"],
                "test_roc_auc_ci95_high": boot["auc_high"],
                "test_pr_auc": apv,
                "test_pr_auc_ci95_low": boot["ap_low"],
                "test_pr_auc_ci95_high": boot["ap_high"],
                "development_qa_threshold_target_specificity": args.target_specificity,
                "development_selected_qa_threshold": thr,
                "development_sensitivity_at_selected_threshold": dsens,
                "development_specificity_at_selected_threshold": dspec,
                **{f"test_{k}_fixed_dev_threshold": v for k, v in cm.items()},
                "is_primary_endpoint": bool(pct == 75.0),
                "bootstrap_valid_replicates": boot["valid"],
            })

    pd.DataFrame(reg_rows).to_csv(out / "qa_regression_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "qa_classification_summary.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(out / "qa_regression_test_predictions.csv", index=False)
    metadata = {
        "roi": args.roi,
        "variant": args.variant,
        "qa_signal_definition": "corresponding metric between input autoseg and model corrected output",
        "true_error_definition": "corresponding metric between input autoseg and approved contour",
        "development": "train + validation",
        "test": "held-out test split from the supervised-correction fold",
        "binary_primary": "true error > development P75",
        "binary_secondary": "true error > development P50",
        "qa_threshold": f"chosen on development for >= {args.target_specificity:.2f} specificity then frozen",
        "test_used_for_fitting_or_threshold_selection": False,
        "metrics": list(METRICS),
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    (out / "qa_protocol.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote QA summaries to {out}")


if __name__ == "__main__":
    main()
