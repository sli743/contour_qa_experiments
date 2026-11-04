#!/usr/bin/env python3
"""Compare hard-output and soft action-map patient-level QA readouts.

Development protocol
--------------------
* Development = train + validation; test remains untouched until final scoring.
* Binary bad-case cutoffs are percentiles of development true error.
* For multivariate soft readouts, regularization is selected by cross-validation
  inside development only. The high-specificity operating threshold is selected
  from out-of-fold development scores, then frozen for test.
* Regression Ridge alpha is selected by development CV, then refit on all
  development patients and evaluated once on test.

All predictor features are deployment-feasible: input/model-output disagreement
or softmax action probabilities measured in autoseg-derived spatial regions.
Approved contours are used only to define QA targets and binary labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.analyze_supervised_correction_qa_signal import (
    bootstrap_classification,
    bootstrap_regression,
    choose_threshold_at_specificity,
    confusion,
    percentile_higher,
    regression_metrics,
)

TARGETS = {
    "1_minus_dice": "true_1_minus_dice",
    "hd95": "true_hd95_mm",
    "assd": "true_assd_mm",
}
HARD_FEATURES = ["qa_1_minus_dice", "qa_hd95_mm", "qa_assd_mm"]
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def load_evaluation(evaluation_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "validation", "test"):
        path = evaluation_dir / f"{split}_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, dtype={"patient_id": str})
        df["analysis_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_soft_features(soft_root: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "validation", "test"):
        path = soft_root / split / "patient_action_score_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, dtype={"patient_id": str})
        df["analysis_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def merge_inputs(evaluation_dir: Path, soft_root: Path) -> pd.DataFrame:
    ev = load_evaluation(evaluation_dir)
    sf = load_soft_features(soft_root)
    drop = [c for c in sf.columns if c in ev.columns and c not in {"patient_id", "analysis_split"}]
    sf = sf.drop(columns=drop)
    out = ev.merge(sf, on=["patient_id", "analysis_split"], how="inner", validate="one_to_one")
    expected = len(ev)
    if len(out) != expected:
        raise RuntimeError(f"Soft-feature/evaluation merge lost cases: evaluation={expected}, merged={len(out)}")
    return out


def is_soft_stat(c: str, region: str) -> bool:
    prefixes = tuple(f"{region}_{name}_" for name in ("action", "add", "remove"))
    if not c.startswith(prefixes):
        return False
    # Deliberately exclude absolute voxel counts; retain normalized distribution/coherence statistics.
    return "_n_ge_" not in c


def build_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    local = sorted(c for c in df.columns if is_soft_stat(c, "local"))
    boundary = sorted(c for c in df.columns if is_soft_stat(c, "boundary"))
    missing_hard = [c for c in HARD_FEATURES if c not in df.columns]
    if missing_hard:
        raise KeyError(f"Missing hard QA features: {missing_hard}")
    if not local or not boundary:
        raise RuntimeError(f"Expected local and boundary soft features, got local={len(local)} boundary={len(boundary)}")
    compact_suffixes = [
        "action_mean", "action_p95", "action_p99", "action_top100_mean", "action_frac_ge_0p1",
        "add_p95", "add_p99", "remove_p95", "remove_p99",
    ]
    local_compact = [f"local_{s}" for s in compact_suffixes if f"local_{s}" in df.columns]
    boundary_compact = [f"boundary_{s}" for s in compact_suffixes if f"boundary_{s}" in df.columns]
    return {
        "hard_all": HARD_FEATURES,
        "soft_local_compact": local_compact,
        "soft_boundary_compact": boundary_compact,
        "soft_compact_combined": local_compact + boundary_compact,
        "soft_local": local,
        "soft_boundary": boundary,
        "soft_local_boundary": local + boundary,
        "hard_plus_soft": HARD_FEATURES + local + boundary,
    }


def prepare_xy(df: pd.DataFrame, features: List[str], target: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = list(features) + [target]
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(d.to_numpy(float)).all(axis=1)
    return df.loc[finite].copy(), d.loc[finite, features].to_numpy(float), d.loc[finite, target].to_numpy(float)


def classification_model(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="liblinear")),
    ])


def regression_model(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])


def choose_classification_C(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[float, int, float]:
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    folds = min(5, n_pos, n_neg)
    if folds < 2:
        return 1.0, 0, np.nan
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best = None
    for C in C_GRID:
        score = float(np.mean(cross_val_score(classification_model(C), X, y, cv=cv, scoring="roc_auc")))
        candidate = (score, -abs(np.log10(C)), C)
        if best is None or candidate > best:
            best = candidate
    return float(best[2]), folds, float(best[0])


def oof_classification_scores(X: np.ndarray, y: np.ndarray, C: float, folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        return np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(classification_model(C), X, y, cv=cv, method="predict_proba")[:, 1]


def choose_ridge_alpha(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[float, int, float]:
    folds = min(5, len(y))
    if folds < 2:
        return 1.0, 0, np.nan
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    best = None
    for alpha in ALPHA_GRID:
        score = float(np.mean(cross_val_score(regression_model(alpha), X, y, cv=cv, scoring="neg_mean_absolute_error")))
        candidate = (score, -abs(np.log10(alpha)), alpha)
        if best is None or candidate > best:
            best = candidate
    return float(best[2]), folds, float(best[0])


def correlations(y: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    if len(y) < 2 or np.std(y) == 0 or np.std(score) == 0:
        return np.nan, np.nan
    return float(stats.pearsonr(score, y).statistic), float(stats.spearmanr(score, y).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-dir", required=True)
    ap.add_argument("--soft-feature-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--bad-percentiles", nargs="+", type=float, default=[75.0, 50.0])
    ap.add_argument("--target-specificity", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=88)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = merge_inputs(Path(args.evaluation_dir), Path(args.soft_feature_root))
    groups = build_feature_groups(df)
    dev = df[df.analysis_split.isin(["train", "validation"])].copy()
    test = df[df.analysis_split == "test"].copy()

    class_rows = []
    reg_rows = []
    pred_rows = []

    for target_i, (metric, target_col) in enumerate(TARGETS.items()):
        corresponding_hard = {
            "1_minus_dice": "qa_1_minus_dice",
            "hd95": "qa_hd95_mm",
            "assd": "qa_assd_mm",
        }[metric]
        target_groups = {"hard_corresponding": [corresponding_hard], **groups}

        for group_i, (group_name, features) in enumerate(target_groups.items()):
            seed = args.seed + target_i * 1000 + group_i * 37
            ddev, Xdev, ydev = prepare_xy(dev, features, target_col)
            dtest, Xtest, ytest = prepare_xy(test, features, target_col)
            if len(ddev) < 4 or len(dtest) < 1:
                continue

            # Regression QA.
            alpha, nfold, cv_neg_mae = choose_ridge_alpha(Xdev, ydev, seed)
            reg = regression_model(alpha).fit(Xdev, ydev)
            pred = reg.predict(Xtest)
            tmp = pd.DataFrame({"x": pred, "y": ytest})
            rm = regression_metrics(tmp, "x", "y", 1.0, 0.0)
            rb = bootstrap_regression(tmp, "x", "y", 1.0, 0.0, args.n_boot, seed + 11)
            raw_pearson, raw_spearman = correlations(ytest, pred)
            reg_rows.append({
                "roi": args.roi, "variant": args.variant, "target_metric": metric,
                "feature_group": group_name, "n_features": len(features),
                "development_n": len(ddev), "test_n": len(dtest),
                "selected_ridge_alpha": alpha, "development_cv_folds": nfold,
                "development_cv_neg_mae": cv_neg_mae,
                "test_mae": rm["mae"], "test_mae_ci95_low": rb["mae_low"], "test_mae_ci95_high": rb["mae_high"],
                "test_rmse": rm["rmse"], "test_r2": rm["r2"],
                "test_pearson_r": raw_pearson, "test_spearman_rho": raw_spearman,
                "test_spearman_ci95_low": rb["spearman_low"], "test_spearman_ci95_high": rb["spearman_high"],
            })
            for pid, yt, yp in zip(dtest.patient_id.astype(str), ytest, pred):
                pred_rows.append({"patient_id": pid, "target_metric": metric, "feature_group": group_name,
                                  "task": "regression", "true_value": float(yt), "score_or_prediction": float(yp)})

            # Binary QA at each development-defined bad-case cutoff.
            for pct in args.bad_percentiles:
                cutoff = percentile_higher(dev[target_col], pct)
                ydev_cls = (ydev > cutoff).astype(int)
                ytest_cls = (ytest > cutoff).astype(int)
                if len(np.unique(ydev_cls)) < 2:
                    continue
                C, cfold, cv_auc = choose_classification_C(Xdev, ydev_cls, seed + int(pct))
                oof = oof_classification_scores(Xdev, ydev_cls, C, cfold, seed + int(pct))
                qa_thr, dev_sens, dev_spec = choose_threshold_at_specificity(ydev_cls, oof, args.target_specificity)
                clf = classification_model(C).fit(Xdev, ydev_cls)
                score = clf.predict_proba(Xtest)[:, 1]
                auc = float(roc_auc_score(ytest_cls, score)) if len(np.unique(ytest_cls)) == 2 else np.nan
                apv = float(average_precision_score(ytest_cls, score)) if ytest_cls.sum() else np.nan
                boot = bootstrap_classification(ytest_cls, score, args.n_boot, seed + int(pct) + 19)
                cm = confusion(ytest_cls, score, qa_thr)
                class_rows.append({
                    "roi": args.roi, "variant": args.variant, "target_metric": metric,
                    "feature_group": group_name, "n_features": len(features), "bad_percentile": pct,
                    "true_error_cutoff": cutoff, "development_n": len(ddev), "test_n": len(dtest),
                    "test_positive_n": int(ytest_cls.sum()), "test_negative_n": int((1-ytest_cls).sum()),
                    "selected_logistic_C": C, "development_cv_folds": cfold, "development_cv_auc": cv_auc,
                    "test_roc_auc": auc, "test_roc_auc_ci95_low": boot["auc_low"], "test_roc_auc_ci95_high": boot["auc_high"],
                    "test_pr_auc": apv, "test_pr_auc_ci95_low": boot["ap_low"], "test_pr_auc_ci95_high": boot["ap_high"],
                    "development_oof_threshold_at_target_specificity": qa_thr,
                    "development_oof_sensitivity": dev_sens, "development_oof_specificity": dev_spec,
                    "test_sensitivity": cm["sensitivity"], "test_specificity": cm["specificity"],
                    "test_ppv": cm["ppv"], "test_npv": cm["npv"],
                })
                for pid, yt, sc in zip(dtest.patient_id.astype(str), ytest_cls, score):
                    pred_rows.append({"patient_id": pid, "target_metric": metric, "feature_group": group_name,
                                      "task": f"classification_p{pct:g}", "true_value": int(yt),
                                      "score_or_prediction": float(sc)})

    pd.DataFrame(class_rows).to_csv(out / "soft_qa_classification_summary.csv", index=False)
    pd.DataFrame(reg_rows).to_csv(out / "soft_qa_regression_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "soft_qa_test_predictions.csv", index=False)
    manifest = {
        "roi": args.roi, "variant": args.variant,
        "development_splits": ["train", "validation"], "test_split": "test",
        "approved_used_for_features": False,
        "feature_groups": groups,
        "hard_corresponding_is_single_matching_input_output_metric": True,
        "classification": {
            "bad_percentiles": args.bad_percentiles,
            "target_specificity": args.target_specificity,
            "C_grid": C_GRID,
            "regularization_selected_by_development_stratified_cv": True,
            "operating_threshold_selected_from_development_oof_scores": True,
        },
        "regression": {"alpha_grid": ALPHA_GRID, "regularization_selected_by_development_cv": True},
    }
    (out / "soft_qa_protocol_and_features.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved soft QA readout analysis to {out}")


if __name__ == "__main__":
    main()
