#!/usr/bin/env python3
"""Aggregate the predeclared best supervised-correction QA readouts across outer folds.

This script intentionally evaluates a fixed set of readout families selected before
looking at the remaining outer-fold test sets:

* handcrafted_soft_local
* handcrafted_soft_local_boundary
* joint_spatial_action_head
* spatial_plus_compact_soft (fusion)

Each outer fold supplies one untouched test partition. Regression predictions are
pooled in the native metric units. Binary labels use that fold's development-set
P75/P50 true-error cutoff, and operating decisions use that fold's development-OOF
threshold, so no test labels are used for threshold selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

TARGETS = ("1_minus_dice", "hd95", "assd")
PERCENTILES = (75, 50)
METHODS = {
    "handcrafted_soft_local": {
        "kind": "soft",
        "feature_group": "soft_local",
        "description": "Full handcrafted P(add/remove/action) distribution summaries in the autoseg-local region.",
    },
    "handcrafted_soft_local_boundary": {
        "kind": "soft",
        "feature_group": "soft_local_boundary",
        "description": "Concatenated full handcrafted soft summaries from autoseg-local and autoseg-boundary regions.",
    },
    "joint_spatial_action_head": {
        "kind": "spatial",
        "feature_group": "learned_spatial_action_head",
        "description": "Joint 3-target tiny 3D CNN on the 12-channel pooled local/boundary action tensor.",
    },
    "spatial_plus_compact_soft": {
        "kind": "fusion",
        "feature_group": "spatial_plus_compact_soft",
        "description": "Joint spatial embedding fused with 18 compact handcrafted local/boundary soft features.",
    },
}

# Predeclared from fold-0 evidence. Cross-fold outputs still report all methods so
# this mapping can be confirmed or rejected rather than silently re-selected.
PRIMARY_MAPPING = {
    "classification": {
        "1_minus_dice": "joint_spatial_action_head",
        "hd95": "joint_spatial_action_head",
        "assd": "joint_spatial_action_head",
    },
    "regression": {
        "1_minus_dice": "handcrafted_soft_local",
        "hd95": "joint_spatial_action_head",
        "assd": "handcrafted_soft_local_boundary",
    },
}


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _reg_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    out = {"n": int(len(y))}
    if len(y) == 0:
        return out | {k: float("nan") for k in ("mae", "rmse", "r2", "pearson", "spearman")}
    out["mae"] = float(mean_absolute_error(y, pred))
    out["rmse"] = float(math.sqrt(mean_squared_error(y, pred)))
    out["r2"] = float(r2_score(y, pred)) if len(y) >= 2 else float("nan")
    out["pearson"] = float(pearsonr(y, pred).statistic) if len(y) >= 2 and np.std(y) > 0 and np.std(pred) > 0 else float("nan")
    out["spearman"] = float(spearmanr(y, pred).statistic) if len(y) >= 2 and np.std(y) > 0 and np.std(pred) > 0 else float("nan")
    return out


def _classification_metrics(y: np.ndarray, score: np.ndarray, pred_label: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(score) & np.isfinite(pred_label)
    y = y[mask].astype(int)
    score = score[mask]
    pred_label = pred_label[mask].astype(int)
    out = {
        "n": int(len(y)),
        "positive_n": int(np.sum(y == 1)),
        "negative_n": int(np.sum(y == 0)),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    tp = int(np.sum((y == 1) & (pred_label == 1)))
    fp = int(np.sum((y == 0) & (pred_label == 1)))
    tn = int(np.sum((y == 0) & (pred_label == 0)))
    fn = int(np.sum((y == 1) & (pred_label == 0)))
    out.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn})
    out["sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
    out["specificity"] = tn / (tn + fp) if tn + fp else float("nan")
    out["ppv"] = tp / (tp + fp) if tp + fp else float("nan")
    out["npv"] = tn / (tn + fn) if tn + fn else float("nan")
    return out


def _bootstrap_regression(y, pred, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    maes, rhos = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], pred[idx]
        maes.append(mean_absolute_error(yy, pp))
        if np.std(yy) > 0 and np.std(pp) > 0:
            rhos.append(spearmanr(yy, pp).statistic)
    def ci(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (float("nan"), float("nan"))
    mae_ci = ci(maes)
    rho_ci = ci(rhos)
    return {
        "mae_ci95_low": mae_ci[0], "mae_ci95_high": mae_ci[1],
        "spearman_ci95_low": rho_ci[0], "spearman_ci95_high": rho_ci[1],
    }


def _bootstrap_classification(y, score, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    aucs, aps = [], []
    if len(pos) == 0 or len(neg) == 0:
        return {k: float("nan") for k in ("roc_auc_ci95_low", "roc_auc_ci95_high", "pr_auc_ci95_low", "pr_auc_ci95_high")}
    for _ in range(n_boot):
        # Stratified bootstrap keeps both classes represented.
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        yy, ss = y[idx], score[idx]
        aucs.append(roc_auc_score(yy, ss))
        aps.append(average_precision_score(yy, ss))
    return {
        "roc_auc_ci95_low": float(np.percentile(aucs, 2.5)),
        "roc_auc_ci95_high": float(np.percentile(aucs, 97.5)),
        "pr_auc_ci95_low": float(np.percentile(aps, 2.5)),
        "pr_auc_ci95_high": float(np.percentile(aps, 97.5)),
    }


def _read_soft(fold_dir: Path, method: str, target: str, percentile: int):
    fg = METHODS[method]["feature_group"]
    analysis_dir = fold_dir / "soft_qa_readout_analysis"
    pred = pd.read_csv(analysis_dir / "soft_qa_test_predictions.csv")
    cls_sum = pd.read_csv(analysis_dir / "soft_qa_classification_summary.csv")
    reg = pred[(pred.target_metric == target) & (pred.feature_group == fg) & (pred.task == "regression")].copy()
    cls = pred[(pred.target_metric == target) & (pred.feature_group == fg) & (pred.task == f"classification_p{percentile}")].copy()
    row = cls_sum[(cls_sum.target_metric == target) & (cls_sum.feature_group == fg) & (cls_sum.bad_percentile == float(percentile))]
    if len(row) != 1:
        raise RuntimeError(f"Expected one soft classification summary row for {method}/{target}/P{percentile} in {fold_dir}; got {len(row)}")
    row = row.iloc[0]
    threshold = float(row.development_oof_threshold_at_target_specificity)
    reg_out = reg.rename(columns={"true_value": "true_value", "score_or_prediction": "prediction"})[["patient_id", "true_value", "prediction"]]
    cls_out = cls.rename(columns={"true_value": "y_true", "score_or_prediction": "score"})[["patient_id", "y_true", "score"]]
    cls_out["fold_threshold"] = threshold
    cls_out["predicted_label"] = (cls_out.score >= threshold).astype(int)
    cls_out["true_error_cutoff"] = float(row.true_error_cutoff)
    return reg_out, cls_out


def _read_spatial(fold_dir: Path, target: str, percentile: int):
    analysis_dir = fold_dir / "spatial_qa_head_analysis"
    pred = pd.read_csv(analysis_dir / "spatial_qa_test_predictions.csv")
    cls_sum = pd.read_csv(analysis_dir / "spatial_qa_classification_summary.csv")
    reg = pred[(pred.target_metric == target) & (pred.task == "regression")].copy()
    cls = pred[(pred.target_metric == target) & (pred.task == f"classification_p{percentile}")].copy()
    row = cls_sum[(cls_sum.target_metric == target) & (cls_sum.bad_percentile == float(percentile))]
    if len(row) != 1:
        raise RuntimeError(f"Expected one spatial classification summary row for {target}/P{percentile} in {fold_dir}; got {len(row)}")
    row = row.iloc[0]
    threshold = float(row.development_oof_threshold_at_target_specificity)
    reg_out = reg.rename(columns={"score_or_prediction": "prediction"})[["patient_id", "true_value", "prediction"]]
    cls_out = cls.rename(columns={"true_value": "y_true", "score_or_prediction": "score"})[["patient_id", "y_true", "score"]]
    cls_out["fold_threshold"] = threshold
    cls_out["predicted_label"] = (cls_out.score >= threshold).astype(int)
    cls_out["true_error_cutoff"] = float(row.true_error_cutoff)
    return reg_out, cls_out


def _read_fusion(fold_dir: Path, target: str, percentile: int):
    analysis_dir = fold_dir / "spatial_soft_fusion_qa_analysis"
    pred = pd.read_csv(analysis_dir / "fusion_qa_test_predictions.csv")
    cls_sum = pd.read_csv(analysis_dir / "fusion_qa_classification_summary.csv")
    reg = pred[pred.metric == target].copy()
    row = cls_sum[(cls_sum.metric == target) & (cls_sum.bad_percentile == float(percentile))]
    if len(row) != 1:
        raise RuntimeError(f"Expected one fusion classification summary row for {target}/P{percentile} in {fold_dir}; got {len(row)}")
    row = row.iloc[0]
    threshold = float(row.dev_operating_threshold)
    cutoff = float(row.true_error_cutoff)
    reg_out = reg.rename(columns={"true_error": "true_value", "predicted_error": "prediction"})[["patient_id", "true_value", "prediction"]]
    cls_out = reg_out.rename(columns={"true_value": "true_error", "prediction": "score"})[["patient_id", "true_error", "score"]]
    cls_out["y_true"] = (cls_out.true_error > cutoff).astype(int)
    cls_out["fold_threshold"] = threshold
    cls_out["predicted_label"] = (cls_out.score >= threshold).astype(int)
    cls_out["true_error_cutoff"] = cutoff
    cls_out = cls_out[["patient_id", "y_true", "score", "fold_threshold", "predicted_label", "true_error_cutoff"]]
    return reg_out, cls_out


def _load_method(fold_dir: Path, method: str, target: str, percentile: int):
    kind = METHODS[method]["kind"]
    if kind == "soft":
        return _read_soft(fold_dir, method, target, percentile)
    if kind == "spatial":
        return _read_spatial(fold_dir, target, percentile)
    if kind == "fusion":
        return _read_fusion(fold_dir, target, percentile)
    raise ValueError(kind)


def _macro_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out = {}
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
        out[f"fold_{col}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        out[f"fold_{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", default="/scratch/shaohan/contour_QA/supervised_autoseg_correction")
    ap.add_argument("--roi", required=True)
    ap.add_argument("--variant", default="large_decoder_ft_last2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--output-zip", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated readout methods to aggregate. Default: all fixed candidate methods.",
    )
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    selected_methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = [m for m in selected_methods if m not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}; choices={sorted(METHODS)}")
    if not selected_methods:
        raise ValueError("At least one method must be selected")
    root = Path(args.study_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "best_qa_5fold" / args.roi / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_predictions = []
    cls_predictions = []
    per_fold_reg = []
    per_fold_cls = []

    for fold in folds:
        fold_dir = root / "capacity_finetune_ablation" / args.roi / f"fold_{fold}" / args.variant
        if not fold_dir.is_dir():
            raise FileNotFoundError(fold_dir)
        print(f"Loading fold {fold}: {fold_dir}")
        for method in selected_methods:
            for target in TARGETS:
                # Regression is independent of the binary percentile; load once using P75 metadata path.
                reg, _ = _load_method(fold_dir, method, target, 75)
                reg["roi"] = args.roi
                reg["variant"] = args.variant
                reg["fold"] = fold
                reg["method"] = method
                reg["target_metric"] = target
                reg_predictions.append(reg)
                rm = _reg_metrics(reg.true_value.to_numpy(float), reg.prediction.to_numpy(float))
                per_fold_reg.append({"roi": args.roi, "variant": args.variant, "fold": fold, "method": method, "target_metric": target, **rm})
                for percentile in PERCENTILES:
                    _, cls = _load_method(fold_dir, method, target, percentile)
                    cls["roi"] = args.roi
                    cls["variant"] = args.variant
                    cls["fold"] = fold
                    cls["method"] = method
                    cls["target_metric"] = target
                    cls["bad_percentile"] = percentile
                    cls_predictions.append(cls)
                    cm = _classification_metrics(
                        cls.y_true.to_numpy(float), cls.score.to_numpy(float), cls.predicted_label.to_numpy(float)
                    )
                    per_fold_cls.append({"roi": args.roi, "variant": args.variant, "fold": fold, "method": method, "target_metric": target, "bad_percentile": percentile, **cm})

    reg_pred = pd.concat(reg_predictions, ignore_index=True)
    cls_pred = pd.concat(cls_predictions, ignore_index=True)
    fold_reg = pd.DataFrame(per_fold_reg)
    fold_cls = pd.DataFrame(per_fold_cls)

    # Each outer-fold test partition should contribute each patient once per method/target.
    dup = reg_pred.duplicated(["method", "target_metric", "patient_id"], keep=False)
    if dup.any():
        examples = reg_pred.loc[dup, ["method", "target_metric", "patient_id", "fold"]].head(20)
        raise RuntimeError(f"Duplicate patient test predictions across outer folds detected:\n{examples}")

    pooled_reg_rows = []
    for (method, target), g in reg_pred.groupby(["method", "target_metric"], sort=False):
        y = g.true_value.to_numpy(float)
        p = g.prediction.to_numpy(float)
        row = {"roi": args.roi, "variant": args.variant, "method": method, "target_metric": target, "folds": len(folds)}
        row.update({f"pooled_{k}": v for k, v in _reg_metrics(y, p).items()})
        row.update(_bootstrap_regression(y, p, args.n_boot, args.seed + sum(map(ord, method + "|" + target)) % 100000))
        fr = fold_reg[(fold_reg.method == method) & (fold_reg.target_metric == target)]
        row.update(_macro_stats(fr, ["mae", "rmse", "r2", "pearson", "spearman"]))
        pooled_reg_rows.append(row)
    pooled_reg = pd.DataFrame(pooled_reg_rows)

    pooled_cls_rows = []
    for (method, target, percentile), g in cls_pred.groupby(["method", "target_metric", "bad_percentile"], sort=False):
        y = g.y_true.to_numpy(float)
        s = g.score.to_numpy(float)
        pl = g.predicted_label.to_numpy(float)
        row = {"roi": args.roi, "variant": args.variant, "method": method, "target_metric": target, "bad_percentile": percentile, "folds": len(folds)}
        row.update({f"pooled_{k}": v for k, v in _classification_metrics(y, s, pl).items()})
        row.update(_bootstrap_classification(y.astype(int), s, args.n_boot, args.seed + sum(map(ord, method + "|" + target + "|" + str(percentile))) % 100000))
        fc = fold_cls[(fold_cls.method == method) & (fold_cls.target_metric == target) & (fold_cls.bad_percentile == percentile)]
        row.update(_macro_stats(fc, ["roc_auc", "pr_auc", "sensitivity", "specificity", "ppv", "npv"]))
        pooled_cls_rows.append(row)
    pooled_cls = pd.DataFrame(pooled_cls_rows)

    # Predeclared primary rows: these are not reselected using the five-fold result.
    primary_reg = []
    for target, method in PRIMARY_MAPPING["regression"].items():
        if method not in selected_methods:
            continue
        x = pooled_reg[(pooled_reg.target_metric == target) & (pooled_reg.method == method)].copy()
        if len(x):
            x.insert(0, "task", "regression")
            primary_reg.append(x)
    primary_cls = []
    for target, method in PRIMARY_MAPPING["classification"].items():
        if method not in selected_methods:
            continue
        x = pooled_cls[(pooled_cls.target_metric == target) & (pooled_cls.method == method) & (pooled_cls.bad_percentile == 75)].copy()
        if len(x):
            x.insert(0, "task", "classification_p75")
            primary_cls.append(x)
    primary_parts = primary_reg + primary_cls
    primary = pd.concat(primary_parts, ignore_index=True, sort=False) if primary_parts else pd.DataFrame()

    reg_pred.to_csv(out_dir / "best_qa_crossfold_test_predictions_regression.csv", index=False)
    cls_pred.to_csv(out_dir / "best_qa_crossfold_test_predictions_classification.csv", index=False)
    fold_reg.to_csv(out_dir / "best_qa_crossfold_regression_per_fold.csv", index=False)
    fold_cls.to_csv(out_dir / "best_qa_crossfold_classification_per_fold.csv", index=False)
    pooled_reg.to_csv(out_dir / "best_qa_crossfold_regression_summary.csv", index=False)
    pooled_cls.to_csv(out_dir / "best_qa_crossfold_classification_summary.csv", index=False)
    primary.to_csv(out_dir / "best_qa_crossfold_predeclared_primary_summary.csv", index=False)

    protocol = {
        "roi": args.roi,
        "variant": args.variant,
        "outer_folds": folds,
        "n_boot": args.n_boot,
        "methods": {m: METHODS[m] for m in selected_methods},
        "predeclared_primary_mapping": PRIMARY_MAPPING,
        "classification_note": (
            "Binary bad-case labels use each outer fold's development-set P75/P50 true-error cutoff. "
            "Operating decisions use that fold's development-OOF threshold targeting high specificity. "
            "Pooled sensitivity/specificity therefore combine fold-specific thresholds without test-set tuning."
        ),
        "selection_note": (
            "All four candidate readouts are reported across five folds. The primary mapping was declared from fold-0 evidence "
            "before evaluating folds 1-4 and is not reselected inside this script."
        ),
    }
    (out_dir / "best_qa_crossfold_protocol.json").write_text(json.dumps(protocol, indent=2))

    readme = f"""# Cross-fold supervised-correction QA summary\n\nROI: `{args.roi}`  \nCorrection representation: `{args.variant}`  \nOuter folds: `{','.join(map(str, folds))}`\n\nCandidate readouts are fixed to handcrafted soft-local, handcrafted soft-local+boundary, the joint learned spatial action head, and spatial+compact-soft fusion. The CSV summaries report both pooled held-out predictions and mean±SD of per-fold metrics.\n\nThe predeclared fold-0 mapping is stored in `best_qa_crossfold_predeclared_primary_summary.csv`; use the full candidate summaries to decide whether that mapping generalizes before extending the workflow across ROIs.\n"""
    (out_dir / "README.md").write_text(readme)

    zip_path = Path(args.output_zip) if args.output_zip else out_dir.parent / f"{args.variant}_best_qa_5fold_results_no_models.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out_dir)))
    print(f"Wrote cross-fold summary: {out_dir}")
    print(f"Wrote uploadable ZIP: {zip_path}")


if __name__ == "__main__":
    main()
