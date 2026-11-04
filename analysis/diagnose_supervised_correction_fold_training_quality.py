#!/usr/bin/env python3
"""Summarize whether outer-fold QA variability tracks upstream training quality.

This diagnostic is intentionally read-only. It combines, per outer fold:
  * training history at the validation-loss-selected checkpoint,
  * validation add/remove action-score discrimination,
  * held-out joint-spatial QA classification and regression performance,
  * training-critical configuration values.

The goal is to distinguish "bad QA because the correction/error-localization model
trained poorly" from ordinary split/generalization variability. Correlations across
five folds are descriptive only; n=5 is too small for inferential claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr


DEFAULT_FOLDS = "0,1,2,3,4"
DEFAULT_METHOD = "joint_spatial_action_head"


def parse_folds(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def get_nested(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def canonical(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)


def safe_float(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def safe_corr(x: pd.Series, y: pd.Series, kind: str) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    try:
        if kind == "pearson":
            return float(pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)
        return float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)
    except Exception:
        return np.nan


def find_qa_tables(study_root: Path, roi: str, variant: str) -> tuple[Path, Path]:
    roots = [
        study_root / "best_qa_5fold" / roi / variant,
        study_root / "joint_spatial_qa_5fold" / roi / variant,
    ]
    for root in roots:
        cls = root / "best_qa_crossfold_classification_per_fold.csv"
        reg = root / "best_qa_crossfold_regression_per_fold.csv"
        if cls.is_file() and reg.is_file():
            return cls, reg
    tried = "\n".join(str(r) for r in roots)
    raise FileNotFoundError(
        "Could not find cross-fold QA per-fold tables. Tried:\n" + tried
    )


def action_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    sub = df[df["region"].astype(str) == "autoseg_local"].copy()
    out: dict[str, float] = {}
    for action in ("add", "remove"):
        row = sub[sub["action"].astype(str) == action]
        if row.empty:
            continue
        r = row.iloc[0]
        out[f"validation_{action}_roc_auc"] = safe_float(r.get("roc_auc"))
        out[f"validation_{action}_average_precision"] = safe_float(r.get("average_precision"))
        out[f"validation_{action}_prevalence"] = safe_float(r.get("positive_prevalence"))
        out[f"validation_{action}_ap_over_prevalence"] = safe_float(r.get("ap_over_prevalence"))
    return out


def history_metrics(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    hist = pd.read_csv(path)
    if hist.empty:
        raise RuntimeError(f"Empty training history: {path}")
    for col in ("epoch", "validation_total"):
        if col not in hist:
            raise KeyError(f"Missing {col!r} in {path}")
    valid = pd.to_numeric(hist["validation_total"], errors="coerce")
    if valid.notna().sum() == 0:
        raise RuntimeError(f"No finite validation_total values in {path}")
    best_idx = valid.idxmin()
    best = hist.loc[best_idx]
    gain = pd.to_numeric(hist.get("validation_correction_gain"), errors="coerce")
    best_gain_idx = gain.idxmax() if gain is not None and gain.notna().any() else best_idx
    best_gain = hist.loc[best_gain_idx]
    requested_epochs = int(get_nested(cfg, "training.num_epochs") or len(hist))
    patience = int(get_nested(cfg, "training.early_stopping_patience") or 0)

    def val(row: pd.Series, key: str) -> float:
        return safe_float(row.get(key))

    return {
        "epochs_ran": int(len(hist)),
        "requested_epochs": requested_epochs,
        "early_stopped": bool(len(hist) < requested_epochs),
        "early_stopping_patience": patience,
        "best_epoch": int(best["epoch"]),
        "best_validation_total": val(best, "validation_total"),
        "train_total_at_best": val(best, "train_total"),
        "train_minus_validation_total_at_best": val(best, "train_total") - val(best, "validation_total"),
        "validation_correction_gain_at_best": val(best, "validation_correction_gain"),
        "validation_needed_recall_at_best": val(best, "validation_needed_correction_recall"),
        "validation_add_recall_at_best": val(best, "validation_missing_correction_recall"),
        "validation_remove_recall_at_best": val(best, "validation_excess_correction_recall"),
        "validation_unnecessary_change_at_best": val(best, "validation_unnecessary_change_fraction"),
        "best_gain_epoch": int(best_gain["epoch"]),
        "best_validation_correction_gain": val(best_gain, "validation_correction_gain"),
        "last_validation_total": val(hist.iloc[-1], "validation_total"),
    }


def qa_metrics_for_fold(
    cls_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    fold: int,
    method: str,
    bad_percentile: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    c = cls_df[
        (pd.to_numeric(cls_df["fold"], errors="coerce") == fold)
        & (cls_df["method"].astype(str) == method)
        & (pd.to_numeric(cls_df["bad_percentile"], errors="coerce") == bad_percentile)
    ]
    for metric in ("1_minus_dice", "assd", "hd95"):
        row = c[c["target_metric"].astype(str) == metric]
        if not row.empty:
            r = row.iloc[0]
            out[f"qa_{metric}_roc_auc"] = safe_float(r.get("roc_auc"))
            out[f"qa_{metric}_pr_auc"] = safe_float(r.get("pr_auc"))
            out[f"qa_{metric}_sensitivity"] = safe_float(r.get("sensitivity"))
            out[f"qa_{metric}_specificity"] = safe_float(r.get("specificity"))
            out[f"qa_{metric}_positive_n"] = safe_float(r.get("positive_n"))
    rfold = reg_df[
        (pd.to_numeric(reg_df["fold"], errors="coerce") == fold)
        & (reg_df["method"].astype(str) == method)
    ]
    for metric in ("1_minus_dice", "assd", "hd95"):
        row = rfold[rfold["target_metric"].astype(str) == metric]
        if not row.empty:
            r = row.iloc[0]
            out[f"qa_{metric}_spearman"] = safe_float(r.get("spearman"))
            out[f"qa_{metric}_pearson"] = safe_float(r.get("pearson"))
            out[f"qa_{metric}_r2"] = safe_float(r.get("r2"))
            out[f"qa_{metric}_mae"] = safe_float(r.get("mae"))
    return out


CONFIG_KEYS = [
    "dino.repo_path",
    "dino.checkpoint_path",
    "dino.config_name",
    "decoder.channels",
    "decoder.blocks_per_stage",
    "training.device",
    "training.batch_size",
    "training.accumulation_steps",
    "training.num_workers",
    "training.num_epochs",
    "training.early_stopping_patience",
    "training.learning_rate",
    "training.weight_decay",
    "training.grad_clip",
    "training.precision",
    "training.paired_augmentation.enabled",
    "training.paired_augmentation.probability",
    "training.paired_augmentation.rotation_deg",
    "training.paired_augmentation.translation_voxels",
    "training.paired_augmentation.scale_min",
    "training.paired_augmentation.scale_max",
    "loss.type",
    "loss.preserve_class_weight",
    "loss.add_class_weight",
    "loss.remove_class_weight",
    "transfer_learning.finetune_last_n_blocks",
    "transfer_learning.encoder_learning_rate",
    "transfer_learning.encoder_weight_decay",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", default="/scratch/shaohan/contour_QA/supervised_autoseg_correction")
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--variant", default="large_decoder_ft_last2")
    ap.add_argument("--folds", default=DEFAULT_FOLDS)
    ap.add_argument("--method", default=DEFAULT_METHOD)
    ap.add_argument("--bad-percentile", type=int, default=75)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    study_root = Path(args.study_root)
    folds = parse_folds(args.folds)
    output = Path(args.output_dir) if args.output_dir else (
        study_root / "training_quality_diagnostics" / args.roi / args.variant
    )
    output.mkdir(parents=True, exist_ok=True)

    cls_path, reg_path = find_qa_tables(study_root, args.roi, args.variant)
    cls_df = pd.read_csv(cls_path)
    reg_df = pd.read_csv(reg_path)

    rows: list[dict[str, Any]] = []
    configs: dict[int, dict[str, Any]] = {}
    for fold in folds:
        out = study_root / "capacity_finetune_ablation" / args.roi / f"fold_{fold}" / args.variant
        cfg_path = out / "config.yaml"
        history_path = out / "models" / f"capacity_ft_ablation_{args.variant}" / "logs" / "history.csv"
        action_path = out / "action_score_diagnostic" / "action_score_discrimination.csv"
        missing = [str(p) for p in (cfg_path, history_path, action_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing fold {fold} diagnostic prerequisites:\n" + "\n".join(missing)
            )
        cfg = yaml.safe_load(cfg_path.read_text())
        configs[fold] = cfg
        row: dict[str, Any] = {"roi": args.roi, "variant": args.variant, "fold": fold}
        row.update(history_metrics(history_path, cfg))
        row.update(action_metrics(action_path))
        row.update(qa_metrics_for_fold(cls_df, reg_df, fold, args.method, args.bad_percentile))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    summary.to_csv(output / "fold_training_quality_vs_qa.csv", index=False)

    # Verify the fields that should be identical across folds.
    config_rows = []
    for key in CONFIG_KEYS:
        values = {fold: canonical(get_nested(configs[fold], key)) for fold in folds}
        uniq = set(values.values())
        row: dict[str, Any] = {"config_key": key, "all_equal": len(uniq) == 1}
        row.update({f"fold_{fold}": values[fold] for fold in folds})
        config_rows.append(row)
    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(output / "training_config_consistency.csv", index=False)

    upstream = [
        "best_epoch",
        "best_validation_total",
        "train_minus_validation_total_at_best",
        "validation_correction_gain_at_best",
        "validation_needed_recall_at_best",
        "validation_add_recall_at_best",
        "validation_remove_recall_at_best",
        "validation_unnecessary_change_at_best",
        "validation_add_roc_auc",
        "validation_add_average_precision",
        "validation_remove_roc_auc",
        "validation_remove_average_precision",
    ]
    downstream = [
        "qa_1_minus_dice_roc_auc",
        "qa_assd_roc_auc",
        "qa_hd95_roc_auc",
        "qa_1_minus_dice_spearman",
        "qa_assd_spearman",
        "qa_hd95_spearman",
    ]
    corr_rows = []
    for x in upstream:
        if x not in summary:
            continue
        for y in downstream:
            if y not in summary:
                continue
            pair = summary[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            corr_rows.append({
                "upstream_metric": x,
                "downstream_qa_metric": y,
                "n_folds": int(len(pair)),
                "pearson_r": safe_corr(summary[x], summary[y], "pearson"),
                "spearman_rho": safe_corr(summary[x], summary[y], "spearman"),
                "interpretation": "descriptive_only_n5",
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(output / "training_quality_vs_qa_correlations.csv", index=False)

    # Compact markdown report to make the diagnostic easy to inspect in a job log or upload.
    unequal = config_df[~config_df["all_equal"]]
    report = [
        f"# Training-quality diagnostic: {args.roi} / {args.variant}",
        "",
        f"Folds: {folds}",
        f"QA method: `{args.method}`; binary bad-case percentile: P{args.bad_percentile}",
        "",
        "## Configuration consistency",
        "",
    ]
    if unequal.empty:
        report.append("All training-critical configuration fields checked here are identical across folds.")
    else:
        report.append("WARNING: training-critical configuration mismatches were found:")
        report.extend([f"- `{k}`" for k in unequal["config_key"].tolist()])
    report.extend([
        "",
        "## Fold table",
        "",
        "The main CSV is `fold_training_quality_vs_qa.csv`. Key columns are shown below.",
        "",
    ])
    show_cols = [
        "fold", "best_epoch", "best_validation_total",
        "validation_add_average_precision", "validation_remove_average_precision",
        "validation_add_roc_auc", "validation_remove_roc_auc",
        "qa_1_minus_dice_roc_auc", "qa_assd_roc_auc", "qa_hd95_roc_auc",
        "qa_1_minus_dice_spearman", "qa_assd_spearman", "qa_hd95_spearman",
    ]
    show = summary[[c for c in show_cols if c in summary.columns]].copy()
    report.append("```text")
    report.append(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    report.append("```")
    report.extend([
        "",
        "## Interpretation rule",
        "",
        "If low-QA folds also have systematically worse validation action AUC/AP or worse validation-loss/correction metrics, upstream optimization quality is a plausible contributor. If action discrimination is comparable across folds while QA varies strongly, split/generalization variability is the more likely explanation.",
        "",
        "`training_quality_vs_qa_correlations.csv` is exploratory only: five folds are far too few for inferential correlation testing.",
    ])
    (output / "training_quality_diagnostic_report.md").write_text("\n".join(report) + "\n")

    metadata = {
        "study_root": str(study_root),
        "roi": args.roi,
        "variant": args.variant,
        "folds": folds,
        "qa_method": args.method,
        "bad_percentile": args.bad_percentile,
        "classification_per_fold": str(cls_path),
        "regression_per_fold": str(reg_path),
        "correlation_warning": "Fold-level correlations are descriptive only because n_folds is typically 5.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(show.to_string(index=False))
    print(f"\nConfig mismatches: {len(unequal)}")
    if not unequal.empty:
        print(unequal[["config_key"] + [f"fold_{f}" for f in folds]].to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
