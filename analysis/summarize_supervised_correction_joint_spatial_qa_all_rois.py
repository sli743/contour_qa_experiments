#!/usr/bin/env python3
"""Summarize the fixed joint-spatial supervised-correction QA method across ROIs.

The input for every ROI is its five-outer-fold summary produced by
``summarize_supervised_correction_best_qa_folds`` with only
``joint_spatial_action_head`` selected. Patient observations are never pooled across
organs because the same patient can contribute multiple ROIs and the true-error
scale differs by anatomy. Cross-ROI summaries therefore macro-average ROI-level
performance statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

METHOD = "joint_spatial_action_head"
TARGETS = ("1_minus_dice", "hd95", "assd")


def read_active_rois(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    required = {"roi", "data_group", "profile"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def macro_summary(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_rois"] = int(g.roi.nunique())
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            row[f"{col}_median"] = float(np.median(vals)) if len(vals) else float("nan")
            row[f"{col}_q25"] = float(np.percentile(vals, 25)) if len(vals) else float("nan")
            row[f"{col}_q75"] = float(np.percentile(vals, 75)) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", default="/scratch/shaohan/contour_QA/supervised_autoseg_correction")
    ap.add_argument("--all-roi-root", default="/scratch/shaohan/contour_QA/all_roi_NSRQA_study")
    ap.add_argument("--variant", default="large_decoder_ft_last2")
    ap.add_argument("--active-rois", default=None)
    ap.add_argument("--rois", default=None, help="Optional comma-separated ROI subset; default all active ROIs.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--output-zip", default=None)
    args = ap.parse_args()

    study_root = Path(args.study_root)
    active_path = Path(args.active_rois) if args.active_rois else Path(args.all_roi_root) / "manifests" / "active_rois.tsv"
    active = read_active_rois(active_path)
    if args.rois:
        requested = [x.strip() for x in args.rois.split(",") if x.strip()]
        unknown = [r for r in requested if r not in set(active.roi)]
        if unknown:
            raise ValueError(f"Requested ROIs are not active: {unknown}")
        active = active[active.roi.isin(requested)].copy()
        active["_order"] = active.roi.map({r: i for i, r in enumerate(requested)})
        active = active.sort_values("_order").drop(columns="_order")
    if active.empty:
        raise RuntimeError("No active ROIs selected")

    out_dir = Path(args.output_dir) if args.output_dir else study_root / "joint_spatial_qa_all_rois" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_parts, reg_parts, fold_cls_parts, fold_reg_parts = [], [], [], []
    missing = []
    for meta in active.to_dict("records"):
        roi = str(meta["roi"])
        roi_dir = study_root / "joint_spatial_qa_5fold" / roi / args.variant
        cls_path = roi_dir / "best_qa_crossfold_classification_summary.csv"
        reg_path = roi_dir / "best_qa_crossfold_regression_summary.csv"
        fcls_path = roi_dir / "best_qa_crossfold_classification_per_fold.csv"
        freg_path = roi_dir / "best_qa_crossfold_regression_per_fold.csv"
        needed = [cls_path, reg_path, fcls_path, freg_path]
        absent = [str(p) for p in needed if not p.is_file()]
        if absent:
            missing.append({"roi": roi, "missing": absent})
            continue

        cls = pd.read_csv(cls_path)
        reg = pd.read_csv(reg_path)
        fcls = pd.read_csv(fcls_path)
        freg = pd.read_csv(freg_path)
        cls = cls[cls.method == METHOD].copy()
        reg = reg[reg.method == METHOD].copy()
        fcls = fcls[fcls.method == METHOD].copy()
        freg = freg[freg.method == METHOD].copy()
        if len(cls) != len(TARGETS) * 2 or len(reg) != len(TARGETS):
            raise RuntimeError(
                f"Unexpected joint-spatial summary shape for {roi}: classification={len(cls)} regression={len(reg)}"
            )
        for df in (cls, reg, fcls, freg):
            df["data_group"] = meta["data_group"]
            df["profile"] = meta["profile"]
            if "global_bad_volume_pct" in meta:
                df["registry_global_bad_volume_pct"] = meta["global_bad_volume_pct"]
            if "local_bad_hd95_mm" in meta:
                df["registry_local_bad_hd95_mm"] = meta["local_bad_hd95_mm"]
        cls_parts.append(cls)
        reg_parts.append(reg)
        fold_cls_parts.append(fcls)
        fold_reg_parts.append(freg)

    if missing:
        lines = [f"{x['roi']}: {', '.join(x['missing'])}" for x in missing]
        raise FileNotFoundError("Missing per-ROI five-fold summaries:\n" + "\n".join(lines))

    cls_all = pd.concat(cls_parts, ignore_index=True)
    reg_all = pd.concat(reg_parts, ignore_index=True)
    fold_cls_all = pd.concat(fold_cls_parts, ignore_index=True)
    fold_reg_all = pd.concat(fold_reg_parts, ignore_index=True)

    # Primary cross-ROI classification endpoint is P75. P50 is retained in by-ROI tables.
    cls_p75 = cls_all[cls_all.bad_percentile == 75].copy()
    cls_metrics = [
        "fold_roc_auc_mean", "fold_pr_auc_mean", "pooled_roc_auc", "pooled_pr_auc",
        "pooled_sensitivity", "pooled_specificity", "pooled_ppv", "pooled_npv",
    ]
    reg_metrics = [
        "fold_spearman_mean", "fold_pearson_mean", "fold_r2_mean", "fold_mae_mean",
        "pooled_spearman", "pooled_pearson", "pooled_r2", "pooled_mae",
    ]
    cls_metrics = [c for c in cls_metrics if c in cls_p75.columns]
    reg_metrics = [c for c in reg_metrics if c in reg_all.columns]

    overall_cls = macro_summary(cls_p75, ["target_metric"], cls_metrics)
    overall_reg = macro_summary(reg_all, ["target_metric"], reg_metrics)
    profile_cls = macro_summary(cls_p75, ["profile", "target_metric"], cls_metrics)
    profile_reg = macro_summary(reg_all, ["profile", "target_metric"], reg_metrics)

    # A compact table aimed at the main scientific question: how does QA performance vary by anatomy?
    compact = cls_p75[[
        "roi", "data_group", "profile", "target_metric", "fold_roc_auc_mean", "fold_roc_auc_sd",
        "pooled_roc_auc", "pooled_pr_auc", "pooled_sensitivity", "pooled_specificity",
    ]].merge(
        reg_all[[
            "roi", "target_metric", "fold_spearman_mean", "fold_spearman_sd",
            "pooled_spearman", "pooled_r2", "pooled_mae",
        ]], on=["roi", "target_metric"], how="outer"
    )

    cls_all.to_csv(out_dir / "joint_spatial_classification_by_roi.csv", index=False)
    reg_all.to_csv(out_dir / "joint_spatial_regression_by_roi.csv", index=False)
    fold_cls_all.to_csv(out_dir / "joint_spatial_classification_per_fold_all_rois.csv", index=False)
    fold_reg_all.to_csv(out_dir / "joint_spatial_regression_per_fold_all_rois.csv", index=False)
    compact.to_csv(out_dir / "joint_spatial_roi_metric_summary.csv", index=False)
    overall_cls.to_csv(out_dir / "joint_spatial_classification_macro_across_rois.csv", index=False)
    overall_reg.to_csv(out_dir / "joint_spatial_regression_macro_across_rois.csv", index=False)
    profile_cls.to_csv(out_dir / "joint_spatial_classification_by_anatomy_profile.csv", index=False)
    profile_reg.to_csv(out_dir / "joint_spatial_regression_by_anatomy_profile.csv", index=False)

    protocol = {
        "method": METHOD,
        "correction_variant": args.variant,
        "active_roi_manifest": str(active_path),
        "rois": active.roi.tolist(),
        "n_rois": int(len(active)),
        "outer_folds_per_roi": [0, 1, 2, 3, 4],
        "classification_primary_bad_percentile": 75,
        "fixed_method_definition": {
            "decoder": "large decoder [512,256,128,64,32], two residual blocks/stage",
            "encoder_adaptation": "last two 3DINO transformer blocks fine-tuned",
            "qa_readout": "joint 3-target spatial action head",
            "spatial_input": "12-channel local/boundary x avg/max x P(action/add/remove), adaptive pooled to 32^3",
            "local_radius_voxels": 6,
            "boundary_radius_voxels": 3,
        },
        "cross_roi_note": (
            "Patients are pooled only across the five mutually exclusive outer-test folds within each ROI. "
            "Patient predictions are not pooled across different ROIs. Cross-ROI and profile summaries macro-average ROI-level metrics."
        ),
    }
    (out_dir / "joint_spatial_all_roi_protocol.json").write_text(json.dumps(protocol, indent=2))

    readme = f"""# All-ROI five-fold joint-spatial QA sweep\n\nFixed correction representation: `{args.variant}`  \nFixed QA readout: `{METHOD}`  \nROIs: **{len(active)}**\n\nEvery ROI uses the same five-outer-fold patient split protocol and the same method hyperparameters. The all-ROI tables do **not** pool patient observations across organs; they first compute held-out performance within each ROI, then macro-summarize ROI-level metrics.\n\nPrimary binary endpoint: development-derived P75 bad-contour definition. P50 results remain available in `joint_spatial_classification_by_roi.csv`.\n"""
    (out_dir / "README.md").write_text(readme)

    zip_path = Path(args.output_zip) if args.output_zip else out_dir.parent / f"{args.variant}_joint_spatial_all_roi_results_no_models.zip"
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out_dir)))
    print(f"Wrote all-ROI summary: {out_dir}")
    print(f"Wrote uploadable ZIP: {zip_path}")


if __name__ == "__main__":
    main()
