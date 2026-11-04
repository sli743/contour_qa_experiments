#!/usr/bin/env python3
"""Analyze whether deployment-style action-score summaries rank bad autoseg cases.

Inputs are produced by diagnose_supervised_correction_action_scores.py.
No model inference occurs here.

This is a diagnostic on a small validation cohort, not a trained/validated gating model.
Any gate threshold selected on these same cases is optimistic and must be tested on an
independent fold/test set before being interpreted as performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


TARGETS = [
    "dice_error",
    "needed_correction_fraction_local",
    "missing_fraction_local",
    "excess_fraction_local",
]


def safe_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return np.nan, np.nan
    r = spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)


def safe_auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    if y.size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def feature_columns(df):
    prefixes = ("action_", "add_", "remove_")
    excluded_suffixes = ("_voxels",)
    out = []
    for c in df.columns:
        if not c.startswith(prefixes):
            continue
        if c.endswith(excluded_suffixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def correlation_table(df, features):
    rows = []
    for f in features:
        for target in TARGETS:
            if target not in df:
                continue
            rho, p = safe_spearman(df[f], df[target])
            rows.append({"feature": f, "target": target, "spearman_rho": rho, "p_value": p, "n": len(df)})
    return pd.DataFrame(rows).sort_values(["target", "spearman_rho"], ascending=[True, False])


def bad_case_auc_table(df, features, dice_thresholds):
    rows = []
    for thr in dice_thresholds:
        y = (df["input_target_dice"] < thr).astype(int)
        for f in features:
            rows.append({
                "definition": f"input_dice<{thr:g}",
                "feature": f,
                "n": len(df),
                "n_bad": int(y.sum()),
                "n_good": int((1-y).sum()),
                "roc_auc": safe_auc(y, df[f]),
            })
    # A threshold-free severity definition that is useful for tiny cohorts.
    cutoff = float(df["dice_error"].quantile(0.75))
    y = (df["dice_error"] >= cutoff).astype(int)
    for f in features:
        rows.append({
            "definition": "worst_dice_error_quartile",
            "feature": f,
            "n": len(df),
            "n_bad": int(y.sum()),
            "n_good": int((1-y).sum()),
            "roc_auc": safe_auc(y, df[f]),
        })
    return pd.DataFrame(rows).sort_values(["definition", "roc_auc"], ascending=[True, False])


def selective_gate_sweep(gate_df, threshold_df, features, correction_threshold):
    """Exploratory same-validation gate sweep; deliberately reports optimism warning."""
    corr = threshold_df[np.isclose(threshold_df["threshold"], correction_threshold)].copy()
    m = gate_df.merge(
        corr[["patient_id", "dice_gain", "corrected_target_dice", "needed_correction_recall", "correction_precision"]],
        on="patient_id", how="inner",
    )
    rows = []
    for f in features:
        vals = np.sort(m[f].dropna().unique())
        if vals.size == 0:
            continue
        # Gate boundaries at observed values plus a select-none boundary.
        candidates = np.concatenate(([np.inf], vals[::-1]))
        for gate_thr in candidates:
            selected = m[f] >= gate_thr
            effective_gain = np.where(selected, m["dice_gain"], 0.0)
            rows.append({
                "feature": f,
                "correction_threshold": correction_threshold,
                "gate_threshold": gate_thr,
                "n_cases": len(m),
                "n_selected": int(selected.sum()),
                "selected_fraction": float(selected.mean()),
                "mean_effective_dice_gain": float(np.mean(effective_gain)),
                "median_effective_dice_gain": float(np.median(effective_gain)),
                "n_improved": int(np.sum(selected & (m["dice_gain"] > 0))),
                "n_harmed": int(np.sum(selected & (m["dice_gain"] < 0))),
                "worst_effective_dice_gain": float(np.min(effective_gain)),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic-dir", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dice-thresholds", default="0.90,0.95")
    ap.add_argument("--correction-thresholds", default="0.10,0.20")
    args = ap.parse_args()

    d = Path(args.diagnostic_dir)
    out = Path(args.output_dir) if args.output_dir else d / "patient_gating"
    out.mkdir(parents=True, exist_ok=True)

    gate_path = d / "patient_action_score_summary.csv"
    threshold_path = d / "threshold_sweep_by_patient.csv"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Missing {gate_path}. Re-run the action-score diagnostic with the patient-gating patch.")
    if not threshold_path.is_file():
        raise FileNotFoundError(threshold_path)

    gate = pd.read_csv(gate_path)
    threshold_df = pd.read_csv(threshold_path)
    features = feature_columns(gate)
    dice_thresholds = [float(x) for x in args.dice_thresholds.split(",") if x.strip()]
    correction_thresholds = [float(x) for x in args.correction_thresholds.split(",") if x.strip()]

    corr = correlation_table(gate, features)
    corr.to_csv(out / "gate_feature_severity_correlations.csv", index=False)

    auc = bad_case_auc_table(gate, features, dice_thresholds)
    auc.to_csv(out / "gate_feature_bad_case_auc.csv", index=False)

    sweeps = []
    for t in correction_thresholds:
        sweeps.append(selective_gate_sweep(gate, threshold_df, features, t))
    sweep = pd.concat(sweeps, ignore_index=True) if sweeps else pd.DataFrame()
    sweep.to_csv(out / "selective_correction_gate_sweep.csv", index=False)

    if not sweep.empty:
        best = (
            sweep.sort_values(
                ["mean_effective_dice_gain", "n_harmed", "selected_fraction"],
                ascending=[False, True, True],
            )
            .groupby(["correction_threshold", "feature"], as_index=False)
            .first()
        )
        best.to_csv(out / "selective_correction_gate_best_by_feature.csv", index=False)

    patient_cols = [
        "patient_id", "input_target_dice", "dice_error", "needed_correction_fraction_local",
        "missing_fraction_local", "excess_fraction_local",
    ]
    # Add a compact set of interpretable score summaries for manual review.
    for c in [
        "action_p99", "action_p99_5", "action_p99_9", "action_top100_mean",
        "remove_p99", "remove_p99_5", "remove_top100_mean",
        "add_p99", "add_p99_5", "add_top100_mean",
        "action_frac_ge_0p1", "remove_frac_ge_0p1", "add_frac_ge_0p1",
        "action_largest_component_frac_ge_0p1", "remove_largest_component_frac_ge_0p1",
    ]:
        if c in gate:
            patient_cols.append(c)
    gate[patient_cols].sort_values("input_target_dice").to_csv(out / "patient_gate_review.csv", index=False)

    print("\nTop severity correlations (Dice error):")
    print(corr[corr.target == "dice_error"].head(15).to_string(index=False))
    print("\nTop bad-case AUCs:")
    print(auc.groupby("definition", as_index=False).head(10).to_string(index=False))
    print(f"\nSaved patient-gating diagnostic to: {out}")
    print("NOTE: selective gate thresholds are optimized/evaluated on the same validation cases and are diagnostic only.")


if __name__ == "__main__":
    main()
