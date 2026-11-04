#!/usr/bin/env python3
"""Summarize the SpinalCord deployment-matched reference-frame ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

METRICS = {
    "1_minus_dice": ("true_1_minus_dice", "qa_1_minus_dice"),
    "assd": ("true_assd_mm", "qa_assd_mm"),
    "hd95": ("true_hd95_mm", "qa_hd95_mm"),
    "hd99": ("true_hd99_mm", "qa_hd99_mm"),
    "max_endpoint": ("true_max_abs_endpoint_error_mm", "qa_max_abs_endpoint_error_mm"),
}

CORRECTION_COLUMNS = [
    "missing_correction_recall", "addition_precision",
    "excess_correction_recall", "removal_precision",
    "overall_error_correction_recall", "change_precision",
    "overcorrection_fraction_of_change", "voxel_error_reduction_fraction",
]

CLEAN_COLUMNS = [
    "clean_1_minus_dice", "clean_assd_mm", "clean_hd95_mm", "clean_hd99_mm",
    "clean_max_abs_endpoint_error_mm",
]

SENTINELS = ["91", "49", "110", "135", "160", "100", "136", "152", "106", "95", "129", "145"]


def _finite_pair(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    out = df.copy()
    out[a] = pd.to_numeric(out[a], errors="coerce")
    out[b] = pd.to_numeric(out[b], errors="coerce")
    return out[np.isfinite(out[a]) & np.isfinite(out[b])].copy()


def auc_stats(dev: pd.DataFrame, test: pd.DataFrame, true_col: str, qa_col: str) -> dict:
    d = _finite_pair(dev, true_col, qa_col)
    t = _finite_pair(test, true_col, qa_col)
    if d.empty or t.empty:
        return {"cutoff": np.nan, "n": len(t), "n_positive": 0, "auroc": np.nan, "auprc": np.nan, "spearman": np.nan}
    cutoff = float(d[true_col].quantile(0.75))
    y = (t[true_col].to_numpy(float) > cutoff).astype(int)
    score = t[qa_col].to_numpy(float)
    auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
    ap = average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan
    rho = spearmanr(t[true_col].to_numpy(float), score).statistic if len(t) >= 3 else np.nan
    return {
        "cutoff": cutoff, "n": len(t), "n_positive": int(y.sum()),
        "auroc": float(auc) if np.isfinite(auc) else np.nan,
        "auprc": float(ap) if np.isfinite(ap) else np.nan,
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def paired_bootstrap_auc(base: pd.DataFrame, other: pd.DataFrame, true_col: str, qa_col: str, cutoff: float, n_boot: int, seed: int) -> dict:
    cols = ["patient_id", true_col, qa_col]
    a = base[cols].rename(columns={qa_col: "score_base", true_col: "truth"})
    b = other[["patient_id", qa_col]].rename(columns={qa_col: "score_other"})
    m = a.merge(b, on="patient_id", how="inner")
    m = m[np.isfinite(pd.to_numeric(m["truth"], errors="coerce")) & np.isfinite(pd.to_numeric(m["score_base"], errors="coerce")) & np.isfinite(pd.to_numeric(m["score_other"], errors="coerce"))].copy()
    if len(m) < 4:
        return {"n_paired": len(m), "auc_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    y = (m["truth"].to_numpy(float) > cutoff).astype(int)
    sb = m["score_base"].to_numpy(float)
    so = m["score_other"].to_numpy(float)
    if len(np.unique(y)) < 2:
        return {"n_paired": len(m), "auc_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    observed = float(roc_auc_score(y, so) - roc_auc_score(y, sb))
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(m)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(float(roc_auc_score(yy, so[idx]) - roc_auc_score(yy, sb[idx])))
    if not diffs:
        return {"n_paired": len(m), "auc_diff": observed, "ci_low": np.nan, "ci_high": np.nan}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_paired": len(m), "auc_diff": observed, "ci_low": float(lo), "ci_high": float(hi)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--exclude-patients", default="95,129,145", help="Manually confirmed acquisition-limited IDs; empty string disables filtering.")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=88)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(root / "manifests" / "model_configs.tsv", sep="\t")
    excluded = {x.strip() for x in str(args.exclude_patients).split(",") if x.strip()}

    actual = {}
    clean = {}
    for _, mr in manifest.iterrows():
        arm = str(mr["arm"])
        actual[arm] = {}
        clean[arm] = {}
        for mode in ("autoseg_centered", "approved_reference"):
            ap = root / "actual_autoseg_results" / arm / "metrics" / "SpinalCord" / f"{mode}_metrics.csv"
            cp = root / "clean_reconstruction" / arm / "metrics" / "SpinalCord" / f"{mode}_clean.csv"
            if not ap.is_file(): raise FileNotFoundError(ap)
            if not cp.is_file(): raise FileNotFoundError(cp)
            adf = pd.read_csv(ap, dtype={"patient_id": str})
            cdf = pd.read_csv(cp, dtype={"patient_id": str})
            adf["patient_id"] = adf["patient_id"].astype(str)
            cdf["patient_id"] = cdf["patient_id"].astype(str)
            actual[arm][mode] = adf
            clean[arm][mode] = cdf

    auc_rows = []
    for filtered in (False, True):
        for mode in ("autoseg_centered", "approved_reference"):
            for arm, modes in actual.items():
                df = modes[mode].copy()
                if filtered: df = df[~df["patient_id"].isin(excluded)]
                dev = df[df["shape_model_split"].isin(["training1", "validation1"])]
                test = df[df["shape_model_split"] == "testing1"]
                for metric, (true_col, qa_col) in METRICS.items():
                    st = auc_stats(dev, test, true_col, qa_col)
                    auc_rows.append({"arm": arm, "localization_mode": mode, "exclude_acquisition_limited": filtered, "metric": metric, **st})
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(out / "qa_auc_summary.csv", index=False)

    correction_rows = []
    clean_rows = []
    for filtered in (False, True):
        for arm, modes in actual.items():
            for mode, df0 in modes.items():
                df = df0[df0["shape_model_split"] == "testing1"].copy()
                if filtered: df = df[~df["patient_id"].isin(excluded)]
                row = {"arm": arm, "localization_mode": mode, "exclude_acquisition_limited": filtered, "n": len(df)}
                for c in CORRECTION_COLUMNS:
                    vals = pd.to_numeric(df.get(c), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    row[f"median_{c}"] = float(vals.median()) if len(vals) else np.nan
                correction_rows.append(row)

                cdf = clean[arm][mode]
                cdf = cdf[cdf["shape_model_split"] == "testing1"].copy()
                if filtered: cdf = cdf[~cdf["patient_id"].isin(excluded)]
                crow = {"arm": arm, "localization_mode": mode, "exclude_acquisition_limited": filtered, "n": len(cdf)}
                for c in CLEAN_COLUMNS:
                    vals = pd.to_numeric(cdf.get(c), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    crow[f"median_{c}"] = float(vals.median()) if len(vals) else np.nan
                    crow[f"p90_{c}"] = float(vals.quantile(.90)) if len(vals) else np.nan
                clean_rows.append(crow)
    pd.DataFrame(correction_rows).to_csv(out / "correction_summary.csv", index=False)
    pd.DataFrame(clean_rows).to_csv(out / "clean_identity_summary.csv", index=False)

    # Paired AUROC differences against Arm A in the deployable localization.
    boot_rows = []
    base_arm = "centered"
    for filtered in (False, True):
        base_all = actual[base_arm]["autoseg_centered"]
        if filtered: base_all = base_all[~base_all["patient_id"].isin(excluded)]
        base_dev = base_all[base_all["shape_model_split"].isin(["training1", "validation1"])]
        base_test = base_all[base_all["shape_model_split"] == "testing1"]
        for metric, (true_col, qa_col) in METRICS.items():
            cutoff = float(pd.to_numeric(base_dev[true_col], errors="coerce").dropna().quantile(.75))
            for arm in actual:
                if arm == base_arm: continue
                other = actual[arm]["autoseg_centered"]
                if filtered: other = other[~other["patient_id"].isin(excluded)]
                other = other[other["shape_model_split"] == "testing1"]
                st = paired_bootstrap_auc(base_test, other, true_col, qa_col, cutoff, args.n_boot, args.seed + sum(map(ord, arm + metric)))
                boot_rows.append({"baseline_arm": base_arm, "arm": arm, "exclude_acquisition_limited": filtered, "metric": metric, "cutoff": cutoff, **st})
    pd.DataFrame(boot_rows).to_csv(out / "paired_auc_bootstrap_vs_centered.csv", index=False)

    # Patient-level sentinel table for visual/mechanistic follow-up.
    sent_rows = []
    for arm, modes in actual.items():
        df = modes["autoseg_centered"]
        df = df[df["patient_id"].isin(SENTINELS)].copy()
        for _, r in df.iterrows():
            sent_rows.append({
                "patient_id": str(r["patient_id"]), "arm": arm,
                "true_1_minus_dice": r.get("true_1_minus_dice", np.nan),
                "true_hd99_mm": r.get("true_hd99_mm", np.nan),
                "true_max_endpoint_mm": r.get("true_max_abs_endpoint_error_mm", np.nan),
                "residual_1_minus_dice": r.get("residual_1_minus_dice", np.nan),
                "residual_hd99_mm": r.get("residual_hd99_mm", np.nan),
                "residual_max_endpoint_mm": r.get("residual_max_abs_endpoint_error_mm", np.nan),
                "missing_correction_recall": r.get("missing_correction_recall", np.nan),
                "excess_correction_recall": r.get("excess_correction_recall", np.nan),
                "change_precision": r.get("change_precision", np.nan),
                "voxel_error_reduction_fraction": r.get("voxel_error_reduction_fraction", np.nan),
                "acquisition_limited_manual": str(r["patient_id"]) in excluded,
            })
    sent_df = pd.DataFrame(sent_rows)
    if not sent_df.empty:
        sent_df["patient_sort"] = sent_df["patient_id"].map(lambda x: SENTINELS.index(str(x)) if str(x) in SENTINELS else 999)
        sent_df = sent_df.sort_values(["patient_sort", "arm"]).drop(columns="patient_sort")
    sent_df.to_csv(out / "sentinel_patient_comparison.csv", index=False)

    filtered_auc = auc_df[(auc_df["exclude_acquisition_limited"]) & (auc_df["localization_mode"] == "autoseg_centered")]
    lines = [
        "# SpinalCord reference-conditioning ablation", "",
        "Primary condition: actual Month3 autoseg, `autoseg_centered` localization.",
        f"Manual acquisition-limited sensitivity exclusion: {', '.join(sorted(excluded)) if excluded else 'none'}.",
        "", "## Filtered Month3 P75 AUROC", "",
        "| Metric | centered | +context | +jitter | +context+jitter |",
        "|---|---:|---:|---:|---:|",
    ]
    arm_order = ["centered", "centered_context", "centered_jitter", "centered_context_jitter"]
    for metric in METRICS:
        vals = []
        for arm in arm_order:
            q = filtered_auc[(filtered_auc["metric"] == metric) & (filtered_auc["arm"] == arm)]
            vals.append(float(q.iloc[0]["auroc"]) if len(q) else np.nan)
        lines.append(f"| {metric} | " + " | ".join("NA" if not np.isfinite(v) else f"{v:.6f}" for v in vals) + " |")
    lines += [
        "", "## Interpretation guide", "",
        "- `centered` tests whether matching training to candidate-centered deployment is sufficient.",
        "- `centered_context - centered` isolates explicit global spatial conditioning.",
        "- `centered_jitter - centered` isolates crop-jitter robustness.",
        "- `centered_context_jitter` is the proposed combined model.",
        "- Do not accept an AUC gain if clean identity or change precision degrades materially.",
        "- Patient 91 is the main reference-frame sentinel; patients 106/152 are clean overcorrection sentinels.",
        "", "See CSV outputs for paired bootstrap CIs, correction decomposition, clean identity, and sentinel cases.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({
        "root": str(root), "excluded_patients": sorted(excluded), "n_boot": args.n_boot,
        "primary_localization": "autoseg_centered", "p75_cutoff_source": "training1+validation1 only",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reference-conditioning ablation report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
