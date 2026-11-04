#!/usr/bin/env python3
"""Compare SpinalCord NSR-QA localization/reference-frame choices.

This diagnostic is intentionally separate from the primary comparison report.
It compares the already-trained models under three geometries:

* approved_reference      -- oracle localization
* autoseg_centered        -- historical deployable candidate-centered crop
* autoseg_xy_ct_z         -- candidate in-plane center + raw CT slice-axis midpoint

Canonical true-error labels are taken from the approved-reference masks so the
same patient label is used for every localization mode.  This prevents a change
in crop geometry from changing the definition of the ground-truth error during
this localization experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

from analysis.spinalcord_diagnostic_utils import (
    correction_decomposition,
    pair_metrics,
    read_binary_lps,
)

MODES = ("approved_reference", "autoseg_centered", "autoseg_xy_ct_z")
MODE_LABEL = {
    "approved_reference": "Approved reference (oracle)",
    "autoseg_centered": "Autoseg centered",
    "autoseg_xy_ct_z": "Autoseg X/Y + CT Z",
}
METRICS = {
    "one_minus_dice": ("true_one_minus_dice", "qa_one_minus_dice"),
    "assd": ("true_assd_mm", "qa_assd_mm"),
    "hd95": ("true_hd95_mm", "qa_hd95_mm"),
    "hd99": ("true_hd99_mm", "qa_hd99_mm"),
    "max_endpoint": ("true_max_abs_endpoint_error_mm", "qa_max_abs_endpoint_error_mm"),
}


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(score, method="average")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def percentile_higher(values: Iterable[float], q: float) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    try:
        return float(np.percentile(x, q, method="higher"))
    except TypeError:
        return float(np.percentile(x, q, interpolation="higher"))


def safe_fraction(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after) or before <= 0:
        return np.nan
    return float((before - after) / before)


def read_case(case_dir: Path) -> Dict[str, float]:
    approved, _gimg, spacing = read_binary_lps(case_dir / "approved.nii.gz")
    autoseg, _aimg, _ = read_binary_lps(case_dir / "autoseg_input.nii.gz")
    recon, _rimg, _ = read_binary_lps(case_dir / "nsrqa_reconstruction.nii.gz")
    true = pair_metrics(approved, autoseg, spacing)
    qa = pair_metrics(autoseg, recon, spacing)
    residual = pair_metrics(approved, recon, spacing)
    decomp = correction_decomposition(approved, autoseg, recon)
    row: Dict[str, float] = {}
    for k, v in true.items(): row[f"true_{k}"] = v
    for k, v in qa.items(): row[f"qa_{k}"] = v
    for k, v in residual.items(): row[f"residual_{k}"] = v
    row.update(decomp)
    for metric in ("one_minus_dice", "assd_mm", "hd95_mm", "hd99_mm", "max_abs_endpoint_error_mm"):
        row[f"{metric}_fraction_corrected"] = safe_fraction(true.get(metric, np.nan), residual.get(metric, np.nan))
    return row


def collect_actual(compare_root: Path, models: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for model in models:
        for mode in MODES:
            csv_path = compare_root / "actual_autoseg_results" / model / "metrics" / "SpinalCord" / f"{mode}_metrics.csv"
            if not csv_path.is_file():
                print(f"WARNING missing: {csv_path}")
                continue
            d = pd.read_csv(csv_path, dtype={"patient_id": str})
            for rec in d.to_dict("records"):
                case_dir = Path(str(rec["case_dir"]))
                needed = [case_dir / x for x in ("approved.nii.gz", "autoseg_input.nii.gz", "nsrqa_reconstruction.nii.gz")]
                if not all(p.is_file() for p in needed):
                    print(f"WARNING incomplete viewer case: {case_dir}")
                    continue
                row = {
                    "model": model,
                    "localization_mode": mode,
                    "patient_id": str(rec["patient_id"]),
                    "shape_model_split": str(rec["shape_model_split"]),
                    "case_dir": str(case_dir),
                }
                row.update(read_case(case_dir))
                rows.append(row)
    return pd.DataFrame(rows)


def collect_clean(compare_root: Path, models: List[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        for mode in MODES:
            p = compare_root / "clean_approved_reconstruction" / model / "metrics" / "SpinalCord" / f"{mode}_metrics.csv"
            if not p.is_file():
                continue
            d = pd.read_csv(p, dtype={"patient_id": str})
            d["model"] = model
            rows.extend(d.to_dict("records"))
    return pd.DataFrame(rows)


def canonical_truth(actual: pd.DataFrame) -> pd.DataFrame:
    d = actual[actual["localization_mode"] == "approved_reference"].copy()
    # True metrics do not depend on model; keep one row per patient/split.
    cols = ["patient_id", "shape_model_split"] + [v[0] for v in METRICS.values()]
    return d[cols].drop_duplicates(["patient_id", "shape_model_split"])


def performance_table(actual: pd.DataFrame, exclude_patients: set[str]) -> pd.DataFrame:
    truth = canonical_truth(actual)
    rows = []
    for model in sorted(actual["model"].unique()):
        for excluded in (False, True):
            canonical = truth.copy()
            if excluded and exclude_patients:
                canonical = canonical[~canonical["patient_id"].isin(exclude_patients)]
            dev = canonical[canonical["shape_model_split"].isin(["training1", "validation1"])]
            test_truth = canonical[canonical["shape_model_split"] == "testing1"]
            for metric, (true_col, qa_col) in METRICS.items():
                threshold = percentile_higher(dev[true_col], 75.0)
                for mode in MODES:
                    score = actual[
                        (actual["model"] == model)
                        & (actual["localization_mode"] == mode)
                        & (actual["shape_model_split"] == "testing1")
                    ][["patient_id", qa_col]].copy()
                    if excluded and exclude_patients:
                        score = score[~score["patient_id"].isin(exclude_patients)]
                    paired = test_truth[["patient_id", true_col]].merge(score, on="patient_id", how="inner")
                    y = (pd.to_numeric(paired[true_col], errors="coerce") >= threshold).astype(int).to_numpy()
                    q = pd.to_numeric(paired[qa_col], errors="coerce").to_numpy(float)
                    rows.append({
                        "model": model,
                        "exclude_manual_incomplete": excluded,
                        "localization_mode": mode,
                        "metric": metric,
                        "n": int(len(paired)),
                        "p75_threshold_from_approved_reference_dev": threshold,
                        "n_positive": int(y.sum()),
                        "spearman": safe_spearman(paired[true_col], paired[qa_col]),
                        "auc": auc_rank(y, q),
                    })
    return pd.DataFrame(rows)


def correction_summary(actual: pd.DataFrame, exclude_patients: set[str]) -> pd.DataFrame:
    metrics = [
        "missing_correction_recall", "addition_precision",
        "excess_correction_recall", "removal_precision",
        "overall_error_correction_recall", "change_precision",
        "max_abs_endpoint_error_mm_fraction_corrected",
        "hd99_mm_fraction_corrected", "one_minus_dice_fraction_corrected",
    ]
    rows = []
    for excluded in (False, True):
        d = actual[actual["shape_model_split"] == "testing1"].copy()
        if excluded and exclude_patients:
            d = d[~d["patient_id"].isin(exclude_patients)]
        for (model, mode), g in d.groupby(["model", "localization_mode"]):
            row = {"model": model, "localization_mode": mode, "exclude_manual_incomplete": excluded, "n": len(g)}
            for m in metrics:
                vals = pd.to_numeric(g[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                row[f"median_{m}"] = float(vals.median()) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def clean_summary(clean: pd.DataFrame, exclude_patients: set[str]) -> pd.DataFrame:
    if clean.empty:
        return clean
    metrics = ["clean_1_minus_dice", "clean_hd99_mm", "clean_max_abs_endpoint_error_mm"]
    rows = []
    for excluded in (False, True):
        d = clean[clean["shape_model_split"] == "testing1"].copy()
        if excluded and exclude_patients:
            d = d[~d["patient_id"].astype(str).isin(exclude_patients)]
        for (model, mode), g in d.groupby(["model", "localization_mode"]):
            row = {"model": model, "localization_mode": mode, "exclude_manual_incomplete": excluded, "n": len(g)}
            for m in metrics:
                vals = pd.to_numeric(g[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                row[f"median_{m}"] = float(vals.median()) if len(vals) else np.nan
                row[f"p90_{m}"] = float(vals.quantile(0.9)) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def patient_localization_table(actual: pd.DataFrame, model: str) -> pd.DataFrame:
    d = actual[(actual["model"] == model) & (actual["shape_model_split"] == "testing1")].copy()
    keep = [
        "patient_id", "localization_mode", "true_max_abs_endpoint_error_mm",
        "residual_max_abs_endpoint_error_mm", "max_abs_endpoint_error_mm_fraction_corrected",
        "missing_correction_recall", "excess_correction_recall", "change_precision",
        "qa_hd99_mm", "qa_one_minus_dice",
    ]
    d = d[keep]
    wide = d.pivot(index="patient_id", columns="localization_mode")
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    return wide.reset_index()


def md_table(df: pd.DataFrame, cols: List[str], digits: int = 4) -> str:
    if df.empty: return "_No rows._"
    x = df[cols].copy()
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in x.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *body])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models", nargs="+", default=["previous_generation", "v4_corruption"])
    p.add_argument("--exclude-patients", nargs="*", default=[])
    args = p.parse_args()

    root = Path(args.compare_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    excluded = {str(x) for x in args.exclude_patients}

    actual = collect_actual(root, args.models)
    if actual.empty:
        raise RuntimeError("No localization-result cases were found")
    clean = collect_clean(root, args.models)

    actual.to_csv(out / "case_metrics.csv", index=False)
    perf = performance_table(actual, excluded); perf.to_csv(out / "qa_performance_by_reference_frame.csv", index=False)
    corr = correction_summary(actual, excluded); corr.to_csv(out / "correction_by_reference_frame.csv", index=False)
    clean_s = clean_summary(clean, excluded); clean_s.to_csv(out / "clean_identity_by_reference_frame.csv", index=False)
    patient = patient_localization_table(actual, "v4_corruption"); patient.to_csv(out / "v4_testing1_patient_localization_comparison.csv", index=False)

    focus = perf[(perf["model"] == "v4_corruption") & (perf["exclude_manual_incomplete"] == bool(excluded))]
    lines = [
        "# SpinalCord reference-frame diagnostic",
        "",
        "This diagnostic changes only the evaluation crop reference frame. The neural-network checkpoints are unchanged.",
        "Canonical true-error values and P75 thresholds come from `approved_reference` development masks so every localization mode is judged against the same patient-level truth definition.",
        "",
        f"Manual acquisition-limited exclusions for the filtered view: `{', '.join(sorted(excluded, key=lambda x: int(x) if x.isdigit() else x)) or 'none'}`.",
        "",
        "## V4 Month-3 QA performance",
        "",
        md_table(focus, ["metric", "localization_mode", "n", "n_positive", "spearman", "auc"]),
        "",
        "## Month-3 correction behavior",
        "",
    ]
    cfocus = corr[(corr["model"] == "v4_corruption") & (corr["exclude_manual_incomplete"] == bool(excluded))]
    lines += [md_table(cfocus, [
        "localization_mode", "n", "median_missing_correction_recall",
        "median_excess_correction_recall", "median_change_precision",
        "median_max_abs_endpoint_error_mm_fraction_corrected", "median_hd99_mm_fraction_corrected",
    ]), ""]
    if not clean_s.empty:
        lines += ["## Clean approved-contour identity", ""]
        q = clean_s[(clean_s["model"] == "v4_corruption") & (clean_s["exclude_manual_incomplete"] == bool(excluded))]
        lines += [md_table(q, [
            "localization_mode", "n", "median_clean_1_minus_dice", "median_clean_hd99_mm",
            "median_clean_max_abs_endpoint_error_mm", "p90_clean_max_abs_endpoint_error_mm",
        ]), ""]
    lines += [
        "## Interpretation",
        "",
        "`autoseg_xy_ct_z` keeps candidate-derived in-plane localization but replaces the candidate-derived slice-axis center with the raw CT acquisition midpoint. Improvement over `autoseg_centered` therefore supports the hypothesis that longitudinal recentering of the candidate removes useful endpoint-position information. Failure to improve suggests that CT midpoint alone is not an adequate anatomical reference frame.",
        "",
        "This is a diagnostic evaluation of already-trained models. If the hybrid frame is beneficial, the next controlled training experiment should prepare training data in the same frame rather than relying on this evaluation-only geometry shift.",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({
        "compare_root": str(root),
        "models": args.models,
        "modes": list(MODES),
        "manual_excluded_patients": sorted(excluded),
        "canonical_truth_mode": "approved_reference",
        "hybrid_mode_definition": "candidate in-plane center + raw CT acquisition midpoint along CT slice axis",
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
