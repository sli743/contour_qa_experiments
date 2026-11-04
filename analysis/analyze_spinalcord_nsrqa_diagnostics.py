#!/usr/bin/env python3
"""Four-level diagnostic analysis for the SpinalCord NSR-QA comparison.

Levels
------
1. Clean approved-contour identity reconstruction.
2. Correction of the deterministic shared synthetic probe bank.
3. Correction decomposition on actual autosegmentations.
4. Final QA-signal performance against true actual-autoseg error.

The script also joins the all-month acquisition audit so metrics can be shown
with and without confirmed/likely incomplete acquisitions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.spinalcord_diagnostic_utils import (
    correction_decomposition,
    endpoint_metrics,
    pair_metrics,
    prefixed,
    read_binary_lps,
)

MODELS = ("previous_generation", "v4_corruption")
MODES = ("autoseg_centered", "approved_reference")
SPLITS = ("training1", "validation1", "testing1")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den != 0 else np.nan


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    xx = pd.to_numeric(pd.Series(x), errors="coerce")
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    m = xx.notna() & yy.notna() & np.isfinite(xx) & np.isfinite(yy)
    if int(m.sum()) < 3 or xx[m].nunique() < 2 or yy[m].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(xx[m], yy[m]).statistic)


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


def percentile_higher(x: Iterable[float], q: float) -> float:
    vals = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan
    try:
        return float(np.percentile(vals, q, method="higher"))
    except TypeError:
        return float(np.percentile(vals, q, interpolation="higher"))


def acquisition_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["patient_id", "acquisition_review_status", "exclude_incomplete_acquisition"])
    d = pd.read_csv(path, dtype={"patient_id": str})
    d["patient_id"] = d["patient_id"].astype(str)
    status = d.get("acquisition_review_status", pd.Series("no_flag", index=d.index)).astype(str)
    d["exclude_incomplete_acquisition"] = status.isin(["confirmed_incomplete", "likely_incomplete"])
    keep = [c for c in [
        "patient_id", "month_membership", "acquisition_review_status", "exclude_incomplete_acquisition",
        "ct_si_extent_mm", "approved_si_span_mm", "approved_inferior_margin_to_ct_edge_mm",
        "approved_superior_margin_to_ct_edge_mm",
    ] if c in d.columns]
    return d[keep].drop_duplicates("patient_id")


def read_actual_cases(compare_root: Path, acq: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    acq_map = acq.set_index("patient_id").to_dict("index") if len(acq) else {}
    for model in MODELS:
        for mode in MODES:
            csv_path = compare_root / "actual_autoseg_results" / model / "metrics" / "SpinalCord" / f"{mode}_metrics.csv"
            if not csv_path.is_file():
                print(f"WARNING missing actual-autoseg CSV: {csv_path}")
                continue
            d = pd.read_csv(csv_path, dtype={"patient_id": str})
            for r in d.itertuples(index=False):
                case = Path(str(r.case_dir))
                app_p = case / "approved.nii.gz"; auto_p = case / "autoseg_input.nii.gz"; rec_p = case / "nsrqa_reconstruction.nii.gz"
                if not (app_p.is_file() and auto_p.is_file() and rec_p.is_file()):
                    print(f"WARNING missing case masks: {case}")
                    continue
                g, _gi, spacing = read_binary_lps(app_p)
                a, _ai, _ = read_binary_lps(auto_p)
                rec, _ri, _ = read_binary_lps(rec_p)
                true = pair_metrics(g, a, spacing)
                residual = pair_metrics(g, rec, spacing)
                qa = pair_metrics(a, rec, spacing)
                decomp = correction_decomposition(g, a, rec)
                row: Dict = {
                    "model": model,
                    "localization_mode": mode,
                    "shape_model_split": str(r.shape_model_split),
                    "patient_id": str(r.patient_id),
                    "case_dir": str(case),
                    **prefixed(true, "true_"),
                    **prefixed(residual, "residual_"),
                    **prefixed(qa, "qa_"),
                    **decomp,
                }
                for metric in ("one_minus_dice", "assd_mm", "hd95_mm", "hd99_mm", "max_hd_mm", "max_abs_endpoint_error_mm"):
                    before = float(true.get(metric, np.nan)); after = float(residual.get(metric, np.nan))
                    row[f"{metric}_reduction"] = before - after if np.isfinite(before) and np.isfinite(after) else np.nan
                    row[f"{metric}_fraction_corrected"] = safe_div(before - after, before)
                row.update(acq_map.get(str(r.patient_id), {}))
                row.setdefault("acquisition_review_status", "no_flag")
                row.setdefault("exclude_incomplete_acquisition", False)
                rows.append(row)
    return pd.DataFrame(rows)


def clean_cases(compare_root: Path, acq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    acq_map = acq.set_index("patient_id").to_dict("index") if len(acq) else {}
    for model in MODELS:
        for mode in MODES:
            p = compare_root / "clean_approved_reconstruction" / model / "metrics" / "SpinalCord" / f"{mode}_metrics.csv"
            if not p.is_file():
                print(f"WARNING missing clean-reconstruction CSV: {p}")
                continue
            d = pd.read_csv(p, dtype={"patient_id": str})
            d["model"] = model
            for rec in d.to_dict("records"):
                rec.update(acq_map.get(str(rec["patient_id"]), {}))
                rec.setdefault("acquisition_review_status", "no_flag")
                rec.setdefault("exclude_incomplete_acquisition", False)
                rows.append(rec)
    return pd.DataFrame(rows)


def summarize_clean(d: pd.DataFrame) -> pd.DataFrame:
    if not len(d): return pd.DataFrame()
    metrics = ["clean_1_minus_dice", "clean_assd_mm", "clean_hd95_mm", "clean_hd99_mm", "clean_max_abs_endpoint_error_mm"]
    rows = []
    for filtered in (False, True):
        dd = d[~d["exclude_incomplete_acquisition"].fillna(False).astype(bool)] if filtered else d
        for (model, mode, split), g in dd.groupby(["model", "localization_mode", "shape_model_split"]):
            row = {"model": model, "localization_mode": mode, "shape_model_split": split, "exclude_incomplete": filtered, "n": len(g)}
            for m in metrics:
                vals = pd.to_numeric(g[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                row[f"median_{m}"] = float(vals.median()) if len(vals) else np.nan
                row[f"p90_{m}"] = float(vals.quantile(0.9)) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def probe_token(level) -> str:
    token = str(level).rstrip("0").rstrip(".").replace(".", "p")
    return token if token else "0"


def analyze_probe_csv(compare_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model in MODELS:
        p = compare_root / "synthetic_probe" / model / "probe_eval_sweep_detailed.csv"
        if not p.is_file(): continue
        d = pd.read_csv(p, dtype={"patient_id": str})
        d["model"] = model
        d["dice_error_input"] = 1.0 - pd.to_numeric(d["dice_input_vs_target"], errors="coerce")
        d["dice_error_recon"] = 1.0 - pd.to_numeric(d["dice_recon_vs_target"], errors="coerce")
        d["dice_error_reduction"] = d["dice_error_input"] - d["dice_error_recon"]
        d["dice_error_fraction_corrected"] = d["dice_error_reduction"] / d["dice_error_input"].replace(0, np.nan)
        for metric in ("hd95", "assd"):
            before = pd.to_numeric(d[f"{metric}_input_vs_target"], errors="coerce")
            after = pd.to_numeric(d[f"{metric}_recon_vs_target"], errors="coerce")
            d[f"{metric}_error_reduction"] = before - after
            d[f"{metric}_error_fraction_corrected"] = (before - after) / before.replace(0, np.nan)
        rows.append(d)
    detailed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not len(detailed): return detailed, pd.DataFrame()
    agg = detailed.groupby(["model", "probe_type", "level"], dropna=False).agg(
        n=("patient_id", "size"),
        median_dice_error_input=("dice_error_input", "median"),
        median_dice_error_recon=("dice_error_recon", "median"),
        median_dice_fraction_corrected=("dice_error_fraction_corrected", "median"),
        median_hd95_fraction_corrected=("hd95_error_fraction_corrected", "median"),
        median_assd_fraction_corrected=("assd_error_fraction_corrected", "median"),
    ).reset_index()
    return detailed, agg


def deep_probe_geometry(compare_root: Path, bank_manifest: Path) -> pd.DataFrame:
    if not bank_manifest.is_file():
        print(f"WARNING probe manifest missing: {bank_manifest}")
        return pd.DataFrame()
    manifest = pd.read_csv(bank_manifest, dtype={"patient_id": str})
    target_cache: Dict[str, Tuple[np.ndarray, Tuple[float, float, float]]] = {}
    input_cache: Dict[str, np.ndarray] = {}
    rows = []
    total = len(manifest) * len(MODELS)
    counter = 0
    for mr in manifest.itertuples(index=False):
        pid = str(mr.patient_id)
        if pid not in target_cache:
            t, _ti, spacing = read_binary_lps(mr.target_path)
            target_cache[pid] = (t, spacing)
        else:
            t, spacing = target_cache[pid]
        key = str(mr.input_path)
        if key not in input_cache:
            inp, _ii, _ = read_binary_lps(mr.input_path)
            input_cache[key] = inp
        else:
            inp = input_cache[key]
        true_ep = endpoint_metrics(t, inp, spacing[0])
        for model in MODELS:
            counter += 1
            vis = compare_root / "synthetic_probe" / model / "probe_visualization_shared"
            rec_path = vis / f"reconstruction_{mr.probe_type}_level_{probe_token(mr.level)}_patient_{pid}.nii.gz"
            if not rec_path.is_file():
                continue
            rec, _ri, _ = read_binary_lps(rec_path)
            residual_ep = endpoint_metrics(t, rec, spacing[0])
            decomp = correction_decomposition(t, inp, rec)
            rows.append({
                "model": model, "patient_id": pid, "case_id": int(mr.case_index),
                "probe_type": str(mr.probe_type), "level": float(mr.level),
                **prefixed(true_ep, "true_"), **prefixed(residual_ep, "residual_"), **decomp,
                "endpoint_error_reduction_mm": true_ep["max_abs_endpoint_error_mm"] - residual_ep["max_abs_endpoint_error_mm"],
                "endpoint_error_fraction_corrected": safe_div(
                    true_ep["max_abs_endpoint_error_mm"] - residual_ep["max_abs_endpoint_error_mm"],
                    true_ep["max_abs_endpoint_error_mm"],
                ),
            })
            if counter % 500 == 0:
                print(f"Deep synthetic geometry [{counter}/{total}]")
    return pd.DataFrame(rows)


def summarize_actual(d: pd.DataFrame) -> pd.DataFrame:
    if not len(d): return pd.DataFrame()
    metrics = [
        "missing_correction_recall", "addition_precision", "excess_correction_recall", "removal_precision",
        "overall_error_correction_recall", "change_precision", "voxel_error_reduction_fraction",
        "one_minus_dice_fraction_corrected", "hd99_mm_fraction_corrected", "max_abs_endpoint_error_mm_fraction_corrected",
    ]
    rows = []
    for filtered in (False, True):
        dd = d[~d["exclude_incomplete_acquisition"].fillna(False).astype(bool)] if filtered else d
        for (model, mode, split), g in dd.groupby(["model", "localization_mode", "shape_model_split"]):
            row = {"model": model, "localization_mode": mode, "shape_model_split": split, "exclude_incomplete": filtered, "n": len(g)}
            for m in metrics:
                vals = pd.to_numeric(g[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                row[f"median_{m}"] = float(vals.median()) if len(vals) else np.nan
                row[f"p25_{m}"] = float(vals.quantile(0.25)) if len(vals) else np.nan
                row[f"p75_{m}"] = float(vals.quantile(0.75)) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def qa_performance(d: pd.DataFrame) -> pd.DataFrame:
    if not len(d): return pd.DataFrame()
    pairs = {
        "1_minus_dice": ("true_one_minus_dice", "qa_one_minus_dice"),
        "assd": ("true_assd_mm", "qa_assd_mm"),
        "hd95": ("true_hd95_mm", "qa_hd95_mm"),
        "hd99": ("true_hd99_mm", "qa_hd99_mm"),
        "max_endpoint": ("true_max_abs_endpoint_error_mm", "qa_max_abs_endpoint_error_mm"),
    }
    rows = []
    for model in MODELS:
        for mode in MODES:
            mm = d[(d["model"] == model) & (d["localization_mode"] == mode)].copy()
            if not len(mm): continue
            dev = mm[mm["shape_model_split"].isin(["training1", "validation1"])]
            for metric, (truth, qa) in pairs.items():
                cutoff = percentile_higher(dev[truth], 75.0)
                for filtered in (False, True):
                    use = mm[~mm["exclude_incomplete_acquisition"].fillna(False).astype(bool)] if filtered else mm
                    for split in SPLITS:
                        g = use[use["shape_model_split"] == split]
                        sp = safe_spearman(g[qa], g[truth])
                        auc = np.nan; bad = good = 0
                        if split == "testing1" and np.isfinite(cutoff):
                            td = g[[truth, qa]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                            y = (td[truth].to_numpy(float) > cutoff).astype(int)
                            bad, good = int(y.sum()), int((y == 0).sum())
                            auc = auc_rank(y, td[qa].to_numpy(float))
                        rows.append({
                            "model": model, "localization_mode": mode, "metric": metric,
                            "shape_model_split": split, "exclude_incomplete": filtered,
                            "n": int(len(g)), "spearman": sp,
                            "development_p75_cutoff": cutoff,
                            "test_p75_auc": auc, "test_bad": bad, "test_good": good,
                        })
    return pd.DataFrame(rows)


def plot_clean(clean: pd.DataFrame, out: Path) -> None:
    if not len(clean): return
    d = clean[clean["localization_mode"] == "approved_reference"].copy()
    labels, vals = [], []
    for model in MODELS:
        for split in SPLITS:
            x = pd.to_numeric(d[(d["model"] == model) & (d["shape_model_split"] == split)]["clean_1_minus_dice"], errors="coerce").dropna().to_numpy()
            if len(x): labels.append(f"{model}\n{split}"); vals.append(x)
    if vals:
        fig, ax = plt.subplots(figsize=(10, 5.5)); ax.boxplot(vals, showfliers=False)
        ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Clean approved reconstruction 1-Dice"); ax.set_title("Identity preservation by split")
        ax.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def plot_actual_correction(actual: pd.DataFrame, out: Path) -> None:
    if not len(actual): return
    d = actual[(actual["localization_mode"] == "autoseg_centered") & (actual["shape_model_split"] == "testing1")].copy()
    labels, vals = [], []
    for model in MODELS:
        x = pd.to_numeric(d[d["model"] == model]["overall_error_correction_recall"], errors="coerce").dropna().to_numpy()
        if len(x): labels.append(model); vals.append(x)
    if vals:
        fig, ax = plt.subplots(figsize=(6.5, 5.2)); ax.boxplot(vals, showfliers=False)
        ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Fraction of true error voxels beneficially corrected"); ax.set_title("Month3 actual-autoseg correction recall")
        ax.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def plot_hd99_scatter(actual: pd.DataFrame, out: Path) -> None:
    if not len(actual): return
    d = actual[(actual["localization_mode"] == "autoseg_centered") & (actual["shape_model_split"] == "testing1")].copy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model, marker in (("previous_generation", "o"), ("v4_corruption", "x")):
        g = d[d["model"] == model]
        ax.scatter(g["qa_hd99_mm"], g["true_hd99_mm"], marker=marker, alpha=0.7, label=model)
    ax.set_xlabel("QA disagreement HD99 (mm)"); ax.set_ylabel("True autoseg HD99 (mm)")
    ax.set_title("Month3 HD99: QA signal vs true error"); ax.grid(alpha=0.2); ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def md_table(df: pd.DataFrame, cols: List[str], max_rows: int = 30) -> List[str]:
    if df is None or not len(df): return ["No rows available."]
    d = df[cols].head(max_rows).copy()
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "NA" if not np.isfinite(v) else f"{v:.4f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in d.itertuples(index=False, name=None): lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-root", required=True)
    p.add_argument("--acquisition-audit-csv", default=None)
    p.add_argument("--probe-bank-manifest", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--skip-deep-probe-geometry", action="store_true")
    args = p.parse_args()

    root = Path(args.compare_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    acq_path = Path(args.acquisition_audit_csv).expanduser().resolve() if args.acquisition_audit_csv else root / "diagnostics" / "acquisition_audit" / "all_months_acquisition_audit.csv"
    acq = acquisition_table(acq_path)

    clean = clean_cases(root, acq); clean.to_csv(out / "clean_reconstruction_cases.csv", index=False)
    clean_summary = summarize_clean(clean); clean_summary.to_csv(out / "clean_reconstruction_summary.csv", index=False)

    probe, probe_summary = analyze_probe_csv(root)
    probe.to_csv(out / "synthetic_probe_correction_cases.csv", index=False)
    probe_summary.to_csv(out / "synthetic_probe_correction_summary.csv", index=False)

    if args.probe_bank_manifest:
        bank = Path(args.probe_bank_manifest).expanduser().resolve()
    else:
        manifests = sorted((root / "shared_probe_bank").glob("*/probe_bank_manifest.csv"))
        bank = manifests[0] if manifests else Path("__missing__")
    deep = pd.DataFrame()
    if not args.skip_deep_probe_geometry:
        deep = deep_probe_geometry(root, bank)
        deep.to_csv(out / "synthetic_probe_geometry_diagnostics.csv", index=False)
        if len(deep):
            deep.groupby(["model", "probe_type", "level"], dropna=False).agg(
                n=("patient_id", "size"),
                median_endpoint_fraction_corrected=("endpoint_error_fraction_corrected", "median"),
                median_missing_correction_recall=("missing_correction_recall", "median"),
                median_excess_correction_recall=("excess_correction_recall", "median"),
                median_change_precision=("change_precision", "median"),
            ).reset_index().to_csv(out / "synthetic_probe_geometry_summary.csv", index=False)

    actual = read_actual_cases(root, acq); actual.to_csv(out / "actual_autoseg_correction_diagnostics.csv", index=False)
    actual_summary = summarize_actual(actual); actual_summary.to_csv(out / "actual_autoseg_correction_summary.csv", index=False)
    qa = qa_performance(actual); qa.to_csv(out / "qa_signal_performance_diagnostics.csv", index=False)

    plot_clean(clean, out / "clean_identity_1minusdice_by_split.png")
    plot_actual_correction(actual, out / "actual_correction_recall_testing1.png")
    plot_hd99_scatter(actual, out / "qa_vs_true_hd99_testing1.png")

    # Compact report focused on the diagnostic fork, not manuscript polish.
    report = [
        "# SpinalCord NSR-QA four-level diagnostic report", "",
        "This report separates identity preservation, synthetic correction capacity, real-error correction/transfer, and final QA-signal performance. "
        "HD95/HD99/max-HD below are recomputed explicitly from masks; they do **not** assume that the external clinical metric named `haus` is HD95.", "",
        "## Acquisition audit", "",
    ]
    if len(acq):
        flagged = acq[acq["exclude_incomplete_acquisition"].fillna(False).astype(bool)]
        report.append(f"Confirmed/likely incomplete acquisitions excluded in filtered summaries: **{len(flagged)}**.")
        if len(flagged):
            report += [""] + md_table(flagged, [c for c in ["patient_id", "month_membership", "acquisition_review_status", "approved_si_span_mm", "ct_si_extent_mm"] if c in flagged.columns])
    else:
        report.append("Acquisition audit CSV was not available; filtered analyses are identical to unfiltered analyses.")

    report += ["", "## Level 1 — clean approved reconstruction", "",
               "This is the identity-preservation/generalization test. `approved_reference` is the pure clean-shape test; `autoseg_centered` feeds the same perfect contour on the deployment crop and isolates localization sensitivity.", ""]
    if len(clean_summary):
        sel = clean_summary[(clean_summary["localization_mode"] == "approved_reference") & (~clean_summary["exclude_incomplete"])]
        report += md_table(sel, ["model", "shape_model_split", "n", "median_clean_1_minus_dice", "median_clean_hd99_mm", "median_clean_max_abs_endpoint_error_mm"])
    report += ["", "## Level 2 — shared synthetic-probe correction", "",
               "Positive fraction-corrected values mean reconstruction moved toward the approved target; 1 means full correction and values below 0 mean reconstruction made the probe worse.", ""]
    if len(probe_summary):
        endpoint_types = ["cut_upper_half", "cut_lower_half", "dilated", "eroded"]
        sel = probe_summary[probe_summary["probe_type"].astype(str).isin(endpoint_types)]
        if not len(sel): sel = probe_summary
        report += md_table(sel, ["model", "probe_type", "level", "n", "median_dice_fraction_corrected", "median_hd95_fraction_corrected", "median_assd_fraction_corrected"], max_rows=24)

    report += ["", "## Level 3 — actual-autoseg correction", "",
               "`overall_error_correction_recall` is the fraction of true missing/excess error voxels beneficially changed by the reconstruction. `change_precision` is the fraction of all model changes that were beneficial rather than overcorrection.", ""]
    if len(actual_summary):
        sel = actual_summary[(actual_summary["localization_mode"] == "autoseg_centered") & (actual_summary["shape_model_split"] == "testing1")]
        report += md_table(sel, ["model", "exclude_incomplete", "n", "median_overall_error_correction_recall", "median_change_precision", "median_one_minus_dice_fraction_corrected", "median_hd99_mm_fraction_corrected", "median_max_abs_endpoint_error_mm_fraction_corrected"])

    report += ["", "## Level 4 — final QA signal", "",
               "P75 thresholds are fitted on Month1+2 true error for the same explicitly recomputed metric. `exclude_incomplete=True` removes confirmed/likely incomplete-acquisition cases only; broader boundary-limited review cases remain included.", ""]
    if len(qa):
        sel = qa[(qa["localization_mode"] == "autoseg_centered") & (qa["shape_model_split"] == "testing1") & qa["metric"].isin(["1_minus_dice", "hd95", "hd99", "max_endpoint"])]
        report += md_table(sel, ["model", "metric", "exclude_incomplete", "n", "spearman", "development_p75_cutoff", "test_p75_auc", "test_bad", "test_good"], max_rows=30)

    report += ["", "## Interpretation guide", "",
        "- **Poor clean reconstruction on all splits:** decoder/identity-preservation is a primary bottleneck.",
        "- **Good train clean reconstruction but worse validation/Month3:** shape-prior generalization is a primary bottleneck.",
        "- **Strong synthetic correction but weak actual correction:** synthetic-to-real transfer/context is the primary bottleneck.",
        "- **Weak synthetic and actual correction:** training objective/model capacity is more important than further corruption-distribution tuning.",
        "- **Good actual correction but weak QA correlation:** reconstruction disagreement is a poor readout even though the corrector itself is useful; consider a different QA score/readout.",
        "- **Large change after excluding incomplete acquisition:** valid CT support/FOV must be modeled explicitly before judging the detector.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "analysis_metadata.json").write_text(json.dumps({
        "compare_root": str(root), "acquisition_audit_csv": str(acq_path), "probe_bank_manifest": str(bank),
        "hd_definition": "HD95/HD99/max-HD recomputed from concatenated bidirectional voxel-surface distances in physical space",
        "external_haus_note": "No assumption is made about the definition of the external clinical metric named 'haus'.",
    }, indent=2), encoding="utf-8")
    print(f"Wrote diagnostic report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
