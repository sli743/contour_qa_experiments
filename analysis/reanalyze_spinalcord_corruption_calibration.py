#!/usr/bin/env python3
"""Lightweight v2 reanalysis of an existing SpinalCord corruption calibration.

This script intentionally reuses the CSV outputs from the expensive calibration
run. It does not read NIfTI files, regenerate controlled candidates, or resample
the stochastic training bank.

Primary corrections/additions:
  * bounded mixed support distance with sparse endpoint presence+magnitude terms;
  * full / endpoint-excluded / endpoint-only coverage sensitivity;
  * explicit endpoint prevalence/severity by real cohort and synthetic branch;
  * corrected nearest-branch enrichment, including same-patient development fits;
  * group-level support/gap attribution for uncovered cases;
  * reduced-feature corrected controlled-family refit from stored candidate rows;
  * skipped-case audit; and
  * decision-support summary for the next corruption-design experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis.spinalcord_corruption_support import (
    SPARSE_ENDPOINT_FEATURES,
    fit_mixed_scaler,
    group_support_summary,
    mixed_grouped_distance,
    mixed_support_coverage,
)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_No rows._"
    x = df.head(max_rows).copy()
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in x.iterrows():
        vals = []
        for col in cols:
            v = row[col]
            if isinstance(v, (float, np.floating)):
                vals.append("NA" if not np.isfinite(v) else f"{float(v):.3f}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def load_inputs(root: Path) -> Dict[str, pd.DataFrame]:
    paths = {
        "real": root / "real" / "case_features.csv",
        "bank": root / "training_coverage" / "synthetic_bank_features.csv",
        "legacy_coverage": root / "training_coverage" / "real_to_synthetic_nearest_neighbor.csv",
        "manifest": root / "cohort_manifest.csv",
        "controlled": root / "fits" / "controlled_family_fits.csv",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required existing calibration outputs:\n" + "\n".join(missing))
    return {k: pd.read_csv(v) for k, v in paths.items()}


def mode_groups(groups: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, List[str]]]:
    full = {k: list(v) for k, v in groups.items()}
    excluded = {
        k: [x for x in v if x not in SPARSE_ENDPOINT_FEATURES]
        for k, v in full.items()
    }
    excluded = {k: v for k, v in excluded.items() if v}
    endpoint = {"endpoint": list(SPARSE_ENDPOINT_FEATURES)}
    return {
        "full_mixed": full,
        "endpoint_excluded": excluded,
        "endpoint_only": endpoint,
    }


def coverage_summary_from_cases(cases: pd.DataFrame, mode: str, threshold: float) -> List[Dict[str, object]]:
    rows = []
    for cohort, g in cases.groupby("cohort"):
        covered = bool_series(g["covered_by_current_training_support"])
        rows.append({
            "mode": mode,
            "cohort": cohort,
            "n": len(g),
            "covered": int(covered.sum()),
            "coverage_fraction": float(covered.mean()),
            "median_nearest_distance": float(g["nearest_pooled_training_distance"].median()),
            "p90_nearest_distance": float(g["nearest_pooled_training_distance"].quantile(0.90)),
            "p95_nearest_distance": float(g["nearest_pooled_training_distance"].quantile(0.95)),
            "coverage_distance_threshold": float(threshold),
        })
    return rows


def endpoint_prevalence_tables(
    real: pd.DataFrame,
    bank: pd.DataFrame,
    presence_threshold_mm: float,
    thresholds_mm: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bank_nonclean = bank[~bank["branch"].isin(["clean", "rejected_to_clean"])].copy()
    sources: List[Tuple[str, str, pd.DataFrame]] = [
        ("real", "development", real[real.cohort == "development"]),
        ("real", "month3", real[real.cohort == "month3"]),
        ("synthetic", "all_nonclean", bank_nonclean),
    ]
    for branch, g in bank_nonclean.groupby("branch"):
        sources.append(("synthetic_branch", str(branch), g))

    prev_rows: List[Dict[str, object]] = []
    sev_rows: List[Dict[str, object]] = []
    for source_type, source, df in sources:
        for feature in SPARSE_ENDPOINT_FEATURES:
            vals = pd.to_numeric(df[feature], errors="coerce").dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            positive = vals[vals > presence_threshold_mm]
            sev_rows.append({
                "source_type": source_type,
                "source": source,
                "endpoint_feature": feature,
                "n_total": len(vals),
                "n_positive": len(positive),
                "positive_fraction": float(len(positive) / len(vals)),
                "positive_p25_mm": float(np.percentile(positive, 25)) if len(positive) else np.nan,
                "positive_p50_mm": float(np.percentile(positive, 50)) if len(positive) else np.nan,
                "positive_p75_mm": float(np.percentile(positive, 75)) if len(positive) else np.nan,
                "positive_p90_mm": float(np.percentile(positive, 90)) if len(positive) else np.nan,
                "positive_p95_mm": float(np.percentile(positive, 95)) if len(positive) else np.nan,
                "positive_max_mm": float(np.max(positive)) if len(positive) else np.nan,
            })
            for requested in thresholds_mm:
                effective = presence_threshold_mm if requested <= 0 else float(requested)
                count = int(np.sum(vals > effective))
                prev_rows.append({
                    "source_type": source_type,
                    "source": source,
                    "endpoint_feature": feature,
                    "threshold_label": "any" if requested <= 0 else f">{requested:g}mm",
                    "threshold_mm": float(effective),
                    "n_total": len(vals),
                    "n_above": count,
                    "fraction_above": float(count / len(vals)),
                })
    prev = pd.DataFrame(prev_rows)
    severity = pd.DataFrame(sev_rows)

    real_dev = prev[(prev.source_type == "real") & (prev.source == "development")][
        ["endpoint_feature", "threshold_label", "threshold_mm", "fraction_above", "n_total", "n_above"]
    ].rename(columns={
        "fraction_above": "real_development_fraction",
        "n_total": "real_development_n",
        "n_above": "real_development_n_above",
    })
    branches = prev[prev.source_type == "synthetic_branch"].copy()
    comparison = branches.merge(real_dev, on=["endpoint_feature", "threshold_label", "threshold_mm"], how="left")
    comparison["synthetic_minus_real_pp"] = 100.0 * (comparison.fraction_above - comparison.real_development_fraction)
    comparison["synthetic_to_real_ratio"] = np.where(
        comparison.real_development_fraction > 0,
        comparison.fraction_above / comparison.real_development_fraction,
        np.nan,
    )
    return prev, severity, comparison


def enrichment_table(
    cases: pd.DataFrame,
    bank: pd.DataFrame,
    branch_col: str,
    same_patient: bool,
) -> pd.DataFrame:
    bank_nonclean = bank[~bank.branch.isin(["clean", "rejected_to_clean"])].copy()
    if same_patient:
        cases = cases[cases.cohort == "development"].copy()
        cases = cases[cases[branch_col].fillna("") != ""]
        # Baseline is the aggregate branch mix among the same-patient candidate
        # pools actually available for the development patients being evaluated.
        pids = set(cases.patient_id.astype(int))
        baseline_bank = bank_nonclean[bank_nonclean.patient_id.astype(int).isin(pids)]
    else:
        baseline_bank = bank_nonclean
    baseline = baseline_bank.groupby("branch").size().rename("bank_n").reset_index()
    baseline["bank_fraction"] = baseline.bank_n / baseline.bank_n.sum()
    nearest = cases.groupby(branch_col).size().rename("nearest_n").reset_index().rename(columns={branch_col: "branch"})
    nearest["nearest_fraction"] = nearest.nearest_n / max(nearest.nearest_n.sum(), 1)
    out = baseline.merge(nearest, on="branch", how="outer").fillna({"bank_n": 0, "bank_fraction": 0, "nearest_n": 0, "nearest_fraction": 0})
    out["enrichment_ratio"] = np.where(out.bank_fraction > 0, out.nearest_fraction / out.bank_fraction, np.nan)
    out["nearest_minus_bank_pp"] = 100.0 * (out.nearest_fraction - out.bank_fraction)
    out["comparison"] = "same_patient" if same_patient else "pooled"
    return out.sort_values("enrichment_ratio", ascending=False)


def corrected_controlled_refit(
    real: pd.DataFrame,
    controlled: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Re-score stored controlled candidates without regenerating NIfTI masks.

    Only candidate features already saved by the original run can be reused, so
    this corrected refit uses a reduced but balanced five-group phenotype.
    """
    reduced_groups = {
        "global_size": ["missing_fraction_gt", "excess_fraction_gt", "relative_volume_change_signed"],
        "surface": ["assd_mm", "hd95_mm"],
        "longitudinal": list(SPARSE_ENDPOINT_FEATURES),
        "displacement": ["centroid_shift_mm"],
        "localization": ["largest_missing_component_fraction", "largest_excess_component_fraction"],
    }
    features = [f for xs in reduced_groups.values() for f in xs]
    scaler = fit_mixed_scaler(real[real.cohort == "development"], features)
    real_by_pid = {int(r.patient_id): r.to_dict() for _, r in real.iterrows()}
    mapping = {
        "missing_fraction_gt": "candidate_missing_fraction_gt",
        "excess_fraction_gt": "candidate_excess_fraction_gt",
        "relative_volume_change_signed": "candidate_relative_volume_change_signed",
        "assd_mm": "candidate_assd_mm",
        "hd95_mm": "candidate_hd95_mm",
        "centroid_shift_mm": "candidate_centroid_shift_mm",
        "inferior_missing_mm": "candidate_inferior_missing_mm",
        "superior_missing_mm": "candidate_superior_missing_mm",
        "inferior_excess_mm": "candidate_inferior_excess_mm",
        "superior_excess_mm": "candidate_superior_excess_mm",
        "largest_missing_component_fraction": "candidate_largest_missing_component_fraction",
        "largest_excess_component_fraction": "candidate_largest_excess_component_fraction",
    }
    rows = []
    for _, r in controlled.iterrows():
        pid = int(r.patient_id)
        if pid not in real_by_pid:
            continue
        candidate = {name: r.get(src, np.nan) for name, src in mapping.items()}
        dist, group_dist = mixed_grouped_distance(real_by_pid[pid], candidate, scaler, reduced_groups)
        row = r.to_dict()
        row["corrected_reduced_phenotype_distance"] = dist
        for group, value in group_dist.items():
            row[f"corrected_distance_{group}"] = value
        rows.append(row)
    rescored = pd.DataFrame(rows)
    if rescored.empty:
        return rescored, pd.DataFrame()

    family_best = (
        rescored.sort_values(["patient_id", "family", "corrected_reduced_phenotype_distance", "weighted_error_dice"], ascending=[True, True, True, False])
        .groupby(["patient_id", "cohort", "family"], as_index=False)
        .first()
    )
    best_rows = []
    for (pid, cohort), g in family_best.groupby(["patient_id", "cohort"]):
        g = g.sort_values(["corrected_reduced_phenotype_distance", "weighted_error_dice"], ascending=[True, False])
        first = g.iloc[0]
        second = g.iloc[1] if len(g) > 1 else None
        out = first.to_dict()
        out["corrected_best_family"] = first.family
        out["corrected_best_distance"] = float(first.corrected_reduced_phenotype_distance)
        out["corrected_second_family"] = str(second.family) if second is not None else ""
        out["corrected_second_distance"] = float(second.corrected_reduced_phenotype_distance) if second is not None else np.nan
        out["corrected_fit_margin"] = (
            float(second.corrected_reduced_phenotype_distance - first.corrected_reduced_phenotype_distance)
            if second is not None else np.nan
        )
        best_rows.append(out)
    return rescored, pd.DataFrame(best_rows)


def skipped_case_audit(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = manifest[manifest.status != "ok"].copy()
    if x.empty:
        return x, pd.DataFrame()
    reason = x.reason.fillna("").astype(str)
    x["missing_ct"] = reason.str.contains("/images/")
    x["missing_approved"] = reason.str.contains("labels-approved")
    x["missing_autoseg"] = reason.str.contains("labels-autoseg")
    x["category"] = np.select(
        [
            x.missing_ct & x.missing_approved & x.missing_autoseg,
            x.missing_ct & ~x.missing_approved & ~x.missing_autoseg,
            ~x.missing_ct & (x.missing_approved | x.missing_autoseg),
        ],
        ["missing_ct_and_both_masks", "missing_ct_only", "missing_mask_file"],
        default="other",
    )
    summary = x.groupby(["cohort", "category"]).size().rename("n").reset_index()
    ids = x.groupby(["cohort", "category"])["patient_id"].apply(lambda s: ",".join(map(str, sorted(map(int, s))))).rename("patient_ids").reset_index()
    summary = summary.merge(ids, on=["cohort", "category"], how="left")
    return x, summary


def design_gap_summary(
    real: pd.DataFrame,
    bank: pd.DataFrame,
    endpoint_prev: pd.DataFrame,
    pooled_enrichment: pd.DataFrame,
    corrected_best: pd.DataFrame,
    group_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    presence_threshold_mm: float,
) -> pd.DataFrame:
    dev = real[real.cohort == "development"]
    syn = bank[~bank.branch.isin(["clean", "rejected_to_clean"])]
    rows: List[Dict[str, object]] = []

    def q(df: pd.DataFrame, col: str, p: float) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").dropna().quantile(p))

    for col, label in (("missing_fraction_gt", "missing volume"), ("hd95_mm", "HD95")):
        rows.append({
            "design_signal": f"overall_{col}",
            "evidence_class": "severity_distribution",
            "real_development_p50": q(dev, col, 0.50),
            "real_development_p95": q(dev, col, 0.95),
            "current_synthetic_p50": q(syn, col, 0.50),
            "current_synthetic_p95": q(syn, col, 0.95),
            "interpretation": f"Current synthetic {label} distribution compared with real development distribution.",
            "design_implication": "reduce generic severity" if q(syn, col, 0.50) > q(dev, col, 0.50) and q(syn, col, 0.95) > q(dev, col, 0.95) else "no simple global severity increase justified",
        })

    # Inferior endpoint frequency is the most clinically structured gap.
    ep = endpoint_prev[
        (endpoint_prev.endpoint_feature == "inferior_missing_mm")
        & (endpoint_prev.threshold_label == ">20mm")
    ]
    real_row = ep[(ep.source_type == "real") & (ep.source == "development")]
    syn_row = ep[(ep.source_type == "synthetic") & (ep.source == "all_nonclean")]
    if len(real_row) and len(syn_row):
        rv = float(real_row.iloc[0].fraction_above)
        sv = float(syn_row.iloc[0].fraction_above)
        rows.append({
            "design_signal": "inferior_missing_gt20mm_frequency",
            "evidence_class": "structured_endpoint_frequency",
            "real_development_p50": np.nan,
            "real_development_p95": np.nan,
            "current_synthetic_p50": np.nan,
            "current_synthetic_p95": np.nan,
            "real_frequency": rv,
            "synthetic_frequency": sv,
            "real_to_synthetic_frequency_ratio": rv / sv if sv > 0 else np.inf,
            "interpretation": "Frequency of clinically large inferior endpoint truncation.",
            "design_implication": "add/dedicate inferior endpoint-truncation branch" if rv > 2 * max(sv, 1e-6) else "current endpoint frequency is not clearly underrepresented",
        })

    smooth = pooled_enrichment[pooled_enrichment.branch == "smooth_warp"]
    if len(smooth):
        er = float(smooth.iloc[0].enrichment_ratio)
        rows.append({
            "design_signal": "smooth_warp_nearest_neighbor_enrichment",
            "evidence_class": "nearest_branch_enrichment",
            "real_development_p50": np.nan,
            "real_development_p95": np.nan,
            "current_synthetic_p50": np.nan,
            "current_synthetic_p95": np.nan,
            "enrichment_ratio": er,
            "interpretation": "How often smooth-warp samples are nearest to real errors relative to their bank frequency.",
            "design_implication": "increase smooth-warp sampling frequency" if er >= 1.5 else "no strong smooth-warp enrichment",
        })

    # Corrected controlled-family results give interpretable severity estimates
    # without rerunning candidate generation.
    if not corrected_best.empty:
        dev_best = corrected_best[corrected_best.cohort == "development"]
        n_dev = max(len(dev_best), 1)
        for family, parameter in (("smooth_warp", "parameter_max_displacement_mm"), ("inferior_cut", "parameter_length_mm")):
            g = dev_best[dev_best.corrected_best_family == family]
            vals = pd.to_numeric(g.get(parameter, pd.Series(dtype=float)), errors="coerce").dropna()
            if len(g):
                rows.append({
                    "design_signal": f"{family}_corrected_best_fit",
                    "evidence_class": "corrected_controlled_family_fit",
                    "n_supporting_cases": len(g),
                    "best_family_fraction": float(len(g) / n_dev),
                    "real_development_p50": float(vals.median()) if len(vals) else np.nan,
                    "real_development_p95": float(vals.quantile(0.95)) if len(vals) else np.nan,
                    "current_synthetic_p50": np.nan,
                    "current_synthetic_p95": np.nan,
                    "interpretation": "Corrected reduced-feature controlled-family prevalence and fitted physical severity.",
                    "design_implication": (
                        "increase smooth-warp sampling while keeping roughly the fitted 2–6 mm support"
                        if family == "smooth_warp" else
                        "add an explicit inferior endpoint-truncation branch centered on the fitted length distribution"
                    ),
                })
        for family in ("global_inward_offset", "global_outward_offset"):
            g = dev_best[dev_best.corrected_best_family == family]
            if len(g):
                vals = pd.to_numeric(g.parameter_magnitude_mm, errors="coerce").dropna()
                syn_g = bank[bank.branch == "global_boundary_offset"]
                syn50 = q(syn_g, "generator_global_boundary_offset_magnitude_mm", 0.50) if "generator_global_boundary_offset_magnitude_mm" in bank else np.nan
                syn95 = q(syn_g, "generator_global_boundary_offset_magnitude_mm", 0.95) if "generator_global_boundary_offset_magnitude_mm" in bank else np.nan
                real50 = float(vals.median()) if len(vals) else np.nan
                rows.append({
                    "design_signal": f"{family}_best_fit_magnitude",
                    "evidence_class": "corrected_controlled_family_fit",
                    "n_supporting_cases": len(g),
                    "best_family_fraction": float(len(g) / n_dev),
                    "real_development_p50": real50,
                    "real_development_p95": float(vals.quantile(0.95)) if len(vals) else np.nan,
                    "current_synthetic_p50": syn50,
                    "current_synthetic_p95": syn95,
                    "interpretation": "Corrected reduced-feature morphology-equivalent boundary magnitude.",
                    "design_implication": (
                        "shift global boundary offsets milder"
                        if len(vals) >= 3 and np.isfinite(real50) and np.isfinite(syn50) and syn50 > 1.2 * real50
                        else "do not expand boundary-offset severity"
                    ),
                })

    # Separate support-range problems from sampling-frequency problems.
    dev_groups = group_summary[group_summary.cohort == "development"]
    for _, g in dev_groups.iterrows():
        if float(g.coverage_fraction) < 0.80:
            rows.append({
                "design_signal": f"{g['group']}_support_coverage",
                "evidence_class": "group_support",
                "group_coverage_fraction": float(g.coverage_fraction),
                "real_development_p50": np.nan,
                "real_development_p95": np.nan,
                "current_synthetic_p50": np.nan,
                "current_synthetic_p95": np.nan,
                "interpretation": f"Only {float(g.coverage_fraction):.1%} of development cases fall within group-specific current synthetic support.",
                "design_implication": (
                    "reshape surface-error phenotypes; do not simply increase severity"
                    if g["group"] == "surface" else
                    "increase diversity of localized error geometry"
                    if g["group"] == "localization" else
                    "inspect and expand this phenotype group selectively"
                ),
            })

    ep_mode = sensitivity[(sensitivity["mode"] == "endpoint_only") & (sensitivity.cohort == "development")]
    if len(ep_mode):
        ep_cov = float(ep_mode.iloc[0].coverage_fraction)
        rows.append({
            "design_signal": "endpoint_severity_support",
            "evidence_class": "coverage_sensitivity",
            "group_coverage_fraction": ep_cov,
            "real_development_p50": np.nan,
            "real_development_p95": np.nan,
            "current_synthetic_p50": np.nan,
            "current_synthetic_p95": np.nan,
            "interpretation": "Endpoint-only support asks whether sampled synthetic endpoint severities span the real endpoint phenotypes, independent of their sampling frequency.",
            "design_implication": (
                "endpoint severity range is adequate; increase structured endpoint sampling frequency rather than widening its maximum"
                if ep_cov >= 0.90 else
                "endpoint severity support itself is incomplete; adjust both frequency and range"
            ),
        })

    # Branches that are never/rarely nearest despite substantial sampling are
    # candidates for downweighting or redesign.
    for branch in ("global_boundary_offset", "stochastic_region_dropout"):
        b = pooled_enrichment[pooled_enrichment.branch == branch]
        if len(b):
            er = float(b.iloc[0].enrichment_ratio)
            rows.append({
                "design_signal": f"{branch}_nearest_enrichment",
                "evidence_class": "nearest_branch_enrichment",
                "enrichment_ratio": er,
                "real_development_p50": np.nan,
                "real_development_p95": np.nan,
                "current_synthetic_p50": np.nan,
                "current_synthetic_p95": np.nan,
                "interpretation": "Nearest-match frequency relative to how often this branch appears in the current synthetic bank.",
                "design_implication": (
                    "downweight or redesign this branch" if er < 0.5 else "no strong evidence for increasing this branch"
                ),
            })
    return pd.DataFrame(rows)


def make_figures(
    out_dir: Path,
    coverage_cases: pd.DataFrame,
    coverage_threshold: float,
    endpoint_prev: pd.DataFrame,
    pooled_enrichment: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    plt.figure(figsize=(7, 4.8))
    for cohort, g in coverage_cases.groupby("cohort"):
        plt.hist(g.nearest_pooled_training_distance, bins=18, alpha=0.55, label=cohort)
    plt.axvline(coverage_threshold, linestyle="--", linewidth=1.5, label=f"synthetic P95 NN = {coverage_threshold:.3f}")
    plt.xlabel("Corrected mixed nearest-training distance")
    plt.ylabel("Cases")
    plt.title("Corrected current-training support coverage")
    plt.legend(); plt.tight_layout()
    p = out_dir / "corrected_training_support_distance.png"
    plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close(); made.append(p.name)

    ep = endpoint_prev[
        (endpoint_prev.endpoint_feature == "inferior_missing_mm")
        & (endpoint_prev.threshold_label == ">20mm")
        & (
            ((endpoint_prev.source_type == "real") & (endpoint_prev.source == "development"))
            | (endpoint_prev.source_type == "synthetic_branch")
        )
    ].copy()
    if not ep.empty:
        ep["label"] = np.where(ep.source_type == "real", "real development", ep.source)
        ep = ep.sort_values("fraction_above", ascending=True)
        plt.figure(figsize=(7.2, max(4.5, 0.45 * len(ep) + 1.5)))
        plt.barh(ep.label, 100.0 * ep.fraction_above)
        plt.xlabel("Cases / samples with >20 mm inferior missing extent (%)")
        plt.title("Inferior endpoint truncation: real vs synthetic branches")
        plt.tight_layout()
        p = out_dir / "inferior_missing_gt20mm_prevalence.png"
        plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close(); made.append(p.name)

    if not pooled_enrichment.empty:
        x = pooled_enrichment.sort_values("enrichment_ratio")
        plt.figure(figsize=(7.2, 4.8))
        plt.barh(x.branch, x.enrichment_ratio)
        plt.axvline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("Nearest-match enrichment ratio")
        plt.title("Corrected pooled nearest-branch enrichment")
        plt.tight_layout()
        p = out_dir / "nearest_branch_enrichment.png"
        plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close(); made.append(p.name)

    dev = group_summary[group_summary.cohort == "development"].sort_values("coverage_fraction")
    if not dev.empty:
        plt.figure(figsize=(7.2, 4.8))
        plt.barh(dev.group, 100.0 * dev.coverage_fraction)
        plt.xlabel("Development cases within group-specific synthetic support (%)")
        plt.xlim(0, 100)
        plt.title("Support coverage by phenotype group")
        plt.tight_layout()
        p = out_dir / "group_support_coverage.png"
        plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close(); made.append(p.name)
    return made


def write_report(
    out_dir: Path,
    sensitivity: pd.DataFrame,
    legacy_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    endpoint_prev: pd.DataFrame,
    endpoint_sev: pd.DataFrame,
    pooled_enrichment: pd.DataFrame,
    same_enrichment: pd.DataFrame,
    corrected_best: pd.DataFrame,
    gap_summary: pd.DataFrame,
    skipped_summary: pd.DataFrame,
    figures: Sequence[str],
    presence_threshold_mm: float,
) -> Path:
    primary = sensitivity[sensitivity["mode"] == "full_mixed"].copy()
    dev_primary = primary[primary.cohort == "development"]
    m3_primary = primary[primary.cohort == "month3"]
    lines = [
        "# SpinalCord corruption calibration reanalysis v2",
        "",
        "## Why this reanalysis exists",
        "",
        "The original support metric treated every phenotype variable as an ordinary continuous feature. Several endpoint variables were zero-inflated in development and received an effective scale of 0.001 mm, so a small endpoint mismatch could dominate the entire distance. This reanalysis reuses the existing CSVs but replaces that metric with a bounded group-balanced mixed distance.",
        "",
        f"Endpoint presence is defined using a development-resolution-aware threshold of **{presence_threshold_mm:.3f} mm**. If one contour has an endpoint error and the other does not, that endpoint contributes the maximum feature distance. If both have an endpoint error, only their positive magnitudes are compared, using a positive-development severity scale. Every feature distance is capped at 1 before equal-weight phenotype groups are combined.",
        "",
        "No NIfTI files, controlled corruptions, or stochastic training samples are regenerated by this reanalysis.",
        "",
        "## Corrected current-training coverage",
        "",
    ]
    if len(dev_primary):
        r = dev_primary.iloc[0]
        lines.append(f"- Development coverage: **{int(r.covered)}/{int(r.n)} ({r.coverage_fraction:.1%})**")
    if len(m3_primary):
        r = m3_primary.iloc[0]
        lines.append(f"- Month3 coverage: **{int(r.covered)}/{int(r.n)} ({r.coverage_fraction:.1%})**")
    lines += ["", "### Sensitivity to endpoint features", "", markdown_table(sensitivity, 20)]
    if not legacy_summary.empty:
        lines += [
            "",
            "### Legacy coverage (for audit only)",
            "",
            "The following values are retained only to show how the old distance behaved. They should not be used to tune the corruption generator.",
            "",
            markdown_table(legacy_summary, 10),
        ]
    lines += [
        "",
        "## Which phenotype groups are under-covered?",
        "",
        markdown_table(group_summary.sort_values(["cohort", "coverage_fraction"]), 30),
        "",
        "## Endpoint frequency: real errors versus current training generator",
        "",
    ]
    key_ep = endpoint_prev[
        (endpoint_prev.endpoint_feature.isin(["inferior_missing_mm", "superior_missing_mm"]))
        & (endpoint_prev.threshold_label.isin(["any", ">3mm", ">20mm", ">30mm"]))
        & (
            ((endpoint_prev.source_type == "real") & (endpoint_prev.source == "development"))
            | (endpoint_prev.source_type == "synthetic")
            | (endpoint_prev.source_type == "synthetic_branch")
        )
    ][["source_type", "source", "endpoint_feature", "threshold_label", "n_total", "n_above", "fraction_above"]]
    lines += [markdown_table(key_ep, 100), "", "### Positive endpoint severity", ""]
    key_sev = endpoint_sev[
        endpoint_sev.endpoint_feature.isin(["inferior_missing_mm", "superior_missing_mm"])
    ][["source_type", "source", "endpoint_feature", "n_positive", "positive_fraction", "positive_p50_mm", "positive_p95_mm", "positive_max_mm"]]
    lines += [markdown_table(key_sev, 60)]
    lines += [
        "",
        "## Which current training branches resemble real errors?",
        "",
        "### Pooled nearest-neighbor enrichment",
        "",
        markdown_table(pooled_enrichment[["branch", "bank_fraction", "nearest_fraction", "enrichment_ratio", "nearest_minus_bank_pp"]], 20),
        "",
        "### Same-patient development nearest-neighbor enrichment",
        "",
        "This comparison removes normal inter-patient anatomy as much as possible by matching each development real error only to synthetic corruptions generated from that same patient's approved contour.",
        "",
        markdown_table(same_enrichment[["branch", "bank_fraction", "nearest_fraction", "enrichment_ratio", "nearest_minus_bank_pp"]], 20),
    ]
    if not corrected_best.empty:
        fam = corrected_best[corrected_best.cohort == "development"].groupby("corrected_best_family").size().rename("n").reset_index()
        fam["fraction"] = fam.n / fam.n.sum()
        lines += [
            "",
            "## Corrected controlled-family refit",
            "",
            "The original controlled candidate masks are not regenerated. The stored candidate phenotype fields are rescored with a reduced five-group mixed distance. Because signed-surface quantiles and directional centroid components were not stored per candidate, this is a reduced-feature correction and should be treated as supportive rather than definitive.",
            "",
            markdown_table(fam.sort_values("n", ascending=False), 30),
        ]
    lines += [
        "",
        "## Corruption-design gap summary",
        "",
        markdown_table(gap_summary, 30),
        "",
        "## Skipped-case audit",
        "",
        markdown_table(skipped_summary, 30),
        "",
        "Missing cases are not silently recovered in this reanalysis. If `missing_ct_only` cases have valid masks and the CT filename is merely different, resolve that path issue before treating clinical error frequencies as final.",
        "",
        "## Interpretation guardrails",
        "",
        "- Month1+Month2 determine all distance/scaling rules. Month3 remains a temporal audit.",
        "- Coverage measures similarity to the *sampled current corruption bank*, not proof that the network learned to correct every covered case.",
        "- Branch enrichment is stronger evidence about which existing synthetic families resemble real errors than raw branch frequency alone, but it is still a nearest-neighbor diagnostic rather than a causal label.",
        "- The corrected controlled-family refit uses only phenotype fields stored in the existing CSV; a future full rerun will use the corrected mixed distance directly during candidate fitting.",
        "- Do not set training probabilities equal to clinical prevalence automatically. Rare severe errors can justify deliberate oversampling.",
        "",
        "## Figures",
        "",
    ]
    for f in figures:
        lines += [f"![{f}](figures/{f})", ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration-root", required=True, help="Existing spinalcord_corruption_calibration output directory")
    ap.add_argument("--analysis-config", default="configs/dino_shape/spinalcord_corruption_calibration.yaml")
    ap.add_argument("--output-dir", default=None, help="Defaults to <calibration-root>/reanalysis_v2")
    ap.add_argument("--coverage-percentile", type=float, default=95.0)
    ap.add_argument("--endpoint-thresholds-mm", default="0,3,10,20,30,40")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    root = Path(args.calibration_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "reanalysis_v2"
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.analysis_config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_yaml(cfg_path)
    groups = {k: list(v) for k, v in cfg["phenotype_groups"].items()}
    inputs = load_inputs(root)
    real, bank = inputs["real"], inputs["bank"]
    development = real[real.cohort == "development"].copy()
    features = sorted({x for names in groups.values() for x in names})
    scaler = fit_mixed_scaler(development, features)
    presence_threshold = float(scaler["inferior_missing_mm"]["presence_threshold"])

    (out / "corrected_scaler.json").write_text(json.dumps(scaler, indent=2) + "\n", encoding="utf-8")

    sensitivity_rows: List[Dict[str, object]] = []
    coverage_by_mode: Dict[str, pd.DataFrame] = {}
    primary_bank_nn = None
    primary_threshold = np.nan
    for mode, mode_group in mode_groups(groups).items():
        cases, bank_nn, threshold = mixed_support_coverage(
            real, bank, scaler, mode_group, args.coverage_percentile, True
        )
        cases["mode"] = mode
        coverage_by_mode[mode] = cases
        sensitivity_rows.extend(coverage_summary_from_cases(cases, mode, threshold))
        if mode == "full_mixed":
            primary_bank_nn = bank_nn
            primary_threshold = threshold
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out / "coverage_sensitivity.csv", index=False)
    primary = coverage_by_mode["full_mixed"]
    primary.to_csv(out / "corrected_coverage_cases.csv", index=False)
    pd.DataFrame(coverage_summary_from_cases(primary, "full_mixed", primary_threshold)).to_csv(
        out / "corrected_coverage_summary.csv", index=False
    )
    if primary_bank_nn is not None:
        primary_bank_nn.to_csv(out / "corrected_synthetic_nn_reference.csv", index=False)

    group_summary = group_support_summary(real, bank, scaler, groups, args.coverage_percentile, True)
    group_summary.to_csv(out / "group_support_summary.csv", index=False)

    thresholds = [float(x.strip()) for x in args.endpoint_thresholds_mm.split(",") if x.strip()]
    endpoint_prev, endpoint_sev, endpoint_branch = endpoint_prevalence_tables(
        real, bank, presence_threshold, thresholds
    )
    endpoint_prev.to_csv(out / "endpoint_prevalence_comparison.csv", index=False)
    endpoint_sev.to_csv(out / "endpoint_positive_severity.csv", index=False)
    endpoint_branch.to_csv(out / "endpoint_branch_comparison.csv", index=False)

    pooled_enrichment_rows = []
    for cohort, g in primary.groupby("cohort"):
        x = enrichment_table(g, bank, "nearest_training_branch", same_patient=False)
        x["cohort"] = cohort
        pooled_enrichment_rows.append(x)
    pooled_enrichment = pd.concat(pooled_enrichment_rows, ignore_index=True)
    pooled_enrichment.to_csv(out / "nearest_branch_enrichment.csv", index=False)
    same_enrichment = enrichment_table(
        primary[primary.cohort == "development"], bank,
        "nearest_same_patient_training_branch", same_patient=True,
    )
    same_enrichment.to_csv(out / "same_patient_branch_enrichment.csv", index=False)

    controlled_rescored, corrected_best = corrected_controlled_refit(real, inputs["controlled"])
    controlled_rescored.to_csv(out / "corrected_controlled_family_fits.csv", index=False)
    corrected_best.to_csv(out / "corrected_best_family_per_case.csv", index=False)

    skipped, skipped_summary = skipped_case_audit(inputs["manifest"])
    skipped.to_csv(out / "skipped_case_audit.csv", index=False)
    skipped_summary.to_csv(out / "skipped_case_summary.csv", index=False)

    legacy = inputs["legacy_coverage"].copy()
    legacy_summary_rows = []
    if not legacy.empty:
        legacy["covered_by_current_training_support"] = bool_series(legacy.covered_by_current_training_support)
        for cohort, g in legacy.groupby("cohort"):
            legacy_summary_rows.append({
                "cohort": cohort,
                "n": len(g),
                "legacy_covered": int(g.covered_by_current_training_support.sum()),
                "legacy_coverage_fraction": float(g.covered_by_current_training_support.mean()),
                "legacy_median_distance": float(g.nearest_pooled_training_distance.median()),
                "legacy_p95_distance": float(g.nearest_pooled_training_distance.quantile(0.95)),
                "legacy_threshold": float(g.coverage_distance_threshold.iloc[0]),
                "status": "audit_only_zero_IQR_unstable",
            })
    legacy_summary = pd.DataFrame(legacy_summary_rows)
    legacy_summary.to_csv(out / "legacy_coverage_audit.csv", index=False)

    pooled_dev = pooled_enrichment[pooled_enrichment.cohort == "development"].copy()
    gap_summary = design_gap_summary(
        real, bank, endpoint_prev, pooled_dev, corrected_best, group_summary, sensitivity, presence_threshold
    )
    gap_summary.to_csv(out / "corruption_design_gap_summary.csv", index=False)

    figures = make_figures(fig_dir, primary, primary_threshold, endpoint_prev, pooled_dev, group_summary)
    report = write_report(
        out, sensitivity, legacy_summary, group_summary, endpoint_prev, endpoint_sev,
        pooled_dev, same_enrichment, corrected_best, gap_summary, skipped_summary,
        figures, presence_threshold,
    )

    print("SpinalCord calibration reanalysis v2 complete")
    print(f"  input:  {root}")
    print(f"  output: {out}")
    print(f"  report: {report}")
    print(f"  endpoint presence threshold: {presence_threshold:.3f} mm")
    for _, r in sensitivity[sensitivity["mode"] == "full_mixed"].iterrows():
        print(f"  corrected {r.cohort} coverage: {r.coverage_fraction:.1%} ({int(r.covered)}/{int(r.n)})")


if __name__ == "__main__":
    main()
