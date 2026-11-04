#!/usr/bin/env python3
"""Aggregate the Lung_L corruption-vs-no-corruption real-autoseg experiment.

This script deliberately does not manufacture a good/bad label from Dice/HD95.
It freezes and exposes the per-patient QA rankings first.  Manual contour-error
behavior/domain labels can be joined later without changing the scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


AUTO_METRICS = [
    "qa_1_minus_dice",
    "qa_hd95_mm",
    "qa_assd_mm",
    "qa_relative_volume_difference",
]
TRUE_METRICS = [
    "true_1_minus_dice",
    "true_hd95_mm",
    "true_assd_mm",
    "true_relative_volume_difference",
]
CLEAN_METRICS = [
    "clean_1_minus_dice",
    "clean_hd95_mm",
    "clean_hd99_mm",
    "clean_assd_mm",
    "clean_relative_volume_difference",
]
PRIMARY = "qa_1_minus_dice"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--study-root",
        default="/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_v1",
    )
    p.add_argument("--top-fraction", type=float, default=0.20)
    p.add_argument("--discordant-n", type=int, default=5)
    return p.parse_args()


def load_manifest(root: Path) -> pd.DataFrame:
    path = root / "experiment_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"arm": str})
    required = {"task_index", "arm", "seed", "config", "checkpoint"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing columns: {sorted(missing)}")
    return df


def load_outputs(root: Path, manifest: pd.DataFrame):
    auto_rows, clean_rows = [], []
    for r in manifest.itertuples(index=False):
        d = root / "evaluations" / str(r.arm) / f"seed{int(r.seed)}"
        ap = d / "actual_autoseg_testing.csv"
        cp = d / "clean_approved_testing.csv"
        if not ap.is_file():
            raise FileNotFoundError(ap)
        if not cp.is_file():
            raise FileNotFoundError(cp)
        a = pd.read_csv(ap, dtype={"patient_id": str})
        c = pd.read_csv(cp, dtype={"patient_id": str})
        a = a[a["shape_model_split"].astype(str) == "testing1"].copy()
        c = c[c["shape_model_split"].astype(str) == "testing1"].copy()
        a["arm"] = str(r.arm); a["seed"] = int(r.seed)
        c["arm"] = str(r.arm); c["seed"] = int(r.seed)
        auto_rows.append(a); clean_rows.append(c)
    return pd.concat(auto_rows, ignore_index=True), pd.concat(clean_rows, ignore_index=True)


def aggregate_arm(auto: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    merged = auto.merge(
        clean[["patient_id", "arm", "seed"] + [m for m in CLEAN_METRICS if m in clean.columns]],
        on=["patient_id", "arm", "seed"], how="left", validate="one_to_one",
    )
    keys = ["arm", "patient_id"]
    out_rows: List[Dict] = []
    for (arm, pid), g in merged.groupby(keys, sort=False):
        row: Dict = {"arm": arm, "patient_id": str(pid), "n_seeds": int(g["seed"].nunique())}
        for m in AUTO_METRICS + CLEAN_METRICS:
            if m not in g.columns:
                continue
            vals = pd.to_numeric(g[m], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
        for m in TRUE_METRICS:
            if m in g.columns:
                vals = pd.to_numeric(g[m], errors="coerce").to_numpy(float)
                vals = vals[np.isfinite(vals)]
                row[m] = float(np.median(vals)) if len(vals) else np.nan
        # Same-metric auto-vs-approved normality contrast where available.
        pairs = {
            "qa_1_minus_dice": "clean_1_minus_dice",
            "qa_hd95_mm": "clean_hd95_mm",
            "qa_assd_mm": "clean_assd_mm",
            "qa_relative_volume_difference": "clean_relative_volume_difference",
        }
        for qa, cl in pairs.items():
            if qa in g.columns and cl in g.columns:
                delta = pd.to_numeric(g[qa], errors="coerce") - pd.to_numeric(g[cl], errors="coerce")
                row[f"delta_{qa}_minus_approved_mean"] = float(delta.mean())
                row[f"delta_{qa}_minus_approved_std"] = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    score = f"{PRIMARY}_mean"
    out["qa_rank"] = out.groupby("arm")[score].rank(method="min", ascending=False).astype(int)
    n_by_arm = out.groupby("arm")["patient_id"].transform("count").astype(float)
    out["qa_rank_fraction"] = out["qa_rank"] / n_by_arm
    return out


def paired_wide(agg: pd.DataFrame) -> pd.DataFrame:
    arms = sorted(agg["arm"].unique())
    if set(arms) != {"no_corruption", "nsrqa_corruption"}:
        raise ValueError(f"Expected exactly two arms, got {arms}")
    a = agg[agg.arm == "nsrqa_corruption"].copy()
    b = agg[agg.arm == "no_corruption"].copy()
    keep = [c for c in agg.columns if c not in {"arm"}]
    w = a[keep].merge(b[keep], on="patient_id", suffixes=("_nsrqa", "_baseline"), validate="one_to_one")
    for base in [f"{m}_mean" for m in AUTO_METRICS] + ["qa_rank", "qa_rank_fraction"]:
        x, y = f"{base}_nsrqa", f"{base}_baseline"
        if x in w and y in w:
            w[f"{base}_nsrqa_minus_baseline"] = pd.to_numeric(w[x], errors="coerce") - pd.to_numeric(w[y], errors="coerce")
    # True metrics are duplicated after merge; preserve one copy after checking practical equality.
    for m in TRUE_METRICS:
        x, y = f"{m}_nsrqa", f"{m}_baseline"
        if x in w and y in w:
            xx = pd.to_numeric(w[x], errors="coerce"); yy = pd.to_numeric(w[y], errors="coerce")
            if np.nanmax(np.abs(xx.to_numpy(float) - yy.to_numpy(float))) > 1e-6:
                print(f"WARNING: true metric differs across arms: {m}")
            w[m] = xx
    return w


def inspection_shortlist(w: pd.DataFrame, top_fraction: float, discordant_n: int) -> pd.DataFrame:
    n = len(w)
    top_n = max(1, int(np.ceil(float(top_fraction) * n)))
    reasons: Dict[str, set] = {str(pid): set() for pid in w.patient_id}

    for arm in ("nsrqa", "baseline"):
        rank = f"qa_rank_{arm}"
        for pid in w.nsmallest(top_n, rank).patient_id.astype(str):
            reasons[pid].add(f"top_{int(round(top_fraction*100))}pct_{arm}")

    diff_col = "qa_rank_nsrqa_minus_baseline"
    if diff_col in w:
        for pid in w.assign(_abs=w[diff_col].abs()).nlargest(int(discordant_n), "_abs").patient_id.astype(str):
            reasons[pid].add("largest_rank_disagreement")

    keep = [pid for pid, rs in reasons.items() if rs]
    out = w[w.patient_id.astype(str).isin(keep)].copy()
    out["inspection_reason"] = out.patient_id.astype(str).map(lambda p: ";".join(sorted(reasons[p])))
    return out.sort_values(["qa_rank_nsrqa", "qa_rank_baseline", "patient_id"])


def corr_text(x: pd.Series, y: pd.Series) -> str:
    z = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(z) < 3:
        return "NA"
    return f"{z.iloc[:,0].corr(z.iloc[:,1], method='spearman'):.3f}"


def main():
    args = parse_args()
    root = Path(args.study_root).expanduser().resolve()
    manifest = load_manifest(root)
    auto, clean = load_outputs(root, manifest)
    agg = aggregate_arm(auto, clean)
    wide = paired_wide(agg)
    shortlist = inspection_shortlist(wide, args.top_fraction, args.discordant_n)

    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    auto.to_csv(out / "per_seed_actual_autoseg_scores.csv", index=False)
    clean.to_csv(out / "per_seed_clean_approved_scores.csv", index=False)
    agg.to_csv(out / "testing_patient_scores_long.csv", index=False)
    wide.to_csv(out / "testing_patient_scores_paired.csv", index=False)
    shortlist.to_csv(out / "inspection_shortlist.csv", index=False)

    score_n = wide["qa_1_minus_dice_mean_nsrqa"]
    score_b = wide["qa_1_minus_dice_mean_baseline"]
    lines = [
        "# Lung_L NSR-QA corruption-vs-no-corruption experiment",
        "",
        f"Testing patients: {len(wide)}; seeds per arm: {manifest.groupby('arm').size().to_dict()}.",
        "",
        "## Primary interpretation",
        "",
        "The primary QA signal is 1-Dice(autoseg, reconstruction), averaged across seeds. "
        "No Dice/HD threshold is used here to declare a case good or bad. Rankings are frozen first for later visual/domain adjudication.",
        "",
        "## Descriptive checks (not ground-truth performance endpoints)",
        "",
        f"- NSR-QA score vs true 1-Dice Spearman: {corr_text(score_n, wide['true_1_minus_dice'])}",
        f"- No-corruption score vs true 1-Dice Spearman: {corr_text(score_b, wide['true_1_minus_dice'])}",
        f"- NSR-QA score vs true HD95 Spearman: {corr_text(score_n, wide['true_hd95_mm'])}",
        f"- No-corruption score vs true HD95 Spearman: {corr_text(score_b, wide['true_hd95_mm'])}",
        "",
        "These correlations are descriptive only because approved-vs-autoseg metrics do not define the intended manual review target.",
        "",
        "## Visual review",
        "",
        f"`inspection_shortlist.csv` contains the top {args.top_fraction:.0%} by each arm plus the {args.discordant_n} largest rank disagreements.",
        "Review these first, then adjudicate the full testing set if the goal is to claim ranking-for-human-review performance.",
        "",
        "## Files",
        "",
        "- `testing_patient_scores_paired.csv`: one row per test patient, both ensemble rankings side by side",
        "- `testing_patient_scores_long.csv`: one row per arm/patient",
        "- `per_seed_actual_autoseg_scores.csv`: raw per-seed real-autoseg QA signals",
        "- `per_seed_clean_approved_scores.csv`: raw per-seed approved-contour identity signals",
        "- `inspection_shortlist.csv`: high-signal and arm-discordant cases for initial visual review",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote experiment analysis to {out}")


if __name__ == "__main__":
    main()
