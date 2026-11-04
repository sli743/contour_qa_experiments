#!/usr/bin/env python3
"""Generate a manuscript-friendly Markdown report for actual-autoseg ranking/classification.

The report intentionally restricts the main tables to *corresponding metrics*:
QA 1-Dice -> true 1-Dice, QA HD95 -> true HD95, QA ASSD -> true ASSD,
and QA relative-volume difference -> true relative-volume difference.

Ranking and classification are reported separately. The default primary condition is
candidate-derived (autoseg-centered) localization on testing1. Classification uses the
P75 high-error threshold learned from training1+validation1, matching the analysis code.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["1_minus_dice", "hd95", "assd", "relative_volume_difference"]
METRIC_LABEL = {
    "1_minus_dice": "1-Dice",
    "hd95": "HD95",
    "assd": "ASSD",
    "relative_volume_difference": "RelVol",
}


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def _fmt(v, digits: int = 3) -> str:
    if pd.isna(v):
        return "NA"
    return f"{float(v):.{digits}f}"


def _fmt_auc_cell(row: pd.Series) -> str:
    if pd.isna(row["roc_auc_testing1"]):
        return "NA"
    mark = "" if bool(row["reliable_roc_min_class_n"]) else "†"
    return f"{row['roc_auc_testing1']:.3f}{mark}"


def _markdown_table(df: pd.DataFrame, columns: List[str], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_files = sorted((root / "ranking_classification").glob("*/ranking_summary.csv"))
    cls_files = sorted((root / "ranking_classification").glob("*/classification_summary.csv"))
    if not rank_files or not cls_files:
        raise RuntimeError(f"Missing per-ROI results under {root / 'ranking_classification'}")
    ranking = pd.concat([pd.read_csv(f) for f in rank_files], ignore_index=True)
    classification = pd.concat([pd.read_csv(f) for f in cls_files], ignore_index=True)
    classification["reliable_roc_min_class_n"] = _bool_series(classification["reliable_roc_min_class_n"])
    return ranking, classification


def corresponding_ranking(ranking: pd.DataFrame, localization_mode: str, subset: str) -> pd.DataFrame:
    x = ranking[
        (ranking["localization_mode"] == localization_mode)
        & (ranking["subset"] == subset)
        & (ranking["qa_metric"] == ranking["true_metric"])
        & ranking["qa_metric"].isin(METRICS)
    ].copy()
    return x


def corresponding_classification(
    classification: pd.DataFrame, localization_mode: str, percentile: float
) -> pd.DataFrame:
    x = classification[
        (classification["localization_mode"] == localization_mode)
        & (classification["true_error_percentile_from_development"] == percentile)
        & (classification["qa_metric"] == classification["true_metric"])
        & classification["qa_metric"].isin(METRICS)
    ].copy()
    return x


def make_auc_figure(rows: pd.DataFrame, metric: str, out_path: Path) -> None:
    x = rows[rows["qa_metric"] == metric].copy()
    x = x[np.isfinite(pd.to_numeric(x["roc_auc_testing1"], errors="coerce"))].copy()
    if x.empty:
        return
    x = x.sort_values("roc_auc_testing1", ascending=True)
    y = np.arange(len(x))
    auc = x["roc_auc_testing1"].to_numpy(float)
    low = x["roc_auc_ci95_low"].to_numpy(float)
    high = x["roc_auc_ci95_high"].to_numpy(float)
    lower = np.maximum(0.0, auc - low)
    upper = np.maximum(0.0, high - auc)

    fig_h = max(4.5, 0.34 * len(x) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    ax.errorbar(auc, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=2)
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(x["roi"].tolist())
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("ROC AUC (95% bootstrap CI)")
    ax.set_title(f"P75 high-error classification: {METRIC_LABEL[metric]}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="actual_autoseg_regression root")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--localization-mode", default="autoseg_centered", choices=["autoseg_centered", "approved_reference"])
    p.add_argument("--ranking-subset", default="testing1")
    p.add_argument("--classification-percentile", type=float, default=75.0)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "ranking_classification_report"
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ranking, classification = load_results(root)
    rank = corresponding_ranking(ranking, args.localization_mode, args.ranking_subset)
    cls = corresponding_classification(classification, args.localization_mode, args.classification_percentile)
    if rank.empty or cls.empty:
        raise RuntimeError("No corresponding-metric rows matched requested analysis condition")

    # Save the exact source tables used by the report.
    rank.to_csv(out / "corresponding_metric_ranking.csv", index=False)
    cls.to_csv(out / "corresponding_metric_classification_p75.csv", index=False)

    # Per-ROI compact ranking table.
    rp = rank.pivot(index="roi", columns="qa_metric", values="spearman_rho").reindex(columns=METRICS)
    rp = rp.reset_index().sort_values("roi")
    rank_table = pd.DataFrame({"ROI": rp["roi"]})
    for m in METRICS:
        rank_table[METRIC_LABEL[m]] = rp[m].map(lambda v: _fmt(v))

    # Per-ROI compact classification table.
    cls_map: Dict[tuple[str, str], pd.Series] = {(str(r.roi), str(r.qa_metric)): r for _, r in cls.iterrows()}
    rois = sorted(set(rank["roi"]).union(cls["roi"]))
    cls_table = pd.DataFrame({"ROI": rois})
    for m in METRICS:
        cls_table[METRIC_LABEL[m]] = [
            _fmt_auc_cell(cls_map[(roi, m)]) if (roi, m) in cls_map else "NA" for roi in rois
        ]

    # Cross-ROI summaries. Classification is reported both for all computable AUCs and the stricter reliable subset.
    rank_summary_rows = []
    cls_summary_rows = []
    for m in METRICS:
        xr = rank[rank["qa_metric"] == m]
        xc = cls[cls["qa_metric"] == m]
        xc_valid = xc[np.isfinite(pd.to_numeric(xc["roc_auc_testing1"], errors="coerce"))]
        xc_rel = xc_valid[xc_valid["reliable_roc_min_class_n"]]
        rank_summary_rows.append({
            "Metric": METRIC_LABEL[m],
            "Median Spearman": _fmt(xr["spearman_rho"].median()),
            "Positive ROIs": f"{int((xr['spearman_rho'] > 0).sum())}/{len(xr)}",
            "ρ ≥ 0.3": f"{int((xr['spearman_rho'] >= 0.3).sum())}/{len(xr)}",
            "Median C-index": _fmt(xr["concordance_index"].median()),
            "Median top-25% recall": _fmt(xr["top_25pct_recall_at_k"].median()),
            "Median enrichment": _fmt(xr["top_25pct_enrichment_over_random"].median()),
        })
        cls_summary_rows.append({
            "Metric": METRIC_LABEL[m],
            "Computable ROIs": f"{len(xc_valid)}/{len(xc)}",
            "Median AUC": _fmt(xc_valid["roc_auc_testing1"].median()),
            "AUC ≥ 0.7": f"{int((xc_valid['roc_auc_testing1'] >= 0.7).sum())}/{len(xc_valid)}" if len(xc_valid) else "NA",
            "Reliable ROIs": f"{len(xc_rel)}/{len(xc)}",
            "Median AUC, reliable": _fmt(xc_rel["roc_auc_testing1"].median()) if len(xc_rel) else "NA",
        })
    rank_summary = pd.DataFrame(rank_summary_rows)
    cls_summary = pd.DataFrame(cls_summary_rows)

    # Figures: one corresponding-metric P75 AUC figure per metric.
    figure_links = []
    for m in METRICS:
        name = f"p75_auc_{m}.png"
        path = fig_dir / name
        make_auc_figure(cls, m, path)
        if path.exists():
            figure_links.append((METRIC_LABEL[m], f"figures/{name}"))

    md: List[str] = []
    md.append("# Actual-autoseg NSR-QA ranking and high-error classification summary")
    md.append("")
    md.append(
        f"Primary condition: `{args.localization_mode}` localization; ranking evaluated on `{args.ranking_subset}`. "
        f"Only corresponding metric pairs are shown (the same metric is used for the NSR-QA signal and true autoseg error)."
    )
    md.append("")
    md.append("## Ranking")
    md.append("")
    md.append("The table reports held-out Spearman rank correlation between each NSR-QA disagreement metric and the corresponding true autoseg error metric.")
    md.append("")
    md.append(_markdown_table(rank_summary, list(rank_summary.columns), list(rank_summary.columns)))
    md.append("")
    md.append("### Ranking by ROI (Spearman ρ)")
    md.append("")
    md.append(_markdown_table(rank_table, list(rank_table.columns), list(rank_table.columns)))
    md.append("")
    md.append("## Classification")
    md.append("")
    md.append(
        f"High error is defined by the {args.classification_percentile:g}th percentile of the corresponding true-error metric in the development cohort (`training1 + validation1`); the threshold is then applied unchanged to `testing1`. "
        "ROC AUC uses the corresponding raw NSR-QA disagreement metric as the score."
    )
    md.append("")
    md.append(_markdown_table(cls_summary, list(cls_summary.columns), list(cls_summary.columns)))
    md.append("")
    md.append("### Classification by ROI (ROC AUC)")
    md.append("")
    md.append(_markdown_table(cls_table, list(cls_table.columns), list(cls_table.columns)))
    md.append("")
    md.append("† Fewer than four positive or four negative testing cases; interpret the AUC as highly uncertain. `NA` means the P75 development threshold produced only one class in testing, so ROC AUC is undefined.")
    md.append("")
    md.append("### AUC figures")
    md.append("")
    for label, rel in figure_links:
        md.append(f"#### {label}")
        md.append("")
        md.append(f"![{label} P75 ROC AUC by ROI]({rel})")
        md.append("")
    md.append("## Interpretation guardrails")
    md.append("")
    md.append("- Ranking and classification are separate endpoints: rank correlation measures ordering of error severity, while ROC AUC measures discrimination of a development-defined high-error group.")
    md.append("- P75 is a distribution-defined high-error endpoint, not a clinically validated failure threshold.")
    md.append("- The main report intentionally does not select cross-metric predictor/target pairs post hoc.")
    md.append("- Very small positive/negative test counts can produce extreme AUCs (including 0 or 1); use the reliability marker and confidence intervals when interpreting individual ROIs.")
    md.append("")

    report_path = out / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
