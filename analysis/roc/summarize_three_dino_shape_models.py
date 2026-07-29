#!/usr/bin/env python3
"""Combine pairwise ROC outputs into one three-model summary report."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot_random_dir", required=True)
    p.add_argument("--benchmark_random_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    a = Path(args.pilot_random_dir)
    b = Path(args.benchmark_random_dir)
    out = Path(args.out_dir)
    tables = out / "tables"
    report = out / "report"
    tables.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)

    roc_a = pd.read_csv(a / "tables/roc_summary_models.csv")
    roc_b = pd.read_csv(b / "tables/roc_summary_models.csv")
    roc = pd.concat([roc_a, roc_b], ignore_index=True)
    roc = roc.drop_duplicates(["model", "metric_display"], keep="first")
    rename = {
        "Denoising": "Random-morphology DAE",
        "Pilot without denoising": "Pilot without denoising",
    }
    # In the second pairwise run, the benchmark-aware model is passed as pilot.
    source_b_models = set(roc_b["model"].astype(str))
    if "Pilot without denoising" in source_b_models:
        mask = roc.index.isin(roc_b.index)  # harmless fallback; corrected below by source concat
    roc["model"] = roc["model"].replace(rename)

    # Reconstruct explicitly to avoid ambiguous pairwise labels.
    pilot = roc_a[roc_a["model"] == "Pilot without denoising"].copy()
    pilot["model"] = "Pilot without denoising"
    random_a = roc_a[roc_a["model"] == "Denoising"].copy()
    random_a["model"] = "Random-morphology DAE"
    benchmark = roc_b[roc_b["model"] == "Pilot without denoising"].copy()
    benchmark["model"] = "Probe-trained DAE (upper limit)"
    combined = pd.concat([pilot, benchmark, random_a], ignore_index=True)
    combined.to_csv(tables / "roc_summary_three_models.csv", index=False)

    best = (combined.sort_values("auc", ascending=False)
            .groupby("model", as_index=False).first())
    cols = [c for c in ["model", "metric_display", "auc", "auc_ci_low", "auc_ci_high",
                           "sens_at_95spec", "threshold_at_95spec"] if c in best.columns]

    md = []
    md += ["# Frozen-3DINO contour DAE: three-model ROC comparison", ""]
    md += ["## Scientific roles", ""]
    md += ["- **Pilot without denoising:** identity-reconstruction baseline.",
           "- **Probe-trained DAE:** benchmark-aware upper limit; its training corruption families overlap the fixed evaluation probes.",
           "- **Random-morphology DAE:** proposed leakage-resistant model trained only with random erosion/dilation, translation, and local ball-shaped additions/removals.", ""]
    md += ["## Best ROC detector per model", "", best[cols].to_markdown(index=False, floatfmt=".3f"), ""]
    md += ["## Full ROC summary", "", combined.to_markdown(index=False, floatfmt=".3f"), ""]
    md += ["## Interpretation", "",
           "The probe-trained DAE should be interpreted as an approximate upper limit on this synthetic benchmark, not as the primary generalization result. The random-morphology DAE is evaluated on the same fixed probes but is not trained on the benchmark half-space cuts or fixed severity grid.", ""]
    (report / "three_model_ROC_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote three-model summary to: {out}")


if __name__ == "__main__":
    main()
