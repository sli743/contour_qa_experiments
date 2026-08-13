#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from analysis.generate_four_model_per_probe_reports import (
    METRICS,
    METRIC_LABELS,
    patient_cluster_bootstrap_auc,
    clean_threshold_and_sensitivity,
    cliffs_delta,
    common_probe_types,
    criterion_for_probe,
    dedup_clean,
    is_clean,
    load_models,
    normalize_probe_name,
    parse_models,
    sensitivity_at_specificity,
)


def build_all_probe_dataset(
    obj: dict[str, object],
    probes: list[str],
    volume_threshold_pct: float,
    local_depth_threshold_mm: float,
) -> pd.DataFrame:
    """Pool clean + all supported non-shift probes using row-wise two-criterion labels."""
    df = obj["df"]
    probe_col = obj["probe_col"]
    patient_col = obj["patient_col"]

    clean = dedup_clean(df, probe_col, patient_col).copy()
    clean["__is_clean"] = True
    clean["__family"] = "clean"
    clean["__severity"] = 0.0
    clean["__threshold"] = 1.0
    clean["__severity_ratio"] = 0.0
    clean["__label"] = 0

    pieces = [clean]
    for probe in probes:
        criterion = criterion_for_probe(probe, volume_threshold_pct, local_depth_threshold_mm)
        sub = df.loc[df["__probe_norm"].eq(normalize_probe_name(probe))].copy()
        if sub.empty:
            continue
        severity_col = str(criterion["severity_col"])
        if severity_col not in sub.columns:
            raise ValueError(f"Missing severity column {severity_col!r} for probe {probe!r}")
        sub["__is_clean"] = False
        sub["__family"] = str(criterion["family"])
        sub["__severity"] = pd.to_numeric(sub[severity_col], errors="coerce")
        sub["__threshold"] = float(criterion["threshold"])
        sub["__severity_ratio"] = sub["__severity"] / float(criterion["threshold"])
        sub["__label"] = (sub["__severity"] > float(criterion["threshold"])).astype(int)
        pieces.append(sub)

    return pd.concat(pieces, ignore_index=True, sort=False)


def macro_probe_auc(ds: pd.DataFrame, score_col: str) -> tuple[float, int]:
    vals = []
    for probe, sub in ds.loc[~ds["__is_clean"]].groupby("__probe_norm"):
        # For a per-probe AUC, include the same clean reference set as negatives.
        clean = ds.loc[ds["__is_clean"]]
        tmp = pd.concat([clean, sub], ignore_index=True)
        y = pd.to_numeric(tmp["__label"], errors="coerce")
        s = pd.to_numeric(tmp[score_col], errors="coerce")
        valid = y.notna() & s.notna() & np.isfinite(s)
        yy = y.loc[valid].astype(int).to_numpy()
        ss = s.loc[valid].to_numpy(float)
        if np.unique(yy).size != 2:
            continue
        fpr, tpr, _ = roc_curve(yy, ss)
        vals.append(float(auc(fpr, tpr)))
    return (float(np.mean(vals)) if vals else np.nan, len(vals))


def plot_roc(payload: list[dict[str, object]], metric: str, out_path: Path, roi: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for row in payload:
        y = row["y"]
        score = row["score"]
        if np.unique(y).size != 2:
            continue
        fpr, tpr, _ = roc_curve(y, score)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.1, label=f"{row['model']} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.1, label="Chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"{roi}: all supported probes pooled\n{METRIC_LABELS[metric]}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_signal_vs_normalized_severity(
    datasets: dict[str, pd.DataFrame], metric: str, out_path: Path, roi: str
) -> None:
    """Plot detector signal vs severity normalized by the relevant bad threshold.

    x=1 is the common decision boundary:
      global: |volume change| / global volume threshold
      local:  HD95(input,target) / local HD95 threshold
    """
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    bins = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf])

    for name, ds in datasets.items():
        corr = ds.loc[~ds["__is_clean"]].copy()
        x = pd.to_numeric(corr["__severity_ratio"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(corr[f"__score__{metric}"], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size == 0:
            continue
        ax.scatter(x, y, s=10, alpha=0.10)

        temp = pd.DataFrame({"x": x, "y": y})
        temp["bin"] = pd.cut(temp["x"], bins=bins, right=False, include_lowest=True)
        med = temp.groupby("bin", observed=True).agg(x=("x", "median"), y=("y", "median"), n=("y", "size"))
        med = med.loc[med["n"] > 0].sort_values("x")
        if not med.empty:
            ax.plot(med["x"], med["y"], marker="o", markersize=4, linewidth=1.8, label=name)

    ax.axvline(1.0, linestyle="--", linewidth=1.6, label="Good/bad separation")
    ax.text(1.0, 0.99, " good  |  bad ", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.set_xlabel("Normalized defect severity (1 = bad-contour threshold)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{roi}: detector signal vs normalized defect severity\nall local + global probes pooled")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_auc_comparison(summary: pd.DataFrame, out_path: Path, roi: str) -> None:
    metrics = list(METRICS)
    models = list(summary["model"].drop_duplicates())
    x = np.arange(len(metrics), dtype=float)
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for i, model in enumerate(models):
        sub = summary.loc[summary["model"].eq(model)].set_index("metric")
        vals = [sub.loc[m, "auc"] if m in sub.index else np.nan for m in metrics]
        ax.bar(x + (i - (len(models)-1)/2)*width, vals, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=20, ha="right")
    ax.set_ylabel("Pooled ROC AUC")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{roi}: four-model comparison across all supported probes")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Four-model pooled-probe QA comparison using separate local/global bad-contour criteria")
    ap.add_argument("--model", action="append", required=True, help="NAME=CSV; repeat four times")
    ap.add_argument("--roi", required=True)
    ap.add_argument("--volume-threshold-pct", type=float, required=True)
    ap.add_argument("--local-depth-threshold-mm", type=float, required=True,
                    help="Local criterion uses each model row's HD95(input,target) in mm")
    ap.add_argument("--exclude-shifts", action="store_true", default=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    models = parse_models(args.model)
    loaded = load_models(models)
    probes = common_probe_types(loaded, exclude_shifts=True)
    if not probes:
        raise RuntimeError("No common non-shift probe types found across the four model CSVs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        name: build_all_probe_dataset(obj, probes, args.volume_threshold_pct, args.local_depth_threshold_mm)
        for name, obj in loaded.items()
    }

    summary_rows = []
    family_rows = []
    roc_payload = {m: [] for m in METRICS}

    for model_i, (model_name, _) in enumerate(models):
        ds = datasets[model_name]
        corr = ds.loc[~ds["__is_clean"]]
        for family in ("global", "local"):
            fam = corr.loc[corr["__family"].eq(family)]
            family_rows.append({
                "roi": args.roi, "model": model_name, "family": family,
                "n": int(len(fam)), "n_bad": int(pd.to_numeric(fam["__label"], errors="coerce").fillna(0).sum()),
                "n_good": int((pd.to_numeric(fam["__label"], errors="coerce").fillna(0) == 0).sum()),
            })

        for metric_i, metric in enumerate(METRICS):
            score_col = f"__score__{metric}"
            y = pd.to_numeric(ds["__label"], errors="coerce")
            s = pd.to_numeric(ds[score_col], errors="coerce")
            valid = y.notna() & s.notna() & np.isfinite(s)
            yy = y.loc[valid].astype(int).to_numpy()
            ss = s.loc[valid].to_numpy(float)

            row = {
                "roi": args.roi, "model": model_name, "metric": metric,
                "n_total": int(valid.sum()), "n_positive": int(yy.sum()), "n_negative": int((yy == 0).sum()),
                "n_clean": int(ds.loc[ds["__is_clean"]].shape[0]),
                "n_corrupted": int(corr.shape[0]),
            }
            if np.unique(yy).size == 2:
                fpr, tpr, _ = roc_curve(yy, ss)
                row["auc"] = float(auc(fpr, tpr))
                rng = np.random.default_rng(args.seed + model_i * 100 + metric_i)
                row["auc_ci_low"], row["auc_ci_high"] = patient_cluster_bootstrap_auc(
                    ds.loc[valid], "__label", score_col, args.n_boot, rng
                )
                row["sens_at_95spec"], row["achieved_specificity"], row["roc_threshold_95spec"] = sensitivity_at_specificity(yy, ss, 0.95)
                roc_payload[metric].append({"model": model_name, "y": yy, "score": ss})
            else:
                row.update({"auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
                            "sens_at_95spec": np.nan, "achieved_specificity": np.nan, "roc_threshold_95spec": np.nan})

            clean_scores = pd.to_numeric(ds.loc[ds["__is_clean"], score_col], errors="coerce").to_numpy(float)
            bad_scores = pd.to_numeric(ds.loc[(~ds["__is_clean"]) & ds["__label"].eq(1), score_col], errors="coerce").to_numpy(float)
            good_corr_scores = pd.to_numeric(ds.loc[(~ds["__is_clean"]) & ds["__label"].eq(0), score_col], errors="coerce").to_numpy(float)
            row["clean_score_median"] = float(np.nanmedian(clean_scores)) if np.isfinite(clean_scores).any() else np.nan
            row["good_corruption_score_median"] = float(np.nanmedian(good_corr_scores)) if np.isfinite(good_corr_scores).any() else np.nan
            row["bad_score_median"] = float(np.nanmedian(bad_scores)) if np.isfinite(bad_scores).any() else np.nan
            row["bad_minus_clean_median"] = row["bad_score_median"] - row["clean_score_median"] if np.isfinite(row["bad_score_median"]) and np.isfinite(row["clean_score_median"]) else np.nan
            row["bad_vs_clean_cliffs_delta"] = cliffs_delta(bad_scores, clean_scores)
            row["clean_95pct_threshold"], row["bad_sensitivity_at_clean95"] = clean_threshold_and_sensitivity(clean_scores, bad_scores, 0.95)
            row["macro_probe_auc"], row["n_probes_with_auc"] = macro_probe_auc(ds, score_col)
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    family_summary = pd.DataFrame(family_rows)
    summary.to_csv(args.output_dir / "all_probes_metric_summary.csv", index=False)
    family_summary.to_csv(args.output_dir / "family_label_counts.csv", index=False)

    row_labels = []
    for probe in probes:
        c = criterion_for_probe(probe, args.volume_threshold_pct, args.local_depth_threshold_mm)
        row_labels.append({"probe_type": probe, "family": c["family"], "severity_measure": c["severity_label"],
                           "threshold": c["threshold"], "bad_if": c["criterion_text"]})
    pd.DataFrame(row_labels).to_csv(args.output_dir / "probe_criteria.csv", index=False)

    for metric in METRICS:
        plot_roc(roc_payload[metric], metric, args.output_dir / f"roc_all_probes_{metric}.png", args.roi)
        plot_signal_vs_normalized_severity(datasets, metric, args.output_dir / f"signal_vs_normalized_defect_size_{metric}.png", args.roi)
    plot_auc_comparison(summary, args.output_dir / "auc_comparison_all_probes.png", args.roi)

    def f3(x):
        return "NA" if not np.isfinite(x) else f"{x:.3f}"

    md = [
        f"# {args.roi} — four-model comparison across all probes", "",
        "## Analysis definition", "",
        "This report pools all supported non-shift probe types into one QA analysis while retaining the two corruption-family criteria.", "",
        f"- **Global probes:** bad if absolute input-target volume change > **{args.volume_threshold_pct:g}%**.",
        f"- **Local probes (`big_bite`, `big_blob`):** bad if HD95(input,target) > **{args.local_depth_threshold_mm:g} mm**.",
        "- **Shift probes are excluded** because neither volume-change nor local-depth labeling is appropriate for pure translation.",
        "- Clean contours are included once per patient, not once per probe type.",
        "- The pooled ROC therefore asks whether one detector score separates all substantial supported defects from clean and below-threshold contours.",
        "- Repair accuracy is not the endpoint; larger input-reconstruction disagreement for bad contours is the desired signal.",
        f"- AUC confidence intervals use a **patient-cluster bootstrap** ({args.n_boot} replicates): patients are sampled with replacement and all probe rows from each sampled patient are retained together.", "",
        "## Good/bad counts by family", "",
        "| Model | Family | n | bad | good/below-threshold |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in family_summary.iterrows():
        md.append(f"| {r['model']} | {r['family']} | {int(r['n'])} | {int(r['n_bad'])} | {int(r['n_good'])} |")

    md += ["", "## Pooled detector performance", ""]
    for metric in METRICS:
        md += [f"### {METRIC_LABELS[metric]}", "",
               "| Model | Pooled AUC (95% patient-cluster bootstrap CI) | Sens @95% spec | Bad sens @ clean-95% threshold | Bad-clean signal lift | Cliff's delta | Mean per-probe AUC |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        sub = summary.loc[summary["metric"].eq(metric)]
        for _, r in sub.iterrows():
            auc_txt = "NA" if not np.isfinite(r["auc"]) else f"{r['auc']:.3f} [{f3(r['auc_ci_low'])}, {f3(r['auc_ci_high'])}]"
            md.append(
                f"| {r['model']} | {auc_txt} | {f3(r['sens_at_95spec'])} | {f3(r['bad_sensitivity_at_clean95'])} | "
                f"{f3(r['bad_minus_clean_median'])} | {f3(r['bad_vs_clean_cliffs_delta'])} | {f3(r['macro_probe_auc'])} |"
            )
        md += ["", f"![ROC](roc_all_probes_{metric}.png)", "",
               f"![Signal versus normalized defect size](signal_vs_normalized_defect_size_{metric}.png)", ""]

    md += [
        "## Reading the normalized defect-size plots", "",
        "The x-axis expresses each corruption in units of its own bad-contour threshold. Therefore **x = 1** is the common separation line:", "",
        f"- for global probes, `x = |volume change| / {args.volume_threshold_pct:g}%`;",
        f"- for local probes, `x = HD95(input,target) / {args.local_depth_threshold_mm:g} mm`.", "",
        "This allows local and global corruptions to appear on one intuitive plot without pretending that percent volume and millimeters are the same physical quantity.", "",
        "## Caveat", "",
        "For historical random `big_bite`/`big_blob` evaluations, each model may contain a different realized local probe geometry. Local good/bad labels are therefore computed from each model row's own HD95(input,target), not borrowed from another model's realization.", "",
        "![AUC comparison](auc_comparison_all_probes.png)", "",
    ]
    (args.output_dir / "all_probes_four_model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    metadata = {
        "roi": args.roi,
        "models": {name: str(path) for name, path in models},
        "included_probes": probes,
        "excluded_shifts": True,
        "global_volume_threshold_pct": args.volume_threshold_pct,
        "local_hd95_threshold_mm": args.local_depth_threshold_mm,
        "normalized_severity_boundary": 1.0,
        "auc_ci_bootstrap_unit": "patient",
        "auc_ci_n_boot": args.n_boot,
        "auc_ci_seed": args.seed,
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote pooled four-model report: {args.output_dir / 'all_probes_four_model_comparison.md'}")


if __name__ == "__main__":
    main()
