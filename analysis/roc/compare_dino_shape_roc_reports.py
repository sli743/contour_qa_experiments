#!/usr/bin/env python3
"""Reproducible ROC reports for frozen-3DINO contour autoencoders.

Compares two probe-sweep CSVs generated with the same held-out cases and
corruption protocol:
  1. pilot frozen-3DINO decoder trained without denoising
  2. frozen-3DINO denoising decoder

The script produces a self-documenting report for each model and a paired
comparison report. Confidence intervals use patient-cluster bootstrap by
default so repeated probes from one patient are resampled together.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

METRICS = [
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
    "abs_input_recon_volume_diff_frac",
]
DISPLAY = {
    "one_minus_dice_input_vs_recon": "1-Dice(input,recon)",
    "hd95_input_vs_recon": "HD95(input,recon)",
    "assd_input_vs_recon": "ASSD(input,recon)",
    "abs_input_recon_volume_diff_frac": "Volume diff(input,recon)",
}
PROBE_ORDER = [
    "clean", "big_bite", "big_blob", "cut_upper_half", "cut_lower_half",
    "cut_posterior_half", "dilated", "eroded", "shifted",
]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_") or "value"


def finite_number(value, default=np.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(p)
    return cfg


def checkpoint_from_config(cfg: Dict) -> str:
    value = cfg.get("paths", {}).get("trained_decoder_checkpoint")
    if value:
        return str(value)
    if cfg.get("output_root") and cfg.get("experiment_name"):
        return str(Path(cfg["output_root"]) / cfg["experiment_name"] / "checkpoints/best_decoder.pth")
    return "not provided"


def dino_checkpoint_from_config(cfg: Dict) -> str:
    return str(cfg.get("dino", {}).get("checkpoint_path", "not provided"))


def prepare_dataframe(csv_path: str, model_name: str, roi: str) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    required = {"patient_id", "probe_type", "input_voxels", "target_voxels", "recon_voxels"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = df.copy()
    out["source_csv"] = str(path)
    out["model"] = model_name
    out["patient_id"] = out["patient_id"].astype(str)
    if "roi" not in out.columns:
        out["roi"] = roi

    for col in [
        "level", "requested_level", "applied_level", "input_voxels", "target_voxels",
        "recon_voxels", "dice_input_vs_target", "dice_input_vs_recon",
        "dice_recon_vs_target", "hd95_input_vs_target", "hd95_input_vs_recon",
        "hd95_recon_vs_target", "assd_input_vs_target", "assd_input_vs_recon",
        "assd_recon_vs_target",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    level_col = next((c for c in ["requested_level", "level", "applied_level"] if c in out.columns), None)
    out["requested_level_numeric"] = out[level_col].fillna(0.0) if level_col else 0.0
    out["is_clean"] = (out["probe_type"].astype(str) == "clean") | (out["requested_level_numeric"] == 0)

    if "one_minus_dice_input_vs_recon" not in out.columns:
        out["one_minus_dice_input_vs_recon"] = 1.0 - out["dice_input_vs_recon"]
    target_denom = out["target_voxels"].replace(0, np.nan)
    input_denom = out["input_voxels"].replace(0, np.nan)
    out["actual_volume_change_pct"] = 100.0 * ((out["input_voxels"] - out["target_voxels"]) / target_denom).abs()
    out.loc[out["is_clean"], "actual_volume_change_pct"] = 0.0
    out["abs_input_recon_volume_diff_frac"] = ((out["input_voxels"] - out["recon_voxels"]) / input_denom).abs()

    if "dice_correction_gain" not in out.columns and {"dice_recon_vs_target", "dice_input_vs_target"}.issubset(out.columns):
        out["dice_correction_gain"] = out["dice_recon_vs_target"] - out["dice_input_vs_target"]
    if "assd_improvement" not in out.columns and {"assd_input_vs_target", "assd_recon_vs_target"}.issubset(out.columns):
        out["assd_improvement"] = out["assd_input_vs_target"] - out["assd_recon_vs_target"]
    if "hd95_improvement" not in out.columns and {"hd95_input_vs_target", "hd95_recon_vs_target"}.issubset(out.columns):
        out["hd95_improvement"] = out["hd95_input_vs_target"] - out["hd95_recon_vs_target"]
    return out


def build_dataset(
    df: pd.DataFrame,
    volume_threshold_pct: float,
    include_shift: bool,
    clean_policy: str,
    probe_types: Optional[Sequence[str]],
) -> pd.DataFrame:
    work = df.copy()
    if not include_shift:
        work = work[work["probe_type"].astype(str) != "shifted"].copy()
    if probe_types:
        allowed = set(probe_types) | {"clean"}
        work = work[work["probe_type"].astype(str).isin(allowed)].copy()

    clean = work[work["is_clean"]].copy()
    if clean_policy == "one_per_patient":
        # Every level-zero probe maps to the same clean contour/reconstruction.
        # Keep one row per patient to avoid duplicated negative samples.
        clean = clean.sort_values(["patient_id", "probe_type"]).drop_duplicates("patient_id", keep="first")
    elif clean_policy != "all_level_zero":
        raise ValueError(f"Unknown clean policy: {clean_policy}")

    corrupt = work[(~work["is_clean"]) & (work["actual_volume_change_pct"] > volume_threshold_pct)].copy()
    clean["y_true"] = 0
    clean["class_label"] = "clean"
    corrupt["y_true"] = 1
    corrupt["class_label"] = f"corrupt_volume_gt_{volume_threshold_pct:g}pct"
    dataset = pd.concat([clean, corrupt], ignore_index=True)
    if dataset["y_true"].nunique() != 2:
        raise ValueError(f"Both classes are required. Counts: {dataset['y_true'].value_counts().to_dict()}")
    return dataset


def auc_pairwise(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def roc_curve_manual(y: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y, scores = y[order], scores[order]
    p, n = int(np.sum(y == 1)), int(np.sum(y == 0))
    if p == 0 or n == 0:
        raise ValueError("ROC requires both classes")
    distinct_end = np.r_[scores[1:] != scores[:-1], True]
    idx = np.flatnonzero(distinct_end)
    tp = np.cumsum(y == 1)[idx]
    fp = np.cumsum(y == 0)[idx]
    return (
        np.r_[0.0, fp / n, 1.0],
        np.r_[0.0, tp / p, 1.0],
        np.r_[np.inf, scores[idx], -np.inf],
    )


def operating_points(fpr, tpr, thresholds, specificities=(0.90, 0.95), sensitivities=(0.90, 0.95)) -> Dict[str, float]:
    spec = 1.0 - fpr
    result: Dict[str, float] = {}
    j = int(np.nanargmax(tpr - fpr))
    result.update(
        youden_threshold=float(thresholds[j]),
        youden_sensitivity=float(tpr[j]),
        youden_specificity=float(spec[j]),
    )
    for target in specificities:
        valid = np.flatnonzero(spec >= target)
        key = int(round(100 * target))
        if len(valid):
            i = valid[np.argmax(tpr[valid])]
            result[f"sens_at_{key}spec"] = float(tpr[i])
            result[f"threshold_at_{key}spec"] = float(thresholds[i])
            result[f"actual_spec_at_{key}spec"] = float(spec[i])
        else:
            result[f"sens_at_{key}spec"] = np.nan
            result[f"threshold_at_{key}spec"] = np.nan
            result[f"actual_spec_at_{key}spec"] = np.nan
    for target in sensitivities:
        valid = np.flatnonzero(tpr >= target)
        key = int(round(100 * target))
        if len(valid):
            i = valid[np.argmax(spec[valid])]
            result[f"spec_at_{key}sens"] = float(spec[i])
            result[f"threshold_at_{key}sens"] = float(thresholds[i])
            result[f"actual_sens_at_{key}sens"] = float(tpr[i])
        else:
            result[f"spec_at_{key}sens"] = np.nan
            result[f"threshold_at_{key}sens"] = np.nan
            result[f"actual_sens_at_{key}sens"] = np.nan
    return result


def patient_bootstrap_auc(
    dataset: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float, np.ndarray]:
    sub = dataset[["patient_id", "y_true", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    patients = np.asarray(sorted(sub["patient_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        frames = []
        for draw_index, patient in enumerate(sampled):
            block = sub[sub["patient_id"].astype(str) == patient].copy()
            block["bootstrap_patient"] = f"{patient}__draw{draw_index}"
            frames.append(block)
        boot = pd.concat(frames, ignore_index=True)
        if boot["y_true"].nunique() < 2:
            continue
        values.append(auc_pairwise(boot["y_true"].to_numpy(), boot[metric].to_numpy()))
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, arr
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), arr


def analyze_model(dataset: pd.DataFrame, metrics: Sequence[str], n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, curve_rows = [], []
    for metric_index, metric in enumerate(metrics):
        if metric not in dataset.columns:
            continue
        sub = dataset[["patient_id", "y_true", metric]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub["y_true"].nunique() < 2:
            continue
        y = sub["y_true"].astype(int).to_numpy()
        scores = sub[metric].astype(float).to_numpy()
        fpr, tpr, thresholds = roc_curve_manual(y, scores)
        auc = auc_pairwise(y, scores)
        boot_mean, ci_low, ci_high, _ = patient_bootstrap_auc(dataset, metric, n_boot, seed + metric_index)
        row = {
            "model": str(dataset["model"].iloc[0]),
            "metric": metric,
            "metric_display": DISPLAY.get(metric, metric),
            "n_patients": int(dataset["patient_id"].nunique()),
            "n_clean": int(np.sum(y == 0)),
            "n_corrupt": int(np.sum(y == 1)),
            "auc": auc,
            "patient_bootstrap_mean_auc": boot_mean,
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            **operating_points(fpr, tpr, thresholds),
        }
        summary_rows.append(row)
        for x, yy, threshold in zip(fpr, tpr, thresholds):
            curve_rows.append({
                "model": row["model"], "metric": metric, "metric_display": row["metric_display"],
                "fpr": float(x), "specificity": float(1.0 - x), "sensitivity": float(yy),
                "threshold": float(threshold) if np.isfinite(threshold) else threshold,
            })
    summary = pd.DataFrame(summary_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    if not summary.empty:
        summary["rank_by_auc"] = np.arange(1, len(summary) + 1)
    return summary, pd.DataFrame(curve_rows)


def paired_bootstrap_auc_difference(
    pilot: pd.DataFrame,
    denoise: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    # Pair by patient/probe/level; the bootstrap unit remains patient.
    keys = ["patient_id", "probe_type", "requested_level_numeric", "y_true"]
    left = pilot[keys + [metric]].rename(columns={metric: "pilot_score"})
    right = denoise[keys + [metric]].rename(columns={metric: "denoise_score"})
    paired = left.merge(right, on=keys, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    patients = np.asarray(sorted(paired["patient_id"].astype(str).unique()))
    if len(patients) == 0:
        return {"n_paired_rows": 0, "n_paired_patients": 0, "auc_difference": np.nan,
                "difference_ci_low": np.nan, "difference_ci_high": np.nan, "bootstrap_p_two_sided": np.nan}
    y = paired["y_true"].astype(int).to_numpy()
    observed = auc_pairwise(y, paired["denoise_score"].to_numpy()) - auc_pairwise(y, paired["pilot_score"].to_numpy())
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        blocks = [paired[paired["patient_id"].astype(str) == p] for p in sampled]
        boot = pd.concat(blocks, ignore_index=True)
        if boot["y_true"].nunique() < 2:
            continue
        yy = boot["y_true"].astype(int).to_numpy()
        differences.append(
            auc_pairwise(yy, boot["denoise_score"].to_numpy())
            - auc_pairwise(yy, boot["pilot_score"].to_numpy())
        )
    arr = np.asarray(differences, dtype=float)
    p_two = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)) if len(arr) else np.nan
    return {
        "n_paired_rows": int(len(paired)),
        "n_paired_patients": int(len(patients)),
        "auc_difference": float(observed),
        "difference_ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
        "difference_ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
        "bootstrap_p_two_sided": float(min(1.0, p_two)) if np.isfinite(p_two) else np.nan,
    }


def clean_calibration(dataset: pd.DataFrame) -> pd.DataFrame:
    clean = dataset[dataset["y_true"] == 0].drop_duplicates("patient_id").copy()
    metrics = ["dice_input_vs_recon", "assd_input_vs_recon", "hd95_input_vs_recon", "abs_input_recon_volume_diff_frac"]
    rows = []
    for metric in metrics:
        if metric not in clean.columns:
            continue
        values = pd.to_numeric(clean[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "model": clean["model"].iloc[0], "metric": metric, "n_patients": len(values),
            "mean": values.mean(), "median": values.median(), "std": values.std(ddof=1),
            "q025": values.quantile(0.025), "q975": values.quantile(0.975),
        })
    return pd.DataFrame(rows)


def repair_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    corrupt = dataset[dataset["y_true"] == 1].copy()
    metrics = ["dice_correction_gain", "assd_improvement", "hd95_improvement"]
    rows = []
    for probe in [p for p in PROBE_ORDER if p != "clean"] + ["ALL"]:
        group = corrupt if probe == "ALL" else corrupt[corrupt["probe_type"].astype(str) == probe]
        if group.empty:
            continue
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            rows.append({
                "model": group["model"].iloc[0], "probe_type": probe, "metric": metric,
                "n": len(values), "mean": values.mean(), "median": values.median(),
                "fraction_improved": float(np.mean(values > 0)),
            })
    return pd.DataFrame(rows)


def plot_roc(curves: pd.DataFrame, summaries: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 6.4))
    for (model, metric), group in curves.groupby(["model", "metric"], sort=False):
        auc_row = summaries[(summaries["model"] == model) & (summaries["metric"] == metric)]
        auc = auc_row["auc"].iloc[0] if len(auc_row) else np.nan
        plt.plot(group["fpr"], group["sensitivity"], linewidth=1.8,
                 label=f"{model}: {DISPLAY.get(metric, metric)} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1.0, label="Random")
    plt.xlabel("False-positive rate")
    plt.ylabel("Sensitivity")
    plt.title("Frozen-3DINO contour QA: ROC comparison")
    plt.xlim(0, 1); plt.ylim(0, 1.02); plt.grid(alpha=0.3)
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_auc_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot(index="metric_display", columns="model", values="auc")
    ax = pivot.plot(kind="bar", figsize=(8.8, 5.2))
    ax.set_ylabel("ROC AUC"); ax.set_xlabel(""); ax.set_ylim(0, 1)
    ax.set_title("AUC comparison by detector metric"); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_clean_calibration(calibration: pd.DataFrame, out_path: Path) -> None:
    subset = calibration[calibration["metric"] == "dice_input_vs_recon"]
    if subset.empty:
        return
    plt.figure(figsize=(6.6, 4.4))
    x = np.arange(len(subset))
    plt.bar(x, subset["mean"])
    plt.errorbar(x, subset["mean"], yerr=subset["std"].fillna(0), fmt="none", capsize=4)
    plt.xticks(x, subset["model"])
    plt.ylabel("Clean Dice(input,reconstruction)")
    plt.title("Clean-contour preservation")
    plt.ylim(0, 1.0); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def markdown_table(df: pd.DataFrame, columns: Sequence[str], floatfmt=".3f") -> str:
    available = [c for c in columns if c in df.columns]
    if not available or df.empty:
        return "No data available."
    return df[available].to_markdown(index=False, floatfmt=floatfmt)


def write_model_report(
    path: Path,
    model_name: str,
    cfg: Dict,
    dataset: pd.DataFrame,
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    repair: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    best = summary.iloc[0]
    probe_counts = dataset[dataset["y_true"] == 1]["probe_type"].value_counts()
    lines = [
        f"# ROC analysis report: {model_name}", "",
        "## Reproducibility metadata", "",
        f"- ROI: `{args.roi}`",
        f"- Evaluation split: `{args.split}` (held-out testing data expected)",
        f"- Unique patients: `{dataset['patient_id'].nunique()}`",
        f"- Model experiment: `{cfg.get('experiment_name', model_name)}`",
        f"- Decoder checkpoint: `{checkpoint_from_config(cfg)}`",
        f"- Frozen 3DINO checkpoint: `{dino_checkpoint_from_config(cfg)}`",
        f"- Probe CSV: `{dataset['source_csv'].iloc[0]}`",
        f"- Positive definition: corrupted contour with absolute input-vs-target volume change > `{args.volume_threshold_pct:g}%`",
        f"- Shift probes included: `{args.include_shift}`",
        f"- Clean-row policy: `{args.clean_policy}`",
        f"- Bootstrap: patient-cluster bootstrap, `{args.n_boot}` iterations, seed `{args.seed}`",
        "- All eligible corruption levels are pooled into the positive class.", "",
        "## Dataset", "",
        f"- Clean rows: `{int((dataset.y_true == 0).sum())}`",
        f"- Corrupted rows: `{int((dataset.y_true == 1).sum())}`",
        "- Corrupted rows by probe type:",
    ]
    lines += [f"  - `{probe}`: {count}" for probe, count in probe_counts.items()]
    lines += ["", "## Main ROC result", "",
              f"Best detector: **{best['metric_display']}**, AUC `{best['auc']:.3f}` "
              f"(95% patient-bootstrap CI `{best['auc_ci_low']:.3f}`–`{best['auc_ci_high']:.3f}`).",
              f"Sensitivity at approximately 95% specificity: `{best.get('sens_at_95spec', np.nan):.3f}` "
              f"at threshold `{best.get('threshold_at_95spec', np.nan):.4g}`.", "",
              "## ROC metric summary", "",
              markdown_table(summary, ["metric_display", "auc", "auc_ci_low", "auc_ci_high", "sens_at_95spec", "threshold_at_95spec", "sens_at_90spec", "spec_at_90sens", "youden_threshold", "youden_sensitivity", "youden_specificity"]),
              "", "## Clean-contour preservation", "",
              markdown_table(calibration, ["metric", "n_patients", "mean", "median", "std", "q025", "q975"]),
              "", "## Shape-repair behavior on positive corruptions", "",
              markdown_table(repair[repair["probe_type"] == "ALL"], ["metric", "n", "mean", "median", "fraction_improved"]),
              "", "## Interpretation note", "",
              "ROC discrimination and shape correction are related but distinct. A detector can achieve high AUC by producing a large input-to-reconstruction discrepancy without necessarily repairing the contour. Clean-contour preservation and correction metrics should therefore be interpreted alongside ROC AUC.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_report(
    path: Path,
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    calibration: pd.DataFrame,
    repair: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Frozen-3DINO contour autoencoder comparison", "",
        "## Study question", "",
        "Does denoising training improve detection and repair of synthetic contour errors relative to the pilot decoder trained without denoising, while preserving clean contours?", "",
        "## Matched design", "",
        f"Both models are evaluated on the same `{args.split}` split, ROI `{args.roi}`, probe definitions, severity levels, and positive-class threshold (> `{args.volume_threshold_pct:g}%` absolute volume change). AUC differences are estimated with a paired patient-cluster bootstrap.", "",
        "## ROC results", "",
        markdown_table(summary, ["model", "metric_display", "auc", "auc_ci_low", "auc_ci_high", "sens_at_95spec", "threshold_at_95spec"]),
        "", "## Paired AUC differences", "",
        "Positive values mean the denoising model has higher AUC.", "",
        markdown_table(differences, ["metric_display", "auc_difference", "difference_ci_low", "difference_ci_high", "bootstrap_p_two_sided", "n_paired_patients", "n_paired_rows"]),
        "", "## Clean-contour preservation", "",
        markdown_table(calibration, ["model", "metric", "n_patients", "mean", "median", "std"]),
        "", "## Repair behavior", "",
        markdown_table(repair[repair["probe_type"] == "ALL"], ["model", "metric", "n", "mean", "median", "fraction_improved"]),
        "", "## Figures", "",
        "![ROC comparison](../figures/roc_comparison.png)", "",
        "![AUC comparison](../figures/auc_comparison.png)", "",
        "![Clean preservation](../figures/clean_preservation.png)", "",
        "## Statistical caveat", "",
        "Synthetic probe rows from one patient are correlated. This report therefore resamples patients rather than individual rows. The analysis still measures performance on the specified synthetic corruption benchmark; it does not by itself establish performance on naturally occurring clinical autosegmentation errors.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_csv", required=True)
    parser.add_argument("--denoising_csv", required=True)
    parser.add_argument("--pilot_config", default=None)
    parser.add_argument("--denoising_config", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--roi", default="Lung_L")
    parser.add_argument("--split", default="test")
    parser.add_argument("--volume_threshold_pct", type=float, default=20.0)
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--probe_types", nargs="+", default=None)
    parser.add_argument("--include_shift", action="store_true")
    parser.add_argument("--clean_policy", choices=["one_per_patient", "all_level_zero"], default="one_per_patient")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()

    out = Path(args.out_dir).expanduser().resolve()
    tables, figures, reports = out / "tables", out / "figures", out / "report"
    for d in [tables, figures, reports]:
        d.mkdir(parents=True, exist_ok=True)

    pilot_cfg = load_config(args.pilot_config)
    denoise_cfg = load_config(args.denoising_config)
    pilot_all = prepare_dataframe(args.pilot_csv, "Pilot without denoising", args.roi)
    denoise_all = prepare_dataframe(args.denoising_csv, "Denoising", args.roi)
    pilot = build_dataset(pilot_all, args.volume_threshold_pct, args.include_shift, args.clean_policy, args.probe_types)
    denoise = build_dataset(denoise_all, args.volume_threshold_pct, args.include_shift, args.clean_policy, args.probe_types)

    pilot.to_csv(tables / "analysis_dataset_pilot.csv", index=False)
    denoise.to_csv(tables / "analysis_dataset_denoising.csv", index=False)

    pilot_summary, pilot_curves = analyze_model(pilot, args.metrics, args.n_boot, args.seed)
    denoise_summary, denoise_curves = analyze_model(denoise, args.metrics, args.n_boot, args.seed + 1000)
    summary = pd.concat([pilot_summary, denoise_summary], ignore_index=True)
    curves = pd.concat([pilot_curves, denoise_curves], ignore_index=True)

    difference_rows = []
    for i, metric in enumerate(args.metrics):
        if metric not in pilot.columns or metric not in denoise.columns:
            continue
        result = paired_bootstrap_auc_difference(pilot, denoise, metric, args.n_boot, args.seed + 2000 + i)
        result.update(metric=metric, metric_display=DISPLAY.get(metric, metric))
        difference_rows.append(result)
    differences = pd.DataFrame(difference_rows)

    calibration = pd.concat([clean_calibration(pilot), clean_calibration(denoise)], ignore_index=True)
    repair = pd.concat([repair_summary(pilot), repair_summary(denoise)], ignore_index=True)

    summary.to_csv(tables / "roc_summary_models.csv", index=False)
    curves.to_csv(tables / "roc_curve_points_models.csv", index=False)
    differences.to_csv(tables / "paired_auc_differences.csv", index=False)
    calibration.to_csv(tables / "clean_preservation_summary.csv", index=False)
    repair.to_csv(tables / "repair_summary.csv", index=False)

    plot_roc(curves, summary, figures / "roc_comparison.png")
    plot_auc_comparison(summary, figures / "auc_comparison.png")
    plot_clean_calibration(calibration, figures / "clean_preservation.png")

    write_model_report(reports / "pilot_without_denoising_ROC_report.md", "Pilot without denoising", pilot_cfg, pilot, pilot_summary, clean_calibration(pilot), repair_summary(pilot), args)
    write_model_report(reports / "denoising_ROC_report.md", "Denoising", denoise_cfg, denoise, denoise_summary, clean_calibration(denoise), repair_summary(denoise), args)
    write_comparison_report(reports / "pilot_vs_denoising_comparison.md", summary, differences, calibration, repair, args)

    metadata = {
        "roi": args.roi, "split": args.split, "volume_threshold_pct": args.volume_threshold_pct,
        "include_shift": args.include_shift, "clean_policy": args.clean_policy,
        "n_boot": args.n_boot, "seed": args.seed, "metrics": args.metrics,
        "pilot_csv": str(Path(args.pilot_csv).expanduser().resolve()),
        "denoising_csv": str(Path(args.denoising_csv).expanduser().resolve()),
        "pilot_config": args.pilot_config, "denoising_config": args.denoising_config,
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote comparison analysis to: {out}")
    print(f"Main report: {reports / 'pilot_vs_denoising_comparison.md'}")
    print(f"Pilot report: {reports / 'pilot_without_denoising_ROC_report.md'}")
    print(f"Denoising report: {reports / 'denoising_ROC_report.md'}")


if __name__ == "__main__":
    main()
