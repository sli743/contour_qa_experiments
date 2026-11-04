#!/usr/bin/env python3
"""Generate manuscript figures for the NSR-QA report from existing experiment outputs.

This module is intentionally read-only with respect to experiments: it does not train or
re-evaluate a model.  It consumes the CSV/PNG assets already produced by the shared-probe,
directional-asymmetry, and actual-autoseg analyses and writes manuscript-ready figures.

Figures generated
-----------------
Figure 1  Framework schematic.
Figure 4a/4b  Separate pooled four-model AUC comparisons for Lung_L and Bronchus.
Figure 5a/5b  Separate proposed-model per-probe ASSD AUC plots for Lung_L and Bronchus.
Figure 6a–6d  Separate Lung_L/Bronchus ASSD disagreement-versus-severity figures by corruption family.
Figure 7a/7b  Separate directional-response and reconstruction-accuracy diagnostics.
Figure 8a/8b  Separate AUC and high-specificity sensitivity loss-ablation figures.
Figure 9      Real-patient P75 classification AUC heatmap across ROIs.

A CSV for the manuscript Table 3 loss ablation is also written.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from analysis.generate_four_model_all_probes_two_criteria import build_all_probe_dataset
from analysis.generate_four_model_per_probe_reports import load_models

# Manuscript typography.  These figures are intended to be inserted as individual
# panels, so use publication-scale fonts rather than compact dashboard defaults.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
    "figure.titlesize": 19,
})

METRICS = [
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
]
METRIC_LABELS = {
    "one_minus_dice_input_vs_recon": "1-Dice",
    "hd95_input_vs_recon": "HD95",
    "assd_input_vs_recon": "ASSD",
    "1_minus_dice": "1-Dice",
    "hd95": "HD95",
    "assd": "ASSD",
}
PROBE_ORDER = [
    "big_bite",
    "big_blob",
    "cut_lower_half",
    "cut_posterior_half",
    "cut_upper_half",
    "dilated",
    "eroded",
]


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def manuscript_model_name(x: str) -> str:
    x = str(x)
    lower = x.lower()
    if "pilot" in lower or "clean" in lower:
        return "Clean AE"
    if "benchmark" in lower:
        return "Benchmark-overlap DAE"
    if "random" in lower:
        return "Random-morphology DAE"
    if "symmetric" in lower or "stochastic-region-dropout" in lower:
        return "Ours"
    return x


def choose_ours(rows: pd.DataFrame) -> pd.DataFrame:
    labels = rows["model"].astype(str)
    mask = labels.str.contains("symmetric", case=False, regex=False)
    if not mask.any():
        mask = labels.str.contains("stochastic-region-dropout", case=False, regex=False)
    if not mask.any():
        raise RuntimeError("Could not identify proposed symmetric stochastic model from 'model' column")
    return rows.loc[mask].copy()


def figure1_framework(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fontsize=14, lw=1.6):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fill=False, linewidth=lw)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
        return p

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.2))

    ax.text(3.0, 5.55, "Training: learn normative reconstruction", ha="center", fontsize=17, fontweight="bold")
    ax.text(9.0, 5.55, "Deployment / QA", ha="center", fontsize=17, fontweight="bold")

    box(0.35, 3.8, 2.1, 1.0, "Physician-approved\ncontour")
    box(3.15, 3.8, 2.1, 1.0, "Stochastic training\ncorruption")
    box(2.0, 1.75, 2.7, 1.1, "Frozen pretrained 3DINO\n+ trainable decoder")
    box(5.0, 0.35, 2.0, 0.9, "Reconstruction loss\nvs clean target")
    arrow(2.45, 4.3, 3.15, 4.3)
    arrow(4.2, 3.8, 3.35, 2.85)
    arrow(1.4, 3.8, 2.55, 2.85)
    arrow(4.1, 1.75, 5.45, 1.1)
    arrow(1.4, 3.8, 5.95, 1.1)

    box(7.2, 3.8, 2.1, 1.0, "Candidate\nautoseg contour")
    box(9.95, 3.8, 1.7, 1.0, "NSR-QA\nreconstruction")
    box(8.3, 1.75, 2.7, 1.1, "Reconstruction disagreement\n1-Dice / HD95 / ASSD")
    box(8.8, 0.25, 1.7, 0.8, "QA anomaly score")
    arrow(9.3, 4.3, 9.95, 4.3)
    arrow(8.25, 3.8, 8.8, 2.85)
    arrow(10.8, 3.8, 10.45, 2.85)
    arrow(9.65, 1.75, 9.65, 1.05)

    ax.text(9.0, 3.25, "approved contour not required", ha="center", va="center", fontsize=12, style="italic")
    ax.plot([6.55, 6.55], [0.1, 5.45], linestyle="--", linewidth=1)
    save_figure(fig, outdir / "figure1_nsrqa_framework")


def load_pooled(root: Path, roi: str) -> pd.DataFrame:
    path = require_file(root / "pooled" / "all_probes_metric_summary.csv", f"{roi} pooled summary")
    df = pd.read_csv(path)
    df["roi"] = roi
    return df


def figure4_pooled_auc(lung_root: Path, bronchus_root: Path, outdir: Path) -> None:
    all_df = pd.concat([load_pooled(lung_root, "Lung_L"), load_pooled(bronchus_root, "Bronchus")], ignore_index=True)
    all_df = all_df[all_df["metric"].isin(METRICS)].copy()
    all_df["model_display"] = all_df["model"].map(manuscript_model_name)
    order = ["Clean AE", "Benchmark-overlap DAE", "Random-morphology DAE", "Ours"]
    offsets = np.linspace(-0.20, 0.20, len(METRICS))

    for suffix, roi in [("4a", "Lung_L"), ("4b", "Bronchus")]:
        fig, ax = plt.subplots(figsize=(9.6, 6.8))
        x = all_df[all_df["roi"] == roi]
        for mi, metric in enumerate(METRICS):
            m = x[x["metric"] == metric].set_index("model_display")
            yy = np.arange(len(order)) + offsets[mi]
            vals = np.array([m.loc[o, "auc"] if o in m.index else np.nan for o in order], float)
            low = np.array([m.loc[o, "auc_ci_low"] if o in m.index else np.nan for o in order], float)
            high = np.array([m.loc[o, "auc_ci_high"] if o in m.index else np.nan for o in order], float)
            err = np.vstack([vals - low, high - vals])
            ax.errorbar(vals, yy, xerr=err, marker="o", markersize=8, linestyle="none", capsize=4, linewidth=1.8, label=METRIC_LABELS[metric])
        ax.axvline(0.5, linestyle="--", linewidth=1.3)
        ax.set_xlim(0.45, 1.0)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_xlabel("ROC AUC (95% patient-cluster bootstrap CI)")
        ax.set_title(f"{roi}: pooled deterministic-probe QA performance")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(title="QA signal", loc="lower right")
        fig.tight_layout()
        save_figure(fig, outdir / f"figure{suffix}_pooled_model_auc_{roi.lower()}")


def load_by_probe(root: Path, roi: str) -> pd.DataFrame:
    path = require_file(root / "by_probe" / "all_probe_metric_summary.csv", f"{roi} per-probe summary")
    df = pd.read_csv(path)
    df["roi"] = roi
    return df


def figure5_probe_heatmap(lung_root: Path, bronchus_root: Path, outdir: Path) -> None:
    df = pd.concat([load_by_probe(lung_root, "Lung_L"), load_by_probe(bronchus_root, "Bronchus")], ignore_index=True)
    df = choose_ours(df)
    df = df[df["metric"] == "assd_input_vs_recon"].copy()

    for suffix, roi in [("5a", "Lung_L"), ("5b", "Bronchus")]:
        sub = df[df["roi"] == roi].set_index("probe_type").reindex(PROBE_ORDER)
        vals = sub["auc"].to_numpy(float).reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(5.8, 7.2))
        im = ax.imshow(vals, vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks([0])
        ax.set_xticklabels([roi])
        ax.set_yticks(np.arange(len(PROBE_ORDER)))
        ax.set_yticklabels([p.replace("_", " ") for p in PROBE_ORDER])
        for i, v in enumerate(vals[:, 0]):
            ax.text(0, i, "NA" if pd.isna(v) else f"{v:.3f}", ha="center", va="center", fontsize=15)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("ASSD(input,reconstruction) ROC AUC")
        ax.set_title(f"{roi}: probe-specific QA performance")
        fig.tight_layout()
        save_figure(fig, outdir / f"figure{suffix}_probe_specific_assd_auc_{roi.lower()}")


def _load_four_model_datasets_from_pooled_metadata(
    report_root: Path,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, object]]:
    """Reconstruct the exact pooled-analysis datasets from saved metadata.

    This deliberately reuses the same `build_all_probe_dataset` routine used by the
    ROC analysis so Figure 6 and the classification analysis share identical local/global
    severity definitions.
    """
    meta_path = require_file(
        report_root / "pooled" / "analysis_metadata.json",
        f"pooled analysis metadata under {report_root}",
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    roi = str(meta["roi"])
    model_items = [(str(name), Path(path)) for name, path in dict(meta["models"]).items()]
    included_probes = [str(x) for x in meta.get("included_probes", [])]
    if not included_probes:
        raise RuntimeError(f"No included_probes recorded in {meta_path}")
    volume_threshold_pct = float(meta["global_volume_threshold_pct"])
    local_hd95_threshold_mm = float(meta["local_hd95_threshold_mm"])

    loaded = load_models(model_items)
    datasets = {
        name: build_all_probe_dataset(
            obj, included_probes, volume_threshold_pct, local_hd95_threshold_mm
        )
        for name, obj in loaded.items()
    }
    return roi, datasets, meta


def _nice_zoom_upper(values: np.ndarray, quantile: float, *, minimum: float) -> float:
    """Choose a reproducible main-panel upper bound without deleting any observations."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite >= 0)]
    if finite.size == 0:
        return minimum
    q = float(np.quantile(finite, quantile))
    # Add a modest visual margin and round upward for readable axes.
    return max(minimum, float(np.ceil(q * 1.08)))


def _figure6_family_plot(
    datasets: dict[str, pd.DataFrame],
    roi: str,
    family: str,
    out_stem: Path,
    *,
    volume_threshold_pct: float,
    local_hd95_threshold_mm: float,
    zoom_quantile: float | None = None,
    plot_data_rows: list[dict[str, object]] | None = None,
) -> None:
    """Draw one family-specific severity-response figure.

    Global and local probes are intentionally never mixed. The normalized x-axis is
    exactly the criterion used by the corresponding ROC analysis:
      global: |volume change| / global volume-change threshold
      local:  HD95(input,target) / local HD95 threshold
    """
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    bins = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, np.inf])

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    prepared: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]] = []
    for raw_name, ds in datasets.items():
        sub = ds.loc[(~ds["__is_clean"]) & ds["__family"].eq(family)].copy()
        x = pd.to_numeric(sub["__severity_ratio"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["__score__assd_input_vs_recon"], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        sub = sub.loc[valid].copy()
        x, y = x[valid], y[valid]
        if x.size == 0:
            continue
        name = manuscript_model_name(raw_name)
        prepared.append((name, sub, x, y))
        all_x.append(x)
        all_y.append(y)
        if plot_data_rows is not None:
            for (_, row), xx, yy in zip(sub.iterrows(), x, y):
                plot_data_rows.append({
                    "roi": roi,
                    "family": family,
                    "model": name,
                    "probe_type": str(row.get("__probe_norm", "")),
                    "normalized_severity": float(xx),
                    "assd_input_vs_recon_mm": float(yy),
                    "is_bad": int(row.get("__label", 0)),
                })

    if not prepared:
        raise RuntimeError(f"No {family!r} probe observations available for {roi}")

    x_limit = y_limit = None
    if zoom_quantile is not None:
        x_limit = _nice_zoom_upper(np.concatenate(all_x), zoom_quantile, minimum=1.5)
        y_limit = _nice_zoom_upper(np.concatenate(all_y), zoom_quantile, minimum=1.0)

    off_panel = 0
    for name, sub, x, y in prepared:
        display = np.ones(x.shape, dtype=bool)
        if x_limit is not None:
            display &= x <= x_limit
        if y_limit is not None:
            display &= y <= y_limit
        off_panel += int((~display).sum())

        # Plot the model trend first so its default Matplotlib color can also be used
        # for the corresponding individual observations. Medians are computed from ALL
        # observations, including those outside the visual zoom.
        tmp = pd.DataFrame({"x": x, "y": y})
        tmp["bin"] = pd.cut(tmp["x"], bins=bins, right=False, include_lowest=True)
        med = tmp.groupby("bin", observed=True).agg(
            x=("x", "median"), y=("y", "median"), n=("y", "size")
        )
        med = med.loc[med["n"] > 0].sort_values("x")
        line, = ax.plot(
            med["x"],
            med["y"],
            marker="o",
            markersize=7,
            linewidth=2.5,
            label=name,
        )
        ax.scatter(
            x[display],
            y[display],
            s=20,
            alpha=0.13,
            color=line.get_color(),
            edgecolors="none",
        )

    ax.axvline(1.0, linestyle="--", linewidth=1.7, label="Substantial-error threshold")
    ax.text(
        1.0, 0.985,
        " below threshold  |  substantial error ",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=12,
    )
    ax.set_ylabel("ASSD(input, reconstruction) [mm]")
    if family == "global":
        ax.set_xlabel(
            f"Absolute volume change / {volume_threshold_pct:g}% threshold"
        )
        subtitle = (
            f"Global probes only; substantial error = |volume change| > "
            f"{volume_threshold_pct:g}%"
        )
    elif family == "local":
        ax.set_xlabel(
            f"HD95(input, target) / {local_hd95_threshold_mm:g} mm threshold"
        )
        subtitle = (
            f"Localized bite/blob probes only; substantial error = "
            f"HD95(input,target) > {local_hd95_threshold_mm:g} mm"
        )
    else:
        raise ValueError(f"Unsupported Figure 6 family: {family}")

    ax.set_title(f"{roi}: ASSD reconstruction disagreement vs defect severity\n{subtitle}")
    ax.grid(alpha=0.22)
    ax.legend(title="Reconstruction model", loc="upper left")
    if x_limit is not None:
        ax.set_xlim(0, x_limit)
    if y_limit is not None:
        ax.set_ylim(0, y_limit)
    if off_panel:
        ax.text(
            0.985, 0.025,
            f"{off_panel} model-probe observations outside the main-axis window\n"
            "(retained in all medians and quantitative analyses)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "alpha": 0.85, "edgecolor": "0.65"},
        )
    fig.tight_layout()
    save_figure(fig, out_stem)


def _figure6_full_range_supplement(
    datasets: dict[str, pd.DataFrame],
    roi: str,
    family: str,
    out_stem: Path,
    *,
    volume_threshold_pct: float,
    local_hd95_threshold_mm: float,
) -> None:
    """Save a transparent full-range companion for a zoomed Bronchus main panel."""
    _figure6_family_plot(
        datasets, roi, family, out_stem,
        volume_threshold_pct=volume_threshold_pct,
        local_hd95_threshold_mm=local_hd95_threshold_mm,
        zoom_quantile=None,
        plot_data_rows=None,
    )


def figure6_family_severity(
    lung_root: Path,
    bronchus_root: Path,
    outdir: Path,
    *,
    bronchus_zoom_quantile: float,
) -> None:
    """Generate four main Figure 6 files and two full-range Bronchus companions."""
    if not (0.90 <= bronchus_zoom_quantile < 1.0):
        raise ValueError("bronchus_zoom_quantile must be in [0.90, 1.0)")

    plot_rows: list[dict[str, object]] = []
    jobs = [
        (lung_root, {"global": "6a", "local": "6b"}),
        (bronchus_root, {"global": "6c", "local": "6d"}),
    ]
    for report_root, suffixes in jobs:
        roi, datasets, meta = _load_four_model_datasets_from_pooled_metadata(report_root)
        volume_threshold_pct = float(meta["global_volume_threshold_pct"])
        local_hd95_threshold_mm = float(meta["local_hd95_threshold_mm"])
        for family in ("global", "local"):
            zoom = bronchus_zoom_quantile if roi == "Bronchus" else None
            main_stem = outdir / (
                f"figure{suffixes[family]}_assd_signal_vs_defect_severity_"
                f"{family}_{roi.lower()}"
            )
            _figure6_family_plot(
                datasets, roi, family, main_stem,
                volume_threshold_pct=volume_threshold_pct,
                local_hd95_threshold_mm=local_hd95_threshold_mm,
                zoom_quantile=zoom,
                plot_data_rows=plot_rows,
            )
            if roi == "Bronchus":
                _figure6_full_range_supplement(
                    datasets, roi, family,
                    outdir / (
                        f"supp_figure6_{family}_bronchus_full_range"
                    ),
                    volume_threshold_pct=volume_threshold_pct,
                    local_hd95_threshold_mm=local_hd95_threshold_mm,
                )

    pd.DataFrame(plot_rows).to_csv(outdir / "figure6_severity_plot_data.csv", index=False)

def figure7_directional(directional_dir: Path, outdir: Path) -> None:
    path = require_file(directional_dir / "directional_challenge_by_level.csv", "directional asymmetry summary")
    df = pd.read_csv(path)
    required = {"direction", "offset_mm", "volume_correction_fraction_median", "dice_correction_gain_median"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Directional CSV missing expected columns: {sorted(missing)}")
    df["signed_offset_mm"] = np.where(df["direction"].astype(str).str.lower().eq("inward"), -df["offset_mm"], df["offset_mm"])
    df = df.sort_values("signed_offset_mm")

    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    ax.plot(df["signed_offset_mm"], df["volume_correction_fraction_median"], marker="o", markersize=8, linewidth=2.2)
    ax.axhline(0, linewidth=1.2)
    ax.axvline(0, linewidth=1.2)
    ax.set_xlabel("Signed input boundary offset (mm)\nnegative = inward; positive = outward")
    ax.set_ylabel("Median volume correction fraction")
    ax.set_title("Directional reconstruction response: asymmetric objective")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, outdir / "figure7a_directional_reconstruction_response")

    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    ax.plot(df["signed_offset_mm"], df["dice_correction_gain_median"], marker="o", markersize=8, linewidth=2.2)
    ax.axhline(0, linewidth=1.2)
    ax.axvline(0, linewidth=1.2)
    ax.set_xlabel("Signed input boundary offset (mm)\nnegative = inward; positive = outward")
    ax.set_ylabel("Median Dice correction gain")
    ax.set_title("Directional effect on reconstruction accuracy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, outdir / "figure7b_directional_reconstruction_accuracy")


def get_dilation_row(report_root: Path, version_word: str) -> pd.Series:
    path = require_file(report_root / "by_probe" / "all_probe_metric_summary.csv", f"{version_word} per-probe summary")
    df = pd.read_csv(path)
    df = df[(df["probe_type"].astype(str).str.lower() == "dilated") & (df["metric"] == "assd_input_vs_recon")].copy()
    labels = df["model"].astype(str)
    preferred = labels.str.contains(version_word, case=False, regex=False)
    if preferred.any():
        df = df.loc[preferred]
    else:
        df = df.loc[labels.str.contains("stochastic-region-dropout", case=False, regex=False)]
    if len(df) != 1:
        raise RuntimeError(f"Expected one {version_word} Lung_L dilation/ASSD row, found {len(df)}")
    return df.iloc[0]


def figure8_loss_ablation(v2_root: Path, v3_root: Path, outdir: Path) -> None:
    asym = get_dilation_row(v2_root, "v2")
    sym = get_dilation_row(v3_root, "symmetric")
    rows = pd.DataFrame([
        {
            "objective": "Asymmetric",
            "auc": float(asym["auc"]),
            "auc_ci_low": float(asym.get("auc_ci_low", np.nan)),
            "auc_ci_high": float(asym.get("auc_ci_high", np.nan)),
            "sensitivity_at_95spec": float(asym.get("sens_at_95spec", np.nan)),
            "achieved_specificity": float(asym.get("achieved_specificity", np.nan)),
        },
        {
            "objective": "Symmetric changed-region",
            "auc": float(sym["auc"]),
            "auc_ci_low": float(sym.get("auc_ci_low", np.nan)),
            "auc_ci_high": float(sym.get("auc_ci_high", np.nan)),
            "sensitivity_at_95spec": float(sym.get("sens_at_95spec", np.nan)),
            "achieved_specificity": float(sym.get("achieved_specificity", np.nan)),
        },
    ])
    rows.to_csv(outdir / "table3_lungl_dilation_loss_ablation.csv", index=False)
    x = np.arange(2)

    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    ax.bar(x, rows["auc"])
    if rows[["auc_ci_low", "auc_ci_high"]].notna().all().all():
        err = np.vstack([rows["auc"] - rows["auc_ci_low"], rows["auc_ci_high"] - rows["auc"]])
        ax.errorbar(x, rows["auc"], yerr=err, fmt="none", capsize=5, linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Asymmetric", "Symmetric"], rotation=10, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ASSD ROC AUC")
    ax.set_title("Lung_L dilation discrimination")
    ax.axhline(0.5, linestyle="--", linewidth=1.2)
    fig.tight_layout()
    save_figure(fig, outdir / "figure8a_symmetric_loss_auc_ablation")

    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    ax.bar(x, rows["sensitivity_at_95spec"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Asymmetric", "Symmetric"], rotation=10, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sensitivity at ≥95% specificity")
    ax.set_title("Lung_L dilation: high-specificity operating point")
    fig.tight_layout()
    save_figure(fig, outdir / "figure8b_symmetric_loss_sensitivity_ablation")


def load_real_classification(path: Path) -> pd.DataFrame:
    df = pd.read_csv(require_file(path, "real-data P75 classification table"))
    # Current report output from generate_actual_autoseg_ranking_classification_report.py.
    if {"qa_metric", "true_metric", "roc_auc_testing1"}.issubset(df.columns):
        df = df[df["qa_metric"] == df["true_metric"]].copy()
        return df
    raise RuntimeError(f"Unrecognized real-data classification schema in {path}")


def figure9_real_data_heatmap(classification_csv: Path, outdir: Path) -> None:
    df = load_real_classification(classification_csv)
    metric_order = ["1_minus_dice", "hd95", "assd"]
    rois = sorted(df["roi"].astype(str).unique())
    pivot = df.pivot_table(index="roi", columns="qa_metric", values="roc_auc_testing1", aggfunc="first").reindex(index=rois, columns=metric_order)

    rel_map = {}
    if "reliable_roc_min_class_n" in df.columns:
        tmp = df.copy()
        tmp["reliable"] = tmp["reliable_roc_min_class_n"].astype(str).str.lower().isin({"true", "1", "yes"})
        rel_map = {(str(r.roi), str(r.qa_metric)): bool(r.reliable) for _, r in tmp.iterrows()}

    fig_h = max(6.5, 0.36 * len(rois) + 1.8)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    arr = pivot.to_numpy(float)
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_order])
    ax.set_yticks(np.arange(len(rois)))
    ax.set_yticklabels(rois)
    for i, roi in enumerate(rois):
        for j, metric in enumerate(metric_order):
            v = pivot.loc[roi, metric]
            if pd.isna(v):
                text = "NA"
            else:
                dagger = "" if rel_map.get((roi, metric), True) else "†"
                text = f"{v:.3f}{dagger}"
            ax.text(j, i, text, ha="center", va="center", fontsize=13)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Testing ROC AUC")
    ax.set_title("Real-patient autosegmentation QA: P75 high-error classification")
    ax.set_xlabel("Corresponding NSR-QA reconstruction-disagreement signal")
    ax.set_ylabel("ROI")
    fig.tight_layout()
    save_figure(fig, outdir / "figure9_real_data_p75_auc_heatmap")


def write_manifest(outdir: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(outdir / "figure_source_manifest.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    scratch = Path("/scratch/shaohan/contour_QA")
    p.add_argument("--output-dir", type=Path, default=scratch / "manuscript_figures" / "nsrqa")
    p.add_argument("--lungl-v3-report", type=Path, default=scratch / "roc_comparison" / "Lung_L_four_models_v3_symmetric_shared_probe")
    p.add_argument("--bronchus-v3-report", type=Path, default=scratch / "roc_comparison" / "Bronchus_four_models_v3_symmetric_shared_probe")
    p.add_argument("--lungl-v2-report", type=Path, default=scratch / "roc_comparison" / "Lung_L_four_models_v2_200_shared_probe")
    p.add_argument(
        "--directional-dir",
        type=Path,
        default=scratch / "dino_stochastic_region_dropout" / "dino_shape_stochastic_region_dropout_Lung_L_v2_global_boundary" / "directional_asymmetry_diagnostic_test",
    )
    p.add_argument(
        "--real-classification-csv",
        type=Path,
        default=scratch / "all_roi_NSRQA_study" / "actual_autoseg_regression" / "ranking_classification_report" / "corresponding_metric_classification_p75.csv",
    )
    p.add_argument(
        "--bronchus-figure6-zoom-quantile",
        type=float,
        default=0.975,
        help=(
            "Quantile used independently for Bronchus global/local x and y main-axis "
            "limits. Full-range companion figures are also saved. Default: 0.975."
        ),
    )
    p.add_argument("--skip", action="append", default=[], choices=["1", "4", "5", "6", "7", "8", "9"], help="Skip a figure number")
    args = p.parse_args()

    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)
    manifest: list[dict[str, str]] = []

    if "1" not in skip:
        figure1_framework(outdir)
        manifest.append({"figure": "Figure 1", "output": "figure1_nsrqa_framework", "source": "schematic generated from method definition"})
    if "4" not in skip:
        figure4_pooled_auc(args.lungl_v3_report, args.bronchus_v3_report, outdir)
        manifest.extend([
            {"figure": "Figure 4a", "output": "figure4a_pooled_model_auc_lung_l", "source": "Lung_L v3 symmetric pooled all_probes_metric_summary.csv"},
            {"figure": "Figure 4b", "output": "figure4b_pooled_model_auc_bronchus", "source": "Bronchus v3 symmetric pooled all_probes_metric_summary.csv"},
        ])
    if "5" not in skip:
        figure5_probe_heatmap(args.lungl_v3_report, args.bronchus_v3_report, outdir)
        manifest.extend([
            {"figure": "Figure 5a", "output": "figure5a_probe_specific_assd_auc_lung_l", "source": "Lung_L v3 symmetric per-probe all_probe_metric_summary.csv"},
            {"figure": "Figure 5b", "output": "figure5b_probe_specific_assd_auc_bronchus", "source": "Bronchus v3 symmetric per-probe all_probe_metric_summary.csv"},
        ])
    if "6" not in skip:
        figure6_family_severity(
            args.lungl_v3_report,
            args.bronchus_v3_report,
            outdir,
            bronchus_zoom_quantile=args.bronchus_figure6_zoom_quantile,
        )
        manifest.extend([
            {"figure": "Figure 6a", "output": "figure6a_assd_signal_vs_defect_severity_global_lung_l", "source": "Lung_L pooled-analysis metadata/raw evaluation CSVs; global probes only; same volume-change criterion as ROC"},
            {"figure": "Figure 6b", "output": "figure6b_assd_signal_vs_defect_severity_local_lung_l", "source": "Lung_L pooled-analysis metadata/raw evaluation CSVs; big_bite/big_blob only; same HD95 criterion as ROC"},
            {"figure": "Figure 6c", "output": "figure6c_assd_signal_vs_defect_severity_global_bronchus", "source": "Bronchus global probes only; same volume-change criterion as ROC; main axes quantile-zoomed"},
            {"figure": "Figure 6d", "output": "figure6d_assd_signal_vs_defect_severity_local_bronchus", "source": "Bronchus big_bite/big_blob only; same HD95 criterion as ROC; main axes quantile-zoomed"},
            {"figure": "Supplementary Figure 6c full range", "output": "supp_figure6_global_bronchus_full_range", "source": "Same Bronchus global observations as Figure 6c without visual zoom"},
            {"figure": "Supplementary Figure 6d full range", "output": "supp_figure6_local_bronchus_full_range", "source": "Same Bronchus local observations as Figure 6d without visual zoom"},
        ])
    if "7" not in skip:
        figure7_directional(args.directional_dir, outdir)
        manifest.extend([
            {"figure": "Figure 7a", "output": "figure7a_directional_reconstruction_response", "source": "v2 directional_challenge_by_level.csv"},
            {"figure": "Figure 7b", "output": "figure7b_directional_reconstruction_accuracy", "source": "v2 directional_challenge_by_level.csv"},
        ])
    if "8" not in skip:
        figure8_loss_ablation(args.lungl_v2_report, args.lungl_v3_report, outdir)
        manifest.extend([
            {"figure": "Figure 8a", "output": "figure8a_symmetric_loss_auc_ablation", "source": "Lung_L v2 and v3 per-probe dilation/ASSD summaries"},
            {"figure": "Figure 8b", "output": "figure8b_symmetric_loss_sensitivity_ablation", "source": "Lung_L v2 and v3 per-probe dilation/ASSD summaries"},
        ])
    if "9" not in skip:
        figure9_real_data_heatmap(args.real_classification_csv, outdir)
        manifest.append({"figure": "Figure 9", "output": "figure9_real_data_p75_auc_heatmap", "source": "actual-autoseg corresponding-metric P75 classification table"})

    write_manifest(outdir, manifest)
    print(f"Wrote manuscript figures to: {outdir}")
    print((outdir / "figure_source_manifest.csv").read_text())


if __name__ == "__main__":
    main()
