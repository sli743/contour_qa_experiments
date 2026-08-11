#!/usr/bin/env python3
"""Diagnose why a contour-reconstruction QA model responds poorly to dilation.

This analysis uses already-computed probe-evaluation CSVs. It does not rerun inference.
For each model it compares clean and dilated cases and asks:

1. Does the dilated input get larger as expected?
2. Does the reconstruction remain near the clean/target volume, or grow with the input?
3. Does reconstruction move toward the target in Dice/volume space?
4. Does the desired input-vs-reconstruction QA signal increase with dilation severity?

The main diagnostic quantity is the volume-follow fraction

    follow = (V_recon - V_target) / (V_input - V_target)

for dilation rows with nonzero input excess. Rough interpretation:
    follow ~ 1 : reconstruction follows the dilation (bad for anomaly signalling)
    follow ~ 0 : reconstruction stays near target size (desired)
    follow < 0  : reconstruction overshoots inward

The script is schema-tolerant for the historical and shared-probe CSVs used in this
project, provided the core input/reconstruction/target volume and Dice columns exist.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = [
    "Pilot AE",
    "Benchmark-leakage DAE",
    "Random-morphology DAE",
    "Stochastic-region-dropout DAE",
]


def parse_model_arg(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--model must be NAME=/path/to.csv")
    name, path = text.split("=", 1)
    return name.strip(), Path(path).expanduser()


def patient_col(df: pd.DataFrame) -> str:
    for c in ["patient_id", "case_id", "file"]:
        if c in df.columns:
            return c
    raise ValueError("CSV has no patient_id, case_id, or file column")


def normalize_patient(v) -> str:
    s = str(v)
    m = re.search(r"patient[_-]?(\d+)", s, flags=re.I)
    if m:
        return m.group(1)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def ensure_volumes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Prefer voxel counts. Fall back to volume fractions, which preserve all ratios.
    mapping = {
        "input_voxels": "volfrac_input",
        "recon_voxels": "volfrac_recon",
        "target_voxels": "volfrac_target",
    }
    for dst, src in mapping.items():
        if dst not in out.columns:
            if src not in out.columns:
                raise ValueError(
                    f"Need either {dst!r} or {src!r}. Available columns: {list(out.columns)}"
                )
            out[dst] = numeric(out, src)
        else:
            out[dst] = numeric(out, dst)
    return out


def prepare_model(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "probe_type" not in df.columns:
        raise ValueError(f"{name}: {path} has no probe_type column")
    df = ensure_volumes(df)
    pcol = patient_col(df)
    df["patient_norm"] = df[pcol].map(normalize_patient)
    df["model"] = name
    df["probe_norm"] = df["probe_type"].astype(str).str.lower().str.strip()
    if "level" in df.columns:
        df["level_num"] = numeric(df, "level")
    else:
        df["level_num"] = np.nan

    vi = numeric(df, "input_voxels")
    vr = numeric(df, "recon_voxels")
    vt = numeric(df, "target_voxels")
    eps = 1e-12

    df["input_over_target"] = vi / (vt + eps)
    df["recon_over_target"] = vr / (vt + eps)
    df["recon_over_input"] = vr / (vi + eps)
    df["input_excess_fraction"] = (vi - vt) / (vt + eps)
    df["recon_excess_fraction"] = (vr - vt) / (vt + eps)
    df["severity_volume_pct"] = 100.0 * np.abs(vi - vt) / (vt + eps)
    denom = vi - vt
    df["volume_follow_fraction"] = np.where(
        np.abs(denom) > eps,
        (vr - vt) / denom,
        np.nan,
    )
    df["volume_correction_fraction"] = 1.0 - df["volume_follow_fraction"]

    # Recompute/standardize core scores.
    df["one_minus_dice_input_vs_recon"] = 1.0 - numeric(df, "dice_input_vs_recon")
    df["relative_volume_difference_input_vs_recon"] = np.abs(vi - vr) / (vi + eps)
    return df


def add_clean_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each patient's clean reconstruction baseline to dilation rows."""
    clean = df[df["probe_norm"].isin(["clean", "none", "identity"])].copy()
    if clean.empty:
        # Shared-bank manifests may encode clean as level 0 of every probe. Do not guess.
        df["clean_recon_over_target"] = np.nan
        df["recon_volume_change_from_clean_pct"] = np.nan
        df["clean_dice_recon_vs_target"] = np.nan
        df["dice_recon_target_change_from_clean"] = np.nan
        return df

    clean = clean.sort_values(["patient_norm"]).drop_duplicates("patient_norm", keep="first")
    clean = clean[["patient_norm", "recon_over_target", "dice_recon_vs_target"]].rename(
        columns={
            "recon_over_target": "clean_recon_over_target",
            "dice_recon_vs_target": "clean_dice_recon_vs_target",
        }
    )
    out = df.merge(clean, on="patient_norm", how="left")
    out["recon_volume_change_from_clean_pct"] = 100.0 * (
        out["recon_over_target"] - out["clean_recon_over_target"]
    )
    out["dice_recon_target_change_from_clean"] = (
        numeric(out, "dice_recon_vs_target") - out["clean_dice_recon_vs_target"]
    )
    return out


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 3 or z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan
    return float(z["x"].corr(z["y"], method="spearman"))


def grouped_summary(dil: pd.DataFrame, bad_threshold_pct: float) -> pd.DataFrame:
    rows: List[Dict] = []
    for model, sub in dil.groupby("model", sort=False):
        severe = sub[sub["severity_volume_pct"] > bad_threshold_pct]
        rows.append({
            "model": model,
            "n_dilated": len(sub),
            "n_severe": len(severe),
            "median_input_over_target_severe": severe["input_over_target"].median(),
            "median_recon_over_target_severe": severe["recon_over_target"].median(),
            "median_recon_over_input_severe": severe["recon_over_input"].median(),
            "median_volume_follow_fraction_severe": severe["volume_follow_fraction"].replace([np.inf, -np.inf], np.nan).median(),
            "median_volume_correction_fraction_severe": severe["volume_correction_fraction"].replace([np.inf, -np.inf], np.nan).median(),
            "median_recon_volume_change_from_clean_pct_severe": severe["recon_volume_change_from_clean_pct"].median(),
            "median_dice_input_target_severe": numeric(severe, "dice_input_vs_target").median(),
            "median_dice_recon_target_severe": numeric(severe, "dice_recon_vs_target").median(),
            "median_dice_input_recon_severe": numeric(severe, "dice_input_vs_recon").median(),
            "rho_severity_vs_recon_over_target": safe_spearman(sub["severity_volume_pct"], sub["recon_over_target"]),
            "rho_severity_vs_follow_fraction": safe_spearman(sub["severity_volume_pct"], sub["volume_follow_fraction"]),
            "rho_severity_vs_1minus_dice_signal": safe_spearman(sub["severity_volume_pct"], sub["one_minus_dice_input_vs_recon"]),
            "rho_severity_vs_hd95_signal": safe_spearman(sub["severity_volume_pct"], numeric(sub, "hd95_input_vs_recon")),
            "rho_severity_vs_assd_signal": safe_spearman(sub["severity_volume_pct"], numeric(sub, "assd_input_vs_recon")),
            "rho_severity_vs_relvol_signal": safe_spearman(sub["severity_volume_pct"], sub["relative_volume_difference_input_vs_recon"]),
        })
    return pd.DataFrame(rows)


def level_summary(dil: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "level_num"] if dil["level_num"].notna().any() else ["model"]
    cols = [
        "severity_volume_pct", "input_over_target", "recon_over_target", "recon_over_input",
        "volume_follow_fraction", "volume_correction_fraction", "recon_volume_change_from_clean_pct",
        "dice_input_vs_target", "dice_recon_vs_target", "dice_input_vs_recon",
        "one_minus_dice_input_vs_recon", "hd95_input_vs_recon", "assd_input_vs_recon",
        "relative_volume_difference_input_vs_recon",
    ]
    rows = []
    for keys, sub in dil.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        row["n"] = len(sub)
        for c in cols:
            if c in sub.columns:
                s = numeric(sub, c).replace([np.inf, -np.inf], np.nan)
                row[f"{c}_median"] = s.median()
                row[f"{c}_q25"] = s.quantile(0.25)
                row[f"{c}_q75"] = s.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def model_order(names: Iterable[str]) -> List[str]:
    names = list(dict.fromkeys(names))
    ordered = [m for m in MODEL_ORDER if m in names]
    ordered += [m for m in names if m not in ordered]
    return ordered


def plot_model_lines(df: pd.DataFrame, ycols: List[Tuple[str, str]], ylabel: str, title: str,
                     out: Path, threshold: float, horizontal_one: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for model in model_order(df["model"].unique()):
        sub = df[df["model"] == model].copy()
        for col, suffix in ycols:
            if col not in sub.columns:
                continue
            s = sub[["severity_volume_pct", col]].replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty: continue
            # Median by actual realized severity rounded to 0.1% is too fragmented; level if present.
            if "level_num" in sub.columns and sub["level_num"].notna().any():
                med = sub.groupby("level_num", as_index=False).agg(
                    severity_volume_pct=("severity_volume_pct", "median"),
                    value=(col, "median"),
                ).sort_values("severity_volume_pct")
            else:
                med = s.sort_values("severity_volume_pct").rename(columns={col: "value"})
            label = model if len(ycols) == 1 else f"{model}: {suffix}"
            ax.plot(med["severity_volume_pct"], med["value"], marker="o", label=label)
    ax.axvline(threshold, linestyle="--", linewidth=1.2, label=f"Bad threshold = {threshold:g}%")
    if horizontal_one:
        ax.axhline(1.0, linestyle=":", linewidth=1.0, label="Target ratio = 1")
    ax.set_xlabel("Dilation severity: |V(input)-V(target)| / V(target) (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_volume_ratios(dil: pd.DataFrame, outdir: Path, threshold: float) -> None:
    # Separate figure per model is much easier to read for the three volume curves.
    for model in model_order(dil["model"].unique()):
        sub = dil[dil["model"] == model]
        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        for col, label in [
            ("input_over_target", "Input / target"),
            ("recon_over_target", "Reconstruction / target"),
            ("recon_over_input", "Reconstruction / input"),
        ]:
            if sub["level_num"].notna().any():
                med = sub.groupby("level_num", as_index=False).agg(
                    severity=("severity_volume_pct", "median"), value=(col, "median")
                ).sort_values("severity")
            else:
                med = sub[["severity_volume_pct", col]].sort_values("severity_volume_pct")
                med = med.rename(columns={"severity_volume_pct": "severity", col: "value"})
            ax.plot(med["severity"], med["value"], marker="o", label=label)
        ax.axvline(threshold, linestyle="--", linewidth=1.2, label=f"Bad threshold = {threshold:g}%")
        ax.axhline(1.0, linestyle=":", linewidth=1.0, label="Target/reference = 1")
        ax.set_xlabel("Dilation severity (%)")
        ax.set_ylabel("Volume ratio")
        ax.set_title(f"{model}: volume response to dilation")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        token = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
        fig.savefig(outdir / f"volume_response_{token}.png", dpi=220)
        fig.savefig(outdir / f"volume_response_{token}.pdf")
        plt.close(fig)


def plot_dice_triplets(dil: pd.DataFrame, outdir: Path, threshold: float) -> None:
    for model in model_order(dil["model"].unique()):
        sub = dil[dil["model"] == model]
        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        for col, label in [
            ("dice_input_vs_target", "Input vs target"),
            ("dice_recon_vs_target", "Reconstruction vs target"),
            ("dice_input_vs_recon", "Input vs reconstruction"),
        ]:
            if col not in sub.columns: continue
            if sub["level_num"].notna().any():
                med = sub.groupby("level_num", as_index=False).agg(
                    severity=("severity_volume_pct", "median"), value=(col, "median")
                ).sort_values("severity")
            else:
                med = sub[["severity_volume_pct", col]].sort_values("severity_volume_pct")
                med = med.rename(columns={"severity_volume_pct": "severity", col: "value"})
            ax.plot(med["severity"], med["value"], marker="o", label=label)
        ax.axvline(threshold, linestyle="--", linewidth=1.2, label=f"Bad threshold = {threshold:g}%")
        ax.set_xlabel("Dilation severity (%)")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{model}: where does reconstruction move?")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        token = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
        fig.savefig(outdir / f"dice_response_{token}.png", dpi=220)
        fig.savefig(outdir / f"dice_response_{token}.pdf")
        plt.close(fig)


def plot_follow_fraction(dil: pd.DataFrame, outdir: Path, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for model in model_order(dil["model"].unique()):
        sub = dil[dil["model"] == model]
        if sub["level_num"].notna().any():
            med = sub.groupby("level_num", as_index=False).agg(
                severity=("severity_volume_pct", "median"),
                follow=("volume_follow_fraction", "median"),
            ).replace([np.inf, -np.inf], np.nan).dropna().sort_values("severity")
        else:
            med = sub[["severity_volume_pct", "volume_follow_fraction"]].replace([np.inf,-np.inf],np.nan).dropna()
            med = med.rename(columns={"severity_volume_pct":"severity", "volume_follow_fraction":"follow"})
        ax.plot(med["severity"], med["follow"], marker="o", label=model)
    ax.axvline(threshold, linestyle="--", linewidth=1.2, label=f"Bad threshold = {threshold:g}%")
    ax.axhline(1.0, linestyle=":", linewidth=1.0, label="Follow input completely")
    ax.axhline(0.0, linestyle="-.", linewidth=1.0, label="Return to target volume")
    ax.set_xlabel("Dilation severity (%)")
    ax.set_ylabel("Volume-follow fraction")
    ax.set_title("Does reconstruction follow the dilated input?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "volume_follow_fraction_four_models.png", dpi=220)
    fig.savefig(outdir / "volume_follow_fraction_four_models.pdf")
    plt.close(fig)


def plot_signal(dil: pd.DataFrame, outdir: Path, threshold: float) -> None:
    metrics = [
        ("one_minus_dice_input_vs_recon", "1 - Dice(input,reconstruction)", "signal_1minus_dice"),
        ("hd95_input_vs_recon", "HD95(input,reconstruction)", "signal_hd95"),
        ("assd_input_vs_recon", "ASSD(input,reconstruction)", "signal_assd"),
        ("relative_volume_difference_input_vs_recon", "Relative volume difference(input,reconstruction)", "signal_rel_volume"),
    ]
    for col, ylabel, stem in metrics:
        if col not in dil.columns or numeric(dil, col).notna().sum() == 0:
            continue
        plot_model_lines(dil, [(col, "")], ylabel, f"QA signal versus dilation severity: {ylabel}", outdir / f"{stem}.png", threshold)


def md_table(df: pd.DataFrame, decimals: int = 3) -> str:
    # No dependency on pandas.to_markdown/tabulate.
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    def fmt(v):
        if pd.isna(v): return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{decimals}f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, summary: pd.DataFrame, threshold: float, dil: pd.DataFrame) -> None:
    compact_cols = [
        "model", "n_dilated", "n_severe",
        "median_input_over_target_severe", "median_recon_over_target_severe",
        "median_volume_follow_fraction_severe", "median_volume_correction_fraction_severe",
        "median_recon_volume_change_from_clean_pct_severe",
        "rho_severity_vs_recon_over_target", "rho_severity_vs_1minus_dice_signal",
        "rho_severity_vs_hd95_signal", "rho_severity_vs_assd_signal",
    ]
    compact = summary[[c for c in compact_cols if c in summary.columns]].copy()

    text = f"""# Four-model dilation behavior diagnostic

## Diagnostic question

This report does **not** ask whether the decoder perfectly repairs dilation. It asks why dilation may produce a weak anomaly signal.

For each dilated input, the analysis follows three objects: input, reconstruction, and clean target. Dilation severity is the realized absolute input-target volume change. The vertical separation line in all severity plots is **{threshold:g}%**, matching the current Lung_L-style global good/bad criterion unless overridden on the command line.

The key quantity is **volume-follow fraction**:

`(V_reconstruction - V_target) / (V_input - V_target)`

Interpretation:

- near **1**: reconstruction grows with/follows the dilated input; this tends to suppress input-reconstruction QA signal;
- near **0**: reconstruction remains near target volume; this is desirable for detecting dilation;
- below **0**: reconstruction overshoots inward;
- values are descriptive, not a training objective.

## Main summary for severe dilation

{md_table(compact)}

## How to distinguish the main failure mechanisms

**Decoder follows dilation:** as severity increases, `reconstruction/target` rises toward `input/target`, volume-follow fraction approaches 1, and input-reconstruction disagreement remains small or decreases.

**Poor clean baseline/calibration:** the clean reconstruction is already biased away from target. Dilated reconstruction may move toward target relative to that biased baseline, producing non-monotonic raw disagreement. Use `recon_volume_change_from_clean_pct` and the clean rows in `diagnostic_rows.csv` to check this.

**Desired detector behavior:** input/target rises with dilation while reconstruction/target remains comparatively stable; volume-follow fraction stays near 0; 1-Dice, HD95, ASSD, or relative-volume input-reconstruction signal rises with severity.

## Figures

### Four-model follow behavior

![Volume follow fraction](plots/volume_follow_fraction_four_models.png)

### QA signals

![1-Dice signal](plots/signal_1minus_dice.png)

![HD95 signal](plots/signal_hd95.png)

![ASSD signal](plots/signal_assd.png)

![Relative-volume signal](plots/signal_rel_volume.png)

## Per-model volume and Dice response

"""
    for model in model_order(dil["model"].unique()):
        token = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
        text += f"\n### {model}\n\n![Volume response](plots/volume_response_{token}.png)\n\n![Dice response](plots/dice_response_{token}.png)\n"

    text += """

## Files for review

Upload `diagnostic_report.md`, `model_dilation_summary.csv`, and `dilation_by_level_summary.csv` first. If interpretation depends on patient heterogeneity, also upload `diagnostic_rows.csv`. The PNG figures are useful if you want visual review, especially the four-model follow-fraction and signal plots.
"""
    (outdir / "diagnostic_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, type=parse_model_arg,
                    help="Repeat: --model 'Pilot AE=/path/to/probe_eval_sweep_detailed.csv'")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--probe", default="dilated")
    ap.add_argument("--bad-volume-threshold-pct", type=float, default=20.0)
    args = ap.parse_args()

    outdir = Path(args.output_dir).expanduser()
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, path in args.model:
        df = add_clean_relative_features(prepare_model(name, path))
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True, sort=False)

    probe = args.probe.lower().strip()
    dil = all_df[all_df["probe_norm"] == probe].copy()
    if dil.empty:
        available = sorted(all_df["probe_norm"].dropna().unique())
        raise ValueError(f"No rows for probe={probe!r}. Available probe types: {available}")

    # retain clean rows plus dilation rows for audit/review
    audit = all_df[all_df["probe_norm"].isin([probe, "clean", "none", "identity"])].copy()
    audit.to_csv(outdir / "diagnostic_rows.csv", index=False)

    summary = grouped_summary(dil, args.bad_volume_threshold_pct)
    summary.to_csv(outdir / "model_dilation_summary.csv", index=False)
    levels = level_summary(dil)
    levels.to_csv(outdir / "dilation_by_level_summary.csv", index=False)

    plot_volume_ratios(dil, plotdir, args.bad_volume_threshold_pct)
    plot_dice_triplets(dil, plotdir, args.bad_volume_threshold_pct)
    plot_follow_fraction(dil, plotdir, args.bad_volume_threshold_pct)
    plot_signal(dil, plotdir, args.bad_volume_threshold_pct)
    write_report(outdir, summary, args.bad_volume_threshold_pct, dil)

    print(f"Wrote: {outdir / 'diagnostic_report.md'}")
    print(f"Wrote: {outdir / 'model_dilation_summary.csv'}")
    print(f"Wrote: {outdir / 'dilation_by_level_summary.csv'}")
    print(f"Wrote: {outdir / 'diagnostic_rows.csv'}")
    print(f"Plots: {plotdir}")


if __name__ == "__main__":
    main()
