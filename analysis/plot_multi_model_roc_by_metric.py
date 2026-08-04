#!/usr/bin/env python3
"""
Create separate multi-model ROC comparison figures for multiple metrics.

Each metric figure overlays an arbitrary number of named models.

The script reads detailed probe-evaluation CSV files, excludes shift probes
when requested, derives the positive class from absolute input-vs-target volume
change, and saves one PNG and one PDF per metric.

Example
-------
python plot_multi_model_roc_by_metric.py \
    --pilot pilot.csv \
    --benchmark benchmark.csv \
    --random-morphology random.csv \
    --volume-threshold-pct 10 \
    --exclude-shifts \
    --roi Bronchus \
    --output-dir roc_by_metric
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


PROBE_CANDIDATES = (
    "probe",
    "probe_type",
    "corruption",
    "corruption_type",
    "perturbation",
    "perturbation_type",
)

PATIENT_CANDIDATES = (
    "patient_id",
    "patient",
    "case_id",
    "case",
)

VOLUME_CHANGE_CANDIDATES = (
    "abs_volume_change_pct",
    "absolute_volume_change_pct",
    "volume_change_pct",
    "input_target_volume_change_pct",
    "input_vs_target_volume_change_pct",
    "volume_diff_pct",
)

DEFAULT_METRICS = (
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
    "relative_volume_difference_input_vs_recon",
)

METRIC_LABELS = {
    "one_minus_dice_input_vs_recon": "1 − Dice(input, reconstruction)",
    "hd95_input_vs_recon": "HD95(input, reconstruction)",
    "assd_input_vs_recon": "ASSD(input, reconstruction)",
    "relative_volume_difference_input_vs_recon": (
        "Relative volume difference(input, reconstruction)"
    ),
}

METRIC_ALIASES = {
    "one_minus_dice_input_recon": "one_minus_dice_input_vs_recon",
    "dice_input_recon": "one_minus_dice_input_vs_recon",
    "hd95_input_recon": "hd95_input_vs_recon",
    "assd_input_recon": "assd_input_vs_recon",
    "volume_diff_input_recon": "relative_volume_difference_input_vs_recon",
}


def column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in df.columns}


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = column_lookup(df)
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def is_shift_probe(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower()
    return text.str.contains(r"shift|translation|translated", regex=True, na=False)


def is_clean_probe(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower().str.strip()
    return text.isin({"clean", "none", "identity", "original", "approved"})


def deduplicate_clean_rows(df: pd.DataFrame, probe_col: str | None) -> pd.DataFrame:
    if probe_col is None:
        return df

    patient_col = find_column(df, PATIENT_CANDIDATES)
    if patient_col is None:
        return df

    clean_mask = is_clean_probe(df[probe_col])
    clean_rows = df.loc[clean_mask].drop_duplicates(
        subset=[patient_col],
        keep="first",
    )
    corrupted_rows = df.loc[~clean_mask]
    return pd.concat([clean_rows, corrupted_rows], ignore_index=True)


def derive_volume_change_pct(df: pd.DataFrame) -> pd.Series:
    volume_col = find_column(df, VOLUME_CHANGE_CANDIDATES)
    if volume_col is not None:
        return pd.to_numeric(df[volume_col], errors="coerce").abs()

    required = {"input_voxels", "target_voxels"}
    if required.issubset(df.columns):
        input_voxels = pd.to_numeric(df["input_voxels"], errors="coerce")
        target_voxels = pd.to_numeric(
            df["target_voxels"],
            errors="coerce",
        ).replace(0, np.nan)
        return 100.0 * (input_voxels - target_voxels).abs() / target_voxels

    raise ValueError(
        "Could not find or derive input-vs-target volume change. "
        "Expected a volume-change percentage column or both "
        "'input_voxels' and 'target_voxels'."
    )


def derive_score(df: pd.DataFrame, metric: str) -> pd.Series:
    canonical = METRIC_ALIASES.get(metric, metric)

    if canonical == "one_minus_dice_input_vs_recon":
        if "dice_input_vs_recon" not in df.columns:
            raise ValueError("Missing column: dice_input_vs_recon")
        dice = pd.to_numeric(df["dice_input_vs_recon"], errors="coerce")
        return 1.0 - dice

    if canonical == "relative_volume_difference_input_vs_recon":
        required = {"input_voxels", "recon_voxels"}
        if not required.issubset(df.columns):
            raise ValueError(
                "Relative volume difference requires input_voxels and recon_voxels."
            )
        input_voxels = pd.to_numeric(
            df["input_voxels"],
            errors="coerce",
        ).replace(0, np.nan)
        recon_voxels = pd.to_numeric(df["recon_voxels"], errors="coerce")
        return (input_voxels - recon_voxels).abs() / input_voxels

    if canonical not in df.columns:
        raise ValueError(
            f"Missing metric column '{canonical}'. "
            f"Available columns: {list(df.columns)}"
        )

    return pd.to_numeric(df[canonical], errors="coerce")


def prepare_roc(
    csv_path: Path,
    metric: str,
    volume_threshold_pct: float,
    exclude_shifts: bool,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    df = pd.read_csv(csv_path)

    probe_col = find_column(df, PROBE_CANDIDATES)
    if exclude_shifts and probe_col is not None:
        df = df.loc[~is_shift_probe(df[probe_col])].copy()

    df = deduplicate_clean_rows(df, probe_col)

    volume_change_pct = derive_volume_change_pct(df)
    labels = (volume_change_pct > volume_threshold_pct).astype(float)

    if probe_col is not None:
        labels.loc[is_clean_probe(df[probe_col])] = 0.0

    scores = derive_score(df, metric)
    valid = labels.notna() & scores.notna()

    y_true = labels.loc[valid].astype(int).to_numpy()
    y_score = scores.loc[valid].to_numpy(dtype=float)

    if np.unique(y_true).size != 2:
        raise ValueError(
            f"{csv_path} does not contain both classes for metric '{metric}' "
            "after filtering."
        )

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    n_positive = int(y_true.sum())
    n_negative = int((y_true == 0).sum())
    return fpr, tpr, roc_auc, n_positive, n_negative


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot separate multi-model ROC comparison figures for Dice, HD95, "
            "ASSD, and relative volume difference."
        )
    )
    parser.add_argument(
        "--model", action="append", required=True, metavar="NAME=CSV",
        help="Repeat for each model, preserving the desired legend order.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=(
            "Metrics to plot. Defaults: one_minus_dice_input_vs_recon, "
            "hd95_input_vs_recon, assd_input_vs_recon, "
            "relative_volume_difference_input_vs_recon."
        ),
    )
    parser.add_argument(
        "--volume-threshold-pct",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--exclude-shifts",
        action="store_true",
    )
    parser.add_argument(
        "--roi",
        default="ROI",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("roc_by_metric"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for item in args.model:
        if "=" not in item:
            raise ValueError(f"--model must be NAME=CSV, got: {item}")
        name, csv_path = item.split("=", 1)
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Model CSV not found: {path}")
        models.append((name, path))

    summary_rows: list[dict[str, object]] = []

    for requested_metric in args.metrics:
        metric = METRIC_ALIASES.get(requested_metric, requested_metric)
        metric_label = METRIC_LABELS.get(metric, metric)

        fig, ax = plt.subplots(figsize=(7.2, 6.4))

        for model_name, csv_path in models:
            fpr, tpr, roc_auc, n_pos, n_neg = prepare_roc(
                csv_path=csv_path,
                metric=metric,
                volume_threshold_pct=args.volume_threshold_pct,
                exclude_shifts=args.exclude_shifts,
            )

            ax.plot(
                fpr,
                tpr,
                linewidth=2.2,
                label=f"{model_name} (AUC={roc_auc:.3f})",
            )

            summary_rows.append(
                {
                    "roi": args.roi,
                    "metric": metric,
                    "model": model_name,
                    "auc": roc_auc,
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "volume_threshold_pct": args.volume_threshold_pct,
                    "shifts_excluded": bool(args.exclude_shifts),
                }
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.2,
            label="Chance",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")

        shift_text = "excluding shifts" if args.exclude_shifts else "including shifts"
        ax.set_title(
            f"{args.roi}: ROC comparison by {metric_label}\n"
            f"Volume-change threshold > {args.volume_threshold_pct:g}%, "
            f"{shift_text}"
        )

        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()

        stem = safe_filename(f"{args.roi}_{metric}_multi_model_roc")
        png_path = args.output_dir / f"{stem}.png"
        pdf_path = args.output_dir / f"{stem}.pdf"

        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

    summary_path = args.output_dir / safe_filename(
        f"{args.roi}_multi_model_auc_by_metric.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
