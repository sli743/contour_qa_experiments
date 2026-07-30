#!/usr/bin/env python3
"""
Plot ROC curves from the three contour-QA experiments on one figure.

The script accepts either:
1. ROC-coordinate CSV files containing FPR/TPR columns, or
2. detailed probe-evaluation CSV files containing labels plus a score column.

Examples
--------
Using ROC-coordinate CSVs:
    python plot_three_model_roc_comparison.py \
        --pilot pilot_roc.csv \
        --benchmark benchmark_roc.csv \
        --random-morphology random_morphology_roc.csv \
        --output three_model_roc_comparison.png

Using detailed probe CSVs:
    python plot_three_model_roc_comparison.py \
        --pilot pilot_probe_eval_sweep_detailed.csv \
        --benchmark benchmark_probe_eval_sweep_detailed.csv \
        --random-morphology random_probe_eval_sweep_detailed.csv \
        --score-column assd_input_recon \
        --volume-threshold-pct 10 \
        --exclude-shifts \
        --output three_model_roc_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


FPR_CANDIDATES = (
    "fpr",
    "false_positive_rate",
    "false positive rate",
)

TPR_CANDIDATES = (
    "tpr",
    "true_positive_rate",
    "sensitivity",
    "true positive rate",
)

LABEL_CANDIDATES = (
    "label",
    "is_positive",
    "positive",
    "target",
    "y_true",
)

PROBE_CANDIDATES = (
    "probe",
    "probe_type",
    "corruption",
    "corruption_type",
    "perturbation",
    "perturbation_type",
)

VOLUME_CHANGE_CANDIDATES = (
    "abs_volume_change_pct",
    "absolute_volume_change_pct",
    "volume_change_pct",
    "input_target_volume_change_pct",
    "input_vs_target_volume_change_pct",
    "volume_diff_pct",
)

SCORE_ALIASES = {
    "assd_input_recon": "assd_input_vs_recon",
    "hd95_input_recon": "hd95_input_vs_recon",
    "dice_input_recon": "dice_input_vs_recon",
    "one_minus_dice_input_recon": "dice_input_vs_recon",
    "volume_diff_input_recon": "derived_volume_diff_input_vs_recon",
}

PATIENT_CANDIDATES = (
    "patient_id",
    "patient",
    "case_id",
    "case",
)


def normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in df.columns}


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = normalize_columns(df)
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def is_shift_probe(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower()
    return text.str.contains(r"shift|translation|translated", regex=True, na=False)


def is_clean_probe(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower().str.strip()
    return text.isin({"clean", "none", "identity", "original", "approved"})


def deduplicate_clean_rows(df: pd.DataFrame, probe_col: str | None) -> pd.DataFrame:
    """
    Keep one clean row per patient when a patient identifier is available.
    This prevents clean contours from being overrepresented because some
    sweep files repeat the clean row for multiple probe families.
    """
    if probe_col is None:
        return df

    patient_col = find_column(df, PATIENT_CANDIDATES)
    if patient_col is None:
        return df

    clean = is_clean_probe(df[probe_col])
    clean_rows = df.loc[clean].drop_duplicates(subset=[patient_col], keep="first")
    corrupted_rows = df.loc[~clean]
    return pd.concat([clean_rows, corrupted_rows], ignore_index=True)


def load_roc_coordinates(
    path: Path,
    score_column: str | None,
    volume_threshold_pct: float,
    exclude_shifts: bool,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    df = pd.read_csv(path)

    fpr_col = find_column(df, FPR_CANDIDATES)
    tpr_col = find_column(df, TPR_CANDIDATES)

    if fpr_col is not None and tpr_col is not None:
        roc_df = (
            df[[fpr_col, tpr_col]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .sort_values(fpr_col)
        )
        if roc_df.empty:
            raise ValueError(f"No valid ROC coordinates in {path}")

        fpr = roc_df[fpr_col].to_numpy(dtype=float)
        tpr = roc_df[tpr_col].to_numpy(dtype=float)

        # Add endpoints when absent.
        if fpr[0] > 0 or tpr[0] > 0:
            fpr = np.insert(fpr, 0, 0.0)
            tpr = np.insert(tpr, 0, 0.0)
        if fpr[-1] < 1 or tpr[-1] < 1:
            fpr = np.append(fpr, 1.0)
            tpr = np.append(tpr, 1.0)

        return fpr, tpr, float(auc(fpr, tpr)), -1, -1

    if score_column is None:
        raise ValueError(
            f"{path} does not contain recognizable FPR/TPR columns. "
            "Provide --score-column to construct the ROC curve from detailed rows."
        )

    requested_score_column = score_column
    if score_column not in df.columns:
        alias = SCORE_ALIASES.get(score_column)
        if alias in df.columns:
            score_column = alias
        elif alias == "derived_volume_diff_input_vs_recon":
            required = {"input_voxels", "recon_voxels"}
            if required.issubset(df.columns):
                denom = pd.to_numeric(df["input_voxels"], errors="coerce").replace(0, np.nan)
                recon = pd.to_numeric(df["recon_voxels"], errors="coerce")
                df[alias] = (recon - denom).abs() / denom
                score_column = alias
            else:
                raise ValueError(
                    f"Score column '{requested_score_column}' is absent from {path}, "
                    "and input_voxels/recon_voxels are unavailable for derivation.\n"
                    f"Available columns: {list(df.columns)}"
                )
        else:
            raise ValueError(
                f"Score column '{requested_score_column}' is absent from {path}.\n"
                f"Available columns: {list(df.columns)}\n"
                "Suggested names for this dataset include: "
                "assd_input_vs_recon, hd95_input_vs_recon, dice_input_vs_recon."
            )

    probe_col = find_column(df, PROBE_CANDIDATES)

    if exclude_shifts and probe_col is not None:
        df = df.loc[~is_shift_probe(df[probe_col])].copy()

    df = deduplicate_clean_rows(df, probe_col)

    label_col = find_column(df, LABEL_CANDIDATES)
    if label_col is not None:
        labels = pd.to_numeric(df[label_col], errors="coerce")
    else:
        volume_col = find_column(df, VOLUME_CHANGE_CANDIDATES)
        if volume_col is not None:
            volume_change = pd.to_numeric(df[volume_col], errors="coerce").abs()
        elif {"input_voxels", "target_voxels"}.issubset(df.columns):
            input_voxels = pd.to_numeric(df["input_voxels"], errors="coerce")
            target_voxels = pd.to_numeric(df["target_voxels"], errors="coerce").replace(0, np.nan)
            volume_change = 100.0 * (input_voxels - target_voxels).abs() / target_voxels
        else:
            raise ValueError(
                f"Could not find or derive volume-change percentage in {path}.\n"
                "Expected a volume-change column or input_voxels and target_voxels.\n"
                f"Available columns: {list(df.columns)}"
            )

        labels = (volume_change > volume_threshold_pct).astype(float)

        # Force clean contours to negative even if a malformed value is present.
        if probe_col is not None:
            labels.loc[is_clean_probe(df[probe_col])] = 0.0

    scores = pd.to_numeric(df[score_column], errors="coerce")
    if requested_score_column == "one_minus_dice_input_recon":
        scores = 1.0 - scores
    valid = labels.notna() & scores.notna()
    labels = labels.loc[valid].astype(int).to_numpy()
    scores = scores.loc[valid].to_numpy(dtype=float)

    if np.unique(labels).size != 2:
        raise ValueError(
            f"{path} does not contain both positive and negative classes after filtering."
        )

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    n_positive = int(labels.sum())
    n_negative = int((labels == 0).sum())
    return fpr, tpr, roc_auc, n_positive, n_negative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw one ROC comparison figure for pilot, benchmark-leakage, "
                    "and random-morphology contour reconstruction models."
    )
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--random-morphology", type=Path, required=True)
    parser.add_argument(
        "--score-column",
        help="Score column used when inputs are detailed probe-evaluation CSVs.",
    )
    parser.add_argument(
        "--volume-threshold-pct",
        type=float,
        default=10.0,
        help="Positive-class threshold for detailed CSVs (default: 10).",
    )
    parser.add_argument(
        "--exclude-shifts",
        action="store_true",
        help="Exclude shift/translation probes before constructing ROC curves.",
    )
    parser.add_argument(
        "--title",
        default="ROC comparison of contour reconstruction models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("three_model_roc_comparison.png"),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models = [
        ("Pilot AE", args.pilot),
        ("Benchmark-leakage DAE", args.benchmark),
        ("Random-morphology DAE", args.random_morphology),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    for label, path in models:
        fpr, tpr, roc_auc, n_pos, n_neg = load_roc_coordinates(
            path=path,
            score_column=args.score_column,
            volume_threshold_pct=args.volume_threshold_pct,
            exclude_shifts=args.exclude_shifts,
        )

        sample_text = ""
        if n_pos >= 0 and n_neg >= 0:
            sample_text = f", n+={n_pos}, n-={n_neg}"

        ax.plot(
            fpr,
            tpr,
            linewidth=2.2,
            label=f"{label} (AUC={roc_auc:.3f}{sample_text})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {args.output}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
