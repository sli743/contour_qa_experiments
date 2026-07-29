#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Box plot of input-recon Dice per probe type"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to probe_eval_detailed.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort probe types by median Dice",
    )
    parser.add_argument(
        "--fig_w",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--fig_h",
        type=float,
        default=6,
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    if "probe_type" not in df.columns or "dice_input_vs_recon" not in df.columns:
        raise ValueError("CSV must contain 'probe_type' and 'dice_input_vs_recon'")

    # Preferred order (same as scatter plot)
    preferred_order = [
        "clean",
        "shifted",
        "eroded",
        "dilated",
        "strong",
        "cut_lower_half",
        "cut_upper_half",
        "cut_posterior_half",
        "big_bite",
        "big_blob",
    ]

    probe_types = [p for p in preferred_order if p in df["probe_type"].unique()]
    probe_types += [p for p in df["probe_type"].unique() if p not in probe_types]

    # Optional sorting by median
    if args.sort:
        medians = df.groupby("probe_type")["dice_input_vs_recon"].median()
        probe_types = sorted(probe_types, key=lambda x: medians.get(x, 0))

    data = [
        df[df["probe_type"] == p]["dice_input_vs_recon"].values
        for p in probe_types
    ]

    # Output path
    if args.out is None:
        out_path = csv_path.parent / "boxplot_input_recon_dice.png"
    else:
        out_path = Path(args.out)

    # Plot
    plt.figure(figsize=(args.fig_w, args.fig_h))

    plt.boxplot(
        data,
        labels=probe_types,
        showfliers=False,   # cleaner
    )

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Input-Recon Dice", fontsize=12)
    plt.title("Input-Recon Dice Distribution per Probe Type", fontsize=14)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved boxplot to: {out_path}")


if __name__ == "__main__":
    main()