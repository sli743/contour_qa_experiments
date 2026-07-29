#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot input-target Dice vs input-recon Dice from probe eval CSV."
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
        help="Output image path. Default: same folder as CSV / input_target_vs_input_recon.png",
    )
    parser.add_argument(
        "--fig_w",
        type=float,
        default=10.0,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--fig_h",
        type=float,
        default=7.0,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=12.0,
        help="Scatter marker size",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["probe_type", "dice_input_vs_target", "dice_input_vs_recon"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

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

    if args.out is None:
        out_path = csv_path.parent / "input_target_vs_input_recon.png"
    else:
        out_path = Path(args.out)

    plt.figure(figsize=(args.fig_w, args.fig_h))

    for probe in probe_types:
        sub = df[df["probe_type"] == probe]
        plt.scatter(
            sub["dice_input_vs_target"],
            sub["dice_input_vs_recon"],
            s=args.marker_size,
            label=probe,
        )

    plt.xlabel("Input-Target Dice", fontsize=14)
    plt.ylabel("Input-Recon Dice", fontsize=14)
    plt.title("Probe Evaluation: Input-Target vs Input-Recon Dice", fontsize=16)
    plt.tick_params(axis="both", labelsize=12)
    plt.grid(True, alpha=0.3)

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=10,
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()