"""Interaction plot for the corruption x CT ablation (reads ablation_arm_summary.csv).

Each panel: x = training corruption (no / yes), one line per CT input (no / yes),
95% paired-bootstrap CIs; the CT-only segmenter is a dashed reference where it applies.

  python -m analysis.plot_ablation_corruption_ct --ablation-dir <...>/seed91_ablation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"I0": ("Contour only", "#2a78d6", "o"), "I1": ("CT + contour", "#eb6834", "s")}

PANELS = [
    ("rho_deep_common_ml", "Deep change ≥ 4.5 mm", "Spearman ρ vs true error", True),
    ("ap_deep_common_ml", "Top-10 AP, deep change ≥ 4.5 mm", "average precision, top-10 true error", True),
    ("rho_qa_hd95_mm", "QA HD95", "Spearman ρ vs true error", True),
    ("mean_change_precision", "Change precision", "share of change toward approved", True),
    ("clean_p90_deep_common_ml", "False alarm, approved input", "deep change at 4.5 mm, p90 (mL)", False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", required=True)
    ap.add_argument("--reference-arm", default="ct_only")
    args = ap.parse_args()
    d = Path(args.ablation_dir)
    s = pd.read_csv(d / "ablation_arm_summary.csv").set_index("arm")
    meta_path = d / "ablation_metadata.json"
    meta = json.loads(meta_path.read_text().replace("NaN", "null")) if meta_path.exists() else {}
    per_seed_path = d / "ablation_per_seed.csv"
    per_seed = pd.read_csv(per_seed_path) if per_seed_path.exists() else None
    if "seeds" in meta:
        seed_ids = [str(k).removeprefix("seed") for k in meta["seeds"]]
        seed_text = f"mean of seeds {', '.join(seed_ids)} (small hollow markers = single seeds)"
    else:
        seed_text = f"seed {meta.get('seed', '?')}"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.6 * len(PANELS), 3.9), facecolor=SURFACE)
    for ax, (key, title, ylab, show_ref) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        labels = {0: [], 1: []}
        for i_key, (lab, col, mk) in SERIES.items():
            cells = [f"C0{i_key}", f"C1{i_key}"]
            if not all(c in s.index and pd.notna(s.loc[c, key]) for c in cells):
                continue
            y = [s.loc[c, key] for c in cells]
            lo = [s.loc[c, key] - s.loc[c, f"{key}_lo"] for c in cells]
            hi = [s.loc[c, f"{key}_hi"] - s.loc[c, key] for c in cells]
            xs = [0 - 0.04, 1 - 0.04] if i_key == "I0" else [0 + 0.04, 1 + 0.04]
            ax.errorbar(xs, y, yerr=[lo, hi], color=col, marker=mk, markersize=8, linewidth=2,
                        elinewidth=1.2, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
            if per_seed is not None and key in per_seed:
                for x, c in zip(xs, cells):
                    vals = per_seed.loc[per_seed.arm == c, key].dropna()
                    ax.scatter([x] * len(vals), vals, s=16, marker=mk,
                               facecolors="none", edgecolors=col, linewidths=1, alpha=0.8, zorder=2)
            labels[0].append((xs[0], y[0]))
            labels[1].append((xs[1], y[1]))
        lo_y, hi_y = ax.get_ylim()
        min_gap = 0.07 * (hi_y - lo_y)
        for side, pts in labels.items():
            pts = sorted(pts, key=lambda p: p[1])
            ys = [p[1] for p in pts]
            if len(ys) == 2 and ys[1] - ys[0] < min_gap:
                mid = (ys[0] + ys[1]) / 2
                ys = [mid - min_gap / 2, mid + min_gap / 2]
            for (x, yv), ty in zip(pts, ys):
                ax.annotate(f"{yv:.2f}", (x, yv), xytext=(x + (0.09 if side else -0.09), ty), textcoords="data",
                            va="center", ha="left" if side else "right", color=INK, fontsize=9)
        if show_ref and args.reference_arm in s.index and pd.notna(s.loc[args.reference_arm, key]):
            r = s.loc[args.reference_arm, key]
            ax.axhline(r, color=INK2, linestyle=(0, (4, 3)), linewidth=1, zorder=1)
            ax.annotate("CT-only segmenter", (0.5, r), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", color=INK2, fontsize=8)
        ax.set_xticks([0, 1], ["no corruption", "corruption"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.axhline(0, color=INK2, linewidth=0.6) if key.startswith("rho") else None
    handles = [plt.Line2D([], [], color=c, marker=m, markersize=8, linewidth=2, label=l) for l, c, m in SERIES.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02),
               labelcolor=INK)
    n = int(s.loc["C1I1", "n"])
    fig.text(0.01, -0.02, f"Lung_L Month3 actual autoseg, n={n}, {seed_text}. Bars: 95% paired bootstrap over patients "
             f"(case sampling only, not training variance).", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "svg"):
        fig.savefig(d / f"ablation_interaction.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {d / 'ablation_interaction.png'}")


if __name__ == "__main__":
    main()
