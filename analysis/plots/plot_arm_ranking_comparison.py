#!/usr/bin/env python3
"""Compare how two QA arms rank OOD Month-3 autosegmentations (ranking view, not AUC).

Reads <study>/evaluations_floor_corrected/<floor-dir>/<arm>_floor_corrected_scores.csv for
both arms (the study may be a folder of junctions pairing arms from different runs).

Figure (rows = scores qa_hd95_mm and qa_change_deep4.5_ml):
  a) OOD patients captured in the top-k, per arm, with the random-ranking expectation
  b) each OOD patient's rank under arm B (x) vs arm A (y); below the diagonal = A ranks it higher
  c) Spearman of the QA score with the true error (1-Dice, HD95) per arm

Example:
  python -m analysis.plots.plot_arm_ranking_comparison --ood-xlsx ood_copy.xlsx --study <study> \
      --roi Lung_L --label-a "NSR-QA v3 fit" --label-b "No corruption" --watch 47,135,148
Writes <study>/analysis/arm_ranking_comparison_<roi>.png and a per-patient .csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from analysis.plots.plot_ood_vs_qa_scores import GRID, INK, MUTED, SURF, load_ood, pid_str

C_A, C_B, C_RAND = "#2a78d6", "#eb6834", "#8a8984"
# (column, label, tie-break column): HD95 is voxel-quantized, so many patients tie
SCORES = [("qa_hd95_mm", "QA HD95 (mm)", "qa_hd99_mm"),
          ("qa_change_deep4.5_ml", "Volume changed ≥4.5 mm (mL)", "qa_change_deep3_ml")]
TRUE = [("true_1_minus_dice", "true 1-Dice"), ("true_hd95_mm", "true HD95")]


def load_arm(study: Path, floor_dir: str, arm: str) -> dict[str, dict]:
    f = study / "evaluations_floor_corrected" / floor_dir / f"{arm}_floor_corrected_scores.csv"
    return {pid_str(r["patient_id"]): r for r in csv.DictReader(f.open(encoding="utf-8"))}


def ranks_desc(vals: dict[str, float], tie: dict[str, float]) -> dict[str, int]:
    order = sorted(vals, key=lambda p: (-vals[p], -tie[p]))
    return {p: i + 1 for i, p in enumerate(order)}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-xlsx", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--arm-a", default="nsrqa_corruption")
    ap.add_argument("--arm-b", default="no_corruption")
    ap.add_argument("--label-a", default="NSR-QA corruption")
    ap.add_argument("--label-b", default="No corruption")
    ap.add_argument("--floor-dir", default="seed91")
    ap.add_argument("--watch", default="", help="comma-separated patient ids to annotate")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    study = Path(args.study)
    ood = load_ood(Path(args.ood_xlsx), args.roi)
    A = load_arm(study, args.floor_dir, args.arm_a)
    B = load_arm(study, args.floor_dir, args.arm_b)
    pats = sorted(set(A) & set(B), key=lambda p: int(p) if p.isdigit() else p)
    n = len(pats)
    is_ood = {p: p in ood for p in pats}
    n_ood = sum(is_ood.values())
    watch = {w.strip() for w in args.watch.split(",") if w.strip()}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), gridspec_kw={"width_ratios": [1.25, 1, 0.8]})
    fig.patch.set_facecolor("white")
    rows_out = {p: {"patient_id": p, "ood": int(is_ood[p])} for p in pats}

    for r, (col, name, tcol_tie) in enumerate(SCORES):
        va = {p: float(A[p][col]) for p in pats}
        vb = {p: float(B[p][col]) for p in pats}
        ra = ranks_desc(va, {p: float(A[p][tcol_tie]) for p in pats})
        rb = ranks_desc(vb, {p: float(B[p][tcol_tie]) for p in pats})
        for p in pats:
            rows_out[p].update({f"{col}_A": va[p], f"{col}_B": vb[p], f"rank_{col}_A": ra[p], f"rank_{col}_B": rb[p]})

        # a) OOD captured in top-k
        ax = axes[r, 0]
        style(ax)
        k = np.arange(1, n + 1)
        for rk, color, lab in ((ra, C_A, args.label_a), (rb, C_B, args.label_b)):
            hits = np.zeros(n)
            for p in pats:
                if is_ood[p]:
                    hits[rk[p] - 1] += 1
            cum = np.cumsum(hits)
            ap_score = np.mean([cum[rk[p] - 1] / rk[p] for p in pats if is_ood[p]])
            ax.step(k, cum, where="post", color=color, lw=2,
                    label=f"{lab}  (top10 {int(cum[9])}/{n_ood}, top20 {int(cum[19])}/{n_ood}, AP {ap_score:.2f})")
        ax.plot(k, k * n_ood / n, color=C_RAND, lw=1.2, ls="--", label=f"random (AP {n_ood / n:.2f})")
        for kk in (10, 20):
            ax.axvline(kk, color=GRID, lw=1.2, zorder=0)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n_ood + 0.5)
        ax.set_xlabel(f"top-k flagged by {name}", color=INK, fontsize=9)
        ax.set_ylabel("OOD cases captured", color=INK, fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        ax.set_title(f"{name}: OOD captured in top-k (n={n}, {n_ood} OOD)\nties broken by {tcol_tie}",
                     color=INK, fontsize=10, loc="left")

        # b) OOD ranks, B vs A
        ax = axes[r, 1]
        style(ax)
        ax.plot([1, n], [1, n], color=C_RAND, lw=1, ls="--")
        ax.axhspan(0.5, 10.5, color=C_A, alpha=0.06, lw=0)
        ax.axvspan(0.5, 10.5, color=C_B, alpha=0.06, lw=0)
        xs = [rb[p] for p in pats if is_ood[p]]
        ys = [ra[p] for p in pats if is_ood[p]]
        ax.scatter(xs, ys, s=46, color=INK, edgecolor="white", lw=1.5, zorder=3)
        better = sum(y < x for x, y in zip(xs, ys))
        for p in pats:
            if is_ood[p] and (p in watch or min(ra[p], rb[p]) <= 10):
                ax.annotate(p, (rb[p], ra[p]), xytext=(4, 3), textcoords="offset points", fontsize=7.5,
                            color=INK if p in watch else MUTED, fontweight="bold" if p in watch else "normal")
        ax.set_ylim(n + 1, 0)
        ax.set_xlim(n + 1, 0)
        ax.set_xlabel(f"rank under {args.label_b} (1 = most suspicious)", color=INK, fontsize=9)
        ax.set_ylabel(f"rank under {args.label_a}", color=INK, fontsize=9)
        mra, mrb = np.median(ys), np.median(xs)
        ax.set_title(f"OOD ranks: {better}/{n_ood} ranked higher by {args.label_a}\n"
                     f"median rank {mra:.0f} vs {mrb:.0f}  (Wilcoxon p={stats.wilcoxon(ys, xs).pvalue:.3f})",
                     color=INK, fontsize=9.5, loc="left")

        # c) Spearman vs true error
        ax = axes[r, 2]
        style(ax)
        ax.grid(axis="x", visible=False)
        w = 0.36
        for j, (tcol, tlab) in enumerate(TRUE):
            for off, arm_rows, vals, color in ((-w / 2, A, va, C_A), (w / 2, B, vb, C_B)):
                rho = stats.spearmanr([vals[p] for p in pats], [float(arm_rows[p][tcol]) for p in pats]).statistic
                ax.bar(j + off, rho, width=w - 0.04, color=color)
                ax.text(j + off, rho + 0.02, f"{rho:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(TRUE)), [t for _, t in TRUE], fontsize=8.5, color=INK)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Spearman ρ with true error", color=INK, fontsize=9)
        ax.set_title(f"{name}\nvs true error (all {n})", color=INK, fontsize=9.5, loc="left")

    handles = [plt.Line2D([], [], color=C_A, lw=6, label=args.label_a),
               plt.Line2D([], [], color=C_B, lw=6, label=args.label_b)]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(args.title or f"{args.roi}: {args.label_a} vs {args.label_b} — ranking of OOD autosegmentations",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = study / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"arm_ranking_comparison_{args.roi.lower()}.png"
    fig.savefig(png, dpi=150)
    with (out / f"arm_ranking_comparison_{args.roi.lower()}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(rows_out.values())).keys()))
        w.writeheader()
        w.writerows(rows_out.values())
    print(png)


if __name__ == "__main__":
    main()
