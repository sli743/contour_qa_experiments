#!/usr/bin/env python3
"""Box plots of a QA score for OOD vs non-OOD Month-3 autosegmentations, per arm.

OOD = rows of the OOD domain workbook for this ROI with "failed limit" == P.
Scores:
  hd95   qa_hd95_mm from evaluations/<arm>/seed91/actual_autoseg_cases/.../metrics.json
  deep   qa_change_deep4.5_ml from evaluations_floor_corrected/<dir>/<arm>_floor_corrected_scores.csv

Example:
  python -m analysis.plots.plot_ood_vs_qa_scores --ood-xlsx ".../OOD_Domain_Assignment_MAD_TOTAL copy.xlsx" \
      --study D:/aiqa/contour_QA/lung_r_..._rcal_v1 --roi Lung_R --score deep
Writes <study>/analysis/ood_vs_qa_{hd95,deep4p5}_<roi>.png and a .csv with the per-patient values.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ARMS = [("nsrqa_corruption", "NSR-QA corruption model"), ("no_corruption", "No-corruption model")]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_IN, C_OOD = "#2a78d6", "#eb6834"
MARK = {"AC": "o", "AV": "^", "I": "s", "H": "v", "GC": "P", "MI": "X", "NDI": "*"}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xlsx_first_sheet(path: Path) -> list[dict]:
    """Minimal stdlib xlsx reader (first sheet, header row -> dicts)."""
    # OneDrive placeholders can refuse Python reads; copy the file locally first
    # (e.g. PowerShell Copy-Item) if this raises PermissionError.
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        sheet = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))[0]
        rows = []
        for r in ET.fromstring(z.read(sheet)).iter(f"{{{NS['m']}}}row"):
            vals = {}
            for c in r.findall("m:c", NS):
                col = re.match(r"[A-Z]+", c.get("r")).group(0)
                v = c.find("m:v", NS)
                if c.get("t") == "inlineStr":
                    text = "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))
                elif v is None:
                    text = ""
                elif c.get("t") == "s":
                    text = shared[int(v.text)]
                else:
                    text = v.text or ""
                vals[col] = text
            rows.append(vals)
    cols = sorted({k for r in rows for k in r}, key=lambda s: (len(s), s))
    header = [rows[0].get(c, "").strip() for c in cols]
    out = []
    for r in rows[1:]:
        out.append({h: r.get(c, "").strip() for h, c in zip(header, cols) if h})
    return out


def pid_str(x: str) -> str:
    try:
        return str(int(float(x)))
    except ValueError:
        return x.strip()


def load_ood(xlsx: Path, roi: str) -> dict[str, set]:
    ood: dict[str, set] = {}
    for r in read_xlsx_first_sheet(xlsx):
        if r.get("roi") == roi and r.get("failed limit", "").upper() == "P":
            ood.setdefault(pid_str(r["patient_id"]), set()).add(r.get("domain", ""))
    return ood


def load_scores(study: Path, roi: str, score: str, floor_dir: str) -> dict[str, dict[str, float]]:
    data = {}
    for arm, _ in ARMS:
        if score == "hd95":
            base = study / "evaluations" / arm / "seed91" / "actual_autoseg_cases" / "autoseg_centered" / roi / "testing1"
            data[arm] = {p.name.removeprefix("patient_"): float(json.loads((p / "metrics.json").read_text())["qa_hd95_mm"])
                         for p in base.glob("patient_*")}
        else:
            f = study / "evaluations_floor_corrected" / floor_dir / f"{arm}_floor_corrected_scores.csv"
            data[arm] = {pid_str(r["patient_id"]): float(r["qa_change_deep4.5_ml"])
                         for r in csv.DictReader(f.open(encoding="utf-8"))}
    return data


def repel(ys, gap):
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        out[order[i]] = max(out[order[i]], out[order[i - 1]] + gap)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-xlsx", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--score", choices=["hd95", "deep"], required=True)
    ap.add_argument("--floor-dir", default="seed91")
    ap.add_argument("--label", default="", help="extra title text, e.g. severity source")
    ap.add_argument("--top-non-ood", type=int, default=5)
    args = ap.parse_args()

    study = Path(args.study)
    roi = args.roi
    ood = load_ood(Path(args.ood_xlsx), roi)
    data = load_scores(study, roi, args.score, args.floor_dir)
    pids = sorted(set.intersection(*(set(v) for v in data.values())), key=int)
    is_ood = {p: p in ood for p in pids}
    unit = "mm" if args.score == "hd95" else "mL"
    ylabel = ("QA HD95 (mm), autoseg vs reconstruction" if args.score == "hd95"
              else "Volume changed by >= 4.5 mm (mL)\nautoseg vs reconstruction")
    what = "QA HD95" if args.score == "hd95" else "volume changed by >= 4.5 mm"

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True, facecolor=SURF)
    rng = np.random.default_rng(0)
    gap = max(max(v.values()) for v in data.values()) * 0.035
    summary = []
    for ax, (arm, title) in zip(axes, ARMS):
        ax.set_facecolor(SURF)
        v = data[arm]
        non = [p for p in pids if not is_ood[p]]
        oo = [p for p in pids if is_ood[p]]
        groups = [[v[p] for p in non], [v[p] for p in oo]]
        bp = ax.boxplot(groups, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 2}, whiskerprops={"color": MUTED},
                        capprops={"color": MUTED})
        for patch, c in zip(bp["boxes"], (C_IN, C_OOD)):
            patch.set_facecolor(c); patch.set_alpha(0.22); patch.set_edgecolor(c); patch.set_linewidth(1.5)

        if args.score == "hd95":
            # HD95 is quantized: spread tied points evenly and list OOD IDs per level.
            for g, members in ((0, non), (1, oo)):
                levels: dict[float, list] = {}
                for p in members:
                    levels.setdefault(round(v[p], 2), []).append(p)
                for lv, ps in levels.items():
                    ps = sorted(ps, key=int)
                    xs = g + (np.linspace(-0.2, 0.2, len(ps)) if len(ps) > 1 else np.zeros(1))
                    for x, p in zip(xs, ps):
                        if g:
                            ax.scatter(x, v[p], marker=MARK.get(sorted(ood[p])[0], "D"), s=46, color=C_OOD,
                                       edgecolor=SURF, linewidth=1.2, zorder=3)
                        else:
                            ax.scatter(x, v[p], s=30, color=C_IN, edgecolor=SURF, linewidth=1.0, zorder=3, alpha=0.85)
                    if g:
                        ax.text(1.33, lv, ", ".join(ps), va="center", ha="left", fontsize=8, color=MUTED)
            ax.set_xlim(-0.5, 2.05)
        else:
            xn = {p: rng.uniform(-0.18, 0.18) for p in non}
            for p in non:
                ax.scatter(xn[p], v[p], s=30, color=C_IN, edgecolor=SURF, linewidth=1.0, zorder=3, alpha=0.85)
            top_non = sorted(non, key=lambda p: -v[p])[: args.top_non_ood]
            for p, y in zip(top_non, repel([v[p] for p in top_non], gap)):
                ax.annotate(p, (xn[p], v[p]), xytext=(-0.5, y), textcoords="data", fontsize=8, color=MUTED,
                            va="center", ha="left", arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8})
            xo = {p: 1 + rng.uniform(-0.18, 0.18) for p in oo}
            for p in oo:
                ax.scatter(xo[p], v[p], marker=MARK.get(sorted(ood[p])[0], "D"), s=46, color=C_OOD,
                           edgecolor=SURF, linewidth=1.2, zorder=3)
            for p, y in zip(oo, repel([v[p] for p in oo], gap)):
                ax.annotate(p, (xo[p], v[p]), xytext=(1.38, y), textcoords="data", fontsize=8, color=MUTED,
                            va="center", ha="left", arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8})
            ax.set_xlim(-0.55, 1.75)

        u = stats.mannwhitneyu(groups[1], groups[0], alternative="greater")
        m0, m1 = np.median(groups[0]), np.median(groups[1])
        rk = {p: i + 1 for i, p in enumerate(sorted(pids, key=lambda q: -v[q]))}
        summary.append((arm, len(non), len(oo), m0, m1, u.pvalue, {p: rk[p] for p in oo},
                        [(p, round(v[p], 2)) for p in sorted(non, key=lambda q: -v[q])[:5]]))
        ax.set_xticks([0, 1], [f"Not OOD\n(n={len(non)})", f"OOD\n(n={len(oo)})"])
        ax.set_title(f"{title}\nmedian {m0:.2f} vs {m1:.2f} {unit}, one-sided Mann-Whitney p={u.pvalue:.3f}",
                     fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(ylabel)
    handles = [plt.Line2D([], [], marker=m, linestyle="", color=C_OOD, markersize=7, label=f"OOD domain {d}")
               for d, m in MARK.items() if any(d in ood[p] for p in pids if is_ood[p])]
    handles.insert(0, plt.Line2D([], [], marker="o", linestyle="", color=C_IN, markersize=7, label="Not OOD"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    extra = f" - {args.label}" if args.label else ""
    fig.suptitle(f"{roi} Month-3 test set: {what}, OOD vs non-OOD (seed 91){extra}",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    tag = "hd95" if args.score == "hd95" else "deep4p5"
    out = study / "analysis" / f"ood_vs_qa_{tag}_{roi.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURF)
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "ood", "ood_domains"] + [f"{tag}_{a}" for a, _ in ARMS])
        for p in pids:
            w.writerow([p, int(is_ood[p]), "|".join(sorted(ood.get(p, [])))] + [data[a][p] for a, _ in ARMS])
    print("OOD not evaluated:", sorted((p for p in ood if p.isdigit() and len(p) < 6 and p not in pids), key=int))
    for arm, n0, n1, m0, m1, pv, ranks, top in summary:
        print(f"{arm}: non-OOD n={n0} median {m0:.2f} | OOD n={n1} median {m1:.2f} | p={pv:.4f}")
        print(f"   OOD ranks {ranks} | top non-OOD {top}")
    print("wrote", out)


if __name__ == "__main__":
    main()
