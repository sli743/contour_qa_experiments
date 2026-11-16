"""Visual + per-case evidence for the corruption x CT ablation.

Tests, case by case, the claims drawn from the 2x2 summary:
  * corruption without CT changes the contour, but not where the real error is;
  * CT without corruption copies the input (few false alarms, but no repair);
  * CT + corruption moves the contour toward approved at the real error.

Per case and cell it measures, from the saved masks:
  deep_ml               network change (input vs reconstruction) >= --depth mm from the
                        other surface, components >= --min-component-ml (same as the score)
  deep_on_error_ml      part of that deep change that lies inside the true autoseg error
                        (i.e. moves the contour toward approved)
  deep_precision        deep_on_error_ml / deep_ml
  error1_ml             substantive true error (1-voxel eroded excess + missing; = true_thick1)
  error1_changed_frac   share of error1 voxels the network changed (any depth)
  error1_deep_frac      share of error1 voxels covered by deep change

Patients are chosen by fixed rules from <ablation>/per_case.csv (not by eye):
  A_top_error        top 10 by true error volume (all shown, whatever the outcome)
  B_c1i1_gain        among top-20 true error, largest rank gain of C1I1 over the other cells
  C_contour_alarm    low-error cases (bottom half) ranked highest by contour-only arms
  D_clean_alarm      largest deep change on the approved contour (clean eval), contour-only
  E_c1i1_miss        top-10 true error that C1I1 ranks below --miss-rank
  E_c1i1_false_pos   bottom-half true error that C1I1 ranks in its top --fp-rank

Each patient gets one page: rows = axial and coronal slice through the focus region,
columns = the four cells. CT (lung window) with approved (blue), autoseg input (orange),
reconstruction (green); deep change shaded violet when it lies on the true error and red
when it does not. Group D shows the clean eval (input = approved).

  python -m analysis.inspect_ablation_cases --ablation-dir <...>/seed91_ablation \
     --cell C0I0=... --cell C1I0=... --cell C0I1=... --cell C1I1=...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

CELLS = ("C0I0", "C1I0", "C0I1", "C1I1")
TITLES = {
    "C0I0": "contour only · no corruption",
    "C1I0": "contour only · corruption",
    "C0I1": "CT + contour · no corruption",
    "C1I1": "CT + contour · corruption",
}
STRUCT = ndimage.generate_binary_structure(3, 1)
APPROVED, AUTOSEG, RECON = "#2a78d6", "#eb6834", "#1baf7a"
ON_ERR, OFF_ERR = "#6250d6", "#e34948"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(p: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0


def deep_change_mask(inp, rec, spacing, vml, depth, min_ml) -> np.ndarray:
    out = np.zeros_like(inp)
    for change, other in ((inp & ~rec, rec), (rec & ~inp, inp)):
        if not change.any():
            continue
        lab, n = ndimage.label(change, STRUCT)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sizes * vml >= min_ml
        keep[0] = False
        kept = keep[lab]
        dist = ndimage.distance_transform_edt(~other, sampling=spacing)
        out |= kept & (dist >= depth - 1e-6)
    return out


def error1(inp, app) -> np.ndarray:
    return ndimage.binary_erosion(inp & ~app, STRUCT) | ndimage.binary_erosion(app & ~inp, STRUCT)


def case_dir(arm_dir: Path, kind: str, pid: str, args) -> Path:
    return arm_dir / f"seed{args.seed}" / kind / args.mode / args.roi / args.split / f"patient_{pid}"


def select_groups(pc: pd.DataFrame, args) -> dict[str, list[str]]:
    w = pc[pc.arm.isin(CELLS)].pivot(index="patient_id", columns="arm", values=f"deep{args.depth:g}_ml")[list(CELLS)]
    r = w.rank(ascending=False, method="first")
    cl = pc[pc.arm.isin(CELLS)].pivot(index="patient_id", columns="arm", values=f"clean_deep{args.depth:g}_ml")
    t = pc[pc.arm == "C1I1"].set_index("patient_id")["true_thick1_ml"]
    tr = t.rank(ascending=False, method="first")
    n = len(t)
    top10 = list(tr.sort_values().index[:10])
    top20 = tr[tr <= 20].index
    gain = (r.loc[top20, ["C0I0", "C1I0", "C0I1"]].mean(axis=1) - r.loc[top20, "C1I1"]).sort_values(ascending=False)
    bottom = tr[tr > n / 2].index
    contour_alarm = r.loc[bottom, ["C0I0", "C1I0"]].min(axis=1).sort_values()
    clean_alarm = cl[["C0I0", "C1I0"]].max(axis=1).sort_values(ascending=False)
    return {
        "A_top_error": top10,
        "B_c1i1_gain": list(gain.index[:5]),
        "C_contour_alarm": list(contour_alarm[contour_alarm <= 15].index[:4]),
        "D_clean_alarm": list(clean_alarm.index[:4]),
        "E_c1i1_miss": [p for p in top10 if r.loc[p, "C1I1"] > args.miss_rank],
        "E_c1i1_false_pos": list(r.loc[bottom, "C1I1"][r.loc[bottom, "C1I1"] <= args.fp_rank].sort_values().index),
    }, r, tr


def case_data(pid: str, arm_dirs: dict, args, clean: bool = False) -> dict:
    kind = "clean_approved_cases" if clean else "actual_autoseg_cases"
    d1 = case_dir(arm_dirs["C1I1"], "actual_autoseg_cases", pid, args)
    img = sitk.ReadImage(str(d1 / "autoseg_input.nii.gz"))
    sp = tuple(reversed(img.GetSpacing()))
    vml = float(np.prod(img.GetSpacing())) / 1000.0
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(d1 / "ct.nii.gz"))).astype(np.float32)
    app = load(d1 / "approved.nii.gz")
    inp = app.copy() if clean else sitk.GetArrayFromImage(img) > 0
    err = inp ^ app
    e1 = error1(inp, app)
    out = {"ct": ct, "app": app, "inp": inp, "err": err, "e1": e1, "vml": vml, "cells": {}}
    for c in CELLS:
        rec = load(case_dir(arm_dirs[c], kind, pid, args) / "nsrqa_reconstruction.nii.gz")
        deep = deep_change_mask(inp, rec, sp, vml, args.depth, args.min_component_ml)
        chg = inp ^ rec
        out["cells"][c] = {"rec": rec, "deep": deep, "on": deep & err, "off": deep & ~err,
                           "deep_ml": deep.sum() * vml, "deep_on_error_ml": (deep & err).sum() * vml,
                           "deep_precision": (deep & err).sum() / deep.sum() if deep.any() else np.nan,
                           "error1_ml": e1.sum() * vml,
                           "error1_changed_frac": (chg & e1).sum() / e1.sum() if e1.any() else np.nan,
                           "error1_deep_frac": (deep & e1).sum() / e1.sum() if e1.any() else np.nan}
    return out


def focus_point(data: dict, group: str) -> tuple[int, int, int]:
    if group.startswith(("A_", "B_", "E_c1i1_miss")) and data["e1"].any():
        m = data["e1"]
    elif group.startswith("C_"):
        m = data["cells"]["C0I0"]["deep"] | data["cells"]["C1I0"]["deep"]
    elif group.startswith("E_c1i1_false"):
        m = data["cells"]["C1I1"]["deep"]
    else:
        m = np.zeros_like(data["app"])
        for c in CELLS:
            m |= data["cells"][c]["deep"]
    if not m.any():
        m = data["err"] if data["err"].any() else data["app"]
    lab, n = ndimage.label(m, STRUCT)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    comp = lab == big
    z = int(np.argmax(comp.sum(axis=(1, 2))))
    y = int(np.argmax(comp.sum(axis=(0, 2))))
    zz, yy, xx = np.nonzero(comp)
    return z, y, int(np.median(xx))


def draw(ax, ct2, app2, inp2, rec2, on2, off2, extent, origin, show_input=True):
    ax.imshow(ct2, cmap="gray", vmin=-1350, vmax=150, extent=extent, origin=origin, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~on2, on2), cmap=ListedColormap([ON_ERR]), alpha=0.65, extent=extent,
              origin=origin, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~off2, off2), cmap=ListedColormap([OFF_ERR]), alpha=0.65, extent=extent,
              origin=origin, interpolation="nearest")
    kw = dict(extent=extent, origin=origin, levels=[0.5], linewidths=1.3)
    if app2.any():
        ax.contour(app2.astype(float), colors=APPROVED, **kw)
    if show_input and inp2.any():
        ax.contour(inp2.astype(float), colors=AUTOSEG, linestyles="--", **kw)
    if rec2.any():
        ax.contour(rec2.astype(float), colors=RECON, **kw)
    ax.set_xticks([])
    ax.set_yticks([])


def page(data: dict, pid: str, group: str, ranks: pd.DataFrame, tr: pd.Series, pc: pd.DataFrame, args):
    z, y, x = focus_point(data, group)
    nz, ny, nx = data["app"].shape
    half = int(args.crop_mm / 1.5 / 2)
    y0, y1 = max(0, y - half), min(ny, y + half)
    x0, x1 = max(0, x - half), min(nx, x + half)
    z0, z1 = max(0, z - half), min(nz, z + half)
    clean = group.startswith("D_")
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.4), facecolor=SURFACE)
    for j, c in enumerate(CELLS):
        cd = data["cells"][c]
        for i, view in enumerate(("axial", "coronal")):
            ax = axes[i, j]
            if view == "axial":
                sl = (z, slice(y0, y1), slice(x0, x1))
                ext, org = (x0 * 1.5, x1 * 1.5, y1 * 1.5, y0 * 1.5), "upper"
            else:
                sl = (slice(z0, z1), y, slice(x0, x1))
                ext, org = (x0 * 1.5, x1 * 1.5, z0 * 1.5, z1 * 1.5), "lower"
            draw(ax, data["ct"][sl], data["app"][sl], data["inp"][sl], cd["rec"][sl], cd["on"][sl], cd["off"][sl],
                 ext, org, show_input=not clean)
            if i == 0:
                ax.set_title(TITLES[c], color=INK, fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f"{view} (slice {z if view == 'axial' else y})", color=INK2, fontsize=9)
        rk = int(ranks.loc[pid, c]) if not clean else None
        prec = "–" if np.isnan(cd["deep_precision"]) else f"{cd['deep_precision']:.0%}"
        if clean:
            txt = f"deep change on approved {cd['deep_ml']:.2f} mL"
        else:
            e1c = "–" if np.isnan(cd["error1_changed_frac"]) else f"{cd['error1_changed_frac']:.0%}"
            txt = (f"score rank {rk}/56 · deep {cd['deep_ml']:.2f} mL\n"
                   f"on true error {prec} · true error changed {e1c}")
        axes[1, j].set_xlabel(txt, color=INK, fontsize=9)
    e1 = data["e1"].sum() * data["vml"]
    head = (f"patient {pid} · group {group} · " +
            ("clean eval: input = approved contour" if clean else
             f"true error {e1:.1f} mL (rank {int(tr.loc[pid])}/56)"))
    fig.suptitle(head, color=INK, fontsize=12, x=0.01, ha="left", y=0.995)
    handles = [Line2D([], [], color=APPROVED, lw=1.6, label="approved")]
    if not clean:
        handles.append(Line2D([], [], color=AUTOSEG, lw=1.6, ls="--", label="autoseg input"))
    handles += [Line2D([], [], color=RECON, lw=1.6, label="reconstruction"),
                Patch(color=ON_ERR, alpha=0.65, label=f"deep change (≥{args.depth:g} mm) on true error"),
                Patch(color=OFF_ERR, alpha=0.65, label=f"deep change (≥{args.depth:g} mm) not on true error")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", required=True)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--seed", default="91")
    ap.add_argument("--mode", default="autoseg_centered")
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--split", default="testing1")
    ap.add_argument("--depth", type=float, default=4.5)
    ap.add_argument("--min-component-ml", type=float, default=0.05)
    ap.add_argument("--miss-rank", type=int, default=15)
    ap.add_argument("--fp-rank", type=int, default=10)
    ap.add_argument("--crop-mm", type=float, default=150)
    ap.add_argument("--skip-all-cases", action="store_true", help="only compute metrics for selected patients")
    args = ap.parse_args()

    abl = Path(args.ablation_dir)
    out = abl / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    arm_dirs = {}
    for spec in args.cell:
        name, _, path = spec.partition("=")
        arm_dirs[name] = Path(path)
    pc = pd.read_csv(abl / "per_case.csv", dtype={"patient_id": str})
    groups, ranks, tr = select_groups(pc, args)
    sel = pd.DataFrame([(g, p) for g, ps in groups.items() for p in ps], columns=["group", "patient_id"])
    sel.to_csv(out / "selected_patients.csv", index=False)
    for g, ps in groups.items():
        print(f"{g}: {', '.join(ps) if ps else '(none)'}")

    # per-case overlap metrics (all patients, actual autoseg)
    pids = sorted(tr.index, key=lambda s: (len(s), s)) if not args.skip_all_cases else sorted(set(sel.patient_id))
    rows = []
    for pid in pids:
        data = case_data(pid, arm_dirs, args)
        for c in CELLS:
            m = {k: v for k, v in data["cells"][c].items() if not isinstance(v, np.ndarray)}
            rows.append({"patient_id": pid, "arm": c, "score_rank": int(ranks.loc[pid, c]),
                         "true_rank": int(tr.loc[pid]), **m})
        print(f"[metrics] {pid}", flush=True)
    met = pd.DataFrame(rows)
    met.to_csv(out / "deep_change_overlap.csv", index=False)

    # summaries: pooled over all patients, and over each group
    def pooled(df):
        return pd.Series({
            "n": df.patient_id.nunique(),
            "deep_ml_total": df.deep_ml.sum(),
            "pooled_deep_precision": df.deep_on_error_ml.sum() / df.deep_ml.sum() if df.deep_ml.sum() else np.nan,
            "median_deep_precision": df.deep_precision.median(),
            "pooled_error1_changed": (df.error1_changed_frac * df.error1_ml).sum() / df.error1_ml.sum()
            if df.error1_ml.sum() else np.nan,
            "pooled_error1_deep": (df.error1_deep_frac * df.error1_ml).sum() / df.error1_ml.sum()
            if df.error1_ml.sum() else np.nan,
        })
    summ = [pooled(met[met.arm == c]).rename(c).to_frame().T.assign(scope="all") for c in CELLS]
    for g, ps in groups.items():
        if g.startswith("D_") or not ps:
            continue
        for c in CELLS:
            summ.append(pooled(met[(met.arm == c) & met.patient_id.isin(ps)]).rename(c).to_frame().T.assign(scope=g))
    summ = pd.concat(summ).rename_axis("arm").reset_index()
    summ.to_csv(out / "deep_change_overlap_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3).to_string(index=False))

    # pages
    for g, ps in groups.items():
        if not ps:
            continue
        with PdfPages(out / f"{g}.pdf") as pdf:
            for pid in ps:
                data = case_data(pid, arm_dirs, args, clean=g.startswith("D_"))
                fig = page(data, pid, g, ranks, tr, pc, args)
                fig.savefig(out / f"{g}_patient_{pid}.png", dpi=110, facecolor=SURFACE, bbox_inches="tight")
                pdf.savefig(fig, facecolor=SURFACE, bbox_inches="tight")
                plt.close(fig)
                print(f"[page] {g} {pid}", flush=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
