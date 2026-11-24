"""Is the superior end of an ROI drawn at a consistent anatomical level?

Motivation (Heart, 2026-11-24): most real Heart errors are one 60-100 mL missing
slab at the superior end. The approved top either equals the autoseg top or lies
10-25 mm above it, with almost nothing in between. A model can learn to flag that
only if the choice to extend follows anatomy, not the physician. This script
measures both tops against landmarks contoured on the same patient:

  * kept vs extended: approved top more than --extended-mm above the autoseg top.
  * offsets (mm, + = superior) of the approved and autoseg tops from each
    landmark, landmark taken from the autoseg (same algorithm every case) and
    from the approved contour.
  * Learnable-convention signal: the approved top sits at a consistent level
    (small spread of approved_top - landmark), and the extended cases are the
    ones whose autoseg stops low (AUC of autoseg_top - landmark, kept vs
    extended, far from 0.5).
    Physician-choice signal: the autoseg offsets of the two groups overlap
    (AUC near 0.5) and the approved offsets are bimodal.
  * Where the slab goes: the fraction of the superior missing slab that lies
    inside the approved great-vessel contours (does the approved Heart include
    the vessel roots?).

Labels are read from the raw data (labels-approved / labels-autoseg); nothing is
resampled except the overlap masks onto the approved grid. Month3 cases are
reported separately (diagnostic only; nothing here trains a model).

  python -m analysis.superior_end_convention --roi Heart \
      --output-dir /scratch/shaohan/contour_QA/heart_superior_convention_v1
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats

from analysis.characterize_lung_error_fields import build_temporal_split

# name -> (landmark ROI, feature of its S/I extent)
LANDMARKS = {
    "carina_centroid": ("Carina", "centroid"),
    "trachea_bottom": ("Trachea", "bottom"),
    "pulmonary_top": ("A_Pulmonary", "top"),
    "pulmonary_centroid": ("A_Pulmonary", "centroid"),
    "aorta_top": ("A_Aorta", "top"),
    "svc_bottom": ("V_VenaCava_S", "bottom"),
}
VESSELS = ("A_Aorta", "A_Pulmonary", "V_VenaCava_S")


def label_path(root: Path, pid: int, roi: str, kind: str) -> Path:
    folder = "labels-approved" if kind == "approved" else "labels-autoseg"
    return root / folder / f"patient{pid}_{roi}_{kind}.nii.gz"


def read_mask(path: Path):
    img = sitk.ReadImage(str(path))
    return img, sitk.GetArrayFromImage(img) > 0


def slice_z_mm(img: sitk.Image, nz: int) -> np.ndarray:
    """World S/I coordinate (LPS z, superior positive) of each array-z slice."""
    sx, sy, _ = img.GetSize()
    return np.array([img.TransformContinuousIndexToPhysicalPoint(((sx - 1) / 2, (sy - 1) / 2, float(k)))[2]
                     for k in range(nz)])


def extent(img: sitk.Image, arr: np.ndarray, min_area_mm2: float) -> Optional[Dict[str, float]]:
    sp = img.GetSpacing()
    area = arr.sum(axis=(1, 2)) * sp[0] * sp[1]
    ok = area > min_area_mm2 if min_area_mm2 > 0 else area > 0
    if not ok.any():
        return None
    z = slice_z_mm(img, arr.shape[0])
    return {"top": float(z[ok].max()), "bottom": float(z[ok].min()),
            "centroid": float((z * area).sum() / area.sum())}


def on_grid(img: sitk.Image, arr: np.ndarray, ref: sitk.Image) -> np.ndarray:
    same = (img.GetSize() == ref.GetSize() and np.allclose(img.GetSpacing(), ref.GetSpacing(), atol=1e-5)
            and np.allclose(img.GetOrigin(), ref.GetOrigin(), atol=1e-4)
            and np.allclose(img.GetDirection(), ref.GetDirection(), atol=1e-5))
    if same:
        return arr
    m = sitk.GetImageFromArray(arr.astype(np.uint8))
    m.CopyInformation(img)
    r = sitk.Resample(m, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(r) > 0


def one_case(args) -> Dict:
    pid, split, cohort, root, roi, min_area, slab_margin_mm = args
    row: Dict = {"patient_id": pid, "split": split, "cohort": cohort}
    try:
        masks = {}
        for kind in ("approved", "autoseg"):
            p = label_path(root, pid, roi, kind)
            if not p.is_file():
                row["status"] = f"missing {kind}"
                return row
            img, arr = read_mask(p)
            e = extent(img, arr, min_area)
            if e is None:
                row["status"] = f"empty {kind}"
                return row
            masks[kind] = (img, arr)
            row[f"{kind}_top"], row[f"{kind}_bottom"] = e["top"], e["bottom"]
        row["top_diff_mm"] = row["approved_top"] - row["autoseg_top"]
        row["bottom_diff_mm"] = row["approved_bottom"] - row["autoseg_bottom"]

        for name, (lroi, feat) in LANDMARKS.items():
            for kind in ("autoseg", "approved"):
                p = label_path(root, pid, lroi, kind)
                if p.is_file():
                    e = extent(*read_mask(p), 0.0)
                    if e is not None:
                        row[f"lm_{kind}_{name}"] = e[feat]

        app_img, app = masks["approved"]
        auto = on_grid(*masks["autoseg"], app_img)
        sp = app_img.GetSpacing()
        vox_ml = sp[0] * sp[1] * sp[2] / 1000.0
        z = slice_z_mm(app_img, app.shape[0])
        upper = (z > row["autoseg_top"] - slab_margin_mm)[:, None, None]
        slab = app & ~auto & upper
        row["superior_slab_ml"] = float(slab.sum() * vox_ml)
        row["approved_ml"] = float(app.sum() * vox_ml)
        row["autoseg_ml"] = float(auto.sum() * vox_ml)
        for v in VESSELS:
            p = label_path(root, pid, v, "approved")
            if not p.is_file():
                continue
            vm = on_grid(*read_mask(p), app_img)
            row[f"slab_in_{v}_frac"] = float((slab & vm).sum() / slab.sum()) if slab.any() else np.nan
            row[f"approved_overlap_{v}_ml"] = float((app & vm).sum() * vox_ml)
            row[f"autoseg_overlap_{v}_ml"] = float((auto & vm).sum() * vox_ml)
        row["status"] = "ok"
    except Exception as exc:  # keep going; report per case
        row["status"] = f"error: {exc}"
    return row


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def q(x: pd.Series) -> str:
    x = x.dropna()
    if x.empty:
        return "-"
    return f"{x.median():+.1f} [{x.quantile(0.25):+.1f}, {x.quantile(0.75):+.1f}]"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def landmark_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in LANDMARKS:
        for src in ("autoseg", "approved"):
            col = f"lm_{src}_{name}"
            if col not in d or d[col].notna().sum() < 5:
                continue
            app_off = d["approved_top"] - d[col]
            auto_off = d["autoseg_top"] - d[col]
            ext, kept = d["extended"], ~d["extended"]
            rows.append({
                "landmark": f"{name} ({src})", "n": int(d[col].notna().sum()),
                "approved top - lm, kept": q(app_off[kept]), "approved top - lm, extended": q(app_off[ext]),
                "autoseg top - lm, kept": q(auto_off[kept]), "autoseg top - lm, extended": q(auto_off[ext]),
                "SD approved off": f"{app_off.std():.1f}", "SD autoseg off": f"{auto_off.std():.1f}",
                "AUC autoseg off (kept > extended)": f"{auc(auto_off[kept].to_numpy(), auto_off[ext].to_numpy()):.2f}",
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi", default="Heart")
    ap.add_argument("--raw-data-root", default="/home/shaohan/radformation-data/data")
    ap.add_argument("--split-yaml", default="configs/dino_shape/month_temporal_nsrqa_split.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--extended-mm", type=float, default=7.5,
                    help="approved top this far above the autoseg top = extended")
    ap.add_argument("--min-area-mm2", type=float, default=100.0,
                    help="a slice counts toward the ROI extent if its area exceeds this")
    ap.add_argument("--slab-margin-mm", type=float, default=5.0,
                    help="superior slab = approved minus autoseg above (autoseg top - margin)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.raw_data_root)
    split, cohorts = build_temporal_split(Path(args.split_yaml))
    tasks = [(pid, s, cohorts.get(pid, ""), root, args.roi, args.min_area_mm2, args.slab_margin_mm)
             for s in ("training1", "validation1", "testing1") for pid in split[s]]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        rows: List[Dict] = list(ex.map(one_case, tasks, chunksize=2))
    df = pd.DataFrame(rows).sort_values(["split", "patient_id"])
    df.to_csv(out / "superior_end_per_case.csv", index=False)

    ok = df[df["status"] == "ok"].copy()
    ok["extended"] = ok["top_diff_mm"] >= args.extended_mm
    ok["dev"] = ok["split"] != "testing1"
    lines = [f"# {args.roi}: is the superior end drawn at a consistent anatomical level?", "",
             f"Cases: {len(df)} (ok {len(ok)}; " + ", ".join(f"{k}: {v}" for k, v in
                                                          df[df['status'] != 'ok']['status'].value_counts().items()) + ")",
             f"Extended = approved top >= {args.extended_mm} mm above the autoseg top (ROI slices > {args.min_area_mm2} mm2).",
             "Offsets: mm, + = superior; median [IQR]. AUC: how well a LOW autoseg top (vs the landmark) "
             "picks out the extended cases (0.5 = not at all, 1 = perfectly).", ""]

    for label, d in (("Months 1+2 (training1 + validation1)", ok[ok["dev"]]), ("Month3 (testing1, diagnostic)", ok[~ok["dev"]])):
        if d.empty:
            continue
        diffs = np.sort(d["top_diff_mm"].round(1).to_numpy())
        lines += [f"## {label}", "",
                  f"n = {len(d)}; extended {int(d['extended'].sum())}, kept {int((~d['extended']).sum())}; "
                  f"approved top - autoseg top (mm), sorted: {' '.join(f'{x:g}' for x in diffs)}", "",
                  f"Bottom end for contrast, approved - autoseg bottom: {q(d['bottom_diff_mm'])}", ""]
        t = landmark_table(d)
        if not t.empty:
            lines += [md_table(t), ""]
        vcols = [c for c in d.columns if c.startswith("slab_in_")]
        if vcols:
            vt = []
            for c in vcols:
                v = c[len("slab_in_"):-len("_frac")]
                vt.append({"vessel (approved)": v,
                           "share of superior slab inside, extended": q(d.loc[d["extended"], c] * 100) + " %",
                           "approved Heart overlap mL, kept": q(d.loc[~d["extended"], f"approved_overlap_{v}_ml"]),
                           "approved Heart overlap mL, extended": q(d.loc[d["extended"], f"approved_overlap_{v}_ml"]),
                           "autoseg Heart overlap mL, all": q(d[f"autoseg_overlap_{v}_ml"])})
            lines += ["Superior slab vs the approved great-vessel contours:", "", md_table(pd.DataFrame(vt)), ""]
        lines += [f"Superior slab volume, extended: {q(d.loc[d['extended'], 'superior_slab_ml'])} mL; "
                  f"kept: {q(d.loc[~d['extended'], 'superior_slab_ml'])} mL", ""]

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [n for n in LANDMARKS if f"lm_autoseg_{n}" in ok and ok[f"lm_autoseg_{n}"].notna().sum() >= 5]
        if names:
            fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
            for ax, n in zip(axes[0], names):
                for dev, mk in ((True, "o"), (False, "^")):
                    for ext, col in ((False, "tab:blue"), (True, "tab:red")):
                        s = ok[(ok["dev"] == dev) & (ok["extended"] == ext)]
                        ax.scatter(s["autoseg_top"] - s[f"lm_autoseg_{n}"], s["approved_top"] - s[f"lm_autoseg_{n}"],
                                   c=col, marker=mk, s=18, alpha=0.8,
                                   label=f"{'M1+2' if dev else 'M3'} {'extended' if ext else 'kept'}")
                lo = np.nanmin(ok["autoseg_top"] - ok[f"lm_autoseg_{n}"]) - 5
                hi = np.nanmax(ok["approved_top"] - ok[f"lm_autoseg_{n}"]) + 5
                ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
                ax.set_xlabel(f"autoseg top - {n} (mm)")
                ax.set_ylabel(f"approved top - {n} (mm)")
            axes[0][0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / "superior_end_vs_landmarks.png", dpi=130)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
