"""Month3 truth census: which ROIs have enough real deep errors to evaluate NSR-QA on?

Before training the generic pipeline (C1I1 vs C0I0) on a new ROI, count its
Month3 (testing1) region-detection truth with exactly the region_detection_ct
definition and the ROI's anatomy-profile settings (make_roi_pipeline_configs):
positive patients, true regions, missing vs extra, and how much of the error
volume is out of reach of the fitted corruptions:
  - segment-scale: region max depth > --segment-mm (corruptions stay <= ~4 mm)
  - beyond_end:   an extra region lying past the approved S/I extent (z_rel < 0 or > 1)
  - end_missing:  a missing region touching the approved first/last slices (truncation)

Reads the all-ROI month-temporal study's autoseg-centred Month3 crops
(<temporal>/actual_autoseg_regression/autoseg_centered/<ROI>/testing1/patient_*),
and the training/validation counts from its preprocess manifests. CPU only.

  python -m analysis.roi_truth_census --output-dir /scratch/.../roi_truth_census_v1
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.calibrate_nsrqa_depth_threshold import load, scan_fov_mask
from analysis.make_roi_pipeline_configs import PROFILE_SETTINGS, REGISTRY
from analysis.region_detection_ct import depth_map, regions

DONE = {"Lung_L", "Lung_R", "Heart", "Esophagus"}


def census_case(job):
    roi, pid, case_dir, s, a = job
    import SimpleITK as sitk

    img = sitk.ReadImage(str(case_dir / "autoseg_input.nii.gz"))
    sp = tuple(reversed(img.GetSpacing()))
    vml = float(np.prod(img.GetSpacing())) / 1000.0
    inp = sitk.GetArrayFromImage(img) > 0
    app = load(case_dir / "approved.nii.gz")
    valid = scan_fov_mask(case_dir / "ct.nii.gz", a.fov_margin_mm)
    t = s["TRUTH_T"]
    depth, lab = depth_map(inp, app, sp, vml, s["MIN_COMPONENT_ML"], valid)
    zs = np.nonzero(app.any(axis=(1, 2)))[0]
    z0, z1 = (zs.min(), zs.max()) if zs.size else (0, inp.shape[0] - 1)
    rows = []
    for rid, core, piece in regions(depth, lab, t, vml, s["MIN_REGION_ML"]):
        kind = "missing" if (piece & app).any() else "extra"
        cz_all = np.nonzero(core.any(axis=(1, 2)))[0]
        z_rel = (float(cz_all.mean()) - z0) / max(z1 - z0, 1)
        pz = np.nonzero(piece.any(axis=(1, 2)))[0]
        rows.append({
            "roi": roi, "patient_id": pid, "region": rid, "kind": kind,
            "core_ml": float(core.sum() * vml), "piece_ml": float(piece.sum() * vml),
            "max_depth_mm": float(depth[core].max()), "z_rel": z_rel,
            "segment_scale": float(depth[core].max()) > a.segment_mm,
            "beyond_end": kind == "extra" and (z_rel < 0 or z_rel > 1),
            "end_missing": kind == "missing" and (pz.min() <= z0 + 1 or pz.max() >= z1 - 1),
        })
    return {"roi": roi, "patient_id": pid, "approved_ml": float(app.sum() * vml),
            "dice_err": float((inp ^ app).sum() / max(inp.sum() + app.sum(), 1))}, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temporal-study", default="/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study")
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--rois", nargs="*", default=None, help="default: every registry ROI with a supported profile")
    ap.add_argument("--segment-mm", type=float, default=10.0)
    ap.add_argument("--fov-margin-mm", type=float, default=4.5)
    ap.add_argument("--min-train", type=int, default=40, help="training1+validation1 patients needed")
    ap.add_argument("--min-positive", type=int, default=15, help="Month3 positive patients needed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reg = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))["rois"]
    rois = args.rois or [r for r, v in reg.items() if v["profile"] in PROFILE_SETTINGS]
    base = Path(args.temporal_study) / "actual_autoseg_regression"

    jobs, n_dev = [], {}
    for roi in rois:
        prof = reg[roi]["profile"]
        if prof not in PROFILE_SETTINGS:
            print(f"skip {roi}: profile {prof} unsupported", flush=True)
            continue
        man = base / "manifests" / f"{roi}_preprocess_manifest.csv"
        if man.is_file():
            m = pd.read_csv(man)
            m = m[(m.status == "ok") & (m.localization_mode == "autoseg_centered")]
            n_dev[roi] = {sp: int((m.split == sp).sum()) for sp in ("training1", "validation1", "testing1")}
        for d in sorted((base / "autoseg_centered" / roi / "testing1").glob("patient_*")):
            if (d / "approved.nii.gz").is_file() and (d / "autoseg_input.nii.gz").is_file():
                jobs.append((roi, d.name.split("_", 1)[1], d, PROFILE_SETTINGS[prof], args))
    print(f"{len(jobs)} Month3 cases over {len(rois)} ROIs", flush=True)

    cases, regs = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (c, r) in enumerate(ex.map(census_case, jobs, chunksize=2)):
            cases.append(c)
            regs.extend(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    cases, regs = pd.DataFrame(cases), pd.DataFrame(regs)
    regs.to_csv(out / "regions.csv", index=False)

    rows = []
    for roi, c in cases.groupby("roi"):
        prof = reg[roi]["profile"]
        s = PROFILE_SETTINGS[prof]
        r = regs[regs.roi == roi] if len(regs) else regs
        per = r.groupby("patient_id").core_ml.sum() if len(r) else pd.Series(dtype=float)
        c = c.assign(true_deep_ml=c.patient_id.map(per).fillna(0.0))
        pos = c.true_deep_ml >= s["POSITIVE_ML"]
        vol = r.core_ml.sum() if len(r) else 0.0
        frac = (lambda col: float(r.loc[r[col], "core_ml"].sum() / vol) if vol > 0 else np.nan)
        dev = n_dev.get(roi, {})
        n_train = dev.get("training1", 0) + dev.get("validation1", 0)
        n_pos = int(pos.sum())
        rows.append({
            "roi": roi, "profile": prof, "done": roi in DONE,
            "n_train_val": n_train, "n_month3": len(c), "n_positive": n_pos,
            "positive_rate": n_pos / max(len(c), 1),
            "n_regions": len(r), "n_missing": int((r.kind == "missing").sum()) if len(r) else 0,
            "n_extra": int((r.kind == "extra").sum()) if len(r) else 0,
            "median_true_deep_ml_pos": float(c.true_deep_ml[pos].median()) if n_pos else 0.0,
            "vol_frac_segment_scale": frac("segment_scale") if len(r) else np.nan,
            "vol_frac_beyond_end": frac("beyond_end") if len(r) else np.nan,
            "vol_frac_end_missing": frac("end_missing") if len(r) else np.nan,
            "n_pos_small_only": int(sum(1 for p in c.patient_id[pos]
                                        if not r[(r.patient_id == p) & (r.segment_scale | r.beyond_end | r.end_missing)].shape[0])),
            "eligible": n_train >= args.min_train and args.min_positive <= n_pos <= len(c) - 5,
        })
    summ = pd.DataFrame(rows).sort_values(["done", "eligible", "n_positive"], ascending=[True, False, False])
    summ.to_csv(out / "census.csv", index=False)
    cases.to_csv(out / "patients.csv", index=False)

    lines = ["# Month3 truth census by ROI", "",
             f"Truth = region_detection_ct definition with each profile's settings; segment-scale = max depth > {args.segment_mm} mm.",
             f"Eligible = train+val >= {args.min_train} and {args.min_positive} <= positives <= n-5.", "",
             summ.to_markdown(index=False, floatfmt=".2f")]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
