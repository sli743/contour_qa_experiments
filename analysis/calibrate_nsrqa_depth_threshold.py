"""Calibrate the depth threshold of the deep-change volume QA score, per arm.

qa_change_deep{t}_ml = volume of network change (input vs reconstruction) whose
voxels sit >= t mm from the other surface. The right t depends on how closely an
arm's reconstruction follows the input: a model that copies fine detail (e.g. the
CT + contour arms) leaves a thinner change shell, so a lower t may separate real
repairs from shell noise; a smoother model needs a higher t.

Two ways to choose t, reported side by side:
  * label-free: the floor of the arm on clean approved inputs. For each t, how
    much deep change does the arm produce when the input has no error? t_floor_pXX
    is the smallest t for which the XX-th percentile of clean deep volume is
    <= --floor-ml. Uses no autoseg labels.
  * supervised: Spearman / precision@k of the score against the true autoseg
    error (thick1 volume, top-k), plus ranks of the --watch cases. Use this to
    check the label-free choice, not to pick t (56 cases, easy to overfit).

Distances are computed once per case; thresholds are then a cheap sweep.

Example:
  python -m analysis.calibrate_nsrqa_depth_threshold \
      --eval-root <study>/evaluations \
      --arms ct_contour_corruption ct_contour_no_corruption contour_only_v3 \
      --watch 148,138,103,47,135 --output-dir <study>/evaluations_floor_corrected/seed91_depth_sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage, stats

STRUCT = ndimage.generate_binary_structure(3, 1)


def load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def scan_fov_mask(ct_path: Path, margin_mm: float = 4.5) -> np.ndarray | None:
    """Voxels inside the scanned field of view, shrunk by margin_mm along z.

    Crops that reach past the scan's superior/inferior end are padded with whole
    constant slices (-1024 HU, which reads as air). The CT-conditioned models can
    grow lung into that padding or carve the flat face where the scan cuts the lung,
    neither of which says anything about the contour. Change there is excluded from
    the QA score. Returns None if the CT is missing or has no padded slices.
    """
    ct_path = Path(ct_path)
    if margin_mm < 0 or not ct_path.exists():
        return None
    img = sitk.ReadImage(str(ct_path))
    ct = sitk.GetArrayFromImage(img)
    flat = ct.reshape(ct.shape[0], -1)
    padded = flat.max(axis=1) == flat.min(axis=1)
    if not padded.any():
        return None
    sz = img.GetSpacing()[2]
    grow = int(np.ceil(margin_mm / sz - 1e-6))
    if grow:
        padded = ndimage.binary_dilation(padded, iterations=grow)
    return np.broadcast_to(~padded[:, None, None], ct.shape).copy()


def change_depths(inp: np.ndarray, rec: np.ndarray, spacing_zyx, vml: float, min_component_ml: float,
                  valid: np.ndarray | None = None) -> np.ndarray:
    """Depth (mm) of every kept change voxel; same definition as score_nsrqa_floor_corrected.

    valid: optional mask (e.g. scan_fov_mask); change outside it is ignored.
    """
    parts = []
    for change, other in ((inp & ~rec, rec), (rec & ~inp, inp)):
        if valid is not None:
            change = change & valid
        if not change.any():
            continue
        dist = ndimage.distance_transform_edt(~other, sampling=spacing_zyx)
        lab, n = ndimage.label(change, STRUCT)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sizes * vml >= min_component_ml
        keep[0] = False
        parts.append(dist[keep[lab]].astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def thick1_ml(inp: np.ndarray, ref: np.ndarray, vml: float) -> float:
    te = ndimage.binary_erosion(inp & ~ref, STRUCT, iterations=1)
    tm = ndimage.binary_erosion(ref & ~inp, STRUCT, iterations=1)
    return float((te.sum() + tm.sum()) * vml)


def case_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {p.name.replace("patient_", ""): p for p in root.iterdir() if (p / "metrics.json").exists()}


def ranks(score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, kind="stable")
    r = np.empty(len(score), dtype=int)
    r[order] = np.arange(1, len(score) + 1)
    return r


def deep_ml(depths: np.ndarray, t: float, vml: float) -> float:
    return float((depths >= t - 1e-6).sum() * vml)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arm names under --eval-root, or label=<other evaluations root>/<arm> for arms from another study")
    ap.add_argument("--seed", default="91")
    ap.add_argument("--mode", default="autoseg_centered")
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--split", default="testing1")
    ap.add_argument("--thresholds", default="1.5,2.25,3,3.75,4.5,5.25,6,7.5,9,12")
    ap.add_argument("--min-component-ml", type=float, nargs="+", default=[0.05])
    ap.add_argument("--fov-margin-mm", type=float, default=4.5,
                    help="ignore change in CT slices outside the scan (constant padding) and within this many mm "
                         "of them; negative disables")
    ap.add_argument("--floor-ml", type=float, default=0.5,
                    help="label-free rule: clean deep volume allowed at the chosen percentile")
    ap.add_argument("--floor-percentiles", default="90,95")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--watch", default="")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    pcts = [float(p) for p in args.floor_percentiles.split(",")]
    watch = [w.strip() for w in args.watch.split(",") if w.strip()]
    eval_root, out_dir = Path(args.eval_root), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_dirs = {}
    for a in args.arms:
        label, _, path = a.partition("=")
        arm_dirs[label] = Path(path) if path else eval_root / label

    def arm_root(arm: str, kind: str) -> Path:
        return arm_dirs[arm] / f"seed{args.seed}" / kind / args.mode / args.roi / args.split

    sweep, per_case, choices = [], [], []
    truth_cache: dict[str, dict] = {}
    for arm in arm_dirs:
        cases = case_dirs(arm_root(arm, "actual_autoseg_cases"))
        clean = case_dirs(arm_root(arm, "clean_approved_cases"))
        if not cases:
            print(f"[{arm}] no cases under {arm_root(arm, 'actual_autoseg_cases')}; skipping")
            continue
        pids = sorted(cases, key=lambda s: (len(s), s))
        for mc in args.min_component_ml:
            act_d, cln_d, vmls = {}, {}, {}
            for pid in pids:
                d = cases[pid]
                img = sitk.ReadImage(str(d / "autoseg_input.nii.gz"))
                sp = tuple(reversed(img.GetSpacing()))
                vml = float(np.prod(img.GetSpacing())) / 1000.0
                vmls[pid] = vml
                inp = sitk.GetArrayFromImage(img) > 0
                rec = load(d / "nsrqa_reconstruction.nii.gz")
                act_d[pid] = change_depths(inp, rec, sp, vml, mc, scan_fov_mask(d / "ct.nii.gz", args.fov_margin_mm))
                if pid not in truth_cache:
                    app = load(d / "approved.nii.gz")
                    m = json.loads((d / "metrics.json").read_text())
                    truth_cache[pid] = {"true_thick1_ml": thick1_ml(inp, app, vml),
                                        "true_error_ml": float((inp ^ app).sum() * vml),
                                        "true_hd95_mm": m["true_hd95_mm"]}
                if pid in clean:
                    c = clean[pid]
                    cin = load(c / "approved.nii.gz")  # clean eval: approved contour is the input
                    crec = load(c / "nsrqa_reconstruction.nii.gz")
                    cln_d[pid] = change_depths(cin, crec, sp, vml, mc, scan_fov_mask(c / "ct.nii.gz", args.fov_margin_mm))
            print(f"[{arm}] min_component_ml={mc}: {len(act_d)} cases, {len(cln_d)} clean", flush=True)

            truth = {k: np.array([truth_cache[p][k] for p in pids]) for k in ("true_thick1_ml", "true_error_ml", "true_hd95_mm")}
            relevant = np.zeros(len(pids), bool)
            relevant[np.argsort(-truth["true_thick1_ml"], kind="stable")[: args.top_k]] = True
            for t in thresholds:
                s = np.array([deep_ml(act_d[p], t, vmls[p]) for p in pids])
                c = np.array([deep_ml(cln_d[p], t, vmls[p]) for p in pids if p in cln_d])
                rk = ranks(s)
                e = {"arm": arm, "min_component_ml": mc, "threshold_mm": t,
                     "frac_cases_nonzero": float((s > 0).mean()),
                     "median_deep_ml": float(np.median(s))}
                for k, v in truth.items():
                    e[f"rho_{k}"] = float(stats.spearmanr(s, v)[0]) if s.std() > 0 else np.nan
                e[f"p@{args.top_k}_top{args.top_k}_true_thick"] = float(relevant[np.argsort(-s, kind="stable")[: args.top_k]].mean())
                if c.size:
                    e["clean_median_ml"] = float(np.median(c))
                    for p in pcts:
                        e[f"clean_p{p:g}_ml"] = float(np.percentile(c, p))
                    sc = np.array([deep_ml(cln_d[p], t, vmls[p]) if p in cln_d else np.nan for p in pids])
                    ok = np.isfinite(sc)
                    e["rho_clean_floor"] = float(stats.spearmanr(s[ok], sc[ok])[0]) if s[ok].std() > 0 and sc[ok].std() > 0 else np.nan
                    # how far actual cases stand out from the clean distribution
                    e["actual_over_clean_p95"] = float((s > np.percentile(c, 95)).mean())
                for w in watch:
                    if w in pids:
                        e[f"rank_{w}"] = int(rk[pids.index(w)])
                if watch:
                    e["watch_mean_rank"] = float(np.mean([e[f"rank_{w}"] for w in watch if f"rank_{w}" in e]))
                sweep.append(e)
                for p, v in zip(pids, s):
                    per_case.append({"arm": arm, "min_component_ml": mc, "threshold_mm": t, "patient_id": p,
                                     f"qa_change_deep_ml": v, **truth_cache[p]})

            arm_rows = [e for e in sweep if e["arm"] == arm and e["min_component_ml"] == mc]
            ch = {"arm": arm, "min_component_ml": mc}
            if cln_d:
                for p in pcts:
                    ok = [e["threshold_mm"] for e in arm_rows if e[f"clean_p{p:g}_ml"] <= args.floor_ml]
                    ch[f"t_floor_p{p:g}"] = min(ok) if ok else np.nan
            best = max(arm_rows, key=lambda e: (np.nan_to_num(e["rho_true_thick1_ml"], nan=-9)))
            ch["t_best_rho_thick1"] = best["threshold_mm"]
            ch["best_rho_thick1"] = best["rho_true_thick1_ml"]
            if watch:
                bw = min(arm_rows, key=lambda e: e["watch_mean_rank"])
                ch["t_best_watch_mean_rank"] = bw["threshold_mm"]
                ch["best_watch_mean_rank"] = bw["watch_mean_rank"]
            choices.append(ch)

    pd.DataFrame(sweep).to_csv(out_dir / "depth_threshold_sweep.csv", index=False)
    pd.DataFrame(per_case).to_csv(out_dir / "depth_threshold_per_case.csv", index=False)
    pd.DataFrame(choices).to_csv(out_dir / "depth_threshold_choices.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(choices).round(3).to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
