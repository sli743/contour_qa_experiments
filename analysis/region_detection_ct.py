"""Depth-matched truth and region-level detection for the corruption x CT ablation.

The patient-level ablation (analysis.ablation_corruption_ct) scores a model by its
deep-change volume (change >= t mm from the other surface) but measures truth as
the 1-voxel-eroded error volume, and calls the k largest errors positive. This
script measures truth with the same depth rule and also scores where the model
edits, not just how much.

Per case and arm (input = Month3 autoseg, reference = approved, rec = model output):

  true error regions   connected pieces of autoseg-vs-approved error (missing and
                       extra lung labelled separately, inside the scan FOV), kept if
                       their part at depth >= t (the "deep core") is >= --min-region-ml.
                       Depth = distance to the other surface, exactly as the QA score.
  detected             the model's deep change covers >= --detect-frac of the core.
  model regions        connected pieces of model change whose deep part is
                       >= --min-region-ml; correct if >= --correct-frac of the piece
                       lies in true error (= moves toward approved).
  clean regions        model regions on the approved input (all false alarms).

Metrics per arm (ratios of sums over patients, so region counts pool across cases):
  region_recall           detected true regions / true regions
  deep_error_recall       deep-core volume covered by deep model change / deep-core volume
  region_precision        correct model regions / model regions
  deep_change_precision   deep model change inside true error / deep model change
  fp_regions_per_case     wrong model regions per patient (autoseg input)
  clean_regions_per_case  model regions per patient on approved input
Patient level, truth = true deep-error volume (true_deep_ml):
  rho_truth_deep          Spearman(deep-change score, true_deep_ml)
  positives               true_deep_ml >= --positive-ml (a cutoff, not a fixed top k)
  recall@k, located_recall@k   positives in the top k (by deep-change score) / positives;
                          "located" also needs >= 1 detected region in that patient
  capture@k               share of total true deep-error volume in the top k
  ap                      average precision over the positives
k = --budgets (a count, or a fraction of patients such as 0.2).

Seeds are pooled as in analysis.ablation_corruption_ct_seeds: metric per seed, then
mean over seeds, with a paired patient bootstrap (same patients for every seed and
arm). Effects: the 2x2 contrasts plus direct C1I1-C0I0 and C1I1-ct_only.

  python -m analysis.region_detection_ct --eval-root <ct study>/evaluations \
    --seeds 91 92 93 --roi Lung_R --exclude 26,100 --workers 8 \
    --output-dir <ct study>/evaluations_floor_corrected/region_detection
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from analysis.ablation_corruption_ct import CELLS, ap_at, effects
from analysis.calibrate_nsrqa_depth_threshold import STRUCT, case_dirs, load, scan_fov_mask

ARMS = {  # label -> arm directory under --eval-root (postprocess job layout)
    "C0I0": "contour_only_no_corruption",
    "C1I0": "contour_only_v3",
    "C0I1": "ct_contour_no_corruption",
    "C1I1": "ct_contour_corruption",
    "ct_only": "ct_only_segmenter",
    # early-fusion variant (image_skip_inputs: ct_mask), trained in its own study;
    # point at it with --arm-root C1I1m=<maskstem study>/evaluations/ct_mask_contour_corruption
    "C1I1m": "ct_mask_contour_corruption",
    # region-growth (leak) corruption, also in its own study (--arm-root)
    "C1I1g": "ct_contour_corruption_growth",
    "C1I0g": "contour_only_growth",
}
DEFAULT_ARMS = ["C0I0", "C1I0", "C0I1", "C1I1", "ct_only"]
DIRECT = [("C1I1", "C0I0"), ("C1I1", "ct_only"), ("C1I1m", "C1I1"), ("C1I1m", "C0I0"),
          ("C1I1g", "C1I1"), ("C1I0g", "C1I0"), ("C1I1g", "C0I0")]


def arm_root(arm: str, args) -> Path:
    """Arm directory (holding seed<S>/): --arm-root override, else <eval-root>/<ARMS[arm]>."""
    for spec in args.arm_root:
        lab, _, path = spec.partition("=")
        if lab == arm:
            return Path(path)
    return Path(args.eval_root) / ARMS[arm]


def depth_map(inp: np.ndarray, other: np.ndarray, spacing, vml: float, min_component_ml: float,
              valid: np.ndarray | None):
    """Per-voxel depth of inp-vs-other change (0 elsewhere), with direction labels.

    Same definition as calibrate_nsrqa_depth_threshold.change_depths: a removed voxel
    (inp & ~other) is measured to `other`, an added voxel (other & ~inp) to `inp`;
    components smaller than min_component_ml and voxels outside `valid` are dropped.
    Returns (depth, labels) where labels > 0 number the kept components (removed and
    added pieces get separate numbers).
    """
    depth = np.zeros(inp.shape, np.float32)
    labels = np.zeros(inp.shape, np.int32)
    offset = 0
    for change, ref in ((inp & ~other, other), (other & ~inp, inp)):
        if valid is not None:
            change = change & valid
        if not change.any():
            continue
        lab, n = ndimage.label(change, STRUCT)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sizes * vml >= min_component_ml
        keep[0] = False
        kept = keep[lab]
        if not kept.any():
            continue
        dist = ndimage.distance_transform_edt(~ref, sampling=spacing)
        depth[kept] = dist[kept]
        labels[kept] = lab[kept] + offset
        offset += n
    return depth, labels


def regions(depth: np.ndarray, labels: np.ndarray, t: float, vml: float, min_region_ml: float):
    """Components whose deep part (depth >= t) is >= min_region_ml: list of (id, deep core mask, piece mask)."""
    deep = depth >= t - 1e-6
    if not deep.any():
        return []
    deep_counts = np.bincount(labels[deep], minlength=labels.max() + 1)
    ids = [i for i in np.nonzero(deep_counts * vml >= min_region_ml)[0] if i > 0]
    out = []
    for i in ids:
        piece = labels == i
        out.append((int(i), piece & deep, piece))
    return out


def score_case(job):
    arm, pid, case_dir, clean_dir, a = job
    import SimpleITK as sitk

    img = sitk.ReadImage(str(case_dir / "autoseg_input.nii.gz"))
    sp = tuple(reversed(img.GetSpacing()))
    vml = float(np.prod(img.GetSpacing())) / 1000.0
    inp = sitk.GetArrayFromImage(img) > 0
    rec = load(case_dir / "nsrqa_reconstruction.nii.gz")
    app = load(case_dir / "approved.nii.gz")
    valid = scan_fov_mask(case_dir / "ct.nii.gz", a["fov_margin_mm"])
    t, mr, mc = a["t"], a["min_region_ml"], a["min_component_ml"]

    err = inp ^ app
    if valid is not None:
        err &= valid
    t_depth, t_lab = depth_map(inp, app, sp, vml, mc, valid)
    m_depth, m_lab = depth_map(inp, rec, sp, vml, mc, valid)
    t_deep = t_depth >= t - 1e-6
    m_deep = m_depth >= t - 1e-6
    chg = inp ^ rec

    zs = np.nonzero(app.any(axis=(1, 2)))[0]
    z0, z1 = (zs.min(), zs.max()) if zs.size else (0, inp.shape[0] - 1)
    reg_rows = []
    n_det = 0
    for rid, core, piece in regions(t_depth, t_lab, t, vml, mr):
        core_n = int(core.sum())
        cov = float((core & m_deep).sum() / core_n)
        any_cov = float((core & chg).sum() / core_n)
        det = cov >= a["detect_frac"]
        n_det += det
        cz = float(np.nonzero(core.any(axis=(1, 2)))[0].mean())
        reg_rows.append({
            "arm": arm, "patient_id": pid, "region": rid,
            "kind": "missing" if (piece & app).any() else "extra",
            "piece_ml": float(piece.sum() * vml), "core_ml": core_n * vml,
            "max_depth_mm": float(t_depth[core].max()),
            "z_rel": (cz - z0) / max(z1 - z0, 1),  # 0/1 = first/last slice of the approved lung (array z order)
            "deep_coverage": cov, "any_coverage": any_cov, "detected": bool(det),
        })

    model_regs = regions(m_depth, m_lab, t, vml, mr)
    n_correct = sum((piece & err).sum() / piece.sum() >= a["correct_frac"] for _, _, piece in model_regs)

    row = {
        "arm": arm, "patient_id": pid,
        "true_deep_ml": float(t_deep.sum() * vml),
        "n_true_regions": len(reg_rows), "n_detected": int(n_det),
        **{f"n_{w}_{kd}": sum(r["kind"] == kd and (w == "true" or r["detected"]) for r in reg_rows)
           for kd in ("missing", "extra") for w in ("true", "det")},
        "true_core_ml": float(sum(r["core_ml"] for r in reg_rows)),
        "core_covered_ml": float(sum(r["core_ml"] * r["deep_coverage"] for r in reg_rows)),
        "deep_change_ml": float(m_deep.sum() * vml),
        "deep_change_in_error_ml": float((m_deep & err).sum() * vml),
        "n_model_regions": len(model_regs), "n_model_correct": int(n_correct),
        "clean_n_regions": np.nan, "clean_deep_change_ml": np.nan,
    }
    if clean_dir is not None:
        c_app = load(clean_dir / "approved.nii.gz")
        c_rec = load(clean_dir / "nsrqa_reconstruction.nii.gz")
        c_valid = scan_fov_mask(clean_dir / "ct.nii.gz", a["fov_margin_mm"])
        c_depth, c_lab = depth_map(c_app, c_rec, sp, vml, mc, c_valid)
        row["clean_n_regions"] = len(regions(c_depth, c_lab, t, vml, mr))
        row["clean_deep_change_ml"] = float((c_depth >= t - 1e-6).sum() * vml)
    return row, reg_rows


def collect(seed: str, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(args.output_dir) / f"seed{seed}"
    pc_path, rg_path = out / "per_case.csv", out / "regions.csv"
    if args.reuse_cache and pc_path.exists() and rg_path.exists():
        return pd.read_csv(pc_path, dtype={"patient_id": str}), pd.read_csv(rg_path, dtype={"patient_id": str})
    out.mkdir(parents=True, exist_ok=True)
    tail = Path(args.mode) / args.roi / args.split
    a = {k: getattr(args, k) for k in ("t", "min_region_ml", "min_component_ml", "detect_frac",
                                        "correct_frac", "fov_margin_mm")}
    jobs = []
    for arm in args.arms:
        root = arm_root(arm, args) / f"seed{seed}"
        cases = case_dirs(root / "actual_autoseg_cases" / tail)
        clean = case_dirs(root / "clean_approved_cases" / tail)
        if not cases:
            raise SystemExit(f"[seed{seed} {arm}] no cases under {root / 'actual_autoseg_cases' / tail}")
        jobs += [(arm, pid, d, clean.get(pid), a) for pid, d in cases.items()]
    rows, regs = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (row, rr) in enumerate(ex.map(score_case, jobs, chunksize=2)):
            rows.append(row)
            regs += rr
            if (i + 1) % 50 == 0:
                print(f"seed{seed}: {i + 1}/{len(jobs)} cases", flush=True)
    pc, rg = pd.DataFrame(rows), pd.DataFrame(regs)
    pc.to_csv(pc_path, index=False)
    rg.to_csv(rg_path, index=False)
    return pc, rg


def budgets(spec: str, n: int) -> dict[str, int]:
    out = {}
    for b in spec.split(","):
        v = float(b)
        out[f"{b}" if v >= 1 else f"{v:.0%}"] = int(round(v * n)) if v < 1 else int(v)
    return out


def metrics(df: pd.DataFrame, pos: np.ndarray, truth: np.ndarray, ks: dict[str, int]) -> dict:
    """df rows aligned with pos / truth (one row per sampled patient)."""
    s = df.sum(numeric_only=True)
    div = lambda a, b: float(a / b) if b > 0 else np.nan  # noqa: E731
    n = len(df)
    score = df["deep_change_ml"].to_numpy()
    out = {
        "region_recall": div(s.n_detected, s.n_true_regions),
        "recall_missing": div(s.get("n_det_missing", np.nan), s.get("n_true_missing", 0)),
        "recall_extra": div(s.get("n_det_extra", np.nan), s.get("n_true_extra", 0)),
        "deep_error_recall": div(s.core_covered_ml, s.true_core_ml),
        "region_precision": div(s.n_model_correct, s.n_model_regions),
        "deep_change_precision": div(s.deep_change_in_error_ml, s.deep_change_ml),
        "fp_regions_per_case": (s.n_model_regions - s.n_model_correct) / n,
        "clean_regions_per_case": float(df["clean_n_regions"].mean()) if df["clean_n_regions"].notna().all() else np.nan,
        "rho_truth_deep": float(stats.spearmanr(score, truth)[0]) if np.std(score) > 0 and np.std(truth) > 0 else np.nan,
        "ap": ap_at(score, pos),
    }
    order = np.argsort(-score, kind="stable")
    located = df["n_detected"].to_numpy() > 0
    for name, k in ks.items():
        top = order[:k]
        out[f"recall@{name}"] = div(pos[top].sum(), pos.sum())
        out[f"located_recall@{name}"] = div((pos & located)[top].sum(), pos.sum())
        out[f"capture@{name}"] = div(truth[top].sum(), truth.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True, help="<ct study>/evaluations (postprocess layout)")
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument("--arm-root", action="append", default=[],
                    help="LABEL=<arm dir> (containing seed<S>/) for an arm outside --eval-root")
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--mode", default="autoseg_centered")
    ap.add_argument("--split", default="testing1")
    ap.add_argument("--t", type=float, default=4.5, help="depth threshold (mm), same as the QA score")
    ap.add_argument("--min-component-ml", type=float, default=0.05)
    ap.add_argument("--min-region-ml", type=float, default=0.5, help="minimum deep-core volume of a region")
    ap.add_argument("--detect-frac", type=float, default=0.1)
    ap.add_argument("--correct-frac", type=float, default=0.5)
    ap.add_argument("--fov-margin-mm", type=float, default=4.5)
    ap.add_argument("--positive-ml", type=float, default=2.0, help="patient positive if true_deep_ml >= this")
    ap.add_argument("--budgets", default="10,0.2", help="review budgets: counts or fractions of patients")
    ap.add_argument("--exclude", default="", help="comma list of patient ids to leave out (e.g. not image-assessable)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse-cache", action="store_true", help="reuse seed<S>/per_case.csv + regions.csv")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {s: collect(s, args) for s in args.seeds}
    excl = {e.strip() for e in args.exclude.split(",") if e.strip()}
    pids = sorted(set.intersection(*[set(pc[pc.arm == a].patient_id) for pc, _ in data.values() for a in args.arms])
                  - excl, key=lambda s: (len(s), s))
    n = len(pids)
    by = {s: {a: pc[pc.arm == a].set_index("patient_id").loc[pids].reset_index() for a in args.arms}
          for s, (pc, _) in data.items()}
    ref_arm = "C1I1" if "C1I1" in args.arms else args.arms[0]
    truth = next(iter(by.values()))[ref_arm]["true_deep_ml"].to_numpy()
    spread = max(float(np.max(np.abs(d[a]["true_deep_ml"].to_numpy() - truth))) for d in by.values() for a in d)
    pos = truth >= args.positive_ml
    ks = budgets(args.budgets, n)
    print(f"n={n} patients (excluded: {sorted(excl) or 'none'}), positives (true deep >= {args.positive_ml} mL): "
          f"{int(pos.sum())}; max truth difference between arm crops: {spread:.2f} mL")
    tt = pd.DataFrame({"patient_id": pids, "true_deep_ml": truth,
                       "n_true_regions": next(iter(by.values()))[ref_arm]["n_true_regions"].to_numpy()})
    print(tt.sort_values("true_deep_ml", ascending=False).head(15).round(2).to_string(index=False))

    def per_seed(idx=None):
        p = pos if idx is None else pos[idx]
        y = truth if idx is None else truth[idx]
        return {s: {a: metrics(df if idx is None else df.iloc[idx], p, y, ks) for a, df in d.items()}
                for s, d in by.items()}

    def mean_seeds(m):
        first = next(iter(m.values()))
        return {a: {k: float(np.nanmean([m[s][a][k] for s in m])) for k in first[a]} for a in first}

    def contrasts(mm, k):
        e = effects(mm, k) if set(CELLS) <= set(mm) else {}
        for x, y in DIRECT:
            if x in mm and y in mm:
                e[f"{x}-{y}"] = mm[x][k] - mm[y][k]
        return e

    point = per_seed()
    mean_pt = mean_seeds(point)
    keys = list(next(iter(mean_pt.values())))
    rng = np.random.default_rng(0)
    boot_arm = {a: {k: [] for k in keys} for a in args.arms}
    boot_eff = {k: [] for k in keys}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        mm = mean_seeds(per_seed(idx))
        for a in args.arms:
            for k in keys:
                boot_arm[a][k].append(mm[a][k])
        for k in keys:
            boot_eff[k].append(contrasts(mm, k))

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return (np.percentile(v, 2.5), np.percentile(v, 97.5), float((v > 0).mean())) if v.size else (np.nan,) * 3

    rows = []
    for a in args.arms:
        r = {"arm": a, "n": n, "n_positive": int(pos.sum()), "n_seeds": len(by)}
        for k in keys:
            lo, hi, _ = ci(boot_arm[a][k])
            r[k], r[f"{k}_lo"], r[f"{k}_hi"] = mean_pt[a][k], lo, hi
            r[f"{k}_sd"] = float(np.nanstd([point[s][a][k] for s in point], ddof=1)) if len(point) > 1 else np.nan
        rows.append(r)
    summary = pd.DataFrame(rows)
    eff_rows = []
    for k in keys:
        est = contrasts(mean_pt, k)
        seed_vals = {s: contrasts(point[s], k) for s in point}
        for name, v in est.items():
            lo, hi, pp = ci([b[name] for b in boot_eff[k]])
            sv = [seed_vals[s][name] for s in seed_vals]
            eff_rows.append({"metric": k, "effect": name, "estimate": v, "ci_lo": lo, "ci_hi": hi, "p_gt_0": pp,
                             "seed_sd": float(np.std(sv, ddof=1)) if len(sv) > 1 else np.nan,
                             "seeds_gt_0": f"{sum(x > 0 for x in sv)}/{len(sv)}"})
    eff = pd.DataFrame(eff_rows)
    per_seed_df = pd.DataFrame([{"seed": s, "arm": a, **point[s][a]} for s in point for a in args.arms])
    regions_all = pd.concat([rg.assign(seed=s) for s, (_, rg) in data.items() if len(rg)], ignore_index=True) \
        if any(len(rg) for _, rg in data.values()) else pd.DataFrame(columns=["seed", "arm", "patient_id"])
    regions_all = regions_all[regions_all.patient_id.isin(pids)]

    summary.to_csv(out / "region_summary.csv", index=False)
    eff.to_csv(out / "region_effects.csv", index=False)
    per_seed_df.to_csv(out / "region_per_seed.csv", index=False)
    tt.assign(positive=pos).to_csv(out / "patient_truth.csv", index=False)
    regions_all.to_csv(out / "regions_all_seeds.csv", index=False)
    (out / "region_metadata.json").write_text(json.dumps({
        "eval_root": args.eval_root, "seeds": args.seeds, "roi": args.roi, "arms": {a: str(arm_root(a, args)) for a in args.arms},
        "n_patients": n, "excluded": sorted(excl), "n_positive": int(pos.sum()), "positive_ml": args.positive_ml,
        "depth_threshold_mm": args.t, "min_region_ml": args.min_region_ml, "detect_frac": args.detect_frac,
        "correct_frac": args.correct_frac, "budgets": ks, "bootstrap": args.bootstrap,
        "ci": "paired patient bootstrap of the seed-mean (case sampling); _sd = training spread over seeds",
    }, indent=2))

    show = ["region_recall", "recall_missing", "recall_extra", "deep_error_recall", "region_precision", "deep_change_precision",
            "fp_regions_per_case", "clean_regions_per_case", "rho_truth_deep", "ap"] + \
           [f"{m}@{b}" for b in ks for m in ("recall", "located_recall", "capture")]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print()
        print(summary[["arm"] + show].round(3).to_string(index=False))
        print()
        print(eff[eff.effect.isin(["interaction", "corruption_without_ct", "corruption_with_ct",
                                   "ct_with_corruption", "C1I1-C0I0", "C1I1-ct_only", "C1I1m-C1I1", "C1I1m-C0I0",
                                   "C1I1g-C1I1", "C1I0g-C1I0", "C1I1g-C0I0"])]
              .sort_values(["metric", "effect"]).round(3).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
