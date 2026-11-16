"""2x2 ablation of NSR-QA training corruption (C) x CT input (I) on Month3 autoseg.

Cells (same data / seed / schedule, differ only in the factor):
  C0I0 contour only, identity training      C0I1 CT + contour, identity training
  C1I0 contour only, corruption training    C1I1 CT + contour, corruption training
Extra arms (CT removed at test time, CT-only segmenter) are scored the same way and
reported as references.

Per case and arm the script computes, from the saved masks:
  * QA scores: QA HD95 (metrics.json); deep-change volume at a common depth threshold
    and at the arm's label-free calibrated threshold (smallest grid t whose clean-
    approved 90th-percentile deep volume <= --floor-ml).
  * truth: autoseg-vs-approved error volume (thick1 = 1-voxel eroded, as in
    score_nsrqa_floor_corrected) and raw voxel error.
  * network behaviour: change precision (share of input->reconstruction change that
    moves toward approved), error-correction recall, voxel-error reduction.
  * floor: deep-change volume on the clean approved input (false alarm).

Ranking metrics per arm: Spearman vs true error volume, precision@k and AP against
the k worst cases, watch-case ranks. Effects (corruption, CT, interaction and the
simple effects) get a paired bootstrap over patients - one seed, so the CI covers
case sampling only, not training variance.

Example:
  python -m analysis.ablation_corruption_ct \
    --cell C0I0=<nocorr study>/evaluations/no_corruption \
    --cell C1I0=<v3 study>/evaluations/nsrqa_corruption \
    --cell C0I1=<ct study>/evaluations/ct_contour_no_corruption \
    --cell C1I1=<ct study>/evaluations/ct_contour_corruption \
    --extra C1I1_ct_absent=<ct study>/evaluations/ct_contour_corruption_ct_absent@C1I1 \
    --extra C0I1_ct_absent=<ct study>/evaluations/ct_contour_no_corruption_ct_absent@C0I1 \
    --extra ct_only=<ct study>/evaluations/ct_only_segmenter \
    --watch 148,138,103,47,135 --output-dir <ct study>/evaluations_floor_corrected/seed91_ablation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage, stats

from analysis.calibrate_nsrqa_depth_threshold import change_depths, scan_fov_mask, thick1_ml

CELLS = ("C0I0", "C1I0", "C0I1", "C1I1")
LABELS = {
    "C0I0": "contour only, no corruption",
    "C1I0": "contour only, corruption",
    "C0I1": "CT + contour, no corruption",
    "C1I1": "CT + contour, corruption",
}
SCORES = ("qa_hd95_mm", "deep_common_ml", "deep_cal_ml")
BEHAVIOUR = ("change_precision", "error_recall", "error_reduction")


def load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def case_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {p.name.replace("patient_", ""): p for p in root.iterdir() if (p / "metrics.json").exists()}


def per_case_rows(arm: str, arm_dir: Path, args, grid) -> list[dict]:
    sub = Path(f"seed{args.seed}")
    tail = Path(args.mode) / args.roi / args.split
    cases = case_dirs(arm_dir / sub / "actual_autoseg_cases" / tail)
    clean = case_dirs(arm_dir / sub / "clean_approved_cases" / tail)
    if not cases:
        raise SystemExit(f"[{arm}] no cases under {arm_dir / sub / 'actual_autoseg_cases' / tail}")
    rows = []
    for pid in sorted(cases, key=lambda s: (len(s), s)):
        d = cases[pid]
        img = sitk.ReadImage(str(d / "autoseg_input.nii.gz"))
        sp = tuple(reversed(img.GetSpacing()))
        vml = float(np.prod(img.GetSpacing())) / 1000.0
        inp = sitk.GetArrayFromImage(img) > 0
        rec = load(d / "nsrqa_reconstruction.nii.gz")
        app = load(d / "approved.nii.gz")
        m = json.loads((d / "metrics.json").read_text())
        err = inp ^ app
        chg = inp ^ rec
        good = chg & err
        n_err, n_chg = int(err.sum()), int(chg.sum())
        row = {
            "arm": arm, "patient_id": pid,
            "true_thick1_ml": thick1_ml(inp, app, vml),
            "true_error_ml": n_err * vml,
            "true_hd95_mm": m["true_hd95_mm"],
            "qa_hd95_mm": m["qa_hd95_mm"],
            "change_ml": n_chg * vml,
            "harmful_change_ml": int((chg & ~err).sum()) * vml,
            "change_precision": good.sum() / n_chg if n_chg else np.nan,
            "error_recall": good.sum() / n_err if n_err else np.nan,
            "error_reduction": (n_err - int((rec ^ app).sum())) / n_err if n_err else np.nan,
        }
        dep = change_depths(inp, rec, sp, vml, args.min_component_ml, scan_fov_mask(d / "ct.nii.gz", args.fov_margin_mm))
        for t in grid:
            row[f"deep{t:g}_ml"] = float((dep >= t - 1e-6).sum() * vml)
        if pid in clean:
            c = clean[pid]
            cdep = change_depths(load(c / "approved.nii.gz"), load(c / "nsrqa_reconstruction.nii.gz"), sp, vml,
                                 args.min_component_ml, scan_fov_mask(c / "ct.nii.gz", args.fov_margin_mm))
            for t in grid:
                row[f"clean_deep{t:g}_ml"] = float((cdep >= t - 1e-6).sum() * vml)
        rows.append(row)
        print(f"[{arm}] {pid}", flush=True)
    return rows


def calibrated_threshold(df: pd.DataFrame, grid, floor_ml: float, pct: float) -> float:
    for t in grid:
        col = f"clean_deep{t:g}_ml"
        if col in df and df[col].notna().all() and np.percentile(df[col], pct) <= floor_ml:
            return t
    return np.nan


def ap_at(score: np.ndarray, rel: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    r = rel[order]
    if not r.any():
        return np.nan
    hits = np.cumsum(r)
    return float((hits[r] / (np.nonzero(r)[0] + 1)).mean())


def rank_average(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combined QA score: mean of each arm's within-sample rank, per score.

    The frames must be row-aligned (same patients, same order, same resample).
    Ranks are recomputed on whatever rows are passed, so a bootstrap resample is
    re-ranked rather than reusing full-sample ranks. Network-behaviour columns are
    not defined for a combination and are left out.
    """
    base = dfs[0].reset_index(drop=True)
    out = base[["patient_id", "true_thick1_ml"]].copy()
    for s in SCORES:
        # higher score -> higher rank; a missing score ranks as the least suspicious
        out[s] = np.mean([pd.Series(d[s].to_numpy()).rank(method="average", na_option="top").to_numpy()
                          for d in dfs], axis=0)
    return out


def parse_combos(specs: list[str], known) -> dict[str, list[str]]:
    combos = {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        parts = [p for p in rest.split("+") if p]
        if not name or len(parts) < 2:
            raise SystemExit(f"--combine needs NAME=ARM+ARM, got {spec}")
        bad = [p for p in parts if p not in known]
        if bad:
            raise SystemExit(f"--combine {name}: unknown arm(s) {bad}")
        combos[name] = parts
    return combos


def arm_metrics(df: pd.DataFrame, rel: np.ndarray, k: int) -> dict:
    """Metrics for one arm on (possibly resampled) rows; rel = fixed per-patient relevance."""
    y = df["true_thick1_ml"].to_numpy()
    out = {}
    for s in SCORES:
        v = df[s].to_numpy()
        out[f"rho_{s}"] = float(stats.spearmanr(v, y)[0]) if np.std(v) > 0 else np.nan
        out[f"ap_{s}"] = ap_at(v, rel)
        out[f"p@{k}_{s}"] = float(rel[np.argsort(-v, kind="stable")[:k]].mean())
    for b in BEHAVIOUR:
        if b in df and df[b].notna().any():
            out[f"mean_{b}"] = float(np.nanmean(df[b]))
    if "change_ml" in df:
        out["median_change_ml"] = float(df["change_ml"].median())
        out["median_harmful_change_ml"] = float(df["harmful_change_ml"].median())
    if "clean_deep_common_ml" in df and df["clean_deep_common_ml"].notna().all():
        out["clean_p90_deep_common_ml"] = float(np.percentile(df["clean_deep_common_ml"], 90))
        out["clean_mean_deep_common_ml"] = float(df["clean_deep_common_ml"].mean())
    return out


def effects(m: dict[str, dict], key: str) -> dict:
    g = {c: m[c][key] for c in CELLS}
    return {
        "corruption_main": ((g["C1I0"] - g["C0I0"]) + (g["C1I1"] - g["C0I1"])) / 2,
        "ct_main": ((g["C0I1"] - g["C0I0"]) + (g["C1I1"] - g["C1I0"])) / 2,
        "interaction": (g["C1I1"] - g["C0I1"]) - (g["C1I0"] - g["C0I0"]),
        "corruption_without_ct": g["C1I0"] - g["C0I0"],
        "corruption_with_ct": g["C1I1"] - g["C0I1"],
        "ct_without_corruption": g["C0I1"] - g["C0I0"],
        "ct_with_corruption": g["C1I1"] - g["C1I0"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True, help="CELL=<evaluations>/<arm>, CELL in C0I0 C1I0 C0I1 C1I1")
    ap.add_argument("--extra", action="append", default=[],
                    help="label=<evaluations>/<arm>[@CELL] - reference arm; @CELL borrows that cell's calibrated "
                         "threshold when the arm has no clean-approved cases")
    ap.add_argument("--combine", action="append", default=[],
                    help="NAME=ARM+ARM[+ARM] - rank-average combined score of these arms (cells or extras), "
                         "re-ranked inside every bootstrap resample; compared against each of its parts")
    ap.add_argument("--seed", default="91")
    ap.add_argument("--mode", default="autoseg_centered")
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--split", default="testing1")
    ap.add_argument("--grid", default="3,3.75,4.5,5.25,6,7.5,9,12")
    ap.add_argument("--common-threshold", type=float, default=4.5)
    ap.add_argument("--floor-ml", type=float, default=0.5)
    ap.add_argument("--floor-percentile", type=float, default=90)
    ap.add_argument("--min-component-ml", type=float, default=0.05)
    ap.add_argument("--fov-margin-mm", type=float, default=4.5,
                    help="ignore change in CT slices outside the scan (constant padding) and within this many mm "
                         "of them; negative disables")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--watch", default="")
    ap.add_argument("--reuse-cache", action="store_true", help="reuse per_case.csv in --output-dir")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    grid = [float(t) for t in args.grid.split(",")]
    if args.common_threshold not in grid:
        raise SystemExit("--common-threshold must be one of --grid")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    arms: dict[str, tuple[Path, str | None]] = {}
    for spec in args.cell:
        name, _, path = spec.partition("=")
        if name not in CELLS:
            raise SystemExit(f"unknown cell {name}")
        arms[name] = (Path(path), None)
    missing = set(CELLS) - set(arms)
    if missing:
        raise SystemExit(f"missing cells: {sorted(missing)}")
    for spec in args.extra:
        name, _, rest = spec.partition("=")
        path, _, parent = rest.partition("@")
        arms[name] = (Path(path), parent or None)
    combos = parse_combos(args.combine, arms)

    cache = out / "per_case.csv"
    if args.reuse_cache and cache.exists():
        pc = pd.read_csv(cache, dtype={"patient_id": str})
    else:
        rows = []
        for name, (path, _) in arms.items():
            rows.extend(per_case_rows(name, path, args, grid))
        pc = pd.DataFrame(rows)
        pc.to_csv(cache, index=False)

    # thresholds
    thr = {}
    for name, (_, parent) in arms.items():
        t = calibrated_threshold(pc[pc.arm == name], grid, args.floor_ml, args.floor_percentile)
        if np.isnan(t) and parent:
            t = thr.get(parent, np.nan)
        thr[name] = t
    for name in arms:
        sel = pc.arm == name
        t = thr[name] if np.isfinite(thr[name]) else args.common_threshold
        pc.loc[sel, "deep_cal_ml"] = pc.loc[sel, f"deep{t:g}_ml"]
        pc.loc[sel, "deep_common_ml"] = pc.loc[sel, f"deep{args.common_threshold:g}_ml"]
        if f"clean_deep{args.common_threshold:g}_ml" in pc:
            pc.loc[sel, "clean_deep_common_ml"] = pc.loc[sel, f"clean_deep{args.common_threshold:g}_ml"]

    pids = sorted(set.intersection(*[set(pc[pc.arm == a].patient_id) for a in arms]), key=lambda s: (len(s), s))
    by_arm = {a: pc[pc.arm == a].set_index("patient_id").loc[pids].reset_index() for a in arms}
    truth = by_arm["C1I1"]["true_thick1_ml"].to_numpy()
    rel = np.zeros(len(pids), bool)
    rel[np.argsort(-truth, kind="stable")[: args.top_k]] = True
    watch = [w.strip() for w in args.watch.split(",") if w.strip() and w.strip() in pids]

    # point estimates per arm (combinations are re-ranked from their parts)
    view = dict(by_arm)
    for name, parts in combos.items():
        view[name] = rank_average([by_arm[p] for p in parts])
    point = {a: arm_metrics(df, rel, args.top_k) for a, df in view.items()}
    summary = []
    for a, df in view.items():
        label = LABELS.get(a, a) if a not in combos else "rank average of " + " + ".join(combos[a])
        e = {"arm": a, "label": label, "n": len(df),
             "calibrated_threshold_mm": thr.get(a, np.nan), "common_threshold_mm": args.common_threshold, **point[a]}
        for s in SCORES:
            v = df[s].to_numpy()
            rk = stats.rankdata(-v, method="ordinal")
            for w in watch:
                e[f"rank_{w}_{s}"] = int(rk[pids.index(w)])
        summary.append(e)
    summary = pd.DataFrame(summary)

    # paired bootstrap
    rng = np.random.default_rng(0)
    keys = [k for k in point["C1I1"] if all(k in point[c] for c in CELLS)]
    n = len(pids)
    boot_arm = {a: {k: [] for k in point[a]} for a in view}
    boot_eff = {k: [] for k in keys}
    extra_cmp = [(x, p) for x, (_, p) in arms.items() if p in CELLS]
    boot_extra = {(x, k): [] for x, _ in extra_cmp for k in keys if k in point[x]}
    # combination vs each part, on the ranking metrics both have
    combo_cmp = {(c, p, k): [] for c, parts in combos.items() for p in parts for k in point[c] if k in point[p]}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        m = {a: arm_metrics(df.iloc[idx], rel[idx], args.top_k) for a, df in by_arm.items()}
        for name, parts in combos.items():
            m[name] = arm_metrics(rank_average([by_arm[p].iloc[idx] for p in parts]), rel[idx], args.top_k)
        for a in view:
            for k in boot_arm[a]:
                boot_arm[a][k].append(m[a].get(k, np.nan))
        for k in keys:
            boot_eff[k].append(effects(m, k))
        for (x, k) in boot_extra:
            p = arms[x][1]
            boot_extra[(x, k)].append(m[p][k] - m[x][k])
        for (c, p, k) in combo_cmp:
            combo_cmp[(c, p, k)].append(m[c][k] - m[p][k])

    def ci(vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        return (np.percentile(v, 2.5), np.percentile(v, 97.5), float((v > 0).mean())) if v.size else (np.nan,) * 3

    for a in view:
        for k in point[a]:
            lo, hi, _ = ci(boot_arm[a][k])
            summary.loc[summary.arm == a, f"{k}_lo"] = lo
            summary.loc[summary.arm == a, f"{k}_hi"] = hi
    eff_rows = []
    for k in keys:
        pe = effects(point, k)
        for name, val in pe.items():
            lo, hi, pp = ci([b[name] for b in boot_eff[k]])
            eff_rows.append({"metric": k, "effect": name, "estimate": val, "ci_lo": lo, "ci_hi": hi, "p_gt_0": pp})
    for (x, k), vals in boot_extra.items():
        p = arms[x][1]
        lo, hi, pp = ci(vals)
        eff_rows.append({"metric": k, "effect": f"ct_at_test:{p}-{x}", "estimate": point[p][k] - point[x][k],
                         "ci_lo": lo, "ci_hi": hi, "p_gt_0": pp})
    for (c, p, k), vals in combo_cmp.items():
        lo, hi, pp = ci(vals)
        eff_rows.append({"metric": k, "effect": f"combo:{c}-{p}", "estimate": point[c][k] - point[p][k],
                         "ci_lo": lo, "ci_hi": hi, "p_gt_0": pp})
    eff = pd.DataFrame(eff_rows)

    summary.to_csv(out / "ablation_arm_summary.csv", index=False)
    eff.to_csv(out / "ablation_effects.csv", index=False)
    pc.to_csv(cache, index=False)
    (out / "ablation_metadata.json").write_text(json.dumps({
        "arms": {a: {"path": str(p), "threshold_from": par, "calibrated_threshold_mm": thr[a]} for a, (p, par) in arms.items()},
        "combine": combos,
        "n_patients": n, "top_k": args.top_k, "bootstrap": args.bootstrap, "seed": args.seed,
        "floor_rule": f"clean p{args.floor_percentile:g} deep volume <= {args.floor_ml} mL",
        "common_threshold_mm": args.common_threshold, "relevance": "top-k true_thick1_ml",
        "fov_margin_mm": args.fov_margin_mm,
    }, indent=2))

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["arm", "calibrated_threshold_mm"] + [f"rho_{s}" for s in SCORES] + [f"p@{args.top_k}_{s}" for s in SCORES] + \
               [f"mean_{b}" for b in BEHAVIOUR] + ["clean_p90_deep_common_ml"]
        print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
        main_keys = [f"rho_{s}" for s in SCORES] + [f"mean_{b}" for b in BEHAVIOUR] + ["clean_p90_deep_common_ml"]
        print(eff[eff.metric.isin(main_keys)].round(3).to_string(index=False))
        if combos:
            cmp_keys = [f"{m}_deep_common_ml" for m in ("rho", "ap", f"p@{args.top_k}")]
            print(eff[eff.effect.str.startswith("combo:") & eff.metric.isin(cmp_keys)].round(3).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
