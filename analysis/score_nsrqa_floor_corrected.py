"""Floor-corrected NSR-QA scores computed from saved actual-autoseg case outputs.

The standard QA score (1-Dice between the autoseg and its reconstruction) mostly
tracks how well the model can redraw a patient's *normal* organ. This script
computes three alternative scores from the saved case masks, without retraining:

  1. thick_*      disagreement that survives an erosion (concentrated errors only;
                  the model's thin, spread-out reconstruction shell is removed)
  2. xarm_*       disagreement between this arm's reconstruction and a reference
                  arm's reconstruction (the no-corruption model as a difficulty
                  reference; shared decoder limits cancel)
  3. resid_qa     QA 1-Dice minus a leave-one-out prediction of the clean-approved
                  reconstruction error from autoseg-only shape features

and compares each score with the true autoseg error (vs approved).
Only autoseg-derived quantities are used in the scores; the approved contour is
used for the targets (true error) and to fit the LOO difficulty model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage, stats

STRUCT = ndimage.generate_binary_structure(3, 3)  # 26-connected, 1 voxel


def load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def spacing_ml(path: Path) -> float:
    sp = sitk.ReadImage(str(path)).GetSpacing()
    return float(np.prod(sp)) / 1000.0


def one_minus_dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(1.0 - 2.0 * np.logical_and(a, b).sum() / s) if s else 0.0


def thick(mask: np.ndarray, iters: int) -> np.ndarray:
    return ndimage.binary_erosion(mask, STRUCT, iterations=iters) if iters else mask


def largest_cc(mask: np.ndarray) -> int:
    lab, n = ndimage.label(mask, STRUCT)
    return int(np.bincount(lab.ravel())[1:].max()) if n else 0


def disagreement_scores(inp: np.ndarray, ref: np.ndarray, vml: float, prefix: str) -> dict:
    """Excess = in inp not in ref; missing = in ref not in inp."""
    out = {}
    excess = inp & ~ref
    missing = ref & ~inp
    for k in (1, 2):
        te, tm = thick(excess, k), thick(missing, k)
        out[f"{prefix}_thick{k}_ml"] = (te.sum() + tm.sum()) * vml
        out[f"{prefix}_thick{k}_excess_ml"] = te.sum() * vml
        out[f"{prefix}_thick{k}_missing_ml"] = tm.sum() * vml
        out[f"{prefix}_thick{k}_maxcc_ml"] = max(largest_cc(te), largest_cc(tm)) * vml
    return out


def change_depth_scores(inp: np.ndarray, rec: np.ndarray, spacing_zyx: tuple, vml: float,
                        min_component_ml: float, prefix: str = "qa") -> dict:
    """HD-style scores on the network's change, robust to the thin reconstruction shell.

    Each removed voxel (inp & ~rec) gets its distance to the reconstruction and each
    added voxel (rec & ~inp) its distance to the input, i.e. how far the surface moved
    there. Components smaller than min_component_ml are dropped (stray voxels that
    make max-HD noisy). A local repair then shows up as a large depth even when it
    covers too little of the surface to move HD95.
    """
    out = {}
    depth_parts, comp_max = [], []
    for name, change, other in (("removed", inp & ~rec, rec), ("added", rec & ~inp, inp)):
        dist = ndimage.distance_transform_edt(~other, sampling=spacing_zyx)
        lab, n = ndimage.label(change, STRUCT)
        keep = np.zeros(n + 1, dtype=bool)
        if n:
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            keep[1:] = sizes[1:] * vml >= min_component_ml
        kept = keep[lab] & change
        d = dist[kept]
        depth_parts.append(d)
        if n and kept.any():
            maxes = ndimage.maximum(dist, lab, index=np.nonzero(keep)[0])
            comp_max.extend(np.atleast_1d(maxes).tolist())
        out[f"{prefix}_{name}_maxdepth_mm"] = float(d.max()) if d.size else 0.0
        for t in (3.0, 4.5):
            out[f"{prefix}_{name}_deep{t:g}_ml"] = float((d >= t - 1e-6).sum() * vml)
    d = np.concatenate(depth_parts)
    out[f"{prefix}_change_maxdepth_mm"] = float(d.max()) if d.size else 0.0
    out[f"{prefix}_change_p99depth_mm"] = float(np.percentile(d, 99)) if d.size else 0.0
    for t in (3.0, 4.5, 6.0):
        out[f"{prefix}_change_deep{t:g}_ml"] = float((d >= t - 1e-6).sum() * vml)
    comp_max.sort(reverse=True)
    out[f"{prefix}_change_top3depth_sum_mm"] = float(sum(comp_max[:3]))
    return out


def shape_features(inp: np.ndarray) -> dict:
    vol = inp.sum()
    surface = inp & ~ndimage.binary_erosion(inp, STRUCT)
    opened = ndimage.binary_opening(inp, STRUCT, iterations=2)
    border = np.zeros_like(inp)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    return {
        "feat_log_volume": float(np.log(max(vol, 1))),
        "feat_surface_to_volume": float(surface.sum() / max(vol, 1)),
        "feat_fine_detail_fraction": float((inp & ~opened).sum() / max(vol, 1)),
        "feat_border_touch_fraction": float((inp & border).sum() / max(surface.sum(), 1)),
    }


def case_dirs(root: Path) -> dict[str, Path]:
    return {p.name.removeprefix("patient_"): p for p in root.glob("patient_*") if p.is_dir()}


def loo_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.empty_like(y)
    A = np.column_stack([np.ones(len(y)), X])
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        pred[i] = A[i] @ coef
    return pred


def precision_at_k(score: np.ndarray, relevant: np.ndarray, k: int) -> float:
    top = np.argsort(-score, kind="stable")[:k]
    return float(relevant[top].mean())


def ranks(score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, kind="stable")
    r = np.empty(len(score), dtype=int)
    r[order] = np.arange(1, len(score) + 1)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True, help="<study>/evaluations (or evaluations_dev)")
    ap.add_argument("--arms", nargs="+", default=["nsrqa_corruption", "no_corruption"])
    ap.add_argument("--reference-arm", default="no_corruption")
    ap.add_argument("--seed", default="91")
    ap.add_argument("--mode", default="autoseg_centered")
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--split", default="testing1")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--watch", default="", help="comma-separated patient ids to report ranks for")
    ap.add_argument("--positives", default="",
                    help="comma-separated patient ids of notable shape cases; adds precision@k against them")
    ap.add_argument("--min-component-ml", type=float, default=0.05,
                    help="drop change components smaller than this before depth scores")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def arm_root(arm: str, kind: str) -> Path:
        return eval_root / arm / f"seed{args.seed}" / kind / args.mode / args.roi / args.split

    ref_cases = case_dirs(arm_root(args.reference_arm, "actual_autoseg_cases"))
    rows_by_arm: dict[str, list[dict]] = {}
    for arm in args.arms:
        cases = case_dirs(arm_root(arm, "actual_autoseg_cases"))
        clean = case_dirs(arm_root(arm, "clean_approved_cases"))
        rows = []
        for pid in sorted(cases, key=lambda s: (len(s), s)):
            d = cases[pid]
            m = json.loads((d / "metrics.json").read_text())
            vml = spacing_ml(d / "autoseg_input.nii.gz")
            spacing_zyx = tuple(reversed(sitk.ReadImage(str(d / "autoseg_input.nii.gz")).GetSpacing()))
            inp = load(d / "autoseg_input.nii.gz")
            rec = load(d / "nsrqa_reconstruction.nii.gz")
            app = load(d / "approved.nii.gz")
            row = {
                "arm": arm,
                "patient_id": pid,
                "true_1_minus_dice": m["true_1_minus_dice"],
                "true_hd95_mm": m["true_hd95_mm"],
                "qa_1_minus_dice": m["qa_1_minus_dice"],
                "qa_hd95_mm": m["qa_hd95_mm"],
                "qa_hd99_mm": m.get("qa_hd99_mm", np.nan),
                "qa_max_hd_mm": m.get("qa_max_hd_mm", np.nan),
            }
            if pid in clean:
                row["clean_recon_1_minus_dice"] = json.loads((clean[pid] / "metrics.json").read_text())["qa_1_minus_dice"]
            true_err = disagreement_scores(inp, app, vml, "true")
            row["true_error_ml"] = float((inp ^ app).sum() * vml)
            row.update({k: true_err[k] for k in ("true_thick1_ml", "true_thick1_maxcc_ml")})
            row.update(disagreement_scores(inp, rec, vml, "qa"))
            row.update(change_depth_scores(inp, rec, spacing_zyx, vml, args.min_component_ml))
            if arm != args.reference_arm and pid in ref_cases:
                ref_rec = load(ref_cases[pid] / "nsrqa_reconstruction.nii.gz")
                row["xarm_1_minus_dice"] = one_minus_dice(rec, ref_rec)
                row.update(disagreement_scores(ref_rec, rec, vml, "xarm"))  # excess = removed by this arm
            row.update(shape_features(inp))
            rows.append(row)
            print(f"[{arm}] patient {pid}: qa={row['qa_1_minus_dice']:.4f} thick1={row['qa_thick1_ml']:.2f} mL", flush=True)

        feats = [k for k in rows[0] if k.startswith("feat_")]
        if all("clean_recon_1_minus_dice" in r for r in rows):
            X = np.array([[r[f] for f in feats] for r in rows])
            y = np.array([r["clean_recon_1_minus_dice"] for r in rows])
            pred = loo_predict(X, y)
            for r, p in zip(rows, pred):
                r["pred_difficulty"] = float(p)
                r["resid_qa"] = r["qa_1_minus_dice"] - float(p)
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            rows[0]["_loo_r2"] = 1 - ss_res / ss_tot
            rows[0]["_loo_spearman"] = float(stats.spearmanr(y, pred)[0])
        rows_by_arm[arm] = rows

    score_cols = [
        "qa_1_minus_dice", "qa_hd95_mm", "qa_hd99_mm", "qa_max_hd_mm",
        "qa_change_maxdepth_mm", "qa_change_p99depth_mm", "qa_change_top3depth_sum_mm",
        "qa_change_deep3_ml", "qa_change_deep4.5_ml", "qa_change_deep6_ml",
        "qa_removed_maxdepth_mm", "qa_added_maxdepth_mm",
        "qa_removed_deep3_ml", "qa_added_deep3_ml",
        "qa_thick1_ml", "qa_thick1_maxcc_ml", "qa_thick2_ml", "qa_thick2_maxcc_ml",
        "qa_thick1_excess_ml", "qa_thick1_missing_ml",
        "xarm_1_minus_dice", "xarm_thick1_ml", "xarm_thick1_maxcc_ml",
        "resid_qa",
    ]
    targets = ["true_1_minus_dice", "true_error_ml", "true_thick1_ml"]
    watch = [w.strip() for w in args.watch.split(",") if w.strip()]
    positives = {p.strip() for p in args.positives.split(",") if p.strip()}

    summary = []
    for arm, rows in rows_by_arm.items():
        with (out_dir / f"{arm}_floor_corrected_scores.csv").open("w", newline="", encoding="utf-8") as f:
            keys = sorted({k for r in rows for k in r if not k.startswith("_")}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        truth = {t: np.array([r[t] for r in rows]) for t in targets}
        relevant = np.zeros(len(rows), dtype=bool)
        relevant[np.argsort(-truth["true_thick1_ml"], kind="stable")[: args.top_k]] = True
        clean = np.array([r.get("clean_recon_1_minus_dice", np.nan) for r in rows])
        pids = [r["patient_id"] for r in rows]
        for col in score_cols:
            if not all(col in r for r in rows):
                continue
            s = np.array([r[col] for r in rows], dtype=float)
            rk = ranks(s)
            entry = {"arm": arm, "score": col}
            for t in targets:
                entry[f"rho_{t}"] = float(stats.spearmanr(s, truth[t])[0])
            entry["rho_clean_recon"] = float(stats.spearmanr(s, clean)[0]) if np.isfinite(clean).all() else np.nan
            entry[f"p@{args.top_k}_vs_top{args.top_k}_true_thick"] = precision_at_k(s, relevant, args.top_k)
            if positives:
                pos = np.array([p in positives for p in pids])
                entry[f"p@{args.top_k}_vs_positives"] = precision_at_k(s, pos, args.top_k)
                entry["positives_mean_rank"] = float(rk[pos].mean()) if pos.any() else np.nan
            for pid in watch:
                if pid in pids:
                    entry[f"rank_{pid}"] = int(rk[pids.index(pid)])
            summary.append(entry)
        if "_loo_r2" in rows[0]:
            print(f"[{arm}] LOO difficulty model: R2={rows[0]['_loo_r2']:.3f}, spearman={rows[0]['_loo_spearman']:.3f}")

    with (out_dir / "floor_corrected_summary.csv").open("w", newline="", encoding="utf-8") as f:
        keys = list(dict.fromkeys(k for e in summary for k in e))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
