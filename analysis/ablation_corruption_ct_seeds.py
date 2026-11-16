"""Pool the corruption x CT 2x2 ablation over training seeds.

Reads <seed dir>/per_case.csv written by analysis.ablation_corruption_ct (one run
per seed, thresholds already calibrated per arm and seed) and reports:

  * per-seed point metrics and effects (training-seed spread), and
  * the seed-mean of every metric/effect, with a paired patient bootstrap: the
    same resampled patients are used in every seed and arm, the metric is computed
    per seed and then averaged. The CI therefore covers case sampling of the
    seed-averaged estimate; the per-seed table shows training variance.

Writes, in --output-dir:
  ablation_arm_summary.csv   seed-mean per arm (+ _lo/_hi, _sd over seeds); same
                             layout as the single-seed file, so
                             analysis.plot_ablation_corruption_ct works on it
  ablation_effects.csv       seed-mean effects with CI, P(>0) and the number of
                             seeds whose effect is > 0
  ablation_per_seed.csv      per-seed arm metrics
  ablation_effects_per_seed.csv

  python -m analysis.ablation_corruption_ct_seeds \
    --ablation-dir seed91=<...>/seed91_ablation --ablation-dir seed92=<...>/seed92_ablation \
    --output-dir <...>/ablation_seeds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.ablation_corruption_ct import CELLS, SCORES, arm_metrics, effects

MAIN_KEYS = [f"rho_{s}" for s in SCORES] + ["mean_change_precision", "mean_error_recall",
                                             "clean_p90_deep_common_ml"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", action="append", required=True, help="LABEL=<seed ablation dir>")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seeds = {}
    for spec in args.ablation_dir:
        lab, _, path = spec.partition("=")
        pc = pd.read_csv(Path(path) / "per_case.csv", dtype={"patient_id": str})
        if "deep_cal_ml" not in pc or pc["deep_cal_ml"].isna().all():
            raise SystemExit(f"{path}/per_case.csv has no calibrated scores; rerun ablation_corruption_ct")
        seeds[lab] = pc
    arms = sorted(set.intersection(*[set(pc.arm) for pc in seeds.values()]),
                  key=lambda a: (a not in CELLS, CELLS.index(a) if a in CELLS else 0, a))
    if not set(CELLS) <= set(arms):
        raise SystemExit("every seed needs all four cells")
    pids = sorted(set.intersection(*[set(pc[pc.arm == a].patient_id) for pc in seeds.values() for a in arms]),
                  key=lambda s: (len(s), s))
    n = len(pids)
    by = {s: {a: pc[pc.arm == a].set_index("patient_id").loc[pids].reset_index() for a in arms}
          for s, pc in seeds.items()}
    # Truth is the autoseg-vs-approved error: identical across arms/seeds.
    truth = next(iter(by.values()))["C1I1"]["true_thick1_ml"].to_numpy()
    rel = np.zeros(n, bool)
    rel[np.argsort(-truth, kind="stable")[: args.top_k]] = True

    point = {s: {a: arm_metrics(df, rel, args.top_k) for a, df in d.items()} for s, d in by.items()}
    keys = [k for k in point[next(iter(point))]["C1I1"]]
    eff_keys = [k for k in keys if all(np.isfinite(point[s][c].get(k, np.nan)) for s in point for c in CELLS)]

    def mean_over_seeds(m):  # m[seed][arm][key] -> [arm][key]
        return {a: {k: float(np.nanmean([m[s][a].get(k, np.nan) for s in m])) for k in m[next(iter(m))][a]}
                for a in arms}

    mean_pt = mean_over_seeds(point)
    rng = np.random.default_rng(0)
    boot_arm = {a: {k: [] for k in mean_pt[a]} for a in arms}
    boot_eff = {k: [] for k in eff_keys}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        m = {s: {a: arm_metrics(df.iloc[idx], rel[idx], args.top_k) for a, df in d.items()} for s, d in by.items()}
        mm = mean_over_seeds(m)
        for a in arms:
            for k in boot_arm[a]:
                boot_arm[a][k].append(mm[a].get(k, np.nan))
        for k in eff_keys:
            boot_eff[k].append(effects(mm, k))

    def ci(vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        return (np.percentile(v, 2.5), np.percentile(v, 97.5), float((v > 0).mean())) if v.size else (np.nan,) * 3

    rows = []
    for a in arms:
        e = {"arm": a, "n": n, "n_seeds": len(seeds)}
        for k, v in mean_pt[a].items():
            lo, hi, _ = ci(boot_arm[a][k])
            e[k], e[f"{k}_lo"], e[f"{k}_hi"] = v, lo, hi
            e[f"{k}_sd"] = float(np.nanstd([point[s][a].get(k, np.nan) for s in point], ddof=1)) if len(seeds) > 1 else np.nan
        rows.append(e)
    summary = pd.DataFrame(rows)

    eff_rows, eff_seed_rows = [], []
    for k in eff_keys:
        per_seed = {s: effects(point[s], k) for s in point}
        for name, val in effects(mean_pt, k).items():
            lo, hi, pp = ci([b[name] for b in boot_eff[k]])
            sv = [per_seed[s][name] for s in per_seed]
            eff_rows.append({"metric": k, "effect": name, "estimate": val, "ci_lo": lo, "ci_hi": hi, "p_gt_0": pp,
                             "seed_sd": float(np.std(sv, ddof=1)) if len(sv) > 1 else np.nan,
                             "seeds_gt_0": f"{sum(v > 0 for v in sv)}/{len(sv)}"})
            for s in per_seed:
                eff_seed_rows.append({"seed": s, "metric": k, "effect": name, "estimate": per_seed[s][name]})
    eff = pd.DataFrame(eff_rows)
    per_seed_arm = pd.DataFrame([{"seed": s, "arm": a, **point[s][a]} for s in point for a in arms])

    summary.to_csv(out / "ablation_arm_summary.csv", index=False)
    eff.to_csv(out / "ablation_effects.csv", index=False)
    per_seed_arm.to_csv(out / "ablation_per_seed.csv", index=False)
    pd.DataFrame(eff_seed_rows).to_csv(out / "ablation_effects_per_seed.csv", index=False)
    (out / "ablation_metadata.json").write_text(json.dumps({
        "seeds": {s: str(p.partition("=")[2]) for s, p in zip(seeds, args.ablation_dir)},
        "n_patients": n, "top_k": args.top_k, "bootstrap": args.bootstrap,
        "ci": "paired patient bootstrap of the seed-mean (case sampling); seed_sd = training spread",
    }, indent=2))

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(per_seed_arm[["seed", "arm"] + [k for k in MAIN_KEYS if k in per_seed_arm]].round(3).to_string(index=False))
        print()
        print(summary[["arm"] + [k for k in MAIN_KEYS if k in summary]].round(3).to_string(index=False))
        print()
        print(eff[eff.metric.isin(MAIN_KEYS)].round(3).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
