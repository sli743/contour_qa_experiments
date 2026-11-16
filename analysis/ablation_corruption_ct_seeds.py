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

from analysis.ablation_corruption_ct import CELLS, SCORES, arm_metrics, effects, parse_combos, rank_average

MAIN_KEYS = [f"rho_{s}" for s in SCORES] + ["mean_change_precision", "mean_error_recall",
                                             "clean_p90_deep_common_ml"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", action="append", required=True, help="LABEL=<seed ablation dir>")
    ap.add_argument("--combine", action="append", default=None,
                    help="NAME=ARM+ARM rank-average combination; default: the combinations recorded in the "
                         "seed dirs' ablation_metadata.json (must agree across seeds)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seeds, recorded = {}, {}
    for spec in args.ablation_dir:
        lab, _, path = spec.partition("=")
        pc = pd.read_csv(Path(path) / "per_case.csv", dtype={"patient_id": str})
        if "deep_cal_ml" not in pc or pc["deep_cal_ml"].isna().all():
            raise SystemExit(f"{path}/per_case.csv has no calibrated scores; rerun ablation_corruption_ct")
        seeds[lab] = pc
        meta = Path(path) / "ablation_metadata.json"
        recorded[lab] = json.loads(meta.read_text()).get("combine", {}) if meta.exists() else {}
    arms = sorted(set.intersection(*[set(pc.arm) for pc in seeds.values()]),
                  key=lambda a: (a not in CELLS, CELLS.index(a) if a in CELLS else 0, a))
    if not set(CELLS) <= set(arms):
        raise SystemExit("every seed needs all four cells")
    if args.combine is not None:
        combos = parse_combos(args.combine, arms)
    else:
        found = list(recorded.values())
        if any(f != found[0] for f in found):
            raise SystemExit(f"seeds record different --combine sets: {recorded}; pass --combine explicitly")
        combos = parse_combos([f"{c}={'+'.join(p)}" for c, p in found[0].items()], arms)
    all_arms = arms + list(combos)
    pids = sorted(set.intersection(*[set(pc[pc.arm == a].patient_id) for pc in seeds.values() for a in arms]),
                  key=lambda s: (len(s), s))
    n = len(pids)
    by = {s: {a: pc[pc.arm == a].set_index("patient_id").loc[pids].reset_index() for a in arms}
          for s, pc in seeds.items()}
    # Truth is the autoseg-vs-approved error: identical across arms/seeds.
    truth = next(iter(by.values()))["C1I1"]["true_thick1_ml"].to_numpy()
    rel = np.zeros(n, bool)
    rel[np.argsort(-truth, kind="stable")[: args.top_k]] = True

    def metrics_for(d, idx=None):  # d[arm] -> df; combinations re-ranked on the (resampled) rows
        sub = {a: (df if idx is None else df.iloc[idx]) for a, df in d.items()}
        r = rel if idx is None else rel[idx]
        res = {a: arm_metrics(df, r, args.top_k) for a, df in sub.items()}
        for c, parts in combos.items():
            res[c] = arm_metrics(rank_average([sub[p] for p in parts]), r, args.top_k)
        return res

    point = {s: metrics_for(d) for s, d in by.items()}
    keys = [k for k in point[next(iter(point))]["C1I1"]]
    eff_keys = [k for k in keys if all(np.isfinite(point[s][c].get(k, np.nan)) for s in point for c in CELLS)]
    combo_cmp = [(c, p, k) for c, parts in combos.items() for p in parts
                 for k in point[next(iter(point))][c] if k in point[next(iter(point))][p]]

    def mean_over_seeds(m):  # m[seed][arm][key] -> [arm][key]
        return {a: {k: float(np.nanmean([m[s][a].get(k, np.nan) for s in m])) for k in m[next(iter(m))][a]}
                for a in all_arms}

    mean_pt = mean_over_seeds(point)
    rng = np.random.default_rng(0)
    boot_arm = {a: {k: [] for k in mean_pt[a]} for a in all_arms}
    boot_eff = {k: [] for k in eff_keys}
    boot_cmp = {t: [] for t in combo_cmp}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        m = {s: metrics_for(d, idx) for s, d in by.items()}
        mm = mean_over_seeds(m)
        for a in all_arms:
            for k in boot_arm[a]:
                boot_arm[a][k].append(mm[a].get(k, np.nan))
        for k in eff_keys:
            boot_eff[k].append(effects(mm, k))
        for (c, p, k) in combo_cmp:
            boot_cmp[(c, p, k)].append(mm[c][k] - mm[p][k])

    def ci(vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        return (np.percentile(v, 2.5), np.percentile(v, 97.5), float((v > 0).mean())) if v.size else (np.nan,) * 3

    rows = []
    for a in all_arms:
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
    for (c, p, k) in combo_cmp:
        name = f"combo:{c}-{p}"
        lo, hi, pp = ci(boot_cmp[(c, p, k)])
        sv = [point[s][c][k] - point[s][p][k] for s in point]
        eff_rows.append({"metric": k, "effect": name, "estimate": mean_pt[c][k] - mean_pt[p][k],
                         "ci_lo": lo, "ci_hi": hi, "p_gt_0": pp,
                         "seed_sd": float(np.std(sv, ddof=1)) if len(sv) > 1 else np.nan,
                         "seeds_gt_0": f"{sum(v > 0 for v in sv)}/{len(sv)}"})
        for s, v in zip(point, sv):
            eff_seed_rows.append({"seed": s, "metric": k, "effect": name, "estimate": v})
    eff = pd.DataFrame(eff_rows)
    per_seed_arm = pd.DataFrame([{"seed": s, "arm": a, **point[s][a]} for s in point for a in all_arms])

    summary.to_csv(out / "ablation_arm_summary.csv", index=False)
    eff.to_csv(out / "ablation_effects.csv", index=False)
    per_seed_arm.to_csv(out / "ablation_per_seed.csv", index=False)
    pd.DataFrame(eff_seed_rows).to_csv(out / "ablation_effects_per_seed.csv", index=False)
    (out / "ablation_metadata.json").write_text(json.dumps({
        "seeds": {s: str(p.partition("=")[2]) for s, p in zip(seeds, args.ablation_dir)},
        "n_patients": n, "top_k": args.top_k, "bootstrap": args.bootstrap, "combine": combos,
        "ci": "paired patient bootstrap of the seed-mean (case sampling); seed_sd = training spread",
    }, indent=2))

    main_keys = MAIN_KEYS + [f"ap_deep_common_ml", f"p@{args.top_k}_deep_common_ml"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(per_seed_arm[["seed", "arm"] + [k for k in main_keys if k in per_seed_arm]].round(3).to_string(index=False))
        print()
        print(summary[["arm"] + [k for k in main_keys if k in summary]].round(3).to_string(index=False))
        print()
        print(eff[eff.metric.isin(main_keys) & ~eff.effect.str.startswith("combo:")].round(3).to_string(index=False))
        if combos:
            print()
            cmp_keys = ["rho_deep_common_ml", "ap_deep_common_ml", f"p@{args.top_k}_deep_common_ml"]
            print(eff[eff.effect.str.startswith("combo:") & eff.metric.isin(cmp_keys)].round(3).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
