"""Add the region-growth (leak) branch to a fitted NSR-QA training config.

The fitted corruption only produces deep *missing* lung (region dropout); every
branch that adds lung moves the boundary by <= ~4 mm, while real excess errors
are 5-80 mm deep. This adds ``stochastic_region_growth``, the outward
counterpart of the dropout, with its size/depth window taken from the ROI's own
excess-type error components the same way the fit takes the dropout's from
missing-type ones:

  per training patient, the deepest excess component with depth >= --min-depth-mm;
  added_fraction  = [q_lo, q_hi] of fraction_approved_volume (sampled log-uniformly)
  max depth       = [max(q_lo, min-depth), q_hi] of max_depth_from_approved_surface_mm

Field shape parameters (control grid, correlation length, localization) are
copied from the config's dropout spec. Branch weight defaults to the dropout's,
so leaks are as common as dropouts; the other branches keep their relative weights.

  python -m analysis.add_growth_corruption --config in.yaml \
    --error-components <roi>_error_field_characterization_v1/error_components.csv --output out.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SHAPE_KEYS = ("coarse_grid_points_zyx", "correlation_length_mm", "localization_sigma_mm", "random_field_weight",
              "localization_weight", "localization_cutoff_sigma", "upsample_order")


def growth_targets(components_csv: str | Path, min_depth_mm: float = 4.5,
                   quantiles: tuple[float, float] = (0.10, 0.90), min_cases: int = 8) -> dict:
    x = pd.read_csv(components_csv, dtype={"patient_id": str})
    x = x[x["component_type"].astype(str).str.lower().eq("excess")].copy()
    for col in ("fraction_approved_volume", "max_depth_from_approved_surface_mm"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x[np.isfinite(x.fraction_approved_volume) & (x.fraction_approved_volume > 0)
          & (x.max_depth_from_approved_surface_mm >= min_depth_mm)]
    x = (x.sort_values(["patient_id", "max_depth_from_approved_surface_mm"], ascending=[True, False])
          .drop_duplicates("patient_id", keep="first"))
    if len(x) < min_cases:
        raise SystemExit(f"only {len(x)} patients with an excess component >= {min_depth_mm} mm; need {min_cases}")
    qlo, qhi = quantiles
    f = x.fraction_approved_volume.to_numpy(float)
    d = x.max_depth_from_approved_surface_mm.to_numpy(float)
    return {
        "added_fraction": [float(np.quantile(f, qlo)), float(np.quantile(f, qhi))],
        "min_max_depth_mm": max(float(np.quantile(d, qlo)), float(min_depth_mm)),
        "max_max_depth_mm": float(np.quantile(d, qhi)),
        "summary": {
            "source": f"deepest excess component per training patient, depth >= {min_depth_mm:g} mm",
            "n_patients": int(len(x)),
            "fraction_p50": float(np.median(f)), "depth_p50_mm": float(np.median(d)),
            "quantiles": [qlo, qhi], "components_csv": str(Path(components_csv).resolve()),
        },
    }


def add_growth(cfg: dict, targets: dict, weight: float | None = None, max_attempts: int = 24,
               search_steps: int = 16, rel_tolerance: float = 0.5) -> dict:
    out = copy.deepcopy(cfg)
    srd = out["stochastic_region_dropout"]
    drop = srd.get("dropout", {})
    spec = {k: copy.deepcopy(drop[k]) for k in SHAPE_KEYS if k in drop}
    spec.update({
        "added_fraction": targets["added_fraction"],
        "added_fraction_sampling": "log",
        "added_fraction_relative_tolerance": float(rel_tolerance),
        "min_max_depth_mm": targets["min_max_depth_mm"],
        "max_max_depth_mm": targets["max_max_depth_mm"],
        "max_attempts": int(max_attempts),
        "threshold_search_steps": int(search_steps),
        "calibration": targets["summary"],
    })
    srd["growth"] = spec
    bp = srd.setdefault("branch_probabilities", {})
    bp["stochastic_region_growth"] = float(weight if weight is not None else bp.get("stochastic_region_dropout", 0.20))
    return out


def smoke(cfg: dict, n_labels: int = 3, draws: int = 10, seed: int = 0) -> None:
    """Draw leaks on a few training labels and print acceptance, size and depth."""
    import random
    import time

    import SimpleITK as sitk

    from training.corruptions.stochastic_region_dropout import stochastic_region_growth

    srd = cfg["stochastic_region_dropout"]
    spacing = tuple(float(v) for v in srd.get("spacing_zyx", [1.5, 1.5, 1.5]))
    files = sorted(Path(cfg["data"]["train_label_dir"]).rglob("*.nii.gz"))[:n_labels]
    if not files:
        raise SystemExit(f"no labels under {cfg['data']['train_label_dir']}")
    rng = random.Random(seed)
    for f in files:
        target = sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0
        vml = float(np.prod(spacing)) / 1000.0
        res = []
        for i in range(draws):
            t0 = time.perf_counter()
            out, m = stochastic_region_growth(target, spacing, srd["growth"], rng, cache_key=f.name)
            added = out & ~target
            res.append((not m.get("growth_rejected_to_clean", 0), added.sum() * vml, m["growth_depth_max_mm"],
                        bool(np.array_equal(out & target, target)), m["growth_total_attempts"], time.perf_counter() - t0))
        acc = [r for r in res if r[0]]
        print(f"{f.name}: lung {target.sum() * vml:.0f} mL, accepted {len(acc)}/{draws}, "
              f"added mL {[round(r[1], 1) for r in acc]}, max depth mm {[round(r[2], 1) for r in acc]}, "
              f"target kept {all(r[3] for r in res)}, attempts {[r[4] for r in res]}, "
              f"s first draw (incl. cache) {res[0][5]:.1f}, s/draw after {np.mean([r[5] for r in res[1:]] or [0]):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--error-components", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-depth-mm", type=float, default=4.5)
    ap.add_argument("--quantiles", default="0.1,0.9")
    ap.add_argument("--weight", type=float, default=None, help="branch weight; default = dropout's")
    ap.add_argument("--smoke", type=int, default=0, help="draw this many leaks on 3 training labels and print stats")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    t = growth_targets(args.error_components, args.min_depth_mm, tuple(float(v) for v in args.quantiles.split(",")))
    new = add_growth(cfg, t, args.weight)
    Path(args.output).write_text(yaml.safe_dump(new, sort_keys=False), encoding="utf-8")
    print(json.dumps(t, indent=2))
    if args.smoke:
        smoke(new, draws=args.smoke)


if __name__ == "__main__":
    main()
