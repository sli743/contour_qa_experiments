#!/usr/bin/env python3
"""Fit Lung_L NSR-QA corruption severity to real approved/autoseg error metrics.

The corruption vocabulary and mixture probabilities are held fixed.  Only four
family-level severity multipliers are optimized:

* smooth_warp
* distance_noise (also used by the combined branch)
* stochastic_region_dropout (fraction and accepted depth scale together)
* global_boundary_offset

The fit target is the realized distribution of real approved-vs-autoseg errors
in training1.  Candidate settings are scored by robust quantile matching of
1-Dice, ASSD, HD95, missing fraction, and excess fraction.  For Lung_L, HD95 is
explicitly emphasized; HD99 is saved as an audit metric but has zero fitting
weight.

The deliberate clean branch is not included in severity fitting.  Synthetic
samples are drawn from the fixed non-clean branch mixture, and no-op/rejected
corruptions are penalized separately.  This prevents the 20% clean branch from
artificially pulling the fitted severity toward zero.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

from analysis.calibrate_lung_l_nsrqa_severity import (
    build_patient_index,
    export_example_case,
    finite_series,
    forced_branch,
    image_array,
    pair_metrics_plus,
    quantile_comparison,
    select_example_patients,
    write_plots,
)


BRANCHES = [
    "smooth_warp",
    "distance_noise",
    "combined",
    "stochastic_region_dropout",
    "global_boundary_offset",
]

# Per-process image cache. Candidate fitting is deliberately parallelized by
# candidate, and each worker reuses its already-loaded training masks.
_MASK_CACHE: Dict[str, Tuple[sitk.Image, np.ndarray, Tuple[float, float, float]]] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/dino_shape/lung_l_nsrqa_synthetic_severity_fit.yaml",
    )
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def resolve_path(value: str | Path, cwd: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return path


def load_initial_training_config(cfg: Mapping, cwd: Path) -> Tuple[Dict, Path, str]:
    primary = resolve_path(cfg["initial_training_config"], cwd)
    if primary.is_file():
        return yaml.safe_load(primary.read_text()), primary, "initial_training_config"
    fallback = resolve_path(cfg["fallback_training_config"], cwd)
    if not fallback.is_file():
        raise FileNotFoundError(
            f"Neither initial nor fallback training config exists: {primary}, {fallback}"
        )
    print(f"Initial calibrated config not found: {primary}")
    print(f"Falling back to: {fallback}")
    return yaml.safe_load(fallback.read_text()), fallback, "fallback_training_config"


def assert_original_mixture(training_cfg: Mapping, cfg: Mapping) -> None:
    spec = training_cfg.get("stochastic_region_dropout", {})
    expected = cfg["fixed_mixture"]
    clean = float(spec.get("clean_probability", -1.0))
    if not math.isclose(clean, float(expected["clean_probability"]), abs_tol=1e-8):
        raise ValueError(
            f"clean_probability changed: {clean} != {expected['clean_probability']}"
        )
    got = spec.get("branch_probabilities", {})
    for branch in BRANCHES:
        gv = float(got.get(branch, -1.0))
        ev = float(expected["branch_probabilities"][branch])
        if not math.isclose(gv, ev, abs_tol=1e-8):
            raise ValueError(f"branch probability changed for {branch}: {gv} != {ev}")


def _scale_pair(
    values: Sequence[float], multiplier: float, lower: float, upper: float
) -> List[float]:
    a, b = float(values[0]), float(values[1])
    out = [min(upper, max(lower, a * multiplier)), min(upper, max(lower, b * multiplier))]
    out.sort()
    if out[1] <= out[0]:
        out[1] = min(upper, out[0] + max(1e-4, 0.05 * max(out[0], 1e-3)))
    return [float(out[0]), float(out[1])]


def config_with_multipliers(
    initial_cfg: Mapping, multipliers: Mapping[str, float], limits: Mapping
) -> Dict:
    out = copy.deepcopy(dict(initial_cfg))
    spec = out["stochastic_region_dropout"]

    spec["smooth_warp"]["max_displacement_mm"] = _scale_pair(
        spec["smooth_warp"]["max_displacement_mm"],
        float(multipliers["smooth"]),
        float(limits["smooth_mm"][0]),
        float(limits["smooth_mm"][1]),
    )
    spec["distance_noise"]["amplitude_mm"] = _scale_pair(
        spec["distance_noise"]["amplitude_mm"],
        float(multipliers["noise"]),
        float(limits["noise_mm"][0]),
        float(limits["noise_mm"][1]),
    )
    spec["global_boundary_offset"]["magnitude_mm"] = _scale_pair(
        spec["global_boundary_offset"]["magnitude_mm"],
        float(multipliers["global"]),
        float(limits["global_mm"][0]),
        float(limits["global_mm"][1]),
    )

    drop = spec["dropout"]
    drop["removed_fraction"] = _scale_pair(
        drop["removed_fraction"],
        float(multipliers["dropout"]),
        float(limits["dropout_fraction"][0]),
        float(limits["dropout_fraction"][1]),
    )
    d_lo = float(drop.get("min_max_depth_mm", 0.0)) * float(multipliers["dropout"])
    d_hi = float(drop.get("max_max_depth_mm", 0.0)) * float(multipliers["dropout"])
    d_lo = min(float(limits["dropout_depth_mm"][1]), max(float(limits["dropout_depth_mm"][0]), d_lo))
    d_hi = min(float(limits["dropout_depth_mm"][1]), max(float(limits["dropout_depth_mm"][0]), d_hi))
    if d_hi < d_lo:
        d_lo, d_hi = d_hi, d_lo
    if d_hi <= d_lo:
        d_hi = min(float(limits["dropout_depth_mm"][1]), d_lo + 0.5)
    drop["min_max_depth_mm"] = float(d_lo)
    drop["max_max_depth_mm"] = float(d_hi)

    out.setdefault("study_metadata", {})["synthetic_severity_fit_multipliers"] = {
        k: float(v) for k, v in multipliers.items()
    }
    return out


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    names = list(BRANCHES)
    vals = [max(0.0, float(weights.get(name, 0.0))) for name in names]
    total = sum(vals)
    if total <= 0:
        raise ValueError("Non-clean branch weights sum to zero")
    draw = rng.random() * total
    acc = 0.0
    for name, weight in zip(names, vals):
        acc += weight
        if draw <= acc:
            return name
    return names[-1]


def make_common_tasks(
    patient_ids: Sequence[int], training_cfg: Mapping, n_samples: int, seed: int
) -> List[Tuple[int, int, str, int]]:
    if not patient_ids:
        raise ValueError("No patient IDs available for fitting")
    branch_weights = training_cfg["stochastic_region_dropout"]["branch_probabilities"]
    rng = random.Random(int(seed))
    tasks: List[Tuple[int, int, str, int]] = []
    for i in range(int(n_samples)):
        pid = int(patient_ids[rng.randrange(len(patient_ids))])
        branch = _weighted_choice(rng, branch_weights)
        sample_seed = rng.randrange(1, 2**31 - 1)
        tasks.append((i, pid, branch, sample_seed))
    return tasks


def _load_mask_cached(path: str) -> Tuple[sitk.Image, np.ndarray, Tuple[float, float, float]]:
    cached = _MASK_CACHE.get(path)
    if cached is not None:
        return cached
    ref, arr = image_array(Path(path))
    spacing = tuple(float(v) for v in ref.GetSpacing()[::-1])
    _MASK_CACHE[path] = (ref, arr, spacing)
    return ref, arr, spacing


def cropped_pair_metrics_plus(
    clean: np.ndarray, corrupted: np.ndarray, spacing_zyx: Sequence[float], pad_vox: int = 3
) -> Dict[str, float]:
    union = clean | corrupted
    pts = np.argwhere(union)
    if not len(pts):
        return pair_metrics_plus(clean, corrupted, spacing_zyx)
    lo = np.maximum(0, pts.min(axis=0) - int(pad_vox))
    hi = np.minimum(np.asarray(union.shape) - 1, pts.max(axis=0) + int(pad_vox))
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
    return pair_metrics_plus(clean[sl], corrupted[sl], spacing_zyx)


def robust_target(real_df: pd.DataFrame, objective_cfg: Mapping) -> Tuple[Dict, pd.DataFrame]:
    probs = [float(q) for q in objective_cfg["quantiles"]]
    qweights = {float(k): float(v) for k, v in objective_cfg["quantile_weights"].items()}
    metric_weights = {str(k): float(v) for k, v in objective_cfg["metric_weights"].items()}
    target: Dict[str, Dict] = {}
    rows: List[Dict] = []
    for metric, metric_weight in metric_weights.items():
        values = finite_series(real_df, metric)
        if len(values) < int(objective_cfg.get("minimum_real_cases", 12)):
            if metric_weight > 0:
                raise RuntimeError(f"Too few finite real values for fitted metric {metric}: {len(values)}")
            continue
        arr = values.to_numpy(dtype=float)
        q = {p: float(np.quantile(arr, p)) for p in probs}
        p10, p25, p50, p75, p90 = [float(np.quantile(arr, p)) for p in (0.10, 0.25, 0.50, 0.75, 0.90)]
        # Scale by the observed robust spread. The median term handles nearly
        # constant positive metrics; the tiny epsilon only prevents division by zero.
        scale = max(p90 - p10, p75 - p25, 0.25 * abs(p50), 1e-8)
        target[metric] = {
            "weight": metric_weight,
            "quantiles": q,
            "scale": float(scale),
            "qweights": qweights,
            "n": int(len(arr)),
        }
        row = {
            "metric": metric,
            "fit_weight": metric_weight,
            "n": int(len(arr)),
            "robust_scale": float(scale),
        }
        for p, value in q.items():
            row[f"q{int(round(p * 100)):02d}"] = value
        rows.append(row)
    return target, pd.DataFrame(rows)


def objective_from_rows(
    rows: Sequence[Mapping], target: Mapping[str, Mapping], objective_cfg: Mapping
) -> Tuple[float, Dict, List[Dict]]:
    n_total = len(rows)
    changed_threshold = float(objective_cfg.get("minimum_changed_fraction_gt", 1e-8))
    changed_rows = [r for r in rows if float(r.get("changed_fraction_gt", 0.0)) > changed_threshold]
    noop_rate = 1.0 - (len(changed_rows) / max(n_total, 1))
    if len(changed_rows) < int(objective_cfg.get("minimum_changed_samples", 30)):
        return float("inf"), {
            "n_samples": n_total,
            "n_changed": len(changed_rows),
            "noop_rate": noop_rate,
            "quantile_loss": float("inf"),
            "noop_penalty": float("inf"),
        }, []

    frame = pd.DataFrame(changed_rows)
    loss_sum = 0.0
    metric_weight_sum = 0.0
    breakdown: List[Dict] = []
    for metric, spec in target.items():
        weight = float(spec["weight"])
        if weight <= 0:
            continue
        values = finite_series(frame, metric)
        if len(values) < int(objective_cfg.get("minimum_changed_samples", 30)):
            return float("inf"), {
                "n_samples": n_total,
                "n_changed": len(changed_rows),
                "noop_rate": noop_rate,
                "quantile_loss": float("inf"),
                "noop_penalty": float("inf"),
            }, breakdown
        arr = values.to_numpy(dtype=float)
        qloss = 0.0
        qweight_sum = 0.0
        for q, real_value in spec["quantiles"].items():
            syn_value = float(np.quantile(arr, q))
            qw = float(spec["qweights"].get(float(q), 1.0))
            residual = (syn_value - float(real_value)) / float(spec["scale"])
            qloss += qw * residual * residual
            qweight_sum += qw
            breakdown.append({
                "metric": metric,
                "quantile": float(q),
                "real_value": float(real_value),
                "synthetic_value": syn_value,
                "robust_scale": float(spec["scale"]),
                "standardized_residual": float(residual),
                "metric_weight": weight,
                "quantile_weight": qw,
            })
        qloss /= max(qweight_sum, 1e-12)
        loss_sum += weight * qloss
        metric_weight_sum += weight

    quantile_loss = loss_sum / max(metric_weight_sum, 1e-12)
    target_noop = float(objective_cfg.get("target_noop_rate", 0.05))
    excess_noop = max(0.0, noop_rate - target_noop)
    noop_penalty = float(objective_cfg.get("noop_penalty_weight", 4.0)) * excess_noop * excess_noop
    total = quantile_loss + noop_penalty
    return float(total), {
        "n_samples": n_total,
        "n_changed": len(changed_rows),
        "noop_rate": float(noop_rate),
        "quantile_loss": float(quantile_loss),
        "noop_penalty": float(noop_penalty),
    }, breakdown


def simulate_candidate(
    candidate_id: str,
    multipliers: Mapping[str, float],
    initial_cfg: Mapping,
    limits: Mapping,
    tasks: Sequence[Tuple[int, int, str, int]],
    label_paths: Mapping[int, str],
    target: Mapping[str, Mapping],
    objective_cfg: Mapping,
) -> Dict:
    try:
        try:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        except Exception:
            pass
        candidate_cfg = config_with_multipliers(initial_cfg, multipliers, limits)
        spec = candidate_cfg["stochastic_region_dropout"]
        rows: List[Dict] = []
        branch_noops = {b: [0, 0] for b in BRANCHES}
        for sample_id, pid, branch, seed in tasks:
            _ref, clean, spacing = _load_mask_cached(label_paths[int(pid)])
            corrupted, meta = forced_branch(
                clean,
                spacing,
                spec,
                branch,
                random.Random(int(seed)),
                cache_key=f"severity-fit-{pid}",
            )
            changed = np.logical_xor(clean, corrupted)
            changed_fraction = float(changed.sum() / max(int(clean.sum()), 1))
            branch_noops[branch][1] += 1
            if changed_fraction <= float(objective_cfg.get("minimum_changed_fraction_gt", 1e-8)):
                branch_noops[branch][0] += 1
                metrics = {
                    "one_minus_dice": 0.0,
                    "assd_mm": 0.0,
                    "hd95_mm": 0.0,
                    "hd99_mm": 0.0,
                    "missing_fraction_gt": 0.0,
                    "excess_fraction_gt": 0.0,
                    "changed_fraction_gt": 0.0,
                }
            else:
                metrics = cropped_pair_metrics_plus(clean, corrupted, spacing)
            rows.append({
                "sample_id": int(sample_id),
                "patient_id": int(pid),
                "branch": branch,
                "changed_fraction_gt": changed_fraction,
                **meta,
                **metrics,
            })
        objective, summary, breakdown = objective_from_rows(rows, target, objective_cfg)
        result = {
            "ok": True,
            "candidate_id": candidate_id,
            "objective": objective,
            **{f"scale_{k}": float(v) for k, v in multipliers.items()},
            **summary,
        }
        for branch, (noops, total) in branch_noops.items():
            result[f"noop_{branch}"] = float(noops / max(total, 1))
        # Compact per-metric aggregate residual helps diagnose what drove the fit.
        if breakdown:
            bdf = pd.DataFrame(breakdown)
            for metric, g in bdf.groupby("metric"):
                result[f"rmse_{metric}"] = float(np.sqrt(np.average(
                    g["standardized_residual"].to_numpy() ** 2,
                    weights=g["quantile_weight"].to_numpy(),
                )))
        return result
    except Exception as exc:
        return {
            "ok": False,
            "candidate_id": candidate_id,
            **{f"scale_{k}": float(v) for k, v in multipliers.items()},
            "error": repr(exc),
            "objective": float("inf"),
        }


def evaluate_candidates(
    candidates: Sequence[Tuple[str, Mapping[str, float]]],
    initial_cfg: Mapping,
    limits: Mapping,
    tasks: Sequence[Tuple[int, int, str, int]],
    label_paths: Mapping[int, str],
    target: Mapping[str, Mapping],
    objective_cfg: Mapping,
    workers: int,
) -> pd.DataFrame:
    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futures = {
            ex.submit(
                simulate_candidate,
                cid,
                scales,
                initial_cfg,
                limits,
                tasks,
                label_paths,
                target,
                objective_cfg,
            ): cid
            for cid, scales in candidates
        }
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            done += 1
            print(
                f"  candidate {done}/{len(futures)} {result['candidate_id']}: "
                f"objective={result.get('objective', float('inf')):.4f}"
            )
    return pd.DataFrame(results).sort_values("objective", kind="stable")


def _clip_scale(value: float, bounds: Sequence[float]) -> float:
    return float(min(float(bounds[1]), max(float(bounds[0]), float(value))))


def stage1_candidates(search_cfg: Mapping, seed: int) -> List[Tuple[str, Dict[str, float]]]:
    bounds = search_cfg["multiplier_bounds"]
    names = ["smooth", "noise", "dropout", "global"]
    out: List[Tuple[str, Dict[str, float]]] = []
    seen = set()

    def add(cid: str, vals: Mapping[str, float]) -> None:
        clean = {k: _clip_scale(float(vals[k]), bounds[k]) for k in names}
        key = tuple(round(clean[k], 6) for k in names)
        if key not in seen:
            seen.add(key)
            out.append((cid, clean))

    add("stage1_baseline", {k: 1.0 for k in names})
    for scale in search_cfg.get("isotropic_scales", [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]):
        add(f"stage1_iso_{scale:g}", {k: float(scale) for k in names})

    rng = np.random.default_rng(int(seed))
    n_random = int(search_cfg.get("n_random_candidates", 36))
    for i in range(n_random):
        vals = {}
        for name in names:
            lo, hi = [float(v) for v in bounds[name]]
            vals[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        add(f"stage1_random_{i:03d}", vals)
    return out


def refinement_candidates(
    best: Mapping, search_cfg: Mapping, seed: int
) -> List[Tuple[str, Dict[str, float]]]:
    bounds = search_cfg["multiplier_bounds"]
    names = ["smooth", "noise", "dropout", "global"]
    center = {k: float(best[f"scale_{k}"]) for k in names}
    out: List[Tuple[str, Dict[str, float]]] = [("stage2_best_stage1", dict(center))]
    seen = {tuple(round(center[k], 6) for k in names)}
    rng = np.random.default_rng(int(seed))
    sigma = float(search_cfg.get("refine_log_sigma", 0.25))
    for i in range(int(search_cfg.get("n_refine_candidates", 24))):
        vals = {
            k: _clip_scale(center[k] * float(np.exp(rng.normal(0.0, sigma))), bounds[k])
            for k in names
        }
        key = tuple(round(vals[k], 6) for k in names)
        if key in seen:
            continue
        seen.add(key)
        out.append((f"stage2_refine_{i:03d}", vals))
    # Add one-at-a-time local perturbations so the result is interpretable even
    # when random refinement happens to miss a useful coordinate direction.
    for name in names:
        for factor in (0.8, 1.25):
            vals = dict(center)
            vals[name] = _clip_scale(vals[name] * factor, bounds[name])
            key = tuple(round(vals[k], 6) for k in names)
            if key not in seen:
                seen.add(key)
                out.append((f"stage2_{name}_{factor:g}", vals))
    return out


def validation_sample_worker(task: Tuple) -> Dict:
    sample_id, pid, branch, seed, label_path, best_cfg, objective_cfg = task
    try:
        try:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        except Exception:
            pass
        _ref, clean, spacing = _load_mask_cached(label_path)
        corrupted, meta = forced_branch(
            clean,
            spacing,
            best_cfg["stochastic_region_dropout"],
            branch,
            random.Random(int(seed)),
            cache_key=f"severity-validation-{pid}",
        )
        changed = np.logical_xor(clean, corrupted)
        changed_fraction = float(changed.sum() / max(int(clean.sum()), 1))
        if changed_fraction <= float(objective_cfg.get("minimum_changed_fraction_gt", 1e-8)):
            metrics = {
                "dice": 1.0,
                "one_minus_dice": 0.0,
                "assd_mm": 0.0,
                "hd95_mm": 0.0,
                "hd99_mm": 0.0,
                "missing_fraction_gt": 0.0,
                "excess_fraction_gt": 0.0,
                "changed_fraction_gt": 0.0,
            }
            is_noop = True
        else:
            metrics = cropped_pair_metrics_plus(clean, corrupted, spacing)
            is_noop = False
        return {
            "ok": True,
            "sample_id": int(sample_id),
            "patient_id": int(pid),
            "branch": branch,
            "is_noop": int(is_noop),
            **meta,
            **metrics,
        }
    except Exception as exc:
        return {
            "ok": False,
            "sample_id": int(sample_id),
            "patient_id": int(pid),
            "branch": branch,
            "error": repr(exc),
        }


def run_validation_audit(
    best_cfg: Mapping,
    patient_ids: Sequence[int],
    labels: Mapping[int, Path],
    objective_cfg: Mapping,
    audit_cfg: Mapping,
    workers: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = make_common_tasks(patient_ids, best_cfg, int(audit_cfg.get("n_samples", 600)), int(seed))
    tasks = [
        (sid, pid, branch, sample_seed, str(labels[pid]), best_cfg, objective_cfg)
        for sid, pid, branch, sample_seed in common
    ]
    rows: List[Dict] = []
    failures: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futures = [ex.submit(validation_sample_worker, task) for task in tasks]
        done = 0
        for fut in as_completed(futures):
            row = fut.result()
            (rows if row.get("ok") else failures).append(row)
            done += 1
            if done % 50 == 0 or done == len(futures):
                print(f"  validation {done}/{len(futures)}")
    good = pd.DataFrame(rows).sort_values("sample_id") if rows else pd.DataFrame()
    bad = pd.DataFrame(failures).sort_values("sample_id") if failures else pd.DataFrame()
    return good, bad


def fitted_severity_strata(best_cfg: Mapping) -> Dict:
    spec = best_cfg["stochastic_region_dropout"]

    def triple(pair: Sequence[float]) -> Dict[str, float]:
        a, b = float(pair[0]), float(pair[1])
        return {"p10": a, "p50": 0.5 * (a + b), "p90": b}

    drop = spec["dropout"]
    depth = [float(drop["min_max_depth_mm"]), float(drop["max_max_depth_mm"])]
    return {
        "smooth_warp": triple(spec["smooth_warp"]["max_displacement_mm"]),
        "distance_noise": triple(spec["distance_noise"]["amplitude_mm"]),
        "global_boundary_offset": triple(spec["global_boundary_offset"]["magnitude_mm"]),
        "dropout_fraction": triple(drop["removed_fraction"]),
        "dropout_depth": triple(depth),
    }


def plot_candidate_objectives(candidate_df: pd.DataFrame, outdir: Path) -> None:
    good = candidate_df[np.isfinite(pd.to_numeric(candidate_df["objective"], errors="coerce"))].copy()
    if not len(good):
        return
    good = good.sort_values("objective").reset_index(drop=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(len(good)), good["objective"].to_numpy(dtype=float), marker="o", linewidth=1)
    plt.xlabel("candidate rank")
    plt.ylabel("quantile-fit objective")
    plt.title("Lung_L synthetic severity candidate fit")
    plt.tight_layout()
    plt.savefig(outdir / "figures" / "candidate_objective_rank.png", dpi=180)
    plt.close()


def write_report(
    outdir: Path,
    cfg: Mapping,
    initial_path: Path,
    initial_source: str,
    best_row: Mapping,
    best_cfg: Mapping,
    target_df: pd.DataFrame,
    candidates: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation_failures: pd.DataFrame,
    examples: Sequence[int],
) -> None:
    fit_metrics = cfg["objective"]["metric_weights"]
    validation_changed = validation_df[validation_df.get("is_noop", 0).astype(int) == 0].copy() if len(validation_df) else pd.DataFrame()
    noop_rate = float(validation_df["is_noop"].mean()) if len(validation_df) and "is_noop" in validation_df else float("nan")
    clean_prob = float(best_cfg["stochastic_region_dropout"]["clean_probability"])
    effective_clean = clean_prob + (1.0 - clean_prob) * noop_rate if np.isfinite(noop_rate) else float("nan")

    lines = [
        "# Lung_L NSR-QA realized synthetic-severity fit",
        "",
        "## Fixed experimental design",
        "",
        "The original NSR-QA corruption vocabulary and branch probabilities are fixed. Only four family-level severity multipliers are fitted. The combined branch inherits the fitted smooth-warp and distance-noise severities.",
        "",
        f"Initial config source: `{initial_source}` = `{initial_path}`",
        f"Clean probability (not fitted): {clean_prob:.3f}",
        f"Branch probabilities (not fitted): `{json.dumps(best_cfg['stochastic_region_dropout']['branch_probabilities'], sort_keys=True)}`",
        "",
        "## Objective",
        "",
        "Severity is selected by matching realized synthetic and real approved-vs-autoseg quantiles at P25/P50/P75/P90. Each residual is divided by that real metric's robust spread. Intentional clean samples are excluded from fitting; non-clean samples that make no voxel change are penalized separately.",
        "",
        "Metric weights:",
    ]
    for metric, weight in fit_metrics.items():
        suffix = " (audit only)" if float(weight) == 0 else ""
        lines.append(f"- `{metric}`: {float(weight):g}{suffix}")
    lines += [
        "",
        "For Lung_L, **HD95 is intentionally emphasized and HD99 has zero fitting weight**. HD99 remains in the final audit files and plots.",
        "",
        "Target quantiles:",
        "",
        target_df.to_markdown(index=False),
        "",
        "## Best fit",
        "",
        f"Objective: {float(best_row['objective']):.5f}",
        f"- smooth multiplier: {float(best_row['scale_smooth']):.4f}",
        f"- distance-noise multiplier: {float(best_row['scale_noise']):.4f}",
        f"- dropout multiplier: {float(best_row['scale_dropout']):.4f}",
        f"- global-offset multiplier: {float(best_row['scale_global']):.4f}",
        f"- fit no-op rate: {float(best_row.get('noop_rate', float('nan'))):.4f}",
        "",
        "Final severity config:",
        "",
        "```yaml",
        yaml.safe_dump(best_cfg["stochastic_region_dropout"], sort_keys=False).rstrip(),
        "```",
        "",
        "## Independent exact-sampler audit",
        "",
        f"Validation draws: {len(validation_df)} successful; {len(validation_failures)} failed.",
        f"Non-clean no-op rate: {noop_rate:.4f}",
        f"Expected effective clean fraction during training = intentional clean + rejected/no-op non-clean: {effective_clean:.4f}",
        "",
        f"Candidates evaluated: {len(candidates)}",
        "",
        "## Inspection bank",
        "",
        f"Patients: {', '.join(map(str, examples))}",
        "Open `example_bank/` with `TrainingCorruptionExampleViewer`. The low/typical/high groups correspond to the fitted severity range minimum/midpoint/maximum, not empirical quantiles of real error.",
        "",
        "## Main outputs",
        "",
        "- `best_fit_training_config.yaml`: fitted config to use for the corruption arm",
        "- `best_fit_multipliers.json`: four fitted family multipliers",
        "- `candidate_search.csv`: complete objective search",
        "- `target_quantiles.csv`: real fitting targets and metric weights",
        "- `best_fit_validation_audit.csv`: independent exact-sampler draws",
        "- `best_fit_validation_changed_only.csv`: non-no-op synthetic cases used for distribution comparison",
        "- `real_vs_synthetic_quantiles.csv`: final real/synthetic quantile comparison (HD99 included as audit)",
        "- `example_bank/`: visual NIfTI bank",
        "- `figures/`: distribution and objective plots",
    ]
    (outdir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    cfg_path = resolve_path(args.config, cwd)
    cfg = yaml.safe_load(cfg_path.read_text())
    outdir = Path(cfg["output"]["directory"]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    initial_cfg, initial_path, initial_source = load_initial_training_config(cfg, cwd)
    assert_original_mixture(initial_cfg, cfg)

    error_root = Path(cfg["error_field_root"]).expanduser().resolve()
    case_path = error_root / "case_error_field_summary.csv"
    if not case_path.is_file():
        raise FileNotFoundError(case_path)
    real_df = pd.read_csv(case_path)
    exclusions = {int(x) for x in cfg.get("calibration", {}).get("exclude_patient_ids", [])}
    if exclusions:
        real_df = real_df[~real_df["patient_id"].astype(int).isin(exclusions)].copy()
    if len(real_df) < int(cfg["objective"].get("minimum_real_cases", 12)):
        raise RuntimeError(f"Only {len(real_df)} real cases remain for calibration")

    target, target_df = robust_target(real_df, cfg["objective"])
    target_df.to_csv(outdir / "target_quantiles.csv", index=False)

    labels = build_patient_index(Path(cfg["data"]["train_label_dir"]))
    cts = build_patient_index(Path(cfg["data"]["train_ct_dir"]))
    available = sorted(set(labels) & set(cts))
    if exclusions:
        available = [pid for pid in available if pid not in exclusions]
    if not available:
        raise RuntimeError("No training patients available after exclusions")
    label_paths = {int(pid): str(labels[pid]) for pid in available}

    fit_tasks = make_common_tasks(
        available,
        initial_cfg,
        int(cfg["search"].get("samples_per_candidate", 160)),
        int(cfg.get("seed", 91)) + 1001,
    )
    requested_workers = args.workers if args.workers is not None else int(cfg.get("parallel", {}).get("workers", 0))
    slurm_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = requested_workers if requested_workers and requested_workers > 0 else slurm_workers
    workers = max(1, min(int(workers), int(cfg.get("parallel", {}).get("candidate_workers", 4))))

    stage1 = stage1_candidates(cfg["search"], int(cfg.get("seed", 91)) + 2003)
    print(f"Evaluating {len(stage1)} stage-1 candidates with {workers} candidate workers")
    stage1_df = evaluate_candidates(
        stage1,
        initial_cfg,
        cfg["parameter_limits"],
        fit_tasks,
        label_paths,
        target,
        cfg["objective"],
        workers,
    )
    if not len(stage1_df) or not np.isfinite(pd.to_numeric(stage1_df["objective"], errors="coerce")).any():
        raise RuntimeError("No finite stage-1 candidate objective")
    best_stage1 = stage1_df.iloc[0]

    stage2 = refinement_candidates(best_stage1, cfg["search"], int(cfg.get("seed", 91)) + 3001)
    print(f"Evaluating {len(stage2)} stage-2 candidates")
    stage2_df = evaluate_candidates(
        stage2,
        initial_cfg,
        cfg["parameter_limits"],
        fit_tasks,
        label_paths,
        target,
        cfg["objective"],
        workers,
    )
    candidate_df = pd.concat([stage1_df.assign(search_stage="stage1"), stage2_df.assign(search_stage="stage2")], ignore_index=True)
    candidate_df = candidate_df.sort_values("objective", kind="stable").reset_index(drop=True)
    candidate_df.to_csv(outdir / "candidate_search.csv", index=False)
    plot_candidate_objectives(candidate_df, outdir)

    best = candidate_df.iloc[0]
    multipliers = {k: float(best[f"scale_{k}"]) for k in ("smooth", "noise", "dropout", "global")}
    best_cfg = config_with_multipliers(initial_cfg, multipliers, cfg["parameter_limits"])
    best_cfg["experiment_name"] = str(cfg["output"].get("recommended_experiment_name", "dino_shape_stochastic_region_dropout_Lung_L_metric_fit"))
    best_cfg.setdefault("study_metadata", {})["synthetic_severity_fit_objective"] = {
        "quantiles": [float(q) for q in cfg["objective"]["quantiles"]],
        "quantile_weights": cfg["objective"]["quantile_weights"],
        "metric_weights": cfg["objective"]["metric_weights"],
        "objective": float(best["objective"]),
        "hd99_role": "audit_only_zero_fit_weight",
    }
    (outdir / "best_fit_training_config.yaml").write_text(yaml.safe_dump(best_cfg, sort_keys=False))
    (outdir / "best_fit_multipliers.json").write_text(json.dumps({
        "objective": float(best["objective"]),
        "multipliers": multipliers,
        "initial_config": str(initial_path),
        "initial_config_source": initial_source,
    }, indent=2))

    print("Running independent exact-sampler validation for the best fit")
    validation_workers = max(1, min(slurm_workers, int(cfg.get("parallel", {}).get("validation_workers", 8))))
    validation_df, validation_fail = run_validation_audit(
        best_cfg,
        available,
        labels,
        cfg["objective"],
        cfg["validation_audit"],
        validation_workers,
        int(cfg.get("seed", 91)) + 900001,
    )
    validation_df.to_csv(outdir / "best_fit_validation_audit.csv", index=False)
    validation_fail.to_csv(outdir / "best_fit_validation_failures.csv", index=False)
    changed = validation_df[validation_df["is_noop"].astype(int) == 0].copy() if len(validation_df) else pd.DataFrame()
    changed.to_csv(outdir / "best_fit_validation_changed_only.csv", index=False)

    qcmp = quantile_comparison(real_df, changed)
    qcmp.to_csv(outdir / "real_vs_synthetic_quantiles.csv", index=False)
    write_plots(real_df, changed, outdir)

    # Export exact forced-branch examples at fitted range min/mid/max.
    examples = select_example_patients(real_df, available, cfg)
    fitted_sev = fitted_severity_strata(best_cfg)
    (outdir / "fitted_severity_strata.json").write_text(json.dumps(fitted_sev, indent=2))
    bank_root = outdir / "example_bank"
    manifest: List[Dict] = []
    for si, stratum in enumerate([str(x) for x in cfg["example_bank"].get("severity_strata", ["low", "typical", "high"])]):
        for pi, pid in enumerate(examples):
            rows = export_example_case(
                int(pid),
                labels[int(pid)],
                cts[int(pid)],
                bank_root / f"Lung_L_{stratum}" / f"patient_{pid}",
                best_cfg,
                fitted_sev,
                stratum,
                int(cfg.get("seed", 91)) + si * 10000019 + pi * 100003,
                float(cfg["example_bank"].get("crop_margin_mm", 20.0)),
            )
            manifest.extend(rows)
            print(f"  exported fitted {stratum} patient {pid}")
    pd.DataFrame(manifest).to_csv(outdir / "example_bank_manifest.csv", index=False)

    write_report(
        outdir,
        cfg,
        initial_path,
        initial_source,
        best,
        best_cfg,
        target_df,
        candidate_df,
        validation_df,
        validation_fail,
        examples,
    )
    print(f"Wrote synthetic severity fit to: {outdir}")
    print(f"Best-fit training config: {outdir / 'best_fit_training_config.yaml'}")
    print(f"Inspection bank: {bank_root}")


if __name__ == "__main__":
    main()
