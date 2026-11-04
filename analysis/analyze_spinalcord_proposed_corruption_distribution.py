#!/usr/bin/env python3
"""Pre-training validation of a proposed SpinalCord corruption distribution.

The analysis compares three distributions without training a neural network:
  1. real Month1+Month2 clinical autoseg errors;
  2. the already-sampled current NSR-QA corruption bank; and
  3. a newly sampled candidate corruption bank.

Month1+Month2 are the only design cohort. Month3 is reported separately as a
read-only audit and is never used to choose corruption parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.spinalcord_corruption_support import (
    SPARSE_ENDPOINT_FEATURES,
    fit_mixed_scaler,
    group_support_summary,
    mixed_distance_matrices,
    mixed_support_coverage,
)
from analysis.spinalcord_error_geometry import extract_pair_features
from training.corruptions.spinalcord_proposed_distribution import proposed_spinalcord_corruption

CLEAN_BRANCHES = {"clean", "rejected_to_clean"}


def load_yaml(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    text = "|".join([str(base), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % (2**31 - 1)


def extract_pid(path: Path) -> int | None:
    m = re.search(r"patient[_]?(\d+)", path.name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_No rows._"
    x = df.head(max_rows).copy()
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in x.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("NA" if not np.isfinite(v) else f"{float(v):.3f}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def nonclean(df: pd.DataFrame) -> pd.DataFrame:
    x = df[~df.branch.fillna("").isin(CLEAN_BRANCHES)].copy()
    if "changed_fraction_gt" in x:
        x = x[pd.to_numeric(x.changed_fraction_gt, errors="coerce").fillna(0) > 0]
    return x


def label_file_index(nsrqa_cfg: Mapping[str, object]) -> Dict[int, Tuple[str, Path]]:
    data = nsrqa_cfg["data"]
    out: Dict[int, Tuple[str, Path]] = {}
    for split, key in (("training1", "train_label_dir"), ("validation1", "validation_label_dir")):
        root = Path(str(data[key])).expanduser()
        for p in sorted(root.glob("*.nii.gz")):
            pid = extract_pid(p)
            if pid is not None:
                if pid in out:
                    raise RuntimeError(f"Duplicate development label patient {pid}: {out[pid][1]} and {p}")
                out[pid] = (split, p)
    return out


def orientation_audit(img: sitk.Image, pid: int, split: str, path: Path) -> Dict[str, object]:
    size = img.GetSize()
    p0 = img.TransformIndexToPhysicalPoint((0, 0, 0))
    p1 = img.TransformIndexToPhysicalPoint((0, 0, max(int(size[2]) - 1, 0)))
    dz = float(p1[2] - p0[2])
    return {
        "patient_id": int(pid),
        "source_split": split,
        "label_path": str(path),
        "physical_z_first_slice_mm": float(p0[2]),
        "physical_z_last_slice_mm": float(p1[2]),
        "physical_z_delta_mm": dz,
        "inferior_is_low_array_z": bool(dz > 0),
        "direction": json.dumps([float(x) for x in img.GetDirection()]),
        "spacing_xyz": json.dumps([float(x) for x in img.GetSpacing()]),
    }


def generate_patient_shard(
    pid: int,
    split: str,
    label_path: str,
    n_samples: int,
    cfg: Mapping[str, object],
    cfg_hash: str,
    base_seed: int,
    shard_path: str,
    force: bool,
) -> Tuple[str, Dict[str, object]]:
    shard = Path(shard_path)
    if shard.is_file() and not force:
        try:
            old = pd.read_csv(shard)
            if len(old) == n_samples and set(old.get("sampler_config_sha256", [])) == {cfg_hash}:
                img = sitk.ReadImage(label_path)
                return str(shard), orientation_audit(img, pid, split, Path(label_path))
        except Exception:
            pass

    img = sitk.ReadImage(label_path)
    approved = sitk.GetArrayFromImage(img) > 0
    if not approved.any():
        raise ValueError(f"Empty development SpinalCord mask: {label_path}")
    spacing_xyz = tuple(float(x) for x in img.GetSpacing())
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    orient = orientation_audit(img, pid, split, Path(label_path))
    inferior_is_low_z = bool(orient["inferior_is_low_array_z"])

    rows: List[Dict[str, object]] = []
    for sample_idx in range(int(n_samples)):
        rng = random.Random(stable_seed(base_seed, "proposed_bank", pid, split, sample_idx))
        corrupted, meta = proposed_spinalcord_corruption(
            approved,
            spacing_zyx,
            cfg,
            rng,
            inferior_is_low_z=inferior_is_low_z,
            cache_key=f"proposed:{pid}",
        )
        feat, _ = extract_pair_features(
            approved,
            corrupted,
            spacing_zyx,
            patient_id=pid,
            cohort="proposed_synthetic",
            include_components=False,
        )
        row: Dict[str, object] = {
            "patient_id": int(pid),
            "source_split": split,
            "sample_index": int(sample_idx),
            "branch": str(meta.get("branch", "")),
            "requested_branch": str(meta.get("requested_branch", meta.get("branch", ""))),
            "sampler_version": str(cfg.get("sampler_version", "")),
            "sampler_config_sha256": cfg_hash,
            **feat,
        }
        for key, value in meta.items():
            if key in {"branch", "requested_branch"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"generator_{key}"] = value
            else:
                row[f"generator_{key}"] = json.dumps(value, default=str)
        rows.append(row)
    shard.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(shard, index=False)
    return str(shard), orient


def generate_proposed_bank(
    real_dev: pd.DataFrame,
    current_bank: pd.DataFrame,
    nsrqa_cfg: Mapping[str, object],
    cfg: Mapping[str, object],
    out_dir: Path,
    workers: int,
    samples_per_case: int,
    force: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    index = label_file_index(nsrqa_cfg)
    current_counts = current_bank.groupby(current_bank.patient_id.astype(int)).size().to_dict()
    dev_ids = sorted(set(real_dev.patient_id.astype(int)))
    missing = [pid for pid in dev_ids if pid not in index]
    if missing:
        raise FileNotFoundError(f"Prepared development labels missing for real cases: {missing}")
    cfg_hash = sha256_json(cfg)
    shard_dir = out_dir / "proposed_bank_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for pid in dev_ids:
        split, path = index[pid]
        n = int(samples_per_case if samples_per_case > 0 else current_counts.get(pid, 0))
        if n <= 0:
            raise RuntimeError(f"No current-bank sample count for patient {pid}; use --samples-per-case")
        jobs.append((pid, split, path, n, shard_dir / f"patient_{pid}.csv"))

    print(f"Generating proposed bank for {len(jobs)} development patients with {workers} worker(s)", flush=True)
    orientation_rows: List[Dict[str, object]] = []
    paths: List[Path] = []
    if workers <= 1:
        for i, (pid, split, path, n, shard) in enumerate(jobs, 1):
            print(f"[proposed-bank] {i}/{len(jobs)} patient {pid}: {n} samples", flush=True)
            sp, orient = generate_patient_shard(pid, split, str(path), n, cfg, cfg_hash, int(cfg.get("seed", 88)), str(shard), force)
            paths.append(Path(sp)); orientation_rows.append(orient)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_map = {
                ex.submit(generate_patient_shard, pid, split, str(path), n, cfg, cfg_hash, int(cfg.get("seed", 88)), str(shard), force): pid
                for pid, split, path, n, shard in jobs
            }
            done = 0
            for fut in as_completed(future_map):
                pid = future_map[fut]
                sp, orient = fut.result()
                done += 1
                print(f"[proposed-bank] completed {done}/{len(jobs)} patient {pid}", flush=True)
                paths.append(Path(sp)); orientation_rows.append(orient)
    bank = pd.concat([pd.read_csv(p) for p in sorted(paths)], ignore_index=True)
    return bank, pd.DataFrame(orientation_rows).sort_values("patient_id")


def coverage_comparison(
    real: pd.DataFrame,
    current: pd.DataFrame,
    proposed: pd.DataFrame,
    scaler: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
    percentile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float]]:
    cases: Dict[str, pd.DataFrame] = {}
    refs: Dict[str, pd.DataFrame] = {}
    thresholds: Dict[str, float] = {}
    rows: List[Dict[str, object]] = []
    for name, bank in (("current", current), ("proposed", proposed)):
        c, ref, thr = mixed_support_coverage(real, bank, scaler, groups, percentile=percentile, exclude_clean=True)
        cases[name], refs[name], thresholds[name] = c, ref, thr
        for cohort, g in c.groupby("cohort"):
            covered = g.covered_by_current_training_support.astype(bool)
            rows.append({
                "bank": name,
                "criterion": "self_p95_radius",
                "cohort": cohort,
                "n": len(g),
                "coverage_fraction": float(covered.mean()),
                "median_nearest_distance": float(g.nearest_pooled_training_distance.median()),
                "p90_nearest_distance": float(g.nearest_pooled_training_distance.quantile(.90)),
                "support_radius": float(thr),
            })
    current_thr = thresholds["current"]
    for name in ("current", "proposed"):
        c = cases[name]
        for cohort, g in c.groupby("cohort"):
            rows.append({
                "bank": name,
                "criterion": "fixed_current_p95_radius",
                "cohort": cohort,
                "n": len(g),
                "coverage_fraction": float((g.nearest_pooled_training_distance <= current_thr).mean()),
                "median_nearest_distance": float(g.nearest_pooled_training_distance.median()),
                "p90_nearest_distance": float(g.nearest_pooled_training_distance.quantile(.90)),
                "support_radius": float(current_thr),
            })
    merged = cases["current"][["patient_id", "cohort", "nearest_pooled_training_distance", "nearest_same_patient_training_distance"]].rename(columns={
        "nearest_pooled_training_distance": "current_pooled_distance",
        "nearest_same_patient_training_distance": "current_same_patient_distance",
    }).merge(
        cases["proposed"][["patient_id", "cohort", "nearest_pooled_training_distance", "nearest_same_patient_training_distance", "nearest_training_branch", "nearest_same_patient_training_branch"]].rename(columns={
            "nearest_pooled_training_distance": "proposed_pooled_distance",
            "nearest_same_patient_training_distance": "proposed_same_patient_distance",
            "nearest_training_branch": "proposed_nearest_branch",
            "nearest_same_patient_training_branch": "proposed_same_patient_nearest_branch",
        }), on=["patient_id", "cohort"], how="inner"
    )
    merged["pooled_distance_improvement"] = merged.current_pooled_distance - merged.proposed_pooled_distance
    merged["same_patient_distance_improvement"] = merged.current_same_patient_distance - merged.proposed_same_patient_distance
    return pd.DataFrame(rows), merged, cases, thresholds


def group_comparison(real, current, proposed, scaler, groups, percentile) -> pd.DataFrame:
    frames = []
    for name, bank in (("current", current), ("proposed", proposed)):
        x = group_support_summary(real, bank, scaler, groups, percentile=percentile, exclude_clean=True)
        x.insert(0, "bank", name)
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    # Fair fixed-current-radius development comparison per group.
    real_records = real.to_dict("records")
    cur = nonclean(current).to_dict("records")
    prop = nonclean(proposed).to_dict("records")
    fixed_rows = []
    for group, names in groups.items():
        one = {group: list(names)}
        dcc, _ = mixed_distance_matrices(cur, cur, scaler, one)
        np.fill_diagonal(dcc, np.inf)
        dcc[dcc <= 1e-8] = np.inf
        cur_nn = np.min(dcc, axis=1)
        cur_nn[~np.isfinite(cur_nn)] = 0
        thr = float(np.percentile(cur_nn, percentile))
        for name, bank_records in (("current", cur), ("proposed", prop)):
            drb, _ = mixed_distance_matrices(real_records, bank_records, scaler, one)
            nn = np.nanmin(drb, axis=1)
            for cohort in sorted(real.cohort.unique()):
                idx = real.cohort.to_numpy() == cohort
                vals = nn[idx]
                fixed_rows.append({
                    "bank": name,
                    "cohort": cohort,
                    "group": group,
                    "criterion": "fixed_current_p95_radius",
                    "current_group_radius": thr,
                    "coverage_fraction": float(np.mean(vals <= thr)),
                    "median_real_nn_distance": float(np.median(vals)),
                })
    fixed = pd.DataFrame(fixed_rows)
    out["criterion"] = "self_p95_radius"
    return pd.concat([out, fixed], ignore_index=True, sort=False)


def endpoint_tables(real_dev: pd.DataFrame, current: pd.DataFrame, proposed: pd.DataFrame) -> pd.DataFrame:
    thresholds = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
    rows = []
    datasets = [
        ("real_development", real_dev),
        ("current_all", current),
        ("current_nonclean", nonclean(current)),
        ("proposed_all", proposed),
        ("proposed_nonclean", nonclean(proposed)),
    ]
    for name, df in datasets:
        for feature in SPARSE_ENDPOINT_FEATURES:
            vals = pd.to_numeric(df[feature], errors="coerce").dropna().to_numpy(float)
            for t in thresholds:
                effective = 1e-8 if t <= 0 else t
                rows.append({
                    "source": name,
                    "endpoint_feature": feature,
                    "threshold_label": "any" if t <= 0 else f">{t:g}mm",
                    "threshold_mm": effective,
                    "n": len(vals),
                    "n_above": int(np.sum(vals > effective)),
                    "fraction_above": float(np.mean(vals > effective)) if len(vals) else np.nan,
                })
    # Branch-level proposed audit.
    for branch, df in proposed.groupby("branch"):
        for feature in SPARSE_ENDPOINT_FEATURES:
            vals = pd.to_numeric(df[feature], errors="coerce").dropna().to_numpy(float)
            for t in thresholds:
                effective = 1e-8 if t <= 0 else t
                rows.append({
                    "source": f"proposed_branch:{branch}",
                    "endpoint_feature": feature,
                    "threshold_label": "any" if t <= 0 else f">{t:g}mm",
                    "threshold_mm": effective,
                    "n": len(vals),
                    "n_above": int(np.sum(vals > effective)),
                    "fraction_above": float(np.mean(vals > effective)) if len(vals) else np.nan,
                })
    return pd.DataFrame(rows)


def feature_floor(name: str) -> float:
    low = name.lower()
    if "fraction" in low or "relative_volume" in low:
        return 0.05
    if low.endswith("_mm") or "extent_" in low:
        return 1.5
    return 1e-3


def marginal_comparison(real_dev, current, proposed, features) -> pd.DataFrame:
    rows = []
    syns = (("current", nonclean(current)), ("proposed", nonclean(proposed)))
    for feature in features:
        r = pd.to_numeric(real_dev[feature], errors="coerce").dropna().to_numpy(float)
        if not len(r):
            continue
        rq = np.percentile(r, [5, 25, 50, 75, 95])
        scale = max(float(rq[3] - rq[1]), feature_floor(feature))
        for name, df in syns:
            s = pd.to_numeric(df[feature], errors="coerce").dropna().to_numpy(float)
            if not len(s):
                continue
            sq = np.percentile(s, [5, 25, 50, 75, 95])
            rows.append({
                "feature": feature,
                "bank": name,
                "real_p05": rq[0], "real_p25": rq[1], "real_p50": rq[2], "real_p75": rq[3], "real_p95": rq[4],
                "synthetic_p05": sq[0], "synthetic_p25": sq[1], "synthetic_p50": sq[2], "synthetic_p75": sq[3], "synthetic_p95": sq[4],
                "wasserstein": float(wasserstein_distance(r, s)),
                "wasserstein_over_real_scale": float(wasserstein_distance(r, s) / scale),
                "fraction_real_outside_synthetic_p05_p95": float(np.mean((r < sq[0]) | (r > sq[4]))),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        pivot = out.pivot(index="feature", columns="bank", values="wasserstein_over_real_scale")
        if {"current", "proposed"}.issubset(pivot.columns):
            imp = (pivot["current"] - pivot["proposed"]).rename("proposed_improvement_in_normalized_wasserstein")
            out = out.merge(imp, left_on="feature", right_index=True, how="left")
    return out


def scaled_xy(df: pd.DataFrame, x: str, y: str, real_dev: pd.DataFrame) -> np.ndarray:
    rx = pd.to_numeric(real_dev[x], errors="coerce").dropna().to_numpy(float)
    ry = pd.to_numeric(real_dev[y], errors="coerce").dropna().to_numpy(float)
    mx, my = np.median(rx), np.median(ry)
    sx = max(float(np.percentile(rx, 75) - np.percentile(rx, 25)), feature_floor(x))
    sy = max(float(np.percentile(ry, 75) - np.percentile(ry, 25)), feature_floor(y))
    a = pd.to_numeric(df[x], errors="coerce").to_numpy(float)
    b = pd.to_numeric(df[y], errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    return np.column_stack([(a[ok] - mx) / sx, (b[ok] - my) / sy])


def energy_distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Deterministic cap keeps pairwise matrices small without changing real rows.
    if len(b) > 1500:
        idx = np.linspace(0, len(b) - 1, 1500).astype(int)
        b = b[idx]
    ab = cdist(a, b).mean()
    aa = cdist(a, a).mean()
    bb = cdist(b, b).mean()
    return float(max(2 * ab - aa - bb, 0.0))


def joint_comparison(real_dev, current, proposed, pairs) -> pd.DataFrame:
    rows = []
    for x, y in pairs:
        a = scaled_xy(real_dev, x, y, real_dev)
        for name, df in (("current", nonclean(current)), ("proposed", nonclean(proposed))):
            b = scaled_xy(df, x, y, real_dev)
            rows.append({
                "x_feature": x,
                "y_feature": y,
                "bank": name,
                "energy_distance_2d": energy_distance_2d(a, b),
                "n_real": len(a),
                "n_synthetic": len(b),
            })
    out = pd.DataFrame(rows)
    p = out.pivot(index=["x_feature", "y_feature"], columns="bank", values="energy_distance_2d")
    if {"current", "proposed"}.issubset(p.columns):
        imp = (p.current - p.proposed).rename("proposed_improvement_energy_distance")
        out = out.merge(imp, left_on=["x_feature", "y_feature"], right_index=True, how="left")
    return out


def transformed_embedding(df: pd.DataFrame, scaler, features) -> np.ndarray:
    cols = []
    n = len(df)
    for f in features:
        spec = scaler[f]
        vals = pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        if spec.get("kind") == "sparse_endpoint":
            thr = float(spec["presence_threshold"])
            scale = max(float(spec["positive_scale"]), 1e-6)
            present = (vals > thr).astype(float)
            mag = np.where(present > 0, np.minimum(vals / scale, 5.0), 0.0)
            cols.extend([present[:, None], mag[:, None]])
        else:
            med = float(spec["median"]); scale = max(float(spec["scale"]), 1e-6)
            cols.append(np.clip((vals - med) / scale, -5, 5)[:, None])
    return np.concatenate(cols, axis=1) if cols else np.zeros((n, 0))


def two_sample_classifier(real_dev, bank, scaler, features, repeats, folds, rf_repeats, seed, bank_name) -> pd.DataFrame:
    syn = nonclean(bank)
    common = sorted(set(real_dev.patient_id.astype(int)) & set(syn.patient_id.astype(int)))
    real_map = real_dev.set_index(real_dev.patient_id.astype(int))
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(repeats):
        syn_rows = []
        real_rows = []
        groups = []
        for pid in common:
            g = syn[syn.patient_id.astype(int) == pid]
            if g.empty:
                continue
            syn_rows.append(g.iloc[int(rng.integers(0, len(g)))])
            real_rows.append(real_map.loc[pid])
            groups.extend([pid, pid])
        s = pd.DataFrame(syn_rows)
        r = pd.DataFrame(real_rows)
        X = np.vstack([transformed_embedding(r, scaler, features), transformed_embedding(s, scaler, features)])
        y = np.concatenate([np.ones(len(r), dtype=int), np.zeros(len(s), dtype=int)])
        groups_arr = np.asarray([int(x) for x in list(r.patient_id) + list(s.patient_id)])
        n_splits = min(folds, len(np.unique(groups_arr)))
        splitter = GroupKFold(n_splits=n_splits)
        for model_name in (["logistic"] + (["random_forest"] if rep < rf_repeats else [])):
            preds = np.full(len(y), np.nan)
            for tr, te in splitter.split(X, y, groups_arr):
                if model_name == "logistic":
                    model = Pipeline([
                        ("scale", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed + rep)),
                    ])
                else:
                    model = RandomForestClassifier(
                        n_estimators=250, max_depth=4, min_samples_leaf=3,
                        class_weight="balanced", random_state=seed + rep, n_jobs=1,
                    )
                model.fit(X[tr], y[tr])
                preds[te] = model.predict_proba(X[te])[:, 1]
            rows.append({
                "bank": bank_name,
                "model": model_name,
                "repeat": rep,
                "n_patients": len(common),
                "auc_real_vs_synthetic": float(roc_auc_score(y, preds)),
            })
    return pd.DataFrame(rows)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return np.nan
    return float(v[np.searchsorted(c, q * c[-1], side="left")])


def mixture_search(real_dev, current, draws, seed) -> pd.DataFrame:
    syn = nonclean(current).copy()
    branches = sorted(syn.branch.unique())
    if len(branches) < 2:
        return pd.DataFrame()
    features = [
        "missing_fraction_gt", "excess_fraction_gt", "hd95_mm", "assd_mm",
        "signed_surface_p05_mm", "signed_surface_p95_mm", "inferior_missing_mm",
        "largest_missing_component_fraction",
    ]
    qs = [0.25, 0.50, 0.75, 0.95]
    real_targets = {}
    scales = {}
    for f in features:
        v = pd.to_numeric(real_dev[f], errors="coerce").dropna().to_numpy(float)
        real_targets[f] = np.quantile(v, qs)
        scales[f] = max(float(np.quantile(v, .75) - np.quantile(v, .25)), feature_floor(f))
    row_branch = syn.branch.astype(str).to_numpy()
    vals = {f: pd.to_numeric(syn[f], errors="coerce").fillna(0).to_numpy(float) for f in features}
    n_by_branch = {b: int(np.sum(row_branch == b)) for b in branches}
    real_ep = {t: float(np.mean(real_dev.inferior_missing_mm.to_numpy(float) > t)) for t in (3., 20., 30.)}
    rng = np.random.default_rng(seed)
    candidates = []
    observed = np.asarray([n_by_branch[b] for b in branches], float); observed /= observed.sum()
    candidates.append(("observed_current", observed))
    for i in range(draws):
        candidates.append((f"dirichlet_{i}", rng.dirichlet(np.ones(len(branches)))))
    rows = []
    for name, mix in candidates:
        weights = np.zeros(len(syn), dtype=float)
        for b, w in zip(branches, mix):
            idx = row_branch == b
            weights[idx] = float(w) / max(int(idx.sum()), 1)
        terms = []
        for f in features:
            sq = np.asarray([weighted_quantile(vals[f], weights, q) for q in qs])
            terms.extend(np.minimum(np.abs(sq - real_targets[f]) / scales[f], 2.0))
        for t in (3., 20., 30.):
            p = float(weights[vals["inferior_missing_mm"] > t].sum() / max(weights.sum(), 1e-12))
            terms.append(min(abs(p - real_ep[t]) / max(real_ep[t], 0.05), 2.0))
        row = {"candidate": name, "objective": float(np.mean(terms))}
        row.update({f"weight_{b}": float(w) for b, w in zip(branches, mix)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)



def decision_summary(coverage, patient_match, group_cmp, endpoint, marginal, classifier_summary):
    rows = []
    def add(name, current, proposed, better, note):
        if np.isfinite(current) and np.isfinite(proposed):
            if better == "higher":
                status = "improved" if proposed > current else ("unchanged" if np.isclose(proposed, current) else "worse")
            else:
                status = "improved" if proposed < current else ("unchanged" if np.isclose(proposed, current) else "worse")
        else:
            status = "unavailable"
        rows.append({"criterion": name, "current": current, "proposed": proposed, "better_direction": better, "status": status, "note": note})

    dev = coverage[(coverage.cohort == "development") & (coverage.criterion == "fixed_current_p95_radius")].set_index("bank")
    if {"current", "proposed"}.issubset(dev.index):
        add("fixed_radius_overall_coverage", float(dev.loc["current", "coverage_fraction"]), float(dev.loc["proposed", "coverage_fraction"]), "higher", "Primary pooled support comparison using the current-bank P95 radius.")
    pm = patient_match[patient_match.cohort == "development"]
    add("median_same_patient_distance", float(pm.current_same_patient_distance.median()), float(pm.proposed_same_patient_distance.median()), "lower", "Primary anatomy-matched nearest-neighbor distance.")

    fixed = group_cmp[(group_cmp.cohort == "development") & (group_cmp.criterion == "fixed_current_p95_radius")]
    for group in ("surface", "localization", "longitudinal", "global_size", "displacement"):
        g = fixed[fixed.group == group].set_index("bank")
        if {"current", "proposed"}.issubset(g.index):
            add(f"fixed_radius_{group}_coverage", float(g.loc["current", "coverage_fraction"]), float(g.loc["proposed", "coverage_fraction"]), "higher", f"Development {group} support at current-bank radius.")

    ep = endpoint[(endpoint.endpoint_feature == "inferior_missing_mm") & (endpoint.threshold_label == ">20mm")].set_index("source")
    if {"real_development", "current_all", "proposed_all"}.issubset(ep.index):
        target = float(ep.loc["real_development", "fraction_above"]); cur = float(ep.loc["current_all", "fraction_above"]); prop = float(ep.loc["proposed_all", "fraction_above"])
        add("inferior_missing_gt20mm_absolute_frequency_error", abs(cur-target), abs(prop-target), "lower", f"Absolute difference from real development prevalence ({target:.3f}).")

    for feature in ("missing_fraction_gt", "hd95_mm"):
        m = marginal[marginal.feature == feature].set_index("bank")
        if {"current", "proposed"}.issubset(m.index):
            add(f"normalized_wasserstein_{feature}", float(m.loc["current", "wasserstein_over_real_scale"]), float(m.loc["proposed", "wasserstein_over_real_scale"]), "lower", "Marginal distribution distance to real development.")

    lg = classifier_summary[classifier_summary.model == "logistic"].set_index("bank")
    if {"current", "proposed"}.issubset(lg.index):
        add("real_vs_synthetic_logistic_auc_distance_from_0p5", abs(float(lg.loc["current", "mean"])-0.5), abs(float(lg.loc["proposed", "mean"])-0.5), "lower", "Lower means real and synthetic phenotypes are harder to distinguish.")
    return pd.DataFrame(rows)

def branch_summary(bank: pd.DataFrame, name: str) -> pd.DataFrame:
    x = bank.groupby("branch").size().rename("n").reset_index()
    x["fraction_all"] = x.n / x.n.sum()
    nc = nonclean(bank)
    denom = max(len(nc), 1)
    x["fraction_nonclean"] = x.branch.map(nc.groupby("branch").size() / denom).fillna(0.0)
    x.insert(0, "bank", name)
    return x


def proposed_composition_summary(bank: pd.DataFrame) -> pd.DataFrame:
    """Summarize realized compositional-sampler frequencies from metadata."""
    rows: List[Dict[str, object]] = []
    n_all = max(len(bank), 1)
    nc = nonclean(bank)
    n_nc = max(len(nc), 1)

    def add(component: str, count: int, note: str = "") -> None:
        rows.append({
            "component": component,
            "n": int(count),
            "fraction_all": float(count / n_all),
            "fraction_nonclean": float(count / n_nc),
            "note": note,
        })

    add("clean_or_rejected", int(len(bank) - len(nc)), "Includes requested clean and any no-change/rejected fallbacks.")

    surface_col = "generator_surface_mode"
    if surface_col in bank.columns:
        for mode, count in nc[surface_col].fillna("unknown").value_counts().items():
            add(f"surface:{mode}", int(count), "Surface modes are mutually exclusive within non-clean compositional samples.")

    endpoint_col = "generator_endpoint_applied"
    if endpoint_col in bank.columns:
        applied = pd.to_numeric(bank[endpoint_col], errors="coerce").fillna(0).astype(int) > 0
        add("endpoint:any", int(applied.sum()), "Independent modifier layered on top of the surface mode.")
        if "generator_endpoint_side" in bank.columns:
            for side in ("inferior", "superior"):
                side_mask = applied & bank["generator_endpoint_side"].fillna("").eq(side)
                add(f"endpoint:{side}", int(side_mask.sum()))
                if "generator_endpoint_severity_mode" in bank.columns:
                    for severity in ("mild", "severe"):
                        sev = side_mask & bank["generator_endpoint_severity_mode"].fillna("").eq(severity)
                        if int(sev.sum()):
                            add(f"endpoint:{side}:{severity}", int(sev.sum()))

    bite_col = "generator_local_bite_applied"
    if bite_col in bank.columns:
        bite = pd.to_numeric(bank[bite_col], errors="coerce").fillna(0).astype(int) > 0
        add("local_bite:any", int(bite.sum()), "Independent modifier applied after endpoint truncation.")

    comp_col = "generator_composition_label"
    if comp_col in bank.columns:
        for label, count in bank[comp_col].fillna("unknown").value_counts().head(20).items():
            add(f"composition:{label}", int(count), "Top realized composition labels.")
    return pd.DataFrame(rows)



def length_conditioned_endpoint_audit(
    real_dev: pd.DataFrame, proposed: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit v3 severe-inferior frequency versus approved S/I length.

    Quartile edges are derived only from Month1+Month2 real development cases.
    Proposed rows are then assigned to those frozen bins.
    """
    lengths = pd.to_numeric(real_dev["gt_length_si_mm"], errors="coerce").dropna().to_numpy(float)
    if not len(lengths):
        return pd.DataFrame(), pd.DataFrame()
    edges = np.quantile(lengths, [0.0, 0.25, 0.50, 0.75, 1.0]).astype(float)
    # Guard against tied quartiles without changing the clinically meaningful scale.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    labels = [
        f"Q{i+1}[{edges[i]:.1f},{edges[i+1]:.1f}]mm"
        for i in range(len(edges) - 1)
    ]

    def summarize(source: str, df: pd.DataFrame) -> List[Dict[str, object]]:
        x = df.copy()
        x["_length"] = pd.to_numeric(x["gt_length_si_mm"], errors="coerce")
        x["_inferior"] = pd.to_numeric(x["inferior_missing_mm"], errors="coerce")
        x = x[np.isfinite(x["_length"]) & np.isfinite(x["_inferior"])].copy()
        x["length_bin"] = pd.cut(x["_length"], bins=edges, labels=labels, include_lowest=True)
        rows: List[Dict[str, object]] = []
        for label in labels:
            g = x[x["length_bin"].astype(str) == label]
            if g.empty:
                continue
            severe = g["_inferior"] > 20.0
            positive = g["_inferior"] > 1e-8
            row: Dict[str, object] = {
                "source": source,
                "length_bin": label,
                "n_rows": int(len(g)),
                "n_patients": int(g.patient_id.astype(int).nunique()) if "patient_id" in g.columns else np.nan,
                "median_gt_length_si_mm": float(g["_length"].median()),
                "inferior_any_fraction": float(positive.mean()),
                "inferior_gt20mm_fraction": float(severe.mean()),
                "n_severe": int(severe.sum()),
                "median_severe_inferior_missing_mm": float(g.loc[severe, "_inferior"].median()) if severe.any() else np.nan,
                "median_severe_fraction_of_gt_length": float((g.loc[severe, "_inferior"] / g.loc[severe, "_length"]).median()) if severe.any() else np.nan,
            }
            target_col = "generator_endpoint_severe_probability_unconditional_target"
            if target_col in g.columns:
                vals = pd.to_numeric(g[target_col], errors="coerce").dropna()
                row["mean_generator_severe_unconditional_target"] = float(vals.mean()) if len(vals) else np.nan
            rows.append(row)
        return rows

    audit = pd.DataFrame(
        summarize("real_development", real_dev)
        + summarize("proposed_all", proposed)
        + summarize("proposed_nonclean", nonclean(proposed))
    )

    fraction_rows: List[Dict[str, object]] = []
    for source, df in (
        ("real_development", real_dev),
        ("proposed_all", proposed),
        ("proposed_nonclean", nonclean(proposed)),
    ):
        gt = pd.to_numeric(df["gt_length_si_mm"], errors="coerce").to_numpy(float)
        inf = pd.to_numeric(df["inferior_missing_mm"], errors="coerce").to_numpy(float)
        ok = np.isfinite(gt) & np.isfinite(inf) & (gt > 0) & (inf > 20.0)
        frac = inf[ok] / gt[ok]
        cuts = inf[ok]
        if not len(frac):
            continue
        qf = np.percentile(frac, [5, 25, 50, 75, 95])
        qc = np.percentile(cuts, [5, 25, 50, 75, 95])
        fraction_rows.append({
            "source": source,
            "n_severe": int(len(frac)),
            "cut_fraction_p05": float(qf[0]),
            "cut_fraction_p25": float(qf[1]),
            "cut_fraction_p50": float(qf[2]),
            "cut_fraction_p75": float(qf[3]),
            "cut_fraction_p95": float(qf[4]),
            "cut_mm_p05": float(qc[0]),
            "cut_mm_p25": float(qc[1]),
            "cut_mm_p50": float(qc[2]),
            "cut_mm_p75": float(qc[3]),
            "cut_mm_p95": float(qc[4]),
        })
    return audit, pd.DataFrame(fraction_rows)

def make_figures(out_dir, real_dev, current, proposed, coverage_cases, marg, joint):
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    # Nearest-distance paired comparison.
    dev = coverage_cases[coverage_cases.cohort == "development"]
    if len(dev):
        plt.figure(figsize=(6.5, 5.2))
        plt.scatter(dev.current_same_patient_distance, dev.proposed_same_patient_distance, alpha=.75)
        lim = max(float(np.nanmax(dev[["current_same_patient_distance", "proposed_same_patient_distance"]].to_numpy())), .1)
        plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
        plt.xlabel("Current same-patient nearest distance")
        plt.ylabel("Proposed same-patient nearest distance")
        plt.title("Same-patient real-error matching")
        plt.tight_layout(); p = out_dir / "same_patient_distance_current_vs_proposed.png"; plt.savefig(p, dpi=180); plt.close(); made.append(p.name)
    # Selected marginal boxplots.
    for feature, label in (("missing_fraction_gt", "Missing fraction"), ("hd95_mm", "HD95 (mm)"), ("inferior_missing_mm", "Inferior missing (mm)")):
        vals = [
            pd.to_numeric(real_dev[feature], errors="coerce").dropna().to_numpy(),
            pd.to_numeric(nonclean(current)[feature], errors="coerce").dropna().to_numpy(),
            pd.to_numeric(nonclean(proposed)[feature], errors="coerce").dropna().to_numpy(),
        ]
        plt.figure(figsize=(6.5, 4.8)); plt.boxplot(vals, labels=["Real dev", "Current", "Proposed"], showfliers=False)
        plt.ylabel(label); plt.title(f"Real vs synthetic: {label}"); plt.tight_layout()
        p = out_dir / f"real_current_proposed_{feature}.png"; plt.savefig(p, dpi=180); plt.close(); made.append(p.name)
    # Joint scatter plots.
    pairs = list(joint[["x_feature", "y_feature"]].drop_duplicates().itertuples(index=False, name=None))
    for x, y in pairs:
        plt.figure(figsize=(6.4, 5.2))
        plt.scatter(nonclean(current)[x], nonclean(current)[y], s=10, alpha=.18, label="Current")
        plt.scatter(nonclean(proposed)[x], nonclean(proposed)[y], s=10, alpha=.18, label="Proposed")
        plt.scatter(real_dev[x], real_dev[y], s=28, alpha=.8, label="Real dev")
        plt.xlabel(x); plt.ylabel(y); plt.legend(); plt.title(f"Joint distribution: {x} vs {y}"); plt.tight_layout()
        p = out_dir / f"joint_{x}_vs_{y}.png"; plt.savefig(p, dpi=180); plt.close(); made.append(p.name)
    return made


def report_text(out_dir, cfg, coverage, groups, endpoint, marginal, joint, classifier_summary, mixture, branches, composition, endpoint_length_audit, severe_fraction_audit, orientations, decision, figures):
    dev_cov = coverage[(coverage.cohort == "development") & (coverage.criterion == "fixed_current_p95_radius")]
    lines = [
        "# Proposed SpinalCord corruption-distribution validation",
        "",
        "## Role of this analysis",
        "",
        "This is a pre-training distribution audit. Month1+Month2 clinical errors are the design target. Month3 results, when shown, are a read-only temporal audit and must not be used to retune the sampler for the current temporal experiment.",
        "",
        "The proposed bank matches the current bank sample count per development patient by default, so nearest-neighbor comparisons are not trivially improved by a denser proposed bank.",
        "",
        "## Primary support comparison",
        "",
        markdown_table(coverage.sort_values(["criterion", "cohort", "bank"]), 30),
        "",
        "The `fixed_current_p95_radius` rows are the fairest direct comparison: both banks are judged against the support radius learned from the current synthetic bank. The self-radius rows are retained as an internal-support diagnostic.",
        "",
        "## Group-level support",
        "",
        markdown_table(groups.sort_values(["criterion", "cohort", "group", "bank"]), 60),
        "",
        "## Endpoint-frequency audit",
        "",
    ]
    ep = endpoint[(endpoint.endpoint_feature == "inferior_missing_mm") & endpoint.threshold_label.isin([">3mm", ">20mm", ">30mm"]) & endpoint.source.isin(["real_development", "current_all", "proposed_all", "current_nonclean", "proposed_nonclean"])]
    lines += [markdown_table(ep, 40), "", "Endpoint support and endpoint sampling frequency are deliberately reported separately. A generator can reach a severe endpoint error yet still sample it too rarely.", ""]
    lines += ["## Marginal-distribution comparison", "", markdown_table(marginal.sort_values(["feature", "bank"]), 80), ""]
    lines += ["## Joint-distribution comparison", "", markdown_table(joint.sort_values(["x_feature", "y_feature", "bank"]), 30), ""]
    lines += ["## Real-vs-synthetic classifier", "", markdown_table(classifier_summary, 20), "", "AUC closer to 0.5 means the measured real and synthetic feature distributions are harder to distinguish. This is a diagnostic, not an optimization target by itself.", ""]
    lines += ["## Pre-training decision summary", "", markdown_table(decision, 30), "", "The sampler should not be promoted to training based on one aggregate metric alone; surface/localization support and global-severity calibration must improve together.", "", "## Existing-branch mixture search", "", markdown_table(mixture.head(12), 12), "", "This search asks how well the *existing* branch families could match development errors by reweighting alone. It cannot substitute for genuinely new endpoint/transverse-surface families.", ""]
    lines += ["## Sampled branch frequencies", "", markdown_table(branches, 30), ""]
    if composition is not None and not composition.empty:
        lines += ["## Proposed composition frequencies", "", markdown_table(composition, 60), "", "For compositional samplers, surface mode, endpoint status, and local bite are separate axes; this table is more informative than a mutually-exclusive branch count.", ""]
    if endpoint_length_audit is not None and not endpoint_length_audit.empty:
        lines += ["## Length-conditioned endpoint audit", "", markdown_table(endpoint_length_audit, 40), "", "Length bins are frozen from Month1+Month2 development quartiles. For v3, compare the realized >20-mm inferior-loss frequency with the real frequency and with the generator's mean unconditional target probability in each bin.", ""]
    if severe_fraction_audit is not None and not severe_fraction_audit.empty:
        lines += ["## Severe inferior cut fraction audit", "", markdown_table(severe_fraction_audit, 20), "", "V3 samples severe inferior length as a fraction of approved S/I length, with an absolute 27-50 mm cap. This table checks both representations against development errors.", ""]
    lines += ["## Orientation audit", "", f"Patients audited: **{len(orientations)}**. Unique inferior-array conventions: **{orientations.inferior_is_low_array_z.value_counts().to_dict()}**.", "", "Endpoint truncation uses physical NIfTI direction rather than assuming that low array-z is inferior.", ""]
    lines += ["## Decision rule before training", "", "Prefer the proposed sampler only if Month1+Month2 improves in the same direction across: (1) fixed-radius overall/same-patient support, (2) the surface and localization groups, (3) endpoint frequency without excessive global severity, (4) marginal and joint distribution distances, and (5) real-vs-synthetic classifier AUC moving toward 0.5. Do not accept an apparent gain driven only by a larger self-calibrated support radius.", ""]
    if figures:
        lines += ["## Figures", ""]
        for f in figures:
            lines += [f"![{f}](figures/{f})", ""]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration-root", default="/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study/spinalcord_corruption_calibration")
    ap.add_argument("--study-root", default="/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study")
    ap.add_argument("--nsrqa-config", default=None)
    ap.add_argument("--proposal-config", default="configs/dino_shape/spinalcord_proposed_corruption_distribution.yaml")
    ap.add_argument("--analysis-config", default="configs/dino_shape/spinalcord_corruption_calibration.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--samples-per-case", type=int, default=0, help="0 = match current-bank count for each patient")
    ap.add_argument("--force-regenerate", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    def resolve(p):
        q = Path(p).expanduser(); return q if q.is_absolute() else project_root / q
    cal = Path(args.calibration_root).expanduser().resolve()
    study = Path(args.study_root).expanduser().resolve()
    proposal_path = resolve(args.proposal_config).resolve(); proposal_cfg = load_yaml(proposal_path)
    analysis_path = resolve(args.analysis_config).resolve(); analysis_cfg = load_yaml(analysis_path)
    nsrqa_path = Path(args.nsrqa_config).expanduser().resolve() if args.nsrqa_config else study / "configs" / "SpinalCord" / "nsrqa.yaml"
    nsrqa_cfg = load_yaml(nsrqa_path)
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else cal / "proposed_distribution_validation"
    out.mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)

    real = pd.read_csv(cal / "real" / "case_features.csv")
    current = pd.read_csv(cal / "training_coverage" / "synthetic_bank_features.csv")
    real_dev = real[real.cohort == "development"].copy()
    if real_dev.empty:
        raise RuntimeError("No development real-error rows found")

    proposed, orientations = generate_proposed_bank(real_dev, current, nsrqa_cfg, proposal_cfg, out, args.workers, args.samples_per_case, args.force_regenerate)
    proposed.to_csv(out / "proposed_synthetic_bank_features.csv", index=False)
    orientations.to_csv(out / "orientation_audit.csv", index=False)

    groups = {k: list(v) for k, v in analysis_cfg["phenotype_groups"].items()}
    features = [f for xs in groups.values() for f in xs]
    scaler = fit_mixed_scaler(real_dev, features)
    (out / "support_scaler.json").write_text(json.dumps(scaler, indent=2) + "\n")
    percentile = float(proposal_cfg.get("analysis", {}).get("coverage_percentile", 95.0))

    coverage, patient_match, cases, thresholds = coverage_comparison(real, current, proposed, scaler, groups, percentile)
    coverage.to_csv(out / "support_coverage_comparison.csv", index=False)
    patient_match.to_csv(out / "same_patient_matching_comparison.csv", index=False)
    for name, df in cases.items():
        df.to_csv(out / f"{name}_coverage_cases.csv", index=False)
    group_cmp = group_comparison(real, current, proposed, scaler, groups, percentile)
    group_cmp.to_csv(out / "group_support_comparison.csv", index=False)

    endpoint = endpoint_tables(real_dev, current, proposed)
    endpoint.to_csv(out / "endpoint_prevalence_comparison.csv", index=False)
    marginal = marginal_comparison(real_dev, current, proposed, features)
    marginal.to_csv(out / "marginal_distribution_comparison.csv", index=False)
    pairs = proposal_cfg.get("analysis", {}).get("joint_pairs", [])
    joint = joint_comparison(real_dev, current, proposed, pairs)
    joint.to_csv(out / "joint_distribution_comparison.csv", index=False)

    repeats = int(proposal_cfg.get("analysis", {}).get("classifier_repeats", 30))
    folds = int(proposal_cfg.get("analysis", {}).get("classifier_folds", 5))
    rf_repeats = int(proposal_cfg.get("analysis", {}).get("classifier_rf_repeats", 10))
    seed = int(proposal_cfg.get("seed", 88))
    clf = pd.concat([
        two_sample_classifier(real_dev, current, scaler, features, repeats, folds, rf_repeats, seed + 1, "current"),
        two_sample_classifier(real_dev, proposed, scaler, features, repeats, folds, rf_repeats, seed + 2, "proposed"),
    ], ignore_index=True)
    clf.to_csv(out / "two_sample_classifier_repeats.csv", index=False)
    clf_summary = clf.groupby(["bank", "model"]).auc_real_vs_synthetic.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    clf_summary["distance_from_random_auc"] = (clf_summary["mean"] - 0.5).abs()
    clf_summary.to_csv(out / "two_sample_classifier_summary.csv", index=False)

    draws = int(proposal_cfg.get("analysis", {}).get("mixture_search_draws", 2500))
    mixseed = int(proposal_cfg.get("analysis", {}).get("mixture_search_seed", 188))
    mixture = mixture_search(real_dev, current, draws, mixseed)
    mixture.to_csv(out / "existing_branch_mixture_search.csv", index=False)

    branches = pd.concat([branch_summary(current, "current"), branch_summary(proposed, "proposed")], ignore_index=True)
    branches.to_csv(out / "branch_frequency_comparison.csv", index=False)
    composition = proposed_composition_summary(proposed)
    composition.to_csv(out / "proposed_composition_frequency.csv", index=False)
    endpoint_length_audit, severe_fraction_audit = length_conditioned_endpoint_audit(real_dev, proposed)
    endpoint_length_audit.to_csv(out / "endpoint_length_conditioning_audit.csv", index=False)
    severe_fraction_audit.to_csv(out / "severe_inferior_fraction_comparison.csv", index=False)
    decision = decision_summary(coverage, patient_match, group_cmp, endpoint, marginal, clf_summary)
    decision.to_csv(out / "decision_summary.csv", index=False)
    (out / "resolved_run_config.json").write_text(json.dumps({
        "proposal_config": str(proposal_path), "proposal_config_sha256": sha256_json(proposal_cfg),
        "analysis_config": str(analysis_path), "nsrqa_config": str(nsrqa_path),
        "calibration_root": str(cal), "samples_per_case_override": args.samples_per_case,
        "current_support_radius": thresholds["current"], "proposed_support_radius": thresholds["proposed"],
    }, indent=2) + "\n")

    figures = make_figures(out / "figures", real_dev, current, proposed, patient_match, marginal, joint)
    report_text(out, proposal_cfg, coverage, group_cmp, endpoint, marginal, joint, clf_summary, mixture, branches, composition, endpoint_length_audit, severe_fraction_audit, orientations, decision, figures)

    print("\nProposed SpinalCord corruption distribution validation complete", flush=True)
    print(f"  output: {out}", flush=True)
    print(f"  report: {out / 'report.md'}", flush=True)
    print(f"  proposed bank rows: {len(proposed)}", flush=True)
    dev_fixed = coverage[(coverage.cohort == "development") & (coverage.criterion == "fixed_current_p95_radius")]
    for _, r in dev_fixed.iterrows():
        print(f"  development fixed-radius coverage {r.bank}: {r.coverage_fraction:.1%}", flush=True)
    for _, r in clf_summary[clf_summary.model == "logistic"].iterrows():
        print(f"  real-vs-{r.bank} logistic AUC: {r['mean']:.3f} ± {r['std']:.3f}", flush=True)


if __name__ == "__main__":
    main()
