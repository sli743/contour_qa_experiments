#!/usr/bin/env python3
"""Flag Lung_L approved/autoseg pairs for manual normality inspection.

Purpose
-------
Build a month-stratified review list of Lung_L cases where the approved and
clinical autoseg contours disagree substantially. The historical poor-agreement
criterion is retained as the primary flag:

    Dice < Dice lower control limit  AND  HD95 > HD95 upper control limit

A supplemental HD99 upper-control-limit flag is OR'ed into the inspection list
to capture localized bite/blob-like discrepancies that may involve too little
surface area to move HD95 strongly.

The original Radformation ``metrics.csv`` is preferred for Dice and the
historical ``haus`` (HD95) quantity so the primary flags remain comparable with
the earlier OOD/domain review. HD99 is recomputed directly from approved and
autoseg NIfTI masks. Dice/HD95/ASSD are also recomputed from those masks for
geometry/source auditing and as explicit fallbacks when clinical metrics are
unavailable.

Control limits
--------------
One fixed Lung_L threshold set is fit once on the configured reference months
(default: month1+month2+month3) and then applied unchanged to each month.
The default robust 3-sigma analogue is median +/- k * robust_scale, where
robust_scale is 1.4826*MAD with IQR/1.349 and SD fallbacks. Exact historical
limits can be supplied in the YAML to override any derived threshold.

This script does not classify an approved contour as normal/abnormal. It only
selects cases for manual review of that question.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from scipy import ndimage

from analysis.analyze_nsrqa_domain_performance import (
    MONTH1_PATIENT_IDS,
    MONTH2_PATIENT_IDS,
    MONTH3_PATIENT_IDS,
    load_clinical_metrics,
)


MONTHS = {
    "month1": sorted(MONTH1_PATIENT_IDS, key=lambda x: int(x)),
    "month2": sorted(MONTH2_PATIENT_IDS, key=lambda x: int(x)),
    "month3": sorted(MONTH3_PATIENT_IDS, key=lambda x: int(x)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def norm_pid(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.0+)?", s):
        return str(int(float(s)))
    return s


def resolve_mask(root: Path, pid: str, roi: str, kind: str) -> Path:
    directory = root / ("labels-approved" if kind == "approved" else "labels-autoseg")
    exact = directory / f"patient{pid}_{roi}_{kind}.nii.gz"
    if exact.is_file():
        return exact

    # Avoid substring collisions such as patient 53 matching patient153.
    patient_token = re.compile(rf"patient[_-]?{re.escape(str(pid))}(?!\d)", re.IGNORECASE)
    patterns = [
        f"*patient*{pid}*{roi}*{kind}*.nii.gz",
        f"*patient_{pid}*{roi}*{kind}*.nii.gz",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(directory.glob(pat))
    hits = sorted({p for p in hits if patient_token.search(p.name)})
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected one {kind} mask for patient {pid} {roi}; "
            f"exact={exact}; token-filtered hits={hits}"
        )
    return hits[0]


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def read_pair(data_root: Path, pid: str, roi: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], str, str]:
    ap = resolve_mask(data_root, pid, roi, "approved")
    up = resolve_mask(data_root, pid, roi, "autoseg")
    a_img = sitk.ReadImage(str(ap))
    u_img = sitk.ReadImage(str(up))
    if not same_geometry(a_img, u_img):
        u_img = sitk.Resample(
            u_img, a_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )
    a = sitk.GetArrayFromImage(a_img) > 0
    u = sitk.GetArrayFromImage(u_img) > 0
    spacing_zyx = tuple(float(x) for x in a_img.GetSpacing()[::-1])
    return a, u, spacing_zyx, str(ap), str(up)


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    er = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return mask & ~er


def crop_to_union(a: np.ndarray, u: np.ndarray, pad_vox: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    union = np.logical_or(a, u)
    pts = np.argwhere(union)
    if pts.size == 0:
        return a, u
    lo = np.maximum(pts.min(axis=0) - int(pad_vox), 0)
    hi = np.minimum(pts.max(axis=0) + int(pad_vox) + 1, np.asarray(a.shape))
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    return a[sl], u[sl]


def geometry_metrics(a: np.ndarray, u: np.ndarray, spacing_zyx: Sequence[float]) -> Dict[str, float]:
    # Surface distance transforms dominate runtime/memory. Cropping to the
    # union foreground bbox preserves all relevant surfaces and physical
    # distances while avoiding full-CT distance transforms.
    a, u = crop_to_union(a, u, pad_vox=2)
    a_n = int(a.sum())
    u_n = int(u.sum())
    inter = int(np.logical_and(a, u).sum())
    if a_n + u_n == 0:
        dice = 1.0
    elif a_n == 0 or u_n == 0:
        dice = 0.0
    else:
        dice = 2.0 * inter / float(a_n + u_n)

    sa = surface(a)
    su = surface(u)
    if not sa.any() and not su.any():
        d_all = np.array([0.0], dtype=float)
    elif not sa.any() or not su.any():
        d_all = np.array([np.nan], dtype=float)
    else:
        # distance_transform_edt returns distance to nearest zero. Inverting the
        # target surface makes surface voxels zero and therefore valid targets.
        dt_to_u = ndimage.distance_transform_edt(~su, sampling=spacing_zyx)
        dt_to_a = ndimage.distance_transform_edt(~sa, sampling=spacing_zyx)
        d_all = np.concatenate([dt_to_u[sa], dt_to_a[su]]).astype(float)

    finite = d_all[np.isfinite(d_all)]
    if finite.size:
        hd95 = float(np.percentile(finite, 95))
        hd99 = float(np.percentile(finite, 99))
        hdmax = float(np.max(finite))
        assd = float(np.mean(finite))
    else:
        hd95 = hd99 = hdmax = assd = np.nan

    missing = int(np.logical_and(a, ~u).sum())
    excess = int(np.logical_and(u, ~a).sum())
    denom = max(a_n, 1)
    return {
        "recomputed_dice": float(dice),
        "recomputed_one_minus_dice": float(1.0 - dice),
        "recomputed_hd95_mm": hd95,
        "hd99_mm": hd99,
        "hdmax_mm": hdmax,
        "recomputed_assd_mm": assd,
        "missing_fraction_gt": missing / float(denom),
        "excess_fraction_gt": excess / float(denom),
        "relative_volume_difference": (u_n - a_n) / float(denom),
        "approved_voxels": a_n,
        "autoseg_voxels": u_n,
    }


def metric_worker(task: Tuple[str, str, str]) -> Dict[str, object]:
    pid, roi, data_root = task
    try:
        a, u, spacing, ap, up = read_pair(Path(data_root), pid, roi)
        row: Dict[str, object] = {"patient_id": pid, "approved_path": ap, "autoseg_path": up}
        row.update(geometry_metrics(a, u, spacing))
        return {"ok": True, "row": row}
    except Exception as exc:
        return {"ok": False, "patient_id": pid, "error": repr(exc)}


def robust_center_scale(values: Iterable[float]) -> Tuple[float, float, str, int]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, "none", 0
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = 1.4826 * mad
    method = "1.4826*MAD"
    if not np.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.percentile(x, [25, 75])
        scale = float((q75 - q25) / 1.349)
        method = "IQR/1.349"
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        method = "SD"
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1e-12
        method = "epsilon"
    return center, scale, method, int(x.size)


def clinical_wide(path: str, site: str, roi: str, duplicate_policy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    case_metric, _exact, audit = load_clinical_metrics(path, site, duplicate_policy)
    d = case_metric[case_metric["roi"].astype(str) == roi].copy()
    d["patient_id"] = d["patient_id"].map(norm_pid)
    wanted = d[d["clinical_metric_name"].isin(["dice", "haus"])].copy()
    values = wanted.pivot_table(
        index="patient_id", columns="clinical_metric_name", values="clinical_metric_value", aggfunc="first"
    ).reset_index()
    values = values.rename(columns={"dice": "clinical_dice", "haus": "clinical_hd95_mm"})
    status = wanted.pivot_table(
        index="patient_id", columns="clinical_metric_name", values="clinical_metric_status", aggfunc="first"
    ).reset_index()
    status = status.rename(columns={"dice": "clinical_dice_status", "haus": "clinical_hd95_status"})
    return values.merge(status, on="patient_id", how="outer"), audit


def load_domains(path: Optional[str], site: str, roi: str) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        return pd.DataFrame(columns=["patient_id", "recorded_domain", "domain_conflict"])
    d = pd.read_csv(path)
    if "site" in d.columns:
        d = d[d["site"].astype(str).str.lower() == site.lower()].copy()
    d = d[d["roi"].astype(str).str.strip() == roi].copy()
    d["patient_id"] = d["patient_id"].map(norm_pid)
    d["domain"] = d["domain"].astype(str).str.strip().str.upper().replace({"GC": "CA"})
    rows = []
    for pid, g in d.groupby("patient_id", sort=True):
        vals = sorted({x for x in g["domain"] if x and x not in {"NAN", "NONE"}})
        rows.append({
            "patient_id": pid,
            "recorded_domain": vals[0] if len(vals) == 1 else ("CONFLICT" if vals else ""),
            "domain_conflict": len(vals) > 1,
        })
    return pd.DataFrame(rows)


def month_membership_rows() -> pd.DataFrame:
    rows = []
    counts: Dict[str, int] = {}
    for month, ids in MONTHS.items():
        for pid in ids:
            counts[pid] = counts.get(pid, 0) + 1
            rows.append({"patient_id": pid, "month": month})
    out = pd.DataFrame(rows)
    out["month_overlap_patient"] = out["patient_id"].map(counts).gt(1)
    return out


def fit_limits(df: pd.DataFrame, cfg: dict) -> Tuple[dict, pd.DataFrame]:
    c = cfg["control_limits"]
    months = set(str(x) for x in c.get("reference_months", ["month1", "month2", "month3"]))
    ref = df[df["month"].isin(months)].copy()
    # Patient IDs can appear in more than one declared month list. The same raw
    # mask pair must not receive extra weight when fitting the fixed limits.
    ref = ref.sort_values(["patient_id", "month"]).drop_duplicates("patient_id", keep="first")
    k = float(c.get("k", 3.0))

    specs = {
        "dice": ("dice_for_flagging", "lower"),
        "hd95_mm": ("hd95_for_flagging_mm", "upper"),
        "hd99_mm": ("hd99_mm", "upper"),
    }
    limits = {}
    audit_rows = []
    overrides = c.get("overrides", {}) or {}
    for name, (col, side) in specs.items():
        center, scale, method, n = robust_center_scale(pd.to_numeric(ref[col], errors="coerce"))
        derived = center - k * scale if side == "lower" else center + k * scale
        if name == "dice":
            derived = float(np.clip(derived, 0.0, 1.0))
        else:
            derived = max(0.0, float(derived))
        override = overrides.get(name)
        final = float(override) if override is not None else derived
        limits[name] = final
        audit_rows.append({
            "metric": name,
            "source_column": col,
            "side": side,
            "reference_months": "|".join(sorted(months)),
            "n": n,
            "center_median": center,
            "robust_scale": scale,
            "scale_method": method,
            "k": k,
            "derived_limit": derived,
            "override": override,
            "final_limit": final,
        })
    return limits, pd.DataFrame(audit_rows)


def apply_flags(df: pd.DataFrame, limits: dict, scale_audit: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dl = float(limits["dice"])
    h95 = float(limits["hd95_mm"])
    h99 = float(limits["hd99_mm"])
    out["dice_below_lcl"] = out["dice_for_flagging"] < dl
    out["hd95_above_ucl"] = out["hd95_for_flagging_mm"] > h95
    out["hd99_above_ucl"] = out["hd99_mm"] > h99
    out["primary_poor_agreement"] = out["dice_below_lcl"] & out["hd95_above_ucl"]
    out["hd99_local_extreme"] = out["hd99_above_ucl"]
    out["inspection_union"] = out["primary_poor_agreement"] | out["hd99_local_extreme"]
    out["inspection_reason"] = np.select(
        [
            out["primary_poor_agreement"] & out["hd99_local_extreme"],
            out["primary_poor_agreement"],
            out["hd99_local_extreme"],
        ],
        ["primary_and_hd99", "primary_only", "hd99_only"],
        default="not_flagged",
    )

    scales = scale_audit.set_index("metric").to_dict("index")
    dice_scale = max(float(scales["dice"]["robust_scale"]), 1e-12)
    h95_scale = max(float(scales["hd95_mm"]["robust_scale"]), 1e-12)
    h99_scale = max(float(scales["hd99_mm"]["robust_scale"]), 1e-12)
    out["dice_limit_exceedance_z"] = np.maximum(0.0, (dl - out["dice_for_flagging"]) / dice_scale)
    out["hd95_limit_exceedance_z"] = np.maximum(0.0, (out["hd95_for_flagging_mm"] - h95) / h95_scale)
    out["hd99_limit_exceedance_z"] = np.maximum(0.0, (out["hd99_mm"] - h99) / h99_scale)
    out["inspection_severity_score"] = (
        out["dice_limit_exceedance_z"]
        + out["hd95_limit_exceedance_z"]
        + out["hd99_limit_exceedance_z"]
    )
    return out


def save_plots(df: pd.DataFrame, limits: dict, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    for month in ["month1", "month2", "month3"]:
        d = df[df["month"] == month]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(d["dice_for_flagging"], d["hd95_for_flagging_mm"], alpha=0.72)
        f = d[d["inspection_union"]]
        if not f.empty:
            ax.scatter(f["dice_for_flagging"], f["hd95_for_flagging_mm"], marker="x", s=55)
        ax.axvline(limits["dice"], linestyle="--", linewidth=1)
        ax.axhline(limits["hd95_mm"], linestyle="--", linewidth=1)
        ax.set_xlabel("Clinical Dice")
        ax.set_ylabel("Clinical HD95 / historical haus (mm)")
        ax.set_title(f"Lung_L {month}: primary poor-agreement plane")
        fig.tight_layout()
        fig.savefig(figdir / f"{month}_dice_vs_hd95.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        order = d.sort_values("hd99_mm", ascending=False).reset_index(drop=True)
        ax.scatter(np.arange(len(order)), order["hd99_mm"], alpha=0.75)
        ax.axhline(limits["hd99_mm"], linestyle="--", linewidth=1)
        ax.set_xlabel("Cases sorted by HD99")
        ax.set_ylabel("HD99 (mm)")
        ax.set_title(f"Lung_L {month}: local-extreme HD99 screen")
        fig.tight_layout()
        fig.savefig(figdir / f"{month}_hd99.png", dpi=180)
        plt.close(fig)


def md_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_None._"
    x = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows).copy()
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]):
            x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    headers = list(x.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in x.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in headers) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, limits: dict, limit_audit: pd.DataFrame, failures: pd.DataFrame, outdir: Path) -> None:
    lines = [
        "# Lung_L approved-versus-autoseg inspection flags",
        "",
        "## Flagging rule",
        "",
        "The primary flag preserves the previous poor-agreement logic: **Dice < LCL AND HD95 > UCL**. "
        "A supplemental **HD99 > UCL99** flag is OR'ed into the manual-inspection list to catch localized extreme disagreement. "
        "The HD99 trigger does not redefine the historical primary flag.",
        "",
        f"- Dice LCL: **{limits['dice']:.4f}**",
        f"- HD95 UCL: **{limits['hd95_mm']:.4f} mm**",
        f"- HD99 UCL: **{limits['hd99_mm']:.4f} mm**",
        "",
        "One fixed threshold set is applied unchanged to all months.",
        "",
        "## Control-limit audit",
        "",
        md_table(limit_audit, ["metric", "reference_months", "n", "center_median", "robust_scale", "scale_method", "k", "derived_limit", "override", "final_limit"]),
        "",
        "## Month summary",
        "",
        "| Month | Available | Primary | HD99 extreme | Union | Primary+HD99 | HD99-only |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for month in ["month1", "month2", "month3"]:
        d = df[df["month"] == month]
        lines.append(
            f"| {month} | {len(d)} | {int(d['primary_poor_agreement'].sum())} | "
            f"{int(d['hd99_local_extreme'].sum())} | {int(d['inspection_union'].sum())} | "
            f"{int((d['inspection_reason']=='primary_and_hd99').sum())} | {int((d['inspection_reason']=='hd99_only').sum())} |"
        )

    cols = [
        "patient_id", "recorded_domain", "inspection_reason", "dice_for_flagging",
        "hd95_for_flagging_mm", "hd99_mm", "recomputed_assd_mm", "missing_fraction_gt",
        "excess_fraction_gt", "inspection_severity_score",
    ]
    for month in ["month1", "month2", "month3"]:
        d = df[(df["month"] == month) & df["inspection_union"]].sort_values(
            "inspection_severity_score", ascending=False
        )
        lines.extend(["", f"## {month} cases to inspect", "", md_table(d, cols, 100)])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- These are **inspection candidates**, not automatic claims that the approved contour is abnormal.",
        "- `primary_poor_agreement` uses the historical-style joint Dice+HD95 criterion.",
        "- `hd99_only` is intentionally retained because a local bite/blob can have a large extreme surface distance while leaving Dice and HD95 relatively modest.",
        "- Primary Dice/HD95 values preferentially come from the original Radformation `metrics.csv`; recomputed values are included for audit and fallback.",
        "- A patient appearing in more than one declared month list is retained in each listed month and marked `month_overlap_patient` rather than silently reassigned.",
        "- Domain labels are annotation only and are not used to decide the flag.",
        "",
        "## Files",
        "",
        "- `all_month_cases.csv`: complete month-stratified table.",
        "- `inspection_union.csv`: all flagged rows, ranked within month.",
        "- `month1_inspection.csv`, `month2_inspection.csv`, `month3_inspection.csv`: month-specific review sets.",
        "- `control_limits.csv`: exact threshold derivation/overrides.",
        "- `clinical_metrics_join_audit.csv`: ambiguous/duplicate original metric rows.",
        "- `metric_failures.csv`: NIfTI cases that could not be recomputed.",
        "- `figures/`: month-specific Dice-vs-HD95 and HD99 plots.",
    ])
    if not failures.empty:
        lines.extend(["", "## Metric failures", "", md_table(failures, list(failures.columns), 100)])
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    roi = str(cfg.get("roi", "Lung_L"))
    data_root = Path(cfg["data_root"])
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    membership = month_membership_rows()
    unique_pids = sorted(set(membership["patient_id"]), key=lambda x: int(x))
    requested = args.workers if args.workers is not None else int(cfg.get("parallel", {}).get("workers", 0))
    available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = requested if requested and requested > 0 else available_cpus
    workers = max(1, min(int(workers), len(unique_pids)))
    print(f"Recomputing Lung_L surface metrics for {len(unique_pids)} unique patients with {workers} workers", flush=True)

    rows: List[dict] = []
    fails: List[dict] = []
    tasks = [(pid, roi, str(data_root)) for pid in unique_pids]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(metric_worker, task): task[0] for task in tasks}
        done = 0
        for fut in as_completed(futs):
            result = fut.result()
            done += 1
            if result["ok"]:
                rows.append(result["row"])
                print(f"[{done}/{len(tasks)}] patient {result['row']['patient_id']} complete", flush=True)
            else:
                fails.append({"patient_id": result["patient_id"], "error": result["error"]})
                print(f"[{done}/{len(tasks)}] patient {result['patient_id']} FAILED: {result['error']}", flush=True)

    recomputed = pd.DataFrame(rows)
    failures = pd.DataFrame(fails)
    clinical, clinical_audit = clinical_wide(
        cfg["clinical_metrics_csv"], str(cfg.get("site", "thorax")), roi,
        str(cfg.get("clinical_metrics_duplicate_policy", "exclude")),
    )
    domains = load_domains(cfg.get("domain_csv"), str(cfg.get("site", "thorax")), roi)

    case = membership.merge(recomputed, on="patient_id", how="left")
    case = case.merge(clinical, on="patient_id", how="left")
    case = case.merge(domains, on="patient_id", how="left")
    case["recorded_domain"] = case["recorded_domain"].fillna("")
    case["domain_conflict"] = case["domain_conflict"].fillna(False).astype(bool)

    clinical_dice = case["clinical_dice"] if "clinical_dice" in case.columns else pd.Series(np.nan, index=case.index)
    clinical_hd95 = case["clinical_hd95_mm"] if "clinical_hd95_mm" in case.columns else pd.Series(np.nan, index=case.index)
    case["dice_for_flagging"] = pd.to_numeric(clinical_dice, errors="coerce")
    case["hd95_for_flagging_mm"] = pd.to_numeric(clinical_hd95, errors="coerce")
    case["dice_flag_source"] = np.where(case["dice_for_flagging"].notna(), "metrics.csv", "recomputed_fallback")
    case["hd95_flag_source"] = np.where(case["hd95_for_flagging_mm"].notna(), "metrics.csv:haus", "recomputed_fallback")
    case["dice_for_flagging"] = case["dice_for_flagging"].fillna(case["recomputed_dice"])
    case["hd95_for_flagging_mm"] = case["hd95_for_flagging_mm"].fillna(case["recomputed_hd95_mm"])

    limits, limit_audit = fit_limits(case, cfg)
    flagged = apply_flags(case, limits, limit_audit)
    flagged["month_order"] = flagged["month"].map({"month1": 1, "month2": 2, "month3": 3})
    flagged = flagged.sort_values(["month_order", "inspection_union", "inspection_severity_score"], ascending=[True, False, False])

    flagged.to_csv(outdir / "all_month_cases.csv", index=False)
    limit_audit.to_csv(outdir / "control_limits.csv", index=False)
    clinical_audit.to_csv(outdir / "clinical_metrics_join_audit.csv", index=False)
    failures.to_csv(outdir / "metric_failures.csv", index=False)

    union = flagged[flagged["inspection_union"]].copy()
    union.to_csv(outdir / "inspection_union.csv", index=False)
    for month in ["month1", "month2", "month3"]:
        d = union[union["month"] == month].sort_values("inspection_severity_score", ascending=False)
        d.to_csv(outdir / f"{month}_inspection.csv", index=False)

    save_plots(flagged, limits, outdir / "figures")
    write_report(flagged, limits, limit_audit, failures, outdir)

    metadata = {
        "roi": roi,
        "data_root": str(data_root),
        "clinical_metrics_csv": cfg["clinical_metrics_csv"],
        "domain_csv": cfg.get("domain_csv"),
        "workers": workers,
        "flag_rule": "(Dice<LCL AND HD95>UCL) OR HD99>UCL99",
        "limits": limits,
        "month_counts": {m: int((flagged["month"] == m).sum()) for m in MONTHS},
        "month_flagged_counts": {m: int(((flagged["month"] == m) & flagged["inspection_union"]).sum()) for m in MONTHS},
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote Lung_L inspection flags to {outdir}")
    for m in ["month1", "month2", "month3"]:
        print(f"  {m}: {metadata['month_flagged_counts'][m]} flagged / {metadata['month_counts'][m]} rows")


if __name__ == "__main__":
    main()
