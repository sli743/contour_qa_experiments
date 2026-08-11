#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import auc, roc_curve

PROBE_CANDIDATES = ("probe_type", "probe", "corruption_type", "corruption", "perturbation_type", "perturbation")
PATIENT_CANDIDATES = ("patient_id", "patient", "case_id", "case")
LEVEL_CANDIDATES = ("level", "probe_level", "severity", "magnitude")
VOLUME_CHANGE_CANDIDATES = (
    "abs_volume_change_pct", "absolute_volume_change_pct", "volume_change_pct",
    "input_target_volume_change_pct", "input_vs_target_volume_change_pct", "volume_diff_pct",
)

METRICS = (
    "one_minus_dice_input_vs_recon",
    "hd95_input_vs_recon",
    "assd_input_vs_recon",
    "relative_volume_difference_input_vs_recon",
)
METRIC_LABELS = {
    "one_minus_dice_input_vs_recon": "1 - Dice(input, reconstruction)",
    "hd95_input_vs_recon": "HD95(input, reconstruction)",
    "assd_input_vs_recon": "ASSD(input, reconstruction)",
    "relative_volume_difference_input_vs_recon": "Relative volume difference(input, reconstruction)",
}

LOCAL_PROBES = {"big_bite", "big_blob"}
GLOBAL_PROBES = {"dilated", "eroded", "cut_lower_half", "cut_upper_half", "cut_posterior_half", "strong"}


def parse_models(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --model NAME=CSV, got: {item}")
        name, path = item.split("=", 1)
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{name}: {p}")
        out.append((name, p))
    return out


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def is_clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().isin({"clean", "none", "identity", "original", "approved"})


def is_shift_name(name: str) -> bool:
    return bool(re.search(r"shift|translation|translated", str(name), flags=re.I))


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "probe"


def normalize_probe_name(s: str) -> str:
    return str(s).strip().lower()


def normalize_level(x) -> float:
    return round(float(x), 8)


def parse_level_token(token: str) -> float:
    token = token.replace("m", "-").replace("p", ".")
    return normalize_level(float(token))


def derive_volume_change_pct(df: pd.DataFrame) -> pd.Series:
    col = find_column(df, VOLUME_CHANGE_CANDIDATES)
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce").abs()
    if {"input_voxels", "target_voxels"}.issubset(df.columns):
        inp = pd.to_numeric(df["input_voxels"], errors="coerce")
        tgt = pd.to_numeric(df["target_voxels"], errors="coerce").replace(0, np.nan)
        return 100.0 * (inp - tgt).abs() / tgt
    raise ValueError("Cannot derive absolute input-vs-target volume change percentage")


def derive_score(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "one_minus_dice_input_vs_recon":
        if "dice_input_vs_recon" not in df.columns:
            raise ValueError("Missing dice_input_vs_recon")
        return 1.0 - pd.to_numeric(df["dice_input_vs_recon"], errors="coerce")
    if metric == "relative_volume_difference_input_vs_recon":
        if not {"input_voxels", "recon_voxels"}.issubset(df.columns):
            raise ValueError("Relative volume score requires input_voxels and recon_voxels")
        inp = pd.to_numeric(df["input_voxels"], errors="coerce").replace(0, np.nan)
        rec = pd.to_numeric(df["recon_voxels"], errors="coerce")
        return (inp - rec).abs() / inp
    if metric not in df.columns:
        raise ValueError(f"Missing {metric}")
    return pd.to_numeric(df[metric], errors="coerce")


def dedup_clean(df: pd.DataFrame, probe_col: str, patient_col: str | None) -> pd.DataFrame:
    c = df.loc[is_clean(df[probe_col])].copy()
    if patient_col is not None:
        c = c.drop_duplicates(patient_col, keep="first")
    return c


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        fpr, tpr, _ = roc_curve(yy, score[idx])
        vals.append(auc(fpr, tpr))
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def sensitivity_at_specificity(y: np.ndarray, score: np.ndarray, target_spec: float = 0.95) -> tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y, score)
    spec = 1.0 - fpr
    ok = np.where(spec >= target_spec)[0]
    if ok.size == 0:
        return np.nan, np.nan, np.nan
    j = ok[np.argmax(tpr[ok])]
    return float(tpr[j]), float(spec[j]), float(thr[j])


def clean_threshold_and_sensitivity(clean_scores: np.ndarray, severe_scores: np.ndarray, specificity: float = 0.95) -> tuple[float, float]:
    clean_scores = clean_scores[np.isfinite(clean_scores)]
    severe_scores = severe_scores[np.isfinite(severe_scores)]
    if clean_scores.size == 0 or severe_scores.size == 0:
        return np.nan, np.nan
    threshold = float(np.quantile(clean_scores, specificity, method="higher"))
    return threshold, float(np.mean(severe_scores > threshold))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    d = a[:, None] - b[None, :]
    return float((np.sum(d > 0) - np.sum(d < 0)) / d.size)


def load_models(models: list[tuple[str, Path]]) -> dict[str, dict[str, object]]:
    loaded = {}
    for name, path in models:
        df = pd.read_csv(path)
        probe_col = find_column(df, PROBE_CANDIDATES)
        if probe_col is None:
            raise ValueError(f"{name}: could not find probe column in {path}")
        patient_col = find_column(df, PATIENT_CANDIDATES)
        level_col = find_column(df, LEVEL_CANDIDATES)
        df = df.copy()
        df["__probe_norm"] = df[probe_col].astype(str).map(normalize_probe_name)
        if patient_col is not None:
            df["__patient_norm"] = df[patient_col].astype(str)
        if level_col is not None:
            df["__level_norm"] = pd.to_numeric(df[level_col], errors="coerce").map(lambda x: normalize_level(x) if pd.notna(x) else np.nan)
        df["__volume_change_pct"] = derive_volume_change_pct(df)
        for metric in METRICS:
            df[f"__score__{metric}"] = derive_score(df, metric)
        loaded[name] = {
            "path": path, "df": df, "probe_col": probe_col,
            "patient_col": patient_col, "level_col": level_col,
        }
    return loaded


def common_probe_types(loaded: dict[str, dict[str, object]], exclude_shifts: bool) -> list[str]:
    sets = []
    display_names: dict[str, str] = {}
    for obj in loaded.values():
        df = obj["df"]
        probe_col = obj["probe_col"]
        vals = df.loc[~is_clean(df[probe_col]), probe_col].astype(str).unique()
        probes = set()
        for p in vals:
            norm = normalize_probe_name(p)
            if exclude_shifts and is_shift_name(norm):
                continue
            probes.add(norm)
            display_names.setdefault(norm, p)
        sets.append(probes)
    common = set.intersection(*sets) if sets else set()
    return [display_names[p] for p in sorted(common)]


def probe_family(probe: str) -> str:
    p = normalize_probe_name(probe)
    if p in LOCAL_PROBES:
        return "local"
    if is_shift_name(p):
        return "shift"
    return "global"


def _same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def local_defect_depth_p95_mm(clean_img: sitk.Image, corrupted_img: sitk.Image) -> tuple[float, float, float]:
    """Return (combined p95, missing p95, excess p95) depth from the target boundary."""
    if not _same_geometry(clean_img, corrupted_img):
        raise ValueError("Clean/corrupted NIfTI geometry mismatch while computing local defect depth")
    target = sitk.GetArrayFromImage(clean_img) > 0
    inp = sitk.GetArrayFromImage(corrupted_img) > 0
    sampling_zyx = tuple(float(v) for v in clean_img.GetSpacing()[::-1])

    missing = target & ~inp
    excess = inp & ~target
    values = []
    missing_values = np.array([], dtype=float)
    excess_values = np.array([], dtype=float)
    if missing.any():
        inside_depth = ndimage.distance_transform_edt(target, sampling=sampling_zyx)
        missing_values = inside_depth[missing]
        values.append(missing_values)
    if excess.any():
        outside_depth = ndimage.distance_transform_edt(~target, sampling=sampling_zyx)
        excess_values = outside_depth[excess]
        values.append(excess_values)
    if not values:
        return 0.0, 0.0, 0.0
    joined = np.concatenate(values)
    combined = float(np.percentile(joined, 95))
    miss = float(np.percentile(missing_values, 95)) if missing_values.size else 0.0
    exc = float(np.percentile(excess_values, 95)) if excess_values.size else 0.0
    return combined, miss, exc


def build_local_severity_table(vis_dir: Path) -> pd.DataFrame:
    if not vis_dir.is_dir():
        raise FileNotFoundError(f"Probe visualization directory not found: {vis_dir}")
    rx = re.compile(r"^corrupted_input_(?P<probe>.+)_level_(?P<level>[^_]+)_patient_(?P<patient>.+)\.nii\.gz$")
    rows = []
    for path in sorted(vis_dir.glob("corrupted_input_*_level_*_patient_*.nii.gz")):
        m = rx.match(path.name)
        if not m:
            continue
        probe = m.group("probe")
        if normalize_probe_name(probe) not in LOCAL_PROBES:
            continue
        patient = m.group("patient")
        level = parse_level_token(m.group("level"))
        clean_path = vis_dir / f"clean_input_patient_{patient}.nii.gz"
        if not clean_path.is_file():
            raise FileNotFoundError(f"Missing matching clean contour for {path}: {clean_path}")
        clean_img = sitk.ReadImage(str(clean_path))
        corr_img = sitk.ReadImage(str(path))
        p95, missing_p95, excess_p95 = local_defect_depth_p95_mm(clean_img, corr_img)
        target = sitk.GetArrayFromImage(clean_img) > 0
        inp = sitk.GetArrayFromImage(corr_img) > 0
        target_n = int(target.sum())
        vol_change = 100.0 * abs(int(inp.sum()) - target_n) / target_n if target_n else np.nan
        rows.append({
            "patient_id": str(patient),
            "probe_type": probe,
            "level": level,
            "local_defect_depth_p95_mm": p95,
            "missing_depth_p95_mm": missing_p95,
            "excess_depth_p95_mm": excess_p95,
            "geometry_volume_change_pct": vol_change,
            "corrupted_path": str(path),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No local big_bite/big_blob flat-layout probe files found under {vis_dir}")
    if out.duplicated(["patient_id", "probe_type", "level"]).any():
        dup = out.loc[out.duplicated(["patient_id", "probe_type", "level"], keep=False)]
        raise RuntimeError(f"Duplicate local severity keys in {vis_dir}:\n{dup.head()}")
    return out


def attach_local_severity(
    loaded: dict[str, dict[str, object]],
    severity: pd.DataFrame,
    volume_tolerance_pct: float,
) -> None:
    sev = severity.copy()
    sev["__patient_norm"] = sev["patient_id"].astype(str)
    sev["__probe_norm"] = sev["probe_type"].astype(str).map(normalize_probe_name)
    sev["__level_norm"] = pd.to_numeric(sev["level"], errors="coerce").map(normalize_level)
    keep = [
        "__patient_norm", "__probe_norm", "__level_norm",
        "local_defect_depth_p95_mm", "missing_depth_p95_mm", "excess_depth_p95_mm",
        "geometry_volume_change_pct",
    ]
    sev = sev[keep]

    for model_name, obj in loaded.items():
        if obj["patient_col"] is None or obj["level_col"] is None:
            raise ValueError(
                f"{model_name}: local-depth analysis requires patient and probe-level columns so the "
                "shared benchmark geometry can be matched exactly."
            )
        df = obj["df"]
        merged = df.merge(sev, on=["__patient_norm", "__probe_norm", "__level_norm"], how="left")
        local_mask = merged["__probe_norm"].isin(LOCAL_PROBES)
        nonclean_local = local_mask & ~is_clean(merged[obj["probe_col"]])
        missing = nonclean_local & merged["local_defect_depth_p95_mm"].isna()
        if missing.any():
            sample = merged.loc[missing, ["__patient_norm", "__probe_norm", "__level_norm"]].head(10)
            raise RuntimeError(
                f"{model_name}: {int(missing.sum())} local probe rows could not be matched to geometry from the "
                f"ProbeContourComparer export. Example keys:\n{sample.to_string(index=False)}"
            )
        check = merged.loc[nonclean_local & merged["geometry_volume_change_pct"].notna(), [
            "__volume_change_pct", "geometry_volume_change_pct", "__patient_norm", "__probe_norm", "__level_norm"
        ]].copy()
        if not check.empty:
            diff = (check["__volume_change_pct"] - check["geometry_volume_change_pct"]).abs()
            if np.nanmax(diff.to_numpy(float)) > volume_tolerance_pct:
                bad = check.loc[diff > volume_tolerance_pct].head(10)
                raise RuntimeError(
                    f"{model_name}: local probe geometry does not match the stochastic ProbeContourComparer export "
                    f"within {volume_tolerance_pct:g} percentage points of volume change. This suggests different "
                    f"random probe realizations, so borrowing local depth would be unfair. Examples:\n{bad.to_string(index=False)}"
                )
        obj["df"] = merged


def criterion_for_probe(probe: str, volume_threshold_pct: float, local_depth_threshold_mm: float) -> dict[str, object]:
    family = probe_family(probe)
    if family == "local":
        # Use each model CSV's own input-vs-target HD95 as the realized local
        # geometric severity.  This avoids borrowing a random bite/blob geometry
        # from another model's independently generated probe realization.
        return {
            "family": family,
            "severity_col": "hd95_input_vs_target",
            "threshold": float(local_depth_threshold_mm),
            "severity_label": "Local geometric severity: HD95(input,target) (mm)",
            "criterion_text": f"HD95(input,target) > {local_depth_threshold_mm:g} mm",
        }
    if family == "global":
        return {
            "family": family,
            "severity_col": "__volume_change_pct",
            "threshold": float(volume_threshold_pct),
            "severity_label": "Absolute input-target volume change (%)",
            "criterion_text": f"absolute input-target volume change > {volume_threshold_pct:g}%",
        }
    raise ValueError(f"Shift probe {probe} is not handled by the two-criterion analysis; use --exclude-shifts")


def prepare_probe_dataset(obj: dict[str, object], probe: str, criterion: dict[str, object]) -> pd.DataFrame:
    df = obj["df"]
    probe_col = obj["probe_col"]
    patient_col = obj["patient_col"]
    clean = dedup_clean(df, probe_col, patient_col)
    corr = df.loc[df["__probe_norm"].eq(normalize_probe_name(probe))].copy()
    out = pd.concat([clean, corr], ignore_index=True)
    out["__is_clean"] = is_clean(out[probe_col])
    severity_col = str(criterion["severity_col"])
    out["__severity"] = pd.to_numeric(out.get(severity_col, np.nan), errors="coerce")
    out.loc[out["__is_clean"], "__severity"] = 0.0
    out["__label"] = (out["__severity"] > float(criterion["threshold"])).astype(int)
    out.loc[out["__is_clean"], "__label"] = 0
    return out


def plot_roc_for_metric(rows: list[dict[str, object]], probe: str, metric: str, out_path: Path, roi: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    plotted = 0
    for row in rows:
        y = row["y"]
        score = row["score"]
        if np.unique(y).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y, score)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.1, label=f"{row['model']} (AUC={a:.3f})")
        plotted += 1
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.1, label="Chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"{roi}: {probe}\n{METRIC_LABELS[metric]}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    if plotted:
        ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_signal_vs_severity(
    loaded: dict[str, dict[str, object]],
    probe: str,
    metric: str,
    out_path: Path,
    criterion: dict[str, object],
    roi: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    any_data = False
    for name, obj in loaded.items():
        df = obj["df"]
        sub = df.loc[df["__probe_norm"].eq(normalize_probe_name(probe))].copy()
        severity_col = str(criterion["severity_col"])
        x = pd.to_numeric(sub[severity_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub[f"__score__{metric}"], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        x, y = x[valid], y[valid]
        ax.scatter(x, y, s=11, alpha=0.16)

        # A simple binned/level median line is easier to read than connecting every patient row.
        level_col = obj["level_col"]
        if level_col is not None:
            temp = sub.loc[valid, [level_col]].copy()
            temp["severity"] = x
            temp["score"] = y
            med = temp.groupby(level_col, dropna=True)[["severity", "score"]].median().sort_values("severity")
            if not med.empty:
                ax.plot(med["severity"], med["score"], marker="o", markersize=4.0, linewidth=1.7, label=name)
        else:
            order = np.argsort(x)
            ax.plot(x[order], y[order], marker="o", markersize=2.4, linewidth=1.0, label=name)
        any_data = True

    threshold = float(criterion["threshold"])
    ax.axvline(threshold, linestyle="--", linewidth=1.5, label=f"Good/bad separation ({threshold:g})")
    ax.text(
        threshold, 0.99, " good  |  bad ", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=8,
    )
    ax.set_xlabel(str(criterion["severity_label"]))
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{roi}: {probe} signal vs defect size")
    if any_data:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate one four-model QA report per benchmark probe using separate global/local severity criteria.")
    p.add_argument("--model", action="append", required=True, metavar="NAME=CSV")
    p.add_argument("--roi", required=True)
    p.add_argument("--volume-threshold-pct", type=float, required=True,
                   help="Bad-contour threshold for global probes")
    p.add_argument("--local-depth-threshold-mm", type=float, required=True,
                   help="Bad-contour threshold for big_bite/big_blob using each row's HD95(input,target)")
    p.add_argument("--probe-visualization-dir", type=Path, default=None,
                   help="Deprecated compatibility argument; local severity is now read from each model CSV")
    p.add_argument("--geometry-volume-tolerance-pct", type=float, default=0.5,
                   help="Deprecated compatibility argument")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--exclude-shifts", action="store_true")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=88)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.model)
    loaded = load_models(models)

    # Local probes can be random across historical model evaluations.  Do not
    # borrow local depth from the stochastic model's exported contour geometry.
    # Instead, use each model CSV's own HD95(input,target), which measures the
    # realized corruption against the same clean target for that row.

    probes = common_probe_types(loaded, args.exclude_shifts)
    if not probes:
        raise SystemExit("No common non-clean probe types found across all models")

    all_summary = []
    criteria_rows = []
    index_lines = [
        f"# {args.roi}: four-model comparison by corruption type", "",
        "This analysis intentionally uses **two simple defect-severity definitions** rather than one whole-ROI volume threshold for every corruption.", "",
        f"- Global probes: bad if **absolute input-target volume change > {args.volume_threshold_pct:g}%**.",
        f"- Local probes (`big_bite`, `big_blob`): bad if **HD95(input,target) > {args.local_depth_threshold_mm:g} mm**.",
        "  This is computed independently for each model's realized local probe, so random bite/blob anchors do not have to match across models.",
        "- Shift probes are excluded from this two-criterion report and should be assessed separately by physical displacement.",
        "- Each corruption type gets its own ROC analysis and Markdown report.", "",
        "## Reports", "",
    ]

    for probe_i, probe in enumerate(probes):
        criterion = criterion_for_probe(probe, args.volume_threshold_pct, args.local_depth_threshold_mm)
        criteria_rows.append({
            "probe_type": probe,
            "family": criterion["family"],
            "severity_measure": criterion["severity_label"],
            "bad_if": criterion["criterion_text"],
            "threshold": criterion["threshold"],
        })
        pdir = args.output_dir / safe_name(probe)
        pdir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        roc_payload: dict[str, list[dict[str, object]]] = {m: [] for m in METRICS}

        for model_i, (model_name, _) in enumerate(models):
            obj = loaded[model_name]
            ds = prepare_probe_dataset(obj, probe, criterion)
            corr = ds.loc[~ds["__is_clean"]].copy()
            severe = corr.loc[corr["__label"].eq(1)]
            mild = corr.loc[corr["__label"].eq(0)]
            clean = ds.loc[ds["__is_clean"]]

            for metric_i, metric in enumerate(METRICS):
                score_col = f"__score__{metric}"
                y = pd.to_numeric(ds["__label"], errors="coerce")
                s = pd.to_numeric(ds[score_col], errors="coerce")
                valid = y.notna() & s.notna() & np.isfinite(s) & pd.to_numeric(ds["__severity"], errors="coerce").notna()
                yy = y.loc[valid].astype(int).to_numpy()
                ss = s.loc[valid].to_numpy(float)

                row = {
                    "roi": args.roi, "probe_type": probe, "probe_family": criterion["family"],
                    "severity_measure": criterion["severity_label"], "severity_threshold": criterion["threshold"],
                    "model": model_name, "metric": metric,
                    "n_total": int(valid.sum()), "n_positive": int(yy.sum()), "n_negative": int((yy == 0).sum()),
                    "n_clean": int(len(clean)), "n_mild": int(len(mild)), "n_severe": int(len(severe)),
                }
                if np.unique(yy).size == 2:
                    fpr, tpr, _ = roc_curve(yy, ss)
                    row["auc"] = float(auc(fpr, tpr))
                    rng = np.random.default_rng(args.seed + probe_i * 10000 + model_i * 100 + metric_i)
                    row["auc_ci_low"], row["auc_ci_high"] = bootstrap_auc(yy, ss, args.n_boot, rng)
                    row["sens_at_95spec"], row["achieved_specificity"], row["roc_threshold_95spec"] = sensitivity_at_specificity(yy, ss, 0.95)
                    roc_payload[metric].append({"model": model_name, "y": yy, "score": ss})
                else:
                    row.update({"auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
                                "sens_at_95spec": np.nan, "achieved_specificity": np.nan, "roc_threshold_95spec": np.nan})

                clean_scores = pd.to_numeric(clean[score_col], errors="coerce").to_numpy(float)
                mild_scores = pd.to_numeric(mild[score_col], errors="coerce").to_numpy(float)
                severe_scores = pd.to_numeric(severe[score_col], errors="coerce").to_numpy(float)
                row["clean_score_median"] = float(np.nanmedian(clean_scores)) if np.isfinite(clean_scores).any() else np.nan
                row["mild_score_median"] = float(np.nanmedian(mild_scores)) if np.isfinite(mild_scores).any() else np.nan
                row["severe_score_median"] = float(np.nanmedian(severe_scores)) if np.isfinite(severe_scores).any() else np.nan
                row["severe_minus_clean_median"] = row["severe_score_median"] - row["clean_score_median"] if np.isfinite(row["severe_score_median"]) and np.isfinite(row["clean_score_median"]) else np.nan
                row["severe_vs_clean_cliffs_delta"] = cliffs_delta(severe_scores, clean_scores)
                clean_thr, clean_sens = clean_threshold_and_sensitivity(clean_scores, severe_scores, 0.95)
                row["clean_95pct_threshold"] = clean_thr
                row["severe_sensitivity_at_clean95"] = clean_sens

                corr_x = pd.to_numeric(corr["__severity"], errors="coerce").to_numpy(float)
                corr_y = pd.to_numeric(corr[score_col], errors="coerce").to_numpy(float)
                vv = np.isfinite(corr_x) & np.isfinite(corr_y)
                if np.sum(vv) >= 3 and np.unique(corr_x[vv]).size >= 2:
                    rho, pv = spearmanr(corr_x[vv], corr_y[vv])
                    row["spearman_severity_signal_rho"] = float(rho)
                    row["spearman_severity_signal_p"] = float(pv)
                else:
                    row["spearman_severity_signal_rho"] = np.nan
                    row["spearman_severity_signal_p"] = np.nan
                summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(pdir / "metric_summary.csv", index=False)
        all_summary.extend(summary_rows)

        for metric in METRICS:
            plot_roc_for_metric(roc_payload[metric], probe, metric, pdir / f"roc_{safe_name(metric)}.png", args.roi)
            plot_signal_vs_severity(
                loaded, probe, metric,
                pdir / f"signal_vs_defect_size_{safe_name(metric)}.png",
                criterion, args.roi,
            )

        md = [
            f"# {args.roi} — {probe}: four-model QA comparison", "",
            "## Good/bad definition", "",
            f"- Corruption family: **{criterion['family']}**.",
            f"- Severity measure: **{criterion['severity_label']}**.",
            f"- Bad contour: **{criterion['criterion_text']}**.",
            f"- Included corrupted samples: **{probe} only**.",
            "- Negative class: one clean contour per patient plus milder samples of this same corruption type that do not cross the criterion.",
            "- Higher detector score means larger disagreement between input contour and decoder reconstruction.",
            "- Repair gain is not the primary endpoint; the goal is a larger anomaly signal for substantial defects.", "",
            "## Per-metric results", "",
        ]
        for metric in METRICS:
            sub = summary.loc[summary["metric"].eq(metric)].copy()
            md += [f"### {METRIC_LABELS[metric]}", ""]
            md += [
                "| Model | AUC (95% CI) | Sens @ 95% spec | Severe sens @ clean-95% threshold | Median severe-clean signal lift | Cliff's delta severe vs clean | Spearman rho(severity, signal) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for _, r in sub.iterrows():
                def f3(x): return "NA" if not np.isfinite(x) else f"{x:.3f}"
                auc_txt = "NA" if not np.isfinite(r["auc"]) else f"{r['auc']:.3f} [{f3(r['auc_ci_low'])}, {f3(r['auc_ci_high'])}]"
                md.append(
                    f"| {r['model']} | {auc_txt} | {f3(r['sens_at_95spec'])} | {f3(r['severe_sensitivity_at_clean95'])} | "
                    f"{f3(r['severe_minus_clean_median'])} | {f3(r['severe_vs_clean_cliffs_delta'])} | {f3(r['spearman_severity_signal_rho'])} |"
                )
            md += [
                "",
                f"![ROC — {METRIC_LABELS[metric]}](roc_{safe_name(metric)}.png)", "",
                f"![Signal versus defect size — {METRIC_LABELS[metric]}](signal_vs_defect_size_{safe_name(metric)}.png)", "",
            ]

        md += [
            "## Interpretation guide", "",
            "The dashed vertical line in each signal-versus-defect-size plot is the **ground-truth good/bad separation**, not a detector threshold. "
            "For global probes the x-axis is whole-ROI volume change; for local bite/blob probes the x-axis is focal defect depth. "
            "A desirable QA detector shows low signal to the left of this line and a clear, increasing signal to the right.", "",
            "For the project objective, prioritize **sensitivity at high specificity**, **severe sensitivity at a threshold calibrated from clean contours**, "
            "and **signal increase with ground-truth defect severity** over reconstruction repair accuracy.", "",
        ]
        (pdir / "report.md").write_text("\n".join(md), encoding="utf-8")
        index_lines.append(f"- [{probe}]({safe_name(probe)}/report.md) — {criterion['criterion_text']}")

    pd.DataFrame(criteria_rows).to_csv(args.output_dir / "probe_criteria.csv", index=False)
    pd.DataFrame(all_summary).to_csv(args.output_dir / "all_probe_metric_summary.csv", index=False)
    (args.output_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    metadata = {
        "roi": args.roi,
        "global_volume_threshold_pct": args.volume_threshold_pct,
        "local_depth_threshold_mm": args.local_depth_threshold_mm,
        "local_depth_definition": "95th percentile distance of changed voxels from the clean target boundary; missing and excess voxels handled on their respective sides",
        "probe_visualization_dir": str(args.probe_visualization_dir),
        "geometry_volume_tolerance_pct": args.geometry_volume_tolerance_pct,
        "exclude_shifts": bool(args.exclude_shifts),
        "models": [{"name": n, "csv": str(p)} for n, p in models],
        "probe_types": probes,
        "primary_project_endpoint": "large anomaly signal rather than reconstruction repair",
        "per_probe_population": "one probe family plus one clean row per patient; mild same-family corruptions are negatives",
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated {len(probes)} per-probe four-model reports under {args.output_dir}")


if __name__ == "__main__":
    main()
