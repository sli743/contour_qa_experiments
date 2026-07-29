#!/usr/bin/env python3
"""
Compute AUC(clean vs corruption) for synthetic probe sweep experiments.

AUC is oriented so larger detector score means more abnormal:
  AUC = P(score_corrupt > score_clean) + 0.5 P(score_corrupt == score_clean)

Inputs are one or more probe_eval_sweep_detailed.csv files.
"""

from __future__ import annotations
import argparse, math, re
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROI_NAMES = [
    "A_Aorta","A_Pulmonary","BrachPlx_L","BrachPlx_R","Bronchus","Carina",
    "Esophagus","Heart","Liver","Lung_L","Lung_R","Ribs","SpinalCord",
    "Stomach","Trachea","V_VenaCava_S","NS_Chestwall"
]

SCORE_SPECS = {
    # Deployable detector scores: use only input and reconstruction.
    "one_minus_dice_input_vs_recon": ("dice_input_vs_recon", "one_minus", "deployable"),
    "hd95_input_vs_recon": ("hd95_input_vs_recon", "identity", "deployable"),
    "abs_input_recon_volume_diff_frac": (None, "abs_input_recon_volume_diff_frac", "deployable"),

    # Synthetic-only benchmarking scores: require clean target.
    "one_minus_dice_recon_vs_target": ("dice_recon_vs_target", "one_minus", "synthetic_only"),
    "hd95_recon_vs_target": ("hd95_recon_vs_target", "identity", "synthetic_only"),
    "neg_dice_repair_gain": (None, "neg_dice_repair_gain", "synthetic_only"),
    "neg_hd95_repair_gain": (None, "neg_hd95_repair_gain", "synthetic_only"),

    # Corruption-control score, not a detector.
    "abs_input_volume_change_frac": (None, "abs_input_volume_change_frac", "corruption_control"),
}

PREFERRED_PROBES = [
    "clean","shifted","eroded","dilated","cut_lower","cut_upper","cut_posterior",
    "cut_anterior","cut_left","cut_right","cut_lower_half","cut_upper_half",
    "cut_posterior_half","big_bite","big_blob"
]


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "value"


def infer_run(path: Path) -> str:
    return path.parent.parent.name if path.parent.name == "probe_eval_sweep" else path.stem


def infer_roi(text: str) -> str:
    low = str(text).lower()
    for roi in sorted(ROI_NAMES, key=len, reverse=True):
        if roi.lower() in low:
            return roi
    return "unknown_roi"


def infer_fov(text: str) -> str:
    m = re.search(r"(\d+)x(\d+)x(\d+)", str(text))
    return f"{m.group(1)}x{m.group(2)}x{m.group(3)}" if m else "unknown_fov"


def load_csvs(csvs: List[str]) -> pd.DataFrame:
    frames = []
    for c in csvs:
        p = Path(c)
        if not p.exists():
            print(f"WARNING: missing CSV, skipping: {p}")
            continue
        df = pd.read_csv(p)
        if df.empty:
            print(f"WARNING: empty CSV, skipping: {p}")
            continue
        run = infer_run(p)
        df["source_csv"] = str(p)
        df["run_name"] = df.get("run_name", run)
        if "roi" not in df.columns:
            df["roi"] = infer_roi(run)
        else:
            df["roi"] = df["roi"].fillna(infer_roi(run))
        if "fov" not in df.columns:
            df["fov"] = infer_fov(run)
        else:
            df["fov"] = df["fov"].fillna(infer_fov(run))
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid CSV files loaded.")
    return pd.concat(frames, ignore_index=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "input_voxels","target_voxels","recon_voxels",
        "dice_input_vs_target","dice_input_vs_recon","dice_recon_vs_target",
        "hd95_input_vs_target","hd95_input_vs_recon","hd95_recon_vs_target",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "requested_level" in out.columns:
        out["level"] = pd.to_numeric(out["requested_level"], errors="coerce").fillna(0)
    elif "level" in out.columns:
        out["level"] = pd.to_numeric(out["level"], errors="coerce").fillna(0)
    else:
        out["level"] = 0.0

    if {"input_voxels","target_voxels"}.issubset(out.columns):
        denom = out["target_voxels"].replace(0, np.nan)
        out["abs_input_volume_change_frac"] = ((out["input_voxels"] - out["target_voxels"]) / denom).abs()

    if {"input_voxels","recon_voxels"}.issubset(out.columns):
        denom = out["input_voxels"].replace(0, np.nan)
        out["abs_input_recon_volume_diff_frac"] = ((out["input_voxels"] - out["recon_voxels"]) / denom).abs()

    if {"dice_recon_vs_target","dice_input_vs_target"}.issubset(out.columns):
        out["dice_repair_gain"] = out["dice_recon_vs_target"] - out["dice_input_vs_target"]
        out["neg_dice_repair_gain"] = -out["dice_repair_gain"]

    if {"hd95_input_vs_target","hd95_recon_vs_target"}.issubset(out.columns):
        out["hd95_repair_gain"] = out["hd95_input_vs_target"] - out["hd95_recon_vs_target"]
        out["neg_hd95_repair_gain"] = -out["hd95_repair_gain"]

    return out


def add_scores(df: pd.DataFrame, selected_scores: List[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    specs = SCORE_SPECS if selected_scores is None else {s: SCORE_SPECS[s] for s in selected_scores}
    meta = []
    for score, (source, transform, category) in specs.items():
        if transform == "identity":
            if source not in out.columns: 
                print(f"WARNING: skipping {score}; missing {source}")
                continue
            out[score] = pd.to_numeric(out[source], errors="coerce")
        elif transform == "one_minus":
            if source not in out.columns:
                print(f"WARNING: skipping {score}; missing {source}")
                continue
            out[score] = 1.0 - pd.to_numeric(out[source], errors="coerce")
        else:
            if transform not in out.columns:
                print(f"WARNING: skipping {score}; missing derived {transform}")
                continue
            out[score] = pd.to_numeric(out[transform], errors="coerce")
        meta.append({"score": score, "source_col": source, "transform": transform, "category": category,
                     "oriented": "larger = more abnormal"})
    return out, pd.DataFrame(meta)


def auc_pairwise(clean: np.ndarray, corrupt: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=float)
    corrupt = np.asarray(corrupt, dtype=float)
    clean = clean[np.isfinite(clean)]
    corrupt = corrupt[np.isfinite(corrupt)]
    if len(clean) == 0 or len(corrupt) == 0:
        return np.nan
    comp = corrupt[:, None] - clean[None, :]
    return float((np.sum(comp > 0) + 0.5 * np.sum(comp == 0)) / comp.size)


def cohen_d(clean: np.ndarray, corrupt: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=float); corrupt = np.asarray(corrupt, dtype=float)
    clean = clean[np.isfinite(clean)]; corrupt = corrupt[np.isfinite(corrupt)]
    if len(clean) < 2 or len(corrupt) < 2:
        return np.nan
    pooled = math.sqrt(((len(clean)-1)*np.var(clean, ddof=1) + (len(corrupt)-1)*np.var(corrupt, ddof=1)) / (len(clean)+len(corrupt)-2))
    return float((np.mean(corrupt) - np.mean(clean)) / pooled) if pooled > 0 else np.nan


def compute_auc(df: pd.DataFrame, scores: List[str], min_clean: int, min_corrupt: int) -> pd.DataFrame:
    rows = []
    group_cols = ["source_csv","run_name","roi","fov"]
    for keys, g in df.groupby(group_cols, dropna=False):
        info = dict(zip(group_cols, keys))
        clean = g[g["probe_type"].astype(str) == "clean"].copy()
        if clean.empty:
            clean = g[pd.to_numeric(g["level"], errors="coerce").fillna(0) == 0].copy()
        if clean.empty:
            continue
        for probe, gp in g.groupby("probe_type", dropna=False):
            probe = str(probe)
            if probe == "clean":
                continue
            for level, gl in gp.groupby("level", dropna=False):
                for score in scores:
                    c = pd.to_numeric(clean[score], errors="coerce").to_numpy()
                    q = pd.to_numeric(gl[score], errors="coerce").to_numpy()
                    c = c[np.isfinite(c)]; q = q[np.isfinite(q)]
                    ok = len(c) >= min_clean and len(q) >= min_corrupt
                    rows.append({
                        **info,
                        "probe_type": probe,
                        "level": float(level) if pd.notna(level) else np.nan,
                        "score": score,
                        "n_clean": int(len(c)),
                        "n_corrupt": int(len(q)),
                        "clean_mean": float(np.mean(c)) if len(c) else np.nan,
                        "clean_std": float(np.std(c, ddof=1)) if len(c) > 1 else np.nan,
                        "corrupt_mean": float(np.mean(q)) if len(q) else np.nan,
                        "corrupt_std": float(np.std(q, ddof=1)) if len(q) > 1 else np.nan,
                        "auc": auc_pairwise(c, q) if ok else np.nan,
                        "cohen_d": cohen_d(c, q) if ok else np.nan,
                    })
    return pd.DataFrame(rows)


def thresholds(auc_df: pd.DataFrame, thrs: List[float]) -> pd.DataFrame:
    rows = []
    id_cols = ["source_csv","run_name","roi","fov","probe_type","score"]
    for keys, g in auc_df.groupby(id_cols, dropna=False):
        info = dict(zip(id_cols, keys))
        gg = g.copy()
        gg["auc"] = pd.to_numeric(gg["auc"], errors="coerce")
        gg["level"] = pd.to_numeric(gg["level"], errors="coerce")
        gg = gg.dropna(subset=["auc","level"]).sort_values("level")
        for thr in thrs:
            hit = gg[gg["auc"] >= thr]
            if hit.empty:
                rows.append({**info, "auc_threshold": thr, "detected": False,
                             "first_detectable_level": np.nan, "first_detectable_auc": np.nan})
            else:
                first = hit.iloc[0]
                rows.append({**info, "auc_threshold": thr, "detected": True,
                             "first_detectable_level": float(first["level"]),
                             "first_detectable_auc": float(first["auc"])})
    return pd.DataFrame(rows)


def summary(auc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in auc_df.groupby(["roi","fov","probe_type","score"], dropna=False):
        gg = g.copy()
        gg["auc"] = pd.to_numeric(gg["auc"], errors="coerce")
        gg = gg.dropna(subset=["auc"])
        if gg.empty:
            continue
        rows.append({
            "roi": keys[0], "fov": keys[1], "probe_type": keys[2], "score": keys[3],
            "mean_auc_across_levels": float(gg["auc"].mean()),
            "median_auc_across_levels": float(gg["auc"].median()),
            "max_auc": float(gg["auc"].max()),
            "n_levels": int(len(gg)),
        })
    return pd.DataFrame(rows)


def plot_auc_curves(auc_df: pd.DataFrame, out_dir: Path, thrs: List[float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (roi, fov, probe, score), g in auc_df.groupby(["roi","fov","probe_type","score"], dropna=False):
        gg = g.copy()
        gg["level"] = pd.to_numeric(gg["level"], errors="coerce")
        gg["auc"] = pd.to_numeric(gg["auc"], errors="coerce")
        gg = gg.dropna(subset=["level","auc"]).sort_values("level")
        if gg.empty:
            continue
        plt.figure(figsize=(7.2, 5.0))
        plt.plot(gg["level"], gg["auc"], marker="o", linewidth=2)
        for t in thrs:
            plt.axhline(t, linestyle="--", linewidth=1.1, alpha=0.6)
        plt.ylim(0.45, 1.02)
        plt.xlabel("Requested corruption level")
        plt.ylabel("AUC(clean vs corruption)")
        plt.title(f"{roi} | {fov} | {probe}\n{score}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"auc_vs_level__{sanitize(roi)}__{sanitize(fov)}__{sanitize(probe)}__{sanitize(score)}.png", dpi=300)
        plt.close()


def plot_mean_auc(summary_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return
    agg = summary_df.groupby("score")["mean_auc_across_levels"].mean().reset_index().sort_values("mean_auc_across_levels", ascending=False)
    plt.figure(figsize=(max(8, 0.6 * len(agg)), 5))
    plt.bar(agg["score"], agg["mean_auc_across_levels"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.45, 1.02)
    plt.ylabel("Mean AUC")
    plt.title("Average separability by detector score")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_auc_by_score.png", dpi=300)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scores", nargs="+", default=None, choices=list(SCORE_SPECS.keys()))
    p.add_argument("--auc_thresholds", default="0.8,0.9,0.95")
    p.add_argument("--min_clean", type=int, default=3)
    p.add_argument("--min_corrupt", type=int, default=3)
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thrs = [float(x) for x in args.auc_thresholds.split(",") if x.strip()]

    df = load_csvs(args.csvs)
    df = add_derived(df)
    df, meta = add_scores(df, args.scores)
    scores = meta["score"].tolist()

    df.to_csv(out_dir / "probe_eval_sweep_with_detector_scores.csv", index=False)
    meta.to_csv(out_dir / "detector_score_definitions.csv", index=False)

    auc_df = compute_auc(df, scores, args.min_clean, args.min_corrupt)
    auc_df.to_csv(out_dir / "auc_clean_vs_corruption_long.csv", index=False)

    thr_df = thresholds(auc_df, thrs)
    thr_df.to_csv(out_dir / "auc_detection_thresholds.csv", index=False)

    summ = summary(auc_df)
    summ.to_csv(out_dir / "auc_metric_probe_summary.csv", index=False)

    if not args.no_plots:
        plot_auc_curves(auc_df, out_dir / "plots" / "auc_vs_level", thrs)
        plot_mean_auc(summ, out_dir / "plots")

    (out_dir / "README_auc_sensitivity.md").write_text(
        "AUC(clean vs corruption) = P(score_corrupt > score_clean) + 0.5 P(equal). "
        "All scores are oriented so larger means more abnormal.\n",
        encoding="utf-8",
    )
    print(f"Wrote AUC analysis to: {out_dir}")
    print(f"Scores: {scores}")

if __name__ == "__main__":
    main()
