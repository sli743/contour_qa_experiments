#!/usr/bin/env python3
"""Stratified correction diagnostics for the actual SpinalCord V4 corruption families.

Two contexts are evaluated on clean approved validation/test shapes:
  1) isolated: exactly one V4 family is applied; endpoint families have no
     surface jitter or local bite.
  2) production_composite: an explicit endpoint event is composed after a
     V4 surface mode sampled with the production probabilities. Local bite is
     intentionally omitted here so endpoint correction remains interpretable;
     local bite is tested separately as its own family.

The same corrupted input is passed through both previous-generation and V4
models. This separates model capacity/training from synthetic-to-real transfer.

The report also audits the legacy cut-probe naming. In the current probe code,
``cut_upper_half`` removes LOW array-z indices and ``cut_lower_half`` removes
HIGH array-z indices. Their anatomical endpoint labels are derived from each
NIfTI geometry rather than assumed from the legacy names.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

from analysis.spinalcord_diagnostic_utils import correction_decomposition, pair_metrics
from evaluation.actual_autoseg.eval_actual_autoseg_nsrqa import read_binary
from evaluation.probes.eval_dino_shape_denoising_probe_sweep import load_config, make_model, reconstruct_binary
from training.corruptions.spinalcord_proposed_distribution import (
    apply_forced_v4_endpoint_event,
    apply_forced_v4_surface_mode,
    small_local_bite,
)
from training.spatial_reference import deployment_centered_training_pair, raw_ct_path

ENDPOINT_LEVELS = {
    "inferior_mild": [1.5, 3.0],
    "inferior_moderate": [6.0, 10.5, 15.0, 19.5],
    "inferior_severe": [30.0, 36.0, 42.0, 48.0],
    "superior_mild": [1.5, 3.0],
    "inferior_excess": [30.0, 36.0, 39.0],
    "superior_excess": [1.5, 4.5, 7.5, 9.0],
}
SURFACE_MODES = ["smooth_warp", "micro_jitter", "warp_plus_micro_jitter"]


def geometry_si_mapping(img: sitk.Image) -> Dict[str, object]:
    size = img.GetSize()
    if size[2] < 2:
        raise ValueError("Need at least two z slices")
    low = img.TransformIndexToPhysicalPoint((0, 0, 0))
    high = img.TransformIndexToPhysicalPoint((0, 0, int(size[2]) - 1))
    dz = float(high[2] - low[2])
    if abs(dz) < 1e-6:
        raise ValueError("Array z has no measurable physical S/I component")
    inferior_is_low_z = dz > 0.0  # LPS +physical-z points superiorly.
    low_endpoint = "inferior" if inferior_is_low_z else "superior"
    high_endpoint = "superior" if inferior_is_low_z else "inferior"
    return {
        "physical_z_low_index_mm": float(low[2]),
        "physical_z_high_index_mm": float(high[2]),
        "physical_z_delta_mm": dz,
        "inferior_is_low_z": bool(inferior_is_low_z),
        "low_array_z_endpoint": low_endpoint,
        "high_array_z_endpoint": high_endpoint,
        "legacy_cut_upper_half_anatomical_action": f"{low_endpoint}_endpoint_truncation",
        "legacy_cut_lower_half_anatomical_action": f"{high_endpoint}_endpoint_truncation",
    }


def to_lps(arr: np.ndarray, ref: sitk.Image) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = sitk.GetImageFromArray(np.asarray(arr, dtype=np.uint8))
    img.CopyInformation(ref)
    img = sitk.DICOMOrient(img, "LPS")
    out = sitk.GetArrayFromImage(img) > 0
    sx, sy, sz = [float(v) for v in img.GetSpacing()]
    return out, (sz, sy, sx)


def safe_fraction_corrected(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after) or before <= 1e-8:
        return np.nan
    return float((before - after) / before)


def endpoint_probes(mask, spacing, cfg, inferior_is_low_z):
    for family, levels in ENDPOINT_LEVELS.items():
        for level in levels:
            corrupted, meta = apply_forced_v4_endpoint_event(
                mask, spacing, family, level, inferior_is_low_z=inferior_is_low_z
            )
            yield "isolated", family, level, 0, corrupted, meta


def surface_probes(mask, spacing, cfg, base_seed: int, repeats: int):
    for family in SURFACE_MODES:
        for rep in range(repeats):
            rng = random.Random(base_seed + 1009 * rep + 7919 * (SURFACE_MODES.index(family) + 1))
            corrupted, meta = apply_forced_v4_surface_mode(mask, spacing, cfg, family, rng)
            yield "isolated", f"surface_{family}", np.nan, rep, corrupted, meta


def local_bite_probes(mask, spacing, cfg, base_seed: int, repeats: int):
    for rep in range(repeats):
        rng = random.Random(base_seed + 50021 + 1013 * rep)
        corrupted, meta = small_local_bite(mask, spacing, cfg.get("local_bite", {}), rng, cache_key=None)
        yield "isolated", "local_bite", np.nan, rep, corrupted, meta


def composite_endpoint_probes(mask, spacing, cfg, inferior_is_low_z, base_seed: int, repeats: int):
    weights = cfg.get("surface_mode_probabilities", {})
    names = list(weights)
    vals = np.asarray([float(weights[n]) for n in names], dtype=float)
    vals = vals / vals.sum()
    for family, levels in ENDPOINT_LEVELS.items():
        for level in levels:
            for rep in range(repeats):
                seed = base_seed + 70001 + 10007 * rep + 101 * int(round(level * 10)) + 997 * list(ENDPOINT_LEVELS).index(family)
                rng = random.Random(seed)
                draw = rng.random()
                acc = 0.0
                surface_mode = names[-1]
                for n, w in zip(names, vals):
                    acc += float(w)
                    if draw <= acc:
                        surface_mode = n; break
                surface, smeta = apply_forced_v4_surface_mode(mask, spacing, cfg, surface_mode, rng)
                corrupted, emeta = apply_forced_v4_endpoint_event(
                    surface, spacing, family, level, inferior_is_low_z=inferior_is_low_z
                )
                meta = {**smeta, **emeta, "diagnostic_surface_mode": surface_mode}
                yield "production_composite", family, level, rep, corrupted, meta


def summarize(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metrics = [
        "input_one_minus_dice", "recon_one_minus_dice", "dice_error_fraction_corrected",
        "input_hd99_mm", "recon_hd99_mm", "hd99_fraction_corrected",
        "input_max_abs_endpoint_error_mm", "recon_max_abs_endpoint_error_mm", "endpoint_fraction_corrected",
        "missing_correction_recall", "addition_precision", "excess_correction_recall", "removal_precision",
        "change_precision", "overcorrection_fraction_of_change",
    ]
    rows = []
    group_cols = ["probe_context", "family", "requested_length_mm", "model_key"]
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys)); row["n"] = len(g)
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"median_{m}"] = float(vals.median()) if len(vals) else np.nan
            row[f"p25_{m}"] = float(vals.quantile(.25)) if len(vals) else np.nan
            row[f"p75_{m}"] = float(vals.quantile(.75)) if len(vals) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "family_summary.csv", index=False)
    return out


def plot_metric(summary: pd.DataFrame, out_dir: Path, metric: str, title: str, filename: str) -> None:
    sub = summary[(summary["probe_context"] == "isolated") & summary["requested_length_mm"].notna()].copy()
    if sub.empty: return
    fig, ax = plt.subplots(figsize=(12, 6))
    families = list(ENDPOINT_LEVELS)
    x0 = 0
    ticks=[]; labels=[]
    for fam in families:
        sf = sub[sub["family"] == fam]
        levels = sorted(sf["requested_length_mm"].unique())
        for lev in levels:
            ss = sf[sf["requested_length_mm"] == lev]
            models = list(dict.fromkeys(sub["model_key"].astype(str)))
            offsets = np.linspace(-0.24, 0.24, max(1, len(models)))
            for j, model in enumerate(models):
                sm = ss[ss["model_key"] == model]
                if sm.empty: continue
                y = float(sm.iloc[0][f"median_{metric}"])
                ax.scatter(x0 + float(offsets[j]), y, label=model if x0 == 0 else None)
            ticks.append(x0); labels.append(f"{fam}\n{lev:g}mm"); x0 += 1
        x0 += .5
    ax.axhline(0, linewidth=1)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel(metric.replace("_", " ")); ax.set_title(title); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180); plt.close(fig)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, orient: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# SpinalCord V4 family correction diagnostic", ""]
    lines += [
        "This analysis uses the actual frozen V4 corruption primitives and applies matched synthetic inputs to every model listed in the study manifest.",
        "", "## Legacy cut-probe orientation", "",
    ]
    mapping = orient.groupby(["legacy_cut_upper_half_anatomical_action", "legacy_cut_lower_half_anatomical_action"]).size().reset_index(name="n")
    lines.append(mapping.to_markdown(index=False)); lines.append("")
    lines.append("The legacy names are retained only for provenance. The diagnostic report uses anatomical endpoint labels derived from NIfTI geometry.")
    lines += ["", "## Isolated endpoint families", ""]
    cols = ["family","requested_length_mm","model_key","n","median_endpoint_fraction_corrected","median_dice_error_fraction_corrected","median_hd99_fraction_corrected","median_missing_correction_recall","median_addition_precision","median_excess_correction_recall","median_removal_precision"]
    tab = summary[(summary.probe_context=="isolated") & summary.requested_length_mm.notna()][cols]
    lines.append(tab.to_markdown(index=False, floatfmt=".3f")); lines.append("")
    lines += ["## Surface/local-bite families", ""]
    surf = summary[(summary.probe_context=="isolated") & summary.requested_length_mm.isna()][["family","model_key","n","median_dice_error_fraction_corrected","median_hd99_fraction_corrected","median_change_precision"]]
    lines.append(surf.to_markdown(index=False, floatfmt=".3f")); lines.append("")
    lines += ["## Production-composite endpoint families", ""]
    comp = summary[summary.probe_context=="production_composite"][cols]
    lines.append(comp.to_markdown(index=False, floatfmt=".3f")); lines.append("")
    lines += [
        "## Interpretation guide", "",
        "- Strong isolated-family correction but weak actual-autoseg correction supports a synthetic-to-real/reference-frame problem.",
        "- Weak isolated-family correction means the trained denoiser has not learned that V4 family well, even on its own synthetic support.",
        "- High correction recall with low precision indicates the model changes the right region but also introduces substantial false correction.",
        "- Compare isolated versus production-composite rows to see whether surface perturbation interferes with endpoint restoration.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--compare-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["validation1","testing1"])
    p.add_argument("--surface-repeats", type=int, default=2)
    p.add_argument("--composite-repeats", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument("--max-patients-per-split", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    args=p.parse_args()

    root=Path(args.compare_root).expanduser().resolve(); out=Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest=root/"manifests/model_configs.tsv"; v4_cfg_path=root/"configs/v4_corruption.yaml"
    v4_train=yaml.safe_load(v4_cfg_path.read_text()); v4_cfg=v4_train["spinalcord_proposed_distribution"]
    device=torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model_rows=pd.read_csv(manifest, sep="\t")
    rows=[]; orientation=[]
    base=root/"actual_autoseg_inputs"/"approved_reference"/"SpinalCord"

    for model_i, (_, mr) in enumerate(model_rows.iterrows()):
        model_key=str(mr["model_key"]); model_label=str(mr["model_label"])
        cf=load_config(str(mr["config"]))
        cf.setdefault("paths", {})["trained_decoder_checkpoint"] = str(mr["checkpoint"])
        cf["device"] = str(device)
        model=make_model(cf, device)
        threshold=float(cf.get("evaluation", {}).get("threshold", 0.5))
        print(f"Loaded {model_label} on {device}")

        for split in args.splits:
            cases=sorted((base/split).glob("patient_*"))
            if args.max_patients_per_split>0: cases=cases[:args.max_patients_per_split]
            for ci, case in enumerate(cases):
                pid=case.name.replace("patient_", "", 1); approved_path=case/"approved.nii.gz"
                x, img=read_binary(approved_path, device); mask=x[0,0].detach().cpu().numpy()>0.5
                spacing_xyz=img.GetSpacing(); spacing=(float(spacing_xyz[2]),float(spacing_xyz[1]),float(spacing_xyz[0]))
                geom=geometry_si_mapping(img)
                if model_i == 0:
                    orientation.append({"patient_id":pid,"split":split,**geom})
                base_seed=args.seed + 100000*ci + sum(ord(c) for c in str(pid))
                probes=[]
                probes += list(endpoint_probes(mask,spacing,v4_cfg,geom["inferior_is_low_z"]))
                probes += list(surface_probes(mask,spacing,v4_cfg,base_seed,args.surface_repeats))
                probes += list(local_bite_probes(mask,spacing,v4_cfg,base_seed,args.surface_repeats))
                probes += list(composite_endpoint_probes(mask,spacing,v4_cfg,geom["inferior_is_low_z"],base_seed,args.composite_repeats))
                probes=[p for p in probes if not np.array_equal(p[4], mask)]
                raw_ct = None
                if bool((cf.get("spatial_conditioning") or {}).get("enabled", False)):
                    raw_root = (cf.get("spatial_conditioning") or {}).get("raw_data_root")
                    raw_path = raw_ct_path(raw_root, pid)
                    if not raw_path.is_file():
                        raise FileNotFoundError(raw_path)
                    raw_ct = sitk.ReadImage(str(raw_path))

                transformed=[]
                for pi, probe in enumerate(probes):
                    context,family,level,rep,corr,meta=probe
                    target_t=torch.from_numpy(mask.astype(np.float32))[None]
                    corr_t=torch.from_numpy(corr.astype(np.float32))[None]
                    target_t,corr_t,_changed,spatial_context,rf_meta=deployment_centered_training_pair(
                        target_t, corr_t, img, raw_ct, cf,
                        random.Random(base_seed + 900000 + pi), train=False,
                    )
                    transformed.append((
                        context,family,level,rep,
                        target_t[0].numpy()>0.5, corr_t[0].numpy()>0.5,
                        meta, spatial_context, rf_meta,
                    ))

                for b0 in range(0, len(transformed), max(1,args.batch_size)):
                    chunk=transformed[b0:b0+max(1,args.batch_size)]
                    xb=torch.stack([torch.from_numpy(p[5].astype(np.float32))[None] for p in chunk], dim=0).to(device)
                    contexts=[p[7] for p in chunk]
                    spatial_batch=None
                    if contexts and contexts[0] is not None:
                        spatial_batch=torch.from_numpy(np.stack(contexts).astype(np.float32)).to(device)
                    with torch.no_grad():
                        rec_batch=reconstruct_binary(model,xb,cf,threshold,spatial_context=spatial_batch)
                    for bi,(context,family,level,rep,target_arr,corr,meta,_spatial,rf_meta) in enumerate(chunk):
                        rec_arr=rec_batch[bi,0].detach().cpu().numpy()>0.5
                        app_lps,sp_lps=to_lps(target_arr,img)
                        corr_lps,_=to_lps(corr,img); rec_lps,_=to_lps(rec_arr,img)
                        input_m=pair_metrics(app_lps,corr_lps,sp_lps); recon_m=pair_metrics(app_lps,rec_lps,sp_lps); qa_m=pair_metrics(corr_lps,rec_lps,sp_lps)
                        decomp=correction_decomposition(app_lps,corr_lps,rec_lps)
                        row={"patient_id":pid,"split":split,"probe_context":context,"family":family,"requested_length_mm":level,"repeat":rep,"model_key":model_key,"model_label":model_label,**{f"meta_{k}":v for k,v in meta.items() if isinstance(v,(str,int,float,bool,np.integer,np.floating))},**{f"reference_{k}":v for k,v in rf_meta.items() if isinstance(v,(str,int,float,bool,np.integer,np.floating))}}
                        for pre,m in [("input_",input_m),("recon_",recon_m),("qa_",qa_m)]: row.update({pre+k:v for k,v in m.items()})
                        row.update(decomp)
                        row["dice_error_fraction_corrected"]=safe_fraction_corrected(input_m["one_minus_dice"],recon_m["one_minus_dice"])
                        row["assd_fraction_corrected"]=safe_fraction_corrected(input_m["assd_mm"],recon_m["assd_mm"])
                        row["hd95_fraction_corrected"]=safe_fraction_corrected(input_m["hd95_mm"],recon_m["hd95_mm"])
                        row["hd99_fraction_corrected"]=safe_fraction_corrected(input_m["hd99_mm"],recon_m["hd99_mm"])
                        row["endpoint_fraction_corrected"]=safe_fraction_corrected(input_m["max_abs_endpoint_error_mm"],recon_m["max_abs_endpoint_error_mm"])
                        rows.append(row)
                print(f"[{model_key} {split} {ci+1}/{len(cases)}] patient {pid}: {len(probes)} diagnostic probes")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    df=pd.DataFrame(rows); df.to_csv(out/"detailed_metrics.csv",index=False)
    odf=pd.DataFrame(orientation); odf.to_csv(out/"legacy_cut_orientation_audit.csv",index=False)
    summary=summarize(df,out)
    plot_metric(summary,out,"endpoint_fraction_corrected","Endpoint correction on isolated V4 endpoint families","endpoint_fraction_corrected.png")
    plot_metric(summary,out,"dice_error_fraction_corrected","Dice-error correction on isolated V4 endpoint families","dice_fraction_corrected.png")
    plot_metric(summary,out,"hd99_fraction_corrected","HD99-error correction on isolated V4 endpoint families","hd99_fraction_corrected.png")
    write_report(df,summary,odf,out)
    (out/"metadata.json").write_text(json.dumps({"splits":args.splits,"surface_repeats":args.surface_repeats,"composite_repeats":args.composite_repeats,"v4_config":str(v4_cfg_path),"legacy_probe_semantics":{"cut_upper_half":"removes low array-z endpoint of candidate bbox","cut_lower_half":"removes high array-z endpoint of candidate bbox"}},indent=2),encoding="utf-8")
    print(f"Wrote V4-family diagnostics to {out}")

if __name__=="__main__": main()
