#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

METRICS = {
    "one_minus_dice": ("1 - Dice(input,recon)", lambda d: 1.0 - pd.to_numeric(d["dice_input_vs_recon"], errors="coerce")),
    "hd95": ("HD95(input,recon)", lambda d: pd.to_numeric(d["hd95_input_vs_recon"], errors="coerce")),
    "assd": ("ASSD(input,recon)", lambda d: pd.to_numeric(d["assd_input_vs_recon"], errors="coerce")),
    "rel_volume": ("Relative volume difference(input,recon)", lambda d: (pd.to_numeric(d["input_voxels"], errors="coerce")-pd.to_numeric(d["recon_voxels"], errors="coerce")).abs()/pd.to_numeric(d["input_voxels"], errors="coerce").replace(0,np.nan)),
}


def parse_models(items):
    out=[]
    for item in items:
        if "=" not in item: raise ValueError(f"Expected NAME=CSV, got {item}")
        n,p=item.split("=",1); p=Path(p).expanduser()
        if not p.is_file(): raise FileNotFoundError(p)
        out.append((n,p))
    return out


def clean_mask(s): return s.astype(str).str.lower().eq("clean")
def shift_mask(s): return s.astype(str).str.contains("shift|translation",case=False,regex=True,na=False)


def normalize_columns(df: pd.DataFrame, model_name: str, path: Path) -> pd.DataFrame:
    """Normalize legacy/new probe-sweep schemas and derive repair metrics when absent."""
    out = df.copy()
    required = {
        "patient_id", "probe_type", "input_voxels", "target_voxels", "recon_voxels",
        "dice_input_vs_target", "dice_input_vs_recon", "dice_recon_vs_target",
        "hd95_input_vs_target", "hd95_input_vs_recon", "hd95_recon_vs_target",
        "assd_input_vs_target", "assd_input_vs_recon", "assd_recon_vs_target",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(
            f"{model_name}: {path} is missing required base columns: {missing}. "
            f"Available columns: {sorted(out.columns.tolist())}"
        )

    numeric_cols = sorted(required - {"patient_id", "probe_type"})
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Older pilot/random-morphology CSVs may not store these derived repair fields.
    if "dice_correction_gain" not in out.columns:
        out["dice_correction_gain"] = out["dice_recon_vs_target"] - out["dice_input_vs_target"]
    else:
        out["dice_correction_gain"] = pd.to_numeric(out["dice_correction_gain"], errors="coerce")

    if "hd95_improvement" not in out.columns:
        out["hd95_improvement"] = out["hd95_input_vs_target"] - out["hd95_recon_vs_target"]
    else:
        out["hd95_improvement"] = pd.to_numeric(out["hd95_improvement"], errors="coerce")

    if "assd_improvement" not in out.columns:
        out["assd_improvement"] = out["assd_input_vs_target"] - out["assd_recon_vs_target"]
    else:
        out["assd_improvement"] = pd.to_numeric(out["assd_improvement"], errors="coerce")

    return out


def prepare(df, threshold, exclude_shifts):
    if exclude_shifts: df=df.loc[~shift_mask(df["probe_type"])].copy()
    clean=df.loc[clean_mask(df["probe_type"])].drop_duplicates("patient_id")
    df=pd.concat([clean,df.loc[~clean_mask(df["probe_type"])]],ignore_index=True)
    iv=pd.to_numeric(df["input_voxels"],errors="coerce"); tv=pd.to_numeric(df["target_voxels"],errors="coerce").replace(0,np.nan)
    y=(100*(iv-tv).abs()/tv > threshold).astype(float); y.loc[clean_mask(df["probe_type"])]=0
    return df,y


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",action="append",required=True,metavar="NAME=CSV")
    p.add_argument("--roi",required=True); p.add_argument("--volume-threshold-pct",type=float,required=True)
    p.add_argument("--exclude-shifts",action="store_true"); p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); plot_dir=a.output_dir/"plots"; plot_dir.mkdir(exist_ok=True)
    models=parse_models(a.model)
    auc_rows=[]; clean_rows=[]; repair_rows=[]
    prepared={}
    for name,path in models:
        raw=normalize_columns(pd.read_csv(path), name, path); df,y=prepare(raw,a.volume_threshold_pct,a.exclude_shifts); prepared[name]=(df,y)
        c=raw.loc[clean_mask(raw["probe_type"])].drop_duplicates("patient_id")
        clean_rows.append({"model":name,"n":len(c),"clean_recon_target_dice_mean":pd.to_numeric(c["dice_recon_vs_target"],errors="coerce").mean(),"clean_recon_target_dice_median":pd.to_numeric(c["dice_recon_vs_target"],errors="coerce").median()})
        r=raw.loc[~clean_mask(raw["probe_type"])].copy()
        if a.exclude_shifts: r=r.loc[~shift_mask(r["probe_type"])]
        for probe,sub in r.groupby("probe_type"):
            repair_rows.append({"model":name,"probe_type":probe,"n":len(sub),"dice_gain_median":pd.to_numeric(sub["dice_correction_gain"],errors="coerce").median(),"fraction_dice_improved":float((pd.to_numeric(sub["dice_correction_gain"],errors="coerce")>0).mean()),"hd95_improvement_median":pd.to_numeric(sub["hd95_improvement"],errors="coerce").replace([np.inf,-np.inf],np.nan).median(),"assd_improvement_median":pd.to_numeric(sub["assd_improvement"],errors="coerce").replace([np.inf,-np.inf],np.nan).median()})

    for key,(label,fn) in METRICS.items():
        fig,ax=plt.subplots(figsize=(7.2,6.2))
        for name,_ in models:
            df,y=prepared[name]; score=fn(df); valid=y.notna() & score.notna() & np.isfinite(score)
            yy=y.loc[valid].astype(int).to_numpy(); ss=score.loc[valid].to_numpy(float)
            if np.unique(yy).size<2: continue
            fpr,tpr,_=roc_curve(yy,ss); auc=float(roc_auc_score(yy,ss)); auc_rows.append({"model":name,"metric":key,"label":label,"auc":auc,"n_positive":int(yy.sum()),"n_negative":int((yy==0).sum())})
            ax.plot(fpr,tpr,linewidth=2,label=f"{name} (AUC={auc:.3f})")
        ax.plot([0,1],[0,1],"--",linewidth=1); ax.set(xlabel="False positive rate",ylabel="True positive rate",title=f"{a.roi}: {label}"); ax.legend(loc="lower right",fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(plot_dir/f"roc_{key}.png",dpi=250); fig.savefig(plot_dir/f"roc_{key}.pdf"); plt.close(fig)

    aucdf=pd.DataFrame(auc_rows); clean=pd.DataFrame(clean_rows); repair=pd.DataFrame(repair_rows)
    aucdf.to_csv(a.output_dir/"roc_summary_four_models.csv",index=False); clean.to_csv(a.output_dir/"clean_preservation.csv",index=False); repair.to_csv(a.output_dir/"probe_repair_by_model.csv",index=False)
    pivot=aucdf.pivot(index="model",columns="metric",values="auc")
    fig,ax=plt.subplots(figsize=(9,5.5)); pivot.plot(kind="bar",ax=ax); ax.set_ylabel("AUROC"); ax.set_ylim(0,1); ax.set_title(f"{a.roi}: detector AUROC by model and metric"); ax.legend(title="Metric",fontsize=8); fig.tight_layout(); fig.savefig(plot_dir/"auc_comparison.png",dpi=250); fig.savefig(plot_dir/"auc_comparison.pdf"); plt.close(fig)

    best=aucdf.sort_values("auc",ascending=False).groupby("model",as_index=False).first()[["model","label","auc"]]
    md=[f"# {a.roi}: four-model probe benchmark comparison","",f"Failure definition: absolute input-vs-target volume change > {a.volume_threshold_pct:g}%"+("; shift probes excluded." if a.exclude_shifts else "."),"","Models: Pilot AE, benchmark-leakage DAE, random-morphology DAE, and stochastic-region-dropout DAE.","","## Best detector per model","",best.to_markdown(index=False,floatfmt=".3f"),"","## AUROC by metric","",aucdf.pivot(index="model",columns="label",values="auc").to_markdown(floatfmt=".3f"),"","![](plots/auc_comparison.png)","","## Clean-contour preservation","",clean.to_markdown(index=False,floatfmt=".3f"),"","## Probe correction behavior","",repair.to_markdown(index=False,floatfmt=".3f"),""]
    for key,(label,_) in METRICS.items(): md += [f"## ROC: {label}","",f"![](plots/roc_{key}.png)",""]
    md += ["## Interpretation note","","The benchmark-leakage DAE was trained using benchmark-style corruption families, whereas the stochastic-region-dropout DAE uses benchmark-independent stochastic missing-region corruption. Detection AUROC and correction behavior should therefore be interpreted together rather than treating the benchmark-leakage result as an unbiased generalization estimate.",""]
    (a.output_dir/"four_model_comparison.md").write_text("\n".join(md)); print(f"Wrote {a.output_dir/'four_model_comparison.md'}")

if __name__=="__main__": main()
