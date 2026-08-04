#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["dice_input_vs_recon", "hd95_input_vs_recon", "assd_input_vs_recon"]


def parse_models(items):
    out=[]
    for item in items:
        if "=" not in item: raise ValueError(f"Expected NAME=CSV: {item}")
        name,path=item.split("=",1); path=Path(path).expanduser()
        if not path.is_file(): raise FileNotFoundError(path)
        out.append((name,path))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",action="append",required=True,metavar="NAME=CSV")
    p.add_argument("--roi",required=True)
    p.add_argument("--volume-threshold-pct",type=float,required=True)
    p.add_argument("--exclude-shifts",action="store_true")
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

    probe_rows=[]; clean_rows=[]
    for model,path in parse_models(args.model):
        df=pd.read_csv(path)
        if args.exclude_shifts:
            df=df[~df["probe_type"].astype(str).str.contains("shift|translation",case=False,regex=True)].copy()
        clean=df[df["probe_type"].astype(str).str.lower().eq("clean")].drop_duplicates("patient_id")
        for metric in METRICS:
            if metric in clean:
                vals=pd.to_numeric(clean[metric],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
                clean_rows.append({"roi":args.roi,"model":model,"metric":metric,"n":len(vals),"mean":vals.mean(),"median":vals.median(),"p95":vals.quantile(.95)})
        for (probe,level),sub in df.groupby(["probe_type","level"],dropna=False):
            row={"roi":args.roi,"model":model,"probe_type":probe,"level":level,"n":len(sub)}
            for metric in METRICS:
                if metric in sub:
                    vals=pd.to_numeric(sub[metric],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
                    row[f"{metric}_median"]=vals.median() if len(vals) else np.nan
            probe_rows.append(row)

    pd.DataFrame(probe_rows).to_csv(args.output_dir/"probe_type_summary.csv",index=False)
    pd.DataFrame(clean_rows).to_csv(args.output_dir/"clean_preservation_summary.csv",index=False)
    metadata={
      "roi":args.roi,"include_shift":not args.exclude_shifts,
      "volume_threshold_pct":args.volume_threshold_pct,
      "crop_localization":"approved_label_oracle",
      "deployment_scope":"shape_QA_after_oracle_localization",
      "models":[name for name,_ in parse_models(args.model)],
      "score_definitions":{
        "dice":"1 - dice_input_vs_recon for ROC",
        "hd95":"hd95_input_vs_recon",
        "assd":"assd_input_vs_recon",
        "relative_volume_difference":"abs(input_voxels-recon_voxels)/input_voxels",
      },
    }
    (args.output_dir/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2))
    print(f"Saved summaries to {args.output_dir}")

if __name__=="__main__": main()
