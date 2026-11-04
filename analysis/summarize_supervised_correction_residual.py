#!/usr/bin/env python3
"""Compare residual+preservation diagnostic with the existing augment_100 baseline."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS=['dice_gain','assd_reduction_mm','hd95_reduction_mm']

def summarize(df,label):
    rows=[]
    for m in METRICS:
        x=pd.to_numeric(df[m],errors='coerce').to_numpy(float); x=x[np.isfinite(x)]
        rows.append({'model':label,'metric':m,'n':len(x),'mean':np.mean(x),'median':np.median(x),'fraction_improved':np.mean(x>0)})
    for m in ['correction_volume_fraction','unnecessary_change_fraction','needed_correction_recall','missing_correction_recall','excess_correction_recall']:
        if m in df:
            x=pd.to_numeric(df[m],errors='coerce').to_numpy(float); x=x[np.isfinite(x)]
            rows.append({'model':label,'metric':m,'n':len(x),'mean':np.mean(x),'median':np.median(x),'fraction_improved':np.nan})
    return rows

def stratified(df,label):
    y=pd.to_numeric(df['true_assd_mm'],errors='coerce')
    q50=float(y.quantile(.5)); q75=float(y.quantile(.75))
    groups=[('low_0_50',y<=q50),('mid_50_75',(y>q50)&(y<=q75)),('high_75_100',y>q75)]
    out=[]
    for g,mask in groups:
        part=df.loc[mask]
        for m in METRICS:
            x=pd.to_numeric(part[m],errors='coerce').to_numpy(float); x=x[np.isfinite(x)]
            out.append({'model':label,'stratum':g,'metric':m,'n':len(x),'mean':np.mean(x) if len(x) else np.nan,'median':np.median(x) if len(x) else np.nan,'fraction_improved':np.mean(x>0) if len(x) else np.nan,'q50_true_assd_mm':q50,'q75_true_assd_mm':q75})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study-root',default='/scratch/shaohan/contour_QA/supervised_autoseg_correction')
    ap.add_argument('--roi',default='Bronchus'); ap.add_argument('--fold',type=int,default=0)
    args=ap.parse_args(); root=Path(args.study_root)
    rroot=root/'residual_diagnostics'/args.roi/f'fold_{args.fold}'
    residual_path=rroot/'evaluation'/'validation_metrics.csv'
    baseline_path=root/'diagnostics'/args.roi/f'fold_{args.fold}'/'evaluation'/'augment_100'/'validation_metrics.csv'
    if not residual_path.is_file(): raise FileNotFoundError(residual_path)
    residual=pd.read_csv(residual_path)
    frames=[('residual_preserve_aug',residual)]
    if baseline_path.is_file(): frames.insert(0,('direct_aug100',pd.read_csv(baseline_path)))
    summary=[]; strata=[]
    for label,df in frames:
        summary.extend(summarize(df,label)); strata.extend(stratified(df,label))
    pd.DataFrame(summary).to_csv(rroot/'comparison_summary.csv',index=False)
    pd.DataFrame(strata).to_csv(rroot/'comparison_by_true_assd_stratum.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print('\nASSD-stratified validation results:')
    print(pd.DataFrame(strata).query("metric == 'assd_reduction_mm'").to_string(index=False))
    (rroot/'summary_metadata.json').write_text(json.dumps({'roi':args.roi,'fold':args.fold,'baseline_found':baseline_path.is_file(),'baseline_path':str(baseline_path),'residual_path':str(residual_path)},indent=2))

if __name__=='__main__': main()
