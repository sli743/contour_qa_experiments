#!/usr/bin/env python3
"""Compare selective residual v2 with direct-augmentation and residual-v1 diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS=['dice_gain','assd_reduction_mm','hd95_reduction_mm']
EXTRA=['correction_volume_fraction','unnecessary_change_fraction','needed_correction_recall','missing_correction_recall','excess_correction_recall']


def summarize(df,label):
    rows=[]
    for m in METRICS:
        x=pd.to_numeric(df[m],errors='coerce').to_numpy(float); x=x[np.isfinite(x)]
        rows.append({'model':label,'metric':m,'n':len(x),'mean':np.mean(x),'median':np.median(x),'fraction_improved':np.mean(x>0)})
    for m in EXTRA:
        if m in df:
            x=pd.to_numeric(df[m],errors='coerce').to_numpy(float); x=x[np.isfinite(x)]
            rows.append({'model':label,'metric':m,'n':len(x),'mean':np.mean(x) if len(x) else np.nan,'median':np.median(x) if len(x) else np.nan,'fraction_improved':np.nan})
    return rows


def stratified(df,label,q50,q75):
    y=pd.to_numeric(df['true_assd_mm'],errors='coerce')
    groups=[('low_0_50',y<=q50),('mid_50_75',(y>q50)&(y<=q75)),('high_75_100',y>q75)]
    out=[]
    for g,mask in groups:
        part=df.loc[mask]
        for m in METRICS:
            x=pd.to_numeric(part[m],errors='coerce').to_numpy(float); x=x[np.isfinite(x)]
            out.append({'model':label,'stratum':g,'metric':m,'n':len(x),
                        'mean':np.mean(x) if len(x) else np.nan,
                        'median':np.median(x) if len(x) else np.nan,
                        'fraction_improved':np.mean(x>0) if len(x) else np.nan,
                        'q50_true_assd_mm':q50,'q75_true_assd_mm':q75})
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study-root',default='/scratch/shaohan/contour_QA/supervised_autoseg_correction')
    ap.add_argument('--roi',default='Bronchus')
    ap.add_argument('--fold',type=int,default=0)
    args=ap.parse_args(); root=Path(args.study_root)

    out=root/'residual_selective_v2'/args.roi/f'fold_{args.fold}'
    v2_path=out/'evaluation'/'validation_metrics.csv'
    direct_path=root/'diagnostics'/args.roi/f'fold_{args.fold}'/'evaluation'/'augment_100'/'validation_metrics.csv'
    v1_path=root/'residual_diagnostics'/args.roi/f'fold_{args.fold}'/'evaluation'/'validation_metrics.csv'
    if not v2_path.is_file(): raise FileNotFoundError(v2_path)

    frames=[]
    if direct_path.is_file(): frames.append(('direct_aug100',pd.read_csv(direct_path)))
    if v1_path.is_file(): frames.append(('residual_preserve_v1',pd.read_csv(v1_path)))
    frames.append(('residual_selective_v2',pd.read_csv(v2_path)))

    ref=frames[-1][1]
    y=pd.to_numeric(ref['true_assd_mm'],errors='coerce')
    q50=float(y.quantile(.5)); q75=float(y.quantile(.75))
    summary=[]; strata=[]
    for label,df in frames:
        summary.extend(summarize(df,label)); strata.extend(stratified(df,label,q50,q75))

    sdf=pd.DataFrame(summary); tdf=pd.DataFrame(strata)
    sdf.to_csv(out/'comparison_summary.csv',index=False)
    tdf.to_csv(out/'comparison_by_true_assd_stratum.csv',index=False)
    print(sdf.to_string(index=False))
    print('\nASSD-stratified validation results:')
    print(tdf.query("metric == 'assd_reduction_mm'").to_string(index=False))
    (out/'summary_metadata.json').write_text(json.dumps({
        'roi':args.roi,'fold':args.fold,
        'direct_baseline_found':direct_path.is_file(),'direct_baseline_path':str(direct_path),
        'residual_v1_found':v1_path.is_file(),'residual_v1_path':str(v1_path),
        'residual_v2_path':str(v2_path),
        'q50_true_assd_mm':q50,'q75_true_assd_mm':q75,
    },indent=2))


if __name__=='__main__': main()
