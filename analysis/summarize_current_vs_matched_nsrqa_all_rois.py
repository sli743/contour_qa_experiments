#!/usr/bin/env python3
"""Macro-summarize current-vs-previous-NSR-QA five-fold comparisons across ROIs."""
from __future__ import annotations
import argparse,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd


def macro(df,groups,cols):
    rows=[]
    for keys,g in df.groupby(groups,dropna=False,sort=False):
        if not isinstance(keys,tuple):keys=(keys,)
        r=dict(zip(groups,keys));r['n_rois']=int(g.roi.nunique())
        for c in cols:
            v=pd.to_numeric(g[c],errors='coerce').dropna().to_numpy(float)
            r[c+'_mean']=float(np.mean(v)) if len(v) else np.nan;r[c+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else np.nan;r[c+'_median']=float(np.median(v)) if len(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study-root',default='/scratch/shaohan/contour_QA/supervised_autoseg_correction');ap.add_argument('--all-roi-root',default='/scratch/shaohan/contour_QA/all_roi_NSRQA_study');ap.add_argument('--variant',default='large_decoder_ft_last2');ap.add_argument('--rois',default=None);ap.add_argument('--output-dir',default=None);args=ap.parse_args()
    root=Path(args.study_root);active=pd.read_csv(Path(args.all_roi_root)/'manifests'/'active_rois.tsv',sep='\t')
    if args.rois:
        wanted=[x.strip() for x in args.rois.split(',') if x.strip()];active=active[active.roi.isin(wanted)]
    out=Path(args.output_dir) if args.output_dir else root/'current_vs_previous_nsrqa_all_rois'/args.variant;out.mkdir(parents=True,exist_ok=True)
    regs=[];clss=[];pairs=[];missing=[]
    for meta in active.to_dict('records'):
        d=root/'current_vs_previous_nsrqa_5fold'/meta['roi']/args.variant
        files=[d/'current_vs_nsrqa_regression_summary.csv',d/'current_vs_nsrqa_classification_summary.csv',d/'current_vs_nsrqa_paired_differences.csv']
        if not all(p.is_file() for p in files):missing.append(meta['roi']);continue
        for arr,p in ((regs,files[0]),(clss,files[1]),(pairs,files[2])):
            x=pd.read_csv(p);x['profile']=meta.get('profile','');x['data_group']=meta.get('data_group','');arr.append(x)
    if missing:raise FileNotFoundError('Missing ROI comparisons: '+', '.join(missing))
    R=pd.concat(regs,ignore_index=True);C=pd.concat(clss,ignore_index=True);P=pd.concat(pairs,ignore_index=True)
    R.to_csv(out/'current_vs_nsrqa_regression_by_roi.csv',index=False);C.to_csv(out/'current_vs_nsrqa_classification_by_roi.csv',index=False);P.to_csv(out/'current_vs_nsrqa_paired_differences_by_roi.csv',index=False)
    Cp=C[C.bad_percentile==75].copy()
    macro(Cp,['method','target_metric'],['fold_roc_auc_mean','pooled_roc_auc','pooled_pr_auc','pooled_sensitivity','pooled_specificity']).to_csv(out/'classification_macro_across_rois.csv',index=False)
    macro(R,['method','target_metric'],['fold_spearman_mean','pooled_spearman','pooled_r2','pooled_mae']).to_csv(out/'regression_macro_across_rois.csv',index=False)
    macro(Cp,['profile','method','target_metric'],['fold_roc_auc_mean','pooled_roc_auc']).to_csv(out/'classification_by_anatomy_profile.csv',index=False)
    macro(R,['profile','method','target_metric'],['fold_spearman_mean','pooled_spearman','pooled_r2']).to_csv(out/'regression_by_anatomy_profile.csv',index=False)
    (out/'README.md').write_text('# Current method vs previous NSR-QA\n\nEach ROI is compared with identical five outer folds and matched preprocessing. Cross-ROI summaries macro-average ROI-level metrics; patients are never pooled across organs.\n')
    (out/'protocol.json').write_text(json.dumps({'variant':args.variant,'n_rois':int(active.roi.nunique()),'rois':active.roi.tolist(),'primary_binary_percentile':75,'patient_pooling_across_rois':False},indent=2))
    z=out.parent/f'{args.variant}_current_vs_previous_nsrqa_all_roi_results_no_models.zip'
    if z.exists():z.unlink()
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob('*'):
            if p.is_file():zf.write(p,p.relative_to(out))
    print(f'Wrote {out}\nWrote {z}')
if __name__=='__main__':main()
