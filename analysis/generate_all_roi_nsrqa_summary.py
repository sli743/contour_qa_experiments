#!/usr/bin/env python3
"""Aggregate completed per-ROI four-model reports without pooling patients across ROIs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def fmt(x):
    return 'NA' if pd.isna(x) else f'{float(x):.3f}'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--study-root', default='/scratch/shaohan/contour_QA/all_roi_NSRQA_study'); ap.add_argument('--output-dir', default=None); args=ap.parse_args()
    study=Path(args.study_root).expanduser().resolve(); reports=study/'reports'; out=Path(args.output_dir).expanduser().resolve() if args.output_dir else study/'all_roi_summary'; out.mkdir(parents=True,exist_ok=True)
    pooled=[]; probe=[]
    for d in sorted(reports.glob('*_four_models_NSRQA_shared_probe')):
        p=d/'pooled'/'all_probes_metric_summary.csv'; q=d/'by_probe'/'all_probe_metric_summary.csv'
        if p.is_file(): pooled.append(pd.read_csv(p))
        if q.is_file(): probe.append(pd.read_csv(q))
    if not pooled: raise RuntimeError(f'No completed pooled reports under {reports}')
    P=pd.concat(pooled,ignore_index=True); P.to_csv(out/'all_roi_pooled_summary.csv',index=False)
    Q=pd.concat(probe,ignore_index=True) if probe else pd.DataFrame(); Q.to_csv(out/'all_roi_probe_summary.csv',index=False)
    assd=P[P['metric'].eq('assd_input_vs_recon')].copy()
    nsr=assd[assd['model'].eq('NSR-QA')].sort_values('roi')
    lines=['# All-ROI NSR-QA study summary','', 'ROIs are summarized at the ROI level; patients/probes are **not** merged into a single cross-organ ROC.','', '## NSR-QA pooled ASSD performance','', '| ROI | AUC (patient-cluster CI) | Sens @95% spec |','|---|---:|---:|']
    for _,r in nsr.iterrows(): lines.append(f"| {r.roi} | {fmt(r.auc)} [{fmt(r.auc_ci_low)}, {fmt(r.auc_ci_high)}] | {fmt(r.sens_at_95spec)} |")
    lines += ['', '## Four-model cross-ROI ASSD summary','', '| Model | median ROI AUC | IQR AUC | median ROI sens@95% spec |','|---|---:|---:|---:|']
    for model,g in assd.groupby('model',sort=False):
        q1,q3=np.nanquantile(g['auc'],[.25,.75]); lines.append(f"| {model} | {fmt(np.nanmedian(g['auc']))} | {fmt(q1)}–{fmt(q3)} | {fmt(np.nanmedian(g['sens_at_95spec']))} |")
    if not Q.empty:
        x=Q[(Q['model'].eq('NSR-QA')) & (Q['metric'].eq('assd_input_vs_recon'))]
        lines += ['', '## NSR-QA ASSD by probe family across ROIs','', '| Probe | median ROI AUC | median ROI sens@95% spec | n ROIs |','|---|---:|---:|---:|']
        for probe_name,g in x.groupby('probe_type',sort=True): lines.append(f"| {probe_name} | {fmt(np.nanmedian(g['auc']))} | {fmt(np.nanmedian(g['sens_at_95spec']))} | {g['roi'].nunique()} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); print(f'\nWrote {out}')
if __name__=='__main__': main()
