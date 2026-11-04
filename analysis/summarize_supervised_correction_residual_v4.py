#!/usr/bin/env python3
"""Compare directional residual v4 with the preceding correction models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ['dice_gain', 'assd_reduction_mm', 'hd95_reduction_mm']
EXTRA = [
    'correction_volume_fraction',
    'unnecessary_change_fraction',
    'needed_correction_recall',
    'missing_correction_recall',
    'excess_correction_recall',
]


def summarize(df, label):
    rows = []
    for metric in METRICS:
        x = pd.to_numeric(df[metric], errors='coerce').to_numpy(float)
        x = x[np.isfinite(x)]
        rows.append({
            'model': label, 'metric': metric, 'n': len(x),
            'mean': np.mean(x) if len(x) else np.nan,
            'median': np.median(x) if len(x) else np.nan,
            'fraction_improved': np.mean(x > 0) if len(x) else np.nan,
        })
    for metric in EXTRA:
        if metric not in df:
            continue
        x = pd.to_numeric(df[metric], errors='coerce').to_numpy(float)
        x = x[np.isfinite(x)]
        rows.append({
            'model': label, 'metric': metric, 'n': len(x),
            'mean': np.mean(x) if len(x) else np.nan,
            'median': np.median(x) if len(x) else np.nan,
            'fraction_improved': np.nan,
        })
    return rows


def stratified(df, label, q50, q75):
    y = pd.to_numeric(df['true_assd_mm'], errors='coerce')
    groups = [
        ('low_0_50', y <= q50),
        ('mid_50_75', (y > q50) & (y <= q75)),
        ('high_75_100', y > q75),
    ]
    rows = []
    for stratum, mask in groups:
        part = df.loc[mask]
        for metric in METRICS:
            x = pd.to_numeric(part[metric], errors='coerce').to_numpy(float)
            x = x[np.isfinite(x)]
            rows.append({
                'model': label, 'stratum': stratum, 'metric': metric, 'n': len(x),
                'mean': np.mean(x) if len(x) else np.nan,
                'median': np.median(x) if len(x) else np.nan,
                'fraction_improved': np.mean(x > 0) if len(x) else np.nan,
                'q50_true_assd_mm': q50, 'q75_true_assd_mm': q75,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study-root', default='/scratch/shaohan/contour_QA/supervised_autoseg_correction')
    ap.add_argument('--roi', default='Bronchus')
    ap.add_argument('--fold', type=int, default=0)
    args = ap.parse_args()
    root = Path(args.study_root)
    out = root / 'residual_directional_v4' / args.roi / f'fold_{args.fold}'

    paths = [
        ('direct_aug100', root / 'diagnostics' / args.roi / f'fold_{args.fold}' / 'evaluation' / 'augment_100' / 'validation_metrics.csv'),
        ('residual_preserve_v1', root / 'residual_diagnostics' / args.roi / f'fold_{args.fold}' / 'evaluation' / 'validation_metrics.csv'),
        ('residual_selective_v2', root / 'residual_selective_v2' / args.roi / f'fold_{args.fold}' / 'evaluation' / 'validation_metrics.csv'),
        ('residual_selective_v3', root / 'residual_selective_v3' / args.roi / f'fold_{args.fold}' / 'evaluation' / 'validation_metrics.csv'),
        ('residual_directional_v4', out / 'evaluation' / 'validation_metrics.csv'),
    ]
    if not paths[-1][1].is_file():
        raise FileNotFoundError(paths[-1][1])

    frames = [(label, pd.read_csv(path)) for label, path in paths if path.is_file()]
    ref = frames[-1][1]
    true_assd = pd.to_numeric(ref['true_assd_mm'], errors='coerce')
    q50 = float(true_assd.quantile(0.5))
    q75 = float(true_assd.quantile(0.75))

    summary_rows, stratum_rows = [], []
    for label, df in frames:
        summary_rows.extend(summarize(df, label))
        stratum_rows.extend(stratified(df, label, q50, q75))

    summary = pd.DataFrame(summary_rows)
    strata = pd.DataFrame(stratum_rows)
    summary.to_csv(out / 'comparison_summary.csv', index=False)
    strata.to_csv(out / 'comparison_by_true_assd_stratum.csv', index=False)
    print(summary.to_string(index=False))
    print('\nASSD-stratified validation results:')
    print(strata.query("metric == 'assd_reduction_mm'").to_string(index=False))

    metadata = {
        'roi': args.roi, 'fold': args.fold,
        'q50_true_assd_mm': q50, 'q75_true_assd_mm': q75,
        'models': {label: {'found': path.is_file(), 'path': str(path)} for label, path in paths},
    }
    (out / 'summary_metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
