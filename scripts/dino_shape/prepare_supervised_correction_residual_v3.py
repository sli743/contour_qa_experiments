#!/usr/bin/env python3
"""Prepare selective residual correction v3 diagnostic config.

Changes from residual_selective_v2:
- stronger region-normalized changed-region correction pressure
- lower global reconstruction weights
- slightly wider changed-region band for coherent boundary correction
- keep the same residual parameterization and preservation weight
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study-root', default='/scratch/shaohan/contour_QA/supervised_autoseg_correction')
    ap.add_argument('--roi', default='Bronchus')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--name', default='residual_selective_v3')
    args = ap.parse_args()

    root = Path(args.study_root)
    base_path = root / 'configs' / args.roi / f'fold_{args.fold}.yaml'
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    cfg = yaml.safe_load(base_path.read_text())

    out_root = root / 'residual_selective_v3' / args.roi / f'fold_{args.fold}'
    cfg['output_root'] = str(out_root / 'models')
    cfg['experiment_name'] = args.name
    cfg['correction'] = {
        'output_mode': 'residual_logits',
        'input_logit_magnitude': 2.0,
        'residual_logit_bound': 4.0,
    }
    cfg['loss']['global_dice_weight'] = 0.25
    cfg['loss']['global_bce_weight'] = 0.05
    cfg['loss']['changed_region_bce_weight'] = 0.60
    cfg['loss']['changed_region_reduction'] = 'region_mean'
    cfg['loss']['changed_region_dilation_voxels'] = 4
    cfg['loss']['unchanged_residual_l1_weight'] = 0.10
    cfg['training']['paired_augmentation'] = {
        'enabled': True,
        'probability': 0.8,
        'rotation_deg': 5.0,
        'translation_voxels': 6.0,
        'scale_min': 0.95,
        'scale_max': 1.05,
    }

    cfg_path = out_root / 'config.yaml'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(cfg_path)


if __name__ == '__main__':
    main()
