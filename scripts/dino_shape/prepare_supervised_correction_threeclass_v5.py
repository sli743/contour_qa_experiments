#!/usr/bin/env python3
"""Prepare the v5 preserve/add/remove supervised correction experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study-root', default='/scratch/shaohan/contour_QA/supervised_autoseg_correction')
    ap.add_argument('--roi', default='Bronchus')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--name', default='threeclass_correction_v5')
    args = ap.parse_args()

    root = Path(args.study_root)
    base_path = root / 'configs' / args.roi / f'fold_{args.fold}.yaml'
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    cfg = yaml.safe_load(base_path.read_text())

    out_root = root / 'threeclass_correction_v5' / args.roi / f'fold_{args.fold}'
    cfg['output_root'] = str(out_root / 'models')
    cfg['experiment_name'] = args.name
    cfg.pop('correction', None)
    cfg['loss'] = {
        'type': 'class_balanced_action_ce',
        'preserve_ce_weight': 0.20,
        'add_ce_weight': 0.40,
        'remove_ce_weight': 0.40,
    }
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
