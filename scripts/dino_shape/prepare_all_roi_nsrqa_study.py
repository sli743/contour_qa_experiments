#!/usr/bin/env python3
"""Validate all ROI datasets and materialize four-model configs/manifests.

ROIs missing any required training/validation/test labels or CT folder are
recorded and skipped. Generated configs live under the study root on scratch,
not in the git working tree.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import yaml

MODELS = ("pilot", "benchmark", "random_morphology", "nsrqa")

def nifti_count(p: Path) -> int:
    return sum(1 for _ in p.glob("*.nii.gz")) if p.is_dir() else 0

def data_dirs(data_root: Path, group: str, roi: str):
    out = {}
    for split in ("training1", "validation1", "testing1"):
        base = data_root / group / roi / split / group / roi
        out[split] = {"labels": base / "labels", "ct": base / "ct"}
    return out

def common_config(reg, roi, dirs):
    return {
        "seed": int(reg["seed"]),
        "dino": {
            "repo_path": "/home/shaohan/3DINO",
            "checkpoint_path": "/home/shaohan/3DINO/pretrained/3dino_vit_weights.pth",
            "config_name": "train/vit3d_highres",
        },
        "data": {
            "roi_name": roi,
            "train_label_dir": str(dirs["training1"]["labels"]),
            "validation_label_dir": str(dirs["validation1"]["labels"]),
            "test_label_dir": str(dirs["testing1"]["labels"]),
            "train_ct_dir": str(dirs["training1"]["ct"]),
            "validation_ct_dir": str(dirs["validation1"]["ct"]),
            "test_ct_dir": str(dirs["testing1"]["ct"]),
        },
        "decoder": {"channels": [256, 128, 64, 32]},
        "evaluation": {"threshold": 0.5, "spacing_zyx": [1.5, 1.5, 1.5], "device": "cuda:0", "num_workers": 2},
    }

def pilot_config(common, study_root, roi):
    c = dict(common)
    c.update({
        "experiment_name": f"dino_shape_pilot_{roi}_allroi",
        "output_root": str(study_root / "models" / "pilot"),
        "training": {"batch_size": 1, "accumulation_steps": 4, "num_workers": 2, "num_epochs": 150,
                     "early_stopping_patience": 30, "learning_rate": 1e-4, "weight_decay": 1e-4,
                     "grad_clip": 1.0, "precision": "bf16"},
        "loss": {"dice_weight": 0.8},
    })
    return c

def benchmark_config(common, study_root, roi, dilation_vox):
    c = dict(common)
    c.update({
        "experiment_name": f"dino_shape_benchmark_overlap_{roi}_allroi",
        "output_root": str(study_root / "models" / "benchmark_overlap"),
        "denoising": {
            "corruption_backend": "gpu_torch",
            "clean_probability": 0.20, "two_corruption_probability": 0.30,
            "corruptions": {
                "eroded": {"weight": 0.15, "voxels": [1, 5]},
                "dilated": {"weight": 0.15, "voxels": [1, 5]},
                "shifted": {"weight": 0.10, "voxels": [1, 5]},
                "cut_upper_half": {"weight": 0.10, "fraction": [0.05, 0.35]},
                "cut_lower_half": {"weight": 0.10, "fraction": [0.05, 0.35]},
                "cut_posterior_half": {"weight": 0.10, "fraction": [0.05, 0.30]},
                "big_bite": {"weight": 0.15, "fraction": [0.02, 0.20]},
                "big_blob": {"weight": 0.15, "fraction": [0.02, 0.20]},
            },
        },
        "training": {"device": "cuda:0", "batch_size": 1, "accumulation_steps": 4, "num_workers": 2,
                     "num_epochs": 200, "early_stopping_patience": 40, "learning_rate": 1e-4,
                     "weight_decay": 1e-4, "grad_clip": 1.0, "precision": "bf16"},
        "loss": {"global_dice_weight": 0.50, "global_bce_weight": 0.10,
                 "changed_region_bce_weight": 0.40, "changed_region_voxel_weight": 8.0,
                 "changed_region_dilation_voxels": int(dilation_vox)},
        "study_metadata": {"role": "benchmark_overlap_positive_control"},
    })
    return c

def random_config(common, study_root, roi, profile):
    rm = profile["random_morphology"]
    c = dict(common)
    c.update({
        "experiment_name": f"dino_shape_random_morphology_{roi}_allroi",
        "output_root": str(study_root / "models" / "random_morphology"),
        "random_morphology": {
            "clean_probability": 0.20,
            "morphology": {"probability": 0.80, "erosion_probability": 0.50, "iterations": rm["morphology_iterations"]},
            "translation": {"probability": 0.50, "voxels": rm["translation_voxels"]},
            "local_ball": {"probability": 0.40, "bite_probability": 0.50, "radius_voxels": rm["local_ball_radius_voxels"]},
        },
        "training": {"device": "cuda:0", "batch_size": 1, "accumulation_steps": 4, "num_workers": 2,
                     "num_epochs": 200, "early_stopping_patience": 40, "learning_rate": 1e-4,
                     "weight_decay": 1e-4, "grad_clip": 1.0, "precision": "bf16"},
        "loss": {"global_dice_weight": 0.50, "global_bce_weight": 0.10,
                 "changed_region_bce_weight": 0.40, "changed_region_voxel_weight": 8.0,
                 "changed_region_dilation_voxels": int(profile["changed_region_dilation_voxels"])},
        "study_metadata": {"role": "partial_overlap_random_morphology_control"},
    })
    return c

def nsrqa_config(common, study_root, roi, profile_name, profile):
    c = dict(common)
    c.update({
        "experiment_name": f"dino_shape_NSRQA_{roi}_allroi",
        "output_root": str(study_root / "models" / "nsrqa"),
        "stochastic_region_dropout": {
            "implementation_version": "fast_v4_global_sdf_offset",
            "spacing_zyx": [1.5, 1.5, 1.5],
            "clean_probability": 0.20,
            "branch_probabilities": {"smooth_warp": 0.10, "distance_noise": 0.20, "combined": 0.15,
                                     "stochastic_region_dropout": 0.20, "global_boundary_offset": 0.15},
            **profile["stochastic"],
        },
        "training": {"device": "cuda:0", "batch_size": 1, "accumulation_steps": 4, "num_workers": 8,
                     "prefetch_factor": 2, "validation_cache": {"enabled": True}, "num_epochs": 200,
                     "early_stopping_patience": 40, "learning_rate": 1e-4, "weight_decay": 1e-4,
                     "grad_clip": 1.0, "precision": "bf16",
                     "timing": {"enabled": True, "sync_cuda": True, "log_every_steps": 10}},
        "loss": {"global_dice_weight": 0.40, "global_bce_weight": 0.10,
                 "changed_region_bce_weight": 0.50, "missing_region_bce_weight": 0.00,
                 "changed_region_voxel_weight": 8.0,
                 "changed_region_dilation_voxels": int(profile["changed_region_dilation_voxels"])},
        "study_metadata": {"role": "proposed_NSRQA", "anatomy_profile": profile_name,
                           "crop_localization": "approved_label_oracle",
                           "training_probe_separation": "generic_stochastic_corruptions_vs_deterministic_probe_bank"},
    })
    return c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--registry', default='configs/dino_shape/all_roi_nsrqa_registry.yaml')
    ap.add_argument('--study-root', default=None)
    ap.add_argument('--data-root', default=None)
    args=ap.parse_args()
    reg_path=Path(args.registry).expanduser().resolve()
    reg=yaml.safe_load(reg_path.read_text())
    study_root=Path(args.study_root or reg['study_root']).expanduser().resolve()
    data_root=Path(args.data_root or reg['data_root']).expanduser().resolve()
    manifest_dir=study_root/'manifests'; config_root=study_root/'configs'
    manifest_dir.mkdir(parents=True, exist_ok=True); config_root.mkdir(parents=True, exist_ok=True)
    active=[]; skipped=[]
    for roi_idx,(roi, rspec) in enumerate(reg['rois'].items()):
        group=rspec['data_group']; profile_name=rspec['profile']; profile=reg['profiles'][profile_name]
        dirs=data_dirs(data_root, group, roi)
        missing=[]
        for split, kinds in dirs.items():
            for kind,p in kinds.items():
                if not p.is_dir(): missing.append(f'{split}/{kind}:missing_dir')
                elif nifti_count(p)==0: missing.append(f'{split}/{kind}:no_nii_gz')
        if missing:
            skipped.append({'roi':roi,'data_group':group,'profile':profile_name,'reason':';'.join(missing)})
            continue
        common=common_config(reg,roi,dirs)
        configs={
            'pilot':pilot_config(common,study_root,roi),
            'benchmark':benchmark_config(common,study_root,roi,profile['changed_region_dilation_voxels']),
            'random_morphology':random_config(common,study_root,roi,profile),
            'nsrqa':nsrqa_config(common,study_root,roi,profile_name,profile),
        }
        roi_cfg_dir=config_root/roi; roi_cfg_dir.mkdir(parents=True,exist_ok=True)
        paths={}
        for model,cfg in configs.items():
            p=roi_cfg_dir/f'{model}.yaml'; p.write_text(yaml.safe_dump(cfg,sort_keys=False)); paths[model]=str(p)
        active.append({'roi_index':len(active),'registry_index':roi_idx,'roi':roi,'data_group':group,'profile':profile_name,
                       'global_bad_volume_pct':profile['global_bad_volume_pct'],'local_bad_hd95_mm':profile['local_bad_hd95_mm'],**{f'{m}_config':p for m,p in paths.items()}})
    with open(manifest_dir/'active_rois.tsv','w',newline='') as f:
        fields=list(active[0].keys()) if active else ['roi_index','registry_index','roi','data_group','profile','global_bad_volume_pct','local_bad_hd95_mm','pilot_config','benchmark_config','random_morphology_config','nsrqa_config']
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(active)
    with open(manifest_dir/'skipped_rois.tsv','w',newline='') as f:
        fields=['roi','data_group','profile','reason']; w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(skipped)
    tasks=[]
    for r in active:
        for model in MODELS:
            tasks.append({'task_index':len(tasks),'roi':r['roi'],'model':model,'config':r[f'{model}_config']})
    with open(manifest_dir/'training_tasks.tsv','w',newline='') as f:
        fields=['task_index','roi','model','config']; w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(tasks)
    summary={'registry':str(reg_path),'study_root':str(study_root),'data_root':str(data_root),'n_active_rois':len(active),'n_skipped_rois':len(skipped),'n_training_tasks':len(tasks),'active_rois':[r['roi'] for r in active],'skipped_rois':skipped}
    (manifest_dir/'preflight_summary.json').write_text(json.dumps(summary,indent=2))
    md=['# All-ROI NSR-QA preflight','',f"Active ROIs: **{len(active)}**",f"Skipped ROIs: **{len(skipped)}**",'', '## Active ROI settings','', '| ROI | profile | global bad volume | local bad HD95 |','|---|---|---:|---:|']
    for r in active: md.append(f"| {r['roi']} | {r['profile']} | >{r['global_bad_volume_pct']}% | >{r['local_bad_hd95_mm']} mm |")
    if skipped:
        md += ['', '## Skipped because required data are incomplete','', '| ROI | reason |','|---|---|']
        for r in skipped: md.append(f"| {r['roi']} | {r['reason']} |")
    (manifest_dir/'preflight_summary.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))
    print(f"\nGenerated configs: {config_root}")
    print(f"Training tasks: {manifest_dir/'training_tasks.tsv'}")

if __name__=='__main__': main()
