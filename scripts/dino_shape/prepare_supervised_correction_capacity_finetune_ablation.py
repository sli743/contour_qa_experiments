#!/usr/bin/env python3
"""Prepare controlled decoder-capacity / partial-3DINO-finetuning ablations."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

VARIANTS = {
    "baseline": {
        "channels": [256, 128, 64, 32],
        "blocks_per_stage": 1,
        "finetune_last_n_blocks": 0,
    },
    "large_decoder": {
        "channels": [512, 256, 128, 64, 32],
        "blocks_per_stage": 2,
        "finetune_last_n_blocks": 0,
    },
    "partial_ft_last1": {
        "channels": [256, 128, 64, 32],
        "blocks_per_stage": 1,
        "finetune_last_n_blocks": 1,
    },
    "partial_ft_last2": {
        "channels": [256, 128, 64, 32],
        "blocks_per_stage": 1,
        "finetune_last_n_blocks": 2,
    },
    "large_decoder_ft_last1": {
        "channels": [512, 256, 128, 64, 32],
        "blocks_per_stage": 2,
        "finetune_last_n_blocks": 1,
    },
    "large_decoder_ft_last2": {
        "channels": [512, 256, 128, 64, 32],
        "blocks_per_stage": 2,
        "finetune_last_n_blocks": 2,
    },
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", default="/scratch/shaohan/contour_QA/supervised_autoseg_correction")
    ap.add_argument("--roi", default="Lung_L")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--action-weight", type=float, default=4.0)
    ap.add_argument("--encoder-lr", type=float, default=1e-6)
    args = ap.parse_args()

    root = Path(args.study_root)
    base_path = root / "configs" / args.roi / f"fold_{args.fold}.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    cfg = yaml.safe_load(base_path.read_text())
    v = VARIANTS[args.variant]

    exp_name = f"capacity_ft_ablation_{args.variant}"
    out_root = root / "capacity_finetune_ablation" / args.roi / f"fold_{args.fold}" / args.variant
    cfg["output_root"] = str(out_root / "models")
    cfg["experiment_name"] = exp_name
    cfg.pop("correction", None)
    cfg["loss"] = {
        "type": "weighted_action_ce",
        "preserve_class_weight": 1.0,
        "add_class_weight": float(args.action_weight),
        "remove_class_weight": float(args.action_weight),
    }
    cfg.setdefault("decoder", {})
    cfg["decoder"]["channels"] = list(v["channels"])
    cfg["decoder"]["blocks_per_stage"] = int(v["blocks_per_stage"])
    cfg["transfer_learning"] = {
        "finetune_last_n_blocks": int(v["finetune_last_n_blocks"]),
        "encoder_learning_rate": float(args.encoder_lr),
        "encoder_weight_decay": 1e-2,
    }
    cfg["training"]["paired_augmentation"] = {
        "enabled": True,
        "probability": 0.8,
        "rotation_deg": 5.0,
        "translation_voxels": 6.0,
        "scale_min": 0.95,
        "scale_max": 1.05,
    }
    cfg["ablation"] = {
        "variant": args.variant,
        "purpose": "decoder capacity and conservative partial 3DINO fine-tuning, including large-decoder encoder-depth ablation",
    }

    cfg_path = out_root / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(cfg_path)

if __name__ == "__main__":
    main()
