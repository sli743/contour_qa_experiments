#!/usr/bin/env python3
"""Evaluate a three-class preserve/add/remove correction model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from evaluation.probes.eval_dino_shape_denoising_probe_sweep import (
    assd_binary,
    dice_score_binary,
    hd95_binary,
    volume_vox,
)
from evaluation.supervised_correction.eval_supervised_autoseg_correction import write_csv
from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder
from training.train_supervised_autoseg_correction_threeclass import (
    ADD,
    PRESERVE,
    REMOVE,
    apply_actions,
)


def rows_for(path, split):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def read_binary(path, device):
    img = sitk.ReadImage(path)
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.float32)
    return torch.from_numpy(arr[None, None]).to(device), img


def relvol(a, b):
    return abs(a - b) / b if b > 0 else float("nan")


def make_model(cfg, device):
    model = Frozen3DINOShapeAutoencoder(
        cfg["dino"]["repo_path"],
        cfg["dino"]["checkpoint_path"],
        cfg["dino"].get("config_name", "train/vit3d_highres"),
        cfg["decoder"].get("channels", [256, 128, 64, 32]),
        output_channels=3,
        decoder_blocks_per_stage=int(cfg["decoder"].get("blocks_per_stage", 1)),
        finetune_last_n_blocks=int(cfg.get("transfer_learning", {}).get("finetune_last_n_blocks", 0)),
    ).to(device)
    ckpt = Path(cfg["output_root"]) / cfg["experiment_name"] / "checkpoints" / "best_decoder.pth"
    state = torch.load(ckpt, map_location=device)
    model.decoder.load_state_dict(state["decoder_state_dict"])
    encoder_state = state.get("encoder_trainable_state_dict", {})
    if encoder_state:
        model.encoder.load_trainable_state_dict(encoder_state)
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_rows(model, rows, device):
    results = []
    for index, row in enumerate(rows, 1):
        x, ref = read_binary(row["autoseg"], device)
        target, _ = read_binary(row["approved"], device)
        logits, _ = model(x)
        actions = logits.argmax(dim=1, keepdim=True)
        corrected = apply_actions(x, logits)
        spacing = tuple(float(v) for v in ref.GetSpacing()[::-1])

        dit = float(dice_score_binary(x, target)[0])
        dir_ = float(dice_score_binary(x, corrected)[0])
        drt = float(dice_score_binary(corrected, target)[0])
        hit = float(hd95_binary(x, target, spacing)[0])
        hir = float(hd95_binary(x, corrected, spacing)[0])
        hrt = float(hd95_binary(corrected, target, spacing)[0])
        ait = float(assd_binary(x, target, spacing)[0])
        air = float(assd_binary(x, corrected, spacing)[0])
        art = float(assd_binary(corrected, target, spacing)[0])
        vin = float(volume_vox(x)[0])
        vr = float(volume_vox(corrected)[0])
        vt = float(volume_vox(target)[0])

        changed = (x > 0.5) != (target > 0.5)
        unchanged = ~changed
        pred_changed = corrected != x
        nonpreserve_actions = actions != PRESERVE
        true_missing = (target > 0.5) & (x <= 0.5)
        true_excess = (x > 0.5) & (target <= 0.5)

        def frac(num, den):
            d = float(den.sum())
            return float(num.sum()) / d if d > 0 else float("nan")

        result = {
            "patient_id": row["patient_id"],
            "split": row["split"],
            "outer_fold": row["outer_fold"],
            "cv_bin": row["cv_bin"],
            "true_1_minus_dice": 1 - dit,
            "true_hd95_mm": hit,
            "true_assd_mm": ait,
            "true_rel_volume_diff": relvol(vin, vt),
            "qa_1_minus_dice": 1 - dir_,
            "qa_hd95_mm": hir,
            "qa_assd_mm": air,
            "qa_rel_volume_diff": relvol(vin, vr),
            "corrected_1_minus_dice": 1 - drt,
            "corrected_hd95_mm": hrt,
            "corrected_assd_mm": art,
            "corrected_rel_volume_diff": relvol(vr, vt),
            "dice_gain": drt - dit,
            "hd95_reduction_mm": hit - hrt,
            "assd_reduction_mm": ait - art,
            "correction_volume_fraction": frac(pred_changed, x > 0.5),
            "unnecessary_change_fraction": frac(pred_changed & unchanged, unchanged),
            "needed_correction_recall": frac(pred_changed & changed, changed),
            "missing_correction_recall": frac((actions == ADD) & true_missing, true_missing),
            "excess_correction_recall": frac((actions == REMOVE) & true_excess, true_excess),
            "predicted_add_fraction": frac(actions == ADD, torch.ones_like(actions, dtype=torch.bool)),
            "predicted_remove_fraction": frac(actions == REMOVE, torch.ones_like(actions, dtype=torch.bool)),
            "nonpreserve_action_fraction": frac(nonpreserve_actions, torch.ones_like(actions, dtype=torch.bool)),
            "case_dir": row["case_dir"],
        }
        results.append(result)
        print(
            f"[{index}/{len(rows)}] patient {row['patient_id']} "
            f"trueASSD={ait:.3f} qaASSD={air:.3f} correctedASSD={art:.3f} "
            f"addRecall={result['missing_correction_recall']:.3f} "
            f"removeRecall={result['excess_correction_recall']:.3f}"
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(
        cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )
    model, ckpt = make_model(cfg, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_rows = evaluate_rows(model, rows_for(cfg["manifest"], "train"), device)
    val_rows = evaluate_rows(model, rows_for(cfg["manifest"], "validation"), device)
    test_rows = evaluate_rows(model, rows_for(cfg["manifest"], "test"), device)
    write_csv(output / "train_metrics.csv", train_rows)
    write_csv(output / "validation_metrics.csv", val_rows)
    write_csv(output / "test_metrics.csv", test_rows)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt),
                "train_n": len(train_rows),
                "validation_n": len(val_rows),
                "test_n": len(test_rows),
                "classes": {"preserve": PRESERVE, "add": ADD, "remove": REMOVE},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
