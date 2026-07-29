#!/usr/bin/env python3
"""Train a binary ROI decoder on top of a frozen pretrained 3DINO encoder."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_id_from_name(name: str) -> str:
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    patterns = [
        r"(?:label|mask|seg|proj|ct)_patient_?(.+)$",
        r"patient_?(.+?)(?:_(?:label|mask|seg|approved|ct))?$",
    ]
    for pattern in patterns:
        m = re.search(pattern, stem, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    numbers = re.findall(r"\d+", stem)
    return numbers[-1] if numbers else stem


def list_nifti_files(directory: str | Path):
    directory = Path(directory)
    files = sorted(directory.rglob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found under {directory}")
    return files


class BinaryROIDataset(Dataset):
    def __init__(self, directory: str | Path):
        self.files = list_nifti_files(directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(image)
        mask = (arr > 0).astype(np.float32)
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask, got {mask.shape}: {path}")
        return {
            "mask": torch.from_numpy(mask)[None],
            "patient_id": patient_id_from_name(path.name),
            "path": str(path),
        }


def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    intersection = torch.sum(prob * target, dim=dims)
    denominator = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, target, dice_weight=0.8):
    dice = dice_loss_from_logits(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    total = float(dice_weight) * dice + (1.0 - float(dice_weight)) * bce
    return total, {"total": total, "dice_loss": dice, "bce": bce}


def amp_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def run_epoch(model, loader, optimizer, device, cfg, train):
    model.train(train)
    totals = {"total": 0.0, "dice_loss": 0.0, "bce": 0.0, "n": 0}
    accumulation = int(cfg["training"].get("accumulation_steps", 1))
    precision = cfg["training"].get("precision", "bf16")

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        target = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with amp_context(device, precision):
                logits, _ = model(target)
                loss, pieces = combined_loss(
                    logits, target,
                    dice_weight=float(cfg["loss"].get("dice_weight", 0.8)),
                )
                backward = loss / accumulation
            if train:
                backward.backward()
                if (step + 1) % accumulation == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(
                        model.decoder.parameters(),
                        float(cfg["training"].get("grad_clip", 1.0)),
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        n = target.shape[0]
        totals["n"] += n
        for key in ("total", "dice_loss", "bce"):
            totals[key] += float(pieces[key].detach().float()) * n

    return {k: totals[k] / totals["n"] for k in ("total", "dice_loss", "bce")}


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed_all(int(cfg.get("seed", 88)))

    experiment_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
    ckpt_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_ds = BinaryROIDataset(cfg["data"]["train_label_dir"])
    val_ds = BinaryROIDataset(cfg["data"]["validation_label_dir"])
    workers = int(cfg["training"].get("num_workers", 2))
    batch_size = int(cfg["training"].get("batch_size", 1))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Frozen3DINOShapeAutoencoder(
        repo_path=cfg["dino"]["repo_path"],
        checkpoint_path=cfg["dino"]["checkpoint_path"],
        config_name=cfg["dino"].get("config_name", "train/vit3d_highres"),
        decoder_channels=cfg["decoder"].get("channels", [256, 128, 64, 32]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Train/validation cases: {len(train_ds)}/{len(val_ds)}")
    print(f"Frozen encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Trainable decoder params: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad):,}")
    print(f"3DINO patch size: {model.encoder.patch_size}; input channels: {model.encoder.in_channels}")

    history_path = log_dir / "history.csv"
    best = float("inf")
    epochs = int(cfg["training"].get("num_epochs", 150))
    patience = int(cfg["training"].get("early_stopping_patience", 30))
    bad_epochs = 0

    fields = [
        "epoch", "train_total", "train_dice_loss", "train_bce",
        "validation_total", "validation_dice_loss", "validation_bce"
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            tr = run_epoch(model, train_loader, optimizer, device, cfg, True)
            va = run_epoch(model, val_loader, optimizer, device, cfg, False)
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"validation_{k}": v for k, v in va.items()},
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps(row))

            state = {
                "epoch": epoch,
                "decoder_state_dict": model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_total": va["total"],
                "config": cfg,
            }
            torch.save(state, ckpt_dir / "last_decoder.pth")
            if va["total"] < best:
                best = va["total"]
                bad_epochs = 0
                torch.save(state, ckpt_dir / "best_decoder.pth")
                print(f"Saved best decoder: validation total={best:.6f}")
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break


if __name__ == "__main__":
    main()
