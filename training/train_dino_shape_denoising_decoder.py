#!/usr/bin/env python3
"""Train a denoising contour decoder on top of a frozen pretrained 3DINO encoder.

Training input: online-corrupted approved contour.
Training target: original approved contour.
The encoder remains frozen; only the decoder is optimized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder
from evaluation.probes.eval_dino_shape_probe_sweep_assd import make_corruption


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_id_from_name(name: str) -> str:
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else stem


class BinaryROIDataset(Dataset):
    def __init__(self, directory: str | Path):
        self.files = sorted(Path(directory).rglob("*.nii.gz"))
        if not self.files:
            raise FileNotFoundError(f"No .nii.gz files found under {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        mask = (arr > 0).astype(np.float32)
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask, got {mask.shape}: {path}")
        return {
            "mask": torch.from_numpy(mask)[None],
            "patient_id": patient_id_from_name(path.name),
        }


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def sample_probe(cfg: Dict, rng: random.Random) -> Tuple[str, float]:
    specs = cfg["denoising"]["corruptions"]
    names, weights = [], []
    for name, spec in specs.items():
        if not spec.get("enabled", True):
            continue
        names.append(name)
        weights.append(float(spec.get("weight", 1.0)))
    probe = rng.choices(names, weights=weights, k=1)[0]
    spec = specs[probe]
    if probe in {"eroded", "dilated", "shifted"}:
        lo, hi = spec.get("voxels", [1, 5])
        level = rng.randint(int(lo), int(hi))
    else:
        lo, hi = spec.get("fraction", [0.05, 0.35])
        level = rng.uniform(float(lo), float(hi))
    return probe, float(level)


def corrupt_batch(target: torch.Tensor, patient_ids: Sequence[str], cfg: Dict,
                  train: bool, epoch: int, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    den = cfg["denoising"]
    base_seed = int(cfg.get("seed", 88))
    clean_probability = float(den.get("clean_probability", 0.20))
    two_probability = float(den.get("two_corruption_probability", 0.30))
    outputs, changed = [], []

    for i in range(target.shape[0]):
        clean_i = target[i].detach()
        if train:
            rng = random.Random(base_seed + epoch * 1_000_003 + step * 1009 + i * 97 + random.randrange(1 << 20))
        else:
            rng = random.Random(stable_seed(str(patient_ids[i]), base_seed))

        x = clean_i.clone()
        if rng.random() >= clean_probability:
            count = 2 if rng.random() < two_probability else 1
            for _ in range(count):
                probe, level = sample_probe(cfg, rng)
                x, _ = make_corruption(x, probe, level, rng)
        outputs.append(x)
        changed.append((x != clean_i).float())

    return torch.stack(outputs, dim=0), torch.stack(changed, dim=0)


def dilate_changed_region(changed: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return changed
    kernel = 2 * radius + 1
    return F.max_pool3d(changed, kernel_size=kernel, stride=1, padding=radius)


def dice_per_case(prob_or_mask: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    dims = tuple(range(2, target.ndim))
    inter = (prob_or_mask * target).sum(dim=dims)
    den = prob_or_mask.sum(dim=dims) + target.sum(dim=dims)
    return (2 * inter + eps) / (den + eps)


def denoising_loss(logits: torch.Tensor, target: torch.Tensor, changed: torch.Tensor, cfg: Dict):
    loss_cfg = cfg["loss"]
    prob = torch.sigmoid(logits)
    global_dice = 1.0 - dice_per_case(prob, target).mean()
    global_bce = F.binary_cross_entropy_with_logits(logits, target)

    region = dilate_changed_region(changed, int(loss_cfg.get("changed_region_dilation_voxels", 4)))
    voxel_weight = 1.0 + float(loss_cfg.get("changed_region_voxel_weight", 8.0)) * region
    changed_bce = F.binary_cross_entropy_with_logits(logits, target, weight=voxel_weight)

    total = (
        float(loss_cfg.get("global_dice_weight", 0.50)) * global_dice
        + float(loss_cfg.get("global_bce_weight", 0.10)) * global_bce
        + float(loss_cfg.get("changed_region_bce_weight", 0.40)) * changed_bce
    )
    return total, {
        "total": total,
        "global_dice_loss": global_dice,
        "global_bce": global_bce,
        "changed_region_bce": changed_bce,
    }


def amp_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def run_epoch(model, loader, optimizer, device, cfg, train: bool, epoch: int):
    model.train(train)
    keys = ["total", "global_dice_loss", "global_bce", "changed_region_bce",
            "input_target_dice", "recon_target_dice", "input_recon_dice", "correction_gain"]
    totals = {k: 0.0 for k in keys}
    totals["n"] = 0
    accumulation = int(cfg["training"].get("accumulation_steps", 1))
    precision = cfg["training"].get("precision", "bf16")
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        target = batch["mask"].to(device, non_blocking=True)
        corrupted, changed = corrupt_batch(target, batch["patient_id"], cfg, train, epoch, step)

        with torch.set_grad_enabled(train):
            with amp_context(device, precision):
                logits, _ = model(corrupted)
                loss, pieces = denoising_loss(logits, target, changed, cfg)
                backward = loss / accumulation
            if train:
                backward.backward()
                if (step + 1) % accumulation == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(
                        model.decoder.parameters(), float(cfg["training"].get("grad_clip", 1.0))
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            recon = (torch.sigmoid(logits) >= threshold).float()
            d_it = dice_per_case(corrupted, target)
            d_rt = dice_per_case(recon, target)
            d_ir = dice_per_case(corrupted, recon)
            metrics = {
                "input_target_dice": d_it.mean(),
                "recon_target_dice": d_rt.mean(),
                "input_recon_dice": d_ir.mean(),
                "correction_gain": (d_rt - d_it).mean(),
            }

        n = target.shape[0]
        totals["n"] += n
        for key in ("total", "global_dice_loss", "global_bce", "changed_region_bce"):
            totals[key] += float(pieces[key].detach().float()) * n
        for key, value in metrics.items():
            totals[key] += float(value.detach().float()) * n

    return {k: totals[k] / totals["n"] for k in keys}


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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True, persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=workers,
                            pin_memory=True, persistent_workers=workers > 0)

    device = torch.device(cfg.get("training", {}).get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model = Frozen3DINOShapeAutoencoder(
        repo_path=cfg["dino"]["repo_path"],
        checkpoint_path=cfg["dino"]["checkpoint_path"],
        config_name=cfg["dino"].get("config_name", "train/vit3d_highres"),
        decoder_channels=cfg["decoder"].get("channels", [256, 128, 64, 32]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.decoder.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Train/validation cases: {len(train_ds)}/{len(val_ds)}")
    print(f"Frozen encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Trainable decoder params: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad):,}")

    history_path = log_dir / "history.csv"
    fields = ["epoch"] + [f"train_{k}" for k in run_epoch.__annotations__ if False]  # placeholder removed below
    metric_names = ["total", "global_dice_loss", "global_bce", "changed_region_bce",
                    "input_target_dice", "recon_target_dice", "input_recon_dice", "correction_gain"]
    fields = ["epoch"] + [f"train_{k}" for k in metric_names] + [f"validation_{k}" for k in metric_names]

    best = float("inf")
    bad_epochs = 0
    epochs = int(cfg["training"].get("num_epochs", 200))
    patience = int(cfg["training"].get("early_stopping_patience", 40))
    clean_floor = float(cfg.get("evaluation", {}).get("clean_dice_floor", 0.0))

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            tr = run_epoch(model, train_loader, optimizer, device, cfg, True, epoch)
            va = run_epoch(model, val_loader, optimizer, device, cfg, False, epoch)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                   **{f"validation_{k}": v for k, v in va.items()}}
            writer.writerow(row)
            f.flush()
            print(json.dumps(row))

            # Primary selection remains validation loss. Correction gain is logged
            # explicitly so identity-mapping behavior is visible during training.
            score = va["total"]
            state = {"epoch": epoch, "decoder_state_dict": model.decoder.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "validation_total": score,
                     "validation_correction_gain": va["correction_gain"], "config": cfg}
            torch.save(state, ckpt_dir / "last_decoder.pth")
            if score < best:
                best = score
                bad_epochs = 0
                torch.save(state, ckpt_dir / "best_decoder.pth")
                print(f"Saved best denoising decoder: validation total={best:.6f}, "
                      f"correction_gain={va['correction_gain']:.4f}")
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break


if __name__ == "__main__":
    main()
