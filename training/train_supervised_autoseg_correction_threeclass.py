#!/usr/bin/env python3
"""Train a frozen-3DINO three-class real-error correction model.

Classes are defined relative to the input autosegmentation:
  0 preserve, 1 add foreground, 2 remove foreground.
Only the decoder is trainable. The final corrected contour is obtained by
applying the predicted actions to the input contour rather than reconstructing
the whole contour from scratch.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder
from training.train_supervised_autoseg_correction_decoder import paired_affine_augment

PRESERVE, ADD, REMOVE = 0, 1, 2


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path: str, split: str):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        autoseg = (sitk.GetArrayFromImage(sitk.ReadImage(row["autoseg"])) > 0).astype(np.float32)
        approved = (sitk.GetArrayFromImage(sitk.ReadImage(row["approved"])) > 0).astype(np.float32)
        if autoseg.shape != approved.shape:
            raise ValueError(
                f"Shape mismatch patient {row['patient_id']}: "
                f"{autoseg.shape} vs {approved.shape}"
            )
        return {
            "input": torch.from_numpy(autoseg)[None],
            "target": torch.from_numpy(approved)[None],
            "patient_id": row["patient_id"],
        }


def action_target(autoseg: torch.Tensor, approved: torch.Tensor) -> torch.Tensor:
    """Return [B,D,H,W] long labels: preserve/add/remove."""
    a = autoseg[:, 0] > 0.5
    t = approved[:, 0] > 0.5
    labels = torch.zeros_like(a, dtype=torch.long)
    labels[(~a) & t] = ADD
    labels[a & (~t)] = REMOVE
    return labels


def apply_actions(autoseg: torch.Tensor, action_logits: torch.Tensor) -> torch.Tensor:
    """Apply argmax actions to a binary autoseg tensor [B,1,D,H,W]."""
    actions = action_logits.argmax(dim=1, keepdim=True)
    corrected = autoseg.clone()
    corrected = torch.where(actions == ADD, torch.ones_like(corrected), corrected)
    corrected = torch.where(actions == REMOVE, torch.zeros_like(corrected), corrected)
    return corrected


def dice(mask: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(2, target.ndim))
    inter = (mask * target).sum(dims)
    den = mask.sum(dims) + target.sum(dims)
    return (2 * inter + eps) / (den + eps)


def class_balanced_action_loss(logits: torch.Tensor, labels: torch.Tensor, cfg):
    """Separately normalize preserve/add/remove CE and combine active classes.

    If a class is absent from a batch, its configured weight is omitted and the
    remaining active weights are renormalized. This prevents a missing class
    from diluting the loss while avoiding division by zero.
    """
    ce = F.cross_entropy(logits, labels, reduction="none")
    weights = {
        PRESERVE: float(cfg["loss"].get("preserve_ce_weight", 0.20)),
        ADD: float(cfg["loss"].get("add_ce_weight", 0.40)),
        REMOVE: float(cfg["loss"].get("remove_ce_weight", 0.40)),
    }
    names = {PRESERVE: "preserve_ce", ADD: "add_ce", REMOVE: "remove_ce"}
    terms = {}
    active = []
    for cls in (PRESERVE, ADD, REMOVE):
        mask = labels == cls
        if mask.any():
            terms[names[cls]] = ce[mask].mean()
            if weights[cls] > 0:
                active.append((weights[cls], terms[names[cls]]))
        else:
            terms[names[cls]] = ce.new_zeros(())
    denom = sum(w for w, _ in active)
    if denom <= 0:
        raise ValueError("At least one active class loss weight must be positive")
    total = sum((w / denom) * term for w, term in active)
    return total, terms


def weighted_action_loss(logits: torch.Tensor, labels: torch.Tensor, cfg):
    """Prevalence-aware voxelwise CE with moderate action-class weights.

    Unlike ``class_balanced_action_loss``, this keeps the natural voxel
    prevalence in the objective. Add/remove voxels are up-weighted without
    giving each rare class a fixed fraction of the total loss.
    """
    class_weights = torch.tensor(
        [
            float(cfg["loss"].get("preserve_class_weight", 1.0)),
            float(cfg["loss"].get("add_class_weight", 4.0)),
            float(cfg["loss"].get("remove_class_weight", 4.0)),
        ],
        dtype=logits.dtype,
        device=logits.device,
    )
    total = F.cross_entropy(logits, labels, weight=class_weights, reduction="mean")

    # Per-class CE is retained only for interpretable logging. These terms are
    # not separately normalized or recombined into the training objective.
    ce = F.cross_entropy(logits, labels, reduction="none")
    names = {PRESERVE: "preserve_ce", ADD: "add_ce", REMOVE: "remove_ce"}
    terms = {}
    for cls in (PRESERVE, ADD, REMOVE):
        mask = labels == cls
        terms[names[cls]] = ce[mask].mean() if mask.any() else ce.new_zeros(())
    return total, terms


def action_loss(logits: torch.Tensor, labels: torch.Tensor, cfg):
    loss_type = cfg["loss"].get("type", "class_balanced_action_ce")
    if loss_type == "class_balanced_action_ce":
        return class_balanced_action_loss(logits, labels, cfg)
    if loss_type == "weighted_action_ce":
        return weighted_action_loss(logits, labels, cfg)
    raise ValueError(f"Unknown three-class loss type: {loss_type}")


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def run_epoch(model, loader, optimizer, device, cfg, train: bool):
    model.train(train)
    accumulation = int(cfg["training"].get("accumulation_steps", 1))
    precision = cfg["training"].get("precision", "bf16")
    sums = {
        "total": 0.0,
        "preserve_ce": 0.0,
        "add_ce": 0.0,
        "remove_ce": 0.0,
        "input_target_dice": 0.0,
        "corrected_target_dice": 0.0,
        "correction_gain": 0.0,
        "needed_correction_recall": 0.0,
        "missing_correction_recall": 0.0,
        "excess_correction_recall": 0.0,
        "unnecessary_change_fraction": 0.0,
    }
    n = 0
    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        x = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if train:
            x, target = paired_affine_augment(x, target, cfg)
        labels = action_target(x, target)

        with torch.set_grad_enabled(train):
            with amp_context(device, precision):
                logits, _ = model(x)
                loss, pieces = action_loss(logits, labels, cfg)
            if train:
                (loss / accumulation).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        float(cfg["training"].get("grad_clip", 1.0)),
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            corrected = apply_actions(x, logits)
            input_dice = dice(x, target).mean()
            corrected_dice = dice(corrected, target).mean()
            pred_actions = logits.argmax(dim=1, keepdim=True)
            changed = (x > 0.5) != (target > 0.5)
            pred_changed = pred_actions != PRESERVE
            missing = (x <= 0.5) & (target > 0.5)
            excess = (x > 0.5) & (target <= 0.5)
            unchanged = ~changed

            def frac(num, den):
                d = den.sum().float()
                return (num.sum().float() / d) if float(d) > 0 else d.new_zeros(())

            needed = frac(pred_changed & changed, changed)
            missing_recall = frac((pred_actions == ADD) & missing, missing)
            excess_recall = frac((pred_actions == REMOVE) & excess, excess)
            unnecessary = frac(pred_changed & unchanged, unchanged)

        bs = x.shape[0]
        n += bs
        sums["total"] += float(loss.detach().float()) * bs
        for key in ("preserve_ce", "add_ce", "remove_ce"):
            sums[key] += float(pieces[key].detach().float()) * bs
        sums["input_target_dice"] += float(input_dice) * bs
        sums["corrected_target_dice"] += float(corrected_dice) * bs
        sums["correction_gain"] += float(corrected_dice - input_dice) * bs
        sums["needed_correction_recall"] += float(needed) * bs
        sums["missing_correction_recall"] += float(missing_recall) * bs
        sums["excess_correction_recall"] += float(excess_recall) * bs
        sums["unnecessary_change_fraction"] += float(unnecessary) * bs

    return {k: v / max(n, 1) for k, v in sums.items()}


def save_checkpoint(path, model, optimizer, epoch, best, cfg):
    torch.save(
        {
            "decoder_state_dict": model.decoder.state_dict(),
            "encoder_trainable_state_dict": model.encoder.trainable_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_total": best,
            "config": cfg,
            "output_channels": 3,
            "classes": {"preserve": PRESERVE, "add": ADD, "remove": REMOVE},
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_all(int(cfg.get("seed", 88)))
    exp = Path(cfg["output_root"]) / cfg["experiment_name"]
    ckpt_dir = exp / "checkpoints"
    logs_dir = exp / "logs"
    if args.fresh and exp.exists():
        shutil.rmtree(exp)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (exp / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_ds = PairDataset(read_manifest(cfg["manifest"], "train"))
    val_ds = PairDataset(read_manifest(cfg["manifest"], "validation"))
    workers = int(cfg["training"].get("num_workers", 4))
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 88)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device(
        cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )
    model = Frozen3DINOShapeAutoencoder(
        cfg["dino"]["repo_path"],
        cfg["dino"]["checkpoint_path"],
        cfg["dino"].get("config_name", "train/vit3d_highres"),
        cfg["decoder"].get("channels", [256, 128, 64, 32]),
        output_channels=3,
        decoder_blocks_per_stage=int(cfg["decoder"].get("blocks_per_stage", 1)),
        finetune_last_n_blocks=int(cfg.get("transfer_learning", {}).get("finetune_last_n_blocks", 0)),
    ).to(device)

    decoder_lr = float(cfg["training"].get("learning_rate", 1e-4))
    encoder_lr = float(cfg.get("transfer_learning", {}).get("encoder_learning_rate", 1e-6))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    encoder_weight_decay = float(cfg.get("transfer_learning", {}).get("encoder_weight_decay", 1e-2))
    param_groups = [{
        "params": list(model.decoder.parameters()),
        "lr": decoder_lr,
        "weight_decay": weight_decay,
        "name": "decoder",
    }]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": encoder_lr,
            "weight_decay": encoder_weight_decay,
            "name": "encoder",
        })
    optimizer = torch.optim.AdamW(param_groups)

    print(
        f"decoder_trainable={sum(p.numel() for p in model.decoder.parameters() if p.requires_grad):,} "
        f"encoder_trainable={sum(p.numel() for p in encoder_params):,} "
        f"decoder_lr={decoder_lr:g} encoder_lr={encoder_lr:g}"
    )

    test_n = len(read_manifest(cfg["manifest"], "test"))
    print(
        f"{cfg['roi']} fold={cfg['fold']} train={len(train_ds)} "
        f"validation={len(val_ds)} test={test_n}"
    )
    history = logs_dir / "history.csv"
    best = float("inf")
    best_gain = float("-inf")
    bad = 0
    patience = int(cfg["training"].get("early_stopping_patience", 40))
    epochs = int(cfg["training"].get("num_epochs", 200))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, cfg, False)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"validation_{k}": v for k, v in val_metrics.items()},
            "seconds": time.time() - t0,
        }
        with history.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        score = val_metrics["total"]
        if score < best:
            best = score
            bad = 0
            save_checkpoint(ckpt_dir / "best_decoder.pth", model, optimizer, epoch, best, cfg)
        else:
            bad += 1
        if val_metrics["correction_gain"] > best_gain:
            best_gain = val_metrics["correction_gain"]
            save_checkpoint(ckpt_dir / "best_gain_decoder.pth", model, optimizer, epoch, best, cfg)
        save_checkpoint(ckpt_dir / "last_decoder.pth", model, optimizer, epoch, best, cfg)
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} "
            f"train_gain={train_metrics['correction_gain']:.4f} "
            f"val_gain={val_metrics['correction_gain']:.4f} "
            f"val_add={val_metrics['missing_correction_recall']:.3f} "
            f"val_remove={val_metrics['excess_correction_recall']:.3f} "
            f"val_unnecessary={val_metrics['unnecessary_change_fraction']:.6f} "
            f"best={best:.5f} bad={bad}/{patience}",
            flush=True,
        )
        if bad >= patience:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
