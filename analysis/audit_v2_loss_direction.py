#!/usr/bin/env python3
"""Audit whether v2 loss supervises outward/excess voxels as intended.

This is a loss-mechanics diagnostic, not a benchmark evaluation.  It generates
fixed inward and outward SDF offsets from clean masks, runs the trained decoder,
and checks the exact tensors entering ``denoising_loss``.

The audit answers:
  * Are outward voxels actually present in ``changed`` and its dilated region?
  * Does ``changed_region_bce`` give them non-zero extra weight?
  * Is its logit gradient in the correct direction (toward background)?
  * How much extra supervision do missing voxels receive from missing-region BCE?

Outputs:
  loss_direction_audit_rows.csv
  loss_direction_audit_summary.csv
  report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder
from training.corruptions.deformation_noise import signed_distance
from training.train_dino_shape_stochastic_region_dropout_decoder import (
    BinaryROIDataset,
    amp_context,
    denoising_loss,
    dilate_changed_region,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v2_global_boundary.yaml",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--offset-mm", type=float, default=4.0,
                   help="Use one clearly realized physical offset for the loss audit.")
    p.add_argument("--max-cases", type=int, default=0, help="0 = all cases")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def load_model(cfg: Dict, checkpoint: Path, device: torch.device):
    model = Frozen3DINOShapeAutoencoder(
        repo_path=cfg["dino"]["repo_path"],
        checkpoint_path=cfg["dino"]["checkpoint_path"],
        config_name=cfg["dino"].get("config_name", "train/vit3d_highres"),
        decoder_channels=cfg["decoder"].get("channels", [256, 128, 64, 32]),
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.decoder.load_state_dict(state.get("decoder_state_dict", state), strict=True)
    model.eval()
    return model, state


def get_dataset(cfg: Dict, split: str):
    key = {
        "train": "train_label_dir",
        "validation": "validation_label_dir",
        "test": "test_label_dir",
    }[split]
    return BinaryROIDataset(cfg["data"][key])


def fixed_offset(clean: np.ndarray, spacing_zyx, offset_mm: float, direction: str) -> np.ndarray:
    sdf = signed_distance(clean.astype(bool), spacing_zyx)
    signed_mm = float(offset_mm) if direction == "outward" else -float(offset_mm)
    return (sdf + signed_mm) > 0.0


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any()):
        return float("nan")
    return float(x[mask].detach().float().mean().cpu())


def audit_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    corrupted: torch.Tensor,
    changed: torch.Tensor,
    cfg: Dict,
) -> Dict[str, float]:
    """Inspect exact loss masks, values, and logit gradients for one sample."""
    loss_cfg = cfg["loss"]
    radius = int(loss_cfg.get("changed_region_dilation_voxels", 4))
    extra_weight = float(loss_cfg.get("changed_region_voxel_weight", 8.0))
    changed_loss_weight = float(loss_cfg.get("changed_region_bce_weight", 0.30))
    missing_loss_weight = float(loss_cfg.get("missing_region_bce_weight", 0.20))

    missing = ((target > 0.5) & (corrupted < 0.5)).float()
    excess = ((target < 0.5) & (corrupted > 0.5)).float()
    region = dilate_changed_region(changed, radius)
    voxel_weight = 1.0 + extra_weight * region

    # Hard bookkeeping checks against the exact tensors used by training.
    xor = ((target > 0.5) != (corrupted > 0.5)).float()
    if not torch.equal(changed > 0.5, xor > 0.5):
        raise AssertionError("changed tensor is not target XOR corrupted")
    if bool(((missing > 0.5) & (excess > 0.5)).any()):
        raise AssertionError("missing and excess overlap")
    if not torch.equal((missing + excess) > 0.5, changed > 0.5):
        raise AssertionError("changed != missing U excess")

    per_voxel_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    global_bce = per_voxel_bce.mean()
    changed_bce = (per_voxel_bce * voxel_weight).mean()
    changed_extra = changed_bce - global_bce

    # Decompose the extra weighting by true changed-shell direction.  This is
    # not a replacement loss; it only shows where the existing weighted BCE is
    # spending its extra penalty.
    n_vox = float(target.numel())
    extra_missing = float((per_voxel_bce * (extra_weight * region) * missing).sum().detach().cpu()) / n_vox
    extra_excess = float((per_voxel_bce * (extra_weight * region) * excess).sum().detach().cpu()) / n_vox

    # Gradient of the exact changed-region BCE with respect to logits.  For
    # target=0 excess voxels this MUST be positive, so gradient descent lowers
    # logits toward background.  For target=1 missing voxels it MUST be negative.
    probe_logits = logits.detach().float().clone().requires_grad_(True)
    probe_bce = F.binary_cross_entropy_with_logits(
        probe_logits, target.float(), weight=voxel_weight.float()
    )
    grad_changed = torch.autograd.grad(probe_bce, probe_logits, retain_graph=False)[0]

    # Gradient of the missing-only term, evaluated independently.
    probe_logits2 = logits.detach().float().clone().requires_grad_(True)
    pv2 = F.binary_cross_entropy_with_logits(probe_logits2, target.float(), reduction="none")
    if bool((missing > 0.5).any()):
        missing_bce = (pv2 * missing.float()).sum() / missing.sum()
        grad_missing = torch.autograd.grad(missing_bce, probe_logits2)[0]
    else:
        missing_bce = probe_logits2.new_zeros(())
        grad_missing = torch.zeros_like(probe_logits2)

    with torch.no_grad():
        total, pieces = denoising_loss(logits, target, changed, corrupted, cfg)

    return {
        "changed_voxels": int(changed.sum().item()),
        "missing_voxels": int(missing.sum().item()),
        "excess_voxels": int(excess.sum().item()),
        "dilated_changed_region_voxels": int((region > 0.5).sum().item()),
        "mean_weight_changed_shell": masked_mean(voxel_weight, changed > 0.5),
        "mean_weight_missing_shell": masked_mean(voxel_weight, missing > 0.5),
        "mean_weight_excess_shell": masked_mean(voxel_weight, excess > 0.5),
        "global_bce_recomputed": float(global_bce.detach().cpu()),
        "changed_region_bce_recomputed": float(changed_bce.detach().cpu()),
        "changed_region_extra_over_global_bce": float(changed_extra.detach().cpu()),
        "changed_extra_from_missing_shell": extra_missing,
        "changed_extra_from_excess_shell": extra_excess,
        "missing_region_bce_recomputed": float(missing_bce.detach().cpu()),
        "changed_bce_grad_mean_missing_shell": masked_mean(grad_changed, missing > 0.5),
        "changed_bce_grad_mean_excess_shell": masked_mean(grad_changed, excess > 0.5),
        "changed_bce_grad_abs_mean_missing_shell": masked_mean(grad_changed.abs(), missing > 0.5),
        "changed_bce_grad_abs_mean_excess_shell": masked_mean(grad_changed.abs(), excess > 0.5),
        "missing_bce_grad_abs_mean_missing_shell": masked_mean(grad_missing.abs(), missing > 0.5),
        "weighted_changed_loss_contribution": changed_loss_weight * float(pieces["changed_region_bce"].detach().cpu()),
        "weighted_missing_loss_contribution": missing_loss_weight * float(pieces["missing_region_bce"].detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def run(cfg: Dict, model, dataset, device: torch.device, offset_mm: float, max_cases: int) -> pd.DataFrame:
    spacing = cfg["stochastic_region_dropout"].get("spacing_zyx", [1.5, 1.5, 1.5])
    precision = cfg.get("training", {}).get("precision", "bf16")
    n_cases = len(dataset) if max_cases <= 0 else min(len(dataset), max_cases)
    rows: List[Dict] = []

    for i in range(n_cases):
        item = dataset[i]
        pid = str(item["patient_id"])
        target_cpu = item["mask"].float()  # [1,D,H,W]
        clean_np = target_cpu[0].numpy() > 0.5
        if not clean_np.any():
            continue

        for direction in ("inward", "outward"):
            corrupted_np = fixed_offset(clean_np, spacing, offset_mm, direction)
            changed_np = np.logical_xor(clean_np, corrupted_np)
            missing_np = np.logical_and(clean_np, ~corrupted_np)
            excess_np = np.logical_and(~clean_np, corrupted_np)

            # At the default 4 mm these should be effective for Lung_L.  If a
            # case is nevertheless a rasterized no-op, record and skip it.
            if not changed_np.any():
                rows.append({
                    "patient_id": pid,
                    "direction": direction,
                    "offset_mm": float(offset_mm),
                    "noop": 1,
                })
                continue

            if direction == "inward" and excess_np.any():
                raise AssertionError(f"inward offset has excess voxels: patient={pid}")
            if direction == "outward" and missing_np.any():
                raise AssertionError(f"outward offset has missing voxels: patient={pid}")

            target = target_cpu.unsqueeze(0).to(device)  # [B,1,D,H,W]
            corrupted = torch.from_numpy(corrupted_np.astype(np.float32))[None, None].to(device)
            changed = torch.from_numpy(changed_np.astype(np.float32))[None, None].to(device)

            with torch.no_grad(), amp_context(device, precision):
                logits, _ = model(corrupted)

            stats = audit_logits(logits, target, corrupted, changed, cfg)
            stats.update({
                "patient_id": pid,
                "direction": direction,
                "offset_mm": float(offset_mm),
                "noop": 0,
                "target_voxels": int(clean_np.sum()),
                "corrupted_voxels": int(corrupted_np.sum()),
            })
            rows.append(stats)

    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    eff = rows[rows["noop"] == 0].copy()
    metrics = [
        "changed_voxels",
        "missing_voxels",
        "excess_voxels",
        "mean_weight_changed_shell",
        "global_bce_recomputed",
        "changed_region_bce_recomputed",
        "changed_region_extra_over_global_bce",
        "changed_extra_from_missing_shell",
        "changed_extra_from_excess_shell",
        "missing_region_bce_recomputed",
        "changed_bce_grad_mean_missing_shell",
        "changed_bce_grad_mean_excess_shell",
        "changed_bce_grad_abs_mean_missing_shell",
        "changed_bce_grad_abs_mean_excess_shell",
        "missing_bce_grad_abs_mean_missing_shell",
        "weighted_changed_loss_contribution",
        "weighted_missing_loss_contribution",
        "total_loss",
    ]
    out = []
    for direction, g in eff.groupby("direction"):
        row = {"direction": direction, "n_effective": len(g)}
        for col in metrics:
            if col in g:
                row[f"median_{col}"] = float(g[col].median(skipna=True))
                row[f"mean_{col}"] = float(g[col].mean(skipna=True))
        out.append(row)
    return pd.DataFrame(out)


def fmt(x, digits=5):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}g}"


def write_report(path: Path, cfg: Dict, checkpoint: Path, offset_mm: float,
                 rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    loss_cfg = cfg["loss"]
    inward = summary[summary.direction == "inward"].iloc[0] if (summary.direction == "inward").any() else None
    outward = summary[summary.direction == "outward"].iloc[0] if (summary.direction == "outward").any() else None

    lines = [
        "# V2 loss-direction audit",
        "",
        "## Purpose",
        "",
        "This audit checks the exact tensors and loss terms used by the stochastic-region-dropout trainer. "
        "It is intended to distinguish an implementation bug from an objective-weighting asymmetry.",
        "",
        f"Checkpoint: `{checkpoint}`  ",
        f"Fixed SDF offset: **{offset_mm:g} mm**",
        "",
        "## Configured loss",
        "",
        f"- global Dice weight: **{float(loss_cfg.get('global_dice_weight', 0.40)):.3f}**",
        f"- global BCE weight: **{float(loss_cfg.get('global_bce_weight', 0.10)):.3f}**",
        f"- changed-region BCE weight: **{float(loss_cfg.get('changed_region_bce_weight', 0.30)):.3f}**",
        f"- missing-region BCE weight: **{float(loss_cfg.get('missing_region_bce_weight', 0.20)):.3f}**",
        f"- changed-region voxel multiplier: **1 + {float(loss_cfg.get('changed_region_voxel_weight', 8.0)):.1f} × region**",
        "",
        "## Direction summary",
        "",
        "| direction | n | missing vox | excess vox | changed-shell weight | extra changed BCE: missing | extra changed BCE: excess | changed-BCE grad on missing | changed-BCE grad on excess | missing-BCE grad on missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['direction']} | {int(r['n_effective'])} | "
            f"{fmt(r.get('median_missing_voxels'))} | {fmt(r.get('median_excess_voxels'))} | "
            f"{fmt(r.get('median_mean_weight_changed_shell'))} | "
            f"{fmt(r.get('median_changed_extra_from_missing_shell'))} | "
            f"{fmt(r.get('median_changed_extra_from_excess_shell'))} | "
            f"{fmt(r.get('median_changed_bce_grad_mean_missing_shell'))} | "
            f"{fmt(r.get('median_changed_bce_grad_mean_excess_shell'))} | "
            f"{fmt(r.get('median_missing_bce_grad_abs_mean_missing_shell'))} |"
        )

    lines += [
        "",
        "## Mechanical interpretation",
        "",
        "For the existing changed-region BCE, the expected logit-gradient signs are:",
        "",
        "- missing foreground (`target=1`): **negative gradient**, so gradient descent increases the logit toward foreground;",
        "- excess foreground (`target=0`): **positive gradient**, so gradient descent decreases the logit toward background.",
        "",
    ]

    if outward is not None:
        ex_vox = outward.get("median_excess_voxels", np.nan)
        ex_extra = outward.get("median_changed_extra_from_excess_shell", np.nan)
        ex_grad = outward.get("median_changed_bce_grad_mean_excess_shell", np.nan)
        if ex_vox > 0 and ex_extra > 0 and ex_grad > 0:
            lines += [
                "**Outward/excess voxels are entering the changed-region BCE correctly.** "
                "They have non-zero excess voxels, receive non-zero extra weighted BCE, and the changed-BCE gradient has the correct positive sign toward background.",
                "",
            ]
        else:
            lines += [
                "**WARNING: the outward/excess supervision did not pass all mechanical checks.** "
                "Inspect `loss_direction_audit_rows.csv` before changing the objective.",
                "",
            ]

    if inward is not None and outward is not None:
        miss_extra = inward.get("median_changed_extra_from_missing_shell", np.nan)
        extra_loss = inward.get("median_weighted_missing_loss_contribution", np.nan)
        outward_missing_loss = outward.get("median_weighted_missing_loss_contribution", np.nan)
        lines += [
            "The asymmetry check is then whether missing voxels receive an additional specialized loss that excess voxels do not. "
            f"Median weighted missing-region loss contribution is **{fmt(extra_loss)}** for inward samples versus **{fmt(outward_missing_loss)}** for outward samples. "
            f"The inward changed-BCE extra attributable to the missing shell is **{fmt(miss_extra)}**.",
            "",
            "If the outward mechanical checks pass while the missing-only loss remains active only for inward samples, that supports an **objective-weighting asymmetry rather than a mask-routing bug**.",
            "",
        ]

    noop = int((rows.get("noop", 0) == 1).sum())
    lines += [
        "## Files",
        "",
        f"No-op rows: **{noop}**",
        "",
        "- `loss_direction_audit_rows.csv`: patient-level raw audit",
        "- `loss_direction_audit_summary.csv`: inward/outward aggregate",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_dir = Path(cfg["output_root"]).expanduser() / cfg["experiment_name"]
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else exp_dir / "checkpoints" / "best_decoder.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else exp_dir / "loss_direction_audit_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Old 3DINO/PyTorch environments reject this newer allocator option.
    # The shell runner also unsets it, but keeping the diagnostic self-contained
    # avoids the same failure when invoked directly.
    import os
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_model(cfg, checkpoint, device)
    dataset = get_dataset(cfg, args.split)
    rows = run(cfg, model, dataset, device, args.offset_mm, args.max_cases)
    summary = summarize(rows)

    rows.to_csv(out_dir / "loss_direction_audit_rows.csv", index=False)
    summary.to_csv(out_dir / "loss_direction_audit_summary.csv", index=False)
    write_report(out_dir / "report.md", cfg, checkpoint, args.offset_mm, rows, summary)

    epoch = state.get("epoch", "unknown") if isinstance(state, dict) else "unknown"
    print(f"Checkpoint epoch: {epoch}")
    print(summary.to_string(index=False))
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
