#!/usr/bin/env python3
"""Extract deployment-feasible features and true QA targets for real autosegs.

For each patient in a supervised-correction fold manifest, this script computes:
  * simple geometry features from the autoseg mask only
  * pooled frozen-3DINO patch-token features from the autoseg mask only
  * optional v6 preserve/add/remove response summaries from the autoseg mask only
  * true autoseg-vs-approved targets: 1-Dice, ASSD, and HD95

The approved contour is used only to construct regression targets. It is never
used to construct predictor features or localization support.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from scipy import ndimage

from evaluation.probes.eval_dino_shape_denoising_probe_sweep import (
    assd_binary,
    dice_score_binary,
    hd95_binary,
)
from models.dino_shape.dino_shape_autoencoder import (
    BinaryMaskDecoder3D,
    Frozen3DINOEncoder,
    remove_symmetric_pad,
)
from training.train_supervised_autoseg_correction_threeclass import ADD, REMOVE


def read_manifest(path: str | Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_binary(path: str | Path):
    img = sitk.ReadImage(str(path))
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)
    return img, arr


def safe_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def geometry_features(mask: np.ndarray, spacing_zyx) -> Dict[str, float]:
    """Autoseg-only geometry features; all coordinates use array z,y,x order."""
    mask = mask.astype(bool)
    coords = np.column_stack(np.where(mask))
    shape = np.asarray(mask.shape, dtype=float)
    spacing = np.asarray(spacing_zyx, dtype=float)
    voxel_volume = float(np.prod(spacing))
    volume_vox = int(mask.sum())

    out = {
        "geom_volume_vox": float(volume_vox),
        "geom_volume_mm3": float(volume_vox * voxel_volume),
        "geom_foreground_fraction": float(mask.mean()),
    }
    if volume_vox == 0:
        for name in (
            "bbox_z_mm", "bbox_y_mm", "bbox_x_mm", "bbox_volume_mm3",
            "bbox_fill_fraction", "centroid_z_norm", "centroid_y_norm",
            "centroid_x_norm", "surface_vox", "surface_fraction",
        ):
            out[f"geom_{name}"] = float("nan")
        return out

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extent_vox = maxs - mins + 1
    extent_mm = extent_vox * spacing
    bbox_vox = int(np.prod(extent_vox))
    centroid = coords.mean(axis=0)

    eroded = ndimage.binary_erosion(
        mask,
        structure=ndimage.generate_binary_structure(3, 1),
        iterations=1,
        border_value=0,
    )
    surface = mask & (~eroded)

    out.update(
        {
            "geom_bbox_z_mm": float(extent_mm[0]),
            "geom_bbox_y_mm": float(extent_mm[1]),
            "geom_bbox_x_mm": float(extent_mm[2]),
            "geom_bbox_volume_mm3": float(bbox_vox * voxel_volume),
            "geom_bbox_fill_fraction": float(volume_vox / bbox_vox),
            "geom_centroid_z_norm": float(centroid[0] / max(shape[0] - 1.0, 1.0)),
            "geom_centroid_y_norm": float(centroid[1] / max(shape[1] - 1.0, 1.0)),
            "geom_centroid_x_norm": float(centroid[2] / max(shape[2] - 1.0, 1.0)),
            "geom_surface_vox": float(surface.sum()),
            "geom_surface_fraction": float(surface.sum() / volume_vox),
        }
    )
    return out


def true_metrics(autoseg: np.ndarray, approved: np.ndarray, spacing_zyx) -> Dict[str, float]:
    x = torch.from_numpy(autoseg.astype(np.float32))[None, None]
    y = torch.from_numpy(approved.astype(np.float32))[None, None]
    d = float(dice_score_binary(x, y)[0])
    a = float(assd_binary(x, y, tuple(float(v) for v in spacing_zyx))[0])
    h = float(hd95_binary(x, y, tuple(float(v) for v in spacing_zyx))[0])
    return {
        "true_1_minus_dice": 1.0 - d,
        "true_assd_mm": a,
        "true_hd95_mm": h,
    }


def _pad_mask_like_encoder(mask: np.ndarray, pad) -> torch.Tensor:
    """Apply the encoder's symmetric x/y/z pad tuple to a [D,H,W] mask."""
    x0, x1, y0, y1, z0, z1 = [int(v) for v in pad]
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.pad(t, (x0, x1, y0, y1, z0, z1), mode="constant", value=0.0)


def _token_support_from_voxel_mask(mask: np.ndarray, pad, patch_size) -> torch.Tensor:
    """Mark 3DINO tokens whose voxel patch intersects an autoseg-only support."""
    padded = _pad_mask_like_encoder(mask, pad)
    pooled = F.max_pool3d(
        padded,
        kernel_size=tuple(int(v) for v in patch_size),
        stride=tuple(int(v) for v in patch_size),
    )
    return pooled[0, 0] > 0


def autoseg_boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    """Deployment-feasible two-sided boundary band derived only from autoseg."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    if radius <= 0:
        eroded = ndimage.binary_erosion(
            mask, structure=ndimage.generate_binary_structure(3, 1), iterations=1
        )
        return mask & (~eroded)
    structure = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(mask, structure=structure, iterations=int(radius))
    eroded = ndimage.binary_erosion(mask, structure=structure, iterations=int(radius))
    return dilated & (~eroded)


def _channel_pool(flat: torch.Tensor, prefix: str, include_tails: bool = True) -> Dict[str, float]:
    """Pool [C,N] local token features without using approved contours."""
    out: Dict[str, float] = {}
    if flat.shape[1] == 0:
        stats = ("mean", "std", "max", "p95") if include_tails else ("mean", "std")
        for stat in stats:
            for i in range(flat.shape[0]):
                out[f"{prefix}_{stat}_{i:04d}"] = float("nan")
        return out

    mean = flat.mean(dim=1).cpu().numpy()
    std = flat.std(dim=1, unbiased=False).cpu().numpy()
    out.update({f"{prefix}_mean_{i:04d}": float(v) for i, v in enumerate(mean)})
    out.update({f"{prefix}_std_{i:04d}": float(v) for i, v in enumerate(std)})
    if include_tails:
        maxv = flat.max(dim=1).values.cpu().numpy()
        p95 = torch.quantile(flat, 0.95, dim=1).cpu().numpy()
        out.update({f"{prefix}_max_{i:04d}": float(v) for i, v in enumerate(maxv)})
        out.update({f"{prefix}_p95_{i:04d}": float(v) for i, v in enumerate(p95)})
    return out


def dino_pool_features(
    feature_grid: torch.Tensor,
    autoseg: np.ndarray,
    pad,
    patch_size,
    boundary_radius: int,
) -> Dict[str, float]:
    """Global and boundary-aware distributional pooling of frozen patch tokens.

    The legacy baseline is preserved exactly as dino_mean_* + dino_std_*.
    New dino_global_* tail features and dino_boundary_* features are added.
    Boundary support is constructed from the autoseg only.
    """
    f = feature_grid.detach().float()[0]  # [C,Dg,Hg,Wg]
    flat = f.flatten(start_dim=1)

    # Preserve the exact previous feature naming for apples-to-apples comparison.
    mean = flat.mean(dim=1).cpu().numpy()
    std = flat.std(dim=1, unbiased=False).cpu().numpy()
    out = {f"dino_mean_{i:04d}": float(v) for i, v in enumerate(mean)}
    out.update({f"dino_std_{i:04d}": float(v) for i, v in enumerate(std)})

    # Tail-sensitive global statistics. Mean/std are not duplicated here.
    maxv = flat.max(dim=1).values.cpu().numpy()
    p95 = torch.quantile(flat, 0.95, dim=1).cpu().numpy()
    out.update({f"dino_global_max_{i:04d}": float(v) for i, v in enumerate(maxv)})
    out.update({f"dino_global_p95_{i:04d}": float(v) for i, v in enumerate(p95)})

    band = autoseg_boundary_band(autoseg, boundary_radius)
    support = _token_support_from_voxel_mask(band, pad, patch_size)
    if tuple(support.shape) != tuple(f.shape[1:]):
        raise ValueError(
            f"Boundary token support {tuple(support.shape)} != DINO grid {tuple(f.shape[1:])}"
        )
    support_flat = support.flatten().to(device=f.device)
    boundary_flat = flat[:, support_flat]
    out["dino_boundary_token_fraction"] = float(support_flat.float().mean().item())
    out["dino_boundary_token_count"] = float(support_flat.sum().item())
    out.update(_channel_pool(boundary_flat, "dino_boundary", include_tails=True))
    return out


def dilated_autoseg_region(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return ndimage.binary_dilation(
        mask.astype(bool),
        structure=ndimage.generate_binary_structure(3, 1),
        iterations=int(radius),
    )


def correction_response_features(probs: np.ndarray, autoseg: np.ndarray, radius: int):
    """Summaries inside an autoseg-only local support; no approved information."""
    region = dilated_autoseg_region(autoseg, radius)
    p_add = probs[ADD][region]
    p_remove = probs[REMOVE][region]
    p_action = p_add + p_remove
    out = {"corr_local_voxels": float(region.sum())}

    def add_family(name: str, values: np.ndarray):
        out[f"corr_{name}_mean"] = float(values.mean()) if values.size else np.nan
        out[f"corr_{name}_max"] = float(values.max()) if values.size else np.nan
        for q in (0.90, 0.95, 0.99, 0.995, 0.999):
            tag = str(q).replace("0.", "p").replace(".", "")
            out[f"corr_{name}_{tag}"] = safe_quantile(values, q)
        for threshold in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
            tag = str(threshold).replace("0.", "0p").replace(".", "p")
            out[f"corr_{name}_frac_ge_{tag}"] = (
                float(np.mean(values >= threshold)) if values.size else np.nan
            )
        for k in (10, 50, 100, 500, 1000):
            if values.size:
                kk = min(k, values.size)
                top = np.partition(values, values.size - kk)[-kk:]
                out[f"corr_{name}_top{k}_mean"] = float(top.mean())
            else:
                out[f"corr_{name}_top{k}_mean"] = np.nan

    add_family("action", p_action)
    add_family("add", p_add)
    add_family("remove", p_remove)
    return out


def build_encoder(cfg, device):
    dino = cfg["dino"]
    model = Frozen3DINOEncoder(
        repo_path=dino["repo_path"],
        checkpoint_path=dino["checkpoint_path"],
        config_name=dino.get("config_name", "train/vit3d_highres"),
    ).to(device)
    model.eval()
    return model


def build_correction_decoder(v6_cfg_path: Path, embed_dim: int, device):
    """Load only the trained v6 decoder; reuse the already-loaded frozen encoder."""
    if not v6_cfg_path.is_file():
        return None, None
    cfg = yaml.safe_load(v6_cfg_path.read_text())
    ckpt = Path(cfg["output_root"]) / cfg["experiment_name"] / "checkpoints" / "best_decoder.pth"
    if not ckpt.is_file():
        return None, ckpt
    decoder = BinaryMaskDecoder3D(
        embed_dim=embed_dim,
        channels=cfg["decoder"].get("channels", [256, 128, 64, 32]),
        output_channels=3,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    decoder.load_state_dict(state["decoder_state_dict"], strict=True)
    decoder.eval()
    return decoder, ckpt


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base supervised-correction fold config.")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--v6-config", default=None, help="Optional v6 config for correction-response features.")
    ap.add_argument("--correction-mode", choices=["auto", "required", "off"], default="auto")
    ap.add_argument("--local-radius", type=int, default=6)
    ap.add_argument("--dino-boundary-radius", type=int, default=3)
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    device = torch.device(cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(cfg, device)

    correction_decoder = None
    correction_ckpt = None
    if args.correction_mode != "off":
        if args.v6_config:
            v6_cfg = Path(args.v6_config).expanduser().resolve()
        else:
            study_root = cfg_path.parents[2]
            v6_cfg = (
                study_root
                / "threeclass_correction_v6_weighted_ce"
                / cfg["roi"]
                / f"fold_{cfg['fold']}"
                / "config.yaml"
            )
        correction_decoder, correction_ckpt = build_correction_decoder(v6_cfg, encoder.embed_dim, device)
        if correction_decoder is None and args.correction_mode == "required":
            raise FileNotFoundError(f"v6 correction model unavailable: config={v6_cfg}, checkpoint={correction_ckpt}")
        if correction_decoder is None:
            print(f"Correction-response features skipped; no trained v6 checkpoint at {correction_ckpt}")
        else:
            print(f"Using v6 correction checkpoint: {correction_ckpt}")

    rows = read_manifest(cfg["manifest"])
    results = []
    for i, row in enumerate(rows, 1):
        aut_img, aut = read_binary(row["autoseg"])
        _, approved = read_binary(row["approved"])
        if aut.shape != approved.shape:
            raise ValueError(f"Patient {row['patient_id']} shape mismatch: {aut.shape} vs {approved.shape}")
        spacing_zyx = tuple(float(v) for v in aut_img.GetSpacing()[::-1])
        x = torch.from_numpy(aut.astype(np.float32))[None, None].to(device)

        features, pad = encoder(x)
        rec = {
            "patient_id": str(row["patient_id"]),
            "roi": cfg["roi"],
            "fold": int(cfg["fold"]),
            "split": row["split"],
            "cv_bin": row.get("cv_bin", ""),
        }
        rec.update(true_metrics(aut, approved, spacing_zyx))
        rec.update(geometry_features(aut, spacing_zyx))
        rec.update(
            dino_pool_features(
                features,
                autoseg=aut,
                pad=pad,
                patch_size=encoder.patch_size,
                boundary_radius=args.dino_boundary_radius,
            )
        )

        if correction_decoder is not None:
            original_size = tuple(int(v) for v in x.shape[2:])
            padded_size = tuple(
                s + a + b
                for s, (a, b) in zip(
                    original_size,
                    [(pad[4], pad[5]), (pad[2], pad[3]), (pad[0], pad[1])],
                )
            )
            logits_padded = correction_decoder(features, padded_size)
            logits = remove_symmetric_pad(logits_padded, pad)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            rec.update(correction_response_features(probs, aut, args.local_radius))

        results.append(rec)
        print(
            f"[{i}/{len(rows)}] {cfg['roi']} patient {row['patient_id']} {row['split']} "
            f"1-Dice={rec['true_1_minus_dice']:.5f} ASSD={rec['true_assd_mm']:.3f} "
            f"HD95={rec['true_hd95_mm']:.3f}"
        )

    out = Path(args.output_csv).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    meta = {
        "config": str(cfg_path),
        "manifest": str(cfg["manifest"]),
        "roi": cfg["roi"],
        "fold": int(cfg["fold"]),
        "n_cases": len(results),
        "predictor_feature_families": ["geometry", "frozen_3dino"] + (["correction_response_v6"] if correction_decoder is not None else []),
        "target_metrics": ["true_1_minus_dice", "true_assd_mm", "true_hd95_mm"],
        "approved_used_for_features": False,
        "correction_checkpoint": str(correction_ckpt) if correction_decoder is not None else None,
        "local_radius_vox": int(args.local_radius),
        "dino_boundary_radius_vox": int(args.dino_boundary_radius),
        "dino_pooling": {
            "legacy": ["global_mean", "global_std"],
            "new_global": ["max", "p95"],
            "boundary": ["mean", "std", "max", "p95"],
            "boundary_support_uses_approved": False,
        },
    }
    out.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
