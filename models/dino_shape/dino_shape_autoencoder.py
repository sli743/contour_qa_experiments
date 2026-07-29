#!/usr/bin/env python3
"""Frozen-3DINO binary ROI reconstruction model.

The pretrained 3DINO encoder is frozen. Only the convolutional decoder is trained.
Inputs are binary ROI masks in the same ROI-centered windows used by the existing
shape-autoencoder experiment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_repo_to_path(repo_path: str | Path) -> Path:
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"3DINO repository not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _triple(value) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    return tuple(int(v) for v in value)


def symmetric_pad_to_multiple(
    x: torch.Tensor, multiple_zyx: Sequence[int]
) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int]]:
    """Pad [B,C,D,H,W] symmetrically to multiples of patch size.

    Returns PyTorch pad tuple (xL,xR,yL,yR,zL,zR).
    """
    spatial = x.shape[2:]
    pads_axis = []
    for size, multiple in zip(spatial, multiple_zyx):
        total = (int(multiple) - int(size) % int(multiple)) % int(multiple)
        before = total // 2
        after = total - before
        pads_axis.append((before, after))
    (z0, z1), (y0, y1), (x0, x1) = pads_axis
    pad = (x0, x1, y0, y1, z0, z1)
    return F.pad(x, pad, mode="constant", value=0.0), pad


def remove_symmetric_pad(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
) -> torch.Tensor:
    x0, x1, y0, y1, z0, z1 = pad
    z_end = x.shape[2] - z1 if z1 else x.shape[2]
    y_end = x.shape[3] - y1 if y1 else x.shape[3]
    x_end = x.shape[4] - x1 if x1 else x.shape[4]
    return x[:, :, z0:z_end, y0:y_end, x0:x_end]


class Frozen3DINOEncoder(nn.Module):
    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        config_name: str = "train/vit3d_highres",
    ) -> None:
        super().__init__()
        repo = add_repo_to_path(repo_path)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"3DINO checkpoint not found: {checkpoint}")

        from dinov2.configs import load_and_merge_config_3d
        from dinov2.eval.setup import build_model_for_eval

        old_cwd = Path.cwd()
        try:
            os.chdir(repo)
            cfg = load_and_merge_config_3d(config_name)
            self.backbone = build_model_for_eval(cfg, str(checkpoint))
        finally:
            os.chdir(old_cwd)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        projection = getattr(getattr(self.backbone, "patch_embed", None), "proj", None)
        if projection is None:
            raise AttributeError("Could not identify 3DINO patch embedding projection.")
        self.patch_size = _triple(projection.kernel_size)
        self.embed_dim = int(projection.out_channels)
        self.in_channels = int(projection.in_channels)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, binary_mask: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        # Existing 3DINO CT model in this project accepts one input channel.
        # For robustness, repeat channels only if the checkpoint expects >1.
        if binary_mask.shape[1] != self.in_channels:
            if binary_mask.shape[1] == 1 and self.in_channels > 1:
                binary_mask = binary_mask.repeat(1, self.in_channels, 1, 1, 1)
            else:
                raise ValueError(
                    f"Input has {binary_mask.shape[1]} channels but 3DINO expects "
                    f"{self.in_channels}."
                )

        padded, pad = symmetric_pad_to_multiple(binary_mask, self.patch_size)
        with torch.no_grad():
            features = self.backbone.forward_features(padded)
        tokens = features["x_norm_patchtokens"]
        grid = tuple(int(s // p) for s, p in zip(padded.shape[2:], self.patch_size))
        expected = int(np.prod(grid))
        if tokens.ndim != 3 or tokens.shape[1] != expected:
            raise ValueError(
                f"Patch tokens have shape {tuple(tokens.shape)}; expected "
                f"[B,{expected},C] for grid {grid}."
            )
        feature_grid = tokens.transpose(1, 2).reshape(
            padded.shape[0], self.embed_dim, *grid
        )
        return feature_grid, pad


class ResidualRefine3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class BinaryMaskDecoder3D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        channels: Sequence[int] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if not channels:
            raise ValueError("decoder channels cannot be empty")
        self.projection = nn.Sequential(
            nn.Conv3d(embed_dim, channels[0], 1, bias=False),
            nn.InstanceNorm3d(channels[0], affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.blocks = nn.ModuleList(
            ResidualRefine3D(a, b)
            for a, b in zip(channels[:-1], channels[1:])
        )
        self.final_refine = ResidualRefine3D(channels[-1], channels[-1])
        self.output = nn.Conv3d(channels[-1], 1, 1)

    def forward(
        self, feature_grid: torch.Tensor, output_size_zyx: Sequence[int]
    ) -> torch.Tensor:
        x = self.projection(feature_grid)

        # Use all configured refinement blocks while increasing resolution.
        sizes = []
        current = tuple(int(v) for v in x.shape[2:])
        target = tuple(int(v) for v in output_size_zyx)
        while current != target:
            next_size = tuple(min(t, max(c + 1, c * 2)) for c, t in zip(current, target))
            sizes.append(next_size)
            current = next_size

        blocks = list(self.blocks) + [self.final_refine]
        if len(sizes) > len(blocks):
            # Remaining stages reuse final refinement block.
            blocks = blocks + [self.final_refine] * (len(sizes) - len(blocks))

        for size, block in zip(sizes, blocks):
            dtype = x.dtype
            x = F.interpolate(
                x.float(), size=size, mode="trilinear", align_corners=False
            ).to(dtype=dtype)
            x = block(x)

        if x.shape[2:] != target:
            x = F.interpolate(
                x.float(), size=target, mode="trilinear", align_corners=False
            ).to(dtype=x.dtype)
            x = self.final_refine(x)
        return self.output(x)  # logits


class Frozen3DINOShapeAutoencoder(nn.Module):
    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        config_name: str = "train/vit3d_highres",
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        self.encoder = Frozen3DINOEncoder(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            config_name=config_name,
        )
        self.decoder = BinaryMaskDecoder3D(
            embed_dim=self.encoder.embed_dim,
            channels=decoder_channels,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, binary_mask: torch.Tensor):
        original_size = tuple(int(v) for v in binary_mask.shape[2:])
        features, pad = self.encoder(binary_mask)
        padded_size = tuple(
            s + a + b
            for s, (a, b) in zip(
                original_size,
                [(pad[4], pad[5]), (pad[2], pad[3]), (pad[0], pad[1])],
            )
        )
        logits_padded = self.decoder(features, padded_size)
        logits = remove_symmetric_pad(logits_padded, pad)
        if logits.shape[2:] != original_size:
            raise RuntimeError(
                f"Output shape {tuple(logits.shape[2:])} != input {original_size}"
            )
        return logits, features

    @torch.no_grad()
    def reconstruct(
        self,
        binary_mask: torch.Tensor,
        threshold: float = 0.5,
        return_probability: bool = False,
    ):
        """Reconstruct a binary mask through the shared evaluator interface."""
        logits, features = self(binary_mask)
        probability = torch.sigmoid(logits)
        reconstruction = (probability >= float(threshold)).float()
        if return_probability:
            return reconstruction, probability, features
        return reconstruction


def load_trained_dino_shape_model(
    config: Mapping[str, Any],
    decoder_checkpoint: str | Path,
    device: torch.device | str,
) -> Frozen3DINOShapeAutoencoder:
    """Build the frozen encoder and load a trained contour decoder."""
    dino_cfg = config["dino"]
    decoder_cfg = config.get("decoder", {})
    model = Frozen3DINOShapeAutoencoder(
        repo_path=Path(dino_cfg["repo_path"]).expanduser(),
        checkpoint_path=Path(dino_cfg["checkpoint_path"]).expanduser(),
        config_name=dino_cfg.get("config_name", "train/vit3d_highres"),
        decoder_channels=decoder_cfg.get("channels", [256, 128, 64, 32]),
    ).to(device)

    checkpoint = Path(decoder_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained decoder checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    decoder_state = state.get("decoder_state_dict", state)
    model.decoder.load_state_dict(decoder_state, strict=True)
    model.eval()
    return model
