#!/usr/bin/env python3
"""Frozen-3DINO CT + contour reconstruction model (image-conditioned NSR-QA).

The same frozen 3DINO encoder is run twice per case: once on the (possibly
corrupted) binary contour and once on the CT crop.  The two patch-token grids
are fused and decoded into a reconstructed contour.

Design choices (see scripts/dino_shape/LUNG_L_CT_CONTOUR_NSRQA.md):

* The fusion starts as the contour-only model.  ``gated_add`` adds a projected
  CT token grid through a per-channel gate initialised at zero; ``concat``
  mixes [contour, CT] tokens with a 1x1 conv initialised to pass the contour
  tokens through unchanged.  CT information enters only as far as training
  finds it useful.
* 3DINO patch tokens are coarse (one token per 16^3 voxels at 1.5 mm), so a
  small trainable full-resolution CT stem feeds the last decoder stages.  Each
  injection is a zero-initialised 1x1 conv, again starting as the baseline.
* ``image_skip_inputs: ct_mask`` (early fusion) feeds the stem [CT, contour] as
  two channels, so its first conv can compare the input boundary with the CT
  voxel by voxel at full resolution; with ``ct`` (default) the contour reaches
  the decoder only through the 16^3 tokens.  The mask channel is gated by
  ``image_present`` together with the CT, so CT dropout also trains the
  token-only path.
* ``image_present`` (one value per case) switches all CT pathways off.  The
  trainer uses it for CT dropout, so the decoder cannot rely on CT alone, and
  the evaluator uses it for the CT-reliance check.
* ``contour_input: empty`` feeds an all-zero contour, giving the CT-only
  segmenter comparison arm with the same architecture.
* The encoder is frozen and the CT crop never changes between corruptions, so
  CT tokens can be cached per case during training.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dino_shape.dino_shape_autoencoder import (
    BinaryMaskDecoder3D,
    Frozen3DINOEncoder,
    remove_symmetric_pad,
)

# Encoding used by the roiwise preprocessing (normalize_ct_to_uint8).
UINT8_CT_MIN_HU = -1024.0
UINT8_CT_MAX_HU = 3324.0

DEFAULT_IMAGE_CFG: Dict[str, Any] = {
    "ct_encoding": "auto",            # auto | hu | uint8_m1024_3324
    "quantize_to_uint8_grid": True,   # make raw-HU eval crops match uint8 training crops
    "hu_window": [-1000.0, 2000.0],   # mapped linearly to [-1, 1]
    "fusion": "gated_add",            # gated_add | concat
    "fusion_hidden_dim": 256,
    "image_skip_channels": 16,
    "image_skip_last_n_stages": 2,    # 0 disables the full-resolution CT stem
    "image_skip_inputs": "ct",        # ct | ct_mask (two-channel CT + contour stem)
    "train_image_dropout": 0.3,
    "contour_input": "mask",          # mask | empty (CT-only segmenter arm)
    "cache_image_tokens": True,
}


def image_cfg(cfg: Mapping) -> Dict[str, Any]:
    out = dict(DEFAULT_IMAGE_CFG)
    out.update(dict(cfg.get("image_conditioning") or {}))
    return out


def image_conditioning_enabled(cfg: Mapping) -> bool:
    return bool((cfg.get("image_conditioning") or {}).get("enabled", False))


# ---------------------------------------------------------------------------
# CT intensity handling
# ---------------------------------------------------------------------------

def ct_image_to_hu(img: sitk.Image, encoding: str = "auto") -> np.ndarray:
    """Return the CT crop in Hounsfield units as float32 [D,H,W]."""
    arr = sitk.GetArrayFromImage(img)
    enc = str(encoding)
    if enc == "auto":
        enc = "uint8_m1024_3324" if img.GetPixelID() == sitk.sitkUInt8 else "hu"
    if enc == "uint8_m1024_3324":
        return arr.astype(np.float32) / 255.0 * (UINT8_CT_MAX_HU - UINT8_CT_MIN_HU) + UINT8_CT_MIN_HU
    if enc == "hu":
        out = arr.astype(np.float32)
        if out.size and float(out.min()) >= 0.0 and float(out.max()) <= 255.0:
            raise ValueError(
                "CT crop looks 0..255-encoded but was read as HU; set "
                "image_conditioning.ct_encoding: uint8_m1024_3324"
            )
        return out
    raise ValueError(f"Unknown ct_encoding: {encoding}")


def normalize_hu(hu: np.ndarray, icfg: Mapping) -> np.ndarray:
    hu = np.asarray(hu, dtype=np.float32)
    if bool(icfg.get("quantize_to_uint8_grid", True)):
        span = UINT8_CT_MAX_HU - UINT8_CT_MIN_HU
        q = np.floor((np.clip(hu, UINT8_CT_MIN_HU, UINT8_CT_MAX_HU) - UINT8_CT_MIN_HU) / span * 255.0)
        hu = q / 255.0 * span + UINT8_CT_MIN_HU
    lo, hi = (float(v) for v in icfg.get("hu_window", [-1000.0, 2000.0]))
    if not hi > lo:
        raise ValueError(f"Invalid hu_window: {lo}, {hi}")
    x = (np.clip(hu, lo, hi) - lo) / (hi - lo)
    return (2.0 * x - 1.0).astype(np.float32)


def load_ct_array(path_or_img: str | Path | sitk.Image, cfg: Mapping) -> np.ndarray:
    """Read a CT crop and return normalised float32 [D,H,W] in [-1, 1]."""
    icfg = image_cfg(cfg)
    img = path_or_img if isinstance(path_or_img, sitk.Image) else sitk.ReadImage(str(path_or_img))
    return normalize_hu(ct_image_to_hu(img, icfg.get("ct_encoding", "auto")), icfg)


def load_ct_tensor(path_or_img, cfg: Mapping, device: torch.device | str) -> torch.Tensor:
    """[1,1,D,H,W] tensor on device."""
    return torch.from_numpy(load_ct_array(path_or_img, cfg))[None, None].to(device)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

def _present_view(present: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return present.to(device=like.device, dtype=like.dtype).view(-1, 1, 1, 1, 1)


class ImageTokenFusion(nn.Module):
    def __init__(self, embed_dim: int, mode: str = "gated_add", hidden_dim: int = 256) -> None:
        super().__init__()
        self.mode = str(mode)
        e = int(embed_dim)
        if self.mode == "gated_add":
            h = int(hidden_dim)
            self.image_proj = nn.Sequential(
                nn.Conv3d(e, h, 1, bias=False),
                nn.InstanceNorm3d(h, affine=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(h, e, 1),
            )
            self.gate = nn.Parameter(torch.zeros(e))
        elif self.mode == "concat":
            self.mix = nn.Conv3d(2 * e, e, 1, bias=True)
            with torch.no_grad():
                self.mix.weight.zero_()
                self.mix.bias.zero_()
                self.mix.weight[:, :e, 0, 0, 0].copy_(torch.eye(e))
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.embed_dim = e

    def forward(self, mask_tokens: torch.Tensor, image_tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        image_tokens = image_tokens.to(dtype=mask_tokens.dtype)
        p = _present_view(present, mask_tokens)
        if self.mode == "gated_add":
            gate = self.gate.to(mask_tokens.dtype).view(1, -1, 1, 1, 1)
            return mask_tokens + gate * self.image_proj(image_tokens) * p
        return self.mix(torch.cat([mask_tokens, image_tokens * p], dim=1))

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        if self.mode == "gated_add":
            g = self.gate.detach().float()
            return {"fusion_gate_abs_mean": float(g.abs().mean()), "fusion_gate_abs_max": float(g.abs().max())}
        w = self.mix.weight.detach().float()
        return {
            "fusion_ct_weight_norm": float(w[:, self.embed_dim:].norm()),
            "fusion_contour_weight_norm": float(w[:, : self.embed_dim].norm()),
        }


SKIP_INPUT_CHANNELS = {"ct": 1, "ct_mask": 2}


class ImageSkipStem(nn.Module):
    def __init__(self, channels: int = 16, in_channels: int = 1) -> None:
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            nn.Conv3d(int(in_channels), c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class ImageGuidedMaskDecoder3D(BinaryMaskDecoder3D):
    """BinaryMaskDecoder3D plus zero-initialised CT injections in its last stages."""

    def __init__(self, *args, expected_steps: int, image_skip_channels: int = 16,
                 image_skip_last_n_stages: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.expected_steps = int(expected_steps)
        self.skip_last_n = max(0, min(int(image_skip_last_n_stages), self.expected_steps))
        injectors = []
        for k in range(self.skip_last_n):
            step = self.expected_steps - self.skip_last_n + k
            conv = nn.Conv3d(int(image_skip_channels), self.stage_input_channels(step), 1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            injectors.append(conv)
        self.image_injectors = nn.ModuleList(injectors)

    def _before_stage(self, x, step_index, n_steps, image_skip=None, image_present=None, **_):
        if image_skip is None or self.skip_last_n == 0:
            return x
        if n_steps != self.expected_steps:
            raise RuntimeError(
                f"Decoder made {n_steps} upsampling steps; image injectors were built for "
                f"{self.expected_steps}. Check the input crop size / patch size."
            )
        k = step_index - (n_steps - self.skip_last_n)
        if k < 0:
            return x
        feat = image_skip
        if tuple(feat.shape[2:]) != tuple(x.shape[2:]):
            feat = F.adaptive_avg_pool3d(feat.float(), tuple(int(v) for v in x.shape[2:]))
        inj = self.image_injectors[k](feat.to(x.dtype))
        if image_present is not None:
            inj = inj * _present_view(image_present, inj)
        return x + inj


class ImageContourHead(nn.Module):
    """All trainable parts; exposed as ``model.decoder`` so the existing trainer,
    optimizer, grad clipping and checkpoint format work unchanged."""

    def __init__(self, embed_dim: int, decoder_channels: Sequence[int], expected_steps: int,
                 icfg: Mapping, spatial_context_dim: int = 0, spatial_context_hidden_dim: int = 64) -> None:
        super().__init__()
        self.fusion = ImageTokenFusion(embed_dim, icfg["fusion"], int(icfg["fusion_hidden_dim"]))
        last_n = int(icfg["image_skip_last_n_stages"])
        self.skip_inputs = str(icfg.get("image_skip_inputs", "ct"))
        if self.skip_inputs not in SKIP_INPUT_CHANNELS:
            raise ValueError(f"image_skip_inputs must be one of {sorted(SKIP_INPUT_CHANNELS)}, got {self.skip_inputs}")
        if self.skip_inputs != "ct" and last_n <= 0:
            raise ValueError("image_skip_inputs=ct_mask needs image_skip_last_n_stages > 0")
        self.image_stem = (ImageSkipStem(int(icfg["image_skip_channels"]), SKIP_INPUT_CHANNELS[self.skip_inputs])
                           if last_n > 0 else None)
        self.mask_decoder = ImageGuidedMaskDecoder3D(
            embed_dim=embed_dim,
            channels=decoder_channels,
            expected_steps=expected_steps,
            image_skip_channels=int(icfg["image_skip_channels"]),
            image_skip_last_n_stages=last_n,
            spatial_context_dim=spatial_context_dim,
            spatial_context_hidden_dim=spatial_context_hidden_dim,
        )

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        out = self.fusion.stats()
        for k, conv in enumerate(self.mask_decoder.image_injectors):
            out[f"image_injector{k}_weight_norm"] = float(conv.weight.detach().float().norm())
        return out


class Frozen3DINOImageContourAutoencoder(nn.Module):
    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        config_name: str = "train/vit3d_highres",
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        image_conditioning: Mapping | None = None,
        spatial_context_dim: int = 0,
        spatial_context_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        icfg = dict(DEFAULT_IMAGE_CFG)
        icfg.update(dict(image_conditioning or {}))
        self.icfg = icfg
        self.encoder = Frozen3DINOEncoder(repo_path, checkpoint_path, config_name)
        self.encoder.configure_finetuning(0)
        self.patch_size = self.encoder.patch_size
        expected_steps = int(math.ceil(math.log2(max(self.patch_size))))
        self.decoder = ImageContourHead(
            self.encoder.embed_dim, decoder_channels, expected_steps, icfg,
            spatial_context_dim=spatial_context_dim,
            spatial_context_hidden_dim=spatial_context_hidden_dim,
        )
        self.contour_input = str(icfg.get("contour_input", "mask"))
        if self.contour_input not in {"mask", "empty"}:
            raise ValueError(f"contour_input must be mask or empty, got {self.contour_input}")
        self.cache_image_tokens = bool(icfg.get("cache_image_tokens", True))
        self._image_token_cache: Dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def clear_image_token_cache(self) -> None:
        self._image_token_cache.clear()

    def _pad(self, x: torch.Tensor, value: float):
        spatial = x.shape[2:]
        pads = []
        for size, multiple in zip(spatial, self.patch_size):
            total = (int(multiple) - int(size) % int(multiple)) % int(multiple)
            pads.append((total // 2, total - total // 2))
        (z0, z1), (y0, y1), (x0, x1) = pads
        pad = (x0, x1, y0, y1, z0, z1)
        return F.pad(x, pad, mode="constant", value=float(value)), pad

    def _encode_image(self, padded_image: torch.Tensor, cache_keys: Sequence[str] | None) -> torch.Tensor:
        if cache_keys is None or not self.cache_image_tokens:
            tokens, _ = self.encoder(padded_image)
            return tokens
        out = []
        for i, key in enumerate(cache_keys):
            key = str(key)
            cached = self._image_token_cache.get(key)
            if cached is None or cached.device != padded_image.device:
                tokens, _ = self.encoder(padded_image[i : i + 1])
                cached = tokens.detach()
                self._image_token_cache[key] = cached
            out.append(cached)
        return torch.cat(out, dim=0)

    def forward(
        self,
        binary_mask: torch.Tensor,
        image: torch.Tensor | None = None,
        image_present: torch.Tensor | None = None,
        spatial_context: torch.Tensor | None = None,
        image_cache_keys: Sequence[str] | None = None,
    ):
        if image is None:
            raise ValueError("Image-conditioned model requires the CT crop (image=...)")
        if tuple(image.shape[2:]) != tuple(binary_mask.shape[2:]) or image.shape[0] != binary_mask.shape[0]:
            raise ValueError(f"CT {tuple(image.shape)} and contour {tuple(binary_mask.shape)} differ")
        batch = binary_mask.shape[0]
        if image_present is None:
            image_present = torch.ones(batch, device=binary_mask.device)
        if self.contour_input == "empty":
            binary_mask = torch.zeros_like(binary_mask)

        original_size = tuple(int(v) for v in binary_mask.shape[2:])
        padded_mask, pad = self._pad(binary_mask, 0.0)
        padded_image, _ = self._pad(image, -1.0)  # air after normalisation
        mask_tokens, _ = self.encoder(padded_mask)
        image_tokens = self._encode_image(padded_image, image_cache_keys)

        fused = self.decoder.fusion(mask_tokens, image_tokens, image_present)
        skip = None
        if self.decoder.image_stem is not None:
            stem_in = padded_image
            if self.decoder.skip_inputs == "ct_mask":
                # contour in {-1, 1} like the CT window; padding = background (-1)
                stem_in = torch.cat([padded_image, 2.0 * padded_mask.to(padded_image.dtype) - 1.0], dim=1)
            skip = self.decoder.image_stem(stem_in)
        logits_padded = self.decoder.mask_decoder(
            fused, tuple(int(v) for v in padded_mask.shape[2:]),
            spatial_context=spatial_context, image_skip=skip, image_present=image_present,
        )
        logits = remove_symmetric_pad(logits_padded, pad)
        if tuple(logits.shape[2:]) != original_size:
            raise RuntimeError(f"Output shape {tuple(logits.shape[2:])} != input {original_size}")
        return logits, fused

    @torch.no_grad()
    def reconstruct(
        self,
        binary_mask: torch.Tensor,
        threshold: float = 0.5,
        return_probability: bool = False,
        spatial_context: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        image_present: torch.Tensor | None = None,
    ):
        logits, features = self(binary_mask, image=image, image_present=image_present,
                                spatial_context=spatial_context)
        probability = torch.sigmoid(logits)
        reconstruction = (probability >= float(threshold)).float()
        if return_probability:
            return reconstruction, probability, features
        return reconstruction


def build_image_contour_model(cfg: Mapping) -> Frozen3DINOImageContourAutoencoder:
    dino_cfg = cfg["dino"]
    sc = cfg.get("spatial_conditioning") or {}
    return Frozen3DINOImageContourAutoencoder(
        repo_path=Path(dino_cfg["repo_path"]).expanduser(),
        checkpoint_path=Path(dino_cfg["checkpoint_path"]).expanduser(),
        config_name=dino_cfg.get("config_name", "train/vit3d_highres"),
        decoder_channels=(cfg.get("decoder") or {}).get("channels", [256, 128, 64, 32]),
        image_conditioning=cfg.get("image_conditioning") or {},
        spatial_context_dim=int(sc.get("context_dim", 12)) if bool(sc.get("enabled", False)) else 0,
        spatial_context_hidden_dim=int(sc.get("hidden_dim", 64)),
    )


def load_trained_image_contour_model(config: Mapping, decoder_checkpoint, device) -> Frozen3DINOImageContourAutoencoder:
    model = build_image_contour_model(config).to(device)
    checkpoint = Path(decoder_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained decoder checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    model.decoder.load_state_dict(state.get("decoder_state_dict", state), strict=True)
    model.eval()
    return model
