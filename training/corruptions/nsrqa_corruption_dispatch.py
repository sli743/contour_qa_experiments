#!/usr/bin/env python3
"""Dispatch NSR-QA training corruptions without changing the trainer objective.

The legacy backend preserves the previous-generation stochastic-region-dropout
sampler.  The SpinalCord proposed backend uses the frozen candidate distribution
stored directly in the generated training config.  Keeping dispatch here makes
old-vs-new experiments differ only in the corruption generator.
"""
from __future__ import annotations

import random
from typing import Dict, Mapping, Tuple

import numpy as np
import torch

from training.corruptions.spinalcord_proposed_distribution import proposed_spinalcord_corruption
from training.corruptions.stochastic_region_dropout import stochastic_region_dropout_corruption

LEGACY_BACKEND = "legacy_stochastic_region_dropout"
SPINALCORD_PROPOSED_BACKEND = "spinalcord_proposed"


def corruption_backend(cfg: Mapping[str, object]) -> str:
    return str(cfg.get("corruption_backend", LEGACY_BACKEND)).strip().lower()


def corruption_signature(cfg: Mapping[str, object]) -> Dict[str, object]:
    """Return the config fragment that determines deterministic corruptions."""
    backend = corruption_backend(cfg)
    if backend == LEGACY_BACKEND:
        return {
            "backend": backend,
            "stochastic_region_dropout": cfg.get("stochastic_region_dropout", {}),
        }
    if backend == SPINALCORD_PROPOSED_BACKEND:
        return {
            "backend": backend,
            "spinalcord_proposed_distribution": cfg.get("spinalcord_proposed_distribution", {}),
        }
    raise ValueError(f"Unknown NSR-QA corruption backend: {backend}")


def apply_nsrqa_corruption(
    target: torch.Tensor,
    cfg: Dict,
    rng: random.Random,
    *,
    cache_key: str | None,
    inferior_is_low_z: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Generate one corrupted contour and changed-region mask.

    Parameters
    ----------
    target:
        Binary ``[1, z, y, x]`` CPU tensor from the approved contour.
    inferior_is_low_z:
        True when decreasing NumPy array-z moves toward patient inferior.
        The legacy backend ignores this flag; the SpinalCord endpoint model uses
        it so superior/inferior corruption does not depend on file orientation.
    """
    backend = corruption_backend(cfg)
    if backend == LEGACY_BACKEND:
        corrupted, changed, metadata = stochastic_region_dropout_corruption(
            target, cfg, rng, cache_key=cache_key
        )
        meta = dict(metadata)
        meta["corruption_backend"] = backend
        return corrupted, changed, meta

    if backend != SPINALCORD_PROPOSED_BACKEND:
        raise ValueError(f"Unknown NSR-QA corruption backend: {backend}")

    proposal = cfg.get("spinalcord_proposed_distribution")
    if not isinstance(proposal, Mapping):
        raise KeyError(
            "corruption_backend=spinalcord_proposed requires "
            "spinalcord_proposed_distribution in the training config"
        )
    spacing_zyx = tuple(
        float(v)
        for v in cfg.get("evaluation", {}).get("spacing_zyx", [1.5, 1.5, 1.5])
    )
    arr = target.detach().cpu().numpy()
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"Expected target [1,z,y,x], got {arr.shape}")
    original = arr[0] > 0.5
    result, metadata = proposed_spinalcord_corruption(
        original,
        spacing_zyx,
        proposal,
        rng,
        inferior_is_low_z=bool(inferior_is_low_z),
        cache_key=cache_key,
    )
    result = np.asarray(result, dtype=bool)
    changed_np = np.logical_xor(original, result)
    corrupted = torch.from_numpy(result.astype(np.float32, copy=False))[None]
    changed = torch.from_numpy(changed_np.astype(np.float32, copy=False))[None]
    meta = dict(metadata)
    meta.update({
        "corruption_backend": backend,
        "sampler_version": str(proposal.get("sampler_version", "")),
        "sampling_strategy": str(proposal.get("sampling_strategy", "")),
        "inferior_is_low_z": int(bool(inferior_is_low_z)),
    })
    return corrupted, changed, meta
