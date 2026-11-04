#!/usr/bin/env python3
"""Candidate SpinalCord corruption distribution for pre-training validation.

This module deliberately does *not* modify the production NSR-QA corruption
function.  It implements the candidate distribution inferred from the
Month1+Month2 calibration so that its geometry can be audited before any model
is trained with it.

Arrays are NumPy z,y,x.  Endpoint truncation receives an explicit
``inferior_is_low_z`` flag from the NIfTI geometry; it never assumes array-z
orientation.
"""
from __future__ import annotations

import random
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage

from training.corruptions.deformation_noise import signed_distance
from training.corruptions.stochastic_region_dropout import (
    smooth_random_warp_fast,
    stochastic_region_dropout,
)

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def _uniform(rng: random.Random, bounds: Sequence[float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    return rng.uniform(lo, hi)


def _triangular(rng: random.Random, spec, default: Sequence[float]) -> float:
    if isinstance(spec, Mapping):
        lo = float(spec.get("min", default[0]))
        hi = float(spec.get("max", default[1]))
        mode = float(spec.get("mode", 0.5 * (lo + hi)))
        return rng.triangular(lo, hi, mode)
    if isinstance(spec, (list, tuple)) and len(spec) >= 2:
        return rng.uniform(float(spec[0]), float(spec[1]))
    return rng.uniform(float(default[0]), float(default[1]))


def _choose_weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    names = list(weights)
    vals = [max(0.0, float(weights[n])) for n in names]
    total = sum(vals)
    if total <= 0:
        raise ValueError("At least one branch weight must be positive")
    draw = rng.random() * total
    acc = 0.0
    for name, weight in zip(names, vals):
        acc += weight
        if draw <= acc:
            return name
    return names[-1]


def endpoint_truncation(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Mapping[str, object],
    rng: random.Random,
    *,
    inferior_is_low_z: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Remove a physically specified length from one S/I endpoint."""
    mask = np.asarray(mask, dtype=bool)
    z = np.where(np.any(mask, axis=(1, 2)))[0]
    if len(z) < 2:
        return mask.copy(), {"endpoint_rejected_to_clean": 1, "endpoint_rejection_reason": "too_short"}

    inferior_probability = float(spec.get("inferior_probability", 0.85))
    side = "inferior" if rng.random() < inferior_probability else "superior"
    if side == "inferior":
        inf = spec.get("inferior", {})
        mild_probability = float(inf.get("mild_probability", 0.30))
        if rng.random() < mild_probability:
            mode = "mild"
            length_mm = _uniform(rng, inf.get("mild_length_mm", [1.5, 3.0]))
        else:
            mode = "severe"
            length_mm = _uniform(rng, inf.get("severe_length_mm", [25.0, 50.0]))
    else:
        mode = "mild"
        length_mm = _uniform(rng, spec.get("superior_length_mm", [1.5, 3.0]))

    spacing_z = float(spacing_zyx[0])
    requested_slices = max(1, int(round(length_mm / max(spacing_z, 1e-6))))
    max_removable = max(len(z) - 1, 1)
    n = min(requested_slices, max_removable)
    out = mask.copy()
    remove_low = (side == "inferior" and inferior_is_low_z) or (side == "superior" and not inferior_is_low_z)
    if remove_low:
        cutoff = int(z.min()) + n
        out[:cutoff] = False
    else:
        cutoff = int(z.max()) - n + 1
        out[cutoff:] = False
    if not out.any():
        return mask.copy(), {"endpoint_rejected_to_clean": 1, "endpoint_rejection_reason": "empty_result"}
    return out, {
        "endpoint_rejected_to_clean": 0,
        "endpoint_side": side,
        "endpoint_severity_mode": mode,
        "endpoint_requested_length_mm": float(length_mm),
        "endpoint_applied_slices": int(n),
        "endpoint_applied_length_mm": float(n * spacing_z),
        "inferior_is_low_z": int(bool(inferior_is_low_z)),
    }


def _signed_distance_2d(mask2d: np.ndarray, spacing_yx: Sequence[float]) -> np.ndarray:
    inside = ndimage.distance_transform_edt(mask2d, sampling=spacing_yx)
    outside = ndimage.distance_transform_edt(~mask2d, sampling=spacing_yx)
    return inside - outside


def transverse_boundary_offset(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Mapping[str, object],
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Uniform transverse offset while preserving the original S/I support."""
    mask = np.asarray(mask, dtype=bool)
    magnitude = _triangular(rng, spec.get("magnitude_mm", {}), [0.5, 1.5])
    outward = rng.random() < float(spec.get("outward_probability", 0.50))
    offset = magnitude if outward else -magnitude
    out = np.zeros_like(mask, dtype=bool)
    fallback_slices = 0
    active = np.where(np.any(mask, axis=(1, 2)))[0]
    for zi in active:
        sdf = _signed_distance_2d(mask[zi], spacing_zyx[1:])
        sl = (sdf + offset) > 0.0
        if not sl.any():
            sl = mask[zi].copy()
            fallback_slices += 1
        out[zi] = sl
    return out, {
        "transverse_offset_mm": float(offset),
        "transverse_offset_magnitude_mm": float(magnitude),
        "transverse_offset_direction": "outward" if outward else "inward",
        "transverse_offset_empty_slice_fallbacks": int(fallback_slices),
    }


def transverse_correlated_boundary_noise(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Mapping[str, object],
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Correlated cross-sectional boundary perturbation with fixed S/I endpoints.

    A smooth 3-D scalar field controls the displacement, but it is applied to a
    *2-D* signed distance transform independently on each original non-empty
    axial slice. Therefore it changes boundary shape without creating/removing
    S/I slices.
    """
    mask = np.asarray(mask, dtype=bool)
    np_rng = np.random.default_rng(rng.randrange(2**32))
    amplitude = _triangular(rng, spec.get("amplitude_mm", {}), [0.5, 2.5])
    sigma_z = _uniform(rng, spec.get("sigma_z_vox", [5.0, 12.0]))
    sigma_yx = _uniform(rng, spec.get("sigma_yx_vox", [2.0, 5.0]))
    noise = np_rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=(sigma_z, sigma_yx, sigma_yx), mode="reflect")
    noise -= float(noise.mean())
    std = float(noise.std())
    if std > 1e-8:
        noise /= std
    noise = np.clip(noise, -3.0, 3.0)

    out = np.zeros_like(mask, dtype=bool)
    fallback_slices = 0
    active = np.where(np.any(mask, axis=(1, 2)))[0]
    for zi in active:
        sdf = _signed_distance_2d(mask[zi], spacing_zyx[1:])
        sl = (sdf + amplitude * noise[zi]) > 0.0
        if not sl.any():
            sl = mask[zi].copy()
            fallback_slices += 1
        out[zi] = sl
    return out, {
        "transverse_noise_amplitude_mm": float(amplitude),
        "transverse_noise_sigma_z_vox": float(sigma_z),
        "transverse_noise_sigma_yx_vox": float(sigma_yx),
        "transverse_noise_empty_slice_fallbacks": int(fallback_slices),
    }


def small_local_bite(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Mapping[str, object],
    rng: random.Random,
    *,
    cache_key: str | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Use the existing irregular dropout machinery but reject endpoint cuts."""
    original = np.asarray(mask, dtype=bool)
    z = np.where(np.any(original, axis=(1, 2)))[0]
    zmin, zmax = int(z.min()), int(z.max())
    outer_attempts = int(spec.get("endpoint_guard_attempts", 6))
    rejected_for_endpoint = 0
    last_meta: Dict[str, object] = {}
    for _ in range(max(1, outer_attempts)):
        result, meta = stochastic_region_dropout(original, spacing_zyx, dict(spec), rng, cache_key=cache_key)
        last_meta = dict(meta)
        if int(meta.get("dropout_rejected_to_clean", 0)):
            continue
        zr = np.where(np.any(result, axis=(1, 2)))[0]
        if len(zr) and int(zr.min()) == zmin and int(zr.max()) == zmax:
            return result, {**last_meta, "local_bite_endpoint_guard_rejections": rejected_for_endpoint}
        rejected_for_endpoint += 1
    return original.copy(), {
        **last_meta,
        "local_bite_rejected_to_clean": 1,
        "local_bite_rejection_reason": "endpoint_support_changed_or_dropout_rejected",
        "local_bite_endpoint_guard_rejections": rejected_for_endpoint,
    }


def _warp_with_candidate_amplitude(mask, spacing_zyx, spec, rng):
    use = dict(spec)
    amp = _triangular(rng, use.pop("amplitude_mm", {}), [2.0, 6.0])
    use["max_displacement_mm"] = [amp, amp]
    out, meta = smooth_random_warp_fast(mask, spacing_zyx, use, rng)
    meta["candidate_warp_sampled_amplitude_mm"] = float(amp)
    return out, meta



def _preserve_si_support(reference: np.ndarray, candidate: np.ndarray) -> Tuple[np.ndarray, int]:
    """Force a surface-only corruption to keep the reference S/I support.

    Smooth 3-D warps can otherwise create or remove terminal slices, which would
    confound the explicitly modeled endpoint axis. Voxels outside the reference
    z support are removed; if either reference endpoint becomes empty, that
    endpoint cross-section is restored from the reference mask.
    """
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool).copy()
    z = np.where(np.any(reference, axis=(1, 2)))[0]
    if len(z) == 0:
        return candidate, 0
    zmin, zmax = int(z.min()), int(z.max())
    candidate[:zmin] = False
    candidate[zmax + 1:] = False
    restored = 0
    if not candidate[zmin].any():
        candidate[zmin] = reference[zmin]
        restored += 1
    if zmax != zmin and not candidate[zmax].any():
        candidate[zmax] = reference[zmax]
        restored += 1
    return candidate, restored



def transverse_micro_jitter(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Mapping[str, object],
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Short-correlation, mildly outward-biased transverse boundary jitter.

    The displacement field is applied independently to 2-D signed-distance
    maps on the original non-empty axial slices, so S/I support is preserved.
    Compared with the v2 transverse noise, the shorter z/yx correlation is
    intended to create many small distributed boundary disagreements rather
    than a few long coherent regions.
    """
    mask = np.asarray(mask, dtype=bool)
    np_rng = np.random.default_rng(rng.randrange(2**32))
    amplitude = _triangular(rng, spec.get("amplitude_mm", {}), [0.8, 2.0])
    outward_bias = _triangular(rng, spec.get("outward_bias_mm", {}), [0.25, 0.90])
    sigma_z = _uniform(rng, spec.get("sigma_z_vox", [0.75, 2.50]))
    sigma_yx = _uniform(rng, spec.get("sigma_yx_vox", [0.75, 1.75]))
    clip_sd = float(spec.get("clip_standard_deviations", 2.5))

    noise = np_rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=(sigma_z, sigma_yx, sigma_yx), mode="reflect")
    noise -= float(noise.mean())
    std = float(noise.std())
    if std > 1e-8:
        noise /= std
    noise = np.clip(noise, -clip_sd, clip_sd)
    displacement = outward_bias + amplitude * noise

    out = np.zeros_like(mask, dtype=bool)
    fallback_slices = 0
    active = np.where(np.any(mask, axis=(1, 2)))[0]
    for zi in active:
        sdf = _signed_distance_2d(mask[zi], spacing_zyx[1:])
        sl = (sdf + displacement[zi]) > 0.0
        if not sl.any():
            sl = mask[zi].copy()
            fallback_slices += 1
        out[zi] = sl
    return out, {
        "micro_jitter_amplitude_mm": float(amplitude),
        "micro_jitter_outward_bias_mm": float(outward_bias),
        "micro_jitter_sigma_z_vox": float(sigma_z),
        "micro_jitter_sigma_yx_vox": float(sigma_yx),
        "micro_jitter_clip_standard_deviations": float(clip_sd),
        "micro_jitter_empty_slice_fallbacks": int(fallback_slices),
    }


def _apply_endpoint_length(
    mask: np.ndarray,
    spacing_z_mm: float,
    length_mm: float,
    side: str,
    *,
    inferior_is_low_z: bool,
) -> Tuple[np.ndarray, int, float]:
    z = np.where(np.any(mask, axis=(1, 2)))[0]
    if len(z) < 2:
        return mask.copy(), 0, 0.0
    requested_slices = max(1, int(round(float(length_mm) / max(float(spacing_z_mm), 1e-6))))
    n = min(requested_slices, max(len(z) - 1, 1))
    out = np.asarray(mask, dtype=bool).copy()
    remove_low = (side == "inferior" and inferior_is_low_z) or (side == "superior" and not inferior_is_low_z)
    if remove_low:
        cutoff = int(z.min()) + n
        out[:cutoff] = False
    else:
        cutoff = int(z.max()) - n + 1
        out[cutoff:] = False
    return out, int(n), float(n * spacing_z_mm)


def _extend_endpoint_length(
    mask: np.ndarray,
    spacing_z_mm: float,
    length_mm: float,
    side: str,
    *,
    inferior_is_low_z: bool,
) -> Tuple[np.ndarray, int, float]:
    """Extend one S/I endpoint by propagating its terminal cross-section.

    This is intentionally an endpoint-only operation: it does not dilate the
    whole SpinalCord.  The terminal axial cross-section is copied into adjacent
    empty slices up to the requested physical length or the available FOV.
    """
    mask = np.asarray(mask, dtype=bool)
    z = np.where(np.any(mask, axis=(1, 2)))[0]
    if len(z) < 1:
        return mask.copy(), 0, 0.0

    requested_slices = max(1, int(round(float(length_mm) / max(float(spacing_z_mm), 1e-6))))
    extend_low = (side == "inferior" and inferior_is_low_z) or (side == "superior" and not inferior_is_low_z)
    out = mask.copy()
    if extend_low:
        endpoint_z = int(z.min())
        available = endpoint_z
        n = min(requested_slices, available)
        if n <= 0:
            return out, 0, 0.0
        template = out[endpoint_z].copy()
        for zi in range(endpoint_z - n, endpoint_z):
            out[zi] = template
    else:
        endpoint_z = int(z.max())
        available = int(mask.shape[0] - 1 - endpoint_z)
        n = min(requested_slices, available)
        if n <= 0:
            return out, 0, 0.0
        template = out[endpoint_z].copy()
        for zi in range(endpoint_z + 1, endpoint_z + n + 1):
            out[zi] = template

    return out, int(n), float(n * spacing_z_mm)


def length_conditioned_endpoint_v3(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Mapping[str, object],
    rng: random.Random,
    *,
    inferior_is_low_z: bool,
    clean_probability: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Sample v3 endpoint status conditional on approved S/I length.

    The severe-inferior probability model was fitted only on Month1+Month2
    development cases for the event inferior_missing_mm > 20 mm. Probabilities
    in the config are unconditional across all samples; this function is called
    only after a non-clean draw and converts them to conditional probabilities.
    """
    mask = np.asarray(mask, dtype=bool)
    z = np.where(np.any(mask, axis=(1, 2)))[0]
    spacing_z = float(spacing_zyx[0])
    if len(z) < 2:
        return mask.copy(), {
            "endpoint_policy": "length_conditioned_v3",
            "endpoint_requested": 0,
            "endpoint_applied": 0,
            "endpoint_rejected_to_clean": 1,
            "endpoint_rejection_reason": "too_short",
        }

    gt_length_mm = float((int(z.max()) - int(z.min()) + 1) * spacing_z)
    model = spec.get("severe_inferior_probability_model", {})
    intercept = float(model.get("intercept", -8.2152121173))
    slope = float(model.get("slope_per_mm", 0.0194447053))
    development_prevalence = float(model.get("development_prevalence", 0.2575757576))
    raw_weight = float(model.get("raw_weight", 0.75))
    raw_prob = float(1.0 / (1.0 + np.exp(-(intercept + slope * gt_length_mm))))
    severe_target = development_prevalence + raw_weight * (raw_prob - development_prevalence)
    severe_target = float(np.clip(
        severe_target,
        float(model.get("min_unconditional_probability", 0.04)),
        float(model.get("max_unconditional_probability", 0.55)),
    ))
    nonclean_prob = max(1.0 - float(clean_probability), 1e-6)
    severe_cond = severe_target / nonclean_prob
    mild_target = float(spec.get("mild_inferior_unconditional_probability", 0.13))
    superior_target = float(spec.get("superior_mild_unconditional_probability", 0.07))
    mild_cond = mild_target / nonclean_prob
    superior_cond = superior_target / nonclean_prob
    total_cond = severe_cond + mild_cond + superior_cond
    if total_cond > 0.98:
        scale = 0.98 / total_cond
        severe_cond *= scale
        mild_cond *= scale
        superior_cond *= scale

    draw = rng.random()
    event = "none"
    side = ""
    severity_mode = "none"
    requested_length_mm = 0.0
    requested_fraction = 0.0
    if draw < severe_cond:
        event = "inferior_severe"
        side = "inferior"
        severity_mode = "severe"
        frac = _triangular(
            rng,
            spec.get("severe_inferior_fraction_of_gt_length", {}),
            [0.073, 0.131],
        )
        abs_bounds = spec.get("severe_inferior_absolute_length_mm", [27.0, 50.0])
        requested_length_mm = float(np.clip(frac * gt_length_mm, float(abs_bounds[0]), float(abs_bounds[1])))
        requested_fraction = float(requested_length_mm / max(gt_length_mm, 1e-6))
    elif draw < severe_cond + mild_cond:
        event = "inferior_mild"
        side = "inferior"
        severity_mode = "mild"
        requested_length_mm = _uniform(rng, spec.get("mild_inferior_length_mm", [1.5, 3.0]))
        requested_fraction = float(requested_length_mm / max(gt_length_mm, 1e-6))
    elif draw < severe_cond + mild_cond + superior_cond:
        event = "superior_mild"
        side = "superior"
        severity_mode = "mild"
        requested_length_mm = _uniform(rng, spec.get("superior_length_mm", [1.5, 3.0]))
        requested_fraction = float(requested_length_mm / max(gt_length_mm, 1e-6))

    meta: Dict[str, object] = {
        "endpoint_policy": "length_conditioned_v3",
        "endpoint_gt_length_si_mm": gt_length_mm,
        "endpoint_severe_probability_raw": raw_prob,
        "endpoint_severe_probability_unconditional_target": severe_target,
        "endpoint_severe_probability_conditional_nonclean": float(severe_cond),
        "endpoint_mild_inferior_probability_conditional_nonclean": float(mild_cond),
        "endpoint_superior_probability_conditional_nonclean": float(superior_cond),
        "endpoint_event": event,
        "endpoint_requested": int(event != "none"),
        "endpoint_applied": 0,
        "endpoint_side": side,
        "endpoint_severity_mode": severity_mode,
        "endpoint_requested_length_mm": float(requested_length_mm),
        "endpoint_requested_fraction_of_gt_length": float(requested_fraction),
        "inferior_is_low_z": int(bool(inferior_is_low_z)),
    }
    if event == "none":
        return mask.copy(), meta

    out, n, applied_mm = _apply_endpoint_length(
        mask, spacing_z, requested_length_mm, side, inferior_is_low_z=inferior_is_low_z
    )
    if n <= 0 or not out.any():
        meta.update({
            "endpoint_rejected_to_clean": 1,
            "endpoint_rejection_reason": "empty_or_too_short",
        })
        return mask.copy(), meta
    meta.update({
        "endpoint_applied": 1,
        "endpoint_rejected_to_clean": 0,
        "endpoint_applied_slices": int(n),
        "endpoint_applied_length_mm": float(applied_mm),
        "endpoint_applied_fraction_of_gt_length": float(applied_mm / max(gt_length_mm, 1e-6)),
    })
    return out, meta


def length_conditioned_endpoint_v4(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Mapping[str, object],
    rng: random.Random,
    *,
    inferior_is_low_z: bool,
    clean_probability: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """V4 endpoint model with moderate truncation and rare over-extension.

    V3 matched the dominant severe-inferior prevalence but left a hard gap
    between 3 and 20 mm and had no explicit endpoint over-extension.  V4 keeps
    the same length-conditioned severe branch and adds development-calibrated
    fixed-probability events for moderate inferior truncation and rare
    inferior/superior excess.
    """
    mask = np.asarray(mask, dtype=bool)
    z = np.where(np.any(mask, axis=(1, 2)))[0]
    spacing_z = float(spacing_zyx[0])
    if len(z) < 2:
        return mask.copy(), {
            "endpoint_policy": "length_conditioned_v4",
            "endpoint_requested": 0,
            "endpoint_applied": 0,
            "endpoint_rejected_to_clean": 1,
            "endpoint_rejection_reason": "too_short",
        }

    gt_length_mm = float((int(z.max()) - int(z.min()) + 1) * spacing_z)
    model = spec.get("severe_inferior_probability_model", {})
    intercept = float(model.get("intercept", -8.2152121173))
    slope = float(model.get("slope_per_mm", 0.0194447053))
    development_prevalence = float(model.get("development_prevalence", 0.2575757576))
    raw_weight = float(model.get("raw_weight", 0.75))
    raw_prob = float(1.0 / (1.0 + np.exp(-(intercept + slope * gt_length_mm))))
    severe_target = development_prevalence + raw_weight * (raw_prob - development_prevalence)
    severe_target = float(np.clip(
        severe_target,
        float(model.get("min_unconditional_probability", 0.04)),
        float(model.get("max_unconditional_probability", 0.55)),
    ))

    nonclean_prob = max(1.0 - float(clean_probability), 1e-6)
    unconditional = {
        "inferior_severe": severe_target,
        "inferior_moderate": float(spec.get("moderate_inferior_unconditional_probability", 3.0 / 66.0)),
        "inferior_mild": float(spec.get("mild_inferior_unconditional_probability", 0.13)),
        "superior_mild": float(spec.get("superior_mild_unconditional_probability", 0.07)),
        "inferior_excess": float(spec.get("inferior_excess_unconditional_probability", 1.0 / 66.0)),
        "superior_excess": float(spec.get("superior_excess_unconditional_probability", 4.0 / 66.0)),
    }
    conditional = {name: max(0.0, value / nonclean_prob) for name, value in unconditional.items()}
    total_cond = float(sum(conditional.values()))
    scale = 1.0
    if total_cond > 0.98:
        scale = 0.98 / total_cond
        conditional = {name: value * scale for name, value in conditional.items()}

    draw = rng.random()
    cumulative = 0.0
    event = "none"
    for name in (
        "inferior_severe",
        "inferior_moderate",
        "inferior_mild",
        "superior_mild",
        "inferior_excess",
        "superior_excess",
    ):
        cumulative += conditional[name]
        if draw < cumulative:
            event = name
            break

    side = ""
    severity_mode = "none"
    operation = "none"
    requested_length_mm = 0.0
    requested_fraction = 0.0

    if event == "inferior_severe":
        side = "inferior"; severity_mode = "severe"; operation = "missing"
        frac = _triangular(rng, spec.get("severe_inferior_fraction_of_gt_length", {}), [0.073, 0.131])
        abs_bounds = spec.get("severe_inferior_absolute_length_mm", [27.0, 50.0])
        requested_length_mm = float(np.clip(frac * gt_length_mm, float(abs_bounds[0]), float(abs_bounds[1])))
    elif event == "inferior_moderate":
        side = "inferior"; severity_mode = "moderate"; operation = "missing"
        requested_length_mm = _uniform(rng, spec.get("moderate_inferior_length_mm", [4.5, 20.0]))
    elif event == "inferior_mild":
        side = "inferior"; severity_mode = "mild"; operation = "missing"
        requested_length_mm = _uniform(rng, spec.get("mild_inferior_length_mm", [1.5, 3.0]))
    elif event == "superior_mild":
        side = "superior"; severity_mode = "mild"; operation = "missing"
        requested_length_mm = _uniform(rng, spec.get("superior_length_mm", [1.5, 3.0]))
    elif event == "inferior_excess":
        side = "inferior"; severity_mode = "excess"; operation = "excess"
        requested_length_mm = _uniform(rng, spec.get("inferior_excess_length_mm", [30.0, 40.0]))
    elif event == "superior_excess":
        side = "superior"; severity_mode = "excess"; operation = "excess"
        requested_length_mm = _uniform(rng, spec.get("superior_excess_length_mm", [1.5, 9.0]))

    if requested_length_mm > 0:
        requested_fraction = float(requested_length_mm / max(gt_length_mm, 1e-6))

    meta: Dict[str, object] = {
        "endpoint_policy": "length_conditioned_v4",
        "endpoint_gt_length_si_mm": gt_length_mm,
        "endpoint_severe_probability_raw": raw_prob,
        "endpoint_severe_probability_unconditional_target": severe_target,
        "endpoint_probability_rescale_factor": float(scale),
        "endpoint_event": event,
        "endpoint_operation": operation,
        "endpoint_requested": int(event != "none"),
        "endpoint_applied": 0,
        "endpoint_side": side,
        "endpoint_severity_mode": severity_mode,
        "endpoint_requested_length_mm": float(requested_length_mm),
        "endpoint_requested_fraction_of_gt_length": float(requested_fraction),
        "inferior_is_low_z": int(bool(inferior_is_low_z)),
    }
    for name, target in unconditional.items():
        meta[f"endpoint_{name}_probability_unconditional_target"] = float(target)
        meta[f"endpoint_{name}_probability_conditional_nonclean"] = float(conditional[name])

    if event == "none":
        return mask.copy(), meta

    if operation == "missing":
        out, n, applied_mm = _apply_endpoint_length(
            mask, spacing_z, requested_length_mm, side, inferior_is_low_z=inferior_is_low_z
        )
        rejection_reason = "empty_or_too_short"
    else:
        out, n, applied_mm = _extend_endpoint_length(
            mask, spacing_z, requested_length_mm, side, inferior_is_low_z=inferior_is_low_z
        )
        rejection_reason = "insufficient_fov_margin"

    if n <= 0 or not out.any():
        meta.update({
            "endpoint_rejected_to_clean": 1,
            "endpoint_rejection_reason": rejection_reason,
        })
        return mask.copy(), meta

    meta.update({
        "endpoint_applied": 1,
        "endpoint_rejected_to_clean": 0,
        "endpoint_applied_slices": int(n),
        "endpoint_applied_length_mm": float(applied_mm),
        "endpoint_applied_fraction_of_gt_length": float(applied_mm / max(gt_length_mm, 1e-6)),
    })
    return out, meta



def apply_forced_v4_endpoint_event(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    event: str,
    length_mm: float,
    *,
    inferior_is_low_z: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply one explicit V4 endpoint event for diagnostic evaluation.

    This helper uses exactly the same endpoint primitives as the production V4
    sampler but bypasses event-probability sampling.  It therefore changes no
    training behavior and is intended for stratified correction diagnostics.
    """
    original = np.asarray(mask, dtype=bool)
    mapping = {
        "inferior_mild": ("inferior", "missing", "mild"),
        "inferior_moderate": ("inferior", "missing", "moderate"),
        "inferior_severe": ("inferior", "missing", "severe"),
        "superior_mild": ("superior", "missing", "mild"),
        "inferior_excess": ("inferior", "excess", "excess"),
        "superior_excess": ("superior", "excess", "excess"),
    }
    if event not in mapping:
        raise ValueError(f"Unsupported V4 endpoint event: {event}")
    side, operation, severity = mapping[event]
    spacing_z = float(spacing_zyx[0])
    if operation == "missing":
        out, n, applied_mm = _apply_endpoint_length(
            original, spacing_z, float(length_mm), side,
            inferior_is_low_z=inferior_is_low_z,
        )
    else:
        out, n, applied_mm = _extend_endpoint_length(
            original, spacing_z, float(length_mm), side,
            inferior_is_low_z=inferior_is_low_z,
        )
    return out, {
        "endpoint_event": event,
        "endpoint_side": side,
        "endpoint_operation": operation,
        "endpoint_severity_mode": severity,
        "endpoint_requested_length_mm": float(length_mm),
        "endpoint_applied_slices": int(n),
        "endpoint_applied_length_mm": float(applied_mm),
        "endpoint_applied": int(n > 0),
        "inferior_is_low_z": int(bool(inferior_is_low_z)),
    }


def apply_forced_v4_surface_mode(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    cfg: Mapping[str, object],
    mode: str,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply one explicit V4 surface mode while preserving S/I support."""
    original = np.asarray(mask, dtype=bool)
    result = original.copy()
    meta: Dict[str, object] = {"surface_mode": mode}
    if mode == "smooth_warp":
        result, m = _warp_with_candidate_amplitude(
            result, spacing_zyx, cfg.get("smooth_warp", {}), rng
        )
        meta.update(m)
    elif mode == "micro_jitter":
        result, m = transverse_micro_jitter(
            result, spacing_zyx, cfg.get("micro_jitter", {}), rng
        )
        meta.update(m)
    elif mode == "warp_plus_micro_jitter":
        combo = cfg.get("warp_plus_micro_jitter", {})
        result, m1 = _warp_with_candidate_amplitude(
            result, spacing_zyx, combo.get("smooth_warp", {}), rng
        )
        result, m2 = transverse_micro_jitter(
            result, spacing_zyx, combo.get("micro_jitter", {}), rng
        )
        meta.update(m1); meta.update(m2)
    else:
        raise ValueError(f"Unsupported V4 surface mode: {mode}")
    result, restored = _preserve_si_support(original, result)
    meta["surface_si_endpoint_slices_restored"] = int(restored)
    meta["surface_changed_voxel_fraction"] = float(np.logical_xor(original, result).mean())
    return result, meta

def _surface_corruption_v3(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    cfg: Mapping[str, object],
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply v3 surface mode with short-correlation micro-jitter."""
    original = np.asarray(mask, dtype=bool)
    mode = _choose_weighted(rng, cfg["surface_mode_probabilities"])
    result = original.copy()
    meta: Dict[str, object] = {"surface_mode": mode}
    if mode == "smooth_warp":
        result, m1 = _warp_with_candidate_amplitude(result, spacing_zyx, cfg.get("smooth_warp", {}), rng)
        meta.update(m1)
    elif mode == "micro_jitter":
        result, m1 = transverse_micro_jitter(result, spacing_zyx, cfg.get("micro_jitter", {}), rng)
        meta.update(m1)
    elif mode == "warp_plus_micro_jitter":
        combo = cfg.get("warp_plus_micro_jitter", {})
        result, m1 = _warp_with_candidate_amplitude(result, spacing_zyx, combo.get("smooth_warp", {}), rng)
        result, m2 = transverse_micro_jitter(result, spacing_zyx, combo.get("micro_jitter", {}), rng)
        meta.update(m1)
        meta.update(m2)
    else:
        raise ValueError(f"Unknown v3 surface mode: {mode}")
    result, restored = _preserve_si_support(original, result)
    meta["surface_si_endpoint_slices_restored"] = int(restored)
    meta["surface_changed_voxel_fraction"] = float(np.logical_xor(original, result).mean())
    return result, meta


def _proposed_spinalcord_corruption_v3(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    cfg: Mapping[str, object],
    rng: random.Random,
    *,
    inferior_is_low_z: bool,
    cache_key: str | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """V3 sampler: surface base + length-conditioned endpoint + local bite."""
    original = np.asarray(mask, dtype=bool)
    if not original.any():
        raise ValueError("Cannot corrupt an empty SpinalCord mask")
    clean_probability = float(cfg.get("clean_probability", 0.20))
    if rng.random() < clean_probability:
        return original.copy(), {
            "branch": "clean",
            "requested_branch": "clean",
            "surface_mode": "none",
            "endpoint_requested": 0,
            "endpoint_applied": 0,
            "endpoint_event": "none",
            "local_bite_requested": 0,
            "local_bite_applied": 0,
            "composition_label": "clean",
        }

    result, surface_meta = _surface_corruption_v3(original, spacing_zyx, cfg, rng)
    surface_mode = str(surface_meta["surface_mode"])
    branch = f"surface_{surface_mode}"
    meta: Dict[str, object] = dict(surface_meta)

    endpoint_result, endpoint_meta = length_conditioned_endpoint_v3(
        result,
        spacing_zyx,
        cfg.get("endpoint_v3", {}),
        rng,
        inferior_is_low_z=inferior_is_low_z,
        clean_probability=clean_probability,
    )
    meta.update(endpoint_meta)
    endpoint_applied = int(endpoint_meta.get("endpoint_applied", 0))
    if endpoint_applied:
        result = endpoint_result

    local_bite_requested = int(rng.random() < float(cfg.get("local_bite_probability", 0.05)))
    local_bite_applied = 0
    if local_bite_requested:
        bite_result, bite_meta = small_local_bite(
            result, spacing_zyx, cfg.get("local_bite", {}), rng, cache_key=cache_key
        )
        meta.update(bite_meta)
        rejected = int(bite_meta.get("local_bite_rejected_to_clean", 0)) or int(
            bite_meta.get("dropout_rejected_to_clean", 0)
        )
        if not rejected and np.logical_xor(result, bite_result).any():
            result = bite_result
            local_bite_applied = 1

    if not result.any():
        result = original.copy()
        branch = "rejected_to_clean"
        meta["empty_result_fallback"] = 1
    if not np.logical_xor(original, result).any():
        branch = "rejected_to_clean"
        meta["no_change_fallback"] = 1

    components = [surface_mode]
    if endpoint_applied:
        components.append(str(endpoint_meta.get("endpoint_event", "endpoint")))
    if local_bite_applied:
        components.append("local_bite")
    changed = np.logical_xor(original, result)
    return result, {
        "branch": branch,
        "requested_branch": f"surface_{surface_mode}",
        "surface_mode": surface_mode,
        "endpoint_requested": int(endpoint_meta.get("endpoint_requested", 0)),
        "endpoint_applied": endpoint_applied,
        "local_bite_requested": local_bite_requested,
        "local_bite_applied": local_bite_applied,
        "composition_label": "+".join(components),
        "changed_voxel_fraction": float(changed.mean()),
        "input_target_volume_change_pct": float(
            100.0 * abs(int(result.sum()) - int(original.sum())) / max(int(original.sum()), 1)
        ),
        "retained_volume_fraction": float(result.sum() / max(int(original.sum()), 1)),
        **meta,
    }

def _proposed_spinalcord_corruption_v4(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    cfg: Mapping[str, object],
    rng: random.Random,
    *,
    inferior_is_low_z: bool,
    cache_key: str | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """V4 sampler: v3 surface base + corrected endpoint support + uncached bite."""
    original = np.asarray(mask, dtype=bool)
    if not original.any():
        raise ValueError("Cannot corrupt an empty SpinalCord mask")
    clean_probability = float(cfg.get("clean_probability", 0.20))
    if rng.random() < clean_probability:
        return original.copy(), {
            "branch": "clean",
            "requested_branch": "clean",
            "surface_mode": "none",
            "endpoint_requested": 0,
            "endpoint_applied": 0,
            "endpoint_event": "none",
            "local_bite_requested": 0,
            "local_bite_applied": 0,
            "local_bite_geometry_cache_disabled": 1,
            "composition_label": "clean",
        }

    result, surface_meta = _surface_corruption_v3(original, spacing_zyx, cfg, rng)
    surface_mode = str(surface_meta["surface_mode"])
    branch = f"surface_{surface_mode}"
    meta: Dict[str, object] = dict(surface_meta)

    endpoint_result, endpoint_meta = length_conditioned_endpoint_v4(
        result,
        spacing_zyx,
        cfg.get("endpoint_v4", {}),
        rng,
        inferior_is_low_z=inferior_is_low_z,
        clean_probability=clean_probability,
    )
    meta.update(endpoint_meta)
    endpoint_applied = int(endpoint_meta.get("endpoint_applied", 0))
    if endpoint_applied:
        result = endpoint_result

    local_bite_requested = int(rng.random() < float(cfg.get("local_bite_probability", 0.05)))
    local_bite_applied = 0
    if local_bite_requested:
        # V4 deliberately disables the patient-level target-geometry cache.
        # The local-bite input is the already surface/endpoint-corrupted mask,
        # so its geometry is not invariant across samples or epochs.
        bite_result, bite_meta = small_local_bite(
            result, spacing_zyx, cfg.get("local_bite", {}), rng, cache_key=None
        )
        meta.update(bite_meta)
        rejected = int(bite_meta.get("local_bite_rejected_to_clean", 0)) or int(
            bite_meta.get("dropout_rejected_to_clean", 0)
        )
        if not rejected and np.logical_xor(result, bite_result).any():
            result = bite_result
            local_bite_applied = 1
    meta["local_bite_geometry_cache_disabled"] = 1

    if not result.any():
        result = original.copy()
        branch = "rejected_to_clean"
        meta["empty_result_fallback"] = 1
    if not np.logical_xor(original, result).any():
        branch = "rejected_to_clean"
        meta["no_change_fallback"] = 1

    components = [surface_mode]
    if endpoint_applied:
        components.append(str(endpoint_meta.get("endpoint_event", "endpoint")))
    if local_bite_applied:
        components.append("local_bite")
    changed = np.logical_xor(original, result)
    return result, {
        "branch": branch,
        "requested_branch": f"surface_{surface_mode}",
        "surface_mode": surface_mode,
        "endpoint_requested": int(endpoint_meta.get("endpoint_requested", 0)),
        "endpoint_applied": endpoint_applied,
        "local_bite_requested": local_bite_requested,
        "local_bite_applied": local_bite_applied,
        "composition_label": "+".join(components),
        "changed_voxel_fraction": float(changed.mean()),
        "input_target_volume_change_pct": float(
            100.0 * abs(int(result.sum()) - int(original.sum())) / max(int(original.sum()), 1)
        ),
        "retained_volume_fraction": float(result.sum() / max(int(original.sum()), 1)),
        **meta,
    }


def _surface_corruption_v2(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    cfg: Mapping[str, object],
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply one compositional-v2 surface mode while preserving S/I support."""
    original = np.asarray(mask, dtype=bool)
    mode = _choose_weighted(rng, cfg["surface_mode_probabilities"])
    result = original.copy()
    meta: Dict[str, object] = {"surface_mode": mode}

    if mode == "smooth_warp":
        result, m1 = _warp_with_candidate_amplitude(
            result, spacing_zyx, cfg.get("smooth_warp", {}), rng
        )
        meta.update(m1)
    elif mode == "transverse_distance_noise":
        result, m1 = transverse_correlated_boundary_noise(
            result, spacing_zyx, cfg.get("transverse_distance_noise", {}), rng
        )
        meta.update(m1)
    elif mode == "warp_plus_transverse_noise":
        combo = cfg.get("warp_plus_transverse_noise", {})
        result, m1 = _warp_with_candidate_amplitude(
            result, spacing_zyx, combo.get("smooth_warp", {}), rng
        )
        result, m2 = transverse_correlated_boundary_noise(
            result, spacing_zyx, combo.get("transverse_distance_noise", {}), rng
        )
        meta.update(m1)
        meta.update(m2)
    else:
        raise ValueError(f"Unknown v2 surface mode: {mode}")

    result, restored = _preserve_si_support(original, result)
    meta["surface_si_endpoint_slices_restored"] = int(restored)
    meta["surface_changed_voxel_fraction"] = float(np.logical_xor(original, result).mean())
    return result, meta


def _proposed_spinalcord_corruption_v2(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    cfg: Mapping[str, object],
    rng: random.Random,
    *,
    inferior_is_low_z: bool,
    cache_key: str | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Compositional v2 sampler: surface base + optional endpoint + local bite."""
    original = np.asarray(mask, dtype=bool)
    if not original.any():
        raise ValueError("Cannot corrupt an empty SpinalCord mask")

    clean_probability = float(cfg.get("clean_probability", 0.20))
    if rng.random() < clean_probability:
        return original.copy(), {
            "branch": "clean",
            "requested_branch": "clean",
            "surface_mode": "none",
            "endpoint_requested": 0,
            "endpoint_applied": 0,
            "local_bite_requested": 0,
            "local_bite_applied": 0,
            "composition_label": "clean",
        }

    result, surface_meta = _surface_corruption_v2(original, spacing_zyx, cfg, rng)
    surface_mode = str(surface_meta["surface_mode"])
    branch = f"surface_{surface_mode}"
    meta: Dict[str, object] = dict(surface_meta)

    endpoint_requested = int(rng.random() < float(cfg.get("endpoint_probability", 0.40)))
    endpoint_applied = 0
    if endpoint_requested:
        endpoint_result, endpoint_meta = endpoint_truncation(
            result,
            spacing_zyx,
            cfg.get("endpoint_truncation", {}),
            rng,
            inferior_is_low_z=inferior_is_low_z,
        )
        meta.update(endpoint_meta)
        if not int(endpoint_meta.get("endpoint_rejected_to_clean", 0)):
            result = endpoint_result
            endpoint_applied = 1

    local_bite_requested = int(rng.random() < float(cfg.get("local_bite_probability", 0.05)))
    local_bite_applied = 0
    if local_bite_requested:
        bite_result, bite_meta = small_local_bite(
            result,
            spacing_zyx,
            cfg.get("local_bite", {}),
            rng,
            cache_key=cache_key,
        )
        meta.update(bite_meta)
        rejected = int(bite_meta.get("local_bite_rejected_to_clean", 0)) or int(
            bite_meta.get("dropout_rejected_to_clean", 0)
        )
        if not rejected and np.logical_xor(result, bite_result).any():
            result = bite_result
            local_bite_applied = 1

    if not result.any():
        result = original.copy()
        branch = "rejected_to_clean"
        meta["empty_result_fallback"] = 1

    if not np.logical_xor(original, result).any():
        # Keep no-op samples visible in the audit instead of silently calling
        # them a successful corruption.
        branch = "rejected_to_clean"
        meta["no_change_fallback"] = 1

    components = [surface_mode]
    if endpoint_applied:
        components.append("endpoint")
    if local_bite_applied:
        components.append("local_bite")
    composition_label = "+".join(components)

    changed = np.logical_xor(original, result)
    return result, {
        "branch": branch,
        "requested_branch": f"surface_{surface_mode}",
        "surface_mode": surface_mode,
        "endpoint_requested": endpoint_requested,
        "endpoint_applied": endpoint_applied,
        "local_bite_requested": local_bite_requested,
        "local_bite_applied": local_bite_applied,
        "composition_label": composition_label,
        "changed_voxel_fraction": float(changed.mean()),
        "input_target_volume_change_pct": float(
            100.0 * abs(int(result.sum()) - int(original.sum())) / max(int(original.sum()), 1)
        ),
        "retained_volume_fraction": float(result.sum() / max(int(original.sum()), 1)),
        **meta,
    }

def proposed_spinalcord_corruption(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    cfg: Mapping[str, object],
    rng: random.Random,
    *,
    inferior_is_low_z: bool,
    cache_key: str | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Sample one corruption from the requested candidate distribution.

    ``sampling_strategy`` selects the requested surface-first composition;
    configs without a compositional flag retain the original v1 mutually-
    exclusive sampler for reproducibility.
    """
    strategy = str(cfg.get("sampling_strategy", "exclusive_v1"))
    if strategy == "compositional_v4":
        return _proposed_spinalcord_corruption_v4(
            mask,
            spacing_zyx,
            cfg,
            rng,
            inferior_is_low_z=inferior_is_low_z,
            cache_key=cache_key,
        )
    if strategy == "compositional_v3":
        return _proposed_spinalcord_corruption_v3(
            mask,
            spacing_zyx,
            cfg,
            rng,
            inferior_is_low_z=inferior_is_low_z,
            cache_key=cache_key,
        )
    if strategy == "compositional_v2":
        return _proposed_spinalcord_corruption_v2(
            mask,
            spacing_zyx,
            cfg,
            rng,
            inferior_is_low_z=inferior_is_low_z,
            cache_key=cache_key,
        )

    original = np.asarray(mask, dtype=bool)
    if not original.any():
        raise ValueError("Cannot corrupt an empty SpinalCord mask")
    clean_probability = float(cfg.get("clean_probability", 0.20))
    if rng.random() < clean_probability:
        return original.copy(), {"branch": "clean", "requested_branch": "clean"}

    branch = _choose_weighted(rng, cfg["nonclean_branch_probabilities"])
    requested = branch
    result = original.copy()
    meta: Dict[str, object] = {}

    if branch == "smooth_warp":
        result, meta = _warp_with_candidate_amplitude(result, spacing_zyx, cfg.get("smooth_warp", {}), rng)
    elif branch == "transverse_distance_noise":
        result, meta = transverse_correlated_boundary_noise(result, spacing_zyx, cfg.get("transverse_distance_noise", {}), rng)
    elif branch == "combined":
        result, meta1 = _warp_with_candidate_amplitude(result, spacing_zyx, cfg.get("combined", {}).get("smooth_warp", {}), rng)
        result, meta2 = transverse_correlated_boundary_noise(result, spacing_zyx, cfg.get("combined", {}).get("transverse_distance_noise", {}), rng)
        meta = {**meta1, **meta2}
    elif branch == "endpoint_truncation":
        result, meta = endpoint_truncation(
            original, spacing_zyx, cfg.get("endpoint_truncation", {}), rng,
            inferior_is_low_z=inferior_is_low_z,
        )
        if int(meta.get("endpoint_rejected_to_clean", 0)):
            branch = "rejected_to_clean"
    elif branch == "local_bite":
        result, meta = small_local_bite(
            original, spacing_zyx, cfg.get("local_bite", {}), rng, cache_key=cache_key,
        )
        if int(meta.get("local_bite_rejected_to_clean", 0)) or int(meta.get("dropout_rejected_to_clean", 0)):
            branch = "rejected_to_clean"
    elif branch == "transverse_boundary_offset":
        result, meta = transverse_boundary_offset(result, spacing_zyx, cfg.get("transverse_boundary_offset", {}), rng)
    else:
        raise ValueError(f"Unknown proposed branch: {branch}")

    if not result.any():
        result = original.copy()
        branch = "rejected_to_clean"
        meta["empty_result_fallback"] = 1

    changed = np.logical_xor(original, result)
    return result, {
        "branch": branch,
        "requested_branch": requested,
        "changed_voxel_fraction": float(changed.mean()),
        "input_target_volume_change_pct": float(100.0 * abs(int(result.sum()) - int(original.sum())) / max(int(original.sum()), 1)),
        "retained_volume_fraction": float(result.sum() / max(int(original.sum()), 1)),
        **meta,
    }
