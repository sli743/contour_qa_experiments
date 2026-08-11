#!/usr/bin/env python3
"""Benchmark-independent stochastic missing-region corruptions.

The new branch generates a smooth random scalar field, localizes it around a
random target-boundary seed, and removes one connected target-intersecting
region. Acceptance is controlled by generic descriptors (removed fraction,
penetration depth, connectivity), not by benchmark probe orientation or level.

Performance note: target-invariant geometry is computed once per corruption
sample and reused across rejection attempts. Expensive acceptance checks are
ordered from cheap to expensive, and random-field smoothing is performed on
the coarse control grid before the single full-resolution interpolation.
"""
from __future__ import annotations

import random
import time
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage

from training.corruptions.deformation_noise import (
    _uniform,
    signed_distance,
    signed_distance_noise,
    smooth_random_warp,
)


_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)
_TARGET_GEOMETRY_CACHE: Dict[str, Dict[str, object]] = {}
_COORD_CACHE: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _cache_geometry(cache_key: str | None, target: np.ndarray, spacing_zyx: Sequence[float]) -> Dict[str, object]:
    """Worker-local cache for target-invariant geometry."""
    if cache_key and cache_key in _TARGET_GEOMETRY_CACHE:
        return _TARGET_GEOMETRY_CACHE[cache_key]
    boundary = _boundary(target)
    boundary_points = np.argwhere(boundary)
    if len(boundary_points) == 0:
        raise ValueError("Target has no boundary voxels.")
    geometry = {
        "boundary_points": boundary_points,
        "target_voxels": int(target.sum()),
        "depth_map_mm": ndimage.distance_transform_edt(target, sampling=spacing_zyx).astype(np.float32, copy=False),
        "target_component_count": _component_count(target),
    }
    if cache_key:
        _TARGET_GEOMETRY_CACHE[cache_key] = geometry
    return geometry


def _coarse_sigma_from_mm(shape: Sequence[int], grid_shape: Sequence[int], spacing_zyx: Sequence[float], sigma_mm: float) -> Tuple[float, float, float]:
    """Convert a physical correlation length to coarse-grid Gaussian sigmas."""
    out = []
    for n, g, sp in zip(shape, grid_shape, spacing_zyx):
        physical_extent = max((int(n) - 1) * float(sp), float(sp))
        coarse_spacing = physical_extent / max(int(g) - 1, 1)
        out.append(max(float(sigma_mm) / coarse_spacing, 0.35))
    return tuple(out)


def _smooth_coarse_random_field(np_rng: np.random.Generator, grid_shape: Sequence[int], shape: Sequence[int], spacing_zyx: Sequence[float], correlation_mm: float, order: int = 1) -> np.ndarray:
    coarse = np_rng.normal(0.0, 1.0, size=tuple(int(v) for v in grid_shape)).astype(np.float32)
    coarse = ndimage.gaussian_filter(
        coarse,
        sigma=_coarse_sigma_from_mm(shape, grid_shape, spacing_zyx, correlation_mm),
        mode="reflect",
    )
    field = _upsample_field_fast(coarse, shape, order=order)
    field -= float(field.mean())
    std = float(field.std())
    if std > 1e-8:
        field /= std
    return field


def _base_coords(shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = tuple(int(v) for v in shape)
    coords = _COORD_CACHE.get(key)
    if coords is None:
        coords = tuple(np.arange(v, dtype=np.float32) for v in key)
        _COORD_CACHE[key] = coords
    return coords


def smooth_random_warp_fast(mask: np.ndarray, spacing_zyx: Sequence[float], spec: Dict, rng: random.Random) -> Tuple[np.ndarray, Dict[str, float]]:
    """Smooth warp using coarse-grid smoothing before full-resolution interpolation."""
    shape = mask.shape
    grid_cfg = spec.get("control_grid_points_zyx", {})
    grid_min = grid_cfg.get("min", [4, 4, 4])
    grid_max = grid_cfg.get("max", [8, 8, 8])
    grid_shape = tuple(rng.randint(int(a), int(b)) for a, b in zip(grid_min, grid_max))
    np_rng = np.random.default_rng(rng.randrange(2**32))
    sigma_vox = _uniform(rng, spec.get("gaussian_sigma_vox", [4.0, 10.0]))
    # Preserve the old parameter's approximate physical meaning.
    sigma_mm = float(sigma_vox) * float(np.mean(np.asarray(spacing_zyx, dtype=np.float32)))
    max_displacement_mm = _uniform(rng, spec.get("max_displacement_mm", [1.5, 6.0]))
    displacement_vox = [
        _smooth_coarse_random_field(np_rng, grid_shape, shape, spacing_zyx, sigma_mm, order=int(spec.get("upsample_order", 1)))
        for _ in range(3)
    ]
    disp = np.stack(displacement_vox, axis=0)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)[:, None, None, None]
    magnitude_mm = np.sqrt(np.sum((disp * spacing) ** 2, axis=0))
    current_max = float(np.max(magnitude_mm))
    if current_max > 0:
        disp *= max_displacement_mm / current_max
    z, y, x = _base_coords(shape)
    sample_coords = [
        z[:, None, None] + disp[0],
        y[None, :, None] + disp[1],
        x[None, None, :] + disp[2],
    ]
    from training.corruptions.deformation_noise import signed_distance
    sdf = signed_distance(mask, spacing_zyx)
    warped_sdf = ndimage.map_coordinates(
        sdf, sample_coords, order=1, mode="constant", cval=float(np.min(sdf)), prefilter=False
    )
    return warped_sdf > 0.0, {
        "max_displacement_mm": float(max_displacement_mm),
        "warp_sigma_vox": float(sigma_vox),
        "control_grid_z": int(grid_shape[0]),
        "control_grid_y": int(grid_shape[1]),
        "control_grid_x": int(grid_shape[2]),
        "warp_fast_coarse_smoothing": 1,
    }



def global_signed_distance_offset(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Dict,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply a generic whole-surface offset in physical distance space.

    This is intentionally different from the benchmark's discrete binary
    erosion/dilation operators.  The contour is represented by a signed
    distance field (positive inside), and a continuously sampled signed
    physical offset is applied to the zero level set.

    Positive offset -> outward boundary displacement.
    Negative offset -> inward boundary displacement.
    """
    mask = mask.astype(bool, copy=False)
    if not mask.any():
        raise ValueError("Cannot offset an empty mask.")

    magnitude_bounds = spec.get("magnitude_mm", [1.5, 6.0])
    magnitude_mm = _uniform(rng, magnitude_bounds)
    outward_probability = float(spec.get("outward_probability", 0.50))
    outward = rng.random() < outward_probability
    offset_mm = float(magnitude_mm if outward else -magnitude_mm)

    sdf = signed_distance(mask, spacing_zyx)
    result = (sdf + offset_mm) > 0.0

    # A large inward offset could erase an unusually small structure.  Reject
    # such a sample rather than silently changing the target semantics.
    if not result.any():
        return mask.copy(), {
            "global_boundary_offset_mm": offset_mm,
            "global_boundary_offset_magnitude_mm": float(magnitude_mm),
            "global_boundary_offset_direction": "outward" if outward else "inward",
            "global_boundary_offset_rejected_to_clean": 1,
            "global_boundary_offset_rejection_reason": "empty_result",
        }

    target_voxels = int(mask.sum())
    input_voxels = int(result.sum())
    return result, {
        "global_boundary_offset_mm": offset_mm,
        "global_boundary_offset_magnitude_mm": float(magnitude_mm),
        "global_boundary_offset_direction": "outward" if outward else "inward",
        "global_boundary_offset_rejected_to_clean": 0,
        "global_boundary_offset_volume_change_pct": float(
            100.0 * abs(input_voxels - target_voxels) / max(target_voxels, 1)
        ),
    }

def _best_seed_component_for_fraction(score: np.ndarray, target: np.ndarray, seed: Sequence[int], target_fraction: float, target_voxels: int, search_steps: int) -> Tuple[np.ndarray, float, float]:
    """Binary-search a threshold so the seed-connected component matches target volume."""
    values = score[target]
    lo = float(np.min(values))
    hi = float(np.max(values))
    desired = float(target_fraction) * max(int(target_voxels), 1)
    best_region = np.zeros_like(target, dtype=bool)
    best_threshold = hi
    best_error = float("inf")
    for _ in range(max(2, int(search_steps))):
        threshold = 0.5 * (lo + hi)
        region = _component_containing_seed(target & (score >= threshold), seed)
        size = int(region.sum())
        error = abs(size - desired)
        if error < best_error:
            best_error = error
            best_region = region
            best_threshold = threshold
        # Raising threshold can only shrink the super-level set.
        if size > desired:
            lo = threshold
        else:
            hi = threshold
    return best_region, float(best_threshold), float(best_error / max(target_voxels, 1))


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=_STRUCTURE_6, border_value=0)
    return mask & ~eroded


def _component_containing_seed(region: np.ndarray, seed_zyx: Sequence[int]) -> np.ndarray:
    labels, count = ndimage.label(region, structure=_STRUCTURE_6)
    if count == 0:
        return np.zeros_like(region, dtype=bool)
    label = int(labels[tuple(int(v) for v in seed_zyx)])
    if label > 0:
        return labels == label
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (1 + int(np.argmax(sizes)))


def _component_count(mask: np.ndarray) -> int:
    """Only count components; avoid the heavier full topology descriptor."""
    if not mask.any():
        return 0
    _, count = ndimage.label(mask, structure=_STRUCTURE_6)
    return int(count)


def _depth_metrics_from_map(region: np.ndarray, depth_map_mm: np.ndarray) -> Dict[str, float]:
    if not region.any():
        return {
            "missing_depth_mean_mm": 0.0,
            "missing_depth_median_mm": 0.0,
            "missing_depth_p95_mm": 0.0,
            "missing_depth_max_mm": 0.0,
        }
    values = depth_map_mm[region]
    return {
        "missing_depth_mean_mm": float(np.mean(values)),
        "missing_depth_median_mm": float(np.median(values)),
        "missing_depth_p95_mm": float(np.percentile(values, 95)),
        "missing_depth_max_mm": float(np.max(values)),
    }


def _upsample_field_fast(coarse: np.ndarray, shape: Sequence[int], order: int = 1) -> np.ndarray:
    """Interpolate a coarse field once; linear interpolation avoids cubic prefilter cost."""
    zoom = [float(s) / float(c) for s, c in zip(shape, coarse.shape)]
    field = ndimage.zoom(
        coarse, zoom=zoom, order=int(order), mode="reflect",
        prefilter=bool(int(order) > 1),
    )
    slices = tuple(slice(0, int(s)) for s in shape)
    field = field[slices]
    if field.shape != tuple(shape):
        pad = [(0, max(0, int(s) - int(g))) for s, g in zip(shape, field.shape)]
        field = np.pad(field, pad, mode="edge")[slices]
    return field.astype(np.float32, copy=False)


def _localized_score(field: np.ndarray, seed: Sequence[int], spacing: np.ndarray,
                     localization_sigma_mm: float, field_weight: float,
                     localization_weight: float, cutoff_sigma: float) -> np.ndarray:
    """Add a Gaussian localization envelope only in a finite physical ROI."""
    score = (field_weight * field).astype(np.float32, copy=True)
    shape = np.asarray(field.shape, dtype=int)
    seed = np.asarray(seed, dtype=int)
    radius_vox = np.ceil(cutoff_sigma * localization_sigma_mm / spacing).astype(int)
    lo = np.maximum(seed - radius_vox, 0)
    hi = np.minimum(seed + radius_vox + 1, shape)

    z = (np.arange(lo[0], hi[0], dtype=np.float32) - seed[0]) * spacing[0]
    y = (np.arange(lo[1], hi[1], dtype=np.float32) - seed[1]) * spacing[1]
    x = (np.arange(lo[2], hi[2], dtype=np.float32) - seed[2]) * spacing[2]
    d2 = z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2
    envelope = np.exp(-0.5 * d2 / (localization_sigma_mm ** 2)).astype(np.float32, copy=False)
    score[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += localization_weight * envelope
    score[tuple(seed)] += 4.0
    return score


def stochastic_region_dropout(
    target: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Dict,
    rng: random.Random,
    cache_key: str | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Remove one smooth, irregular, boundary-connected target region."""
    overall_start = time.perf_counter()
    target = target.astype(bool, copy=False)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)

    # Worker-local target geometry cache survives across epochs for patients handled by this worker.
    t0 = time.perf_counter()
    geometry = _cache_geometry(cache_key, target, spacing_zyx)
    boundary_points = geometry["boundary_points"]
    target_voxels = int(geometry["target_voxels"])
    depth_map_mm = geometry["depth_map_mm"]
    target_component_count = int(geometry["target_component_count"])
    invariant_seconds = time.perf_counter() - t0

    np_rng = np.random.default_rng(rng.randrange(2**32))
    grid_bounds = spec.get("coarse_grid_points_zyx", {})
    grid_min = grid_bounds.get("min", [4, 4, 4])
    grid_max = grid_bounds.get("max", [8, 8, 8])
    removed_bounds = spec.get("removed_fraction", [0.05, 0.35])
    corr_bounds = spec.get("correlation_length_mm", [15.0, 45.0])
    envelope_bounds = spec.get("localization_sigma_mm", [20.0, 55.0])
    field_weight = float(spec.get("random_field_weight", 1.0))
    localization_weight = float(spec.get("localization_weight", 1.5))
    envelope_cutoff_sigma = float(spec.get("localization_cutoff_sigma", 3.5))
    max_attempts = int(spec.get("max_attempts", 8))
    min_depth = float(spec.get("min_max_depth_mm", 6.0))
    max_depth = float(spec.get("max_max_depth_mm", 55.0))
    fraction_tolerance = float(spec.get("removed_fraction_tolerance", 0.08))
    max_remaining_components = int(spec.get("max_remaining_components", 1))
    max_extra_remaining_components = spec.get("max_extra_remaining_components")
    threshold_search_steps = int(spec.get("threshold_search_steps", 8))

    profile = {
        "dropout_timing_invariant_seconds": invariant_seconds,
        "dropout_timing_field_seconds": 0.0,
        "dropout_timing_envelope_seconds": 0.0,
        "dropout_timing_component_seconds": 0.0,
        "dropout_timing_depth_seconds": 0.0,
        "dropout_timing_topology_seconds": 0.0,
    }
    rejection_counts: Dict[str, int] = {}
    last_meta: Dict[str, object] = {}

    for attempt in range(1, max_attempts + 1):
        grid_shape = tuple(rng.randint(int(a), int(b)) for a, b in zip(grid_min, grid_max))
        correlation_mm = _uniform(rng, corr_bounds)

        # Smooth on the tiny control grid, then interpolate once. This removes the
        # dominant full-volume Gaussian filter from every rejection attempt.
        t0 = time.perf_counter()
        field = _smooth_coarse_random_field(
            np_rng, grid_shape, target.shape, spacing_zyx, correlation_mm,
            order=int(spec.get("upsample_order", 1)),
        )
        profile["dropout_timing_field_seconds"] += time.perf_counter() - t0

        seed = boundary_points[rng.randrange(len(boundary_points))]
        localization_sigma_mm = _uniform(rng, envelope_bounds)
        t0 = time.perf_counter()
        score = _localized_score(
            field, seed, spacing, localization_sigma_mm,
            field_weight, localization_weight, envelope_cutoff_sigma,
        )
        profile["dropout_timing_envelope_seconds"] += time.perf_counter() - t0

        target_fraction = _uniform(rng, removed_bounds)
        # Reuse one random field and search threshold values for the requested
        # seed-connected volume instead of regenerating whole fields repeatedly.
        t0 = time.perf_counter()
        region, threshold, fraction_search_error = _best_seed_component_for_fraction(
            score, target, seed, target_fraction, target_voxels, threshold_search_steps
        )
        profile["dropout_timing_component_seconds"] += time.perf_counter() - t0
        removed_voxels = int(region.sum())
        removed_fraction = removed_voxels / max(target_voxels, 1)

        # Cheap rejection first. Do NOT compute depth/topology for obviously invalid proposals.
        reasons = []
        if removed_voxels == 0:
            reasons.append("empty_removed_region")
        if abs(removed_fraction - target_fraction) > fraction_tolerance:
            reasons.append("removed_fraction_outside_tolerance")
        if reasons:
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            last_meta = {
                "dropout_attempt": attempt,
                "dropout_rejection_reason": ";".join(reasons),
                "dropout_target_removed_fraction": float(target_fraction),
                "dropout_removed_fraction": float(removed_fraction),
                "dropout_removed_voxels": removed_voxels,
                "dropout_correlation_length_mm": float(correlation_mm),
                "dropout_localization_sigma_mm": float(localization_sigma_mm),
                "dropout_threshold": threshold,
                "dropout_fraction_search_error": float(fraction_search_error),
            }
            continue

        # Reuse the single target distance transform here.
        t0 = time.perf_counter()
        depth_meta = _depth_metrics_from_map(region, depth_map_mm)
        profile["dropout_timing_depth_seconds"] += time.perf_counter() - t0
        if depth_meta["missing_depth_max_mm"] < min_depth:
            reasons.append("maximum_depth_below_minimum")
        if depth_meta["missing_depth_max_mm"] > max_depth:
            reasons.append("maximum_depth_above_maximum")

        remaining = target & ~region
        if not remaining.any():
            reasons.append("empty_remaining_mask")
            remaining_component_count = 0
        elif not reasons:
            # Topology is the final expensive acceptance check and only needs component count.
            t0 = time.perf_counter()
            remaining_component_count = _component_count(remaining)
            profile["dropout_timing_topology_seconds"] += time.perf_counter() - t0
            component_limit = (
                target_component_count + int(max_extra_remaining_components)
                if max_extra_remaining_components is not None
                else max_remaining_components
            )
            if remaining_component_count > component_limit:
                reasons.append("too_many_remaining_components")
        else:
            remaining_component_count = -1

        for reason in reasons:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        last_meta = {
            "dropout_attempt": attempt,
            "dropout_rejection_reason": ";".join(reasons),
            "dropout_target_removed_fraction": float(target_fraction),
            "dropout_removed_fraction": float(removed_fraction),
            "dropout_removed_voxels": removed_voxels,
            "dropout_boundary_connected": 1,
            "dropout_correlation_length_mm": float(correlation_mm),
            "dropout_localization_sigma_mm": float(localization_sigma_mm),
            "dropout_threshold": threshold,
            "dropout_fraction_search_error": float(fraction_search_error),
            "dropout_control_grid_z": int(grid_shape[0]),
            "dropout_control_grid_y": int(grid_shape[1]),
            "dropout_control_grid_x": int(grid_shape[2]),
            "dropout_seed_z": int(seed[0]),
            "dropout_seed_y": int(seed[1]),
            "dropout_seed_x": int(seed[2]),
            "dropout_remaining_component_count": int(remaining_component_count),
            **depth_meta,
        }
        if not reasons:
            profile["dropout_total_attempts"] = attempt
            profile["dropout_rejection_counts_json"] = str(rejection_counts)
            profile["dropout_timing_total_seconds"] = time.perf_counter() - overall_start
            return remaining, {**last_meta, **profile}

    profile["dropout_total_attempts"] = max_attempts
    profile["dropout_rejection_counts_json"] = str(rejection_counts)
    profile["dropout_timing_total_seconds"] = time.perf_counter() - overall_start
    return target.copy(), {
        **last_meta,
        **profile,
        "dropout_rejected_to_clean": 1,
    }


def stochastic_region_dropout_corruption(
    clean: torch.Tensor,
    cfg: Dict,
    rng: random.Random,
    cache_key: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Mixed corruption distribution for the first dropout experiment."""
    overall_start = time.perf_counter()
    if clean.ndim != 4 or clean.shape[0] != 1:
        raise ValueError(f"Expected [1,D,H,W], got {tuple(clean.shape)}")

    spec = cfg["stochastic_region_dropout"]
    spacing_zyx = tuple(float(v) for v in spec.get("spacing_zyx", [1.5, 1.5, 1.5]))
    original = clean.detach().cpu().numpy()[0] > 0.5
    if not original.any():
        raise ValueError("Cannot corrupt an empty approved contour.")

    clean_probability = float(spec.get("clean_probability", 0.20))
    branches = spec.get("branch_probabilities", {})
    if rng.random() < clean_probability:
        branch = "clean"
    else:
        names = [
            "smooth_warp",
            "distance_noise",
            "combined",
            "stochastic_region_dropout",
            "global_boundary_offset",
        ]
        defaults = [0.10, 0.20, 0.15, 0.20, 0.15]
        weights = [float(branches.get(name, default)) for name, default in zip(names, defaults)]
        total = sum(weights)
        if total <= 0:
            raise ValueError("At least one non-clean branch probability must be positive.")
        draw = rng.random() * total
        cumulative = 0.0
        branch = names[-1]
        for name, weight in zip(names, weights):
            cumulative += weight
            if draw <= cumulative:
                branch = name
                break

    metadata: Dict[str, object] = {"branch": branch}
    result = original.copy()
    branch_start = time.perf_counter()
    if branch == "smooth_warp":
        result, meta = smooth_random_warp_fast(result, spacing_zyx, spec.get("smooth_warp", {}), rng)
        metadata.update(meta)
    elif branch == "distance_noise":
        result, meta = signed_distance_noise(result, spacing_zyx, spec.get("distance_noise", {}), rng)
        metadata.update(meta)
    elif branch == "combined":
        result, meta = smooth_random_warp_fast(result, spacing_zyx, spec.get("smooth_warp", {}), rng)
        metadata.update(meta)
        result, meta = signed_distance_noise(result, spacing_zyx, spec.get("distance_noise", {}), rng)
        metadata.update(meta)
    elif branch == "stochastic_region_dropout":
        result, meta = stochastic_region_dropout(original, spacing_zyx, spec.get("dropout", {}), rng, cache_key=cache_key)
        metadata.update(meta)
        if int(meta.get("dropout_rejected_to_clean", 0)):
            metadata["requested_branch"] = branch
            metadata["branch"] = "rejected_to_clean"
    elif branch == "global_boundary_offset":
        result, meta = global_signed_distance_offset(
            original, spacing_zyx, spec.get("global_boundary_offset", {}), rng
        )
        metadata.update(meta)
        if int(meta.get("global_boundary_offset_rejected_to_clean", 0)):
            metadata["requested_branch"] = branch
            metadata["branch"] = "rejected_to_clean"
    metadata["branch_corruption_seconds"] = time.perf_counter() - branch_start

    if not result.any():
        result = original.copy()
        metadata["requested_branch"] = metadata.get("branch", branch)
        metadata["branch"] = "rejected_to_clean"
        metadata["empty_result_fallback"] = 1

    changed_np = np.logical_xor(result, original)
    metadata["changed_voxel_fraction"] = float(changed_np.mean())
    metadata["input_target_volume_change_pct"] = float(
        100.0 * abs(int(result.sum()) - int(original.sum())) / max(int(original.sum()), 1)
    )
    metadata["retained_volume_fraction"] = float(result.sum() / max(int(original.sum()), 1))
    metadata["corruption_function_seconds"] = time.perf_counter() - overall_start
    corrupted = torch.from_numpy(result.astype(np.float32, copy=False))[None]
    changed = torch.from_numpy(changed_np.astype(np.float32, copy=False))[None]
    return corrupted, changed, metadata
