#!/usr/bin/env python3
"""Benchmark-independent continuous 3D contour corruptions.

The module intentionally contains no erosion, dilation, translation, cut,
blob, or bite operators. Approved masks are corrupted with smooth displacement
fields and/or spatially correlated signed-distance noise.
"""
from __future__ import annotations

import random
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage


def _uniform(rng: random.Random, bounds: Sequence[float]) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def _integer_triplet(rng: random.Random, lo: Sequence[int], hi: Sequence[int]) -> Tuple[int, int, int]:
    return tuple(rng.randint(int(a), int(b)) for a, b in zip(lo, hi))


def signed_distance(mask: np.ndarray, spacing_zyx: Sequence[float]) -> np.ndarray:
    """Positive inside, negative outside signed distance in millimetres."""
    mask = mask.astype(bool)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing_zyx)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing_zyx)
    return inside - outside


def _upsample_field(coarse: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    zoom = [float(s) / float(c) for s, c in zip(shape, coarse.shape)]
    field = ndimage.zoom(coarse, zoom=zoom, order=3, mode="reflect", prefilter=True)
    slices = tuple(slice(0, int(s)) for s in shape)
    field = field[slices]
    if field.shape != tuple(shape):
        pad = [(0, max(0, int(s) - int(g))) for s, g in zip(shape, field.shape)]
        field = np.pad(field, pad, mode="edge")
        field = field[slices]
    return field.astype(np.float32, copy=False)


def smooth_random_warp(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Dict,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Warp a mask with a spatially smooth random displacement field."""
    shape = mask.shape
    grid_cfg = spec.get("control_grid_points_zyx", {})
    grid_min = grid_cfg.get("min", [4, 4, 4])
    grid_max = grid_cfg.get("max", [8, 8, 8])
    grid_shape = _integer_triplet(rng, grid_min, grid_max)

    np_rng = np.random.default_rng(rng.randrange(2**32))
    displacement_vox = []
    sigma_vox = _uniform(rng, spec.get("gaussian_sigma_vox", [4.0, 10.0]))
    max_displacement_mm = _uniform(rng, spec.get("max_displacement_mm", [1.5, 6.0]))

    for axis in range(3):
        coarse = np_rng.normal(0.0, 1.0, size=grid_shape).astype(np.float32)
        field = _upsample_field(coarse, shape)
        field = ndimage.gaussian_filter(field, sigma=sigma_vox, mode="reflect")
        displacement_vox.append(field)

    disp = np.stack(displacement_vox, axis=0)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)[:, None, None, None]
    magnitude_mm = np.sqrt(np.sum((disp * spacing) ** 2, axis=0))
    current_max = float(np.max(magnitude_mm))
    if current_max > 0:
        disp *= max_displacement_mm / current_max

    coords = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    sample_coords = [coords[a] + disp[a] for a in range(3)]
    sdf = signed_distance(mask, spacing_zyx)
    warped_sdf = ndimage.map_coordinates(
        sdf,
        sample_coords,
        order=1,
        mode="constant",
        cval=float(np.min(sdf)),
        prefilter=False,
    )
    result = warped_sdf > 0.0
    return result, {
        "max_displacement_mm": float(max_displacement_mm),
        "warp_sigma_vox": float(sigma_vox),
        "control_grid_z": int(grid_shape[0]),
        "control_grid_y": int(grid_shape[1]),
        "control_grid_x": int(grid_shape[2]),
    }


def _cleanup_small_components(mask: np.ndarray, spec: Dict) -> Tuple[np.ndarray, Dict[str, int]]:
    """Remove thresholding speckle without imposing benchmark-like morphology."""
    if not bool(spec.get("remove_small_components", False)):
        return mask.astype(bool, copy=False), {
            "removed_small_component_voxels": 0,
            "removed_small_component_count": 0,
        }

    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return mask.astype(bool, copy=False), {
            "removed_small_component_voxels": 0,
            "removed_small_component_count": 0,
        }

    sizes = np.bincount(labels.ravel())
    foreground_sizes = sizes[1:]
    keep = np.zeros(count + 1, dtype=bool)

    if bool(spec.get("keep_largest_component", False)):
        keep[1 + int(np.argmax(foreground_sizes))] = True
    else:
        minimum = max(1, int(spec.get("min_component_voxels", 1)))
        keep[1:] = foreground_sizes >= minimum
        # Never erase all foreground solely because every component was small.
        if not np.any(keep[1:]):
            keep[1 + int(np.argmax(foreground_sizes))] = True

    cleaned = keep[labels]
    removed = mask.astype(bool) & ~cleaned
    return cleaned, {
        "removed_small_component_voxels": int(removed.sum()),
        "removed_small_component_count": int(count - int(keep[1:].sum())),
    }


def signed_distance_noise(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    spec: Dict,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Perturb the implicit surface with correlated zero-mean noise."""
    np_rng = np.random.default_rng(rng.randrange(2**32))
    sigma_vox = _uniform(rng, spec.get("gaussian_sigma_vox", [1.5, 6.0]))
    amplitude_mm = _uniform(rng, spec.get("amplitude_mm", [0.5, 3.0]))

    noise = np_rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=sigma_vox, mode="reflect")
    noise -= float(noise.mean())
    scale = float(noise.std())
    if scale > 1e-8:
        noise /= scale
    noise = np.clip(noise, -3.0, 3.0)

    sdf = signed_distance(mask, spacing_zyx)
    result = (sdf + amplitude_mm * noise) > 0.0
    result, cleanup_meta = _cleanup_small_components(result, spec)
    metadata = {
        "distance_noise_amplitude_mm": float(amplitude_mm),
        "distance_noise_sigma_vox": float(sigma_vox),
    }
    metadata.update(cleanup_meta)
    return result, metadata


def _topology_metrics(mask: np.ndarray) -> Dict[str, float]:
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count > 0:
        sizes = np.bincount(labels.ravel())[1:]
        largest_fraction = float(sizes.max() / max(int(mask.sum()), 1))
    else:
        largest_fraction = 0.0
    try:
        from skimage.measure import euler_number
        euler = float(euler_number(mask.astype(np.uint8), connectivity=1))
    except Exception:
        euler = float("nan")
    return {
        "component_count": int(count),
        "largest_component_fraction": largest_fraction,
        "euler_characteristic": euler,
    }


def deformation_noise_corruption(
    clean: torch.Tensor,
    cfg: Dict,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Corrupt one [1,D,H,W] binary mask and return mask/change/metadata."""
    if clean.ndim != 4 or clean.shape[0] != 1:
        raise ValueError(f"Expected [1,D,H,W], got {tuple(clean.shape)}")

    spec = cfg["deformation_noise"]
    spacing_zyx = tuple(float(v) for v in spec.get("spacing_zyx", [1.5, 1.5, 1.5]))
    original = clean.detach().cpu().numpy()[0] > 0.5
    if not original.any():
        raise ValueError("Cannot corrupt an empty approved contour.")

    clean_probability = float(spec.get("clean_probability", 0.20))
    if rng.random() < clean_probability:
        result = original.copy()
        metadata: Dict[str, object] = {
            "branch": "clean",
            "accepted_attempt": 0,
            "retained_volume_fraction": 1.0,
            "input_target_volume_change_pct": 0.0,
            "changed_voxel_fraction": 0.0,
            "removed_small_component_voxels": 0,
            "removed_small_component_count": 0,
        }
    else:
        branches = spec.get("branch_probabilities", {})
        names = ["smooth_warp", "distance_noise", "combined"]
        weights = [float(branches.get(name, default)) for name, default in zip(names, [0.40, 0.25, 0.15])]
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

        validity = spec.get("validity", {})
        min_retained = float(validity.get("min_retained_volume_fraction", 0.55))
        max_change = float(validity.get("max_volume_change_fraction", 0.50))
        max_attempts = int(validity.get("max_attempts", 8))
        original_volume = float(original.sum())
        original_topology = _topology_metrics(original)
        accepted = None
        accepted_meta: Dict[str, object] = {}
        last_rejected_meta: Dict[str, object] = {}

        for attempt in range(1, max_attempts + 1):
            candidate = original.copy()
            candidate_meta: Dict[str, object] = {"branch": branch, "accepted_attempt": attempt}
            if branch in {"smooth_warp", "combined"}:
                candidate, meta = smooth_random_warp(
                    candidate, spacing_zyx, spec.get("smooth_warp", {}), rng
                )
                candidate_meta.update(meta)
            if branch in {"distance_noise", "combined"}:
                candidate, meta = signed_distance_noise(
                    candidate, spacing_zyx, spec.get("distance_noise", {}), rng
                )
                candidate_meta.update(meta)

            candidate_volume = float(candidate.sum())
            retained = candidate_volume / max(original_volume, 1.0)
            volume_change = abs(candidate_volume - original_volume) / max(original_volume, 1.0)
            changed_fraction = float(np.mean(candidate != original))
            candidate_topology = _topology_metrics(candidate)

            max_components = validity.get("max_component_count", None)
            max_extra_components = validity.get("max_extra_components", None)
            if max_extra_components is not None:
                allowed_component_count = (
                    int(original_topology["component_count"])
                    + int(max_extra_components)
                )
            elif max_components is not None:
                allowed_component_count = int(max_components)
            else:
                allowed_component_count = None

            min_largest_fraction = float(
                validity.get("min_largest_component_fraction", 0.0)
            )
            min_largest_fraction_ratio = validity.get(
                "min_largest_component_fraction_ratio", None
            )
            if min_largest_fraction_ratio is not None:
                relative_minimum = (
                    float(original_topology["largest_component_fraction"])
                    * float(min_largest_fraction_ratio)
                )
                absolute_floor = float(
                    validity.get(
                        "min_absolute_largest_component_fraction",
                        0.0,
                    )
                )
                min_largest_fraction = max(relative_minimum, absolute_floor)

            rejection_reasons = []
            if not candidate.any():
                rejection_reasons.append("empty_mask")
            if np.array_equal(candidate, original):
                rejection_reasons.append("unchanged_mask")
            if retained < min_retained:
                rejection_reasons.append("retained_volume_fraction_below_minimum")
            if volume_change > max_change:
                rejection_reasons.append("volume_change_fraction_above_maximum")
            if (
                allowed_component_count is not None
                and candidate_topology["component_count"] > allowed_component_count
            ):
                rejection_reasons.append("too_many_components_relative_to_target")
            if (
                candidate_topology["largest_component_fraction"]
                < min_largest_fraction
            ):
                rejection_reasons.append(
                    "largest_component_fraction_below_relative_threshold"
                )

            valid = not rejection_reasons
            if not valid:
                last_rejected_meta = {
                    **candidate_meta,
                    "requested_branch": branch,
                    "rejection_reason": ";".join(rejection_reasons),
                    "last_rejected_component_count": int(
                        candidate_topology["component_count"]
                    ),
                    "last_rejected_largest_component_fraction": float(
                        candidate_topology["largest_component_fraction"]
                    ),
                    "last_rejected_changed_fraction_of_target": float(
                        np.count_nonzero(candidate != original)
                        / max(original_volume, 1.0)
                    ),
                    "allowed_component_count": allowed_component_count,
                    "required_largest_component_fraction": float(
                        min_largest_fraction
                    ),
                }

            if valid:
                accepted = candidate
                candidate_meta.update({
                    "retained_volume_fraction": retained,
                    "input_target_volume_change_pct": 100.0 * volume_change,
                    "changed_voxel_fraction": changed_fraction,
                })
                accepted_meta = candidate_meta
                break

        if accepted is None:
            # Keep the target unchanged rather than introduce a benchmark-like
            # fallback operation. Preserve the final rejected-attempt diagnostics.
            result = original.copy()
            metadata = {
                **last_rejected_meta,
                "requested_branch": branch,
                "branch": "rejected_to_clean",
                "accepted_attempt": max_attempts,
                "retained_volume_fraction": 1.0,
                "input_target_volume_change_pct": 0.0,
                "changed_voxel_fraction": 0.0,
            }
        else:
            result = accepted
            metadata = accepted_meta

    out = torch.from_numpy(result.astype(np.float32))[None].to(clean.device)
    changed = (out != clean).float()
    original_topology = _topology_metrics(original)
    corrupted_topology = _topology_metrics(result)
    changed_voxels = int(np.count_nonzero(result != original))
    target_voxels = int(original.sum())
    union_voxels = int(np.count_nonzero(result | original))
    metadata.update({
        "changed_voxels": changed_voxels,
        "target_voxels": target_voxels,
        "changed_voxel_fraction": float(changed.mean().item()),
        "changed_fraction_of_target": float(changed_voxels / max(target_voxels, 1)),
        "symmetric_difference_over_union": float(changed_voxels / max(union_voxels, 1)),
        "input_target_volume_change_pct": float(
            100.0 * abs(int(result.sum()) - target_voxels) / max(target_voxels, 1)
        ),
        "target_component_count": original_topology["component_count"],
        "input_component_count": corrupted_topology["component_count"],
        "component_count_change": int(
            corrupted_topology["component_count"] - original_topology["component_count"]
        ),
        "input_largest_component_fraction": corrupted_topology["largest_component_fraction"],
        "input_euler_characteristic": corrupted_topology["euler_characteristic"],
        "removed_small_component_voxels": int(
            metadata.get("removed_small_component_voxels", 0)
        ),
        "removed_small_component_count": int(
            metadata.get("removed_small_component_count", 0)
        ),
    })
    return out, changed, metadata
