"""Spatial-reference utilities for deployment-matched NSR-QA training/evaluation.

The candidate mask remains the only 3DINO input.  Optional global spatial
conditioning is computed outside the encoder from crop/candidate/raw-CT
geometry so the pretrained encoder input representation is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import SimpleITK as sitk
import torch



@dataclass(frozen=True)
class RawCTGeometry:
    size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]


def read_raw_ct_geometry(path: str | Path) -> RawCTGeometry:
    """Read only image header geometry; do not materialize the CT voxel buffer."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return RawCTGeometry(
        size=tuple(int(v) for v in reader.GetSize()),
        spacing=tuple(float(v) for v in reader.GetSpacing()),
        origin=tuple(float(v) for v in reader.GetOrigin()),
        direction=tuple(float(v) for v in reader.GetDirection()),
    )


def _raw_size(raw_ct: sitk.Image | RawCTGeometry) -> np.ndarray:
    return np.asarray(raw_ct.GetSize() if isinstance(raw_ct, sitk.Image) else raw_ct.size, dtype=float)


def _raw_spacing(raw_ct: sitk.Image | RawCTGeometry) -> np.ndarray:
    return np.asarray(raw_ct.GetSpacing() if isinstance(raw_ct, sitk.Image) else raw_ct.spacing, dtype=float)


def _raw_continuous_index(point_phys_xyz: Sequence[float], raw_ct: sitk.Image | RawCTGeometry) -> np.ndarray:
    if isinstance(raw_ct, sitk.Image):
        return np.asarray(
            raw_ct.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in point_phys_xyz)),
            dtype=float,
        )
    point = np.asarray(point_phys_xyz, dtype=float)
    origin = np.asarray(raw_ct.origin, dtype=float)
    spacing = np.asarray(raw_ct.spacing, dtype=float)
    direction = np.asarray(raw_ct.direction, dtype=float).reshape(3, 3)
    return np.linalg.solve(direction, point - origin) / spacing


SPATIAL_CONTEXT_FEATURES = (
    "crop_center_raw_x",
    "crop_center_raw_y",
    "crop_center_raw_z",
    "candidate_center_raw_x",
    "candidate_center_raw_y",
    "candidate_center_raw_z",
    "candidate_span_over_fov_x",
    "candidate_span_over_fov_y",
    "candidate_span_over_fov_z",
    "raw_ct_span_over_fov_x",
    "raw_ct_span_over_fov_y",
    "raw_ct_span_over_fov_z",
)


def _mask_bbox_zyx(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.column_stack(np.where(np.asarray(mask) > 0))
    if coords.size == 0:
        raise ValueError("Cannot define candidate reference frame from an empty mask")
    return coords.min(axis=0).astype(float), coords.max(axis=0).astype(float)


def image_center_physical(img: sitk.Image) -> np.ndarray:
    idx = 0.5 * (np.asarray(img.GetSize(), dtype=float) - 1.0)
    return np.asarray(img.TransformContinuousIndexToPhysicalPoint(tuple(idx)), dtype=float)


def bbox_midpoint_physical(mask: np.ndarray, img: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    lo_zyx, hi_zyx = _mask_bbox_zyx(mask)
    mid_zyx = 0.5 * (lo_zyx + hi_zyx)
    idx_xyz = (float(mid_zyx[2]), float(mid_zyx[1]), float(mid_zyx[0]))
    center = np.asarray(img.TransformContinuousIndexToPhysicalPoint(idx_xyz), dtype=float)
    span_xyz = (hi_zyx - lo_zyx)[::-1] * np.asarray(img.GetSpacing(), dtype=float)
    return center, span_xyz


def normalized_raw_index(point_phys_xyz: Sequence[float], raw_ct: sitk.Image | RawCTGeometry) -> np.ndarray:
    idx = _raw_continuous_index(point_phys_xyz, raw_ct)
    den = np.maximum(_raw_size(raw_ct) - 1.0, 1.0)
    return idx / den


def spatial_context_vector(
    crop_img: sitk.Image,
    candidate_mask: np.ndarray,
    raw_ct: sitk.Image | RawCTGeometry,
    *,
    crop_center_phys_xyz: Sequence[float] | None = None,
    candidate_center_phys_xyz: Sequence[float] | None = None,
    candidate_span_xyz_mm: Sequence[float] | None = None,
) -> np.ndarray:
    """Return the 12-D generic spatial context used by the ablation.

    Coordinates are expressed in raw-CT continuous-index fractions, which keeps
    them orientation-safe and dimensionless.  Extents are normalized by the
    current NSR-QA FOV center-to-center span.
    """
    if candidate_center_phys_xyz is None or candidate_span_xyz_mm is None:
        center, span = bbox_midpoint_physical(candidate_mask, crop_img)
        if candidate_center_phys_xyz is None:
            candidate_center_phys_xyz = center
        if candidate_span_xyz_mm is None:
            candidate_span_xyz_mm = span
    if crop_center_phys_xyz is None:
        crop_center_phys_xyz = image_center_physical(crop_img)

    crop_center_norm = normalized_raw_index(crop_center_phys_xyz, raw_ct)
    candidate_center_norm = normalized_raw_index(candidate_center_phys_xyz, raw_ct)
    fov_span = (np.asarray(crop_img.GetSize(), dtype=float) - 1.0) * np.asarray(crop_img.GetSpacing(), dtype=float)
    fov_span = np.maximum(fov_span, 1e-6)
    candidate_span_norm = np.asarray(candidate_span_xyz_mm, dtype=float) / fov_span
    raw_span = (_raw_size(raw_ct) - 1.0) * _raw_spacing(raw_ct)
    raw_span_norm = raw_span / fov_span
    out = np.concatenate((crop_center_norm, candidate_center_norm, candidate_span_norm, raw_span_norm)).astype(np.float32)
    if out.shape != (len(SPATIAL_CONTEXT_FEATURES),):
        raise AssertionError(out.shape)
    return out


def raw_ct_path(raw_data_root: str | Path, patient_id: str) -> Path:
    return Path(raw_data_root).expanduser().resolve() / "images" / f"patient{patient_id}_image.nii.gz"


def shift_tensor_zyx(x: torch.Tensor, shift_zyx: Sequence[int]) -> torch.Tensor:
    """Translate a [..., Z,Y,X] tensor with zero fill (never wrap)."""
    if x.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions, got {tuple(x.shape)}")
    shifts = tuple(int(v) for v in shift_zyx)
    spatial = tuple(int(v) for v in x.shape[-3:])
    out = torch.zeros_like(x)
    src = []
    dst = []
    for n, s in zip(spatial, shifts):
        if abs(s) >= n:
            return out
        if s >= 0:
            src.append(slice(0, n - s))
            dst.append(slice(s, n))
        else:
            src.append(slice(-s, n))
            dst.append(slice(0, n + s))
    prefix = (slice(None),) * (x.ndim - 3)
    out[prefix + tuple(dst)] = x[prefix + tuple(src)]
    return out


def _allowed_shift_zyx(target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _mask_bbox_zyx(target_mask)
    shape = np.asarray(target_mask.shape, dtype=int)
    min_shift = -lo.astype(int)
    max_shift = (shape - 1 - hi.astype(int)).astype(int)
    return min_shift, max_shift


def deployment_centered_training_pair(
    target: torch.Tensor,
    corrupted: torch.Tensor,
    crop_img: sitk.Image,
    raw_ct: sitk.Image | None,
    cfg: Mapping,
    rng,
    *,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray | None, dict]:
    """Reframe a synthetic pair to the corrupted candidate's deployment crop.

    The prepared source image is approved-centered.  Both target and corrupted
    masks are translated by the same integer-voxel amount so the corrupted bbox
    midpoint is at the tensor center.  Optional training-only FOV jitter then
    moves the crop by a small amount while guaranteeing the approved target is
    not clipped.
    """
    rf = dict(cfg.get("reference_frame_training") or {})
    if not bool(rf.get("candidate_centered", False)):
        context = None
        if bool((cfg.get("spatial_conditioning") or {}).get("enabled", False)):
            if raw_ct is None:
                raise ValueError("raw CT is required for spatial conditioning")
            arr = (corrupted[0].detach().cpu().numpy() > 0.5)
            context = spatial_context_vector(crop_img, arr, raw_ct)
        changed = (target != corrupted).float()
        return target, corrupted, changed, context, {"reference_frame_mode": "approved_centered_source"}

    target_arr = target[0].detach().cpu().numpy() > 0.5
    corrupted_arr = corrupted[0].detach().cpu().numpy() > 0.5
    lo, hi = _mask_bbox_zyx(corrupted_arr)
    candidate_mid_zyx = 0.5 * (lo + hi)
    tensor_mid_zyx = 0.5 * (np.asarray(corrupted_arr.shape, dtype=float) - 1.0)
    base_shift = np.rint(tensor_mid_zyx - candidate_mid_zyx).astype(int)

    min_shift, max_shift = _allowed_shift_zyx(target_arr)
    base_target_clips = bool(np.any(base_shift < min_shift) or np.any(base_shift > max_shift))

    jitter_shift = np.zeros(3, dtype=int)
    jitter_cfg = dict(rf.get("fov_jitter") or {})
    jitter_enabled = train and bool(jitter_cfg.get("enabled", False))
    # Candidate-centering is allowed to reproduce deployment clipping. Jitter
    # itself must not introduce additional clipping, so it is suppressed when
    # the candidate-centered crop already clips the target and otherwise only
    # accepted when the complete target remains in-FOV.
    if jitter_enabled and not base_target_clips and rng.random() < float(jitter_cfg.get("probability", 1.0)):
        fraction = float(jitter_cfg.get("fraction_of_fov", 0.025))
        max_mm = float(jitter_cfg.get("max_mm", 6.0))
        spacing_zyx = np.asarray(crop_img.GetSpacing(), dtype=float)[::-1]
        shape_zyx = np.asarray(corrupted_arr.shape, dtype=float)
        fov_mm_zyx = (shape_zyx - 1.0) * spacing_zyx
        max_jitter_mm_zyx = np.minimum(max_mm, fraction * fov_mm_zyx)
        max_jitter_vox_zyx = np.maximum(0, np.floor(max_jitter_mm_zyx / spacing_zyx + 1e-8).astype(int))
        tries = int(jitter_cfg.get("max_resample_attempts", 20))
        for _ in range(max(1, tries)):
            # Positive crop-center jitter makes anatomy appear at lower tensor indices.
            crop_jitter_vox = np.asarray([
                rng.randint(-int(m), int(m)) if m > 0 else 0 for m in max_jitter_vox_zyx
            ], dtype=int)
            proposed = base_shift - crop_jitter_vox
            if np.all(proposed >= min_shift) and np.all(proposed <= max_shift):
                jitter_shift = -crop_jitter_vox
                break

    total_shift = base_shift + jitter_shift
    target_out = shift_tensor_zyx(target, total_shift)
    corrupted_out = shift_tensor_zyx(corrupted, total_shift)
    changed_out = (target_out != corrupted_out).float()

    # Global geometry for context is based on the hypothetical output crop.
    # Output tensor center maps back to source index (center - data_shift).
    source_center_zyx = tensor_mid_zyx - total_shift.astype(float)
    source_center_xyz = tuple(float(v) for v in source_center_zyx[::-1])
    crop_center_phys = np.asarray(crop_img.TransformContinuousIndexToPhysicalPoint(source_center_xyz), dtype=float)
    candidate_center_phys, candidate_span_xyz = bbox_midpoint_physical(corrupted_arr, crop_img)

    context = None
    if bool((cfg.get("spatial_conditioning") or {}).get("enabled", False)):
        if raw_ct is None:
            raise ValueError("raw CT is required for spatial conditioning")
        context = spatial_context_vector(
            crop_img,
            corrupted_arr,
            raw_ct,
            crop_center_phys_xyz=crop_center_phys,
            candidate_center_phys_xyz=candidate_center_phys,
            candidate_span_xyz_mm=candidate_span_xyz,
        )

    spacing_zyx = np.asarray(crop_img.GetSpacing(), dtype=float)[::-1]
    meta = {
        "reference_frame_mode": "corrupted_candidate_bbox_midpoint",
        "reference_frame_shift_z_vox": int(total_shift[0]),
        "reference_frame_shift_y_vox": int(total_shift[1]),
        "reference_frame_shift_x_vox": int(total_shift[2]),
        "candidate_centering_target_clips": int(base_target_clips),
        "fov_jitter_shift_z_vox": int(jitter_shift[0]),
        "fov_jitter_shift_y_vox": int(jitter_shift[1]),
        "fov_jitter_shift_x_vox": int(jitter_shift[2]),
        "fov_jitter_shift_z_mm": float(jitter_shift[0] * spacing_zyx[0]),
        "fov_jitter_shift_y_mm": float(jitter_shift[1] * spacing_zyx[1]),
        "fov_jitter_shift_x_mm": float(jitter_shift[2] * spacing_zyx[2]),
    }
    return target_out, corrupted_out, changed_out, context, meta
