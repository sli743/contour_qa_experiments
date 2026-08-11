#!/usr/bin/env python3
"""Collect stochastic-region-dropout previews with CT and optional ROI cropping.

Input preview filenames:
  clean_target_patient_<id>.nii.gz
  corrupted_input_branch_<branch>_sample_<n>_patient_<id>.nii.gz
  changed_region_branch_<branch>_sample_<n>_patient_<id>.nii.gz
  missing_region_branch_<branch>_sample_<n>_patient_<id>.nii.gz

Output is a flat directory that the companion Slicer script can scan.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import yaml

PATIENT_RE = re.compile(r"patient_(.+?)\.nii\.gz$")
CLEAN_RE = re.compile(r"^clean_target_patient_(.+)\.nii\.gz$")


def patient_id(path: Path) -> str:
    m = PATIENT_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract patient ID from {path.name}")
    return m.group(1)


def extract_id_from_data_name(path: Path) -> str:
    nums = re.findall(r"\d+", path.name)
    return nums[-1] if nums else path.stem.replace(".nii", "")


def all_nii(root: Path) -> list[Path]:
    return sorted(root.rglob("*.nii.gz")) if root.is_dir() else []


def resolve_ct_dir(args: argparse.Namespace) -> Path:
    if args.ct_dir:
        root = Path(args.ct_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CT directory not found: {root}")
        return root

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        key = {"train": "train_ct_dir", "validation": "validation_ct_dir", "test": "test_ct_dir"}[args.split]
        value = cfg.get("data", {}).get(key)
        if value:
            root = Path(value).expanduser().resolve()
            if root.is_dir():
                return root

        label_key = {"train": "train_label_dir", "validation": "validation_label_dir", "test": "test_label_dir"}[args.split]
        label_dir = cfg.get("data", {}).get(label_key)
        if label_dir:
            label_path = Path(label_dir).expanduser().resolve()
            candidates = [
                Path(str(label_path).replace("/labels", "/images")),
                label_path.parent / "images",
                label_path.parent.parent / "images",
            ]
            for candidate in candidates:
                if candidate.is_dir():
                    return candidate

    raise RuntimeError("Could not locate CT directory. Pass --ct-dir or provide a config with a *_ct_dir entry.")


def build_ct_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in all_nii(root):
        index.setdefault(extract_id_from_data_name(path), path)
    return index


def geometry_matches(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def resample_to_reference(image: sitk.Image, ref: sitk.Image, nearest: bool) -> sitk.Image:
    return sitk.Resample(
        image,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear,
        0,
        image.GetPixelID(),
    )


def bbox_from_masks(images: Iterable[sitk.Image], margin_mm: float) -> tuple[list[int], list[int]]:
    union = None
    ref = None
    for image in images:
        ref = image if ref is None else ref
        arr = sitk.GetArrayViewFromImage(image) > 0
        union = arr.copy() if union is None else (union | arr)
    if ref is None or union is None or not union.any():
        raise ValueError("No foreground found for crop")

    zyx = np.argwhere(union)
    lo_zyx = zyx.min(axis=0)
    hi_zyx = zyx.max(axis=0) + 1
    spacing_xyz = np.asarray(ref.GetSpacing(), dtype=float)
    margin_xyz = np.ceil(float(margin_mm) / spacing_xyz).astype(int)
    lo_xyz = lo_zyx[::-1] - margin_xyz
    hi_xyz = hi_zyx[::-1] + margin_xyz
    size_xyz = np.asarray(ref.GetSize(), dtype=int)
    lo_xyz = np.maximum(lo_xyz, 0)
    hi_xyz = np.minimum(hi_xyz, size_xyz)
    return lo_xyz.astype(int).tolist(), (hi_xyz - lo_xyz).astype(int).tolist()


def crop_image(image: sitk.Image, index_xyz: list[int], size_xyz: list[int]) -> sitk.Image:
    return sitk.RegionOfInterest(image, size=size_xyz, index=index_xyz)


def output_name(src: Path) -> str:
    name = src.name
    if name.startswith("clean_target_"):
        return name.replace("clean_target_", "clean_input_", 1)
    return name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preview-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config")
    p.add_argument("--ct-dir")
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--margin-mm", type=float, default=20.0)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    src = Path(args.preview_dir).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ct_index = build_ct_index(resolve_ct_dir(args))

    clean_files = sorted(p for p in src.glob("clean_target_patient_*.nii.gz") if CLEAN_RE.match(p.name))
    if not clean_files:
        raise FileNotFoundError(f"No clean preview masks found in {src}")

    manifest_rows = []
    for clean_path in clean_files:
        pid = patient_id(clean_path)
        related = sorted(src.glob(f"*patient_{pid}.nii.gz"))
        clean = sitk.ReadImage(str(clean_path))
        masks: list[tuple[Path, sitk.Image]] = []
        for path in related:
            image = sitk.ReadImage(str(path))
            if not geometry_matches(image, clean):
                image = resample_to_reference(image, clean, nearest=True)
            masks.append((path, image))

        ct_path = ct_index.get(str(pid))
        if ct_path is None:
            raise FileNotFoundError(f"No CT found for patient {pid}")
        ct = sitk.ReadImage(str(ct_path))
        if not geometry_matches(ct, clean):
            ct = resample_to_reference(ct, clean, nearest=False)

        if args.crop:
            crop_sources = [image for path, image in masks if path.name.startswith(("clean_target_", "corrupted_input_"))]
            index_xyz, size_xyz = bbox_from_masks(crop_sources, args.margin_mm)
            ct = crop_image(ct, index_xyz, size_xyz)
            masks = [(path, crop_image(image, index_xyz, size_xyz)) for path, image in masks]
        else:
            index_xyz = [0, 0, 0]
            size_xyz = list(clean.GetSize())

        ct_out = out / f"original_ct_patient_{pid}.nii.gz"
        if args.overwrite or not ct_out.exists():
            sitk.WriteImage(ct, str(ct_out), True)

        for path, image in masks:
            dst = out / output_name(path)
            if args.overwrite or not dst.exists():
                sitk.WriteImage(image, str(dst), True)

        manifest_rows.append((pid, str(ct_path), index_xyz, size_xyz, len(related)))
        print(f"Collected patient {pid}: {len(related)} masks, crop size={size_xyz}")

    import csv
    with open(out / "collection_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "source_ct", "crop_index_xyz", "crop_size_xyz", "n_preview_files"])
        for pid, ct_path, index_xyz, size_xyz, n in manifest_rows:
            w.writerow([pid, ct_path, " ".join(map(str, index_xyz)), " ".join(map(str, size_xyz)), n])

    metadata = src / "metadata.csv"
    if metadata.exists():
        shutil.copy2(metadata, out / "metadata.csv")
    print(f"Collected preview directory: {out}")


if __name__ == "__main__":
    main()
