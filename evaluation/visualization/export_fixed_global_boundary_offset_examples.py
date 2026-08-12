#!/usr/bin/env python3
"""Export deterministic inward/outward SDF-offset examples for Slicer review.

This is a pre-training sanity check for thin ROIs such as Bronchus.  It calls
exactly the same global SDF offset implementation used by training, but fixes
both direction and magnitude so every requested physical offset can be
inspected consistently across cases.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from training.corruptions.stochastic_region_dropout import signed_distance


def patient_id(path: Path) -> str:
    m = re.search(r"patient_(.+)\.nii\.gz$", path.name)
    return m.group(1) if m else path.stem


def write_like(arr: np.ndarray, ref: sitk.Image, path: Path) -> None:
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    sitk.WriteImage(out, str(path))


def sibling_ct_path(label_path: Path, pid: str) -> Path:
    ct = label_path.parent.parent / "ct" / f"ct_patient_{pid}.nii.gz"
    if not ct.is_file():
        raise FileNotFoundError(f"Matching CT not found: {ct}")
    return ct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    ap.add_argument("--max-cases", type=int, default=8)
    ap.add_argument("--offset-mm", type=float, action="append", default=None,
                    help="Repeat for each magnitude; defaults to 1.5, 2.0, 2.5, 3.0 mm.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    key = {"train": "train_label_dir", "validation": "validation_label_dir", "test": "test_label_dir"}[args.split]
    label_dir = Path(cfg["data"][key]).expanduser().resolve()
    files = sorted(label_dir.rglob("*.nii.gz"))[: args.max_cases]
    if not files:
        raise FileNotFoundError(f"No label files under {label_dir}")

    spacing = tuple(float(x) for x in cfg["stochastic_region_dropout"].get("spacing_zyx", [1.5, 1.5, 1.5]))
    magnitudes = args.offset_mm or [1.5, 2.0, 2.5, 3.0]
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(cfg["output_root"]) / cfg["experiment_name"] / "global_boundary_preview"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_path in files:
        ref = sitk.ReadImage(str(label_path))
        clean = sitk.GetArrayFromImage(ref) > 0
        pid = patient_id(label_path)
        if not clean.any():
            continue

        write_like(clean.astype(np.uint8), ref, out_dir / f"clean_target_patient_{pid}.nii.gz")
        ct_path = sibling_ct_path(label_path, pid)
        ct_img = sitk.ReadImage(str(ct_path))
        if (ct_img.GetSize() != ref.GetSize() or
                not np.allclose(ct_img.GetSpacing(), ref.GetSpacing()) or
                not np.allclose(ct_img.GetOrigin(), ref.GetOrigin()) or
                not np.allclose(ct_img.GetDirection(), ref.GetDirection())):
            raise RuntimeError(f"CT/label geometry mismatch for patient {pid}: {ct_path}")
        sitk.WriteImage(ct_img, str(out_dir / f"original_ct_patient_{pid}.nii.gz"))

        sdf = signed_distance(clean, spacing)
        target_vox = int(clean.sum())
        for magnitude in magnitudes:
            for direction, sign in (("inward", -1.0), ("outward", 1.0)):
                result = (sdf + sign * float(magnitude)) > 0.0
                changed = np.logical_xor(result, clean)
                missing = clean & ~result
                excess = ~clean & result
                result_vox = int(result.sum())
                rejected = int(not result.any())

                token = f"{float(magnitude):g}".replace(".", "p")
                stem = f"global_boundary_{direction}_{token}mm_patient_{pid}"
                write_like(result.astype(np.uint8), ref, out_dir / f"corrupted_input_{stem}.nii.gz")
                write_like(changed.astype(np.uint8), ref, out_dir / f"changed_region_{stem}.nii.gz")
                write_like(missing.astype(np.uint8), ref, out_dir / f"missing_region_{stem}.nii.gz")
                write_like(excess.astype(np.uint8), ref, out_dir / f"excess_region_{stem}.nii.gz")

                rows.append({
                    "patient_id": pid,
                    "direction": direction,
                    "offset_magnitude_mm": float(magnitude),
                    "signed_offset_mm": sign * float(magnitude),
                    "target_voxels": target_vox,
                    "corrupted_voxels": result_vox,
                    "changed_voxels": int(changed.sum()),
                    "missing_voxels": int(missing.sum()),
                    "excess_voxels": int(excess.sum()),
                    "volume_change_pct": 100.0 * abs(result_vox - target_vox) / max(target_vox, 1),
                    "retained_volume_fraction": result_vox / max(target_vox, 1),
                    "empty_result": rejected,
                })

    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved fixed global-boundary preview to: {out_dir}")
    print(f"Cases: {len(set(r['patient_id'] for r in rows))}; rows: {len(rows)}")


if __name__ == "__main__":
    main()
