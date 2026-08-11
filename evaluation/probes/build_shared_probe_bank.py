#!/usr/bin/env python3
"""Build a deterministic, model-independent probe bank for contour-QA evaluation.

Each corrupted mask is generated exactly once from the clean target and saved to disk.
All models then evaluate these same NIfTI inputs, eliminating RNG/anchor bias.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from evaluation.probes.eval_dino_shape_denoising_probe_sweep import (
    PREFERRED_PROBE_ORDER,
    extract_patient_id,
    load_config,
    make_corruption,
    parse_float_list,
    parse_int_list,
)


def stable_probe_seed(base_seed: int, patient_id: str, probe_type: str, level: float) -> int:
    token = f"{base_seed}|{patient_id}|{probe_type}|{float(level):.12g}".encode("utf-8")
    digest = hashlib.blake2b(token, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def level_token(level: float) -> str:
    x = float(level)
    if x.is_integer():
        return str(int(x))
    return f"{x:g}".replace("-", "m").replace(".", "p")


def write_mask(mask: torch.Tensor, reference: sitk.Image, path: Path) -> None:
    arr = (mask.detach().cpu().numpy()[0] > 0.5).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Build deterministic shared probe bank")
    p.add_argument("--config", required=True, help="Any eval config with the desired test_label_dir")
    p.add_argument("--split", default="test", choices=["train", "valid", "validation", "test"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument("--cut-levels", default="0,0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--local-frac-levels", default="0,0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--voxel-levels", default="0,1,2,3,4,5")
    p.add_argument("--probe-types", nargs="*", default=PREFERRED_PROBE_ORDER)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    cf = load_config(args.config)
    split_alias = "validation" if args.split in {"valid", "validation"} else args.split
    label_dir = Path(cf["data"][f"{split_alias}_label_dir"]).expanduser().resolve()
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)

    out = Path(args.output_dir).expanduser().resolve()
    inputs_dir = out / "inputs"
    targets_dir = out / "targets"
    out.mkdir(parents=True, exist_ok=True)

    level_map = {
        "clean": [0.0],
        "big_bite": parse_float_list(args.local_frac_levels),
        "big_blob": parse_float_list(args.local_frac_levels),
        "cut_lower_half": parse_float_list(args.cut_levels),
        "cut_upper_half": parse_float_list(args.cut_levels),
        "cut_posterior_half": parse_float_list(args.cut_levels),
        "dilated": parse_int_list(args.voxel_levels),
        "eroded": parse_int_list(args.voxel_levels),
        "shifted": parse_int_list(args.voxel_levels),
    }

    rows = []
    label_files = sorted(label_dir.rglob("*.nii.gz"))
    if not label_files:
        raise RuntimeError(f"No NIfTI labels found under {label_dir}")

    for case_index, label_path in enumerate(label_files):
        ref = sitk.ReadImage(str(label_path))
        clean_np = (sitk.GetArrayFromImage(ref) > 0).astype(np.float32)
        clean = torch.from_numpy(clean_np[None])
        patient_id = extract_patient_id(str(label_path))

        target_path = targets_dir / f"clean_input_patient_{patient_id}.nii.gz"
        if args.overwrite or not target_path.exists():
            write_mask(clean, ref, target_path)

        for probe_type in args.probe_types:
            if probe_type not in level_map:
                raise ValueError(f"Unknown probe type: {probe_type}")
            for level in level_map[probe_type]:
                seed = stable_probe_seed(args.seed, str(patient_id), probe_type, float(level))
                rng = random.Random(seed)
                corrupted, meta = make_corruption(clean, probe_type, float(level), rng)
                input_path = inputs_dir / (
                    f"corrupted_input_{probe_type}_level_{level_token(level)}_patient_{patient_id}.nii.gz"
                )
                if args.overwrite or not input_path.exists():
                    write_mask(corrupted, ref, input_path)

                rows.append({
                    "case_index": case_index,
                    "patient_id": str(patient_id),
                    "probe_type": probe_type,
                    "level": float(level),
                    "probe_seed": int(seed),
                    "target_path": str(target_path),
                    "input_path": str(input_path),
                    "target_sha256": sha256_file(target_path),
                    "input_sha256": sha256_file(input_path),
                    "target_voxels": int(clean.sum().item()),
                    "input_voxels": int(corrupted.sum().item()),
                    "requested_level": meta.get("requested_level", level),
                    "applied_level": meta.get("applied_level", level),
                    "adapted": bool(meta.get("adapted", False)),
                    "shift_zyx": meta.get("shift_zyx", ""),
                })

    manifest = pd.DataFrame(rows)
    manifest_path = out / "probe_bank_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "format_version": 1,
        "base_seed": args.seed,
        "split": args.split,
        "source_config": str(Path(args.config).resolve()),
        "source_label_dir": str(label_dir),
        "n_patients": int(manifest["patient_id"].nunique()),
        "n_probe_rows": int(len(manifest)),
        "probe_types": list(args.probe_types),
        "cut_levels": args.cut_levels,
        "local_frac_levels": args.local_frac_levels,
        "voxel_levels": args.voxel_levels,
        "seed_key": "blake2b(base_seed|patient_id|probe_type|level)",
    }
    (out / "probe_bank_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Shared probe bank: {out}")
    print(f"Manifest: {manifest_path}")
    print(f"Patients: {metadata['n_patients']}; probe rows: {metadata['n_probe_rows']}")


if __name__ == "__main__":
    main()
