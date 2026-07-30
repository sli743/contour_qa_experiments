#!/usr/bin/env python3
"""Export all denoising-AE probe contours for ProbeContourComparer without metrics.

This is the fast save-only companion to eval_dino_shape_denoising_probe_sweep.py.
It performs batched reconstruction but skips Dice/HD95/ASSD and plots.

Flat output structure (compatible with ProbeContourComparer.py):
  probe_visualization_<split>/
    clean_input_patient_<id>.nii.gz
    original_ct_patient_<id>.nii.gz                 # when *_ct_dir is configured
    corrupted_input_<probe>_level_<level>_patient_<id>.nii.gz
    reconstruction_<probe>_level_<level>_patient_<id>.nii.gz
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import torch

from evaluation.probes.eval_dino_shape_denoising_probe_sweep import (
    PREFERRED_PROBE_ORDER,
    assd_binary,
    dice_score_binary,
    extract_patient_id,
    hd95_binary,
    get_eval_loader,
    load_config,
    make_corruption,
    make_model,
    parse_float_list,
    parse_int_list,
    preprocess_clean_target,
    reconstruct_binary,
    select_case_indices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DINO denoising probe contours without surface metrics")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "validation", "test"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument("--sample_ratio", type=float, default=1.0)
    p.add_argument("--inference_batch_size", type=int, default=8)
    p.add_argument("--output_subdir", default=None)
    p.add_argument("--cut_levels", default="0,0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--local_frac_levels", default="0,0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--voxel_levels", default="0,1,2,3,4,5")
    p.add_argument("--probe_types", nargs="*", default=PREFERRED_PROBE_ORDER)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--layout", choices=["flat", "patient"], default="flat",
                   help="flat keeps ProbeContourComparer compatibility; patient creates one folder per patient")
    p.add_argument("--save_metrics", action=argparse.BooleanOptionalAction, default=False,
                   help="save per-patient metrics.csv (recommended with --layout patient)")
    return p.parse_args()


def format_level(level) -> str:
    value = float(level)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p").replace("-", "m")


def probe_variant(probe_type: str, level) -> str:
    return f"{probe_type}_level_{format_level(level)}"


def stable_probe_seed(base_seed: int, case_index: int, probe_type: str, level) -> int:
    token = f"{probe_type}|{float(level):.12g}".encode("utf-8")
    digest = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
    return int((base_seed + 100000 * case_index + digest) % (2**32 - 1))


def save_like(mask: torch.Tensor, ref: sitk.Image, path: Path) -> None:
    arr = (mask.detach().cpu().numpy()[0] > 0.5).astype(np.uint8)
    image = sitk.GetImageFromArray(arr)
    if tuple(image.GetSize()) != tuple(ref.GetSize()):
        raise ValueError(f"Geometry size mismatch for {path}: tensor={image.GetSize()} ref={ref.GetSize()}")
    image.CopyInformation(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)


def find_ct(cf: Dict, split: str, patient_id: str) -> Optional[Path]:
    split_alias = "validation" if split in {"valid", "validation"} else split
    ct_dir = cf.get("data", {}).get(f"{split_alias}_ct_dir")
    if not ct_dir:
        return None
    root = Path(ct_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    for path in sorted(root.rglob("*.nii.gz")):
        if extract_patient_id(str(path)) == str(patient_id):
            return path
    return None


def should_write(path: Path, overwrite: bool) -> bool:
    return overwrite or not path.exists()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cf = load_config(args.config)
    device_name = cf.get("evaluation", {}).get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cf["device"] = str(device)

    output_root = Path(cf.get("evaluation", {}).get(
        "output_root", Path(cf["output_root"]) / cf["experiment_name"]
    )).expanduser().resolve()
    out_dir = output_root / (args.output_subdir or f"probe_visualization_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"> Output: {out_dir}")
    model = make_model(cf, device)
    loader = get_eval_loader(cf, split=args.split)
    chosen = select_case_indices(len(loader.dataset), args.sample_ratio, args.seed)

    level_map = {
        "clean": [0.0],
        "big_bite": parse_float_list(args.local_frac_levels),
        "big_blob": parse_float_list(args.local_frac_levels),
        "cut_lower_half": parse_float_list(args.cut_levels),
        "cut_posterior_half": parse_float_list(args.cut_levels),
        "cut_upper_half": parse_float_list(args.cut_levels),
        "dilated": parse_int_list(args.voxel_levels),
        "eroded": parse_int_list(args.voxel_levels),
        "shifted": parse_int_list(args.voxel_levels),
    }

    case_index = -1
    selected_number = 0
    ct_exported = 0
    ct_missing = 0
    for batch in loader:
        target_raw = batch.get("target_image", batch["proj_image"]).to(device)
        names = batch["proj_name"]
        target = preprocess_clean_target(target_raw, cf)

        for b in range(target.shape[0]):
            case_index += 1
            if case_index not in chosen:
                continue
            selected_number += 1
            clean = target[b:b+1]
            source_path = names[b] if isinstance(names, (list, tuple)) else str(names)
            patient_id = extract_patient_id(source_path)
            ref = sitk.ReadImage(str(source_path))
            print(f"[{selected_number}/{len(chosen)}] patient {patient_id}")

            patient_dir = out_dir / f"patient_{patient_id}" if args.layout == "patient" else out_dir
            patient_dir.mkdir(parents=True, exist_ok=True)
            clean_name = "ground_truth.nii.gz" if args.layout == "patient" else f"clean_input_patient_{patient_id}.nii.gz"
            clean_path = patient_dir / clean_name
            if should_write(clean_path, args.overwrite):
                save_like(clean[0], ref, clean_path)

            ct_path = find_ct(cf, args.split, patient_id)
            if ct_path is not None:
                ct_name = "ct.nii.gz" if args.layout == "patient" else f"original_ct_patient_{patient_id}.nii.gz"
                dst = patient_dir / ct_name
                if should_write(dst, args.overwrite):
                    shutil.copy2(ct_path, dst)
                ct_exported += 1
            else:
                ct_missing += 1

            jobs = []
            for probe_type in args.probe_types:
                if probe_type not in level_map:
                    raise ValueError(f"Unknown probe type: {probe_type}")
                for level in level_map[probe_type]:
                    rng = random.Random(stable_probe_seed(args.seed, case_index, probe_type, level))
                    corrupted_1, _ = make_corruption(clean[0], probe_type, level, rng)
                    corrupted = corrupted_1.unsqueeze(0)
                    jobs.append({
                        "probe_type": probe_type,
                        "level": level,
                        "corrupted": corrupted,
                        "clean_equivalent": bool(torch.equal(corrupted, clean)),
                    })

            # Cache the clean reconstruction once for all level-zero jobs.
            clean_recon = None
            if any(j["clean_equivalent"] for j in jobs):
                clean_recon = reconstruct_binary(model, clean, cf, threshold=args.threshold)

            pending = [j for j in jobs if not j["clean_equivalent"]]
            bs = max(1, int(args.inference_batch_size))
            for offset in range(0, len(pending), bs):
                chunk = pending[offset:offset+bs]
                x = torch.cat([j["corrupted"] for j in chunk], dim=0)
                try:
                    recon = reconstruct_binary(model, x, cf, threshold=args.threshold)
                except torch.cuda.OutOfMemoryError as exc:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA OOM at batch size {bs}; retry with --inference_batch_size 4, 2, or 1"
                    ) from exc
                for i, job in enumerate(chunk):
                    job["recon"] = recon[i:i+1]

            metric_rows: List[Dict[str, object]] = []
            spacing = tuple(float(v) for v in cf.get("evaluation", {}).get("spacing_zyx", [1.0, 1.0, 1.0]))
            for job in jobs:
                if job["clean_equivalent"]:
                    job["recon"] = clean_recon
                variant = probe_variant(job["probe_type"], job["level"])
                if args.layout == "patient":
                    variant_dir = patient_dir / variant
                    corrupted_path = variant_dir / "corrupted_input.nii.gz"
                    recon_path = variant_dir / "reconstruction.nii.gz"
                else:
                    corrupted_path = out_dir / f"corrupted_input_{variant}_patient_{patient_id}.nii.gz"
                    recon_path = out_dir / f"reconstruction_{variant}_patient_{patient_id}.nii.gz"
                if should_write(corrupted_path, args.overwrite):
                    save_like(job["corrupted"][0], ref, corrupted_path)
                if should_write(recon_path, args.overwrite):
                    save_like(job["recon"][0], ref, recon_path)

                if args.save_metrics:
                    corrupted = job["corrupted"]
                    recon = job["recon"]
                    dice_in = float(dice_score_binary(corrupted, clean)[0].item())
                    dice_recon = float(dice_score_binary(recon, clean)[0].item())
                    hd95_in = float(hd95_binary(corrupted, clean, spacing)[0].item())
                    hd95_recon = float(hd95_binary(recon, clean, spacing)[0].item())
                    assd_in = float(assd_binary(corrupted, clean, spacing)[0].item())
                    assd_recon = float(assd_binary(recon, clean, spacing)[0].item())
                    metric_rows.append({
                        "patient_id": patient_id,
                        "probe_type": job["probe_type"],
                        "level": float(job["level"]),
                        "dice_input_target": dice_in,
                        "dice_recon_target": dice_recon,
                        "dice_correction_gain": dice_recon - dice_in,
                        "hd95_input_target": hd95_in,
                        "hd95_recon_target": hd95_recon,
                        "hd95_improvement": hd95_in - hd95_recon,
                        "assd_input_target": assd_in,
                        "assd_recon_target": assd_recon,
                        "assd_improvement": assd_in - assd_recon,
                    })

            if args.save_metrics and metric_rows:
                metrics_path = patient_dir / "metrics.csv" if args.layout == "patient" else out_dir / f"metrics_patient_{patient_id}.csv"
                with metrics_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(metric_rows)

    print(f"Done. Use this directory in Probe Contour Comparer: {out_dir}")
    if ct_missing == 0:
        print(f"Original CT exported for all {ct_exported} selected patients.")
    else:
        split_alias = "validation" if args.split in {"valid", "validation"} else args.split
        expected_key = f"{split_alias}_ct_dir"
        print(
            f"Warning: original CT exported for {ct_exported}/{selected_number} selected patients; "
            f"{ct_missing} CT file(s) were not found. Check data.{expected_key} and CT filenames."
        )


if __name__ == "__main__":
    main()
