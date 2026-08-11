#!/usr/bin/env python3
"""Evaluate one frozen-3DINO contour decoder on a pre-generated shared probe bank."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from evaluation.probes.eval_dino_shape_denoising_probe_sweep import (
    assd_binary,
    dice_score_binary,
    hd95_binary,
    load_config,
    make_model,
    reconstruct_binary,
    volume_vox,
)


def read_binary(path: str, device: torch.device) -> tuple[torch.Tensor, sitk.Image]:
    img = sitk.ReadImage(path)
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.float32)
    return torch.from_numpy(arr[None, None]).to(device), img


def check_geometry(a: sitk.Image, b: sitk.Image, name: str) -> None:
    if a.GetSize() != b.GetSize() or not np.allclose(a.GetSpacing(), b.GetSpacing()) or not np.allclose(a.GetOrigin(), b.GetOrigin()) or not np.allclose(a.GetDirection(), b.GetDirection()):
        raise ValueError(f"Geometry mismatch for {name}")


def rel_vol(a: float, b: float) -> float:
    return abs(a - b) / b if b > 0 else np.nan


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate one model on deterministic shared probe bank")
    p.add_argument("--config", required=True)
    p.add_argument("--probe-bank", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="Optional decoder checkpoint override")
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--save-visualization", action="store_true")
    p.add_argument("--visualization-dir", default=None)
    args = p.parse_args()

    cf = load_config(args.config)
    if args.checkpoint:
        cf.setdefault("paths", {})["trained_decoder_checkpoint"] = str(Path(args.checkpoint).expanduser())
    device_name = args.device or cf.get("evaluation", {}).get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cf["device"] = str(device)
    threshold = args.threshold
    if threshold is None:
        threshold = float(cf.get("evaluation", {}).get("threshold", 0.5))

    manifest_path = Path(args.probe_bank).expanduser().resolve() / "probe_bank_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    model = make_model(cf, device)

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    vis = Path(args.visualization_dir).expanduser().resolve() if args.visualization_dir else out / "probe_visualization_shared"
    if args.save_visualization:
        vis.mkdir(parents=True, exist_ok=True)

    rows = []
    bs = max(1, int(args.inference_batch_size))
    grouped = list(manifest.groupby("patient_id", sort=False))
    for patient_number, (patient_id, group) in enumerate(grouped, start=1):
        print(f"[{patient_number}/{len(grouped)}] patient {patient_id}")
        group = group.reset_index(drop=True)
        target, target_img = read_binary(group.loc[0, "target_path"], device)

        if args.save_visualization:
            dst = vis / f"clean_input_patient_{patient_id}.nii.gz"
            if not dst.exists():
                shutil.copy2(group.loc[0, "target_path"], dst)

        for start in range(0, len(group), bs):
            chunk = group.iloc[start:start + bs]
            inputs = []
            refs = []
            for _, r in chunk.iterrows():
                x, img = read_binary(r["input_path"], device)
                check_geometry(img, target_img, r["input_path"])
                inputs.append(x)
                refs.append(img)
            x_in = torch.cat(inputs, dim=0)
            x_target = target.expand(x_in.shape[0], -1, -1, -1, -1)
            with torch.no_grad():
                recon = reconstruct_binary(model, x_in, cf, threshold)

            spacing_zyx = tuple(float(v) for v in target_img.GetSpacing()[::-1])
            dice_it = dice_score_binary(x_in, x_target).cpu().numpy()
            dice_ir = dice_score_binary(x_in, recon).cpu().numpy()
            dice_rt = dice_score_binary(recon, x_target).cpu().numpy()
            hd_it = hd95_binary(x_in, x_target, spacing_zyx).numpy()
            hd_ir = hd95_binary(x_in, recon, spacing_zyx).numpy()
            hd_rt = hd95_binary(recon, x_target, spacing_zyx).numpy()
            assd_it = assd_binary(x_in, x_target, spacing_zyx).numpy()
            assd_ir = assd_binary(x_in, recon, spacing_zyx).numpy()
            assd_rt = assd_binary(recon, x_target, spacing_zyx).numpy()
            vin = volume_vox(x_in).cpu().numpy()
            vrec = volume_vox(recon).cpu().numpy()
            vtgt = volume_vox(x_target).cpu().numpy()

            for i, (_, r) in enumerate(chunk.iterrows()):
                rows.append({
                    "patient_id": str(patient_id),
                    "case_id": int(r["case_index"]),
                    "probe_type": r["probe_type"],
                    "level": float(r["level"]),
                    "probe_seed": int(r["probe_seed"]),
                    "input_sha256": r["input_sha256"],
                    "target_sha256": r["target_sha256"],
                    "dice_input_vs_target": float(dice_it[i]),
                    "dice_input_vs_recon": float(dice_ir[i]),
                    "dice_recon_vs_target": float(dice_rt[i]),
                    "hd95_input_vs_target": float(hd_it[i]),
                    "hd95_input_vs_recon": float(hd_ir[i]),
                    "hd95_recon_vs_target": float(hd_rt[i]),
                    "assd_input_vs_target": float(assd_it[i]),
                    "assd_input_vs_recon": float(assd_ir[i]),
                    "assd_recon_vs_target": float(assd_rt[i]),
                    "input_voxels": float(vin[i]),
                    "recon_voxels": float(vrec[i]),
                    "target_voxels": float(vtgt[i]),
                    "volfrac_input": float(vin[i] / np.prod(x_in.shape[2:])),
                    "volfrac_recon": float(vrec[i] / np.prod(x_in.shape[2:])),
                    "volfrac_target": float(vtgt[i] / np.prod(x_in.shape[2:])),
                    "vol_diff_input_target": float(abs(vin[i] - vtgt[i])),
                    "vol_diff_recon_target": float(abs(vrec[i] - vtgt[i])),
                    "abs_volume_change_pct": float(100.0 * rel_vol(vin[i], vtgt[i])),
                    "relative_volume_difference_input_vs_recon": float(rel_vol(vin[i], vrec[i])),
                })

                if args.save_visualization:
                    token = str(r["level"]).rstrip("0").rstrip(".").replace(".", "p")
                    if token == "": token = "0"
                    input_dst = vis / f"corrupted_input_{r['probe_type']}_level_{token}_patient_{patient_id}.nii.gz"
                    if not input_dst.exists():
                        shutil.copy2(r["input_path"], input_dst)
                    arr = recon[i, 0].detach().cpu().numpy().astype(np.uint8)
                    out_img = sitk.GetImageFromArray(arr)
                    out_img.CopyInformation(target_img)
                    sitk.WriteImage(out_img, str(vis / f"reconstruction_{r['probe_type']}_level_{token}_patient_{patient_id}.nii.gz"), True)

    df = pd.DataFrame(rows)
    detailed = out / "probe_eval_sweep_detailed.csv"
    df.to_csv(detailed, index=False)
    summary = df.groupby(["probe_type", "level"], dropna=False).mean(numeric_only=True).reset_index()
    summary.to_csv(out / "probe_eval_sweep_summary.csv", index=False)

    meta = {
        "source_config": str(Path(args.config).resolve()),
        "checkpoint": str(cf.get("paths", {}).get("trained_decoder_checkpoint", "")),
        "probe_bank": str(Path(args.probe_bank).resolve()),
        "threshold": threshold,
        "n_rows": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
    }
    (out / "evaluation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote detailed CSV: {detailed}")


if __name__ == "__main__":
    main()
