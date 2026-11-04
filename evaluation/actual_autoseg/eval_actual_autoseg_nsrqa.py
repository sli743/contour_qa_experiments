#!/usr/bin/env python3
"""Run NSR-QA on real autosegmentations and compute clinical-error/QA metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List

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
from analysis.spinalcord_diagnostic_utils import (
    correction_decomposition,
    pair_metrics,
    read_binary_lps,
)
from training.spatial_reference import (
    SPATIAL_CONTEXT_FEATURES,
    raw_ct_path,
    spatial_context_vector,
)


def read_binary(path: Path, device: torch.device):
    img = sitk.ReadImage(str(path))
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.float32)
    return torch.from_numpy(arr[None, None]).to(device), img


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def rel_vol(a: float, b: float) -> float:
    return abs(a - b) / b if b > 0 else np.nan


def write_mask(arr: np.ndarray, ref: sitk.Image, path: Path) -> None:
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(path), True)


def finite_or_none(v: float):
    return float(v) if np.isfinite(v) else None



def spatial_context_for_case(cf: Dict, patient_id: str, input_img: sitk.Image, input_arr: np.ndarray, device: torch.device):
    scfg = dict(cf.get("spatial_conditioning") or {})
    if not bool(scfg.get("enabled", False)):
        return None, {}
    root = scfg.get("raw_data_root")
    if not root:
        raise KeyError("spatial_conditioning.raw_data_root is required for conditioned evaluation")
    path = raw_ct_path(root, patient_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_ct = sitk.ReadImage(str(path))
    vec = spatial_context_vector(input_img, input_arr, raw_ct)
    if len(vec) != int(scfg.get("context_dim", len(SPATIAL_CONTEXT_FEATURES))):
        raise ValueError(f"Spatial context length {len(vec)} does not match configured context_dim")
    tensor = torch.from_numpy(vec[None]).to(device=device, dtype=torch.float32)
    values = {f"spatial_{name}": float(value) for name, value in zip(SPATIAL_CONTEXT_FEATURES, vec)}
    return tensor, values

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--input-root", required=True, help="Root produced by preprocess_actual_autoseg.py")
    p.add_argument("--roi", required=True)
    p.add_argument("--mode", choices=["autoseg_centered", "autoseg_xy_ct_z", "approved_reference"], required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument(
        "--case-output-root",
        default=None,
        help=(
            "Optional separate viewer-output root. Inputs are read from --input-root, "
            "while CT/approved/autoseg plus reconstruction/diagnostics/metrics.json are "
            "written under <case-output-root>/<mode>/<roi>/<split>/patient_<id>. "
            "If omitted, preserves the previous in-place behavior."
        ),
    )
    p.add_argument("--model-name", default="NSR-QA")
    p.add_argument("--corruption-label", default="")
    args = p.parse_args()

    cf = load_config(args.config)
    if args.checkpoint:
        cf.setdefault("paths", {})["trained_decoder_checkpoint"] = str(Path(args.checkpoint).expanduser())
    device_name = args.device or cf.get("evaluation", {}).get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cf["device"] = str(device)
    threshold = float(args.threshold if args.threshold is not None else cf.get("evaluation", {}).get("threshold", 0.5))
    model = make_model(cf, device)

    input_base = Path(args.input_root).expanduser().resolve()
    root = input_base / args.mode / args.roi
    case_output_base = Path(args.case_output_root).expanduser().resolve() if args.case_output_root else None
    config_path = Path(args.config).expanduser().resolve()
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else Path(str(cf.get("paths", {}).get("trained_decoder_checkpoint", ""))).expanduser()
    rows: List[Dict] = []
    for split in ("training1", "validation1", "testing1"):
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"Missing split directory; skipping: {split_dir}")
            continue
        case_dirs = sorted(p for p in split_dir.glob("patient_*") if p.is_dir())
        for idx, input_case_dir in enumerate(case_dirs, start=1):
            pid = input_case_dir.name.replace("patient_", "", 1)
            input_path = input_case_dir / "autoseg_input.nii.gz"
            target_path = input_case_dir / "approved.nii.gz"
            ct_path = input_case_dir / "ct.nii.gz"
            if not (input_path.is_file() and target_path.is_file() and ct_path.is_file()):
                print(f"[{args.roi}] {split} patient {pid}: incomplete preprocessed case; skip")
                continue

            if case_output_base is None:
                case_dir = input_case_dir
            else:
                case_dir = case_output_base / args.mode / args.roi / split / f"patient_{pid}"
                case_dir.mkdir(parents=True, exist_ok=True)
                # Keep each model result self-contained and directly consumable by
                # ActualAutosegNSRQAViewer, including when copied off the cluster.
                for filename in ("ct.nii.gz", "approved.nii.gz", "autoseg_input.nii.gz"):
                    shutil.copy2(input_case_dir / filename, case_dir / filename)

            x_in, input_img = read_binary(input_path, device)
            x_target, target_img = read_binary(target_path, device)
            ct_img = sitk.ReadImage(str(ct_path))
            if not same_geometry(input_img, target_img) or not same_geometry(input_img, ct_img):
                raise RuntimeError(f"Geometry mismatch: {case_dir}")

            input_arr = sitk.GetArrayFromImage(input_img) > 0
            spatial_context, spatial_values = spatial_context_for_case(
                cf, pid, input_img, input_arr, device
            )
            with torch.no_grad():
                recon = reconstruct_binary(
                    model, x_in, cf, threshold, spatial_context=spatial_context
                )

            spacing_zyx = tuple(float(v) for v in target_img.GetSpacing()[::-1])
            dice_true = float(dice_score_binary(x_in, x_target).item())
            dice_qa = float(dice_score_binary(x_in, recon).item())
            hd_true = float(hd95_binary(x_in, x_target, spacing_zyx).item())
            hd_qa = float(hd95_binary(x_in, recon, spacing_zyx).item())
            assd_true = float(assd_binary(x_in, x_target, spacing_zyx).item())
            assd_qa = float(assd_binary(x_in, recon, spacing_zyx).item())
            vin = float(volume_vox(x_in).item())
            vtgt = float(volume_vox(x_target).item())
            vrec = float(volume_vox(recon).item())

            recon_arr = recon[0, 0].detach().cpu().numpy() > 0.5
            target_arr = sitk.GetArrayFromImage(target_img) > 0
            write_mask(recon_arr, target_img, case_dir / "nsrqa_reconstruction.nii.gz")
            write_mask(target_arr & ~input_arr, target_img, case_dir / "missing_autoseg_vs_approved.nii.gz")
            write_mask(input_arr & ~target_arr, target_img, case_dir / "excess_autoseg_vs_approved.nii.gz")
            write_mask(recon_arr & ~input_arr, target_img, case_dir / "missing_input_vs_reconstruction.nii.gz")
            write_mask(input_arr & ~recon_arr, target_img, case_dir / "excess_input_vs_reconstruction.nii.gz")

            approved_lps, _app_img_lps, spacing_lps_zyx = read_binary_lps(case_dir / "approved.nii.gz")
            autoseg_lps, _auto_img_lps, _ = read_binary_lps(case_dir / "autoseg_input.nii.gz")
            recon_lps, _rec_img_lps, _ = read_binary_lps(case_dir / "nsrqa_reconstruction.nii.gz")
            true_full = pair_metrics(approved_lps, autoseg_lps, spacing_lps_zyx)
            qa_full = pair_metrics(autoseg_lps, recon_lps, spacing_lps_zyx)
            residual_full = pair_metrics(approved_lps, recon_lps, spacing_lps_zyx)
            correction = correction_decomposition(approved_lps, autoseg_lps, recon_lps)

            row = {
                "patient_id": str(pid),
                "roi": args.roi,
                "localization_mode": args.mode,
                "shape_model_split": split,
                "regression_split": "development" if split in {"training1", "validation1"} else "test",
                "case_dir": str(case_dir),
                "source_case_dir": str(input_case_dir),
                "model_name": str(args.model_name),
                "corruption_label": str(args.corruption_label),
                "source_config": str(config_path),
                "source_config_sha256": config_sha256,
                "checkpoint": str(checkpoint_path),
                "true_1_minus_dice": 1.0 - dice_true,
                "true_hd95_mm": hd_true,
                "true_assd_mm": assd_true,
                "true_relative_volume_difference": rel_vol(vin, vtgt),
                "qa_1_minus_dice": 1.0 - dice_qa,
                "qa_hd95_mm": hd_qa,
                "qa_assd_mm": assd_qa,
                "qa_relative_volume_difference": rel_vol(vin, vrec),
                "autoseg_voxels": vin,
                "approved_voxels": vtgt,
                "reconstruction_voxels": vrec,
                "true_hd99_mm": true_full["hd99_mm"],
                "true_max_hd_mm": true_full["max_hd_mm"],
                "true_max_abs_endpoint_error_mm": true_full["max_abs_endpoint_error_mm"],
                "true_inferior_signed_error_mm": true_full["inferior_signed_error_mm"],
                "true_superior_signed_error_mm": true_full["superior_signed_error_mm"],
                "qa_hd99_mm": qa_full["hd99_mm"],
                "qa_max_hd_mm": qa_full["max_hd_mm"],
                "qa_max_abs_endpoint_error_mm": qa_full["max_abs_endpoint_error_mm"],
                "residual_1_minus_dice": residual_full["one_minus_dice"],
                "residual_assd_mm": residual_full["assd_mm"],
                "residual_hd95_mm": residual_full["hd95_mm"],
                "residual_hd99_mm": residual_full["hd99_mm"],
                "residual_max_abs_endpoint_error_mm": residual_full["max_abs_endpoint_error_mm"],
                **correction,
                **spatial_values,
                "threshold": threshold,
            }
            row["all_primary_metrics_finite"] = bool(all(np.isfinite(row[k]) for k in (
                "true_1_minus_dice", "true_hd95_mm", "true_assd_mm", "true_relative_volume_difference",
                "qa_1_minus_dice", "qa_hd95_mm", "qa_assd_mm", "qa_relative_volume_difference",
            )))
            rows.append(row)

            patient_meta = dict(row)
            for k, v in list(patient_meta.items()):
                if isinstance(v, float):
                    patient_meta[k] = finite_or_none(v)
            (case_dir / "metrics.json").write_text(json.dumps(patient_meta, indent=2), encoding="utf-8")
            print(f"[{args.roi}] {args.mode} {split} [{idx}/{len(case_dirs)}] patient {pid}")

    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    meta = {
        "source_config": str(config_path),
        "source_config_sha256": config_sha256,
        "checkpoint": str(checkpoint_path),
        "model_name": str(args.model_name),
        "corruption_label": str(args.corruption_label),
        "input_root": str(input_base),
        "case_output_root": str(case_output_base) if case_output_base is not None else str(input_base),
        "roi": args.roi,
        "localization_mode": args.mode,
        "threshold": threshold,
        "n_rows": int(len(df)),
        "n_finite_rows": int(df["all_primary_metrics_finite"].sum()) if len(df) else 0,
        "metric_definitions": {
            "true": "autosegmentation versus approved contour",
            "qa": "autosegmentation versus NSR-QA reconstruction",
            "relative_volume_true": "abs(V_autoseg - V_approved) / V_approved",
            "relative_volume_qa": "abs(V_autoseg - V_reconstruction) / V_reconstruction",
        },
    }
    out_csv.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
