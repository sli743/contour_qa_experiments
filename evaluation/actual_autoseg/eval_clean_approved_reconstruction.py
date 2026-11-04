#!/usr/bin/env python3
"""Evaluate clean approved-contour identity reconstruction for NSR-QA models.

The approved contour itself is fed to the network. Two preprocessed localization
geometries can be evaluated:
  * approved_reference: ideal/approved-centered identity test
  * autoseg_centered: same perfect contour placed on the deployment crop, which
    isolates localization/crop sensitivity without introducing contour error.
  * autoseg_xy_ct_z: same perfect contour on a hybrid crop whose in-plane center
    is candidate-derived while the longitudinal/slice-axis center is anchored to
    the raw CT acquisition midpoint.

Outputs are directly consumable by ActualAutosegNSRQAViewer: autoseg_input is
intentionally a copy of approved, so the viewer's QA disagreement is exactly the
clean reconstruction error.
"""
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

from analysis.spinalcord_diagnostic_utils import pair_metrics, read_binary_lps
from evaluation.actual_autoseg.eval_actual_autoseg_nsrqa import (
    finite_or_none,
    read_binary,
    same_geometry,
    spatial_context_for_case,
    write_mask,
)
from evaluation.probes.eval_dino_shape_denoising_probe_sweep import (
    load_config,
    make_model,
    reconstruct_binary,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-root", required=True)
    p.add_argument("--roi", default="SpinalCord")
    p.add_argument("--mode", choices=["autoseg_centered", "autoseg_xy_ct_z", "approved_reference"], required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--case-output-root", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--model-name", required=True)
    p.add_argument("--corruption-label", default="")
    args = p.parse_args()

    cf = load_config(args.config)
    cf.setdefault("paths", {})["trained_decoder_checkpoint"] = str(Path(args.checkpoint).expanduser().resolve())
    device_name = args.device or cf.get("evaluation", {}).get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cf["device"] = str(device)
    threshold = float(args.threshold if args.threshold is not None else cf.get("evaluation", {}).get("threshold", 0.5))
    model = make_model(cf, device)

    input_base = Path(args.input_root).expanduser().resolve()
    source_root = input_base / args.mode / args.roi
    output_base = Path(args.case_output_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    cfg_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    rows: List[Dict] = []

    for split in ("training1", "validation1", "testing1"):
        split_dir = source_root / split
        if not split_dir.is_dir():
            continue
        cases = sorted(p for p in split_dir.glob("patient_*") if p.is_dir())
        for i, source_case in enumerate(cases, 1):
            pid = source_case.name.replace("patient_", "", 1)
            approved_path = source_case / "approved.nii.gz"
            ct_path = source_case / "ct.nii.gz"
            if not approved_path.is_file() or not ct_path.is_file():
                continue

            x_clean, approved_img = read_binary(approved_path, device)
            ct_img = sitk.ReadImage(str(ct_path))
            if not same_geometry(approved_img, ct_img):
                raise RuntimeError(f"CT/approved geometry mismatch: {source_case}")
            approved_arr = sitk.GetArrayFromImage(approved_img) > 0
            spatial_context, spatial_values = spatial_context_for_case(
                cf, pid, approved_img, approved_arr, device
            )
            with torch.no_grad():
                recon = reconstruct_binary(
                    model, x_clean, cf, threshold, spatial_context=spatial_context
                )

            out_case = output_base / args.mode / args.roi / split / f"patient_{pid}"
            out_case.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ct_path, out_case / "ct.nii.gz")
            shutil.copy2(approved_path, out_case / "approved.nii.gz")
            shutil.copy2(approved_path, out_case / "autoseg_input.nii.gz")

            recon_arr = recon[0, 0].detach().cpu().numpy() > 0.5
            write_mask(recon_arr, approved_img, out_case / "nsrqa_reconstruction.nii.gz")
            # Viewer-compatible diagnostic masks. Since input==approved, true-error
            # masks are empty by construction.
            empty = np.zeros_like(approved_arr, dtype=bool)
            write_mask(empty, approved_img, out_case / "missing_autoseg_vs_approved.nii.gz")
            write_mask(empty, approved_img, out_case / "excess_autoseg_vs_approved.nii.gz")
            write_mask(recon_arr & ~approved_arr, approved_img, out_case / "missing_input_vs_reconstruction.nii.gz")
            write_mask(approved_arr & ~recon_arr, approved_img, out_case / "excess_input_vs_reconstruction.nii.gz")

            # Re-read in LPS so endpoint signs are explicit and consistent.
            app_lps, _img_lps, spacing_zyx = read_binary_lps(out_case / "approved.nii.gz")
            rec_lps, _rec_img, _ = read_binary_lps(out_case / "nsrqa_reconstruction.nii.gz")
            m = pair_metrics(app_lps, rec_lps, spacing_zyx)
            row = {
                "patient_id": str(pid),
                "roi": args.roi,
                "localization_mode": args.mode,
                "shape_model_split": split,
                "regression_split": "development" if split in {"training1", "validation1"} else "test",
                "experiment_type": "clean_approved_reconstruction",
                "model_name": args.model_name,
                "corruption_label": args.corruption_label,
                "case_dir": str(out_case),
                "source_case_dir": str(source_case),
                "source_config": str(config_path),
                "source_config_sha256": cfg_hash,
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "threshold": threshold,
                "clean_1_minus_dice": m["one_minus_dice"],
                "clean_assd_mm": m["assd_mm"],
                "clean_hd95_mm": m["hd95_mm"],
                "clean_hd99_mm": m["hd99_mm"],
                "clean_max_hd_mm": m["max_hd_mm"],
                "clean_relative_volume_difference": m["relative_volume_difference"],
                "clean_inferior_missing_mm": m["inferior_missing_mm"],
                "clean_inferior_excess_mm": m["inferior_excess_mm"],
                "clean_superior_missing_mm": m["superior_missing_mm"],
                "clean_superior_excess_mm": m["superior_excess_mm"],
                "clean_max_abs_endpoint_error_mm": m["max_abs_endpoint_error_mm"],
                "clean_si_span_difference_signed_mm": m["si_span_difference_signed_mm"],
                # Existing viewer keys.
                "true_1_minus_dice": 0.0,
                "true_hd95_mm": 0.0,
                "true_assd_mm": 0.0,
                "true_relative_volume_difference": 0.0,
                "qa_1_minus_dice": m["one_minus_dice"],
                "qa_hd95_mm": m["hd95_mm"],
                "qa_assd_mm": m["assd_mm"],
                "qa_relative_volume_difference": m["relative_volume_difference"],
                **spatial_values,
            }
            rows.append(row)
            meta = {k: (finite_or_none(v) if isinstance(v, float) else v) for k, v in row.items()}
            (out_case / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[{args.model_name}] clean {args.mode} {split} [{i}/{len(cases)}] patient {pid}")

    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    (out_csv.with_suffix(".metadata.json")).write_text(json.dumps({
        "experiment_type": "clean_approved_reconstruction",
        "model_name": args.model_name,
        "localization_mode": args.mode,
        "source_config": str(config_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "metric_definition_note": "HD95/HD99/max-HD are recomputed explicitly from binary masks in LPS physical geometry.",
        "n_rows": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
