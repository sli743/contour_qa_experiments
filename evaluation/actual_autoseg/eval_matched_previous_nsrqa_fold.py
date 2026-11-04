#!/usr/bin/env python3
"""Evaluate a fold-matched previous NSR-QA model on real autosegmentations.

No reconstructed NIfTI volumes are written: this comparison only needs the
patient-level input-vs-reconstruction QA metrics and input-vs-approved targets.
"""
from __future__ import annotations

import argparse
import csv
import json
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
)


def read_binary(path: str | Path, device: torch.device):
    img = sitk.ReadImage(str(path))
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.float32)
    return torch.from_numpy(arr[None, None]).to(device), img


def geometry_equal(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device_name = args.device or cfg.get("evaluation", {}).get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cfg["device"] = str(device)
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    model = make_model(cfg, device)

    with Path(args.manifest).open(newline="") as f:
        rows = list(csv.DictReader(f))

    out = []
    for i, row in enumerate(rows, 1):
        x, x_img = read_binary(row["autoseg"], device)
        target, t_img = read_binary(row["approved"], device)
        if not geometry_equal(x_img, t_img):
            raise RuntimeError(f"Input/approved geometry mismatch for patient {row['patient_id']}")
        with torch.no_grad():
            recon = reconstruct_binary(model, x, cfg, threshold)
        spacing_zyx = tuple(float(v) for v in t_img.GetSpacing()[::-1])

        true_dice = float(dice_score_binary(x, target).item())
        qa_dice = float(dice_score_binary(x, recon).item())
        true_hd95 = float(hd95_binary(x, target, spacing_zyx).item())
        qa_hd95 = float(hd95_binary(x, recon, spacing_zyx).item())
        true_assd = float(assd_binary(x, target, spacing_zyx).item())
        qa_assd = float(assd_binary(x, recon, spacing_zyx).item())

        out.append({
            "patient_id": str(row["patient_id"]),
            "roi": cfg.get("data", {}).get("roi_name", ""),
            "outer_fold": int(row.get("outer_fold", -1)),
            "split": row["split"],
            "true_1_minus_dice": 1.0 - true_dice,
            "true_hd95_mm": true_hd95,
            "true_assd_mm": true_assd,
            "qa_1_minus_dice": 1.0 - qa_dice,
            "qa_hd95_mm": qa_hd95,
            "qa_assd_mm": qa_assd,
            "threshold": threshold,
        })
        print(
            f"[{i}/{len(rows)}] patient={row['patient_id']} split={row['split']} "
            f"trueASSD={true_assd:.4f} nsrqaASSD={qa_assd:.4f}"
        )

    df = pd.DataFrame(out)
    output = Path(args.output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "config": str(Path(args.config).expanduser().resolve()),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "n": int(len(df)),
        "qa_definition": "real autosegmentation versus previous NSR-QA reconstruction",
        "true_error_definition": "real autosegmentation versus approved contour",
        "reconstruction_volumes_saved": False,
    }, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
