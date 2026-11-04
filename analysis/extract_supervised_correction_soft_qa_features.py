#!/usr/bin/env python3
"""Extract deployment-feasible patient-level soft correction-response QA features.

Features use only the input autosegmentation and model P(add/remove/preserve):
  * autoseg-local: dilation of the input contour
  * autoseg-boundary: inside/outside band around the input contour surface
Approved contours are never used to define features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
import yaml

from analysis.diagnose_supervised_correction_action_scores import (
    ADD, REMOVE, binary_dilate, largest_component_fraction, make_model, read_binary, read_rows,
    weighted_score_summary,
)


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    if radius <= 0:
        inner = ndimage.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
        return mask & (~inner)
    outer = ndimage.binary_dilation(mask, structure=structure, iterations=radius)
    inner = ndimage.binary_erosion(mask, structure=structure, iterations=radius, border_value=0)
    return outer & (~inner)


def add_region_features(out, probs, region, region_prefix, thresholds, topks):
    p_add = probs[ADD]
    p_remove = probs[REMOVE]
    p_action = p_add + p_remove
    out[f"{region_prefix}voxels"] = int(region.sum())
    for arr, action_name in ((p_action, "action"), (p_add, "add"), (p_remove, "remove")):
        prefix = f"{region_prefix}{action_name}"
        out.update(weighted_score_summary(arr[region], prefix, thresholds, topks))
        for thr in thresholds:
            tag = str(thr).replace(".", "p")
            high = region & (arr >= thr)
            out[f"{prefix}_largest_component_frac_ge_{tag}"] = largest_component_fraction(high)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--local-radius", type=int, default=6)
    ap.add_argument("--boundary-radius", type=int, default=3)
    ap.add_argument("--score-thresholds", default="0.02,0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--topks", default="10,50,100,500,1000")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model, checkpoint = make_model(cfg, args.checkpoint, device)
    rows = read_rows(cfg["manifest"], args.split)
    thresholds = [float(x) for x in args.score_thresholds.split(",") if x.strip()]
    topks = [int(x) for x in args.topks.split(",") if x.strip()]

    records = []
    for i, row in enumerate(rows, 1):
        x = read_binary(row["autoseg"], device)
        logits, _ = model(x)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        autoseg = (x[0, 0].cpu().numpy() > 0.5)
        local = binary_dilate(autoseg, args.local_radius)
        boundary = boundary_band(autoseg, args.boundary_radius)
        rec = {
            "patient_id": str(row["patient_id"]),
            "split": args.split,
            "autoseg_voxels": int(autoseg.sum()),
        }
        add_region_features(rec, probs, local, "local_", thresholds, topks)
        add_region_features(rec, probs, boundary, "boundary_", thresholds, topks)
        records.append(rec)
        print(f"[{i}/{len(rows)}] {args.split} patient={row['patient_id']}", flush=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    metadata = {
        "config": str(Path(args.config).resolve()), "checkpoint": str(checkpoint), "split": args.split,
        "n_cases": len(records), "local_radius_voxels": args.local_radius,
        "boundary_radius_voxels": args.boundary_radius, "score_thresholds": thresholds, "topks": topks,
        "approved_used_for_features": False,
        "regions": {
            "local": "dilation of input autoseg only",
            "boundary": "inside/outside morphological band around input autoseg surface only",
        },
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
