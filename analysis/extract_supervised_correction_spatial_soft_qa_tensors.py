#!/usr/bin/env python3
"""Extract fixed-size spatial soft-action tensors for patient-level QA.

The frozen correction model produces P(add) and P(remove) volumes.  This script
creates deployment-feasible spatial tensors using only those probabilities and
regions derived from the input autosegmentation.  Approved contours are never
used to construct predictor tensors.

For each of two regions (autoseg-local and autoseg-boundary), and each score
(P(action), P(add), P(remove)), we store adaptive-average-pooled and
adaptive-max-pooled maps.  With the defaults this gives 12 channels at 32^3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from analysis.diagnose_supervised_correction_action_scores import (
    ADD,
    REMOVE,
    binary_dilate,
    make_model,
    read_binary,
    read_rows,
)
from analysis.extract_supervised_correction_soft_qa_features import boundary_band


CHANNEL_NAMES = []
for region in ("local", "boundary"):
    for pool in ("avg", "max"):
        for score in ("action", "add", "remove"):
            CHANNEL_NAMES.append(f"{region}_{pool}_{score}")


def pooled_region_maps(
    probs: torch.Tensor,
    region: torch.Tensor,
    output_size: tuple[int, int, int],
) -> list[torch.Tensor]:
    """Return avg/max pooled action/add/remove maps, each shaped [1,1,D,H,W]."""
    p_add = probs[:, ADD : ADD + 1]
    p_remove = probs[:, REMOVE : REMOVE + 1]
    p_action = p_add + p_remove
    region = region.to(dtype=probs.dtype)
    maps = [p_action * region, p_add * region, p_remove * region]
    avg = [F.adaptive_avg_pool3d(x, output_size) for x in maps]
    mx = [F.adaptive_max_pool3d(x, output_size) for x in maps]
    return avg + mx


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--local-radius", type=int, default=6)
    ap.add_argument("--boundary-radius", type=int, default=3)
    ap.add_argument("--spatial-size", default="32,32,32", help="D,H,W")
    args = ap.parse_args()

    spatial_size = tuple(int(x) for x in args.spatial_size.split(","))
    if len(spatial_size) != 3 or min(spatial_size) <= 0:
        raise ValueError(f"Invalid --spatial-size: {args.spatial_size}")

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model, checkpoint = make_model(cfg, args.checkpoint, device)
    rows = read_rows(cfg["manifest"], args.split)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for i, row in enumerate(rows, 1):
        x = read_binary(row["autoseg"], device)
        logits, _ = model(x)
        probs = torch.softmax(logits.float(), dim=1)

        autoseg_np = x[0, 0].detach().cpu().numpy() > 0.5
        local_np = binary_dilate(autoseg_np, args.local_radius)
        boundary_np = boundary_band(autoseg_np, args.boundary_radius)
        local = torch.from_numpy(local_np).to(device=device).unsqueeze(0).unsqueeze(0)
        boundary = torch.from_numpy(boundary_np).to(device=device).unsqueeze(0).unsqueeze(0)

        channels = []
        channels.extend(pooled_region_maps(probs, local, spatial_size))
        channels.extend(pooled_region_maps(probs, boundary, spatial_size))
        tensor = torch.cat(channels, dim=1)[0].detach().cpu().numpy().astype(np.float16)
        if tensor.shape[0] != len(CHANNEL_NAMES):
            raise RuntimeError(f"Expected {len(CHANNEL_NAMES)} channels, got {tensor.shape}")

        pid = str(row["patient_id"])
        path = out / f"patient_{pid}.npz"
        np.savez_compressed(path, spatial=tensor)
        index_rows.append({"patient_id": pid, "split": args.split, "path": path.name})
        print(f"[{i}/{len(rows)}] {args.split} patient={pid} tensor={tuple(tensor.shape)}", flush=True)

    import pandas as pd

    pd.DataFrame(index_rows).to_csv(out / "index.csv", index=False)
    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint),
        "split": args.split,
        "n_cases": len(index_rows),
        "spatial_size": list(spatial_size),
        "local_radius_voxels": args.local_radius,
        "boundary_radius_voxels": args.boundary_radius,
        "channels": CHANNEL_NAMES,
        "dtype": "float16",
        "approved_used_for_predictor_tensor": False,
        "pooling": "adaptive average and adaptive max pooling of masked probability maps",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved spatial QA tensors to {out}")


if __name__ == "__main__":
    main()
