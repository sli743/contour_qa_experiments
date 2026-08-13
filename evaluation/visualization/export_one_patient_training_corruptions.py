#!/usr/bin/env python3
"""Export one actual example from each stochastic training branch for one patient.

The exporter calls the same mixed-corruption function used by training.  It
forces one branch at a time only for visualization, while leaving each branch's
internal magnitude/shape sampling unchanged.  The global SDF-offset branch is
exported once inward and once outward because both directions are part of that
single training branch.
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import re
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from training.corruptions.stochastic_region_dropout import stochastic_region_dropout_corruption


BRANCHES = (
    "smooth_warp",
    "distance_noise",
    "combined",
    "stochastic_region_dropout",
    "global_boundary_offset",
)


def patient_id(path: Path) -> str:
    match = re.search(r"patient_(.+?)\.nii(?:\.gz)?$", path.name)
    if match:
        return match.group(1)
    nums = re.findall(r"\d+", path.name)
    return nums[-1] if nums else path.stem.replace(".nii", "")


def write_like(array: np.ndarray, reference: sitk.Image, path: Path) -> None:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def geometry_matches(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def find_ct(label_path: Path, pid: str) -> Path | None:
    # Preprocessed ROIwise layout: .../<ROI>/labels and sibling .../<ROI>/ct.
    ct_dir = label_path.parent.parent / "ct"
    if not ct_dir.is_dir():
        return None
    preferred = ct_dir / f"ct_patient_{pid}.nii.gz"
    if preferred.is_file():
        return preferred
    matches = [p for p in sorted(ct_dir.rglob("*.nii.gz")) if patient_id(p) == pid]
    return matches[0] if matches else None


def forced_branch_cfg(cfg: Dict, branch: str, direction: str | None = None) -> Dict:
    out = copy.deepcopy(cfg)
    spec = out["stochastic_region_dropout"]
    spec["clean_probability"] = 0.0
    spec["branch_probabilities"] = {name: float(name == branch) for name in BRANCHES}
    if branch == "global_boundary_offset" and direction is not None:
        spec.setdefault("global_boundary_offset", {})["outward_probability"] = (
            1.0 if direction == "outward" else 0.0
        )
    return out


def select_label(files: Iterable[Path], requested_pid: str | None) -> Path:
    files = list(files)
    if not files:
        raise FileNotFoundError("No label NIfTI files found.")
    if requested_pid is None:
        return files[0]
    for path in files:
        if patient_id(path) == requested_pid:
            return path
    available = ", ".join(patient_id(p) for p in files[:20])
    raise FileNotFoundError(
        f"Patient {requested_pid!r} not found. First available patient IDs: {available}"
    )


def generate_accepted(
    clean: torch.Tensor,
    cfg: Dict,
    branch: str,
    seed: int,
    cache_key: str,
    direction: str | None,
    max_tries: int,
):
    branch_cfg = forced_branch_cfg(cfg, branch, direction)
    last = None
    for attempt in range(max_tries):
        rng = random.Random(seed + attempt * 10007)
        corrupted, changed, meta = stochastic_region_dropout_corruption(
            clean, branch_cfg, rng, cache_key=cache_key
        )
        last = (corrupted, changed, meta, attempt + 1)
        if str(meta.get("branch")) != "rejected_to_clean" and bool(changed.any()):
            return last
    assert last is not None
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    ap.add_argument("--patient-id", default=None, help="Default: first patient in the selected split.")
    ap.add_argument("--seed", type=int, default=None, help="Default: config seed.")
    ap.add_argument("--max-tries", type=int, default=30)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--require-ct", action="store_true", help="Fail if a geometry-matched original CT is not found.")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())
    split_key = {
        "train": "train_label_dir",
        "validation": "validation_label_dir",
        "test": "test_label_dir",
    }[args.split]
    label_dir = Path(cfg["data"][split_key]).expanduser().resolve()
    label_path = select_label(sorted(label_dir.rglob("*.nii.gz")), args.patient_id)
    pid = patient_id(label_path)

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(cfg["output_root"]).expanduser()
        / cfg["experiment_name"]
        / "training_corruption_examples"
        / f"patient_{pid}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = sitk.ReadImage(str(label_path))
    clean_np = (sitk.GetArrayFromImage(ref) > 0).astype(np.uint8)
    if not clean_np.any():
        raise ValueError(f"Selected approved contour is empty: {label_path}")
    clean = torch.from_numpy(clean_np.astype(np.float32))[None]

    write_like(clean_np, ref, out_dir / f"clean_target_patient_{pid}.nii.gz")

    ct_path = find_ct(label_path, pid)
    if ct_path is not None:
        ct = sitk.ReadImage(str(ct_path))
        if not geometry_matches(ct, ref):
            raise RuntimeError(f"CT/label geometry mismatch: {ct_path} vs {label_path}")
        sitk.WriteImage(ct, str(out_dir / f"original_ct_patient_{pid}.nii.gz"))
    else:
        if args.require_ct:
            raise FileNotFoundError(f"No sibling original CT found for patient {pid}: label={label_path}")
        print(f"WARNING: no sibling CT found for patient {pid}; contour examples will still be exported.")

    base_seed = int(args.seed if args.seed is not None else cfg.get("seed", 91))
    requests = [
        ("smooth_warp", None),
        ("distance_noise", None),
        ("combined", None),
        ("stochastic_region_dropout", None),
        ("global_boundary_offset", "inward"),
        ("global_boundary_offset", "outward"),
    ]
    rows = []

    for index, (branch, direction) in enumerate(requests):
        corrupted, changed, meta, tries = generate_accepted(
            clean=clean,
            cfg=cfg,
            branch=branch,
            direction=direction,
            seed=base_seed + index * 100003,
            cache_key=f"preview:{label_path}",
            max_tries=args.max_tries,
        )
        corrupted_np = (corrupted[0].cpu().numpy() > 0.5).astype(np.uint8)
        changed_np = (changed[0].cpu().numpy() > 0.5).astype(np.uint8)
        missing_np = ((clean_np > 0) & (corrupted_np == 0)).astype(np.uint8)
        excess_np = ((clean_np == 0) & (corrupted_np > 0)).astype(np.uint8)

        tag = branch if direction is None else f"{branch}_{direction}"
        write_like(corrupted_np, ref, out_dir / f"corrupted_input_{tag}_patient_{pid}.nii.gz")
        write_like(changed_np, ref, out_dir / f"changed_region_{tag}_patient_{pid}.nii.gz")
        write_like(missing_np, ref, out_dir / f"missing_region_{tag}_patient_{pid}.nii.gz")
        write_like(excess_np, ref, out_dir / f"excess_region_{tag}_patient_{pid}.nii.gz")

        row = {
            "patient_id": pid,
            "requested_branch": branch,
            "requested_direction": direction or "",
            "tries": tries,
            "target_voxels": int(clean_np.sum()),
            "corrupted_voxels": int(corrupted_np.sum()),
            "changed_voxels": int(changed_np.sum()),
            "missing_voxels": int(missing_np.sum()),
            "excess_voxels": int(excess_np.sum()),
            **meta,
        }
        rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "README.txt").write_text(
        "One visualization sample from each stochastic training branch.\n"
        "global_boundary_offset is shown in both inward and outward directions because both are sampled in training.\n"
        "Each corrupted_input file is accompanied by changed/missing/excess masks.\n"
        f"Config: {config_path}\nSplit: {args.split}\nPatient: {pid}\n",
        encoding="utf-8",
    )
    print(f"Saved one-patient training corruption examples to: {out_dir}")
    print(f"Patient: {pid}; exported branch examples: {len(rows)}")


if __name__ == "__main__":
    main()
