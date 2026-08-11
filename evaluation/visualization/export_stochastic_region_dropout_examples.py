#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import SimpleITK as sitk
import torch
import yaml

from training.corruptions.stochastic_region_dropout import stochastic_region_dropout_corruption


def patient_id(path: Path) -> str:
    nums = re.findall(r"\d+", path.name)
    return nums[-1] if nums else path.stem


def write_like(array, reference, path: Path):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--max_cases", type=int, default=10)
    p.add_argument("--samples_per_case", type=int, default=10)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    key = {"train":"train_label_dir", "validation":"validation_label_dir", "test":"test_label_dir"}[args.split]
    files = sorted(Path(cfg["data"][key]).rglob("*.nii.gz"))[:args.max_cases]
    out = Path(args.output_dir) if args.output_dir else Path(cfg["output_root"]) / cfg["experiment_name"] / "corruption_preview"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for case_index, path in enumerate(files):
        ref = sitk.ReadImage(str(path))
        clean_np = (sitk.GetArrayFromImage(ref) > 0).astype("uint8")
        clean = torch.from_numpy(clean_np.astype("float32"))[None]
        pid = patient_id(path)
        write_like(clean_np, ref, out / f"clean_target_patient_{pid}.nii.gz")
        for sample_index in range(args.samples_per_case):
            rng = random.Random(int(cfg.get("seed", 91)) + case_index * 1009 + sample_index * 97)
            corrupted, changed, meta = stochastic_region_dropout_corruption(clean, cfg, rng)
            branch = str(meta.get("branch", "unknown"))
            stem = f"branch_{branch}_sample_{sample_index}_patient_{pid}"
            write_like(corrupted[0].numpy().astype("uint8"), ref, out / f"corrupted_input_{stem}.nii.gz")
            write_like(changed[0].numpy().astype("uint8"), ref, out / f"changed_region_{stem}.nii.gz")
            missing = ((clean > 0.5) & (corrupted < 0.5)).float()
            excess = ((clean < 0.5) & (corrupted > 0.5)).float()
            write_like(missing[0].numpy().astype("uint8"), ref, out / f"missing_region_{stem}.nii.gz")
            write_like(excess[0].numpy().astype("uint8"), ref, out / f"excess_region_{stem}.nii.gz")
            rows.append({"patient_id":pid,"sample_index":sample_index,**meta})

    if rows:
        fields = sorted({k for row in rows for k in row})
        with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"Saved stochastic-region-dropout preview to: {out}")


if __name__ == "__main__":
    main()
