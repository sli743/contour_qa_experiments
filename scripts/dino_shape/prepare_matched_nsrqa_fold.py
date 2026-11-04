#!/usr/bin/env python3
"""Prepare a split/preprocessing-matched previous-NSR-QA baseline for one outer fold.

The stochastic-region-dropout NSR-QA training recipe is copied verbatim from the
existing all-ROI NSR-QA config. Only patient split/data paths and output paths are
changed. Approved masks come from the exact autoseg-derived grids used by the
supervised-correction experiment, which isolates learning strategy from crop/split
changes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml


def read_manifest(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ensure_symlink(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--study-root", default="/scratch/shaohan/contour_QA/supervised_autoseg_correction")
    ap.add_argument("--all-roi-root", default="/scratch/shaohan/contour_QA/all_roi_NSRQA_study")
    ap.add_argument("--baseline-root", default=None)
    args = ap.parse_args()

    study_root = Path(args.study_root).expanduser().resolve()
    all_roi_root = Path(args.all_roi_root).expanduser().resolve()
    baseline_root = Path(args.baseline_root).expanduser().resolve() if args.baseline_root else study_root / "matched_previous_nsrqa_5fold"

    manifest = study_root / "manifests" / args.roi / f"fold_{args.fold}.csv"
    reference_cfg_path = all_roi_root / "configs" / args.roi / "nsrqa.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not reference_cfg_path.is_file():
        raise FileNotFoundError(
            f"Previous NSR-QA reference config missing: {reference_cfg_path}. "
            "Run the existing all-ROI NSR-QA prepare step first."
        )

    rows = read_manifest(manifest)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest}")
    expected_splits = {"train", "validation", "test"}
    got_splits = {r.get("split", "") for r in rows}
    if not expected_splits.issubset(got_splits):
        raise RuntimeError(f"Manifest lacks required splits {expected_splits}: got {got_splits}")

    fold_root = baseline_root / args.roi / f"fold_{args.fold}"
    data_root = fold_root / "matched_clean_labels"
    split_dirs = {s: data_root / s for s in expected_splits}
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        # Remove stale links/files only; never follow/remove source files.
        for p in d.iterdir():
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    split_counts = {s: 0 for s in expected_splits}
    spacings = []
    for r in rows:
        split = r["split"]
        if split not in expected_splits:
            continue
        src = Path(r["approved"]).expanduser().resolve()
        pid = str(r["patient_id"])
        dst = split_dirs[split] / f"patient{pid}_{args.roi}_approved.nii.gz"
        ensure_symlink(src, dst)
        split_counts[split] += 1
        raw_spacing = r.get("spacing_xyz", "")
        if raw_spacing:
            try:
                spacings.append(tuple(float(x) for x in json.loads(raw_spacing)))
            except Exception:
                pass

    for split, n in split_counts.items():
        if n == 0:
            raise RuntimeError(f"No {split} cases materialized for {args.roi} fold {args.fold}")

    ref_cfg = yaml.safe_load(reference_cfg_path.read_text())
    cfg = copy.deepcopy(ref_cfg)
    cfg["experiment_name"] = f"matched_previous_nsrqa_{args.roi}_fold{args.fold}"
    cfg["output_root"] = str(fold_root / "models")
    # Never inherit a trained-checkpoint override from a legacy evaluation config;
    # the matched baseline must load the checkpoint trained for this outer fold.
    if isinstance(cfg.get("paths"), dict):
        cfg["paths"].pop("trained_decoder_checkpoint", None)
    cfg.setdefault("data", {})
    cfg["data"]["roi_name"] = args.roi
    cfg["data"]["train_label_dir"] = str(split_dirs["train"])
    cfg["data"]["validation_label_dir"] = str(split_dirs["validation"])
    cfg["data"]["test_label_dir"] = str(split_dirs["test"])
    # CT is not consumed by the shape-only NSR-QA trainer/evaluator. Remove old
    # split-specific CT paths to avoid implying they are part of this baseline.
    for key in ("train_ct_dir", "validation_ct_dir", "test_ct_dir"):
        cfg["data"].pop(key, None)

    if spacings:
        unique = {tuple(round(v, 6) for v in s) for s in spacings}
        if len(unique) != 1:
            raise RuntimeError(f"Mixed spacing within matched fold: {sorted(unique)}")
        xyz = next(iter(unique))
        cfg.setdefault("evaluation", {})["spacing_zyx"] = list(xyz[::-1])

    # Preserve the previous NSR-QA model/corruption/loss/training recipe exactly.
    # Metadata makes clear which pieces were intentionally matched.
    cfg["comparison_metadata"] = {
        "role": "previous_NSRQA_split_and_preprocessing_matched_baseline",
        "outer_fold": int(args.fold),
        "source_previous_nsrqa_config": str(reference_cfg_path),
        "matched_supervised_manifest": str(manifest),
        "matched_grid": "same autoseg-derived preprocessed approved masks used by supervised correction",
        "training_input": "approved masks with previous NSR-QA stochastic synthetic corruptions",
        "encoder": "previous NSR-QA frozen 3DINO encoder (unchanged from source config/trainer)",
        "test_input": "real autosegmentation on the same matched grid",
        "no_test_patients_used_for_training": True,
    }

    config_path = fold_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (fold_root / "matched_split_counts.json").write_text(json.dumps(split_counts, indent=2))
    print(config_path)


if __name__ == "__main__":
    main()
