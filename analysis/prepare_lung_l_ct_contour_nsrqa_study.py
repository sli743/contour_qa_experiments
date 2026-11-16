#!/usr/bin/env python3
"""Prepare the CT + contour NSR-QA study (Lung_L or Lung_R, Month1+2 -> Month3).

Every arm copies the v3 family-validated contour-only config (same split,
corruption families/severities, loss, optimizer, seed) and changes only what
the arm is about:

  ct_contour_corruption     CT + contour, v3 corruption (main model)
  ct_contour_no_corruption  CT + contour, identity training (clean_probability 1.0)
  ct_only_segmenter         contour input forced empty, identity targets: a plain
                            CT segmenter with the same encoder/decoder; its QA
                            score is "autoseg vs an independent segmentation"

The contour-only v3 model and the contour-only no-corruption baseline are not
retrained; the postprocess step links their existing evaluations.

Writes <study>/configs/<arm>_seed<seed>.yaml and <study>/experiment_manifest.csv
in the format read by scripts/dino_shape/run_lung_l_nsrqa_corruption_ablation_*.sh,
and links <study>/actual_autoseg_preprocessed to the v3 study's Month3 crops
(which already contain ct.nii.gz on the deployment crop).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import yaml

ARMS = ("ct_contour_corruption", "ct_contour_no_corruption", "ct_only_segmenter")

IMAGE_CONDITIONING = {
    "enabled": True,
    "ct_encoding": "auto",
    "quantize_to_uint8_grid": True,
    "hu_window": [-1000.0, 2000.0],
    "fusion": "gated_add",
    "fusion_hidden_dim": 256,
    "image_skip_channels": 16,
    "image_skip_last_n_stages": 2,
    "train_image_dropout": 0.3,
    "contour_input": "mask",
    "cache_image_tokens": True,
}


def ct_dir_for(cfg: dict, split_key: str) -> str:
    explicit = cfg["data"].get(f"{split_key}_ct_dir")
    if explicit:
        return str(explicit)
    label_dir = Path(cfg["data"][f"{split_key}_label_dir"])
    return str(label_dir.parent / "ct")


def count_nii(path: str) -> int:
    p = Path(path)
    return len(list(p.rglob("*.nii.gz"))) if p.is_dir() else -1


def arm_config(source: dict, arm: str, study_root: Path, seed: int, roi: str = "Lung_L") -> dict:
    cfg = copy.deepcopy(source)
    name = f"{roi.lower()}_{arm}_seed{seed}"
    cfg["experiment_name"] = name
    cfg["output_root"] = str(study_root / "models" / arm)
    cfg["seed"] = int(seed)
    for key in ("train", "validation", "test"):
        cfg["data"][f"{key}_ct_dir"] = ct_dir_for(source, key)

    image = copy.deepcopy(IMAGE_CONDITIONING)
    if arm in {"ct_contour_no_corruption", "ct_only_segmenter"}:
        cfg["stochastic_region_dropout"]["clean_probability"] = 1.0
    if arm == "ct_only_segmenter":
        image["contour_input"] = "empty"
        image["train_image_dropout"] = 0.0
    cfg["image_conditioning"] = image

    if bool((cfg.get("reference_frame_training") or {}).get("candidate_centered", False)):
        raise SystemExit("Source config uses candidate-centered training; not supported with CT conditioning.")

    meta = dict(cfg.get("study_metadata") or {})
    meta.update({
        "experiment": f"{roi} CT + contour NSR-QA, Month3 temporal holdout",
        "roi": roi,
        "arm": arm,
        "seed": int(seed),
        "image_conditioning_change": (
            "frozen 3DINO run on CT and on contour; gated token fusion (zero-init) plus "
            "zero-init full-resolution CT injections in the last decoder stages"
        ),
        "intervention": {
            "ct_contour_corruption": "v3 family-validated corruption; CT added as a second encoder input",
            "ct_contour_no_corruption": "identity training (clean_probability 1.0); CT added",
            "ct_only_segmenter": "contour input forced empty; identity targets; CT-only segmentation",
        }[arm],
        "source_contour_only_config": str(meta.get("source_calibrated_config", "")),
    })
    cfg["study_metadata"] = meta
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-config", required=True,
                    help="v3 contour-only config, e.g. <v3 study>/configs/nsrqa_corruption_seed91.yaml")
    ap.add_argument("--source-actual-autoseg", required=True,
                    help="<v3 study>/actual_autoseg_preprocessed (Month3 crops with ct.nii.gz)")
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--roi", default=None,
                    help="ROI name used in experiment names; default: the source config's data.roi_name, else Lung_L")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override training.num_epochs (smoke runs only).")
    ap.add_argument("--per-seed-files", action="store_true",
                    help="write experiment_manifest_seed<S>.csv / experiment_metadata_seed<S>.json "
                         "so extra seeds can share the study root without clobbering seed 91")
    args = ap.parse_args()

    source = yaml.safe_load(Path(args.source_config).read_text(encoding="utf-8"))
    src_roi = (source.get("data") or {}).get("roi_name")
    roi = args.roi or src_roi or "Lung_L"
    if src_roi and src_roi != roi:
        raise SystemExit(f"--roi {roi} does not match the source config's data.roi_name={src_roi}")
    study = Path(args.study_root)
    (study / "configs").mkdir(parents=True, exist_ok=True)
    (study / "logs").mkdir(parents=True, exist_ok=True)

    # Every training/validation label needs a CT crop of the same patient.
    audit = {}
    for key in ("train", "validation", "test"):
        labels = count_nii(source["data"][f"{key}_label_dir"])
        cts = count_nii(ct_dir_for(source, key))
        audit[key] = {"labels": labels, "ct": cts, "ct_dir": ct_dir_for(source, key)}
        if key != "test" and (cts <= 0 or cts < labels):
            raise SystemExit(f"{key}: {labels} labels but {cts} CT crops in {ct_dir_for(source, key)}")
    print("CT crop audit:", json.dumps(audit, indent=2))

    link = study / "actual_autoseg_preprocessed"
    src = Path(args.source_actual_autoseg).resolve()
    if not (src / "autoseg_centered").is_dir():
        raise SystemExit(f"Missing Month3 crops: {src}/autoseg_centered")
    if link.is_symlink() or link.exists():
        if link.resolve() != src:
            raise SystemExit(f"{link} exists and points elsewhere ({link.resolve()})")
    else:
        os.symlink(src, link, target_is_directory=True)

    rows = []
    for idx, arm in enumerate(args.arms):
        cfg = arm_config(source, arm, study, args.seed, roi)
        if args.epochs is not None:
            cfg["training"]["num_epochs"] = int(args.epochs)
            cfg["study_metadata"]["smoke_run_epochs"] = int(args.epochs)
        cfg_path = study / "configs" / f"{arm}_seed{args.seed}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        model_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
        rows.append({
            "task_index": idx,
            "arm": arm,
            "seed": args.seed,
            "config": str(cfg_path),
            "experiment_name": cfg["experiment_name"],
            "model_dir": str(model_dir),
            "checkpoint": str(model_dir / "checkpoints" / "best_decoder.pth"),
        })
        print(f"[{idx}] {arm}: {cfg_path}")

    tag = f"_seed{args.seed}" if args.per_seed_files else ""
    manifest = study / f"experiment_manifest{tag}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (study / f"experiment_metadata{tag}.json").write_text(json.dumps({
        "source_config": str(Path(args.source_config).resolve()),
        "source_actual_autoseg": str(src),
        "seed": args.seed,
        "roi": roi,
        "arms": list(args.arms),
        "ct_crop_audit": audit,
        "image_conditioning_defaults": IMAGE_CONDITIONING,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
