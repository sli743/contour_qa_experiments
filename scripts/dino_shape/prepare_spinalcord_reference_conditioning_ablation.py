#!/usr/bin/env python3
"""Prepare the four-arm SpinalCord reference-frame conditioning ablation.

All arms inherit the already calibrated V4 corruption config, temporal split,
coverage-safe FOV, loss, optimizer, decoder width, and frozen 3DINO encoder.
Only deployment-matched candidate centering, optional global spatial context,
and optional small training-only FOV jitter differ.
"""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import yaml

from training.spatial_reference import SPATIAL_CONTEXT_FEATURES


ARMS = (
    ("centered", False, False),
    ("centered_context", True, False),
    ("centered_jitter", False, True),
    ("centered_context_jitter", True, True),
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-compare-root",
        default="/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare",
        help="Existing controlled V4 comparison root; configs/v4_corruption.yaml is the source config.",
    )
    p.add_argument(
        "--output-root",
        default="/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_reference_ablation",
    )
    p.add_argument(
        "--raw-data-root",
        default="../radformation-data/data",
        help="Raw Radformation root containing images/, labels-approved/, labels-autoseg/.",
    )
    p.add_argument("--jitter-fraction-of-fov", type=float, default=0.025)
    p.add_argument("--jitter-max-mm", type=float, default=6.0)
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    source_root = Path(args.source_compare_root).expanduser().resolve()
    source_cfg_path = source_root / "configs" / "v4_corruption.yaml"
    if not source_cfg_path.is_file():
        raise FileNotFoundError(source_cfg_path)
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    if str(source_cfg.get("corruption_backend", "")) != "spinalcord_proposed":
        raise ValueError(f"Expected V4 spinalcord_proposed source config: {source_cfg_path}")

    raw_root = Path(args.raw_data_root).expanduser()
    if not raw_root.is_absolute():
        raw_root = (project_root / raw_root).resolve()
    if not (raw_root / "images").is_dir():
        raise FileNotFoundError(f"Raw CT image directory not found: {raw_root / 'images'}")

    out_root = Path(args.output_root).expanduser().resolve()
    cfg_dir = out_root / "configs"
    manifest_dir = out_root / "manifests"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm_index, (arm, use_context, use_jitter) in enumerate(ARMS):
        cfg = deepcopy(source_cfg)
        cfg["experiment_name"] = f"dino_shape_NSRQA_SpinalCord_reference_{arm}"
        cfg["output_root"] = str(out_root / "models" / arm)
        # Reframing happens after V4 corruption. Validation uses the same
        # candidate-centered representation but never random jitter.
        cfg["reference_frame_training"] = {
            "candidate_centered": True,
            "center_definition": "corrupted_candidate_bbox_midpoint",
            "integer_grid_translation": True,
            "fov_jitter": {
                "enabled": bool(use_jitter),
                "training_only": True,
                "probability": 1.0,
                "fraction_of_fov": float(args.jitter_fraction_of_fov),
                "max_mm": float(args.jitter_max_mm),
                "max_resample_attempts": 20,
                "reject_jitter_if_approved_target_would_clip": True,
                "candidate_centering_may_reproduce_deployment_clipping": True,
            },
        }
        cfg["spatial_conditioning"] = {
            "enabled": bool(use_context),
            "raw_data_root": str(raw_root),
            "context_dim": len(SPATIAL_CONTEXT_FEATURES),
            "hidden_dim": 64,
            "injection": "decoder_projection_film",
            "feature_names": list(SPATIAL_CONTEXT_FEATURES),
            "coordinate_system": "raw_ct_continuous_index_fraction",
            "encoder_input_unchanged": True,
        }
        # The legacy validation cache stores only corrupted/changed masks and
        # therefore cannot preserve a corruption-specific candidate recentering.
        cfg.setdefault("training", {}).setdefault("validation_cache", {})["enabled"] = False
        cfg.setdefault("study_metadata", {}).update({
            "comparison_study": "SpinalCord deployment-matched reference-frame ablation",
            "comparison_arm": arm,
            "source_v4_config": str(source_cfg_path),
            "v4_corruption_distribution_unchanged": True,
            "candidate_centered_training": True,
            "spatial_conditioning": bool(use_context),
            "training_fov_jitter": bool(use_jitter),
            "month3_used_for_design": False,
            "primary_evaluation_localization": "autoseg_centered",
        })
        path = cfg_dir / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        checkpoint = Path(cfg["output_root"]) / cfg["experiment_name"] / "checkpoints" / "best_decoder.pth"
        rows.append({
            "arm_index": arm_index,
            "arm": arm,
            "model_key": arm,
            "model_label": arm.replace("_", " ").title(),
            "use_spatial_context": int(use_context),
            "use_fov_jitter": int(use_jitter),
            "config": str(path),
            "checkpoint": str(checkpoint),
        })

    # Retain a canonical copy so the existing V4-family diagnostic can use the
    # exact same corruption definition without reaching back into another study.
    (cfg_dir / "v4_corruption.yaml").write_text(source_cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = manifest_dir / "model_configs.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    provenance = {
        "source_compare_root": str(source_root),
        "source_v4_config": str(source_cfg_path),
        "raw_data_root": str(raw_root),
        "output_root": str(out_root),
        "arms": [r["arm"] for r in rows],
        "controlled_constants": [
            "patient split", "coverage-safe FOV", "V4 corruption distribution",
            "3DINO encoder", "decoder channels", "loss", "optimizer", "seed", "epochs",
        ],
        "interventions": {
            "candidate_centered_training": "all four arms",
            "spatial_context": "B and D only",
            "training_only_fov_jitter": "C and D only",
        },
        "spatial_context_features": list(SPATIAL_CONTEXT_FEATURES),
        "jitter_rule": {
            "fraction_of_fov": float(args.jitter_fraction_of_fov),
            "max_mm": float(args.jitter_max_mm),
            "per_axis_max": "min(max_mm, fraction_of_fov * center-to-center FOV span)",
            "jitter_induced_approved_target_clipping": "rejected",
            "candidate_centering_clipping": "allowed to match deployment; logged per sample",
        },
    }
    (manifest_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"Prepared reference-frame ablation: {out_root}")
    for row in rows:
        print(
            f"  {row['arm_index']}: {row['arm']} "
            f"context={row['use_spatial_context']} jitter={row['use_fov_jitter']}"
        )
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
