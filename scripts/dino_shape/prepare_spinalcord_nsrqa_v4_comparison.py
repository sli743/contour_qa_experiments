#!/usr/bin/env python3
"""Prepare the controlled previous-generation vs v4 SpinalCord NSR-QA study.

The source month-temporal large-FOV study already fixes patient membership,
preprocessing/FOV, architecture, loss, and optimizer settings. This preparer
copies its generated SpinalCord NSR-QA config twice and changes only the
corruption backend/output provenance for the v4 arm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_spinalcord_manifest(study_root: Path) -> dict:
    path = study_root / "manifests" / "active_rois.tsv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [r for r in rows if str(r.get("roi", "")).strip() == "SpinalCord"]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one SpinalCord row in {path}; found {len(rows)}")
    return rows[0]


def audit_v4_endpoint_fov(source_root: Path, v4_distribution: dict) -> dict:
    """Audit development FOV headroom for v4 S/I endpoint over-extension.

    The large-FOV crop is centered at the approved SpinalCord bbox midpoint.
    For that centering convention, the per-side longitudinal headroom is half
    of (output center-to-center span - approved bbox center span).  V4 may
    deliberately extend an endpoint, so record whether the configured support
    can fit before training instead of silently clipping it.
    """
    spec_path = source_root / "manifests" / "spinalcord_large_fov_spec.json"
    audit_path = source_root / "manifests" / "spinalcord_large_fov_coverage_audit.csv"
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    z_span = float(spec["center_to_center_span_xyz_mm"][2])
    rows = []
    with audit_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("split", "")).strip() not in {"training1", "validation1"}:
                continue
            approved_span = float(row["raw_approved_bbox_center_span_z_mm"])
            rows.append({
                "patient_id": str(row.get("patient_id", "")),
                "split": str(row.get("split", "")),
                "approved_bbox_center_span_z_mm": approved_span,
                "per_side_endpoint_headroom_mm": 0.5 * (z_span - approved_span),
            })
    if not rows:
        raise RuntimeError(f"No development rows found in FOV audit: {audit_path}")

    endpoint = v4_distribution.get("endpoint_v4", {})
    excess_support = endpoint.get("inferior_excess_length_mm", [0.0, 0.0])
    excess_min = float(excess_support[0])
    excess_max = float(excess_support[1])
    min_row = min(rows, key=lambda r: r["per_side_endpoint_headroom_mm"])
    min_headroom = float(min_row["per_side_endpoint_headroom_mm"])
    if min_headroom + 1e-6 < excess_min:
        raise RuntimeError(
            "V4 endpoint-excess support does not fit the development FOV: "
            f"minimum per-side headroom={min_headroom:.3f} mm for patient "
            f"{min_row['patient_id']} ({min_row['split']}), but configured inferior "
            f"excess begins at {excess_min:.3f} mm. Increase the FOV or reduce the "
            "configured excess support before training."
        )

    status = "full_support_fits" if min_headroom + 1e-6 >= excess_max else "upper_tail_may_clip"
    return {
        "source_fov_spec": str(spec_path),
        "source_fov_coverage_audit": str(audit_path),
        "fov_center_to_center_z_span_mm": z_span,
        "configured_inferior_excess_min_mm": excess_min,
        "configured_inferior_excess_max_mm": excess_max,
        "minimum_development_per_side_headroom_mm": min_headroom,
        "minimum_headroom_patient_id": min_row["patient_id"],
        "minimum_headroom_split": min_row["split"],
        "status": status,
        "note": (
            "All configured inferior-excess support fits the development crop."
            if status == "full_support_fits"
            else "The configured lower bound fits every development crop, but the extreme upper tail can reach the crop face for the longest cords. This is recorded as a preflight warning; no Month3 information is used to change the FOV or sampler."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-study-root",
        default="/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study",
    )
    p.add_argument(
        "--output-root",
        default="/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare",
    )
    p.add_argument(
        "--v4-config",
        default="configs/dino_shape/spinalcord_proposed_corruption_distribution_v4.yaml",
    )
    args = p.parse_args()

    source_root = Path(args.source_study_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    v4_path = Path(args.v4_config).expanduser().resolve()
    if not v4_path.is_file():
        raise FileNotFoundError(v4_path)

    row = read_spinalcord_manifest(source_root)
    previous_source = Path(row["nsrqa_config"]).expanduser().resolve()
    if not previous_source.is_file():
        raise FileNotFoundError(previous_source)
    previous = yaml.safe_load(previous_source.read_text(encoding="utf-8"))
    v4_distribution = yaml.safe_load(v4_path.read_text(encoding="utf-8"))
    fov_endpoint_audit = audit_v4_endpoint_fov(source_root, v4_distribution)
    if fov_endpoint_audit["status"] != "full_support_fits":
        print("WARNING: " + fov_endpoint_audit["note"])
        print(
            "  minimum development endpoint headroom: "
            f"{fov_endpoint_audit['minimum_development_per_side_headroom_mm']:.3f} mm; "
            "configured inferior-excess max: "
            f"{fov_endpoint_audit['configured_inferior_excess_max_mm']:.3f} mm"
        )

    # Explicitly preserve the previous-generation backend even if the trainer's
    # default changes in the future.
    previous_cfg = deepcopy(previous)
    previous_cfg["corruption_backend"] = "legacy_stochastic_region_dropout"
    previous_cfg["experiment_name"] = "dino_shape_NSRQA_SpinalCord_previous_generation"
    previous_cfg["output_root"] = str(out_root / "models" / "previous_generation")
    previous_cfg.setdefault("study_metadata", {}).update({
        "comparison_study": "SpinalCord previous-generation vs v4 corruption",
        "comparison_arm": "previous_generation",
        "source_month_temporal_config": str(previous_source),
        "scientific_intervention": "none_reference_arm",
    })

    v4_cfg = deepcopy(previous)
    v4_cfg["corruption_backend"] = "spinalcord_proposed"
    v4_cfg["spinalcord_proposed_distribution"] = v4_distribution
    v4_cfg["experiment_name"] = "dino_shape_NSRQA_SpinalCord_v4_corruption"
    v4_cfg["output_root"] = str(out_root / "models" / "v4_corruption")
    v4_cfg.setdefault("study_metadata", {}).update({
        "comparison_study": "SpinalCord previous-generation vs v4 corruption",
        "comparison_arm": "v4_corruption",
        "source_month_temporal_config": str(previous_source),
        "scientific_intervention": "training_corruption_distribution_only",
        "v4_distribution_config": str(v4_path),
        "v4_distribution_config_sha256": sha256_file(v4_path),
        "month3_used_for_corruption_design": False,
    })

    config_dir = out_root / "configs"
    manifest_dir = out_root / "manifests"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    previous_out = config_dir / "previous_generation.yaml"
    v4_out = config_dir / "v4_corruption.yaml"
    previous_out.write_text(yaml.safe_dump(previous_cfg, sort_keys=False), encoding="utf-8")
    v4_out.write_text(yaml.safe_dump(v4_cfg, sort_keys=False), encoding="utf-8")

    model_rows = [
        {
            "model_key": "previous_generation",
            "model_label": "Previous-generation NSR-QA",
            "config": str(previous_out),
            "checkpoint": str(Path(previous_cfg["output_root"]) / previous_cfg["experiment_name"] / "checkpoints" / "best_decoder.pth"),
            "corruption_backend": previous_cfg["corruption_backend"],
        },
        {
            "model_key": "v4_corruption",
            "model_label": "V4 corruption NSR-QA",
            "config": str(v4_out),
            "checkpoint": str(Path(v4_cfg["output_root"]) / v4_cfg["experiment_name"] / "checkpoints" / "best_decoder.pth"),
            "corruption_backend": v4_cfg["corruption_backend"],
        },
    ]
    model_manifest = manifest_dir / "model_configs.tsv"
    with model_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(model_rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(model_rows)

    # Link/copy provenance by path rather than altering the source study.
    provenance = {
        "source_study_root": str(source_root),
        "source_active_roi_manifest": str(source_root / "manifests" / "active_rois.tsv"),
        "source_temporal_split_summary": str(source_root / "manifests" / "temporal_split_summary.json"),
        "source_fov_spec": str(source_root / "manifests" / "spinalcord_large_fov_spec.json"),
        "source_previous_config": str(previous_source),
        "source_previous_config_sha256": sha256_file(previous_source),
        "v4_distribution_config": str(v4_path),
        "v4_distribution_config_sha256": sha256_file(v4_path),
        "output_root": str(out_root),
        "controlled_difference": "training corruption distribution only",
        "primary_actual_autoseg_localization": "autoseg_centered",
        "oracle_sensitivity_localization": "approved_reference",
        "primary_test_split": "testing1 (Month3)",
        "development_splits": ["training1", "validation1"],
        "v4_endpoint_fov_audit": fov_endpoint_audit,
    }
    (manifest_dir / "comparison_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (manifest_dir / "v4_endpoint_fov_audit.json").write_text(json.dumps(fov_endpoint_audit, indent=2), encoding="utf-8")

    print("Prepared controlled SpinalCord NSR-QA comparison")
    print(f"  source study: {source_root}")
    print(f"  output root:  {out_root}")
    print(f"  previous:     {previous_out}")
    print(f"  v4:           {v4_out}")
    print(f"  manifest:     {model_manifest}")


if __name__ == "__main__":
    main()
