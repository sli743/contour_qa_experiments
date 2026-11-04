#!/usr/bin/env python3
"""Prepare a one-seed Lung_L NSR-QA corruption ablation on month-temporal splits.

This is deliberately a split-only follow-up to the earlier
``lung_l_nsrqa_corruption_ablation_v1`` experiment:

* training1/validation1 are inherited from the leakage-safe Month1+Month2
  temporal study;
* testing1 is the Month3 temporal holdout;
* a single seed is used for each arm;
* the two arms differ only in whether the original NSR-QA stochastic
  corruption mixture is enabled during training/validation.

The corruption severity source is supplied separately so the same script can be
used either for a direct split-matched rerun of the earlier ablation or, later,
for a newly validated severity configuration.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List

import yaml

from analysis.prepare_lung_l_nsrqa_corruption_ablation import (
    EXPECTED_BRANCHES,
    EXPECTED_CLEAN,
    arm_config,
    validate_original_family_set,
)

DATA_KEYS = (
    "train_label_dir",
    "validation_label_dir",
    "test_label_dir",
    "train_ct_dir",
    "validation_ct_dir",
    "test_ct_dir",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calibrated-config", required=True)
    p.add_argument("--temporal-reference-config", required=True)
    p.add_argument("--temporal-split-summary", required=True)
    p.add_argument("--study-root", required=True)
    p.add_argument("--seed", type=int, default=91)
    return p.parse_args()


def nifti_count(directory: str | Path) -> int:
    p = Path(directory).expanduser().resolve()
    return sum(1 for _ in p.glob("*.nii.gz")) if p.is_dir() else 0


def write_manifest(rows: List[Dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    calibrated_path = Path(args.calibrated_config).expanduser().resolve()
    temporal_cfg_path = Path(args.temporal_reference_config).expanduser().resolve()
    split_summary_path = Path(args.temporal_split_summary).expanduser().resolve()
    study_root = Path(args.study_root).expanduser().resolve()

    for path in (calibrated_path, temporal_cfg_path, split_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = yaml.safe_load(calibrated_path.read_text(encoding="utf-8"))
    temporal_ref = yaml.safe_load(temporal_cfg_path.read_text(encoding="utf-8"))
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    validate_original_family_set(base)

    temporal_data = temporal_ref.get("data", {})
    missing_keys = [k for k in DATA_KEYS if not temporal_data.get(k)]
    if missing_keys:
        raise KeyError(f"Temporal Lung_L config missing data paths: {missing_keys}")

    split = split_summary.get("split", {})
    for key in ("training1", "validation1", "testing1"):
        if key not in split:
            raise KeyError(f"Temporal split summary missing {key}")
    train_ids = set(map(int, split["training1"]))
    val_ids = set(map(int, split["validation1"]))
    test_ids = set(map(int, split["testing1"]))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Temporal split summary contains patient leakage")

    month3_ids = set(map(int, split_summary.get("split", {}).get("testing1", [])))
    if test_ids != month3_ids:
        raise RuntimeError("testing1 does not match temporal split summary Month3 holdout")

    # Strong guard against accidentally pointing back to the legacy ~15% random
    # test split.  The exact available Lung_L count is ROI-specific, but the
    # Month3 source cohort is substantially larger than the old ~20-case split.
    n_test_labels = nifti_count(temporal_data["test_label_dir"])
    if n_test_labels < 40:
        raise RuntimeError(
            f"Temporal Lung_L test_label_dir has only {n_test_labels} cases: "
            f"{temporal_data['test_label_dir']}. Expected a Month3-sized cohort, "
            "not the legacy random testing1 split."
        )

    counts = {
        "training1_labels": nifti_count(temporal_data["train_label_dir"]),
        "validation1_labels": nifti_count(temporal_data["validation_label_dir"]),
        "testing1_labels": n_test_labels,
    }
    for key, n in counts.items():
        if n <= 0:
            raise RuntimeError(f"Empty temporal Lung_L partition: {key}")

    config_dir = study_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (study_root / "models").mkdir(parents=True, exist_ok=True)
    (study_root / "evaluations").mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    seed = int(args.seed)
    for arm in ("nsrqa_corruption", "no_corruption"):
        cfg = arm_config(copy.deepcopy(base), study_root, arm, seed)
        cfg.setdefault("data", {})
        for key in DATA_KEYS:
            cfg["data"][key] = str(Path(temporal_data[key]).expanduser().resolve())

        meta = cfg.setdefault("study_metadata", {})
        meta.update({
            "experiment": "Lung_L real-autoseg NSR-QA corruption ablation, Month3 temporal holdout",
            "split_design": "Month1+Month2 development; Month3 locked temporal test",
            "training1_semantics": "Month1+Month2 temporal-development training subset",
            "validation1_semantics": "Month1+Month2 temporal-development validation subset",
            "testing1_semantics": "Month3 temporal holdout",
            "temporal_split_summary": str(split_summary_path),
            "temporal_reference_config": str(temporal_cfg_path),
            "source_calibrated_config": str(calibrated_path),
            "single_seed_ablation": True,
            "month3_patient_ids_before_roi_availability_filter": len(test_ids),
            "lung_l_test_labels_available": n_test_labels,
        })

        out = config_dir / f"{arm}_seed{seed}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        model_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
        rows.append({
            "task_index": len(rows),
            "arm": arm,
            "seed": seed,
            "config": str(out),
            "experiment_name": cfg["experiment_name"],
            "model_dir": str(model_dir),
            "checkpoint": str(model_dir / "checkpoints" / "best_decoder.pth"),
        })

    # Verify that the only intentional arm difference in the corruption block is
    # clean_probability.  Data paths and seed are shared exactly.
    c0 = yaml.safe_load(Path(rows[0]["config"]).read_text(encoding="utf-8"))
    c1 = yaml.safe_load(Path(rows[1]["config"]).read_text(encoding="utf-8"))
    if c0["data"] != c1["data"]:
        raise RuntimeError("Ablation arms do not share identical temporal data paths")
    if int(c0["seed"]) != int(c1["seed"]):
        raise RuntimeError("Ablation arms do not share the same seed")

    manifest = study_root / "experiment_manifest.csv"
    write_manifest(rows, manifest)

    metadata = {
        "study_root": str(study_root),
        "source_calibrated_config": str(calibrated_path),
        "temporal_reference_config": str(temporal_cfg_path),
        "temporal_split_summary": str(split_summary_path),
        "seed": seed,
        "arms": ["nsrqa_corruption", "no_corruption"],
        "n_training_tasks": 2,
        "split_semantics": {
            "training1": "Month1+Month2 training subset",
            "validation1": "Month1+Month2 validation subset",
            "testing1": "Month3 temporal holdout",
        },
        "nominal_temporal_patient_counts": {
            "training1": len(train_ids),
            "validation1": len(val_ids),
            "testing1": len(test_ids),
        },
        "lung_l_available_label_counts": counts,
        "frozen_original_nsrqa_clean_probability": EXPECTED_CLEAN,
        "frozen_original_nsrqa_branch_probabilities": EXPECTED_BRANCHES,
        "primary_qa_signal": "1-Dice(autoseg, reconstruction)",
        "note": (
            "testing1 is intentionally retained as the directory name for compatibility, "
            "but now means the Month3 temporal holdout rather than the legacy random 15% test split."
        ),
    }
    (study_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Keep a frozen copy of the split definition next to the experiment.
    (study_root / "temporal_split_summary.json").write_text(
        json.dumps(split_summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Prepared Month-temporal Lung_L ablation under {study_root}")
    print(f"  seed: {seed}")
    print(f"  temporal nominal patients: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(
        "  Lung_L available labels: "
        f"train={counts['training1_labels']} val={counts['validation1_labels']} test={counts['testing1_labels']}"
    )
    for row in rows:
        print(f"  [{row['task_index']}] {row['arm']} -> {row['config']}")


if __name__ == "__main__":
    main()
