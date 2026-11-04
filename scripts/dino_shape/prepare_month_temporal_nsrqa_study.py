#!/usr/bin/env python3
"""Prepare a month-based temporal NSR-QA study without touching the old study.

The existing all-ROI trainers/evaluators expect the standard directory names
``training1``, ``validation1`` and ``testing1``.  This script creates a new
symlink-only data tree with exactly that layout:

* training1/validation1: patients from month1 + month2
* testing1: month3 patients only

The month3 set has priority for duplicate patient IDs.  The development pool is
split into training/validation *within month1 and month2* using a fixed seed so
both periods are represented in validation.  Once the temporal data tree is
materialized, the original ``prepare_all_roi_nsrqa_study.py`` is called to
create the same four configs/manifests as the August-13 study, but under a new
study root.

Only symlinks are created by default; the source preprocessed data are never
modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

SPLITS = ("training1", "validation1", "testing1")
KINDS = ("labels", "ct")
PID_RE = re.compile(r"patient_?(\d+)", re.IGNORECASE)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def patient_id(path: Path) -> int:
    m = PID_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse patient ID from {path}")
    return int(m.group(1))


def unique_ints(values: Iterable[int]) -> List[int]:
    return sorted({int(x) for x in values})


def choose_validation(
    month_ids: Sequence[int],
    held_out: set[int],
    fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    dev = np.array(sorted(set(map(int, month_ids)) - held_out), dtype=int)
    if len(dev) < 2:
        raise ValueError(f"Need at least 2 development patients; got {len(dev)}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(dev)
    n_val = int(round(len(dev) * fraction))
    n_val = max(1, min(len(dev) - 1, n_val))
    val = sorted(int(x) for x in shuffled[:n_val])
    train = sorted(int(x) for x in shuffled[n_val:])
    return train, val


def make_split(cfg: Mapping) -> Tuple[Dict[str, List[int]], List[Dict]]:
    month1 = unique_ints(cfg["month1"])
    month2 = unique_ints(cfg["month2"])
    month3 = unique_ints(cfg["month3"])
    test = set(month3)

    overlaps = []
    for month_name, ids in (("month1", month1), ("month2", month2)):
        for pid in sorted(set(ids) & test):
            overlaps.append({
                "patient_id": pid,
                "earlier_month": month_name,
                "later_month": "month3",
                "resolution": "testing1_month3_priority",
            })

    fraction = float(cfg.get("validation_fraction", 0.20))
    seed = int(cfg.get("seed", 88))
    tr1, va1 = choose_validation(month1, test, fraction, seed + 101)
    tr2, va2 = choose_validation(month2, test, fraction, seed + 202)
    split = {
        "training1": sorted(tr1 + tr2),
        "validation1": sorted(va1 + va2),
        "testing1": sorted(test),
    }

    train, val, test_ids = map(set, (split["training1"], split["validation1"], split["testing1"]))
    if train & val or train & test_ids or val & test_ids:
        raise AssertionError("Patient leakage remains after temporal split construction")
    return split, overlaps


def source_dir(root: Path, group: str, roi: str, split: str, kind: str) -> Path:
    return root / group / roi / split / group / roi / kind


def target_dir(root: Path, group: str, roi: str, split: str, kind: str) -> Path:
    return root / group / roi / split / group / roi / kind


def index_source_files(source_root: Path, group: str, roi: str) -> Dict[str, Dict[int, Path]]:
    out: Dict[str, Dict[int, Path]] = {kind: {} for kind in KINDS}
    origins: Dict[Tuple[str, int], List[Path]] = defaultdict(list)
    for old_split in SPLITS:
        for kind in KINDS:
            d = source_dir(source_root, group, roi, old_split, kind)
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.nii.gz")):
                pid = patient_id(p)
                origins[(kind, pid)].append(p)

    duplicates = {k: v for k, v in origins.items() if len(v) > 1}
    if duplicates:
        msg = [f"Duplicate source patient files for {roi}; refusing ambiguous remap:"]
        for (kind, pid), paths in sorted(duplicates.items()):
            msg.append(f"  {kind} patient {pid}: " + ", ".join(map(str, paths)))
        raise RuntimeError("\n".join(msg))

    for (kind, pid), paths in origins.items():
        out[kind][pid] = paths[0]
    return out


def materialize_file(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def sha256_json(obj: Mapping) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_tsv(path: Path, rows: List[Dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def split_source_month(pid: int, cfg: Mapping) -> str:
    memberships = [m for m in ("month1", "month2", "month3") if pid in set(map(int, cfg[m]))]
    if "month3" in memberships:
        return "month3"
    return "+".join(memberships) if memberships else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-config", default="configs/dino_shape/month_temporal_nsrqa_split.yaml")
    ap.add_argument("--study-root", default=None)
    ap.add_argument("--source-data-root", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--copy-files", action="store_true", help="Copy NIfTI files instead of creating symlinks.")
    ap.add_argument("--force", action="store_true", help="Allow rebuilding split/config manifests when models already exist.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    split_cfg_path = Path(args.split_config)
    if not split_cfg_path.is_absolute():
        split_cfg_path = project_root / split_cfg_path
    cfg = load_yaml(split_cfg_path.resolve())

    study_root = Path(args.study_root or cfg["study_root"]).expanduser().resolve()
    source_root = Path(args.source_data_root or cfg["source_data_root"]).expanduser().resolve()
    registry_path = Path(args.registry or cfg.get("registry", "configs/dino_shape/all_roi_nsrqa_registry.yaml"))
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path
    registry_path = registry_path.resolve()
    registry = load_yaml(registry_path)

    reference_rois = list(cfg.get("reference_active_rois", registry["rois"].keys()))
    unknown_rois = [roi for roi in reference_rois if roi not in registry["rois"]]
    if unknown_rois:
        raise ValueError(f"reference_active_rois contains ROIs absent from registry: {unknown_rois}")
    if len(reference_rois) != len(set(reference_rois)):
        raise ValueError("reference_active_rois contains duplicates")
    # Keep the exact completed August-13 ROI set/order. This prevents a newly
    # available ROI (notably NS_Chestwall) from silently changing the study.
    study_registry = dict(registry)
    study_registry["rois"] = {roi: registry["rois"][roi] for roi in reference_rois}

    models_root = study_root / "models"
    if models_root.exists() and any(models_root.rglob("*.pth")) and not args.force:
        raise RuntimeError(
            f"{models_root} already contains checkpoints. Refusing to change the patient split. "
            "Use a new --study-root, or --force only if you intentionally want to invalidate those models."
        )

    split, overlaps = make_split(cfg)
    month_by_pid = {pid: split_source_month(pid, cfg) for pid in set(sum((list(map(int, cfg[m])) for m in ("month1", "month2", "month3")), []))}

    temporal_data_root = study_root / "data" / "preprocessed_roiwise_allrois_1p5mm_nozoom_temporal"
    if temporal_data_root.exists():
        shutil.rmtree(temporal_data_root)
    temporal_data_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = study_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    filtered_registry_path = manifest_dir / "temporal_reference_registry.yaml"
    filtered_registry_path.write_text(yaml.safe_dump(study_registry, sort_keys=False), encoding="utf-8")

    patient_rows = []
    for target_split, ids in split.items():
        for pid in ids:
            patient_rows.append({
                "patient_id": pid,
                "source_month": month_by_pid.get(pid, "unknown"),
                "target_split": target_split,
                "held_out_test": target_split == "testing1",
            })
    patient_rows.sort(key=lambda r: (r["target_split"], int(r["patient_id"])))
    write_tsv(manifest_dir / "temporal_split_patients.tsv", patient_rows,
              ["patient_id", "source_month", "target_split", "held_out_test"])
    write_tsv(manifest_dir / "temporal_split_overlap_resolution.tsv", overlaps,
              ["patient_id", "earlier_month", "later_month", "resolution"])

    audit_rows: List[Dict] = []
    count_rows: List[Dict] = []
    for roi, rspec in study_registry["rois"].items():
        group = rspec["data_group"]
        idx = index_source_files(source_root, group, roi)
        counts = Counter()
        for target_split, ids in split.items():
            for pid in ids:
                label_src = idx["labels"].get(pid)
                ct_src = idx["ct"].get(pid)
                status = "linked"
                reason = ""
                if label_src is None and ct_src is None:
                    status, reason = "missing", "missing_label_and_ct"
                elif label_src is None:
                    status, reason = "missing", "missing_label"
                elif ct_src is None:
                    status, reason = "missing", "missing_ct"
                else:
                    label_dst = target_dir(temporal_data_root, group, roi, target_split, "labels") / label_src.name
                    ct_dst = target_dir(temporal_data_root, group, roi, target_split, "ct") / ct_src.name
                    materialize_file(label_src, label_dst, args.copy_files)
                    materialize_file(ct_src, ct_dst, args.copy_files)
                    counts[target_split] += 1
                audit_rows.append({
                    "roi": roi,
                    "data_group": group,
                    "patient_id": pid,
                    "source_month": month_by_pid.get(pid, "unknown"),
                    "target_split": target_split,
                    "status": status,
                    "reason": reason,
                    "label_source": str(label_src) if label_src else "",
                    "ct_source": str(ct_src) if ct_src else "",
                })
        count_rows.append({
            "roi": roi,
            "data_group": group,
            "training1": counts["training1"],
            "validation1": counts["validation1"],
            "testing1": counts["testing1"],
        })

    write_tsv(manifest_dir / "temporal_source_file_audit.tsv", audit_rows,
              ["roi", "data_group", "patient_id", "source_month", "target_split", "status", "reason", "label_source", "ct_source"])
    write_tsv(manifest_dir / "temporal_roi_split_counts.tsv", count_rows,
              ["roi", "data_group", "training1", "validation1", "testing1"])

    split_payload = {
        "split_config": str(split_cfg_path.resolve()),
        "source_data_root": str(source_root),
        "temporal_data_root": str(temporal_data_root),
        "reference_active_rois": reference_rois,
        "study_root": str(study_root),
        "seed": int(cfg.get("seed", 88)),
        "validation_fraction": float(cfg.get("validation_fraction", 0.20)),
        "validation_strategy": cfg.get("validation_strategy", "stratified_by_month"),
        "overlap_policy": cfg.get("overlap_policy", "month3_test_priority"),
        "split": split,
        "overlap_resolutions": overlaps,
    }
    split_payload["split_sha256"] = sha256_json(split_payload)
    (manifest_dir / "temporal_split_summary.json").write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Month-temporal NSR-QA split",
        "",
        "The downstream code intentionally continues to call these partitions `training1`, `validation1`, and `testing1` so the August-13 analysis/report format is unchanged.",
        "",
        f"- training1: **{len(split['training1'])}** patients from month1+month2",
        f"- validation1: **{len(split['validation1'])}** patients from month1+month2",
        f"- testing1: **{len(split['testing1'])}** patients = month3 temporal holdout",
        f"- duplicate policy: **{cfg.get('overlap_policy', 'month3_test_priority')}**",
        "",
        "## Cross-period duplicate IDs",
        "",
    ]
    if overlaps:
        lines += ["| patient | earlier period | resolution |", "|---:|---|---|"]
        lines += [f"| {r['patient_id']} | {r['earlier_month']} | {r['resolution']} |" for r in overlaps]
    else:
        lines.append("None.")
    lines += ["", "## ROI availability after remapping", "", "| ROI | train | validation | test |", "|---|---:|---:|---:|"]
    for r in count_rows:
        lines.append(f"| {r['roi']} | {r['training1']} | {r['validation1']} | {r['testing1']} |")
    (manifest_dir / "temporal_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Reuse the exact August-13 config generator so architecture, corruption,
    # loss, four model roles and report thresholds remain unchanged.
    prepare_script = project_root / "scripts" / "dino_shape" / "prepare_all_roi_nsrqa_study.py"
    cmd = [
        sys.executable, str(prepare_script),
        "--registry", str(filtered_registry_path),
        "--study-root", str(study_root),
        "--data-root", str(temporal_data_root),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=project_root, check=True)

    metadata = {
        "study_type": "month_temporal_holdout",
        "old_reference_study": "/scratch/shaohan/contour_QA/all_roi_NSRQA_study",
        "new_study_root": str(study_root),
        "temporal_data_root": str(temporal_data_root),
        "testing1_semantics": "month3 temporal holdout",
        "development_semantics": "month1 + month2; validation sampled separately within each month",
        "month3_duplicate_priority": [r["patient_id"] for r in overlaps],
        "preserved_model_and_analysis_workflow": True,
        "reference_active_rois": reference_rois,
    }
    (manifest_dir / "temporal_study_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("\nPrepared temporal study:")
    print(f"  study root: {study_root}")
    print(f"  data root:  {temporal_data_root}")
    print(f"  split:      train={len(split['training1'])} val={len(split['validation1'])} test={len(split['testing1'])}")
    print(f"  audit:      {manifest_dir / 'temporal_roi_split_counts.tsv'}")
    print("\nNext: inspect standardized training-corruption examples, then submit training.")


if __name__ == "__main__":
    main()
