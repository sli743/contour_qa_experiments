#!/usr/bin/env python3
"""Create a frozen patient-level split for the real-error QA study.

The split has two levels:
  1) a held-out QA test cohort that is never used to train segmentation or QA models;
  2) K OOF folds within the QA-development cohort for segmentation model training.

For the Lung_L pilot, eligibility defaults to requiring both the CT image and the
approved Lung_L contour. The output JSON is architecture-independent and should
be reused by nnU-Net, MONAI, and all downstream QA experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CT_RE = re.compile(r"^patient(.+)_image\.nii\.gz$")


def patient_sort_key(patient_id: str) -> Tuple[int, object]:
    """Natural-ish ordering that keeps numeric patient IDs in numeric order."""
    try:
        return (0, int(patient_id))
    except ValueError:
        return (1, patient_id)


def discover_ct_patients(images_dir: Path) -> Dict[str, Path]:
    patients: Dict[str, Path] = {}
    for path in sorted(images_dir.glob("patient*_image.nii.gz")):
        match = CT_RE.match(path.name)
        if match is None:
            continue
        patient_id = match.group(1)
        if patient_id in patients:
            raise RuntimeError(f"Duplicate CT for patient {patient_id}: {path}")
        patients[patient_id] = path
    return patients


def approved_roi_path(labels_approved_dir: Path, patient_id: str, roi: str) -> Path:
    return labels_approved_dir / f"patient{patient_id}_{roi}_approved.nii.gz"


def split_evenly(items: Sequence[str], n_folds: int) -> List[List[str]]:
    """Round-robin assignment after shuffling keeps fold sizes within one case."""
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for i, item in enumerate(items):
        folds[i % n_folds].append(item)
    return folds


def build_split(
    patient_ids: Sequence[str],
    seed: int,
    qa_test_fraction: float,
    n_folds: int,
) -> Tuple[List[str], List[str], List[List[str]]]:
    if not 0.0 < qa_test_fraction < 1.0:
        raise ValueError("qa_test_fraction must be strictly between 0 and 1")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(patient_ids) < n_folds + 1:
        raise ValueError(
            f"Need at least {n_folds + 1} eligible patients, got {len(patient_ids)}"
        )

    shuffled = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # Round rather than floor so a ~20% holdout stays close to the requested size.
    n_test = max(1, int(round(len(shuffled) * qa_test_fraction)))
    n_test = min(n_test, len(shuffled) - n_folds)

    qa_test = shuffled[:n_test]
    qa_development = shuffled[n_test:]
    fold_validations = split_evenly(qa_development, n_folds)

    if any(len(fold) == 0 for fold in fold_validations):
        raise RuntimeError("At least one OOF validation fold is empty")

    return qa_development, qa_test, fold_validations


def write_manifest_csv(
    output_csv: Path,
    all_ids: Iterable[str],
    qa_test: Sequence[str],
    fold_validations: Sequence[Sequence[str]],
    ct_paths: Dict[str, Path],
    labels_approved_dir: Path,
    roi: str,
) -> None:
    qa_test_set = set(qa_test)
    fold_by_patient = {
        patient_id: fold_idx
        for fold_idx, fold_ids in enumerate(fold_validations)
        for patient_id in fold_ids
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "cohort",
                "oof_fold",
                "ct_path",
                "approved_roi_path",
            ],
        )
        writer.writeheader()
        for patient_id in sorted(all_ids, key=patient_sort_key):
            is_test = patient_id in qa_test_set
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "cohort": "qa_test" if is_test else "qa_development",
                    "oof_fold": "" if is_test else fold_by_patient[patient_id],
                    "ct_path": str(ct_paths[patient_id]),
                    "approved_roi_path": str(
                        approved_roi_path(labels_approved_dir, patient_id, roi)
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create frozen QA-test + segmentation OOF patient splits."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Raw Radformation root containing images/ and labels-approved/.",
    )
    parser.add_argument("--roi", default="Lung_L")
    parser.add_argument("--qa-test-fraction", type=float, default=0.20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("configs/real_error_qa/patient_splits_lung_l.json"),
        help=(
            "Output split JSON. Default: "
            "configs/real_error_qa/patient_splits_lung_l.json"
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-roi",
        action="store_true",
        help=(
            "Split all CT patients even if the requested ROI is missing. "
            "For the Lung_L pilot the default is to require the approved ROI."
        ),
    )
    args = parser.parse_args()

    images_dir = args.data_root / "images"
    labels_approved_dir = args.data_root / "labels-approved"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_approved_dir.is_dir():
        raise FileNotFoundError(
            f"Missing approved-label directory: {labels_approved_dir}"
        )

    ct_paths = discover_ct_patients(images_dir)
    if not ct_paths:
        raise RuntimeError(f"No patient*_image.nii.gz files found in {images_dir}")

    missing_roi = [
        patient_id
        for patient_id in ct_paths
        if not approved_roi_path(labels_approved_dir, patient_id, args.roi).is_file()
    ]

    missing_roi_set = set(missing_roi)
    if args.allow_missing_roi:
        eligible_ids = list(ct_paths)
    else:
        eligible_ids = [pid for pid in ct_paths if pid not in missing_roi_set]

    eligible_ids = sorted(eligible_ids, key=patient_sort_key)
    qa_development, qa_test, fold_validations = build_split(
        eligible_ids,
        seed=args.seed,
        qa_test_fraction=args.qa_test_fraction,
        n_folds=args.n_folds,
    )

    dev_set = set(qa_development)
    folds = []
    for fold_idx, val_ids in enumerate(fold_validations):
        val_set = set(val_ids)
        train_ids = [pid for pid in qa_development if pid not in val_set]
        folds.append(
            {
                "fold": fold_idx,
                "train": sorted(train_ids, key=patient_sort_key),
                "val": sorted(val_ids, key=patient_sort_key),
            }
        )

    assert set(qa_test).isdisjoint(dev_set)
    assert set().union(*(set(f["val"]) for f in folds)) == dev_set

    payload = {
        "schema_version": 1,
        "study": "real_error_qa",
        "roi": args.roi,
        "data_root": str(args.data_root.resolve()),
        "seed": args.seed,
        "qa_test_fraction_requested": args.qa_test_fraction,
        "n_folds": args.n_folds,
        "n_ct_patients": len(ct_paths),
        "n_eligible_patients": len(eligible_ids),
        "n_qa_development": len(qa_development),
        "n_qa_test": len(qa_test),
        "missing_approved_roi": sorted(missing_roi, key=patient_sort_key),
        "qa_development": sorted(qa_development, key=patient_sort_key),
        "qa_test": sorted(qa_test, key=patient_sort_key),
        "folds": folds,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    output_csv = args.output_csv or args.output_json.with_suffix(".csv")
    write_manifest_csv(
        output_csv,
        eligible_ids,
        qa_test,
        fold_validations,
        ct_paths,
        labels_approved_dir,
        args.roi,
    )

    print(f"CT patients:       {len(ct_paths)}")
    print(f"Eligible patients: {len(eligible_ids)}")
    print(f"QA development:    {len(qa_development)}")
    print(f"QA test:           {len(qa_test)}")
    print(f"Missing {args.roi}:      {len(missing_roi)}")
    for fold in folds:
        print(
            f"Fold {fold['fold']}: train={len(fold['train'])}, "
            f"val={len(fold['val'])}"
        )
    print(f"Saved split JSON:  {args.output_json}")
    print(f"Saved manifest:    {output_csv}")


if __name__ == "__main__":
    main()
