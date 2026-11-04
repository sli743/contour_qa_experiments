#!/usr/bin/env python3
"""Collect nnU-Net fold validation outputs into one canonical OOF dataset.

Each QA-development patient must appear exactly once, in the validation output
of the fold specified by the frozen patient split. This script intentionally
fails on missing predictions, duplicates, or fold leakage.

Expected nnU-Net results layout (default trainer/plans/configuration):
  $nnUNet_results/
    Dataset501_RadformationLungLRealError/
      nnUNetTrainer__nnUNetPlans__3d_fullres/
        fold_0/validation/radf_p*.nii.gz
        ...
        fold_4/validation/radf_p*.nii.gz

The canonical output is kept outside nnU-Net so later QA code is not coupled to
nnU-Net's internal directory structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def case_id_for_patient(patient_id: str, case_mapping: Mapping[str, str]) -> str:
    try:
        return str(case_mapping[str(patient_id)])
    except KeyError as exc:
        raise KeyError(
            f"Patient {patient_id!r} is missing from real_error_dataset_metadata.json"
        ) from exc


def expected_fold_map(split: Mapping) -> Dict[str, int]:
    dev = {str(x) for x in split["qa_development"]}
    result: Dict[str, int] = {}

    for position, fold in enumerate(split["folds"]):
        fold_idx = int(fold.get("fold", position))
        for patient_id in fold["val"]:
            patient_id = str(patient_id)
            if patient_id in result:
                raise ValueError(
                    f"Patient {patient_id} appears in validation for both fold "
                    f"{result[patient_id]} and fold {fold_idx}"
                )
            result[patient_id] = fold_idx

    if set(result) != dev:
        missing = sorted(dev - set(result))
        extra = sorted(set(result) - dev)
        raise ValueError(
            "Frozen folds do not cover qa_development exactly: "
            f"missing={missing}, extra={extra}"
        )
    return result


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(mode)


def candidate_probability_paths(validation_dir: Path, case_id: str) -> Sequence[Path]:
    # nnU-Net v2 --npz convention is <case>.npz. Keep a second form for
    # compatibility with local/custom trainers that may retain .nii in the stem.
    return (
        validation_dir / f"{case_id}.npz",
        validation_dir / f"{case_id}.nii.gz.npz",
    )


def locate_probability(validation_dir: Path, case_id: str) -> Optional[Path]:
    for path in candidate_probability_paths(validation_dir, case_id):
        if path.is_file():
            return path
    return None


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect five nnU-Net validation folds into one OOF prediction set."
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("configs/real_error_qa/patient_splits_lung_l.json"),
    )
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="RadformationLungLRealError")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument(
        "--nnunet-results",
        type=Path,
        default=None,
        help="Defaults to $nnUNet_results.",
    )
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=None,
        help=(
            "Defaults to $nnUNet_raw. Used to read the exporter metadata that maps "
            "patient IDs to nnU-Net case IDs."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/shaohan/contour_QA/real_error_segqa"),
    )
    parser.add_argument(
        "--file-mode",
        choices=["copy", "symlink", "hardlink"],
        default="symlink",
    )
    parser.add_argument(
        "--require-probabilities",
        action="store_true",
        help="Fail if a --npz probability file is missing for any case.",
    )
    args = parser.parse_args()

    split = load_json(args.split_json)
    if str(split.get("roi")) != "Lung_L":
        raise ValueError(f"Expected Lung_L split, got roi={split.get('roi')!r}")

    nnunet_results_value = args.nnunet_results or os.environ.get("nnUNet_results")
    nnunet_raw_value = args.nnunet_raw or os.environ.get("nnUNet_raw")
    if nnunet_results_value is None:
        raise RuntimeError("Set $nnUNet_results or pass --nnunet-results")
    if nnunet_raw_value is None:
        raise RuntimeError("Set $nnUNet_raw or pass --nnunet-raw")

    dataset = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw_dataset_dir = Path(nnunet_raw_value) / dataset
    metadata_path = raw_dataset_dir / "real_error_dataset_metadata.json"
    metadata = load_json(metadata_path)
    case_mapping = metadata.get("case_id_mapping", {})

    experiment = f"{args.trainer}__{args.plans}__{args.configuration}"
    result_root = Path(nnunet_results_value) / dataset / experiment
    if not result_root.is_dir():
        raise FileNotFoundError(
            f"nnU-Net experiment directory not found: {result_root}\n"
            "Train the five folds first, or override --trainer/--plans/--configuration."
        )

    fold_by_patient = expected_fold_map(split)
    output_dir = (
        args.output_root
        / "predictions"
        / "oof"
        / "nnunet"
        / "Lung_L"
        / args.configuration
    )
    hard_dir = output_dir / "segmentations"
    prob_dir = output_dir / "probabilities"
    reset_dir(hard_dir)
    reset_dir(prob_dir)

    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    missing_probabilities: List[str] = []
    seen_case_ids = set()

    # Iterate over the frozen manifest, not whatever happens to be present in
    # nnU-Net's validation folders. This prevents accidental cross-fold mixing.
    for patient_id in [str(x) for x in split["qa_development"]]:
        fold = fold_by_patient[patient_id]
        case_id = case_id_for_patient(patient_id, case_mapping)
        validation_dir = result_root / f"fold_{fold}" / "validation"
        pred = validation_dir / f"{case_id}.nii.gz"

        if not pred.is_file():
            missing.append(f"patient={patient_id} fold={fold} expected={pred}")
            continue
        if case_id in seen_case_ids:
            raise RuntimeError(f"Duplicate OOF case encountered: {case_id}")
        seen_case_ids.add(case_id)

        canonical_pred = hard_dir / f"{case_id}.nii.gz"
        copy_or_link(pred, canonical_pred, args.file_mode)

        probability = locate_probability(validation_dir, case_id)
        canonical_probability: Optional[Path] = None
        if probability is not None:
            canonical_probability = prob_dir / f"{case_id}.npz"
            copy_or_link(probability, canonical_probability, args.file_mode)
        else:
            missing_probabilities.append(
                f"patient={patient_id} fold={fold} validation={validation_dir}"
            )

        rows.append(
            {
                "patient_id": patient_id,
                "roi": "Lung_L",
                "source_model": "nnunet",
                "source_configuration": args.configuration,
                "source_fold": fold,
                "case_id": case_id,
                "prediction_path": str(canonical_pred.resolve()),
                "probability_path": (
                    str(canonical_probability.resolve())
                    if canonical_probability is not None
                    else ""
                ),
                "nnunet_validation_dir": str(validation_dir.resolve()),
            }
        )

    if missing:
        details = "\n".join(f"  - {x}" for x in missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise RuntimeError(
            f"Missing {len(missing)} expected OOF predictions.\n{details}{more}\n"
            "No manifest was finalized. Check that all five folds completed validation."
        )

    if len(rows) != len(split["qa_development"]):
        raise RuntimeError(
            f"Expected {len(split['qa_development'])} OOF rows, collected {len(rows)}"
        )

    if args.require_probabilities and missing_probabilities:
        details = "\n".join(f"  - {x}" for x in missing_probabilities[:20])
        raise RuntimeError(
            f"Missing {len(missing_probabilities)} probability files despite "
            f"--require-probabilities.\n{details}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "oof_predictions.csv"
    fieldnames = [
        "patient_id",
        "roi",
        "source_model",
        "source_configuration",
        "source_fold",
        "case_id",
        "prediction_path",
        "probability_path",
        "nnunet_validation_dir",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": dataset,
        "roi": "Lung_L",
        "configuration": args.configuration,
        "trainer": args.trainer,
        "plans": args.plans,
        "n_oof_predictions": len(rows),
        "n_probability_files": len(rows) - len(missing_probabilities),
        "n_missing_probability_files": len(missing_probabilities),
        "source_split_json": str(args.split_json.resolve()),
        "nnunet_result_root": str(result_root.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    with (output_dir / "collection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Dataset:                 {dataset}")
    print(f"Configuration:           {args.configuration}")
    print(f"OOF predictions:         {len(rows)}")
    print(f"Probability files:       {len(rows) - len(missing_probabilities)}")
    print(f"Missing probabilities:   {len(missing_probabilities)}")
    print(f"Canonical OOF dir:       {output_dir}")
    print(f"OOF manifest:            {manifest_path}")
    if missing_probabilities:
        print(
            "WARNING: some probability files were not found. Hard segmentations are "
            "complete; rerun training/validation with --npz if probabilities are needed."
        )


if __name__ == "__main__":
    main()
