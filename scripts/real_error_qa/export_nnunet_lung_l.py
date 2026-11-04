#!/usr/bin/env python3
"""Export the frozen RealSeg-QA Lung_L pilot to nnU-Net v2 format.

The QA-development cohort becomes nnU-Net training data (imagesTr/labelsTr).
The fully held-out QA-test cohort is exported to imagesTs, while its approved
masks are stored separately under qa_test_gt/ so they are available for final
study evaluation but are not part of nnU-Net's training labels.

The script also writes an nnU-Net-compatible splits_final.json containing the
exact OOF folds from build_patient_splits.py. Use --install-preprocessed-split
after nnUNetv2_plan_and_preprocess to place this file where nnU-Net reads it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import SimpleITK as sitk


SAFE_CASE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def load_split(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Split JSON not found: {path}\n"
            "Create it first with:\n"
            "  python scripts/real_error_qa/build_patient_splits.py "
            "--data-root ../radformation-data/data --roi Lung_L\n"
            "The split script creates configs/real_error_qa automatically."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    required = {"roi", "qa_development", "qa_test", "folds"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Split JSON is missing keys: {sorted(missing)}")
    return payload


def sanitize_patient_id(patient_id: str) -> str:
    safe = SAFE_CASE_RE.sub("-", str(patient_id)).strip("-")
    if not safe:
        raise ValueError(f"Patient ID cannot be converted to a safe case ID: {patient_id!r}")
    return safe


def nnunet_case_id(patient_id: str) -> str:
    return f"radf_p{sanitize_patient_id(patient_id)}"


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0.0, atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0.0, atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0.0, atol=tol)
    )


def binary_mask_in_ct_geometry(mask_path: Path, ct_img: sitk.Image) -> Tuple[sitk.Image, bool]:
    mask_img = sitk.ReadImage(str(mask_path))
    resampled = False
    if not same_geometry(mask_img, ct_img):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        mask_img = resampler.Execute(mask_img)
        resampled = True

    mask_arr = sitk.GetArrayFromImage(mask_img)
    binary_arr = (mask_arr > 0).astype(np.uint8)
    if binary_arr.sum() == 0:
        raise ValueError(f"Approved mask is empty after geometry normalization: {mask_path}")

    out = sitk.GetImageFromArray(binary_arr)
    out.CopyInformation(ct_img)
    return out, resampled


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown file mode: {mode}")


def expected_paths(data_root: Path, patient_id: str, roi: str) -> Tuple[Path, Path]:
    ct = data_root / "images" / f"patient{patient_id}_image.nii.gz"
    gt = data_root / "labels-approved" / f"patient{patient_id}_{roi}_approved.nii.gz"
    return ct, gt


def write_dataset_json(dataset_dir: Path, n_training: int, roi: str) -> None:
    # Using channel name "CT" activates nnU-Net's CT normalization scheme.
    payload = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, roi: 1},
        "numTraining": n_training,
        "file_ending": ".nii.gz",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_nnunet_splits(split: Mapping) -> List[Dict[str, List[str]]]:
    dev = set(str(x) for x in split["qa_development"])
    result = []
    seen_val = set()

    for expected_fold, fold in enumerate(split["folds"]):
        fold_idx = int(fold.get("fold", expected_fold))
        if fold_idx != expected_fold:
            raise ValueError(
                f"Expected folds in order 0..K-1; found fold={fold_idx} at index {expected_fold}"
            )
        train_ids = [str(x) for x in fold["train"]]
        val_ids = [str(x) for x in fold["val"]]
        if set(train_ids) & set(val_ids):
            raise ValueError(f"Fold {fold_idx} has patient overlap between train and val")
        if set(train_ids) | set(val_ids) != dev:
            raise ValueError(f"Fold {fold_idx} does not partition qa_development exactly")
        seen_val.update(val_ids)
        result.append(
            {
                "train": [nnunet_case_id(pid) for pid in train_ids],
                "val": [nnunet_case_id(pid) for pid in val_ids],
            }
        )

    if seen_val != dev:
        raise ValueError("OOF validation folds do not cover qa_development exactly")
    return result


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def export_case(
    patient_id: str,
    cohort: str,
    data_root: Path,
    dataset_dir: Path,
    roi: str,
    file_mode: str,
) -> Dict[str, object]:
    case_id = nnunet_case_id(patient_id)
    ct_path, gt_path = expected_paths(data_root, patient_id, roi)
    if not ct_path.is_file():
        raise FileNotFoundError(f"Missing CT: {ct_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing approved {roi} contour: {gt_path}")

    if cohort == "qa_development":
        image_out = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        label_out = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
    elif cohort == "qa_test":
        image_out = dataset_dir / "imagesTs" / f"{case_id}_0000.nii.gz"
        label_out = dataset_dir / "qa_test_gt" / f"{case_id}.nii.gz"
    else:
        raise ValueError(f"Unknown cohort: {cohort}")

    link_or_copy(ct_path, image_out, file_mode)
    ct_img = sitk.ReadImage(str(ct_path))
    binary_gt, was_resampled = binary_mask_in_ct_geometry(gt_path, ct_img)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(binary_gt, str(label_out), useCompression=True)

    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "cohort": cohort,
        "ct_source": str(ct_path.resolve()),
        "approved_source": str(gt_path.resolve()),
        "nnunet_image": str(image_out.resolve()),
        "nnunet_label_or_test_gt": str(label_out.resolve()),
        "label_resampled_to_ct": was_resampled,
        "foreground_voxels": int(sitk.GetArrayViewFromImage(binary_gt).sum()),
    }


def write_export_manifest(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    fieldnames = [
        "patient_id",
        "case_id",
        "cohort",
        "ct_source",
        "approved_source",
        "nnunet_image",
        "nnunet_label_or_test_gt",
        "label_resampled_to_ct",
        "foreground_voxels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reset_output_dirs(dataset_dir: Path) -> None:
    for name in ["imagesTr", "labelsTr", "imagesTs", "qa_test_gt"]:
        path = dataset_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the RealSeg-QA Lung_L pilot to nnU-Net v2 format."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("configs/real_error_qa/patient_splits_lung_l.json"),
        help=(
            "Frozen patient split JSON. Default: "
            "configs/real_error_qa/patient_splits_lung_l.json"
        ),
    )
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=None,
        help="nnUNet_raw root. Defaults to the nnUNet_raw environment variable.",
    )
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", default="RadformationLungLRealError")
    parser.add_argument(
        "--file-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place CT images in nnUNet_raw. Labels are always rewritten as binary masks.",
    )
    parser.add_argument(
        "--install-preprocessed-split",
        action="store_true",
        help=(
            "Also install splits_final.json into $nnUNet_preprocessed/DatasetXXX_Name. "
            "Run after nnUNetv2_plan_and_preprocess (or rerun afterward if preprocessing overwrites it)."
        ),
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=None,
        help="Override nnUNet_preprocessed when --install-preprocessed-split is used.",
    )
    args = parser.parse_args()

    split = load_split(args.split_json)
    roi = str(split["roi"])
    if roi != "Lung_L":
        raise ValueError(
            f"This pilot exporter is intentionally Lung_L-specific; split roi={roi!r}"
        )

    nnunet_raw_value = args.nnunet_raw or os.environ.get("nnUNet_raw")
    if nnunet_raw_value is None:
        raise RuntimeError(
            "nnUNet_raw is not set. Pass --nnunet-raw or export nnUNet_raw=/path/to/nnUNet_raw"
        )
    nnunet_raw = Path(nnunet_raw_value)

    if not 0 <= args.dataset_id <= 999:
        raise ValueError("dataset-id must be between 0 and 999")
    dataset_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = nnunet_raw / dataset_folder_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    reset_output_dirs(dataset_dir)

    dev_ids = [str(x) for x in split["qa_development"]]
    test_ids = [str(x) for x in split["qa_test"]]
    if set(dev_ids) & set(test_ids):
        raise ValueError("qa_development and qa_test overlap")

    rows = []
    for patient_id in dev_ids:
        rows.append(
            export_case(
                patient_id,
                "qa_development",
                args.data_root,
                dataset_dir,
                roi,
                args.file_mode,
            )
        )
    for patient_id in test_ids:
        rows.append(
            export_case(
                patient_id,
                "qa_test",
                args.data_root,
                dataset_dir,
                roi,
                args.file_mode,
            )
        )

    write_dataset_json(dataset_dir, n_training=len(dev_ids), roi=roi)
    write_export_manifest(dataset_dir / "real_error_case_manifest.csv", rows)

    nnunet_splits = build_nnunet_splits(split)
    raw_split_path = dataset_dir / "splits_final.json"
    save_json(raw_split_path, nnunet_splits)

    mapping = {
        "dataset": dataset_folder_name,
        "roi": roi,
        "source_split_json": str(args.split_json.resolve()),
        "n_training": len(dev_ids),
        "n_qa_test": len(test_ids),
        "case_id_mapping": {
            str(pid): nnunet_case_id(str(pid)) for pid in dev_ids + test_ids
        },
        "note": (
            "qa_test_gt is intentionally outside labelsTr. It is study ground truth "
            "for final evaluation and must not be used for segmentation or QA training."
        ),
    }
    save_json(dataset_dir / "real_error_dataset_metadata.json", mapping)

    installed_path = None
    if args.install_preprocessed_split:
        preprocessed_value = args.nnunet_preprocessed or os.environ.get("nnUNet_preprocessed")
        if preprocessed_value is None:
            raise RuntimeError(
                "nnUNet_preprocessed is not set. Pass --nnunet-preprocessed or export it."
            )
        preprocessed_dataset = Path(preprocessed_value) / dataset_folder_name
        if not preprocessed_dataset.is_dir():
            raise FileNotFoundError(
                f"Preprocessed dataset does not exist yet: {preprocessed_dataset}\n"
                "Run nnUNetv2_plan_and_preprocess first, then rerun this exporter with "
                "--install-preprocessed-split."
            )
        installed_path = preprocessed_dataset / "splits_final.json"
        shutil.copy2(raw_split_path, installed_path)

    print(f"Dataset:             {dataset_folder_name}")
    print(f"nnUNet raw dataset:  {dataset_dir}")
    print(f"Training cases:      {len(dev_ids)}")
    print(f"Held-out QA cases:   {len(test_ids)}")
    print(f"OOF folds:           {len(nnunet_splits)}")
    print(f"Raw split copy:      {raw_split_path}")
    if installed_path is not None:
        print(f"Installed split:     {installed_path}")
    else:
        print(
            "Next: run nnUNetv2_plan_and_preprocess, then rerun with "
            "--install-preprocessed-split before training."
        )


if __name__ == "__main__":
    main()
