#!/usr/bin/env python3
"""
Convert patient-organized Bronchus probe visualization exports into the flat
filename layout expected by ProbeContourComparer.py.

Expected source layout (one common form):

    SOURCE_ROOT/
      patient_101/
        ct.nii.gz
        ground_truth.nii.gz
        metrics.csv                         # optional
        cut_upper_level_0.5/
          corrupted_input.nii.gz
          reconstruction.nii.gz
        erode_level_2/
          corrupted_input.nii.gz
          reconstruction.nii.gz

Generated flat layout:

    OUTPUT_ROOT/
      original_ct_patient_101.nii.gz
      clean_input_patient_101.nii.gz
      metrics_patient_101.csv               # optional
      corrupted_input_cut_upper_level_0.5_patient_101.nii.gz
      reconstruction_cut_upper_level_0.5_patient_101.nii.gz
      ...

The script accepts multiple experiment roots, so all three Bronchus experiments
can be converted in one command. By default, files are hard-linked, which is
fast and does not duplicate image data. If hard-linking is unavailable across
filesystems, the script falls back to copying unless --no-copy-fallback is used.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable


PATIENT_RE = re.compile(r"^patient[_-]?(?P<id>.+)$", re.IGNORECASE)
NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(NIFTI_SUFFIXES)


def strip_nifti_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def sanitize_token(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def patient_id_from_dir(path: Path) -> str | None:
    match = PATIENT_RE.match(path.name)
    return sanitize_token(match.group("id")) if match else None


def find_patient_dirs(root: Path) -> list[Path]:
    direct = [p for p in root.iterdir() if p.is_dir() and patient_id_from_dir(p)]
    if direct:
        return sorted(direct)

    # Support an extra wrapper directory, e.g. root/probe_visualization_test/patient_101.
    found = [
        p
        for p in root.rglob("*")
        if p.is_dir() and patient_id_from_dir(p)
    ]
    return sorted(found)


def identify_clean_file(patient_dir: Path) -> Path | None:
    preferred = (
        "ground_truth.nii.gz",
        "ground_truth.nii",
        "clean_input.nii.gz",
        "clean_input.nii",
        "approved.nii.gz",
        "approved.nii",
        "target.nii.gz",
        "target.nii",
    )
    for name in preferred:
        candidate = patient_dir / name
        if candidate.is_file():
            return candidate
    return None


def identify_ct_file(patient_dir: Path) -> Path | None:
    preferred = (
        "ct.nii.gz",
        "ct.nii",
        "original_ct.nii.gz",
        "original_ct.nii",
        "image.nii.gz",
        "image.nii",
    )
    for name in preferred:
        candidate = patient_dir / name
        if candidate.is_file():
            return candidate
    return None


def infer_variant(relative_parent: Path, source_file: Path) -> str:
    """
    Prefer the containing variant folder. If the file is directly under the
    patient folder, infer the variant from its filename.
    """
    if relative_parent.parts:
        return sanitize_token("_".join(relative_parent.parts))

    stem = strip_nifti_suffix(source_file.name)
    stem = re.sub(
        r"^(corrupted_input|reconstruction|recon|prediction|pred)[_-]?",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    return sanitize_token(stem)


def destination_name(source: Path, patient_id: str, patient_dir: Path) -> str | None:
    lower = source.name.lower()

    if source == identify_clean_file(patient_dir):
        return f"clean_input_patient_{patient_id}.nii.gz"

    if source == identify_ct_file(patient_dir):
        return f"original_ct_patient_{patient_id}.nii.gz"

    if lower == "metrics.csv":
        return f"metrics_patient_{patient_id}.csv"

    if not is_nifti(source):
        return None

    relative_parent = source.parent.relative_to(patient_dir)
    variant = infer_variant(relative_parent, source)
    if not variant:
        return None

    stem = strip_nifti_suffix(source.name).lower()

    if stem.startswith("corrupted_input") or stem in {"corrupted", "input"}:
        prefix = "corrupted_input"
    elif (
        stem.startswith("reconstruction")
        or stem.startswith("recon")
        or stem.startswith("prediction")
        or stem.startswith("pred")
    ):
        prefix = "reconstruction"
    else:
        return None

    # If a direct filename already includes the variant, infer_variant() uses it.
    return f"{prefix}_{variant}_patient_{patient_id}.nii.gz"


def transfer(
    source: Path,
    destination: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    copy_fallback: bool,
) -> str:
    if destination.exists():
        try:
            same = os.path.samefile(source, destination)
        except OSError:
            same = False
        if same:
            return "already-linked"
        if not overwrite:
            raise FileExistsError(f"Destination exists: {destination}")
        if not dry_run:
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

    if dry_run:
        return f"would-{mode}"

    destination.parent.mkdir(parents=True, exist_ok=True)

    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlinked"
        except OSError:
            if not copy_fallback:
                raise
            shutil.copy2(source, destination)
            return "copied-fallback"

    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return "symlinked"

    if mode == "copy":
        shutil.copy2(source, destination)
        return "copied"

    if mode == "move":
        shutil.move(str(source), str(destination))
        return "moved"

    raise ValueError(f"Unsupported mode: {mode}")


def convert_root(
    root: Path,
    output: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    copy_fallback: bool,
) -> tuple[int, int]:
    root = root.resolve()
    output = output.resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Experiment root does not exist: {root}")

    patient_dirs = find_patient_dirs(root)
    if not patient_dirs:
        raise RuntimeError(
            f"No patient_<id> directories found under {root}. "
            "Point the script to the patient-organized visualization directory."
        )

    print(f"\nExperiment: {root}")
    print(f"Output:     {output}")
    print(f"Patients:   {len(patient_dirs)}")

    written = 0
    skipped = 0

    for patient_dir in patient_dirs:
        patient_id = patient_id_from_dir(patient_dir)
        assert patient_id is not None

        candidates: list[Path] = []
        candidates.extend(p for p in patient_dir.iterdir() if p.is_file())
        candidates.extend(p for p in patient_dir.rglob("*") if p.is_file())

        # Preserve order but remove duplicates.
        unique_candidates = list(dict.fromkeys(candidates))

        patient_written = 0
        for source in unique_candidates:
            name = destination_name(source, patient_id, patient_dir)
            if name is None:
                skipped += 1
                continue

            destination = output / name
            result = transfer(
                source,
                destination,
                mode=mode,
                overwrite=overwrite,
                dry_run=dry_run,
                copy_fallback=copy_fallback,
            )
            print(f"  [{result:15}] {source.relative_to(root)} -> {name}")
            written += 1
            patient_written += 1

        if patient_written == 0:
            print(f"  [warning] no compatible files found for {patient_dir}")

    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten one or more patient-organized Bronchus visualization "
            "directories for ProbeContourComparer.py."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Patient-organized visualization directories for the experiments.",
    )
    parser.add_argument(
        "--output-name",
        default="probe_visualization_test_flat",
        help=(
            "Name of the flat output directory created beside each input root "
            "(default: probe_visualization_test_flat)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Explicit output directory. Valid only when one input root is given."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink", "move"),
        default="hardlink",
        help=(
            "File operation. hardlink is fastest and preserves the originals "
            "(default: hardlink)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without creating files.",
    )
    parser.add_argument(
        "--no-copy-fallback",
        action="store_true",
        help="Do not fall back to copying when a hard link cannot be created.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.output is not None and len(args.roots) != 1:
        print("--output can only be used with one input root.", file=sys.stderr)
        return 2

    total_written = 0
    total_skipped = 0

    try:
        for root in args.roots:
            if args.output is not None:
                output = args.output
            else:
                # Put the new flat directory beside the source visualization dir.
                output = root.resolve().parent / args.output_name

                # Multiple source roots can share the same parent. Avoid collision
                # by incorporating the source directory name when needed.
                if len(args.roots) > 1:
                    output = (
                        root.resolve().parent
                        / f"{root.resolve().name}_flat"
                    )

            written, skipped = convert_root(
                root=root,
                output=output,
                mode=args.mode,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                copy_fallback=not args.no_copy_fallback,
            )
            total_written += written
            total_skipped += skipped

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nDone. Compatible files: {total_written}; "
        f"unrecognized files skipped: {total_skipped}."
    )
    if args.dry_run:
        print("Dry run only; no files were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
