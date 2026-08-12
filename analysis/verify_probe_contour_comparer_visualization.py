#!/usr/bin/env python3
"""Verify the flat Probe Contour Comparer visualization file layout."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def level_token(level: float) -> str:
    x = float(level)
    if x.is_integer():
        return str(int(x))
    return f"{x:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--visualization-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    vis = args.visualization_dir.expanduser().resolve()
    missing: list[str] = []

    for pid in sorted(manifest["patient_id"].unique()):
        for name in (
            f"clean_input_patient_{pid}.nii.gz",
            f"original_ct_patient_{pid}.nii.gz",
        ):
            if not (vis / name).is_file():
                missing.append(name)

    for _, r in manifest.iterrows():
        pid = str(r["patient_id"])
        token = level_token(float(r["level"]))
        for name in (
            f"corrupted_input_{r['probe_type']}_level_{token}_patient_{pid}.nii.gz",
            f"reconstruction_{r['probe_type']}_level_{token}_patient_{pid}.nii.gz",
        ):
            if not (vis / name).is_file():
                missing.append(name)

    if missing:
        preview = "\n".join(f"  - {x}" for x in missing[:30])
        more = "" if len(missing) <= 30 else f"\n  ... and {len(missing) - 30} more"
        raise RuntimeError(
            f"Probe Contour Comparer visualization is incomplete; {len(missing)} files missing:\n"
            f"{preview}{more}"
        )

    print(
        f"Probe Contour Comparer layout OK: {manifest['patient_id'].nunique()} patients, "
        f"{len(manifest)} probe rows, CT + clean + corrupted + reconstruction files present."
    )


if __name__ == "__main__":
    main()
