#!/usr/bin/env python3
"""Export Lung_L NSR-QA synthetic contours for visual inspection.

This is a post-hoc inspection utility. It does not refit corruption severity and
it does not modify training. It loads the family-validated fitted training
configuration (or, optionally, the best unvalidated candidate) and writes:

1. a branch-forced low/typical/high bank compatible with the existing
   TrainingCorruptionExampleViewer; and
2. a true-mixture bank sampled exactly through the training corruption entry
   point, including the intentional clean branch.

Every exported sample includes the clean target, corrupted input, changed,
missing, and excess masks plus realized geometric metrics and sampler metadata.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

from analysis.calibrate_lung_l_nsrqa_severity import (
    build_patient_index,
    crop_union,
    export_example_case,
    image_array,
    pair_metrics_plus,
    resample_to_reference,
    to_image,
)
from analysis.fit_lung_l_nsrqa_synthetic_severity_family_validated import (
    fitted_severity_strata,
)
from training.corruptions.stochastic_region_dropout import (
    stochastic_region_dropout_corruption,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/dino_shape/lung_l_nsrqa_synthetic_inspection_bank.yaml",
    )
    return p.parse_args()


def resolve_path(value: str | Path, cwd: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (cwd / p).resolve()


def choose_training_config(cfg: Mapping, cwd: Path) -> tuple[Dict, Path, bool]:
    root = resolve_path(cfg["calibration_output_dir"], cwd)
    preferred = root / str(cfg.get("validated_config_name", "best_fit_training_config.yaml"))
    if preferred.is_file():
        return yaml.safe_load(preferred.read_text()), preferred, True

    candidate = root / str(cfg.get("candidate_config_name", "best_candidate_training_config.yaml"))
    if candidate.is_file() and bool(cfg.get("allow_unvalidated_candidate_for_inspection", True)):
        print("WARNING: validated best_fit_training_config.yaml was not found.")
        print(f"Using UNVALIDATED candidate for inspection only: {candidate}")
        return yaml.safe_load(candidate.read_text()), candidate, False

    raise FileNotFoundError(
        "No fitted training config available. Expected either "
        f"{preferred} or {candidate}. Run the family-validated calibration first."
    )


def choose_patients(
    case_df: pd.DataFrame,
    available: Sequence[int],
    explicit: Sequence[int],
    n_patients: int,
) -> List[int]:
    avail = sorted({int(x) for x in available})
    avail_set = set(avail)
    if explicit:
        chosen = [int(x) for x in explicit if int(x) in avail_set]
        missing = [int(x) for x in explicit if int(x) not in avail_set]
        if missing:
            print(f"WARNING: requested patient IDs unavailable: {missing}")
        if not chosen:
            raise ValueError("None of selection.patient_ids are available")
        return chosen

    n_patients = min(max(1, int(n_patients)), len(avail))
    x = case_df.copy()
    if "patient_id" not in x:
        return avail[:n_patients]
    x["patient_id"] = pd.to_numeric(x["patient_id"], errors="coerce")
    x = x[np.isfinite(x["patient_id"])].copy()
    x["patient_id"] = x["patient_id"].astype(int)
    x = x[x["patient_id"].isin(avail_set)].copy()

    volume_col = next(
        (c for c in ("approved_volume_cc", "approved_volume_ml", "approved_volume") if c in x.columns),
        None,
    )
    if volume_col is None:
        return x["patient_id"].tolist()[:n_patients] if len(x) else avail[:n_patients]

    x[volume_col] = pd.to_numeric(x[volume_col], errors="coerce")
    x = x[np.isfinite(x[volume_col])].sort_values(volume_col).copy()
    if not len(x):
        return avail[:n_patients]

    probs = np.linspace(0.05, 0.95, n_patients)
    vals = x[volume_col].to_numpy(dtype=float)
    chosen: List[int] = []
    for q in probs:
        target = float(np.quantile(vals, q))
        for idx in np.argsort(np.abs(vals - target)):
            pid = int(x.iloc[int(idx)]["patient_id"])
            if pid not in chosen:
                chosen.append(pid)
                break
    return chosen


def write_mixture_sample(
    pid: int,
    label_path: Path,
    ct_path: Path,
    training_cfg: Mapping,
    outdir: Path,
    sample_index: int,
    seed: int,
    crop_margin_mm: float,
) -> Dict:
    ref, clean = image_array(label_path)
    ct = resample_to_reference(sitk.ReadImage(str(ct_path), sitk.sitkFloat32), ref, nearest=False)
    spacing_zyx = tuple(float(v) for v in ref.GetSpacing()[::-1])

    x = torch.from_numpy(clean.astype(np.float32, copy=False))[None]
    corr_t, changed_t, meta = stochastic_region_dropout_corruption(
        x,
        training_cfg,
        random.Random(int(seed)),
        cache_key=f"inspection-mixture-{pid}-{sample_index}",
    )
    corrupted = corr_t.detach().cpu().numpy()[0] > 0.5
    changed = changed_t.detach().cpu().numpy()[0] > 0.5
    missing = clean & ~corrupted
    excess = corrupted & ~clean
    metrics = pair_metrics_plus(clean, corrupted, spacing_zyx)

    image_map = {
        "ct": ct,
        "clean": to_image(clean, ref),
        "corrupted": to_image(corrupted, ref),
        "changed": to_image(changed, ref),
        "missing": to_image(missing, ref),
        "excess": to_image(excess, ref),
    }
    image_map = crop_union(image_map, [clean, corrupted], crop_margin_mm)
    outdir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "ct": "ct.nii.gz",
        "clean": "clean_target.nii.gz",
        "corrupted": "corrupted_input.nii.gz",
        "changed": "changed_region.nii.gz",
        "missing": "missing_region.nii.gz",
        "excess": "excess_region.nii.gz",
    }
    for key, name in filenames.items():
        sitk.WriteImage(image_map[key], str(outdir / name), True)

    row = {
        "patient_id": int(pid),
        "sample_index": int(sample_index),
        "seed": int(seed),
        "branch": str(meta.get("branch", "unknown")),
        "is_noop": int(not np.any(clean != corrupted)),
        **meta,
        **metrics,
    }
    (outdir / "metadata.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def write_report(
    outdir: Path,
    config_path: Path,
    validated: bool,
    patients: Sequence[int],
    forced_manifest: pd.DataFrame,
    mixture_manifest: pd.DataFrame,
) -> None:
    lines = [
        "# Lung_L NSR-QA synthetic inspection bank",
        "",
        f"Training config: `{config_path}`",
        f"Family-validity passed config: **{'yes' if validated else 'no — candidate used only for inspection'}**",
        f"Patients: `{list(map(int, patients))}`",
        "",
        "## Banks",
        "",
        "- `branch_forced_bank/`: low / typical / high forced examples for every NSR-QA corruption family; compatible with the existing TrainingCorruptionExampleViewer.",
        "- `mixture_bank/`: direct draws through the actual training corruption entry point, including the intentional clean branch.",
        "",
        "## Realized mixture summary",
        "",
    ]
    if len(mixture_manifest):
        counts = mixture_manifest["branch"].value_counts(dropna=False)
        for branch, n in counts.items():
            lines.append(f"- `{branch}`: {int(n)}")
        lines += [
            f"- no-op rate: {float(mixture_manifest['is_noop'].mean()):.3f}",
            f"- median 1-Dice: {float(mixture_manifest['one_minus_dice'].median()):.6f}",
            f"- median HD95: {float(mixture_manifest['hd95_mm'].median()):.3f} mm",
        ]
    lines += [
        "",
        f"Forced examples exported: {len(forced_manifest)}",
        f"Mixture examples exported: {len(mixture_manifest)}",
        "",
    ]
    (outdir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    cfg_path = resolve_path(args.config, cwd)
    cfg = yaml.safe_load(cfg_path.read_text())

    training_cfg, training_cfg_path, validated = choose_training_config(cfg, cwd)
    label_root = resolve_path(cfg["data"]["train_label_dir"], cwd)
    ct_root = resolve_path(cfg["data"]["train_ct_dir"], cwd)
    error_root = resolve_path(cfg["error_field_root"], cwd)
    outdir = resolve_path(cfg["output"]["directory"], cwd)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = build_patient_index(label_root)
    cts = build_patient_index(ct_root)
    available = sorted(set(labels) & set(cts))
    if not available:
        raise RuntimeError(f"No common patient IDs under {label_root} and {ct_root}")

    case_csv = error_root / "case_error_field_summary.csv"
    case_df = pd.read_csv(case_csv) if case_csv.is_file() else pd.DataFrame({"patient_id": available})
    selection = cfg.get("selection", {})
    patients = choose_patients(
        case_df,
        available,
        [int(x) for x in selection.get("patient_ids", [])],
        int(selection.get("n_patients", 5)),
    )
    print(f"Inspection patients: {patients}")
    print(f"Using fitted config: {training_cfg_path}")

    # 1) Branch-forced bank in the pre-existing viewer-compatible layout.
    fitted_sev = fitted_severity_strata(training_cfg)
    (outdir / "fitted_severity_strata.json").write_text(json.dumps(fitted_sev, indent=2))
    forced_root = outdir / "branch_forced_bank"
    forced_rows: List[Dict] = []
    strata = [str(x) for x in cfg.get("forced_bank", {}).get("severity_strata", ["low", "typical", "high"])]
    base_seed = int(cfg.get("seed", 91))
    margin = float(cfg.get("crop_margin_mm", 20.0))
    for si, stratum in enumerate(strata):
        for pi, pid in enumerate(patients):
            rows = export_example_case(
                int(pid),
                labels[int(pid)],
                cts[int(pid)],
                forced_root / f"Lung_L_{stratum}" / f"patient_{pid}",
                training_cfg,
                fitted_sev,
                stratum,
                base_seed + si * 10000019 + pi * 100003,
                margin,
            )
            forced_rows.extend(rows)
            print(f"exported forced {stratum}: patient {pid}")
    forced_manifest = pd.DataFrame(forced_rows)
    forced_manifest.to_csv(outdir / "branch_forced_manifest.csv", index=False)

    # 2) Exact training-mixture draws. These show what the DataLoader sees.
    mixture_root = outdir / "mixture_bank"
    mix_rows: List[Dict] = []
    n_mix = int(cfg.get("mixture_bank", {}).get("samples_per_patient", 8))
    for pi, pid in enumerate(patients):
        for j in range(n_mix):
            seed = base_seed + 500000003 + pi * 1000003 + j * 10007
            row = write_mixture_sample(
                int(pid), labels[int(pid)], cts[int(pid)], training_cfg,
                mixture_root / f"patient_{pid}" / f"sample_{j:03d}",
                j, seed, margin,
            )
            mix_rows.append(row)
            print(
                f"exported mixture patient {pid} sample {j:03d}: "
                f"branch={row.get('branch')} noop={row.get('is_noop')}"
            )
    mixture_manifest = pd.DataFrame(mix_rows)
    mixture_manifest.to_csv(outdir / "mixture_manifest.csv", index=False)

    write_report(outdir, training_cfg_path, validated, patients, forced_manifest, mixture_manifest)
    (outdir / "inspection_source.json").write_text(json.dumps({
        "training_config": str(training_cfg_path),
        "validated_training_config": bool(validated),
        "patients": [int(x) for x in patients],
        "seed": base_seed,
    }, indent=2))
    print(f"\nSynthetic inspection bank written to: {outdir}")
    print(f"Viewer-compatible branch bank: {forced_root}")
    print(f"Exact training-mixture draws: {mixture_root}")


if __name__ == "__main__":
    main()
