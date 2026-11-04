#!/usr/bin/env python3
"""Export all flagged Lung_L cases for the Lung Error Field Viewer.

This is a companion to ``analysis.flag_lung_l_abnormal_inspection_cases``.
It reads that analysis' ``inspection_union.csv`` and writes every flagged case,
plus any explicitly requested manual-review patients from the export config,
in the same Slicer-compatible format produced by
``analysis.characterize_lung_error_fields``::

    output_dir/
      patient_<id>/
        ct.nii.gz
        approved.nii.gz
        autoseg.nii.gz
        real_missing.nii.gz
        real_excess.nii.gz
        approved_surface.nii.gz
        surface_displacement_mm.nii.gz
        smooth_surface_displacement_mm.nii.gz
        local_surface_residual_mm.nii.gz
        surface_*_ge_threshold.nii.gz
        smooth_*_ge_threshold.nii.gz
        local_*_ge_threshold.nii.gz
        largest_missing_component.nii.gz   # when present
        largest_excess_component.nii.gz    # when present
        components.csv
        multiscale_surface_summary.csv
        metrics.json
        flagging.json

The existing ``LungErrorFieldViewer.py`` can scan ``output_dir`` directly.
Patient-level multiprocessing is used because each export independently loads,
resamples, characterizes, and writes one case.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Tuple

import pandas as pd
import yaml

from analysis.characterize_lung_error_fields import (
    build_temporal_split,
    load_domains,
    write_case_export,
)


REQUIRED_FLAG_COLUMNS = {
    "patient_id",
    "month",
    "inspection_reason",
    "inspection_severity_score",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/dino_shape/lung_l_flagged_error_field_export.yaml",
        help="Export configuration YAML.",
    )
    p.add_argument("--workers", type=int, default=None, help="Override patient-level worker count.")
    p.add_argument("--overwrite", action="store_true", help="Re-export patients even if metrics.json already exists.")
    return p.parse_args()


def _safe_float(value):
    try:
        out = float(value)
        return out if pd.notna(out) else None
    except Exception:
        return None


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _append_manual_includes(flags: pd.DataFrame, patient_ids, error_cfg: Mapping) -> pd.DataFrame:
    """Add explicitly requested review cases without changing statistical flags.

    Manual includes are exported only for visualization. They are tagged with
    ``inspection_reason=manual_include`` and do not become statistically flagged
    cases in the upstream monthly inspection analysis.
    """
    manual_ids = sorted({int(x) for x in (patient_ids or [])})
    if not manual_ids:
        return flags

    _split, cohorts = build_temporal_split(Path(error_cfg["split_yaml"]))
    domain_map = load_domains(
        Path(error_cfg["domain_csv"]) if error_cfg.get("domain_csv") else None,
        str(error_cfg.get("roi", "Lung_L")),
        str(error_cfg.get("site", "thorax")),
    )

    existing = set(pd.to_numeric(flags["patient_id"], errors="coerce").dropna().astype(int))
    rows = []
    for pid in manual_ids:
        if pid in existing:
            continue
        cohort = cohorts.get(pid)
        if cohort is None:
            raise ValueError(
                f"Manual include patient {pid} is not present in {error_cfg['split_yaml']}; "
                "cannot determine month/cohort."
            )
        rows.append({
            "patient_id": pid,
            "month": cohort,
            "inspection_reason": "manual_include",
            "inspection_severity_score": float("nan"),
            "primary_poor_agreement": False,
            "hd99_local_extreme": False,
            "dice_below_lcl": False,
            "hd95_above_ucl": False,
            "hd99_above_ucl": False,
            "recorded_domain": domain_map.get(pid, "Unassigned"),
            "domain_conflict": False,
            "dice_flag_source": "manual_include",
            "hd95_flag_source": "manual_include",
        })

    if rows:
        flags = pd.concat([flags, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return flags


def build_flag_payload(row: Mapping) -> Dict[str, object]:
    """Keep the monthly inspection provenance beside the viewer export."""
    keys = [
        "month",
        "inspection_reason",
        "inspection_severity_score",
        "primary_poor_agreement",
        "hd99_local_extreme",
        "dice_below_lcl",
        "hd95_above_ucl",
        "hd99_above_ucl",
        "dice_for_flagging",
        "hd95_for_flagging_mm",
        "hd99_mm",
        "recomputed_dice",
        "recomputed_hd95_mm",
        "recomputed_assd_mm",
        "missing_fraction_gt",
        "excess_fraction_gt",
        "relative_volume_difference",
        "recorded_domain",
        "domain_conflict",
        "dice_flag_source",
        "hd95_flag_source",
    ]
    payload: Dict[str, object] = {}
    bool_keys = {
        "primary_poor_agreement",
        "hd99_local_extreme",
        "dice_below_lcl",
        "hd95_above_ucl",
        "hd99_above_ucl",
        "domain_conflict",
    }
    text_keys = {
        "month",
        "inspection_reason",
        "recorded_domain",
        "dice_flag_source",
        "hd95_flag_source",
    }
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if key in bool_keys:
            payload[key] = _safe_bool(value)
        elif key in text_keys:
            payload[key] = "" if pd.isna(value) else str(value)
        else:
            payload[key] = _safe_float(value)
    return payload


def inspection_strings(row: Mapping) -> Tuple[str, str]:
    month = str(row.get("month", "unknown"))
    reason = str(row.get("inspection_reason", "flagged"))
    score = _safe_float(row.get("inspection_severity_score"))
    score_text = "NA" if score is None else f"{score:.3f}"
    categories = f"monthly_abnormal_review;{month};{reason}"
    details = [
        f"{month} abnormal-shape review",
        f"flag={reason}",
        f"severity score={score_text}",
    ]
    dice = _safe_float(row.get("dice_for_flagging"))
    hd95 = _safe_float(row.get("hd95_for_flagging_mm"))
    hd99 = _safe_float(row.get("hd99_mm"))
    if dice is not None:
        details.append(f"Dice={dice:.4f}")
    if hd95 is not None:
        details.append(f"HD95={hd95:.3f} mm")
    if hd99 is not None:
        details.append(f"HD99={hd99:.3f} mm")
    return categories, "; ".join(details)


def _domain_from_row(row: Mapping) -> str:
    value = row.get("recorded_domain", "")
    if pd.isna(value) or not str(value).strip():
        return "Unassigned"
    return str(value).strip()


def _augment_metrics_json(patient_dir: Path, flag_payload: Mapping[str, object]) -> None:
    path = patient_dir / "metrics.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    else:
        data = {}
    data["flagging"] = dict(flag_payload)
    path.write_text(json.dumps(data, indent=2))
    (patient_dir / "flagging.json").write_text(json.dumps(dict(flag_payload), indent=2))


def export_one(task):
    row, error_cfg, outdir, overwrite = task
    pid = int(row["patient_id"])
    pdir = Path(outdir) / f"patient_{pid}"
    if not overwrite and (pdir / "metrics.json").is_file():
        return {"ok": True, "patient_id": pid, "status": "skipped_existing"}

    month = str(row.get("month", "unknown"))
    domain = _domain_from_row(row)
    categories, reasons = inspection_strings(row)
    viewer_row = {
        "inspection_categories": categories,
        "inspection_reasons": reasons,
    }
    try:
        write_case_export(pid, month, domain, error_cfg, Path(outdir), viewer_row)
        payload = build_flag_payload(row)
        _augment_metrics_json(pdir, payload)
        return {"ok": True, "patient_id": pid, "status": "exported"}
    except Exception as exc:
        return {"ok": False, "patient_id": pid, "status": "failed", "error": repr(exc)}


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    flags_csv = Path(cfg["flags_csv"])
    error_cfg_path = Path(cfg["error_field_config"])
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    if not flags_csv.is_file():
        raise FileNotFoundError(
            f"Flagged-case table not found: {flags_csv}. Run the monthly Lung_L flagging analysis first."
        )
    if not error_cfg_path.is_file():
        raise FileNotFoundError(f"Error-field config not found: {error_cfg_path}")

    flags = pd.read_csv(flags_csv)
    missing_cols = REQUIRED_FLAG_COLUMNS - set(flags.columns)
    if missing_cols:
        raise ValueError(f"{flags_csv} is missing required columns: {sorted(missing_cols)}")

    flags["patient_id"] = pd.to_numeric(flags["patient_id"], errors="raise").astype(int)

    error_cfg = yaml.safe_load(error_cfg_path.read_text())
    flags = _append_manual_includes(
        flags,
        cfg.get("manual_include_patients", []),
        error_cfg,
    )
    flags = flags.sort_values(
        ["month", "inspection_severity_score", "patient_id"],
        ascending=[True, False, True],
        na_position="last",
    ).drop_duplicates("patient_id", keep="first")

    if flags.empty:
        print("No flagged or manually included Lung_L cases to export.")
        pd.DataFrame().to_csv(outdir / "export_manifest.csv", index=False)
        return

    # The characterization config supplies all geometry/error-field settings
    # used by write_case_export. Its own output_dir/calibration_split do not
    # control this export.
    requested = args.workers if args.workers is not None else int(cfg.get("workers", 0))
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = requested if requested and requested > 0 else slurm_cpus
    workers = max(1, min(int(workers), len(flags)))
    overwrite = bool(args.overwrite or cfg.get("overwrite", False))

    print(
        f"Exporting {len(flags)} flagged/manual-review Lung_L cases to {outdir} "
        f"with {workers} patient-level workers (overwrite={overwrite})"
    )

    records = flags.to_dict("records")
    tasks = [(row, error_cfg, str(outdir), overwrite) for row in records]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(export_one, task): int(task[0]["patient_id"]) for task in tasks}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            done += 1
            if result["ok"]:
                print(f"[{done}/{len(tasks)}] patient {result['patient_id']}: {result['status']}")
            else:
                print(f"[{done}/{len(tasks)}] patient {result['patient_id']} FAILED: {result['error']}")

    result_df = pd.DataFrame(results)
    manifest = flags.merge(result_df, on="patient_id", how="left")
    manifest.to_csv(outdir / "export_manifest.csv", index=False)
    failures = manifest[manifest["status"].eq("failed")].copy() if "status" in manifest else pd.DataFrame()
    failures.to_csv(outdir / "export_failures.csv", index=False)

    # Month-specific lists make it easy to choose review cases while keeping a
    # single viewer-compatible directory containing all patient folders.
    for month, group in manifest.groupby("month", sort=True):
        group.sort_values("inspection_severity_score", ascending=False).to_csv(
            outdir / f"{month}_export_manifest.csv", index=False
        )

    metadata = {
        "flags_csv": str(flags_csv),
        "error_field_config": str(error_cfg_path),
        "output_dir": str(outdir),
        "n_selected": int(len(flags)),
        "manual_include_patients": [int(x) for x in cfg.get("manual_include_patients", [])],
        "n_exported": int((result_df.get("status") == "exported").sum()) if len(result_df) else 0,
        "n_skipped_existing": int((result_df.get("status") == "skipped_existing").sum()) if len(result_df) else 0,
        "n_failed": int((~result_df.get("ok", pd.Series(dtype=bool))).sum()) if len(result_df) else 0,
        "workers": workers,
    }
    (outdir / "export_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Viewer root: {outdir}")
    print("Open slicer/LungErrorFieldViewer.py and point 'Inspection root' to that directory.")


if __name__ == "__main__":
    main()
