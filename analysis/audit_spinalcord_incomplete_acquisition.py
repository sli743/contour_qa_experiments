#!/usr/bin/env python3
"""Audit all temporal-study SpinalCord cases for incomplete CT acquisition.

This is an acquisition audit, not a model-based classifier. It measures raw CT
S/I extent and how close the raw approved/autoseg SpinalCord endpoints lie to
actual CT acquisition boundaries. Automated flags are intentionally tiered:
``boundary_limited`` is sensitive, while ``likely_incomplete_acquisition`` also
requires unusually short approved-cord or CT coverage relative to the Month1+2
reference cohort. Manually confirmed cases can be supplied separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

from analysis.spinalcord_diagnostic_utils import resample_binary_to_reference


def load_months(path: Path) -> Dict[str, List[int]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {m: [int(x) for x in cfg.get(m, [])] for m in ("month1", "month2", "month3")}


def memberships(months: Dict[str, List[int]]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for month, pids in months.items():
        for pid in pids:
            out.setdefault(int(pid), []).append(month)
    return out


def load_lps(path: Path) -> sitk.Image:
    return sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")


def z_bounds(arr: np.ndarray):
    z = np.where(np.any(arr, axis=(1, 2)))[0]
    return (int(z.min()), int(z.max())) if len(z) else (None, None)


def percentile_or_nan(x, q):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    return float(np.percentile(x, q)) if len(x) else np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-data-root", default="/home/shaohan/radformation-data/data")
    p.add_argument("--temporal-split-yaml", default="configs/dino_shape/month_temporal_nsrqa_split.yaml")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--roi", default="SpinalCord")
    p.add_argument("--boundary-margin-mm", type=float, default=4.5, help="Endpoint-to-raw-CT-edge distance counted as boundary-limited")
    p.add_argument("--short-percentile", type=float, default=10.0, help="Month1+2 reference percentile used for short CT/cord coverage")
    p.add_argument("--confirmed-incomplete", nargs="*", type=int, default=[95, 129], help="Visually confirmed incomplete-acquisition patient IDs")
    args = p.parse_args()

    raw = Path(args.raw_data_root).expanduser().resolve()
    split_path = Path(args.temporal_split_yaml).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    months = load_months(split_path)
    member = memberships(months)
    confirmed = {int(x) for x in args.confirmed_incomplete}
    rows = []

    for i, pid in enumerate(sorted(member), 1):
        ct_path = raw / "images" / f"patient{pid}_image.nii.gz"
        app_path = raw / "labels-approved" / f"patient{pid}_{args.roi}_approved.nii.gz"
        auto_path = raw / "labels-autoseg" / f"patient{pid}_{args.roi}_autoseg.nii.gz"
        row = {"patient_id": pid, "month_membership": ",".join(member[pid]), "confirmed_incomplete": pid in confirmed}
        missing = [str(x) for x in (ct_path, app_path) if not x.is_file()]
        if missing:
            row.update(status="missing_files", missing_files=";".join(missing))
            rows.append(row); continue
        try:
            ct = load_lps(ct_path)
            app = load_lps(app_path)
            app = resample_binary_to_reference(app, ct)
            app_arr = sitk.GetArrayFromImage(app) > 0
            if not app_arr.any():
                raise ValueError("approved contour empty")
            auto_arr = None
            if auto_path.is_file():
                auto = load_lps(auto_path)
                auto = resample_binary_to_reference(auto, ct)
                auto_arr = sitk.GetArrayFromImage(auto) > 0

            spacing_z = float(ct.GetSpacing()[2])
            nz = int(ct.GetSize()[2])
            a0, a1 = z_bounds(app_arr)
            app_inf_margin = float(a0 * spacing_z)
            app_sup_margin = float((nz - 1 - a1) * spacing_z)
            app_span = float((a1 - a0 + 1) * spacing_z)
            ct_extent = float(nz * spacing_z)
            if auto_arr is not None and auto_arr.any():
                u0, u1 = z_bounds(auto_arr)
                auto_inf_margin = float(u0 * spacing_z)
                auto_sup_margin = float((nz - 1 - u1) * spacing_z)
                auto_span = float((u1 - u0 + 1) * spacing_z)
            else:
                auto_inf_margin = auto_sup_margin = auto_span = np.nan
            row.update({
                "status": "ok",
                "ct_slices_z": nz,
                "ct_spacing_z_mm": spacing_z,
                "ct_si_extent_mm": ct_extent,
                "approved_si_span_mm": app_span,
                "approved_inferior_margin_to_ct_edge_mm": app_inf_margin,
                "approved_superior_margin_to_ct_edge_mm": app_sup_margin,
                "autoseg_si_span_mm": auto_span,
                "autoseg_inferior_margin_to_ct_edge_mm": auto_inf_margin,
                "autoseg_superior_margin_to_ct_edge_mm": auto_sup_margin,
                "approved_inferior_boundary_limited": app_inf_margin <= args.boundary_margin_mm,
                "approved_superior_boundary_limited": app_sup_margin <= args.boundary_margin_mm,
                "approved_any_boundary_limited": min(app_inf_margin, app_sup_margin) <= args.boundary_margin_mm,
                "autoseg_inferior_boundary_limited": bool(np.isfinite(auto_inf_margin) and auto_inf_margin <= args.boundary_margin_mm),
                "autoseg_superior_boundary_limited": bool(np.isfinite(auto_sup_margin) and auto_sup_margin <= args.boundary_margin_mm),
                "raw_ct": str(ct_path), "raw_approved": str(app_path), "raw_autoseg": str(auto_path),
            })
        except Exception as exc:
            row.update(status="error", error=str(exc))
        rows.append(row)
        print(f"[{i}/{len(member)}] patient {pid}: {row['status']}")

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    # Reference thresholds use Month1+2-only patients that are not Month3 test
    # patients, mirroring the temporal study's no-leakage convention.
    ref = ok[ok["month_membership"].str.contains("month1|month2", regex=True) & ~ok["month_membership"].str.contains("month3", regex=False)].copy()
    q = float(args.short_percentile)
    app_short_thr = percentile_or_nan(ref["approved_si_span_mm"], q)
    ct_short_thr = percentile_or_nan(ref["ct_si_extent_mm"], q)

    df["short_approved_coverage"] = False
    df["short_ct_coverage"] = False
    mask_ok = df["status"] == "ok"
    df.loc[mask_ok, "short_approved_coverage"] = pd.to_numeric(df.loc[mask_ok, "approved_si_span_mm"], errors="coerce") <= app_short_thr
    df.loc[mask_ok, "short_ct_coverage"] = pd.to_numeric(df.loc[mask_ok, "ct_si_extent_mm"], errors="coerce") <= ct_short_thr
    df["likely_incomplete_acquisition"] = False
    df.loc[mask_ok, "likely_incomplete_acquisition"] = (
        df.loc[mask_ok, "approved_any_boundary_limited"].astype(bool)
        & (df.loc[mask_ok, "short_approved_coverage"].astype(bool) | df.loc[mask_ok, "short_ct_coverage"].astype(bool))
    )
    df["acquisition_review_status"] = "no_flag"
    df.loc[mask_ok & df["approved_any_boundary_limited"].fillna(False).astype(bool), "acquisition_review_status"] = "boundary_limited_review"
    df.loc[df["likely_incomplete_acquisition"].fillna(False).astype(bool), "acquisition_review_status"] = "likely_incomplete"
    df.loc[df["confirmed_incomplete"].fillna(False).astype(bool), "acquisition_review_status"] = "confirmed_incomplete"

    all_csv = out / "all_months_acquisition_audit.csv"
    df.to_csv(all_csv, index=False)
    candidates = df[df["acquisition_review_status"].isin(["confirmed_incomplete", "likely_incomplete"])].copy()
    candidates = candidates.sort_values(["acquisition_review_status", "approved_si_span_mm", "patient_id"])
    candidates.to_csv(out / "incomplete_acquisition_patient_list.csv", index=False)
    df[df["approved_any_boundary_limited"].fillna(False).astype(bool)].to_csv(out / "boundary_limited_review_list.csv", index=False)

    lines = [f"patient {int(r.patient_id)}\t{r.month_membership}\t{r.acquisition_review_status}" for r in candidates.itertuples()]
    (out / "incomplete_acquisition_patient_list.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    by_month = []
    for month in ("month1", "month2", "month3"):
        g = df[df["month_membership"].str.contains(month, regex=False)]
        by_month.append({
            "month": month,
            "n_patients": int(len(g)),
            "n_confirmed": int((g["acquisition_review_status"] == "confirmed_incomplete").sum()),
            "n_likely": int((g["acquisition_review_status"] == "likely_incomplete").sum()),
            "n_boundary_review": int((g["acquisition_review_status"] == "boundary_limited_review").sum()),
        })
    pd.DataFrame(by_month).to_csv(out / "acquisition_counts_by_month.csv", index=False)

    meta = {
        "raw_data_root": str(raw), "temporal_split_yaml": str(split_path), "roi": args.roi,
        "boundary_margin_mm": args.boundary_margin_mm, "short_percentile": q,
        "reference_definition": "Month1+2 memberships excluding any Month3 patient",
        "reference_n": int(len(ref)), "approved_span_short_threshold_mm": app_short_thr,
        "ct_extent_short_threshold_mm": ct_short_thr, "confirmed_incomplete": sorted(confirmed),
        "automated_definition": "approved endpoint near raw CT edge AND short approved-cord or CT S/I coverage relative to development reference",
    }
    (out / "audit_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = [
        "# SpinalCord incomplete-acquisition audit", "",
        "This audit is based on **raw CT acquisition boundaries**, not NSR-QA outputs.", "",
        f"- Raw data root: `{raw}`",
        f"- Boundary margin: **{args.boundary_margin_mm:g} mm**",
        f"- Development-reference short-coverage percentile: **P{q:g}**",
        f"- Approved-cord short threshold: **{app_short_thr:.2f} mm**",
        f"- CT S/I short threshold: **{ct_short_thr:.2f} mm**", "",
        "## Interpretation", "",
        "`confirmed_incomplete` is reserved for manually verified cases supplied on the command line. "
        "`likely_incomplete` is an automated candidate: the approved cord approaches a raw CT acquisition edge and either the CT or approved-cord S/I coverage is unusually short. "
        "`boundary_limited_review` is deliberately more sensitive and should be visually reviewed rather than automatically excluded.", "",
        "## Patient list", "",
    ]
    if len(candidates):
        report += ["| Patient | Month(s) | Status | Approved span (mm) | CT extent (mm) | Inf edge (mm) | Sup edge (mm) |", "|---:|---|---|---:|---:|---:|---:|"]
        for r in candidates.itertuples():
            report.append(f"| {int(r.patient_id)} | {r.month_membership} | {r.acquisition_review_status} | {float(r.approved_si_span_mm):.1f} | {float(r.ct_si_extent_mm):.1f} | {float(r.approved_inferior_margin_to_ct_edge_mm):.1f} | {float(r.approved_superior_margin_to_ct_edge_mm):.1f} |")
    else:
        report.append("No confirmed/likely incomplete-acquisition cases were found.")
    report += ["", "See `boundary_limited_review_list.csv` for the broader review set and `all_months_acquisition_audit.csv` for every temporal-study patient."]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {all_csv}")
    print(f"Patient list: {out / 'incomplete_acquisition_patient_list.csv'}")


if __name__ == "__main__":
    main()
