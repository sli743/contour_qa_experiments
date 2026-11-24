"""Write the per-ROI configs of the generic NSR-QA pipeline (any ROI in the registry).

The pipeline is the Lung_L/Lung_R one, unchanged in method:
  error-field characterization (training1 only) -> realized-severity fit (Lung_L v3
  method) -> contour-only study prepare -> CT + contour study.

Two configs are derived from the Lung_R copies, which are themselves the Lung_L
settings with only data paths changed:
  configs/dino_shape/lung_r_error_field_characterization.yaml
  configs/dino_shape/lung_r_nsrqa_synthetic_severity_fit_v3method.yaml
Only roi, names, data paths and output directories are replaced.

Starting config of the fit: the ROI's month-temporal all-ROI NSR-QA config
(<temporal study>/configs/<ROI>/nsrqa.yaml). It has the same corruption mixture
and training recipe as the Lung_L v3 fallback config, with the ROI's anatomy-profile
severities; the fit's multiplier search starts from those. (The lungs started from a
first-pass severity calibration instead; the fit config allows either.)
The fit config also enables the end-truncation guard, the only addition to the lung
method: draws that cut an S/I end segment off (held-out truncation) are redrawn.

Prints shell assignments (GROUP, PROFILE, the profile's region-truth settings and
config paths) for the driver script to eval.

  python -m analysis.make_roi_pipeline_configs --roi Heart --scratch /scratch/shaohan/contour_QA
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml

EF_TEMPLATE = "configs/dino_shape/lung_r_error_field_characterization.yaml"
FIT_TEMPLATE = "configs/dino_shape/lung_r_nsrqa_synthetic_severity_fit_v3method.yaml"
REGISTRY = "configs/dino_shape/all_roi_nsrqa_registry.yaml"

# Depth-matched truth and score grids by anatomy profile. Large solid = the lung
# settings. Medium tubular structures are ~15-25 mm across, so a 4.5 mm-deep core
# would almost never exist: 3 mm (2 voxels), smaller regions and positives.
PROFILE_SETTINGS = {
    "large_solid": {"TRUTH_T": 4.5, "MIN_REGION_ML": 0.5, "POSITIVE_ML": 2.0, "MIN_COMPONENT_ML": 0.05,
                    "DEPTHS": "3,4.5,5.25,6,7.5"},
    "medium_tubular": {"TRUTH_T": 3.0, "MIN_REGION_ML": 0.1, "POSITIVE_ML": 0.5, "MIN_COMPONENT_ML": 0.02,
                       "DEPTHS": "1.5,2.25,3,3.75,4.5"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--scratch", default="/scratch/shaohan/contour_QA")
    ap.add_argument("--temporal-study", default="/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study")
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--fit-suffix", default="v3method")
    args = ap.parse_args()

    roi, roi_lc = args.roi, args.roi.lower()
    scratch = Path(args.scratch)
    reg = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    if roi not in reg["rois"]:
        raise SystemExit(f"{roi} is not in {args.registry}")
    group, profile = reg["rois"][roi]["data_group"], reg["rois"][roi]["profile"]
    if profile not in PROFILE_SETTINGS:
        raise SystemExit(f"{roi}: profile {profile} has no region-truth settings yet "
                         f"(supported: {', '.join(PROFILE_SETTINGS)}); thin structures are left out on purpose")

    start = Path(args.temporal_study) / "configs" / roi / "nsrqa.yaml"
    if not start.is_file():
        raise SystemExit(f"missing {start}: prepare the month-temporal all-ROI study first")
    start_cfg = yaml.safe_load(start.read_text(encoding="utf-8"))
    data = start_cfg["data"]
    if data.get("roi_name") not in (None, roi):
        raise SystemExit(f"{start} is for {data.get('roi_name')}, not {roi}")

    ef_dir = scratch / f"{roi_lc}_error_field_characterization_v1"
    fit_dir = scratch / f"{roi_lc}_nsrqa_synthetic_severity_fit_{args.fit_suffix}"
    for d in (ef_dir, fit_dir):
        d.mkdir(parents=True, exist_ok=True)

    ef = yaml.safe_load(Path(EF_TEMPLATE).read_text(encoding="utf-8"))
    ef.update({"experiment_name": f"{roi_lc}_error_field_characterization_v1", "roi": roi,
               "output_dir": str(ef_dir)})
    ef_path = ef_dir / "config.yaml"
    ef_path.write_text(yaml.safe_dump(ef, sort_keys=False), encoding="utf-8")

    fit = yaml.safe_load(Path(FIT_TEMPLATE).read_text(encoding="utf-8"))
    fit.update({"experiment_name": f"{roi_lc}_nsrqa_synthetic_severity_fit_{args.fit_suffix}", "roi": roi,
                "initial_training_config": str(start), "fallback_training_config": str(start),
                "error_field_root": str(ef_dir)})
    fit["data"] = {"train_label_dir": data["train_label_dir"], "train_ct_dir": data["train_ct_dir"]}
    # Truncation is a held-out error type; on tubular ROIs dropout / noise can cut an
    # end segment off. Reject draws that shorten the S/I extent by more than the ROI's
    # real p95 endpoint error (no effect where corruptions never do that).
    fit["end_truncation_guard"] = {"enabled": True, "quantile": 0.95, "min_mm": 3.0, "max_redraws": 8}
    fit["output"] = {"directory": str(fit_dir),
                     "recommended_experiment_name": f"dino_shape_stochastic_region_dropout_{roi}_metric_fit_{args.fit_suffix}"}
    fit_path = fit_dir / "config.yaml"
    fit_path.write_text(yaml.safe_dump(fit, sort_keys=False), encoding="utf-8")

    out = {"GROUP": group, "PROFILE": profile, "START_CONFIG": str(start),
           "EF_CONFIG": str(ef_path), "EF_DIR": str(ef_dir),
           "FIT_CONFIG": str(fit_path), "FIT_DIR": str(fit_dir), **PROFILE_SETTINGS[profile]}
    for k, v in out.items():
        print(f"{k}={shlex.quote(str(v))}")


if __name__ == "__main__":
    main()
