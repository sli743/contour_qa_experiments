#!/usr/bin/env python3
"""Export exact SpinalCord proposed probe-bank examples for lightweight 3D Slicer review.

This script is intentionally separate from the NSR-QA result viewer.  It uses
artifacts from the *pre-training* proposed distribution-validation run to:

1. rank development patients by a real autoseg-error severity metric;
2. rank same-patient proposed-bank samples by the exact bounded, group-balanced
   phenotype distance used by the validation analysis;
3. deterministically regenerate those exact bank samples from patient/split/
   sample_index; and
4. export CT + approved + actual autoseg + selected proposed probe contours on the
   same prepared-label grid for a small Slicer-only viewer.

Month 3 is never selected: the proposed bank contains Month1+Month2 development cases
only and real-case ranking is restricted to cohort == 'development'.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml


# When invoked as ``python analysis/export_spinalcord_v3_probe_examples.py``,
# Python puts ``analysis/`` rather than the repository root on sys.path.
# Add the working-tree root explicitly so the existing analysis/training
# modules resolve without requiring ``python -m`` or PYTHONPATH setup.
AUTO_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(AUTO_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTO_REPO_ROOT))

from analysis.spinalcord_corruption_support import mixed_distance_matrices
from analysis.spinalcord_error_geometry import extract_pair_features
from training.corruptions.spinalcord_proposed_distribution import proposed_spinalcord_corruption


CLEAN_BRANCHES = {"clean", "rejected_to_clean"}
ENDPOINT_FEATURES = [
    "inferior_missing_mm",
    "superior_missing_mm",
    "inferior_excess_mm",
    "superior_excess_mm",
]
SUPPORTED_COVERAGE_ENDPOINT_EVENTS_V3 = ("inferior_severe", "inferior_mild", "superior_mild", "none")
SUPPORTED_COVERAGE_ENDPOINT_EVENTS_V4 = (
    "inferior_severe",
    "inferior_moderate",
    "inferior_mild",
    "superior_mild",
    "inferior_excess",
    "superior_excess",
    "none",
)

DISPLAY_FEATURES = [
    "dice",
    "assd_mm",
    "hd95_mm",
    "missing_fraction_gt",
    "excess_fraction_gt",
    "inferior_missing_mm",
    "inferior_excess_mm",
    "superior_missing_mm",
    "superior_excess_mm",
    "centroid_shift_mm",
    "largest_missing_component_fraction",
    "largest_excess_component_fraction",
]


def stable_seed(base: int, *parts: object) -> int:
    """Exact seed construction used by analyze_spinalcord_proposed_corruption_distribution.py."""
    text = "|".join([str(base), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % (2**31 - 1)


def sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_pid(path: Path) -> int | None:
    m = re.search(r"patient[_]?(\d+)", path.name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_existing(path_text: str | None, fallback: Path | None = None) -> Path | None:
    if path_text:
        p = Path(path_text).expanduser()
        if p.exists():
            return p.resolve()
    if fallback is not None and fallback.exists():
        return fallback.resolve()
    return None


def label_file_index(nsrqa_cfg: Mapping[str, object]) -> Dict[int, Tuple[str, Path]]:
    data = nsrqa_cfg["data"]
    out: Dict[int, Tuple[str, Path]] = {}
    for split, key in (("training1", "train_label_dir"), ("validation1", "validation_label_dir")):
        root = Path(str(data[key])).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Prepared label directory does not exist: {root}")
        for p in sorted(root.rglob("*.nii.gz")):
            pid = extract_pid(p)
            if pid is None:
                continue
            if pid in out:
                raise RuntimeError(f"Duplicate prepared development patient {pid}: {out[pid][1]} and {p}")
            out[pid] = (split, p)
    return out


def inferior_is_low_array_z(img: sitk.Image) -> bool:
    size = img.GetSize()
    p0 = img.TransformIndexToPhysicalPoint((0, 0, 0))
    p1 = img.TransformIndexToPhysicalPoint((0, 0, max(int(size[2]) - 1, 0)))
    return bool(float(p1[2] - p0[2]) > 0)


def resample_mask_to_reference(path: Path, reference: sitk.Image) -> sitk.Image:
    moving = sitk.ReadImage(str(path))
    out = sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    arr = (sitk.GetArrayFromImage(out) > 0).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(reference)
    return sitk.Cast(img, sitk.sitkUInt8)


def resample_ct_to_reference(path: Path, reference: sitk.Image) -> sitk.Image:
    moving = sitk.ReadImage(str(path))
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        -1024.0,
        sitk.sitkFloat32,
    )


def write_binary_like(mask: np.ndarray, reference: sitk.Image, path: Path) -> None:
    img = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
    img.CopyInformation(reference)
    sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), str(path))


def source_paths(raw_root: Path, pid: int) -> Tuple[Path, Path, Path]:
    ct = raw_root / "images" / f"patient{pid}_image.nii.gz"
    approved = raw_root / "labels-approved" / f"patient{pid}_SpinalCord_approved.nii.gz"
    autoseg = raw_root / "labels-autoseg" / f"patient{pid}_SpinalCord_autoseg.nii.gz"
    missing = [p for p in (ct, approved, autoseg) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw files: " + ", ".join(map(str, missing)))
    return ct, approved, autoseg


def numeric_record(row: Mapping[str, object], keys: Iterable[str]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for key in keys:
        value = row.get(key, np.nan)
        try:
            value = float(value)
        except Exception:
            value = np.nan
        out[key] = None if not np.isfinite(value) else value
    return out


def rank_value(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "1_minus_dice":
        return 1.0 - pd.to_numeric(df["dice"], errors="coerce")
    if metric not in df.columns:
        raise KeyError(f"Patient ranking metric {metric!r} is not in real case_features.csv")
    values = pd.to_numeric(df[metric], errors="coerce")
    if metric == "dice":
        # Lower Dice is worse; convert to a larger-is-worse value for sorting.
        return 1.0 - values
    return values


def dominant_endpoint_feature(row: Mapping[str, object]) -> Tuple[str | None, float]:
    """Return the largest real S/I endpoint discrepancy in millimeters."""
    best_name = None
    best_value = 0.0
    for key in ENDPOINT_FEATURES:
        try:
            value = float(row.get(key, 0.0))
        except Exception:
            value = 0.0
        if np.isfinite(value) and value > best_value:
            best_name = key
            best_value = value
    return best_name, float(best_value)


def supported_coverage_endpoint_events(proposal_cfg: Mapping[str, object]) -> Tuple[str, ...]:
    strategy = str(proposal_cfg.get("sampling_strategy", ""))
    if strategy == "compositional_v4":
        return SUPPORTED_COVERAGE_ENDPOINT_EVENTS_V4
    return SUPPORTED_COVERAGE_ENDPOINT_EVENTS_V3


def forced_endpoint_config(proposal_cfg: Mapping[str, object], event: str) -> Dict[str, object]:
    """Build a coverage-only config forcing one supported endpoint stratum.

    The forced bank is inspection-only.  It uses the same candidate sampler but
    disables clean/local-bite draws and sets endpoint probabilities so the
    requested endpoint event is generated.  It never contributes to prevalence
    or distribution statistics.
    """
    supported = supported_coverage_endpoint_events(proposal_cfg)
    if event not in supported:
        raise ValueError(f"Unsupported coverage endpoint event {event!r}; supported={supported}")
    cfg = copy.deepcopy(dict(proposal_cfg))
    cfg["clean_probability"] = 0.0
    cfg["local_bite_probability"] = 0.0
    strategy = str(cfg.get("sampling_strategy", ""))

    endpoint_key = "endpoint_v4" if strategy == "compositional_v4" else "endpoint_v3"
    ep = copy.deepcopy(dict(cfg.get(endpoint_key, {})))
    model = copy.deepcopy(dict(ep.get("severe_inferior_probability_model", {})))

    severe = 0.98 if event == "inferior_severe" else 0.0
    model.update({
        "development_prevalence": severe,
        "raw_weight": 0.0,
        "min_unconditional_probability": severe,
        "max_unconditional_probability": severe,
    })
    ep["severe_inferior_probability_model"] = model
    ep["mild_inferior_unconditional_probability"] = 0.98 if event == "inferior_mild" else 0.0
    ep["superior_mild_unconditional_probability"] = 0.98 if event == "superior_mild" else 0.0
    if strategy == "compositional_v4":
        ep["moderate_inferior_unconditional_probability"] = 0.98 if event == "inferior_moderate" else 0.0
        ep["inferior_excess_unconditional_probability"] = 0.98 if event == "inferior_excess" else 0.0
        ep["superior_excess_unconditional_probability"] = 0.98 if event == "superior_excess" else 0.0
    cfg[endpoint_key] = ep
    return cfg

def generate_forced_endpoint_sample(
    *,
    pid: int,
    source_split: str,
    approved: np.ndarray,
    spacing_zyx: Sequence[float],
    proposal_cfg: Mapping[str, object],
    base_seed: int,
    inferior_low: bool,
    event: str,
    replicate: int,
    max_attempts: int = 200,
) -> Tuple[np.ndarray, Dict[str, object], int]:
    """Generate one coverage-only sample using the same proposed corruption code."""
    cfg = forced_endpoint_config(proposal_cfg, event)
    for attempt in range(max_attempts):
        rng = random.Random(stable_seed(base_seed, "coverage_bank", pid, source_split, event, replicate, attempt))
        probe, meta = proposed_spinalcord_corruption(
            approved,
            spacing_zyx,
            cfg,
            rng,
            inferior_is_low_z=inferior_low,
            cache_key=None,
        )
        actual_event = str(meta.get("endpoint_event", "none"))
        applied = int(meta.get("endpoint_applied", 0))
        if actual_event == event and ((event == "none" and applied == 0) or (event != "none" and applied == 1)):
            return np.asarray(probe, dtype=bool), dict(meta), attempt
    raise RuntimeError(
        f"Patient {pid}: failed to generate forced coverage event {event!r} "
        f"after {max_attempts} attempts"
    )


def annotate_distance_to_real(
    real_row: Mapping[str, object],
    features: Mapping[str, object],
    scaler: Mapping[str, object],
    groups: Mapping[str, object],
) -> Tuple[float, Dict[str, float]]:
    one = pd.DataFrame([dict(features)])
    d, mats = mixed_distance_matrices([real_row], one, scaler, groups)
    return float(d[0, 0]), {name: float(mat[0, 0]) for name, mat in mats.items()}


def compare_regenerated_row(csv_row: Mapping[str, object], regenerated: Mapping[str, object]) -> None:
    """Fail loudly if regeneration no longer reproduces the saved bank feature row."""
    exact_keys = ["branch", "generator_composition_label"]
    for key in exact_keys:
        expected = str(csv_row.get(key, ""))
        if key == "generator_composition_label":
            actual = str(regenerated.get("composition_label", ""))
        else:
            actual = str(regenerated.get(key, ""))
        if expected and expected != "nan" and expected != actual:
            raise RuntimeError(f"Regenerated {key} mismatch: saved={expected!r}, regenerated={actual!r}")


def replay_patient_bank_samples(
    *,
    pid: int,
    source_split: str,
    approved: np.ndarray,
    spacing_zyx: Sequence[float],
    proposal_cfg: Mapping[str, object],
    base_seed: int,
    inferior_low: bool,
    wanted_sample_indices: Sequence[int],
) -> Dict[int, Tuple[np.ndarray, Dict[str, object]]]:
    """Replay the original patient shard in sample-index order.

    The v3 validation bank was generated sequentially per patient with a shared
    ``cache_key=f"proposed:{pid}"``.  ``small_local_bite`` delegates to the
    legacy stochastic-region-dropout implementation, whose geometry cache is
    initialized by the first local-bite call for that patient.  Since the
    compositional v3 contour entering local bite varies by sample, regenerating
    an isolated sample can otherwise differ from the historical saved bank.

    Replaying 0..max(wanted) reproduces the bank-generation history without
    changing the validated v3 sampler.  A unique export cache key avoids
    colliding with any other use in this process while preserving the same
    within-patient cache behavior.
    """
    wanted = sorted({int(x) for x in wanted_sample_indices})
    if not wanted:
        return {}
    if wanted[0] < 0:
        raise ValueError(f"Negative sample index requested for patient {pid}: {wanted[0]}")

    wanted_set = set(wanted)
    replayed: Dict[int, Tuple[np.ndarray, Dict[str, object]]] = {}
    cache_key = f"probe_export_replay:{pid}"
    for sample_idx in range(wanted[-1] + 1):
        rng = random.Random(stable_seed(base_seed, "proposed_bank", pid, source_split, sample_idx))
        probe_arr, meta = proposed_spinalcord_corruption(
            approved,
            spacing_zyx,
            proposal_cfg,
            rng,
            inferior_is_low_z=inferior_low,
            cache_key=cache_key,
        )
        if sample_idx in wanted_set:
            replayed[sample_idx] = (np.asarray(probe_arr, dtype=bool).copy(), dict(meta))
    return replayed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-dir", required=True, help="Extracted proposed_distribution_validation_v3/v4 directory.")
    ap.add_argument(
        "--repo-root",
        default=str(AUTO_REPO_ROOT),
        help="contour_qa_experiments working tree; default is inferred from this script location.",
    )
    ap.add_argument("--proposal-config", default=None)
    ap.add_argument("--calibration-config", default=None)
    ap.add_argument("--nsrqa-config", default=None)
    ap.add_argument("--calibration-root", default=None)
    ap.add_argument("--raw-data-root", default="/home/shaohan/radformation-data/data")
    ap.add_argument("--output-dir", default=None, help="Default: <validation-dir>/slicer_probe_examples")
    ap.add_argument("--top-patients", type=int, default=12, help="Number of worst development patients to export.")
    ap.add_argument(
        "--good-patients",
        type=int,
        default=4,
        help=(
            "Also export this many best development patients by --rank-by. "
            "Set to 0 to export only bad patients."
        ),
    )
    ap.add_argument("--probes-per-patient", type=int, default=5, help="Overall nearest random-bank probes per patient.")
    ap.add_argument(
        "--endpoint-probes-per-patient",
        type=int,
        default=3,
        help=(
            "Random-bank probes that match the patient's dominant S/I endpoint direction "
            "and fall within the endpoint-magnitude tolerance before ranking by the full phenotype distance."
        ),
    )
    ap.add_argument(
        "--endpoint-match-abs-tolerance-mm",
        type=float,
        default=3.0,
        help="Minimum absolute tolerance for endpoint-matched retrieval (default: 3 mm).",
    )
    ap.add_argument(
        "--endpoint-match-relative-tolerance",
        type=float,
        default=0.35,
        help="Relative endpoint-magnitude tolerance; effective tolerance is max(abs, relative*real).",
    )
    ap.add_argument(
        "--largest-inferior-probes",
        type=int,
        default=3,
        help="Also export this many random-bank samples with the largest inferior truncation.",
    )
    ap.add_argument(
        "--coverage-probes-per-event",
        type=int,
        default=2,
        help=(
            "Coverage-only forced samples per endpoint event supported by the selected sampler. "
            "These are never used in population prevalence statistics."
        ),
    )
    ap.add_argument(
        "--coverage-for-good",
        action="store_true",
        help="Also generate forced endpoint-coverage samples for GOOD patients (off by default).",
    )
    ap.add_argument(
        "--rank-by",
        default="hd95_mm",
        choices=["hd95_mm", "assd_mm", "1_minus_dice", "dice", "missing_fraction_gt", "inferior_missing_mm"],
    )
    ap.add_argument("--patient-id", action="append", type=int, default=[], help="Export specified patient(s) instead of top-N ranking.")
    ap.add_argument("--include-clean", action="store_true", help="Allow clean/rejected samples in nearest-probe ranking.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    validation_dir = Path(args.validation_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not validation_dir.is_dir():
        raise FileNotFoundError(validation_dir)

    resolved_path = validation_dir / "resolved_run_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Missing resolved_run_config.json: {resolved_path}")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))

    validation_name = validation_dir.name.lower()
    fallback_proposal = repo_root / "configs/dino_shape/spinalcord_proposed_corruption_distribution_v4.yaml"
    if "v4" not in validation_name:
        fallback_proposal = repo_root / "configs/dino_shape/spinalcord_proposed_corruption_distribution_v3.yaml"
    proposal_config = resolve_existing(
        args.proposal_config or resolved.get("proposal_config"),
        fallback_proposal,
    )
    calibration_config = resolve_existing(
        args.calibration_config or resolved.get("analysis_config"),
        repo_root / "configs/dino_shape/spinalcord_corruption_calibration.yaml",
    )
    nsrqa_config = resolve_existing(args.nsrqa_config or resolved.get("nsrqa_config"))
    calibration_root = resolve_existing(args.calibration_root or resolved.get("calibration_root"))
    for label, path in (
        ("proposal config", proposal_config),
        ("calibration config", calibration_config),
        ("NSR-QA config", nsrqa_config),
        ("calibration root", calibration_root),
    ):
        if path is None:
            raise FileNotFoundError(f"Could not resolve {label}; pass its command-line override explicitly.")

    proposal_cfg = load_yaml(proposal_config)
    calibration_cfg = load_yaml(calibration_config)
    nsrqa_cfg = load_yaml(nsrqa_config)
    cfg_hash = sha256_json(proposal_cfg)
    expected_hash = str(resolved.get("proposal_config_sha256", ""))
    if expected_hash and cfg_hash != expected_hash:
        raise RuntimeError(
            "Proposal config hash differs from the validation run. "
            f"expected={expected_hash}, actual={cfg_hash}, file={proposal_config}"
        )

    bank_path = validation_dir / "proposed_synthetic_bank_features.csv"
    scaler_path = validation_dir / "support_scaler.json"
    real_path = calibration_root / "real" / "case_features.csv"
    for p in (bank_path, scaler_path, real_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    bank = pd.read_csv(bank_path)
    real = pd.read_csv(real_path)
    real = real[real["cohort"].astype(str) == "development"].copy()
    if real.empty:
        raise RuntimeError("No development cases found in real/case_features.csv")
    bank["patient_id"] = pd.to_numeric(bank["patient_id"], errors="raise").astype(int)
    real["patient_id"] = pd.to_numeric(real["patient_id"], errors="raise").astype(int)

    if "sampler_config_sha256" in bank.columns:
        hashes = set(bank["sampler_config_sha256"].dropna().astype(str))
        if hashes != {cfg_hash}:
            raise RuntimeError(f"Bank config hash mismatch: bank={hashes}, selected config={cfg_hash}")

    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    groups = calibration_cfg["phenotype_groups"]

    rank = rank_value(real, args.rank_by)
    real = real.assign(_badness=rank)
    real = real[real["patient_id"].isin(set(bank["patient_id"]))].copy()
    real = real[np.isfinite(pd.to_numeric(real["_badness"], errors="coerce"))].copy()
    if args.patient_id:
        wanted = list(dict.fromkeys(int(x) for x in args.patient_id))
        missing = sorted(set(wanted) - set(real["patient_id"]))
        if missing:
            raise KeyError(f"Requested patient(s) are not in the development bank: {missing}")
        selected_real = pd.concat([real[real.patient_id == pid] for pid in wanted], ignore_index=True)
        selected_real["_selection_group"] = "requested"
        selected_real["_selection_rank"] = np.arange(1, len(selected_real) + 1)
    else:
        ordered_bad = real.sort_values(["_badness", "patient_id"], ascending=[False, True])
        n_bad = max(0, int(args.top_patients))
        bad = ordered_bad.head(n_bad).copy()
        bad["_selection_group"] = "bad"
        bad["_selection_rank"] = np.arange(1, len(bad) + 1)

        used = set(bad["patient_id"].astype(int))
        n_good = max(0, int(args.good_patients))
        good_pool = real[~real["patient_id"].isin(used)].copy()
        good = good_pool.sort_values(["_badness", "patient_id"], ascending=[True, True]).head(n_good).copy()
        good["_selection_group"] = "good"
        good["_selection_rank"] = np.arange(1, len(good) + 1)

        selected_real = pd.concat([bad, good], ignore_index=True)
        if selected_real.empty:
            raise RuntimeError("No patients selected; increase --top-patients or --good-patients.")

    prepared_index = label_file_index(nsrqa_cfg)
    raw_root = Path(args.raw_data_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else validation_dir / "slicer_probe_examples"
    )
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_rows = []
    base_seed = int(proposal_cfg.get("seed", 88))

    for patient_rank, (_, real_row_series) in enumerate(selected_real.iterrows(), start=1):
        real_row = real_row_series.to_dict()
        pid = int(real_row["patient_id"])
        if pid not in prepared_index:
            raise FileNotFoundError(f"Prepared development label missing for patient {pid}")
        source_split, label_path = prepared_index[pid]

        same_all = bank[bank.patient_id == pid].copy()
        same = same_all.copy()
        if not args.include_clean:
            same = same[~same["branch"].fillna("").isin(CLEAN_BRANCHES)].copy()
            if "changed_fraction_gt" in same.columns:
                same = same[pd.to_numeric(same["changed_fraction_gt"], errors="coerce").fillna(0) > 0].copy()
        if same.empty:
            raise RuntimeError(f"No eligible proposed-bank samples for patient {pid}")

        d, group_mats = mixed_distance_matrices([real_row], same, scaler, groups)
        same["phenotype_distance"] = d[0]
        for group, mat in group_mats.items():
            same[f"distance_{group}"] = mat[0]

        # Keep population-distribution validation separate from inspection.
        # These selections all come from the original random bank; they simply
        # expose different retrieval questions for visualization.
        selections = []
        overall = same.sort_values(["phenotype_distance", "sample_index"], ascending=[True, True]).head(
            max(1, args.probes_per_patient)
        ).copy()
        for rank_i, (_, row_i) in enumerate(overall.iterrows(), start=1):
            selections.append(("overall_nearest", rank_i, row_i.to_dict(), None, None))

        endpoint_feature, real_endpoint_value = dominant_endpoint_feature(real_row)
        endpoint_match_tolerance_mm = None
        endpoint_match_random_bank_n = 0
        if endpoint_feature is not None and args.endpoint_probes_per_patient > 0:
            endpoint_sorted = same.copy()
            endpoint_values = pd.to_numeric(endpoint_sorted[endpoint_feature], errors="coerce")
            endpoint_sorted["endpoint_abs_diff_mm"] = (endpoint_values - real_endpoint_value).abs()
            endpoint_match_tolerance_mm = max(
                float(args.endpoint_match_abs_tolerance_mm),
                float(args.endpoint_match_relative_tolerance) * float(real_endpoint_value),
            )
            # Endpoint-matched means what it says: same endpoint error direction
            # must actually be present, and its magnitude must already be close.
            # Do not let an essentially no-cut sample win merely because its
            # remaining phenotype groups are close. If the finite random bank has
            # no qualifying sample, leave this retrieval view empty; the forced
            # coverage bank then answers whether the generator can express it.
            endpoint_sorted = endpoint_sorted[
                np.isfinite(endpoint_values)
                & (endpoint_values > 0.0)
                & (endpoint_sorted["endpoint_abs_diff_mm"] <= endpoint_match_tolerance_mm)
            ].copy()
            endpoint_match_random_bank_n = int(len(endpoint_sorted))
            endpoint_sorted = endpoint_sorted.sort_values(
                ["phenotype_distance", "endpoint_abs_diff_mm", "sample_index"],
                ascending=[True, True, True],
            ).head(args.endpoint_probes_per_patient)
            for rank_i, (_, row_i) in enumerate(endpoint_sorted.iterrows(), start=1):
                selections.append((
                    "endpoint_matched", rank_i, row_i.to_dict(), endpoint_feature, float(row_i["endpoint_abs_diff_mm"])
                ))

        if args.largest_inferior_probes > 0 and "inferior_missing_mm" in same.columns:
            largest_inf = same.sort_values(
                ["inferior_missing_mm", "phenotype_distance", "sample_index"],
                ascending=[False, True, True],
            ).head(args.largest_inferior_probes)
            for rank_i, (_, row_i) in enumerate(largest_inf.iterrows(), start=1):
                selections.append(("largest_inferior_bank", rank_i, row_i.to_dict(), "inferior_missing_mm", None))

        # Replay every unique random-bank sample needed by any retrieval view.
        selected_bank_indices = sorted({int(item[2]["sample_index"]) for item in selections})

        prepared_img = sitk.ReadImage(str(label_path))
        prepared_arr = sitk.GetArrayFromImage(prepared_img) > 0
        if not prepared_arr.any():
            raise ValueError(f"Prepared approved SpinalCord is empty: {label_path}")
        spacing_xyz = tuple(float(v) for v in prepared_img.GetSpacing())
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        inferior_low = inferior_is_low_array_z(prepared_img)

        ct_path, raw_approved_path, raw_autoseg_path = source_paths(raw_root, pid)
        ct_view = resample_ct_to_reference(ct_path, prepared_img)
        autoseg_view = resample_mask_to_reference(raw_autoseg_path, prepared_img)
        autoseg_arr = sitk.GetArrayFromImage(autoseg_view) > 0
        view_real_features, _ = extract_pair_features(
            prepared_arr,
            autoseg_arr,
            spacing_zyx,
            patient_id=pid,
            cohort="prepared_grid_view",
            include_components=False,
        )

        case_dir = output_dir / f"patient_{pid}"
        case_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(ct_view, str(case_dir / "ct.nii.gz"))
        write_binary_like(prepared_arr, prepared_img, case_dir / "approved.nii.gz")
        sitk.WriteImage(autoseg_view, str(case_dir / "autoseg.nii.gz"))

        # Reproduce the historical proposed bank exactly, including its sequential
        # patient-level local-bite cache history.  Do not regenerate selected
        # samples independently.
        replayed = replay_patient_bank_samples(
            pid=pid,
            source_split=source_split,
            approved=prepared_arr,
            spacing_zyx=spacing_zyx,
            proposal_cfg=proposal_cfg,
            base_seed=base_seed,
            inferior_low=inferior_low,
            wanted_sample_indices=selected_bank_indices,
        )

        probe_rows = []
        written_bank_files: Dict[int, str] = {}
        global_probe_rank = 0
        for inspection_kind, inspection_rank, saved, endpoint_match_feature, endpoint_abs_diff_mm in selections:
            global_probe_rank += 1
            sample_idx = int(saved["sample_index"])
            saved_split = str(saved.get("source_split", source_split))
            if saved_split != source_split:
                raise RuntimeError(
                    f"Patient {pid} split mismatch: bank={saved_split}, prepared_index={source_split}"
                )
            probe_arr, meta = replayed[sample_idx]
            compare_regenerated_row(saved, meta)
            regenerated_features, _ = extract_pair_features(
                prepared_arr,
                probe_arr,
                spacing_zyx,
                patient_id=pid,
                cohort="regenerated_proposed_probe",
                include_components=False,
            )
            for key in ("dice", "assd_mm", "hd95_mm", "inferior_missing_mm", "missing_fraction_gt"):
                expected = float(saved[key])
                actual = float(regenerated_features[key])
                if np.isfinite(expected) and np.isfinite(actual) and not np.isclose(expected, actual, rtol=1e-8, atol=1e-8):
                    raise RuntimeError(
                        f"Patient {pid} sample {sample_idx}: regeneration feature mismatch for {key}: "
                        f"saved={expected}, regenerated={actual}"
                    )

            if sample_idx not in written_bank_files:
                filename = f"bank_sample_{sample_idx:03d}.nii.gz"
                write_binary_like(probe_arr, prepared_img, case_dir / filename)
                written_bank_files[sample_idx] = filename
            filename = written_bank_files[sample_idx]
            row = {
                "probe_rank": int(global_probe_rank),
                "inspection_kind": inspection_kind,
                "inspection_rank": int(inspection_rank),
                "probe_source": "random_population_bank",
                "population_statistical_weight": 1,
                "sample_index": sample_idx,
                "filename": filename,
                "phenotype_distance": float(saved["phenotype_distance"]),
                "endpoint_match_feature": endpoint_match_feature or "",
                "endpoint_abs_diff_mm": endpoint_abs_diff_mm,
                "branch": str(saved.get("branch", "")),
                "composition_label": str(saved.get("generator_composition_label", meta.get("composition_label", ""))),
                "surface_mode": str(saved.get("generator_surface_mode", meta.get("surface_mode", ""))),
                "endpoint_event": str(saved.get("generator_endpoint_event", meta.get("endpoint_event", ""))),
                "local_bite_applied": int(float(saved.get("generator_local_bite_applied", meta.get("local_bite_applied", 0)) or 0)),
                **{f"distance_{g}": float(saved[f"distance_{g}"]) for g in group_mats if np.isfinite(saved[f"distance_{g}"])},
                **{f"probe_{k}": v for k, v in numeric_record(saved, DISPLAY_FEATURES).items()},
            }
            probe_rows.append(row)

        # Separate coverage bank: deliberately force each endpoint phenotype
        # through the SAME proposed sampler.  These samples are inspection-only and
        # are explicitly excluded from all prevalence/distribution statistics.
        selection_group = str(real_row.get("_selection_group", "requested"))
        do_coverage = args.coverage_probes_per_event > 0 and (selection_group != "good" or args.coverage_for_good)
        if do_coverage:
            for event in supported_coverage_endpoint_events(proposal_cfg):
                for rep in range(1, args.coverage_probes_per_event + 1):
                    probe_arr, meta, attempt = generate_forced_endpoint_sample(
                        pid=pid,
                        source_split=source_split,
                        approved=prepared_arr,
                        spacing_zyx=spacing_zyx,
                        proposal_cfg=proposal_cfg,
                        base_seed=base_seed,
                        inferior_low=inferior_low,
                        event=event,
                        replicate=rep,
                    )
                    feat, _ = extract_pair_features(
                        prepared_arr,
                        probe_arr,
                        spacing_zyx,
                        patient_id=pid,
                        cohort="forced_coverage_probe",
                        include_components=False,
                    )
                    distance, forced_group_distances = annotate_distance_to_real(
                        real_row, feat, scaler, groups
                    )
                    global_probe_rank += 1
                    filename = f"coverage_{event}_{rep:02d}.nii.gz"
                    write_binary_like(probe_arr, prepared_img, case_dir / filename)
                    forced_endpoint_diff = None
                    if endpoint_feature is not None:
                        try:
                            forced_endpoint_diff = abs(float(feat.get(endpoint_feature, np.nan)) - real_endpoint_value)
                        except Exception:
                            forced_endpoint_diff = None
                    probe_rows.append({
                        "probe_rank": int(global_probe_rank),
                        "inspection_kind": f"coverage_forced_{event}",
                        "inspection_rank": int(rep),
                        "probe_source": "forced_coverage_only",
                        "population_statistical_weight": 0,
                        "sample_index": f"coverage:{event}:{rep}",
                        "filename": filename,
                        "phenotype_distance": float(distance),
                        "endpoint_match_feature": endpoint_feature or "",
                        "endpoint_abs_diff_mm": forced_endpoint_diff,
                        "branch": str(meta.get("branch", "")),
                        "composition_label": str(meta.get("composition_label", "")),
                        "surface_mode": str(meta.get("surface_mode", "")),
                        "endpoint_event": str(meta.get("endpoint_event", event)),
                        "local_bite_applied": int(meta.get("local_bite_applied", 0) or 0),
                        "coverage_generation_attempt": int(attempt),
                        **{f"distance_{g}": v for g, v in forced_group_distances.items()},
                        **{f"probe_{k}": v for k, v in numeric_record(feat, DISPLAY_FEATURES).items()},
                    })

        probe_df = pd.DataFrame(probe_rows)
        probe_df.to_csv(case_dir / "probe_manifest.csv", index=False)
        patient_info = {
            "patient_id": pid,
            "patient_rank": int(patient_rank),
            "selection_group": str(real_row.get("_selection_group", "requested")),
            "selection_rank": int(real_row.get("_selection_rank", patient_rank)),
            "rank_by": args.rank_by,
            "rank_badness_value": float(real_row["_badness"]),
            "source_split": source_split,
            "prepared_label_path": str(label_path),
            "raw_ct_path": str(ct_path),
            "raw_approved_path": str(raw_approved_path),
            "raw_autoseg_path": str(raw_autoseg_path),
            "proposal_config": str(proposal_config),
            "proposal_config_sha256": cfg_hash,
            "sampler_version": str(proposal_cfg.get("sampler_version", "")),
            "distance_method": "mixed_bounded_v2 (same group-balanced distance as distribution validation)",
            "dominant_endpoint_feature": endpoint_feature,
            "dominant_endpoint_real_value_mm": real_endpoint_value,
            "endpoint_match_tolerance_mm": endpoint_match_tolerance_mm,
            "endpoint_match_random_bank_eligible_n": endpoint_match_random_bank_n,
            "endpoint_match_status": (
                "matched" if endpoint_match_random_bank_n > 0
                else ("no_random_bank_match_within_tolerance" if endpoint_feature is not None else "no_dominant_endpoint_error")
            ),
            "random_bank_max_inferior_missing_mm": float(pd.to_numeric(same_all.get("inferior_missing_mm"), errors="coerce").max()) if "inferior_missing_mm" in same_all else None,
            "inspection_policy": {
                "population_bank_unchanged": True,
                "overall_nearest_random_bank": int(args.probes_per_patient),
                "endpoint_matched_random_bank": int(args.endpoint_probes_per_patient),
                "largest_inferior_random_bank": int(args.largest_inferior_probes),
                "forced_coverage_per_endpoint_event": int(args.coverage_probes_per_event if do_coverage else 0),
                "forced_coverage_statistical_weight": 0,
            },
            "real_full_volume_features": numeric_record(real_row, DISPLAY_FEATURES),
            "real_prepared_grid_features": numeric_record(view_real_features, DISPLAY_FEATURES),
            "probes": probe_rows,
        }
        (case_dir / "case_info.json").write_text(json.dumps(patient_info, indent=2), encoding="utf-8")

        root_rows.append({
            "patient_rank": int(patient_rank),
            "selection_group": str(real_row.get("_selection_group", "requested")),
            "selection_rank": int(real_row.get("_selection_rank", patient_rank)),
            "patient_id": pid,
            "source_split": source_split,
            "rank_by": args.rank_by,
            "rank_badness_value": float(real_row["_badness"]),
            "real_dice": float(real_row.get("dice", np.nan)),
            "real_hd95_mm": float(real_row.get("hd95_mm", np.nan)),
            "real_assd_mm": float(real_row.get("assd_mm", np.nan)),
            "real_inferior_missing_mm": float(real_row.get("inferior_missing_mm", np.nan)),
            "nearest_probe_distance": float(probe_df.loc[probe_df["probe_source"] == "random_population_bank", "phenotype_distance"].min()),
            "dominant_endpoint_feature": endpoint_feature,
            "dominant_endpoint_real_value_mm": real_endpoint_value,
            "random_bank_max_inferior_missing_mm": float(pd.to_numeric(same_all.get("inferior_missing_mm"), errors="coerce").max()) if "inferior_missing_mm" in same_all else np.nan,
            "n_exported_probes": int(len(probe_df)),
            "n_random_bank_views": int((probe_df["probe_source"] == "random_population_bank").sum()),
            "n_forced_coverage_views": int((probe_df["probe_source"] == "forced_coverage_only").sum()),
        })
        group = str(real_row.get("_selection_group", "requested"))
        group_rank = int(real_row.get("_selection_rank", patient_rank))
        group_total = int((selected_real["_selection_group"] == group).sum()) if "_selection_group" in selected_real else len(selected_real)
        print(
            f"[{group} {group_rank}/{group_total}] patient {pid}: "
            f"real_{args.rank_by}={float(real_row['_badness']):.4g}, "
            f"nearest proposed distance={float(probe_df.loc[probe_df['probe_source'] == 'random_population_bank', 'phenotype_distance'].min()):.4f}, "
            f"dominant endpoint={endpoint_feature or 'none'}:{real_endpoint_value:.1f} mm, "
            f"max bank inferior cut={float(pd.to_numeric(same_all.get('inferior_missing_mm'), errors='coerce').max()) if 'inferior_missing_mm' in same_all else float('nan'):.1f} mm"
        )

    root_df = pd.DataFrame(root_rows)
    root_df.to_csv(output_dir / "manifest.csv", index=False)
    root_meta = {
        "validation_dir": str(validation_dir),
        "validation_resolved_run_config": str(resolved_path),
        "proposal_config": str(proposal_config),
        "proposal_config_sha256": cfg_hash,
        "calibration_config": str(calibration_config),
        "calibration_root": str(calibration_root),
        "nsrqa_config": str(nsrqa_config),
        "raw_data_root": str(raw_root),
        "rank_by": args.rank_by,
        "top_patients": int(args.top_patients),
        "good_patients": int(args.good_patients),
        "probes_per_patient": int(args.probes_per_patient),
        "endpoint_probes_per_patient": int(args.endpoint_probes_per_patient),
        "largest_inferior_probes": int(args.largest_inferior_probes),
        "endpoint_match_abs_tolerance_mm": float(args.endpoint_match_abs_tolerance_mm),
        "endpoint_match_relative_tolerance": float(args.endpoint_match_relative_tolerance),
        "coverage_probes_per_event": int(args.coverage_probes_per_event),
        "coverage_for_good": bool(args.coverage_for_good),
        "population_bank_modified": False,
        "forced_coverage_included_in_distribution_statistics": False,
        "selected_patient_ids": [int(x) for x in root_df.patient_id.tolist()],
        "viewer_grid": "prepared approved SpinalCord grid; raw autoseg and CT resampled to this grid",
        "month3_used": False,
    }
    (output_dir / "export_info.json").write_text(json.dumps(root_meta, indent=2), encoding="utf-8")

    portable_readme = f"""SpinalCord proposed probe-bank inspection export

This directory is self-contained for offline 3D Slicer inspection.

COPY THIS ENTIRE DIRECTORY to the computer running 3D Slicer.
Do not copy the contour_qa_experiments repository or validation directory
unless you need them for some other purpose.

In 3D Slicer:
  1. Install/load the separate V3ProbeBankViewer scripted module.
  2. Set 'Export root' to THIS directory (the directory containing
     manifest.csv and patient_* subdirectories).
  3. Click 'Scan directory'.
  4. Choose a BAD or GOOD patient and an inspection probe.

Probe roles in probe_manifest.csv:
  overall_nearest          = nearest by the original group-balanced validation distance
  endpoint_matched         = random-bank samples with the SAME dominant endpoint
                             direction and a close endpoint magnitude first; only
                             then ranked by the full phenotype distance. If none
                             satisfy the tolerance, this view is intentionally empty.
  largest_inferior_bank    = largest inferior truncations already present in the
                             ORIGINAL random population bank
  coverage_forced_*        = deliberately stratified endpoint examples generated
                             with the same proposed corruption code for coverage inspection
                             ONLY. Their population_statistical_weight is 0 and they
                             must never be used for prevalence/distribution summaries.

Portable files used by the viewer:
  manifest.csv
  export_info.json
  patient_*/ct.nii.gz
  patient_*/approved.nii.gz
  patient_*/autoseg.nii.gz
  patient_*/probe_rank_*.nii.gz
  patient_*/probe_manifest.csv
  patient_*/case_info.json

The absolute source paths recorded in JSON are provenance only.
The viewer does not access those source paths.
"""
    (output_dir / "README.txt").write_text(portable_readme, encoding="utf-8")

    print(f"\nSaved self-contained Slicer-ready proposed probe examples to: {output_dir}")
    n_bad = int((root_df.get("selection_group") == "bad").sum()) if "selection_group" in root_df else 0
    n_good = int((root_df.get("selection_group") == "good").sum()) if "selection_group" in root_df else 0
    print(
        f"Patients: {len(root_df)} (bad={n_bad}, good={n_good}); "
        f"overall-nearest per patient={args.probes_per_patient}; "
        f"endpoint-matched={args.endpoint_probes_per_patient}; "
        f"largest-inferior={args.largest_inferior_probes}; "
        f"coverage/event={args.coverage_probes_per_event}"
    )


if __name__ == "__main__":
    main()
