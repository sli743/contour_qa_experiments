#!/usr/bin/env python3
"""Diagnose inward/outward correction asymmetry in stochastic-dropout DAE v2.

This diagnostic is intentionally NOT a benchmark evaluation. It challenges the
trained decoder with the exact *family* used in v2 training: continuous signed-
distance whole-boundary offsets at fixed physical magnitudes, separately for
inward and outward directions.

Outputs answer four questions:
  1. Are changed/missing/excess masks internally consistent?
  2. Does the decoder handle inward and outward SDF offsets symmetrically?
  3. What foreground probability does it assign inside missing/excess shells?
  4. Did the training run actually expose the model to both directions?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Use project implementations so this diagnostic exercises the exact same model
# and signed-distance convention as training.
from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder
from training.corruptions.stochastic_region_dropout import global_signed_distance_offset
from training.train_dino_shape_stochastic_region_dropout_decoder import (
    BinaryROIDataset,
    amp_context,
    dice_per_case,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v2_global_boundary.yaml",
    )
    p.add_argument("--checkpoint", default=None,
                   help="Default: <output_root>/<experiment_name>/checkpoints/best_decoder.pth")
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--offset-mm", nargs="+", type=float, default=[1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    p.add_argument("--max-cases", type=int, default=0, help="0 = all cases")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save-example-masks", action="store_true",
                   help="Save a few masks as .npz for debugging; NIfTI is not required for the diagnostic.")
    return p.parse_args()


def finite_mean(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().cpu()) if x.numel() else float("nan")


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def binary_dice_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)
    den = int(a.sum()) + int(b.sum())
    return 1.0 if den == 0 else 2.0 * int(np.logical_and(a, b).sum()) / den


def make_fixed_offset(clean_3d: np.ndarray, spacing_zyx, magnitude_mm: float, direction: str) -> Tuple[np.ndarray, Dict]:
    import random
    if direction not in {"inward", "outward"}:
        raise ValueError(direction)
    spec = {
        "magnitude_mm": [float(magnitude_mm), float(magnitude_mm)],
        "outward_probability": 1.0 if direction == "outward" else 0.0,
    }
    # magnitude and direction are deterministic from spec, RNG is only required by API.
    result, meta = global_signed_distance_offset(clean_3d.astype(bool), spacing_zyx, spec, random.Random(0))
    return result.astype(np.float32), meta


def load_model(cfg: Dict, checkpoint: Path, device: torch.device):
    model = Frozen3DINOShapeAutoencoder(
        repo_path=cfg["dino"]["repo_path"],
        checkpoint_path=cfg["dino"]["checkpoint_path"],
        config_name=cfg["dino"].get("config_name", "train/vit3d_highres"),
        decoder_channels=cfg["decoder"].get("channels", [256, 128, 64, 32]),
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    decoder_state = state.get("decoder_state_dict", state)
    model.decoder.load_state_dict(decoder_state, strict=True)
    model.eval()
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("Expected frozen 3DINO encoder.")
    return model, state


def get_dataset(cfg: Dict, split: str):
    key = {"train": "train_label_dir", "validation": "validation_label_dir", "test": "test_label_dir"}[split]
    return BinaryROIDataset(cfg["data"][key])


def summarize_training_exposure(experiment_dir: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    path = experiment_dir / "logs" / "training_corruption_summary.csv"
    if not path.exists():
        return pd.DataFrame(), {"available": 0}
    df = pd.read_csv(path)
    totals: Dict[str, int] = {}
    for raw in df.get("branch_counts_json", pd.Series(dtype=str)).dropna():
        try:
            d = json.loads(raw)
        except Exception:
            continue
        for k, v in d.items():
            totals[str(k)] = totals.get(str(k), 0) + int(v)

    # Direction may be available only in validation corruption details. We report
    # what can be established from logs and avoid inventing train direction counts.
    details_path = experiment_dir / "logs" / "validation_corruption_details.csv"
    direction_counts = {}
    if details_path.exists():
        detail = pd.read_csv(details_path)
        if "global_boundary_offset_direction" in detail.columns:
            sub = detail[detail.get("branch", "") == "global_boundary_offset"]
            direction_counts = sub["global_boundary_offset_direction"].dropna().astype(str).value_counts().to_dict()

    n_total = sum(totals.values())
    summary: Dict[str, float] = {"available": 1, "logged_train_samples": n_total}
    for k, v in sorted(totals.items()):
        summary[f"train_branch_count__{k}"] = v
        summary[f"train_branch_fraction__{k}"] = v / n_total if n_total else float("nan")
    for k, v in direction_counts.items():
        summary[f"validation_global_offset_direction_count__{k}"] = int(v)
    return df, summary


def run_challenge(cfg: Dict, model, dataset, device: torch.device, offsets: Iterable[float], max_cases: int, out_dir: Path, save_examples: bool) -> pd.DataFrame:
    spacing = cfg["stochastic_region_dropout"].get("spacing_zyx", [1.5, 1.5, 1.5])
    precision = cfg.get("training", {}).get("precision", "bf16")
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    rows: List[Dict] = []
    n_cases = len(dataset) if max_cases <= 0 else min(len(dataset), max_cases)
    example_dir = out_dir / "example_masks"
    if save_examples:
        example_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_cases):
        item = dataset[i]
        clean_t = item["mask"].float()  # [1,D,H,W]
        pid = str(item["patient_id"])
        clean_np = clean_t[0].numpy() > 0.5
        target_vox = int(clean_np.sum())
        if target_vox == 0:
            continue

        for direction in ("inward", "outward"):
            for mag in offsets:
                corrupted_np, meta = make_fixed_offset(clean_np, spacing, float(mag), direction)
                corrupted_bool = corrupted_np > 0.5
                changed_np = np.logical_xor(clean_np, corrupted_bool)
                missing_np = np.logical_and(clean_np, ~corrupted_bool)
                excess_np = np.logical_and(~clean_np, corrupted_bool)

                # Exact mask identities. Any failure here indicates corruption/mask
                # bookkeeping is wrong and should stop interpretation immediately.
                assert not np.logical_and(missing_np, excess_np).any(), "missing/excess overlap"
                assert np.array_equal(changed_np, np.logical_or(missing_np, excess_np)), "changed != missing U excess"
                if direction == "outward":
                    assert int(missing_np.sum()) == 0, f"Outward offset created missing voxels for {pid}, {mag} mm"
                else:
                    assert int(excess_np.sum()) == 0, f"Inward offset created excess voxels for {pid}, {mag} mm"

                # A physical SDF offset can be a no-op after rasterization when
                # the requested displacement is smaller than the distance to the
                # next voxel centre (e.g. 1.5 mm on a ~3 mm grid). This is an
                # expected discretization effect, not a mask-bookkeeping error.
                is_noop = int(changed_np.sum()) == 0

                x = torch.from_numpy(corrupted_np)[None, None].to(device)
                target = clean_t[None].to(device)
                with torch.no_grad(), amp_context(device, precision):
                    logits, _ = model(x)
                    prob = torch.sigmoid(logits)
                recon = (prob >= threshold).float()

                prob_cpu = prob.detach().float().cpu()[0, 0]
                recon_np = recon.detach().cpu().numpy()[0, 0] > 0.5
                input_vox = int(corrupted_bool.sum())
                recon_vox = int(recon_np.sum())
                delta_input = input_vox - target_vox
                follow = float("nan") if is_noop else safe_ratio(recon_vox - target_vox, delta_input)

                missing_t = torch.from_numpy(missing_np)
                excess_t = torch.from_numpy(excess_np)
                changed_t = torch.from_numpy(changed_np)
                p_missing = finite_mean(prob_cpu[missing_t]) if missing_t.any() else float("nan")
                p_excess = finite_mean(prob_cpu[excess_t]) if excess_t.any() else float("nan")
                # Direction-normalized correction probability: 1 is ideal in the
                # changed shell for either sign.
                shell_correct_prob = p_missing if direction == "inward" else (1.0 - p_excess)

                row = {
                    "patient_id": pid,
                    "direction": direction,
                    "offset_mm": float(mag),
                    "signed_offset_mm": float(mag if direction == "outward" else -mag),
                    "target_voxels": target_vox,
                    "input_voxels": input_vox,
                    "recon_voxels": recon_vox,
                    "changed_voxels": int(changed_np.sum()),
                    "missing_voxels": int(missing_np.sum()),
                    "excess_voxels": int(excess_np.sum()),
                    "input_over_target": input_vox / target_vox,
                    "recon_over_target": recon_vox / target_vox,
                    "recon_over_input": recon_vox / input_vox if input_vox else float("nan"),
                    "abs_input_target_volume_change_pct": 100.0 * abs(delta_input) / target_vox,
                    "volume_follow_fraction": follow,
                    "volume_correction_fraction": 1.0 - follow if np.isfinite(follow) else float("nan"),
                    "dice_input_target": binary_dice_np(corrupted_bool, clean_np),
                    "dice_recon_target": binary_dice_np(recon_np, clean_np),
                    "dice_input_recon": binary_dice_np(corrupted_bool, recon_np),
                    "dice_correction_gain": binary_dice_np(recon_np, clean_np) - binary_dice_np(corrupted_bool, clean_np),
                    "mean_prob_missing_region": p_missing,
                    "mean_prob_excess_region": p_excess,
                    "mean_correct_class_prob_changed_shell": shell_correct_prob,
                    "mean_prob_changed_shell": finite_mean(prob_cpu[changed_t]),
                    "mask_identity_ok": 1,
                    "realized_offset_noop": int(is_noop),
                    **{k: v for k, v in meta.items() if np.isscalar(v) or isinstance(v, str)},
                }
                rows.append(row)

                if save_examples and i < 2 and mag in (min(offsets), max(offsets)):
                    np.savez_compressed(
                        example_dir / f"{pid}_{direction}_{mag:g}mm.npz",
                        target=clean_np.astype(np.uint8),
                        input=corrupted_bool.astype(np.uint8),
                        reconstruction=recon_np.astype(np.uint8),
                        changed=changed_np.astype(np.uint8),
                        missing=missing_np.astype(np.uint8),
                        excess=excess_np.astype(np.uint8),
                        probability=prob_cpu.numpy().astype(np.float32),
                    )
    return pd.DataFrame(rows)


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "abs_input_target_volume_change_pct", "input_over_target", "recon_over_target",
        "volume_follow_fraction", "volume_correction_fraction", "dice_input_target",
        "dice_recon_target", "dice_input_recon", "dice_correction_gain",
        "mean_prob_missing_region", "mean_prob_excess_region", "mean_correct_class_prob_changed_shell",
    ]
    out = []
    for (direction, offset), g in rows.groupby(["direction", "offset_mm"], sort=True):
        effective = g[g["realized_offset_noop"] == 0]
        r = {
            "direction": direction, "offset_mm": offset, "n": len(g),
            "n_effective": len(effective),
            "n_noop": int((g["realized_offset_noop"] == 1).sum()),
        }
        for m in metrics:
            vals = pd.to_numeric(effective[m], errors="coerce").dropna()
            r[f"{m}_median"] = float(vals.median()) if len(vals) else float("nan")
            r[f"{m}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            r[f"{m}_q25"] = float(vals.quantile(.25)) if len(vals) else float("nan")
            r[f"{m}_q75"] = float(vals.quantile(.75)) if len(vals) else float("nan")
        out.append(r)
    return pd.DataFrame(out)


def overall_direction_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for direction, g in rows.groupby("direction"):
        g = g[g["realized_offset_noop"] == 0].copy()
        r = {"direction": direction, "n": len(g)}
        for m in ["volume_follow_fraction", "dice_correction_gain", "mean_correct_class_prob_changed_shell",
                  "recon_over_target", "dice_input_recon"]:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            r[f"median_{m}"] = float(vals.median()) if len(vals) else float("nan")
        # monotonic association with physical magnitude
        for m in ["recon_over_target", "dice_input_recon", "mean_correct_class_prob_changed_shell"]:
            tmp = g[["offset_mm", m]].dropna()
            r[f"spearman_offset_vs_{m}"] = float(tmp.corr(method="spearman").iloc[0,1]) if len(tmp) > 2 else float("nan")
        out.append(r)
    return pd.DataFrame(out)


def fmt(v, nd=3):
    try:
        v = float(v)
        return "NA" if not np.isfinite(v) else f"{v:.{nd}f}"
    except Exception:
        return str(v)


def write_report(out_dir: Path, cfg: Dict, ckpt_state: Dict, rows: pd.DataFrame, by_level: pd.DataFrame,
                 direction_summary: pd.DataFrame, exposure: Dict[str, float]):
    inward = direction_summary[direction_summary.direction == "inward"]
    outward = direction_summary[direction_summary.direction == "outward"]
    i = inward.iloc[0] if len(inward) else None
    o = outward.iloc[0] if len(outward) else None

    lines = [
        "# V2 directional-asymmetry diagnostic",
        "",
        "## Purpose",
        "",
        "This is a mechanistic diagnostic, not a benchmark score. The trained v2 decoder is challenged with fixed continuous signed-distance offsets from the same corruption family used during training, separately for inward (missing foreground) and outward (excess foreground) errors.",
        "",
        f"Checkpoint epoch: **{ckpt_state.get('epoch', 'unknown')}**  ",
        f"Checkpoint validation total: **{fmt(ckpt_state.get('validation_total', float('nan')), 6)}**",
        "",
        "## Mask sanity",
        "",
        f"All **{len(rows)}** generated challenge rows passed `changed = missing ∪ excess` and `missing ∩ excess = ∅`. "
        f"There were **{int(rows['realized_offset_noop'].sum())}** rasterized no-op offsets (requested physical offset produced no changed voxels). "
        "These rows are retained in the raw CSV but excluded from shell/follow summaries. Effective outward offsets are excess-only; effective inward offsets are missing-only.",
        "",
        "## Direction-level summary",
        "",
        "| direction | n | median follow fraction | median Dice correction gain | median correct-class probability in changed shell | median recon/target volume |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in direction_summary.iterrows():
        lines.append(
            f"| {r.direction} | {int(r.n)} | {fmt(r.median_volume_follow_fraction)} | {fmt(r.median_dice_correction_gain)} | "
            f"{fmt(r.median_mean_correct_class_prob_changed_shell)} | {fmt(r.median_recon_over_target)} |"
        )

    lines += [
        "",
        "Interpretation of shell probability: for inward/missing voxels, the desired target probability is foreground (`p→1`); for outward/excess voxels, the desired target probability is background (`1-p→1`). Thus the same 'correct-class probability' scale can be compared across directions.",
        "",
        "## Decision logic",
        "",
    ]
    if i is not None and o is not None:
        ci = float(i.median_mean_correct_class_prob_changed_shell)
        co = float(o.median_mean_correct_class_prob_changed_shell)
        fi = float(i.median_volume_follow_fraction)
        fo = float(o.median_volume_follow_fraction)
        if np.isfinite(ci) and np.isfinite(co) and ci - co > 0.15:
            lines.append("**Strong directional asymmetry is present:** the decoder assigns substantially higher target-class probability to inward/missing shells than to outward/excess shells. This points first to training/loss asymmetry rather than benchmark morphology mismatch.")
        elif np.isfinite(fo) and fo > 0.8 and (not np.isfinite(fi) or fi < fo - 0.2):
            lines.append("**Outward-following behavior is present:** reconstruction volume follows outward SDF offsets much more strongly than inward offsets. This again points toward directional training/loss asymmetry.")
        else:
            lines.append("**No large inward/outward asymmetry is established by the simple aggregate thresholds.** Inspect the per-level table. If outward SDF offsets are handled well despite benchmark dilation failure, the next suspect is SDF-offset vs binary-morphology distribution mismatch.")

    lines += [
        "",
        "Specifically:",
        "",
        "- If inward SDF offsets are corrected but outward SDF offsets are not, inspect the loss weighting and excess-region supervision before changing the benchmark or training duration.",
        "- If both signs of SDF offsets are corrected, but benchmark dilation remains poor, compare the geometry of SDF offsets with binary morphological dilation; that supports augmentation/test-family mismatch.",
        "- If neither sign is corrected, the issue is broader than foreground direction (capacity, representation, objective, or corruption strength).",
        "",
        "## Fixed-offset summary",
        "",
        "| direction | offset (mm) | n | input/target vol | recon/target vol | follow | Dice gain | correct-class prob in shell |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in by_level.iterrows():
        lines.append(
            f"| {r.direction} | {r.offset_mm:g} | {int(r.n)} | {fmt(r.input_over_target_median)} | "
            f"{fmt(r.recon_over_target_median)} | {fmt(r.volume_follow_fraction_median)} | "
            f"{fmt(r.dice_correction_gain_median)} | {fmt(r.mean_correct_class_prob_changed_shell_median)} |"
        )

    lines += ["", "## Training exposure available from logs", ""]
    if int(exposure.get("available", 0)):
        branch_keys = sorted(k for k in exposure if k.startswith("train_branch_count__"))
        lines += ["| branch | count | fraction |", "|---|---:|---:|"]
        for k in branch_keys:
            name = k.split("__", 1)[1]
            lines.append(f"| {name} | {int(exposure[k])} | {fmt(exposure.get('train_branch_fraction__'+name))} |")
        dir_keys = sorted(k for k in exposure if k.startswith("validation_global_offset_direction_count__"))
        if dir_keys:
            lines += ["", "Validation-cache global-offset direction counts: " + ", ".join(
                f"{k.split('__',1)[1]}={int(exposure[k])}" for k in dir_keys
            ) + "."]
        lines += ["", "Note: the current training summary logs branch counts but may not log inward/outward direction for every *training* sample. If exact train-direction counts are absent, this report does not infer them from the configured 50/50 probability."]
    else:
        lines.append("Training corruption summary was not found; exposure could not be verified from logs.")

    lines += [
        "",
        "## Files",
        "",
        "- `directional_challenge_rows.csv`: patient × direction × offset raw results",
        "- `directional_challenge_by_level.csv`: aggregate by direction and physical offset",
        "- `direction_summary.csv`: compact inward/outward comparison",
        "- `training_exposure_summary.json`: counts recoverable from run logs",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    experiment_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else experiment_dir / "checkpoints" / "best_decoder.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_dir / "directional_asymmetry_diagnostic_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = cfg.get("training", {}).get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, state = load_model(cfg, checkpoint, device)
    ds = get_dataset(cfg, args.split)
    rows = run_challenge(cfg, model, ds, device, args.offset_mm, args.max_cases, out_dir, args.save_example_masks)
    if rows.empty:
        raise RuntimeError("No diagnostic rows generated.")
    by_level = aggregate(rows)
    direction_summary = overall_direction_summary(rows)
    _, exposure = summarize_training_exposure(experiment_dir)

    rows.to_csv(out_dir / "directional_challenge_rows.csv", index=False)
    by_level.to_csv(out_dir / "directional_challenge_by_level.csv", index=False)
    direction_summary.to_csv(out_dir / "direction_summary.csv", index=False)
    with (out_dir / "training_exposure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(exposure, f, indent=2, sort_keys=True)
    write_report(out_dir, cfg, state, rows, by_level, direction_summary, exposure)

    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint epoch: {state.get('epoch', 'unknown')}")
    print(f"Cases: {rows['patient_id'].nunique()}, rows: {len(rows)}")
    print(direction_summary.to_string(index=False))
    print(f"Wrote: {out_dir / 'report.md'}")
    print(f"Wrote: {out_dir / 'directional_challenge_rows.csv'}")
    print(f"Wrote: {out_dir / 'directional_challenge_by_level.csv'}")
    print(f"Wrote: {out_dir / 'direction_summary.csv'}")


if __name__ == "__main__":
    main()
