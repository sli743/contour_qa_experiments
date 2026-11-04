#!/usr/bin/env python3
"""Diagnose whether three-class correction logits contain useful edit information.

This script does not train anything. It evaluates a trained preserve/add/remove
model on paired real autoseg -> approved cases and asks whether continuous
P(add) / P(remove) scores rank true correction voxels above preserve voxels.

Outputs include:
  * class prevalence in several spatial regions
  * one-vs-rest ROC-AUC and average precision for add/remove
  * sampled voxel scores for plotting/calibration
  * a common action-threshold sweep with patient-level correction metrics

Regions:
  whole_crop          : all voxels
  autoseg_local       : dilation of autoseg only (deployment-feasible support)
  oracle_union_local  : dilation of autoseg OR approved (diagnostic upper-bound support)
  oracle_change_local : dilation of true autoseg XOR approved changes (diagnostic only)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder
from training.train_supervised_autoseg_correction_threeclass import ADD, PRESERVE, REMOVE


CLASS_NAMES = {PRESERVE: "preserve", ADD: "add", REMOVE: "remove"}


def read_rows(path: str, split: str):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def read_binary(path: str, device: torch.device):
    arr = (sitk.GetArrayFromImage(sitk.ReadImage(path)) > 0).astype(np.float32)
    return torch.from_numpy(arr[None, None]).to(device)


def action_labels_np(autoseg: np.ndarray, approved: np.ndarray) -> np.ndarray:
    labels = np.zeros(autoseg.shape, dtype=np.uint8)
    labels[(autoseg == 0) & (approved == 1)] = ADD
    labels[(autoseg == 1) & (approved == 0)] = REMOVE
    return labels


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    structure = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_dilation(mask.astype(bool), structure=structure, iterations=radius)


def make_regions(autoseg: np.ndarray, approved: np.ndarray, radius: int):
    autoseg = autoseg.astype(bool)
    approved = approved.astype(bool)
    changed = autoseg ^ approved
    return {
        "whole_crop": np.ones_like(autoseg, dtype=bool),
        "autoseg_local": binary_dilate(autoseg, radius),
        "oracle_union_local": binary_dilate(autoseg | approved, radius),
        "oracle_change_local": binary_dilate(changed, radius),
    }


def make_model(cfg, checkpoint: str | None, device: torch.device):
    model = Frozen3DINOShapeAutoencoder(
        cfg["dino"]["repo_path"],
        cfg["dino"]["checkpoint_path"],
        cfg["dino"].get("config_name", "train/vit3d_highres"),
        cfg["decoder"].get("channels", [256, 128, 64, 32]),
        output_channels=3,
        decoder_blocks_per_stage=int(cfg["decoder"].get("blocks_per_stage", 1)),
        finetune_last_n_blocks=int(cfg.get("transfer_learning", {}).get("finetune_last_n_blocks", 0)),
    ).to(device)
    if checkpoint is None:
        checkpoint_path = (
            Path(cfg["output_root"])
            / cfg["experiment_name"]
            / "checkpoints"
            / "best_decoder.pth"
        )
    else:
        checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    model.decoder.load_state_dict(state["decoder_state_dict"])
    encoder_state = state.get("encoder_trainable_state_dict", {})
    if encoder_state:
        model.encoder.load_trainable_state_dict(encoder_state)
    model.eval()
    return model, checkpoint_path


def safe_auc_ap(y: np.ndarray, score: np.ndarray, sample_weight: np.ndarray):
    y = y.astype(np.uint8)
    finite = np.isfinite(score) & np.isfinite(sample_weight)
    y = y[finite]
    score = score[finite]
    sample_weight = sample_weight[finite]
    pos_weight = float(sample_weight[y == 1].sum())
    neg_weight = float(sample_weight[y == 0].sum())
    auc = (
        float(roc_auc_score(y, score, sample_weight=sample_weight))
        if pos_weight > 0 and neg_weight > 0 else np.nan
    )
    ap = (
        float(average_precision_score(y, score, sample_weight=sample_weight))
        if pos_weight > 0 else np.nan
    )
    prevalence = pos_weight / (pos_weight + neg_weight) if (pos_weight + neg_weight) > 0 else np.nan
    return auc, ap, prevalence, int(round(pos_weight)), int(round(neg_weight))


def dice_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return float((2.0 * inter + eps) / (a.sum() + b.sum() + eps))


def fraction(num: np.ndarray, den: np.ndarray) -> float:
    d = int(den.sum())
    return float(num.sum() / d) if d > 0 else np.nan


def threshold_actions(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Return [D,H,W] preserve/add/remove actions using one common action threshold."""
    p_add = probs[ADD]
    p_remove = probs[REMOVE]
    actions = np.zeros(p_add.shape, dtype=np.uint8)
    add_mask = (p_add >= threshold) & (p_add >= p_remove)
    remove_mask = (p_remove >= threshold) & (p_remove > p_add)
    actions[add_mask] = ADD
    actions[remove_mask] = REMOVE
    return actions


def apply_actions_np(autoseg: np.ndarray, actions: np.ndarray) -> np.ndarray:
    corrected = autoseg.copy().astype(np.uint8)
    corrected[actions == ADD] = 1
    corrected[actions == REMOVE] = 0
    return corrected


def weighted_score_summary(values: np.ndarray, prefix: str, thresholds, topks):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    out = {}
    if values.size == 0:
        for key in ["mean", "max", "p90", "p95", "p99", "p99_5", "p99_9"]:
            out[f"{prefix}_{key}"] = np.nan
        for thr in thresholds:
            out[f"{prefix}_frac_ge_{str(thr).replace('.', 'p')}"] = np.nan
        for k in topks:
            out[f"{prefix}_top{k}_mean"] = np.nan
        return out

    out[f"{prefix}_mean"] = float(values.mean())
    out[f"{prefix}_max"] = float(values.max())
    for q, name in [(90, "p90"), (95, "p95"), (99, "p99"), (99.5, "p99_5"), (99.9, "p99_9")]:
        out[f"{prefix}_{name}"] = float(np.percentile(values, q))
    for thr in thresholds:
        out[f"{prefix}_frac_ge_{str(thr).replace('.', 'p')}"] = float(np.mean(values >= thr))
    sorted_values = np.sort(values)
    for k in topks:
        kk = min(int(k), sorted_values.size)
        out[f"{prefix}_top{k}_mean"] = float(sorted_values[-kk:].mean()) if kk > 0 else np.nan
    return out


def largest_component_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return 0.0
    structure = ndimage.generate_binary_structure(3, 1)
    labels, nlab = ndimage.label(mask, structure=structure)
    if nlab == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / n) if sizes.size else 0.0


def exact_patient_gate_features(probs: np.ndarray, autoseg: np.ndarray, approved: np.ndarray,
                                local_radius: int, score_thresholds, topks):
    """Deployment-style patient features computed only inside autoseg-derived support.

    Feature computation uses only model probabilities and the autoseg-derived local region.
    Approved is used only below for diagnostic target/severity columns, never to define features.
    """
    local = binary_dilate(autoseg.astype(bool), local_radius)
    p_add = probs[ADD]
    p_remove = probs[REMOVE]
    p_action = p_add + p_remove  # == 1 - p_preserve for softmax output

    out = {
        "local_voxels": int(local.sum()),
        "autoseg_voxels": int(autoseg.sum()),
        "approved_voxels": int(approved.sum()),
    }
    for arr, prefix in [(p_action[local], "action"), (p_add[local], "add"), (p_remove[local], "remove")]:
        out.update(weighted_score_summary(arr, prefix, score_thresholds, topks))

    # Coherence features: a severe structured error may generate a large connected high-score region,
    # while harmless uncertainty can be scattered. These remain autoseg/input-derived.
    for thr in score_thresholds:
        tag = str(thr).replace('.', 'p')
        for arr, prefix in [(p_action, "action"), (p_add, "add"), (p_remove, "remove")]:
            high = local & (arr >= thr)
            out[f"{prefix}_n_ge_{tag}"] = int(high.sum())
            out[f"{prefix}_largest_component_frac_ge_{tag}"] = largest_component_fraction(high)

    changed = autoseg.astype(bool) ^ approved.astype(bool)
    missing = (~autoseg.astype(bool)) & approved.astype(bool)
    excess = autoseg.astype(bool) & (~approved.astype(bool))
    local_n = max(int(local.sum()), 1)
    out.update({
        "input_target_dice": dice_np(autoseg, approved),
        "dice_error": 1.0 - dice_np(autoseg, approved),
        "needed_correction_voxels": int(changed.sum()),
        "missing_voxels": int(missing.sum()),
        "excess_voxels": int(excess.sum()),
        "needed_correction_fraction_local": float(changed[local].sum() / local_n),
        "missing_fraction_local": float(missing[local].sum() / local_n),
        "excess_fraction_local": float(excess[local].sum() / local_n),
    })
    return out


def sample_scores(
    labels: np.ndarray,
    probs: np.ndarray,
    region: np.ndarray,
    patient_id: str,
    region_name: str,
    rng: np.random.Generator,
    max_preserve: int,
    max_action: int,
):
    rows = []
    flat_labels = labels.ravel()
    flat_region = region.ravel()
    flat_probs = probs.reshape(3, -1)
    for cls in (PRESERVE, ADD, REMOVE):
        idx_all = np.flatnonzero(flat_region & (flat_labels == cls))
        cap = max_preserve if cls == PRESERVE else max_action
        idx = idx_all
        if cap > 0 and len(idx_all) > cap:
            idx = rng.choice(idx_all, size=cap, replace=False)
        sample_weight = float(len(idx_all) / len(idx)) if len(idx) else 0.0
        for j in idx:
            rows.append(
                {
                    "patient_id": patient_id,
                    "region": region_name,
                    "true_class": CLASS_NAMES[cls],
                    "p_preserve": float(flat_probs[PRESERVE, j]),
                    "p_add": float(flat_probs[ADD, j]),
                    "p_remove": float(flat_probs[REMOVE, j]),
                    "sample_weight": sample_weight,
                }
            )
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--local-radius", type=int, default=6,
                    help="Dilation radius in voxels for local diagnostic regions.")
    ap.add_argument("--max-preserve-samples", type=int, default=50000,
                    help="Maximum sampled preserve voxels per patient per region.")
    ap.add_argument("--max-action-samples", type=int, default=50000,
                    help="Maximum sampled add/remove voxels per patient per region/class; <=0 keeps all.")
    ap.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument(
        "--discrimination-only", action="store_true",
        help=(
            "Compute class prevalence and sampled voxel add/remove AUC/AP only. "
            "Skip patient gate features, threshold sweeps, and writing sampled_voxel_scores.csv. "
            "Useful for lightweight cross-fold training-quality diagnostics."
        ),
    )
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--gate-score-thresholds", default="0.02,0.05,0.10,0.15,0.20,0.30",
                    help="Thresholds for exact patient-level deployment-style action-score features.")
    ap.add_argument("--gate-topks", default="10,50,100,500,1000",
                    help="Top-k local voxels used for patient-level action-score summaries.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(
        cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )
    model, checkpoint = make_model(cfg, args.checkpoint, device)
    rows = read_rows(cfg["manifest"], args.split)
    if not rows:
        raise RuntimeError(f"No rows for split={args.split} in {cfg['manifest']}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]
    gate_score_thresholds = [float(v) for v in args.gate_score_thresholds.split(",") if v.strip()]
    gate_topks = [int(v) for v in args.gate_topks.split(",") if v.strip()]
    rng = np.random.default_rng(args.seed)

    prevalence_rows = []
    sampled_rows = []
    threshold_rows = []
    patient_gate_rows = []

    for case_index, row in enumerate(rows, 1):
        x = read_binary(row["autoseg"], device)
        target = read_binary(row["approved"], device)
        logits, _ = model(x)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        autoseg = (x[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        approved = (target[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        labels = action_labels_np(autoseg, approved)
        regions = make_regions(autoseg, approved, args.local_radius)

        if not args.discrimination_only:
            gate_row = {"patient_id": row["patient_id"], "split": args.split}
            gate_row.update(exact_patient_gate_features(
                probs, autoseg, approved, args.local_radius, gate_score_thresholds, gate_topks
            ))
            patient_gate_rows.append(gate_row)

        for region_name, region in regions.items():
            n_region = int(region.sum())
            counts = {cls: int(np.logical_and(region, labels == cls).sum()) for cls in (PRESERVE, ADD, REMOVE)}
            prevalence_rows.append({
                "patient_id": row["patient_id"],
                "split": args.split,
                "region": region_name,
                "n_voxels": n_region,
                "preserve_voxels": counts[PRESERVE],
                "add_voxels": counts[ADD],
                "remove_voxels": counts[REMOVE],
                "preserve_fraction": counts[PRESERVE] / n_region if n_region else np.nan,
                "add_fraction": counts[ADD] / n_region if n_region else np.nan,
                "remove_fraction": counts[REMOVE] / n_region if n_region else np.nan,
            })

            sampled_rows.extend(sample_scores(
                labels, probs, region, str(row["patient_id"]), region_name, rng,
                args.max_preserve_samples, args.max_action_samples,
            ))

        input_dice = dice_np(autoseg, approved)
        if not args.discrimination_only:
            changed = labels != PRESERVE
            missing = labels == ADD
            excess = labels == REMOVE
            unchanged = labels == PRESERVE
            needed_count = int(changed.sum())

            for threshold in thresholds:
                actions = threshold_actions(probs, threshold)
                corrected = apply_actions_np(autoseg, actions)
                pred_changed = actions != PRESERVE
                corrected_dice = dice_np(corrected, approved)
                threshold_rows.append({
                    "patient_id": row["patient_id"],
                    "split": args.split,
                    "threshold": threshold,
                    "input_target_dice": input_dice,
                    "corrected_target_dice": corrected_dice,
                    "dice_gain": corrected_dice - input_dice,
                    "needed_correction_voxels": needed_count,
                    "predicted_correction_voxels": int(pred_changed.sum()),
                    "correction_volume_ratio_to_needed": (
                        float(pred_changed.sum() / needed_count) if needed_count > 0 else np.nan
                    ),
                    "needed_correction_recall": fraction(pred_changed & changed, changed),
                    "missing_correction_recall": fraction((actions == ADD) & missing, missing),
                    "excess_correction_recall": fraction((actions == REMOVE) & excess, excess),
                    "correction_precision": fraction(pred_changed & changed, pred_changed),
                    "unnecessary_change_fraction": fraction(pred_changed & unchanged, unchanged),
                    "predicted_add_fraction": float(np.mean(actions == ADD)),
                    "predicted_remove_fraction": float(np.mean(actions == REMOVE)),
                })

        print(
            f"[{case_index}/{len(rows)}] patient={row['patient_id']} "
            f"inputDice={input_dice:.4f} add={int((labels == ADD).sum())} "
            f"remove={int((labels == REMOVE).sum())}",
            flush=True,
        )

    prevalence = pd.DataFrame(prevalence_rows)
    prevalence.to_csv(output / "class_prevalence_by_patient_region.csv", index=False)

    prevalence_summary = (
        prevalence.groupby("region", as_index=False)[
            ["n_voxels", "preserve_voxels", "add_voxels", "remove_voxels"]
        ].sum()
    )
    total = prevalence_summary["n_voxels"].replace(0, np.nan)
    prevalence_summary["preserve_fraction"] = prevalence_summary["preserve_voxels"] / total
    prevalence_summary["add_fraction"] = prevalence_summary["add_voxels"] / total
    prevalence_summary["remove_fraction"] = prevalence_summary["remove_voxels"] / total
    prevalence_summary.to_csv(output / "class_prevalence_summary.csv", index=False)

    sampled_df = pd.DataFrame(sampled_rows)
    if not args.discrimination_only:
        sampled_df.to_csv(output / "sampled_voxel_scores.csv", index=False)

    metric_rows = []
    for region_name in sorted(sampled_df["region"].unique()):
        sub = sampled_df[sampled_df["region"] == region_name]
        weights = sub["sample_weight"].to_numpy(float)
        for action_name, true_name, score_key in (("add", "add", "p_add"), ("remove", "remove", "p_remove")):
            scores = sub[score_key].to_numpy(float)
            y = (sub["true_class"].to_numpy() == true_name).astype(np.uint8)
            auc, apv, prev, n_pos, n_neg = safe_auc_ap(y, scores, weights)
            metric_rows.append({
                "region": region_name,
                "action": action_name,
                "roc_auc": auc,
                "average_precision": apv,
                "positive_prevalence": prev,
                "ap_over_prevalence": apv / prev if np.isfinite(apv) and prev > 0 else np.nan,
                "n_positive": n_pos,
                "n_negative": n_neg,
            })
    pd.DataFrame(metric_rows).to_csv(output / "action_score_discrimination.csv", index=False)

    if not args.discrimination_only:
        patient_gate_df = pd.DataFrame(patient_gate_rows)
        patient_gate_df.to_csv(output / "patient_action_score_summary.csv", index=False)

        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df.to_csv(output / "threshold_sweep_by_patient.csv", index=False)
        metric_cols = [
            "dice_gain", "correction_volume_ratio_to_needed", "needed_correction_recall",
            "missing_correction_recall", "excess_correction_recall", "correction_precision",
            "unnecessary_change_fraction", "predicted_add_fraction", "predicted_remove_fraction",
        ]
        threshold_summary = threshold_df.groupby("threshold", as_index=False)[metric_cols].agg(["mean", "median"])
        threshold_summary.columns = [
            "threshold" if c[0] == "threshold" else f"{c[0]}_{c[1]}" for c in threshold_summary.columns
        ]
        threshold_summary.to_csv(output / "threshold_sweep_summary.csv", index=False)

    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint),
        "split": args.split,
        "n_cases": len(rows),
        "local_radius_voxels": args.local_radius,
        "thresholds": thresholds if not args.discrimination_only else [],
        "discrimination_only": bool(args.discrimination_only),
        "patient_gate_features": {
            "region": "autoseg_local_only",
            "score_thresholds": gate_score_thresholds,
            "topks": gate_topks,
            "approved_used_to_define_gate_features": False,
            "approved_used_only_for_diagnostic_severity_targets": True,
        },
        "sampling": {
            "max_preserve_per_patient_region": args.max_preserve_samples,
            "max_action_per_patient_region_class": args.max_action_samples,
            "inverse_sampling_weights_used_for_auc_ap": True,
            "seed": args.seed,
        },
        "interpretation_note": (
            "oracle_union_local and oracle_change_local use approved contours and are diagnostic-only; "
            "autoseg_local is candidate-derived and deployment-feasible."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nClass prevalence summary:")
    print(prevalence_summary.to_string(index=False))
    print("\nAction-score discrimination:")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"\nSaved diagnostic outputs to: {output}")


if __name__ == "__main__":
    main()
