#!/usr/bin/env python3
"""Fuse learned spatial action-map embeddings with compact handcrafted soft-QA features.

Readout-only experiment on a frozen correction model representation.

Inputs
------
* 12-channel low-resolution spatial action tensor from
  ``extract_supervised_correction_spatial_soft_qa_tensors``.
* 18 compact deployment-feasible soft summary features (9 autoseg-local +
  9 autoseg-boundary) from ``extract_supervised_correction_soft_qa_features``.

The fusion head predicts true 1-Dice, HD95, and ASSD jointly. Its continuous
predictions are also used as binary QA discriminator scores.

Leakage control
---------------
* Development = train + validation; test is untouched until final scoring.
* Development OOF folds each contain an inner fit/early-stop split.
* Compact-feature standardization and target standardization are fit only on
  the corresponding fit partition.
* The median best epoch across OOF folds is used to refit one final model on
  all development patients before a single test evaluation.
* Bad-case cutoffs and ~95%-specificity operating thresholds use development
  outcomes / OOF predictions only.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analysis.analyze_supervised_correction_qa_signal import (
    average_precision,
    bootstrap_classification,
    choose_threshold_at_specificity,
    confusion,
    percentile_higher,
    roc_auc,
)

TARGETS = [
    ("1_minus_dice", "true_1_minus_dice"),
    ("hd95", "true_hd95_mm"),
    ("assd", "true_assd_mm"),
]

COMPACT_SUFFIXES = [
    "action_mean", "action_p95", "action_p99", "action_top100_mean", "action_frac_ge_0p1",
    "add_p95", "add_p99", "remove_p95", "remove_p99",
]
COMPACT_FEATURES = [f"{region}_{suffix}" for region in ("local", "boundary") for suffix in COMPACT_SUFFIXES]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpatialSoftFusionHead(nn.Module):
    """Tiny spatial trunk + compact soft summaries -> joint 3-target regression."""

    def __init__(self, spatial_channels: int, compact_dim: int, base_channels: int = 8, dropout: float = 0.25):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv3d(spatial_channels, c, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, c // 4), c),
            nn.GELU(),
            nn.Conv3d(c, 2 * c, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, (2 * c) // 4), 2 * c),
            nn.GELU(),
            nn.Conv3d(2 * c, 3 * c, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, (3 * c) // 4), 3 * c),
            nn.GELU(),
        )
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.mx = nn.AdaptiveMaxPool3d(1)
        spatial_dim = 6 * c
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + compact_dim, 6 * c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(6 * c, 4 * c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * c, len(TARGETS)),
        )

    def spatial_embedding(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return torch.cat([self.avg(z).flatten(1), self.mx(z).flatten(1)], dim=1)

    def forward(self, spatial: torch.Tensor, compact: torch.Tensor) -> torch.Tensor:
        z = self.spatial_embedding(spatial)
        return self.fusion(torch.cat([z, compact], dim=1))


def load_split(
    tensor_root: Path,
    evaluation_dir: Path,
    soft_root: Path,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_path = tensor_root / split / "index.csv"
    ev_path = evaluation_dir / f"{split}_metrics.csv"
    soft_path = soft_root / split / "patient_action_score_summary.csv"
    for p in (idx_path, ev_path, soft_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    idx = pd.read_csv(idx_path, dtype={"patient_id": str})
    ev = pd.read_csv(ev_path, dtype={"patient_id": str})
    soft = pd.read_csv(soft_path, dtype={"patient_id": str})
    target_cols = [c for _, c in TARGETS]
    missing = [c for c in COMPACT_FEATURES if c not in soft.columns]
    if missing:
        raise KeyError(f"Missing compact soft-QA features in {soft_path}: {missing}")

    merged = idx.merge(ev[["patient_id", *target_cols]], on="patient_id", how="inner", validate="one_to_one")
    merged = merged.merge(soft[["patient_id", *COMPACT_FEATURES]], on="patient_id", how="inner", validate="one_to_one")
    if len(merged) != len(idx):
        raise RuntimeError(f"Merge lost cases for {split}: tensors={len(idx)}, merged={len(merged)}")

    spatial = []
    for row in merged.itertuples(index=False):
        spatial.append(np.load(tensor_root / split / row.path)["spatial"].astype(np.float32))
    Xs = np.stack(spatial)
    Xc = merged[COMPACT_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    y = merged[target_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    finite = np.isfinite(Xc).all(axis=1) & np.isfinite(y).all(axis=1)
    if not finite.all():
        print(f"Dropping {(~finite).sum()} non-finite rows from {split}")
    return merged.loc[finite, "patient_id"].astype(str).to_numpy(), Xs[finite], Xc[finite], y[finite]


def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def make_loader(Xs: np.ndarray, Xc: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(Xc), torch.from_numpy(y))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle,
                      generator=gen if shuffle else None, num_workers=0)


@torch.no_grad()
def predict(model: nn.Module, Xs: np.ndarray, Xc: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(Xc))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False, num_workers=0)
    out = []
    for sb, cb in loader:
        pred = model(sb.to(device=device, dtype=torch.float32), cb.to(device=device, dtype=torch.float32))
        out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0)


def inner_split(n: int, seed: int, fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise RuntimeError("Need at least two patients for early-stop split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_stop = min(max(1, int(round(n * fraction))), n - 1)
    return perm[n_stop:], perm[:n_stop]


def eval_loss(model: nn.Module, Xs: np.ndarray, Xc: np.ndarray, yz: np.ndarray, device: torch.device, batch_size: int) -> float:
    pred = predict(model, Xs, Xc, device, batch_size)
    return float(nn.functional.smooth_l1_loss(torch.from_numpy(pred), torch.from_numpy(yz), beta=0.5))


def train_earlystop(
    Xs_fit: np.ndarray, Xc_fit_raw: np.ndarray, y_fit_raw: np.ndarray,
    Xs_stop: np.ndarray, Xc_stop_raw: np.ndarray, y_stop_raw: np.ndarray,
    *, device: torch.device, seed: int, max_epochs: int, patience: int, min_delta: float,
    batch_size: int, lr: float, weight_decay: float, base_channels: int, dropout: float,
) -> Tuple[SpatialSoftFusionHead, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[Dict[str, float]]]:
    set_seed(seed)
    cm, cs = fit_standardizer(Xc_fit_raw)
    ym, ys = fit_standardizer(y_fit_raw)
    Xc_fit = apply_standardizer(Xc_fit_raw, cm, cs)
    Xc_stop = apply_standardizer(Xc_stop_raw, cm, cs)
    yz_fit = apply_standardizer(y_fit_raw, ym, ys)
    yz_stop = apply_standardizer(y_stop_raw, ym, ys)

    model = SpatialSoftFusionHead(Xs_fit.shape[1], Xc_fit.shape[1], base_channels, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    loader = make_loader(Xs_fit, Xc_fit, yz_fit, batch_size, shuffle=True, seed=seed)
    best_loss = np.inf
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0; count = 0
        for sb, cb, yb in loader:
            sb = sb.to(device=device, dtype=torch.float32)
            cb = cb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(sb, cb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(sb); count += len(sb)
        train_loss = total / max(count, 1)
        stop_loss = eval_loss(model, Xs_stop, Xc_stop, yz_stop, device, batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss, "earlystop_loss": stop_loss})
        if stop_loss < best_loss - min_delta:
            best_loss = stop_loss; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, cm, cs, ym, ys, best_epoch, history


def train_fixed(
    Xs: np.ndarray, Xc_raw: np.ndarray, y_raw: np.ndarray,
    *, epochs: int, device: torch.device, seed: int, batch_size: int, lr: float,
    weight_decay: float, base_channels: int, dropout: float,
) -> Tuple[SpatialSoftFusionHead, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    set_seed(seed)
    cm, cs = fit_standardizer(Xc_raw); ym, ys = fit_standardizer(y_raw)
    Xc = apply_standardizer(Xc_raw, cm, cs); yz = apply_standardizer(y_raw, ym, ys)
    model = SpatialSoftFusionHead(Xs.shape[1], Xc.shape[1], base_channels, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    loader = make_loader(Xs, Xc, yz, batch_size, shuffle=True, seed=seed)
    losses = []
    for _ in range(max(1, epochs)):
        model.train(); total = 0.0; count = 0
        for sb, cb, yb in loader:
            sb = sb.to(device=device, dtype=torch.float32); cb = cb.to(device=device, dtype=torch.float32); yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(sb, cb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            total += float(loss.detach()) * len(sb); count += len(sb)
        losses.append(total / max(count, 1))
    return model, cm, cs, ym, ys, losses


def unstandardize(predz: np.ndarray, ym: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return predz * ys + ym


def reg_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    resid = y - pred
    sse = float(np.sum(resid ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    pearson = stats.pearsonr(pred, y).statistic if len(y) >= 2 and np.std(pred) > 0 and np.std(y) > 0 else np.nan
    spearman = stats.spearmanr(pred, y).statistic if len(y) >= 2 and np.std(pred) > 0 and np.std(y) > 0 else np.nan
    return {
        "n": int(len(y)), "mae": float(np.mean(np.abs(resid))), "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": float(1 - sse / sst) if sst > 0 else np.nan,
        "pearson_r": float(pearson) if np.isfinite(pearson) else np.nan,
        "spearman_rho": float(spearman) if np.isfinite(spearman) else np.nan,
    }


def bootstrap_reg(y: np.ndarray, pred: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed); maes = []; rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y)); m = reg_metrics(y[idx], pred[idx])
        if np.isfinite(m["mae"]): maes.append(m["mae"])
        if np.isfinite(m["spearman_rho"]): rhos.append(m["spearman_rho"])
    def ci(v): return (np.nan, np.nan) if not v else tuple(float(x) for x in np.percentile(v, [2.5, 97.5]))
    ml, mh = ci(maes); sl, sh = ci(rhos)
    return {"mae_low": ml, "mae_high": mh, "spearman_low": sl, "spearman_high": sh, "valid": min(len(maes), len(rhos))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-root", required=True)
    ap.add_argument("--soft-feature-root", required=True)
    ap.add_argument("--evaluation-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--max-epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--earlystop-fraction", type=float, default=0.20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--base-channels", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--bad-percentiles", nargs="+", type=float, default=[75.0, 50.0])
    ap.add_argument("--target-specificity", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    tensor_root = Path(args.tensor_root); soft_root = Path(args.soft_feature_root); evaluation_dir = Path(args.evaluation_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pid_tr, Xs_tr, Xc_tr, y_tr = load_split(tensor_root, evaluation_dir, soft_root, "train")
    pid_va, Xs_va, Xc_va, y_va = load_split(tensor_root, evaluation_dir, soft_root, "validation")
    pid_te, Xs_te, Xc_te, y_te = load_split(tensor_root, evaluation_dir, soft_root, "test")
    pid_dev = np.concatenate([pid_tr, pid_va]); Xs_dev = np.concatenate([Xs_tr, Xs_va]); Xc_dev = np.concatenate([Xc_tr, Xc_va]); y_dev = np.concatenate([y_tr, y_va])
    print(f"Loaded dev={len(pid_dev)} test={len(pid_te)} spatial={Xs_dev.shape[1:]} compact={Xc_dev.shape[1]} device={device}")

    kf = KFold(n_splits=min(args.cv_folds, len(pid_dev)), shuffle=True, random_state=args.seed)
    oof = np.full_like(y_dev, np.nan, dtype=np.float32)
    cv_rows = []
    best_epochs = []
    for fold, (outer_train, held) in enumerate(kf.split(Xs_dev)):
        fit_rel, stop_rel = inner_split(len(outer_train), args.seed + fold * 101, args.earlystop_fraction)
        fit_idx = outer_train[fit_rel]; stop_idx = outer_train[stop_rel]
        print(f"OOF fold {fold+1}: fit={len(fit_idx)} stop={len(stop_idx)} held={len(held)}", flush=True)
        model, cm, cs, ym, ys, best_epoch, hist = train_earlystop(
            Xs_dev[fit_idx], Xc_dev[fit_idx], y_dev[fit_idx], Xs_dev[stop_idx], Xc_dev[stop_idx], y_dev[stop_idx],
            device=device, seed=args.seed + fold * 1009, max_epochs=args.max_epochs, patience=args.patience,
            min_delta=args.min_delta, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            base_channels=args.base_channels, dropout=args.dropout,
        )
        held_compact = apply_standardizer(Xc_dev[held], cm, cs)
        oof[held] = unstandardize(predict(model, Xs_dev[held], held_compact, device, args.batch_size), ym, ys)
        best_epochs.append(best_epoch)
        cv_rows.append({"fold": fold, "n_fit": len(fit_idx), "n_earlystop": len(stop_idx), "n_heldout": len(held),
                        "best_epoch": best_epoch, "best_earlystop_loss": min(r["earlystop_loss"] for r in hist)})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if not np.isfinite(oof).all(): raise RuntimeError("OOF fusion predictions contain non-finite values")

    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    print(f"Refitting final fusion head on all development patients for {final_epochs} epochs", flush=True)
    model, cm, cs, ym, ys, losses = train_fixed(
        Xs_dev, Xc_dev, y_dev, epochs=final_epochs, device=device, seed=args.seed + 9999,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        base_channels=args.base_channels, dropout=args.dropout,
    )
    Xc_te_z = apply_standardizer(Xc_te, cm, cs)
    test_pred = unstandardize(predict(model, Xs_te, Xc_te_z, device, args.batch_size), ym, ys)
    torch.save({"model_state_dict": model.state_dict(), "compact_mean": cm, "compact_std": cs,
                "target_mean": ym, "target_std": ys, "final_epochs": final_epochs,
                "compact_features": COMPACT_FEATURES}, model_dir / "fusion_head.pth")

    reg_rows = []; cls_rows = []; pred_rows = []
    for j, (metric, target_col) in enumerate(TARGETS):
        rm = reg_metrics(y_te[:, j], test_pred[:, j]); rb = bootstrap_reg(y_te[:, j], test_pred[:, j], args.n_boot, args.seed + 500 + j)
        reg_rows.append({"roi": args.roi, "variant": args.variant, "metric": metric, "readout": "spatial_plus_compact_soft",
                         **rm, **rb, "final_epochs": final_epochs})
        for pct in args.bad_percentiles:
            cutoff = percentile_higher(pd.Series(y_dev[:, j]), pct)
            ydev_bin = (y_dev[:, j] > cutoff).astype(int); ytest_bin = (y_te[:, j] > cutoff).astype(int)
            dev_score = oof[:, j]; test_score = test_pred[:, j]
            thr, dev_sens, dev_spec = choose_threshold_at_specificity(ydev_bin, dev_score, args.target_specificity)
            cm_test = confusion(ytest_bin, test_score, thr)
            boot = bootstrap_classification(ytest_bin, test_score, args.n_boot, args.seed + 1000 + j * 10 + int(pct))
            cls_rows.append({"roi": args.roi, "variant": args.variant, "metric": metric, "readout": "spatial_plus_compact_soft",
                             "bad_percentile": pct, "true_error_cutoff": cutoff, "n_test": len(ytest_bin),
                             "n_test_bad": int(ytest_bin.sum()), "roc_auc": roc_auc(ytest_bin, test_score),
                             "pr_auc": average_precision(ytest_bin, test_score), **boot,
                             "dev_operating_threshold": thr, "dev_sensitivity": dev_sens, "dev_specificity": dev_spec,
                             **{f"test_{k}": v for k, v in cm_test.items()}})
        for pid, yt, yp in zip(pid_te, y_te[:, j], test_pred[:, j]):
            pred_rows.append({"patient_id": pid, "metric": metric, "true_error": float(yt), "predicted_error": float(yp),
                              "readout": "spatial_plus_compact_soft"})

    pd.DataFrame(reg_rows).to_csv(out / "fusion_qa_regression_summary.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(out / "fusion_qa_classification_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "fusion_qa_test_predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "fusion_qa_earlystop_summary.csv", index=False)
    pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "train_loss": losses}).to_csv(out / "fusion_qa_final_training_curve.csv", index=False)
    protocol = {
        "roi": args.roi, "variant": args.variant, "readout": "spatial_plus_compact_soft",
        "development": "train+validation", "test_used_for_selection": False,
        "compact_features": COMPACT_FEATURES, "n_compact_features": len(COMPACT_FEATURES),
        "spatial_tensor_shape": list(Xs_dev.shape[1:]), "joint_targets": [m for m, _ in TARGETS],
        "cv_folds": min(args.cv_folds, len(pid_dev)), "inner_earlystop_fraction": args.earlystop_fraction,
        "final_epochs_from_median_oof_best_epoch": final_epochs, "bad_percentiles": args.bad_percentiles,
        "target_specificity": args.target_specificity,
    }
    (out / "fusion_qa_protocol.json").write_text(json.dumps(protocol, indent=2))
    print(pd.DataFrame(reg_rows).to_string(index=False))
    print(pd.DataFrame(cls_rows).to_string(index=False))


if __name__ == "__main__":
    main()
