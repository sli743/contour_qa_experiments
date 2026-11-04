#!/usr/bin/env python3
"""Single-target spatial QA heads with early stopping and direct classification.

This is a readout-only ablation on frozen spatial action-map tensors produced by
``extract_supervised_correction_spatial_soft_qa_tensors``.

Compared with the original joint 3-target regression head, this experiment tests:
  1) one regression head per QA metric;
  2) one direct binary head per metric and bad-case percentile;
  3) development-only early stopping / best-checkpoint selection.

Leakage control
---------------
* Development = train + validation. Test remains untouched until final scoring.
* For each development OOF fold, the fold's training partition is further split
  into fit/early-stop subsets. The best epoch is selected only from that inner
  early-stop subset.
* The median best epoch across OOF folds is then used to refit a final model on
  all development patients before one test evaluation.
* Binary bad-case cutoffs and operating thresholds are defined from development
  data only. The direct classifier's test probability is the QA discriminator.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analysis.analyze_supervised_correction_qa_signal import (
    bootstrap_classification,
    bootstrap_regression,
    choose_threshold_at_specificity,
    confusion,
    percentile_higher,
    regression_metrics,
)

TARGETS = [
    ("1_minus_dice", "true_1_minus_dice"),
    ("hd95", "true_hd95_mm"),
    ("assd", "true_assd_mm"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinySpatialScalarHead(nn.Module):
    """Same small spatial trunk as v1, but one scalar output per task."""

    def __init__(self, in_channels: int, base_channels: int = 8, dropout: float = 0.20):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, c, 3, padding=1, bias=False),
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
        self.head = nn.Sequential(
            nn.Linear(6 * c, 4 * c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * c, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = torch.cat([self.avg(z).flatten(1), self.mx(z).flatten(1)], dim=1)
        return self.head(z).squeeze(1)


def load_split(tensor_root: Path, evaluation_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_path = tensor_root / split / "index.csv"
    ev_path = evaluation_dir / f"{split}_metrics.csv"
    if not idx_path.is_file():
        raise FileNotFoundError(idx_path)
    if not ev_path.is_file():
        raise FileNotFoundError(ev_path)
    idx = pd.read_csv(idx_path, dtype={"patient_id": str})
    ev = pd.read_csv(ev_path, dtype={"patient_id": str})
    cols = [c for _, c in TARGETS]
    merged = idx.merge(ev[["patient_id", *cols]], on="patient_id", how="inner", validate="one_to_one")
    if len(merged) != len(idx):
        raise RuntimeError(f"Target merge lost cases for {split}: tensors={len(idx)}, merged={len(merged)}")
    X: List[np.ndarray] = []
    for row in merged.itertuples(index=False):
        X.append(np.load(tensor_root / split / row.path)["spatial"].astype(np.float32))
    Xn = np.stack(X)
    y = merged[cols].to_numpy(dtype=np.float32)
    finite = np.isfinite(y).all(axis=1)
    return merged.loc[finite, "patient_id"].astype(str).to_numpy(), Xn[finite], y[finite]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=0,
    )


@torch.no_grad()
def scalar_predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False, num_workers=0)
    out = []
    for (xb,) in loader:
        out.append(model(xb.to(device=device, dtype=torch.float32)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def inner_regression_split(n: int, seed: int, val_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = min(max(1, int(round(n * val_fraction))), n - 1)
    return perm[n_val:], perm[:n_val]


def inner_classification_split(y: np.ndarray, seed: int, val_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    # Prefer stratification when both classes have enough examples; otherwise fall back.
    counts = np.bincount(y, minlength=2)
    if counts.min() >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        fit_idx, val_idx = next(sss.split(np.zeros(len(y)), y))
        return fit_idx, val_idx
    return inner_regression_split(len(y), seed, val_fraction)


def _eval_regression_loss(model: nn.Module, X: np.ndarray, yz: np.ndarray, device: torch.device, batch_size: int) -> float:
    pred = scalar_predict(model, X, device, batch_size)
    return float(nn.functional.smooth_l1_loss(torch.from_numpy(pred), torch.from_numpy(yz.astype(np.float64)), beta=0.5))


def train_regression_earlystop(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_stop: np.ndarray,
    y_stop: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    base_channels: int,
    dropout: float,
) -> Tuple[TinySpatialScalarHead, float, float, int, List[Dict[str, float]]]:
    set_seed(seed)
    mean = float(np.mean(y_fit))
    std = float(np.std(y_fit))
    if std <= 1e-8:
        std = 1.0
    y_fit_z = ((y_fit - mean) / std).astype(np.float32)
    y_stop_z = ((y_stop - mean) / std).astype(np.float32)
    loader = make_loader(X_fit, y_fit_z, batch_size, shuffle=True, seed=seed)
    model = TinySpatialScalarHead(X_fit.shape[1], base_channels, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    best_loss = np.inf
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        train_loss = total / max(count, 1)
        stop_loss = _eval_regression_loss(model, X_stop, y_stop_z, device, batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss, "earlystop_loss": stop_loss})
        if stop_loss < best_loss - min_delta:
            best_loss = stop_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return model, mean, std, best_epoch, history


def train_regression_fixed_epochs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
    seed: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    base_channels: int,
    dropout: float,
) -> Tuple[TinySpatialScalarHead, float, float, List[float]]:
    set_seed(seed)
    mean = float(np.mean(y)); std = float(np.std(y))
    if std <= 1e-8: std = 1.0
    yz = ((y - mean) / std).astype(np.float32)
    loader = make_loader(X, yz, batch_size, shuffle=True, seed=seed)
    model = TinySpatialScalarHead(X.shape[1], base_channels, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    losses = []
    for _ in range(max(1, epochs)):
        model.train(); total = 0.0; count = 0
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32); yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            total += float(loss.detach()) * len(xb); count += len(xb)
        losses.append(total / max(count, 1))
    return model, mean, std, losses


def train_classifier_earlystop(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_stop: np.ndarray,
    y_stop: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    base_channels: int,
    dropout: float,
) -> Tuple[TinySpatialScalarHead, int, List[Dict[str, float]]]:
    set_seed(seed)
    loader = make_loader(X_fit, y_fit, batch_size, shuffle=True, seed=seed)
    model = TinySpatialScalarHead(X_fit.shape[1], base_channels, dropout).to(device)
    n_pos = max(int(np.sum(y_fit == 1)), 1); n_neg = max(int(np.sum(y_fit == 0)), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = np.inf; best_epoch = 1; bad_epochs = 0
    best_state = copy.deepcopy(model.state_dict()); history: List[Dict[str, float]] = []
    stop_y = torch.from_numpy(y_stop.astype(np.float32)).to(device)
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0; count = 0
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32); yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            total += float(loss.detach()) * len(xb); count += len(xb)
        model.eval()
        with torch.no_grad():
            stop_logits = torch.from_numpy(scalar_predict(model, X_stop, device, batch_size)).to(device=device, dtype=torch.float32)
            stop_loss = float(loss_fn(stop_logits, stop_y).cpu())
        train_loss = total / max(count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "earlystop_loss": stop_loss})
        if stop_loss < best_loss - min_delta:
            best_loss = stop_loss; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict()); bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, history


def train_classifier_fixed_epochs(
    X: np.ndarray,
    y: np.ndarray,
    *, epochs: int, device: torch.device, seed: int, batch_size: int, lr: float,
    weight_decay: float, base_channels: int, dropout: float,
) -> Tuple[TinySpatialScalarHead, List[float]]:
    set_seed(seed)
    loader = make_loader(X, y, batch_size, shuffle=True, seed=seed)
    model = TinySpatialScalarHead(X.shape[1], base_channels, dropout).to(device)
    n_pos = max(int(np.sum(y == 1)), 1); n_neg = max(int(np.sum(y == 0)), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(max(1, epochs)):
        model.train(); total = 0.0; count = 0
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32); yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            total += float(loss.detach()) * len(xb); count += len(xb)
        losses.append(total / max(count, 1))
    return model, losses


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-root", required=True)
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
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--bad-percentiles", nargs="+", type=float, default=[75.0, 50.0])
    ap.add_argument("--target-specificity", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    tensor_root = Path(args.tensor_root); evaluation_dir = Path(args.evaluation_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pid_tr, Xtr, ytr = load_split(tensor_root, evaluation_dir, "train")
    pid_va, Xva, yva = load_split(tensor_root, evaluation_dir, "validation")
    pid_te, Xte, yte = load_split(tensor_root, evaluation_dir, "test")
    pid_dev = np.concatenate([pid_tr, pid_va]); Xdev = np.concatenate([Xtr, Xva]); ydev = np.concatenate([ytr, yva])
    nfold = min(args.cv_folds, len(Xdev))
    if nfold < 2: raise RuntimeError("Need at least two development patients")
    print(f"Loaded dev={len(Xdev)} test={len(Xte)} shape={Xdev.shape[1:]} device={device}")

    reg_rows: List[Dict[str, object]] = []
    cls_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    epoch_rows: List[Dict[str, object]] = []
    kf = KFold(n_splits=nfold, shuffle=True, random_state=args.seed)
    outer_splits = list(kf.split(Xdev))

    # ---- Single-target regression heads ----
    for j, (metric, _) in enumerate(TARGETS):
        print(f"\n=== single-target regression: {metric} ===", flush=True)
        oof = np.full(len(Xdev), np.nan, dtype=np.float64)
        best_epochs = []
        for fold, (outer_fit, outer_held) in enumerate(outer_splits):
            fit_local, stop_local = inner_regression_split(len(outer_fit), args.seed + 1000*j + fold, args.earlystop_fraction)
            fit_idx = outer_fit[fit_local]; stop_idx = outer_fit[stop_local]
            model, ym, ys, best_epoch, hist = train_regression_earlystop(
                Xdev[fit_idx], ydev[fit_idx, j], Xdev[stop_idx], ydev[stop_idx, j],
                device=device, seed=args.seed + j*1000 + fold*37, max_epochs=args.max_epochs,
                patience=args.patience, min_delta=args.min_delta, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay, base_channels=args.base_channels, dropout=args.dropout,
            )
            oof[outer_held] = scalar_predict(model, Xdev[outer_held], device, args.batch_size) * ys + ym
            best_epochs.append(best_epoch)
            epoch_rows.append({"task":"regression", "target_metric":metric, "bad_percentile":np.nan,
                               "fold":fold, "best_epoch":best_epoch, "epochs_ran":len(hist),
                               "best_earlystop_loss":min(r["earlystop_loss"] for r in hist)})
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        final_epochs = max(1, int(round(float(np.median(best_epochs)))))
        model, ym, ys, losses = train_regression_fixed_epochs(
            Xdev, ydev[:,j], epochs=final_epochs, device=device, seed=args.seed + 9000 + j,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            base_channels=args.base_channels, dropout=args.dropout,
        )
        test_pred = scalar_predict(model, Xte, device, args.batch_size) * ys + ym
        torch.save({"model_state_dict":model.state_dict(), "target_metric":metric, "final_epochs":final_epochs,
                    "target_mean":ym, "target_std":ys, "in_channels":int(Xdev.shape[1]),
                    "base_channels":args.base_channels, "dropout":args.dropout}, model_dir / f"regression_{metric}.pth")
        tmp = pd.DataFrame({"x":test_pred, "y":yte[:,j]})
        rm = regression_metrics(tmp, "x", "y", 1.0, 0.0)
        rb = bootstrap_regression(tmp, "x", "y", 1.0, 0.0, args.n_boot, args.seed + 300 + j)
        reg_rows.append({"roi":args.roi, "variant":args.variant, "target_metric":metric,
                         "feature_group":"single_target_spatial_regression_head", "development_n":len(Xdev), "test_n":len(Xte),
                         "final_epochs_from_median_oof_best":final_epochs,
                         "test_mae":rm["mae"], "test_mae_ci95_low":rb["mae_low"], "test_mae_ci95_high":rb["mae_high"],
                         "test_rmse":rm["rmse"], "test_r2":rm["r2"], "test_pearson_r":rm["pearson_r"],
                         "test_spearman_rho":rm["spearman_rho"], "test_spearman_ci95_low":rb["spearman_low"],
                         "test_spearman_ci95_high":rb["spearman_high"]})
        for pid, yt, yp, op in zip(pid_te, yte[:,j], test_pred, np.interp(np.arange(len(Xte)), np.arange(len(Xte)), test_pred)):
            pred_rows.append({"patient_id":pid, "target_metric":metric, "task":"single_target_regression",
                              "true_value":float(yt), "score_or_prediction":float(yp)})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Direct binary heads ----
    for j, (metric, _) in enumerate(TARGETS):
        for pct in args.bad_percentiles:
            print(f"\n=== direct classification: {metric} P{pct:g} ===", flush=True)
            cutoff = percentile_higher(pd.Series(ydev[:,j]), pct)
            ydev_cls = (ydev[:,j] > cutoff).astype(np.int64)
            ytest_cls = (yte[:,j] > cutoff).astype(np.int64)
            oof_prob = np.full(len(Xdev), np.nan, dtype=np.float64)
            best_epochs = []
            for fold, (outer_fit, outer_held) in enumerate(outer_splits):
                fit_local, stop_local = inner_classification_split(ydev_cls[outer_fit], args.seed + 5000 + j*100 + fold + int(pct), args.earlystop_fraction)
                fit_idx = outer_fit[fit_local]; stop_idx = outer_fit[stop_local]
                model, best_epoch, hist = train_classifier_earlystop(
                    Xdev[fit_idx], ydev_cls[fit_idx], Xdev[stop_idx], ydev_cls[stop_idx],
                    device=device, seed=args.seed + 7000 + j*1000 + fold*41 + int(pct), max_epochs=args.max_epochs,
                    patience=args.patience, min_delta=args.min_delta, batch_size=args.batch_size,
                    lr=args.lr, weight_decay=args.weight_decay, base_channels=args.base_channels, dropout=args.dropout,
                )
                oof_prob[outer_held] = sigmoid_np(scalar_predict(model, Xdev[outer_held], device, args.batch_size))
                best_epochs.append(best_epoch)
                epoch_rows.append({"task":"direct_classification", "target_metric":metric, "bad_percentile":pct,
                                   "fold":fold, "best_epoch":best_epoch, "epochs_ran":len(hist),
                                   "best_earlystop_loss":min(r["earlystop_loss"] for r in hist)})
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            final_epochs = max(1, int(round(float(np.median(best_epochs)))))
            model, losses = train_classifier_fixed_epochs(
                Xdev, ydev_cls, epochs=final_epochs, device=device, seed=args.seed + 12000 + j*100 + int(pct),
                batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                base_channels=args.base_channels, dropout=args.dropout,
            )
            test_prob = sigmoid_np(scalar_predict(model, Xte, device, args.batch_size))
            torch.save({"model_state_dict":model.state_dict(), "target_metric":metric, "bad_percentile":pct,
                        "true_error_cutoff":cutoff, "final_epochs":final_epochs, "in_channels":int(Xdev.shape[1]),
                        "base_channels":args.base_channels, "dropout":args.dropout}, model_dir / f"classifier_{metric}_p{pct:g}.pth")
            qa_thr, dev_sens, dev_spec = choose_threshold_at_specificity(ydev_cls, oof_prob, args.target_specificity)
            auc = float(roc_auc_score(ytest_cls, test_prob)) if len(np.unique(ytest_cls)) == 2 else np.nan
            apv = float(average_precision_score(ytest_cls, test_prob)) if ytest_cls.sum() else np.nan
            boot = bootstrap_classification(ytest_cls, test_prob, args.n_boot, args.seed + j*101 + int(pct))
            cm = confusion(ytest_cls, test_prob, qa_thr)
            cls_rows.append({"roi":args.roi, "variant":args.variant, "target_metric":metric,
                             "feature_group":"direct_binary_spatial_action_head", "bad_percentile":pct,
                             "true_error_cutoff":cutoff, "development_n":len(Xdev), "test_n":len(Xte),
                             "test_positive_n":int(ytest_cls.sum()), "test_negative_n":int((1-ytest_cls).sum()),
                             "final_epochs_from_median_oof_best":final_epochs,
                             "test_roc_auc":auc, "test_roc_auc_ci95_low":boot["auc_low"], "test_roc_auc_ci95_high":boot["auc_high"],
                             "test_pr_auc":apv, "test_pr_auc_ci95_low":boot["ap_low"], "test_pr_auc_ci95_high":boot["ap_high"],
                             "development_oof_threshold_at_target_specificity":qa_thr,
                             "development_oof_sensitivity":dev_sens, "development_oof_specificity":dev_spec,
                             "test_sensitivity":cm["sensitivity"], "test_specificity":cm["specificity"],
                             "test_ppv":cm["ppv"], "test_npv":cm["npv"]})
            for pid, yt, sc in zip(pid_te, ytest_cls, test_prob):
                pred_rows.append({"patient_id":pid, "target_metric":metric, "task":f"direct_classification_p{pct:g}",
                                  "true_value":int(yt), "score_or_prediction":float(sc)})
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(reg_rows).to_csv(out / "spatial_qa_v2_regression_summary.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(out / "spatial_qa_v2_classification_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "spatial_qa_v2_test_predictions.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out / "spatial_qa_v2_earlystop_summary.csv", index=False)
    protocol: Dict[str, object] = {
        "roi": args.roi, "variant": args.variant,
        "development_splits": ["train", "validation"], "test_split": "test",
        "correction_model_frozen": True, "approved_used_for_spatial_predictor_tensor": False,
        "experiment": "single-target regression heads + direct binary heads + development-only early stopping",
        "head": {"architecture":"same tiny 3D CNN trunk as spatial QA v1, scalar output",
                 "base_channels":args.base_channels, "dropout":args.dropout, "lr":args.lr,
                 "weight_decay":args.weight_decay, "max_epochs":args.max_epochs, "patience":args.patience,
                 "min_delta":args.min_delta, "earlystop_fraction":args.earlystop_fraction},
        "epoch_selection": "Each OOF fold selects best epoch on an inner early-stop subset; final epoch count is median OOF best epoch; final model refit on all development patients.",
        "regression": "One independent SmoothL1 standardized-target head per metric.",
        "classification": {"one_independent_BCEWithLogits_head_per_metric_and_bad_percentile":True,
                           "class_weighting":"fold-specific pos_weight = Nneg/Npos",
                           "bad_percentiles":args.bad_percentiles,
                           "target_specificity":args.target_specificity,
                           "operating_threshold_from_development_oof_probabilities":True},
    }
    (out / "spatial_qa_v2_protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"Saved spatial QA v2 analysis to {out}")


if __name__ == "__main__":
    main()
