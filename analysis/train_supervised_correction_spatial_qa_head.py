#!/usr/bin/env python3
"""Train/evaluate a tiny learned spatial QA pooling head on frozen action maps.

The correction model is frozen upstream.  This head receives 12 low-resolution
spatial channels derived from P(add/remove/action) and predicts the three true
segmentation-error metrics jointly: 1-Dice, HD95, and ASSD.

Protocol
--------
* Development = train + validation. Test is untouched until final scoring.
* Architecture/hyperparameters are fixed a priori (no test/validation search).
* 5-fold development OOF predictions select the high-specificity binary QA
  operating threshold for each target.
* A final head is trained on all development patients for the same fixed number
  of epochs and evaluated once on test.
* The continuous prediction for each target is also its binary QA discriminator
  score; separate classification heads are deliberately not trained.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
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


class TinySpatialQAHead(nn.Module):
    """~tens-of-thousands-parameter 3D CNN with learned spatial pooling."""

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
        self.regressor = nn.Sequential(
            nn.Linear(6 * c, 4 * c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * c, len(TARGETS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = torch.cat([self.avg(z).flatten(1), self.mx(z).flatten(1)], dim=1)
        return self.regressor(z)


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
    X = []
    for row in merged.itertuples(index=False):
        arr = np.load(tensor_root / split / row.path)["spatial"].astype(np.float32)
        X.append(arr)
    X = np.stack(X)
    y = merged[cols].to_numpy(dtype=np.float32)
    finite = np.isfinite(y).all(axis=1)
    if not finite.all():
        print(f"Dropping {(~finite).sum()} non-finite target rows from {split}")
    return merged.loc[finite, "patient_id"].astype(str).to_numpy(), X[finite], y[finite]


def train_head(
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    base_channels: int,
    dropout: float,
) -> Tuple[TinySpatialQAHead, np.ndarray, np.ndarray, list[float]]:
    set_seed(seed)
    y_mean = y.mean(axis=0).astype(np.float32)
    y_std = y.std(axis=0).astype(np.float32)
    y_std = np.where(y_std > 1e-8, y_std, 1.0).astype(np.float32)
    yz = (y - y_mean) / y_std

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(yz))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, generator=gen, num_workers=0)
    model = TinySpatialQAHead(X.shape[1], base_channels=base_channels, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    losses = []
    model.train()
    for epoch in range(epochs):
        running = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            running += float(loss.detach()) * len(xb)
            count += len(xb)
        losses.append(running / max(count, 1))
        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"epoch {epoch+1:03d}/{epochs} loss={losses[-1]:.6f}", flush=True)
    return model, y_mean, y_std, losses


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False, num_workers=0)
    out = []
    for (xb,) in loader:
        predz = model(xb.to(device=device, dtype=torch.float32, non_blocking=True)).cpu().numpy()
        out.append(predz)
    return np.concatenate(out, axis=0) * y_std + y_mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-root", required=True)
    ap.add_argument("--evaluation-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--base-channels", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--bad-percentiles", nargs="+", type=float, default=[75.0, 50.0])
    ap.add_argument("--target-specificity", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    tensor_root = Path(args.tensor_root)
    evaluation_dir = Path(args.evaluation_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pid_tr, Xtr, ytr = load_split(tensor_root, evaluation_dir, "train")
    pid_va, Xva, yva = load_split(tensor_root, evaluation_dir, "validation")
    pid_te, Xte, yte = load_split(tensor_root, evaluation_dir, "test")
    pid_dev = np.concatenate([pid_tr, pid_va])
    Xdev = np.concatenate([Xtr, Xva], axis=0)
    ydev = np.concatenate([ytr, yva], axis=0)
    print(f"Loaded dev={len(Xdev)} test={len(Xte)} tensor_shape={Xdev.shape[1:]} device={device}")

    nfold = min(args.cv_folds, len(Xdev))
    if nfold < 2:
        raise RuntimeError("Need at least two development patients")
    kf = KFold(n_splits=nfold, shuffle=True, random_state=args.seed)
    oof = np.full_like(ydev, np.nan, dtype=np.float32)
    cv_rows = []
    for fold, (train_idx, held_idx) in enumerate(kf.split(Xdev)):
        print(f"OOF fold {fold+1}/{nfold}: train={len(train_idx)} held={len(held_idx)}", flush=True)
        model, ym, ys, losses = train_head(
            Xdev[train_idx], ydev[train_idx], device=device, seed=args.seed + fold * 101,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            base_channels=args.base_channels, dropout=args.dropout,
        )
        oof[held_idx] = predict(model, Xdev[held_idx], ym, ys, device, args.batch_size)
        cv_rows.append({"fold": fold, "n_train": len(train_idx), "n_heldout": len(held_idx),
                        "final_train_loss": losses[-1], "min_train_loss": min(losses)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF predictions contain non-finite values")

    print("Training final head on all development patients", flush=True)
    final_model, y_mean, y_std, final_losses = train_head(
        Xdev, ydev, device=device, seed=args.seed + 9999,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        base_channels=args.base_channels, dropout=args.dropout,
    )
    test_pred = predict(final_model, Xte, y_mean, y_std, device, args.batch_size)
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "target_mean": y_mean,
        "target_std": y_std,
        "in_channels": int(Xdev.shape[1]),
        "base_channels": args.base_channels,
        "dropout": args.dropout,
        "targets": [name for name, _ in TARGETS],
    }, model_dir / "spatial_qa_head.pth")

    reg_rows = []
    cls_rows = []
    pred_rows = []
    for j, (metric, _) in enumerate(TARGETS):
        tmp = pd.DataFrame({"x": test_pred[:, j], "y": yte[:, j]})
        rm = regression_metrics(tmp, "x", "y", 1.0, 0.0)
        rb = bootstrap_regression(tmp, "x", "y", 1.0, 0.0, args.n_boot, args.seed + j * 31)
        reg_rows.append({
            "roi": args.roi, "variant": args.variant, "target_metric": metric,
            "feature_group": "learned_spatial_action_head", "development_n": len(Xdev), "test_n": len(Xte),
            "test_mae": rm["mae"], "test_mae_ci95_low": rb["mae_low"], "test_mae_ci95_high": rb["mae_high"],
            "test_rmse": rm["rmse"], "test_r2": rm["r2"], "test_pearson_r": rm["pearson_r"],
            "test_spearman_rho": rm["spearman_rho"], "test_spearman_ci95_low": rb["spearman_low"],
            "test_spearman_ci95_high": rb["spearman_high"],
        })
        for pid, yt, yp in zip(pid_te, yte[:, j], test_pred[:, j]):
            pred_rows.append({"patient_id": pid, "target_metric": metric, "task": "regression",
                              "true_value": float(yt), "score_or_prediction": float(yp)})

        for pct in args.bad_percentiles:
            cutoff = percentile_higher(pd.Series(ydev[:, j]), pct)
            ydev_cls = (ydev[:, j] > cutoff).astype(int)
            ytest_cls = (yte[:, j] > cutoff).astype(int)
            score_dev = oof[:, j]
            score_test = test_pred[:, j]
            qa_thr, dev_sens, dev_spec = choose_threshold_at_specificity(ydev_cls, score_dev, args.target_specificity)
            auc = float(roc_auc_score(ytest_cls, score_test)) if len(np.unique(ytest_cls)) == 2 else np.nan
            apv = float(average_precision_score(ytest_cls, score_test)) if ytest_cls.sum() else np.nan
            boot = bootstrap_classification(ytest_cls, score_test, args.n_boot, args.seed + j * 101 + int(pct))
            cm = confusion(ytest_cls, score_test, qa_thr)
            cls_rows.append({
                "roi": args.roi, "variant": args.variant, "target_metric": metric,
                "feature_group": "learned_spatial_action_head", "bad_percentile": pct,
                "true_error_cutoff": cutoff, "development_n": len(Xdev), "test_n": len(Xte),
                "test_positive_n": int(ytest_cls.sum()), "test_negative_n": int((1-ytest_cls).sum()),
                "test_roc_auc": auc, "test_roc_auc_ci95_low": boot["auc_low"], "test_roc_auc_ci95_high": boot["auc_high"],
                "test_pr_auc": apv, "test_pr_auc_ci95_low": boot["ap_low"], "test_pr_auc_ci95_high": boot["ap_high"],
                "development_oof_threshold_at_target_specificity": qa_thr,
                "development_oof_sensitivity": dev_sens, "development_oof_specificity": dev_spec,
                "test_sensitivity": cm["sensitivity"], "test_specificity": cm["specificity"],
                "test_ppv": cm["ppv"], "test_npv": cm["npv"],
            })
            for pid, yt, sc in zip(pid_te, ytest_cls, score_test):
                pred_rows.append({"patient_id": pid, "target_metric": metric, "task": f"classification_p{pct:g}",
                                  "true_value": int(yt), "score_or_prediction": float(sc)})

    pd.DataFrame(reg_rows).to_csv(out / "spatial_qa_regression_summary.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(out / "spatial_qa_classification_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "spatial_qa_test_predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "spatial_qa_cv_training_summary.csv", index=False)
    pd.DataFrame({"epoch": np.arange(1, len(final_losses) + 1), "train_loss": final_losses}).to_csv(
        out / "spatial_qa_final_training_curve.csv", index=False
    )
    protocol: Dict[str, object] = {
        "roi": args.roi,
        "variant": args.variant,
        "development_splits": ["train", "validation"],
        "test_split": "test",
        "correction_model_frozen": True,
        "approved_used_for_spatial_predictor_tensor": False,
        "learned_head": {
            "architecture": "3 conv blocks -> adaptive avg+max pooling -> 2-layer MLP -> 3 continuous targets",
            "input_channels": int(Xdev.shape[1]),
            "input_spatial_shape": list(Xdev.shape[2:]),
            "base_channels": args.base_channels,
            "dropout": args.dropout,
            "epochs_fixed_a_priori": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": "SmoothL1 on development-standardized targets",
        },
        "oof_protocol": {
            "folds": nfold,
            "purpose": "development-only operating-threshold selection",
            "fixed_hyperparameters_no_cv_model_selection": True,
        },
        "classification": {
            "score": "continuous metric prediction from the same regression head",
            "bad_percentiles": args.bad_percentiles,
            "target_specificity": args.target_specificity,
            "operating_threshold_from_development_oof_predictions": True,
        },
    }
    (out / "spatial_qa_protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"Saved learned spatial QA analysis to {out}")


if __name__ == "__main__":
    main()
