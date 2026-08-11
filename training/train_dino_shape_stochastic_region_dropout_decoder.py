#!/usr/bin/env python3
"""Train a stochastic-region-dropout denoising decoder with frozen 3DINO.

The corruption distribution is benchmark-independent and mixes continuous
boundary deformation with generic depth-controlled missing-region dropout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, get_worker_info

from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder
from training.corruptions.stochastic_region_dropout import stochastic_region_dropout_corruption


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto (default), none, or an explicit checkpoint path",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete this experiment's existing output directory and train from epoch 1.",
    )
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_id_from_name(name: str) -> str:
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else stem


class BinaryROIDataset(Dataset):
    def __init__(self, directory: str | Path):
        self.files = sorted(Path(directory).rglob("*.nii.gz"))
        if not self.files:
            raise FileNotFoundError(f"No .nii.gz files found under {directory}")

    def __len__(self):
        return len(self.files)

    def load_item(self, index):
        path = self.files[index]
        t0 = time.perf_counter()
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        load_seconds = time.perf_counter() - t0
        mask = (arr > 0).astype(np.float32)
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask, got {mask.shape}: {path}")
        return torch.from_numpy(mask)[None], patient_id_from_name(path.name), load_seconds

    def __getitem__(self, index):
        mask, patient_id, load_seconds = self.load_item(index)
        return {
            "mask": mask,
            "patient_id": patient_id,
            "load_seconds": torch.tensor(load_seconds, dtype=torch.float64),
        }


class CorruptedROIDataset(Dataset):
    """Load and corrupt masks inside DataLoader worker processes.

    Training workers keep independent seeded RNG streams, so persistent workers
    continuously generate new corruptions. Validation uses a patient-specific
    stable seed, making the validation corruption deterministic across epochs.
    """

    def __init__(self, directory: str | Path, cfg: Dict, train: bool):
        self.base = BinaryROIDataset(directory)
        self.cfg = cfg
        self.train = bool(train)
        self.base_seed = int(cfg.get("seed", 88))
        self._worker_rng = None

    def __len__(self):
        return len(self.base)

    @property
    def files(self):
        return self.base.files

    def _rng(self, patient_id: str) -> random.Random:
        if not self.train:
            return random.Random(stable_seed(str(patient_id), self.base_seed))
        if self._worker_rng is None:
            info = get_worker_info()
            worker_seed = info.seed if info is not None else torch.initial_seed()
            self._worker_rng = random.Random(int(worker_seed) % (2**63 - 1))
        return self._worker_rng

    def __getitem__(self, index):
        target, patient_id, load_seconds = self.base.load_item(index)
        t0 = time.perf_counter()
        corrupted, changed, metadata = stochastic_region_dropout_corruption(
            target, self.cfg, self._rng(patient_id), cache_key=str(patient_id)
        )
        corruption_seconds = time.perf_counter() - t0
        metadata = dict(metadata)
        metadata["patient_id"] = str(patient_id)
        metadata["dataset_corruption_seconds"] = float(corruption_seconds)
        return {
            "mask": target,
            "corrupted": corrupted,
            "changed": changed,
            "patient_id": str(patient_id),
            "metadata_json": json.dumps(metadata, allow_nan=True),
            "load_seconds": torch.tensor(load_seconds, dtype=torch.float64),
            "corruption_seconds": torch.tensor(corruption_seconds, dtype=torch.float64),
        }




def _pack_bool(mask: torch.Tensor) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    return np.packbits(arr.reshape(-1)), tuple(int(v) for v in arr.shape)


def _unpack_bool(bits: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
    n = int(np.prod(shape))
    arr = np.unpackbits(bits, count=n).reshape(tuple(int(v) for v in shape)).astype(np.float32, copy=False)
    return torch.from_numpy(arr)


def validation_cache_fingerprint(cfg: Dict) -> str:
    payload = {
        "seed": int(cfg.get("seed", 88)),
        "stochastic_region_dropout": cfg["stochastic_region_dropout"],
        "validation_label_dir": str(cfg["data"]["validation_label_dir"]),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_filename(cache_dir: Path, patient_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(patient_id))
    return cache_dir / f"patient_{safe}.npz"


def build_validation_cache(raw_dataset: "CorruptedROIDataset", cache_dir: Path,
                           workers: int, prefetch_factor: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for path in raw_dataset.files:
        patient_id = patient_id_from_name(path.name)
        if not _cache_filename(cache_dir, patient_id).is_file():
            missing.append(patient_id)
    if not missing:
        print(f"Validation corruption cache ready: {cache_dir} ({len(raw_dataset)} cases)")
        return

    print(f"Building validation corruption cache: {len(missing)}/{len(raw_dataset)} cases missing")
    kwargs = dict(
        num_workers=workers, pin_memory=False, persistent_workers=False,
        worker_init_fn=worker_init_fn if workers > 0 else None,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(raw_dataset, batch_size=1, shuffle=False, **kwargs)
    t0 = time.perf_counter()
    written = 0
    for batch in loader:
        patient_id = str(batch["patient_id"][0])
        out = _cache_filename(cache_dir, patient_id)
        if out.is_file():
            continue
        corrupted_bits, corrupted_shape = _pack_bool(batch["corrupted"][0])
        changed_bits, changed_shape = _pack_bool(batch["changed"][0])
        metadata_json = str(batch["metadata_json"][0])
        np.savez_compressed(
            out,
            corrupted_bits=corrupted_bits,
            corrupted_shape=np.asarray(corrupted_shape, dtype=np.int32),
            changed_bits=changed_bits,
            changed_shape=np.asarray(changed_shape, dtype=np.int32),
            metadata_json=np.asarray(metadata_json),
        )
        written += 1
    print(f"Validation corruption cache built: {written} new cases in {time.perf_counter()-t0:.1f}s")


class CachedValidationROIDataset(Dataset):
    def __init__(self, directory: str | Path, cache_dir: Path):
        self.base = BinaryROIDataset(directory)
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.base)

    @property
    def files(self):
        return self.base.files

    def __getitem__(self, index):
        target, patient_id, load_seconds = self.base.load_item(index)
        t0 = time.perf_counter()
        cache_path = _cache_filename(self.cache_dir, patient_id)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing validation cache entry: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            corrupted = _unpack_bool(data["corrupted_bits"], data["corrupted_shape"])
            changed = _unpack_bool(data["changed_bits"], data["changed_shape"])
            metadata_json = str(data["metadata_json"].item())
        cache_seconds = time.perf_counter() - t0
        meta = json.loads(metadata_json)
        meta["validation_cache_hit"] = 1
        meta["validation_cache_load_seconds"] = cache_seconds
        return {
            "mask": target,
            "corrupted": corrupted,
            "changed": changed,
            "patient_id": str(patient_id),
            "metadata_json": json.dumps(meta, allow_nan=True),
            "load_seconds": torch.tensor(load_seconds + cache_seconds, dtype=torch.float64),
            "corruption_seconds": torch.tensor(0.0, dtype=torch.float64),
        }


def worker_init_fn(worker_id: int):
    # Prevent 8 workers from each spawning their own large OpenMP/ITK pools.
    torch.set_num_threads(1)
    try:
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    except AttributeError:
        pass
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)

def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def corrupt_batch(target: torch.Tensor, patient_ids: Sequence[str], cfg: Dict,
                  train: bool, epoch: int, step: int):
    base_seed = int(cfg.get("seed", 88))
    outputs, changed, metadata = [], [], []
    for i in range(target.shape[0]):
        clean_i = target[i].detach()
        if train:
            # Deterministic for a fixed seed/run, while changing by epoch, batch,
            # and item. This makes experiments reproducible without repeating the
            # same corruption every epoch.
            rng = random.Random(
                base_seed + epoch * 1_000_003 + step * 1009 + i * 97
            )
        else:
            rng = random.Random(stable_seed(str(patient_ids[i]), base_seed))
        x, changed_i, meta_i = stochastic_region_dropout_corruption(clean_i, cfg, rng, cache_key=str(patient_ids[i]))
        meta_i = dict(meta_i)
        meta_i["patient_id"] = str(patient_ids[i])
        outputs.append(x)
        changed.append(changed_i)
        metadata.append(meta_i)
    return torch.stack(outputs, dim=0), torch.stack(changed, dim=0), metadata

def dilate_changed_region(changed: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return changed
    kernel = 2 * radius + 1
    return F.max_pool3d(changed, kernel_size=kernel, stride=1, padding=radius)


def dice_per_case(prob_or_mask: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    dims = tuple(range(2, target.ndim))
    inter = (prob_or_mask * target).sum(dim=dims)
    den = prob_or_mask.sum(dim=dims) + target.sum(dim=dims)
    return (2 * inter + eps) / (den + eps)


def denoising_loss(logits: torch.Tensor, target: torch.Tensor, changed: torch.Tensor, corrupted: torch.Tensor, cfg: Dict):
    loss_cfg = cfg["loss"]
    prob = torch.sigmoid(logits)
    global_dice = 1.0 - dice_per_case(prob, target).mean()
    global_bce = F.binary_cross_entropy_with_logits(logits, target)

    region = dilate_changed_region(changed, int(loss_cfg.get("changed_region_dilation_voxels", 4)))
    voxel_weight = 1.0 + float(loss_cfg.get("changed_region_voxel_weight", 8.0)) * region
    changed_bce = F.binary_cross_entropy_with_logits(logits, target, weight=voxel_weight)

    missing = ((target > 0.5) & (corrupted < 0.5)).float()
    missing_voxels = missing.sum()
    if float(missing_voxels.detach()) > 0:
        missing_bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * missing).sum() / missing_voxels
    else:
        missing_bce = logits.new_zeros(())

    total = (
        float(loss_cfg.get("global_dice_weight", 0.40)) * global_dice
        + float(loss_cfg.get("global_bce_weight", 0.10)) * global_bce
        + float(loss_cfg.get("changed_region_bce_weight", 0.30)) * changed_bce
        + float(loss_cfg.get("missing_region_bce_weight", 0.20)) * missing_bce
    )
    return total, {
        "total": total,
        "global_dice_loss": global_dice,
        "global_bce": global_bce,
        "changed_region_bce": changed_bce,
        "missing_region_bce": missing_bce,
    }


def amp_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _cuda_sync(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, optimizer, device, cfg, train: bool, epoch: int):
    model.train(train)
    keys = ["total", "global_dice_loss", "global_bce", "changed_region_bce", "missing_region_bce",
            "input_target_dice", "recon_target_dice", "input_recon_dice", "correction_gain"]
    totals = {k: 0.0 for k in keys}
    totals["n"] = 0
    corruption_rows = []
    accumulation = int(cfg["training"].get("accumulation_steps", 1))
    precision = cfg["training"].get("precision", "bf16")
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    timing_cfg = cfg["training"].get("timing", {})
    timing_enabled = bool(timing_cfg.get("enabled", True))
    sync_cuda = bool(timing_cfg.get("sync_cuda", True))
    log_every = max(1, int(timing_cfg.get("log_every_steps", 10)))
    timing = {
        "data_wait_seconds": 0.0, "worker_load_seconds_sum": 0.0,
        "worker_corruption_seconds_sum": 0.0, "host_to_device_seconds": 0.0,
        "forward_seconds": 0.0, "loss_seconds": 0.0,
        "backward_optimizer_seconds": 0.0, "metrics_seconds": 0.0,
    }
    epoch_start = time.perf_counter()
    previous_end = epoch_start

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch_ready = time.perf_counter()
        timing["data_wait_seconds"] += batch_ready - previous_end
        timing["worker_load_seconds_sum"] += float(batch["load_seconds"].sum())
        timing["worker_corruption_seconds_sum"] += float(batch["corruption_seconds"].sum())

        t0 = time.perf_counter()
        target = batch["mask"].to(device, non_blocking=True)
        corrupted = batch["corrupted"].to(device, non_blocking=True)
        changed = batch["changed"].to(device, non_blocking=True)
        _cuda_sync(device, timing_enabled and sync_cuda)
        timing["host_to_device_seconds"] += time.perf_counter() - t0

        batch_metadata = [json.loads(x) for x in batch["metadata_json"]]
        for meta in batch_metadata:
            corruption_rows.append({"epoch": epoch, "split": "train" if train else "validation", **meta})

        with torch.set_grad_enabled(train):
            _cuda_sync(device, timing_enabled and sync_cuda)
            t0 = time.perf_counter()
            with amp_context(device, precision):
                logits, _ = model(corrupted)
            _cuda_sync(device, timing_enabled and sync_cuda)
            timing["forward_seconds"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            with amp_context(device, precision):
                loss, pieces = denoising_loss(logits, target, changed, corrupted, cfg)
                backward = loss / accumulation
            _cuda_sync(device, timing_enabled and sync_cuda)
            timing["loss_seconds"] += time.perf_counter() - t0

            if train:
                t0 = time.perf_counter()
                backward.backward()
                if (step + 1) % accumulation == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(
                        model.decoder.parameters(), float(cfg["training"].get("grad_clip", 1.0))
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                _cuda_sync(device, timing_enabled and sync_cuda)
                timing["backward_optimizer_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            recon = (torch.sigmoid(logits) >= threshold).float()
            d_it = dice_per_case(corrupted, target)
            d_rt = dice_per_case(recon, target)
            d_ir = dice_per_case(corrupted, recon)
            metrics = {
                "input_target_dice": d_it.mean(),
                "recon_target_dice": d_rt.mean(),
                "input_recon_dice": d_ir.mean(),
                "correction_gain": (d_rt - d_it).mean(),
            }
        _cuda_sync(device, timing_enabled and sync_cuda)
        timing["metrics_seconds"] += time.perf_counter() - t0

        n = target.shape[0]
        totals["n"] += n
        for key in ("total", "global_dice_loss", "global_bce", "changed_region_bce", "missing_region_bce"):
            totals[key] += float(pieces[key].detach().float()) * n
        for key, value in metrics.items():
            totals[key] += float(value.detach().float()) * n

        previous_end = time.perf_counter()
        if timing_enabled and ((step + 1) % log_every == 0 or (step + 1) == len(loader)):
            elapsed = previous_end - epoch_start
            print(
                f"[timing] split={'train' if train else 'validation'} epoch={epoch} "
                f"step={step + 1}/{len(loader)} elapsed={elapsed:.1f}s "
                f"data_wait={timing['data_wait_seconds']:.1f}s "
                f"worker_corruption_sum={timing['worker_corruption_seconds_sum']:.1f}s "
                f"forward={timing['forward_seconds']:.1f}s "
                f"backward_opt={timing['backward_optimizer_seconds']:.1f}s",
                flush=True,
            )

    epoch_seconds = time.perf_counter() - epoch_start
    timing["epoch_seconds"] = epoch_seconds
    timing["samples_per_second"] = totals["n"] / max(epoch_seconds, 1e-9)
    timing["worker_load_seconds_per_sample"] = timing["worker_load_seconds_sum"] / max(totals["n"], 1)
    timing["worker_corruption_seconds_per_sample"] = timing["worker_corruption_seconds_sum"] / max(totals["n"], 1)
    if device.type == "cuda":
        timing["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
        timing["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
    metrics_out = {k: totals[k] / totals["n"] for k in keys}
    return metrics_out, corruption_rows, timing


def resolve_resume_checkpoint(resume_arg: str, ckpt_dir: Path) -> Path | None:
    value = str(resume_arg).strip()
    if value.lower() in {"none", "off", "false", "0"}:
        return None
    if value.lower() == "auto":
        candidate = ckpt_dir / "last_decoder.pth"
        return candidate if candidate.is_file() else None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def history_resume_stats(history_path: Path) -> tuple[float, int]:
    if not history_path.is_file():
        return float("inf"), 0
    rows = []
    with open(history_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("inf"), 0
    vals = []
    for row in rows:
        try:
            vals.append(float(row["validation_total"]))
        except Exception:
            vals.append(float("inf"))
    best = min(vals)
    best_idx = min(range(len(vals)), key=vals.__getitem__)
    bad_epochs = max(0, len(vals) - 1 - best_idx)
    return best, bad_epochs


def append_csv_row(path: Path, row: Dict):
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_corruption_profile(path: Path, epoch: int, split: str, rows: Sequence[Dict]):
    branches = sorted({str(r.get("branch", "unknown")) for r in rows})
    for branch in branches:
        sub = [r for r in rows if str(r.get("branch", "unknown")) == branch]
        times = np.asarray([float(r.get("dataset_corruption_seconds", r.get("corruption_function_seconds", 0.0))) for r in sub], dtype=float)
        attempts = np.asarray([float(r.get("dropout_total_attempts", r.get("dropout_attempt", np.nan))) for r in sub], dtype=float)
        attempts = attempts[np.isfinite(attempts)]
        row = {
            "epoch": epoch, "split": split, "branch": branch, "n": len(sub),
            "corruption_seconds_mean": float(np.mean(times)) if len(times) else float("nan"),
            "corruption_seconds_median": float(np.median(times)) if len(times) else float("nan"),
            "corruption_seconds_p90": float(np.percentile(times, 90)) if len(times) else float("nan"),
            "corruption_seconds_p95": float(np.percentile(times, 95)) if len(times) else float("nan"),
            "corruption_seconds_max": float(np.max(times)) if len(times) else float("nan"),
            "dropout_attempts_mean": float(np.mean(attempts)) if len(attempts) else float("nan"),
            "dropout_attempts_median": float(np.median(attempts)) if len(attempts) else float("nan"),
            "dropout_attempts_p95": float(np.percentile(attempts, 95)) if len(attempts) else float("nan"),
            "dropout_attempts_max": float(np.max(attempts)) if len(attempts) else float("nan"),
        }
        for key in (
            "dropout_timing_invariant_seconds", "dropout_timing_field_seconds",
            "dropout_timing_envelope_seconds", "dropout_timing_component_seconds",
            "dropout_timing_depth_seconds", "dropout_timing_topology_seconds",
            "dropout_timing_total_seconds",
        ):
            vals = [float(r[key]) for r in sub if r.get(key) is not None]
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        append_csv_row(path, row)


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Stochastic-region-dropout training config not found: {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ["experiment_name", "output_root", "dino", "data",
                "decoder", "stochastic_region_dropout", "training", "loss"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing required config sections: {missing}")
    seed_all(int(cfg.get("seed", 88)))

    experiment_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
    if args.fresh:
        if experiment_dir.exists():
            print(f"Fresh run requested: removing existing experiment directory: {experiment_dir}", flush=True)
            shutil.rmtree(experiment_dir)
        args.resume = "none"
    ckpt_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_ds = CorruptedROIDataset(cfg["data"]["train_label_dir"], cfg, train=True)
    raw_val_ds = CorruptedROIDataset(cfg["data"]["validation_label_dir"], cfg, train=False)
    requested_workers = int(cfg["training"].get("num_workers", 8))
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    workers = max(0, min(requested_workers, available_cpus))
    batch_size = int(cfg["training"].get("batch_size", 1))
    prefetch_factor = int(cfg["training"].get("prefetch_factor", 2))
    loader_kwargs = dict(
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn if workers > 0 else None,
    )
    if workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 88)))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator, **loader_kwargs
    )

    validation_cache_cfg = cfg["training"].get("validation_cache", {})
    validation_cache_enabled = bool(validation_cache_cfg.get("enabled", True))
    if validation_cache_enabled:
        fingerprint = validation_cache_fingerprint(cfg)
        cache_root = Path(validation_cache_cfg.get(
            "directory", experiment_dir / "validation_corruption_cache"
        )) / fingerprint
        build_validation_cache(raw_val_ds, cache_root, workers, prefetch_factor)
        val_ds = CachedValidationROIDataset(cfg["data"]["validation_label_dir"], cache_root)
    else:
        val_ds = raw_val_ds
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, **loader_kwargs
    )

    device = torch.device(cfg.get("training", {}).get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model = Frozen3DINOShapeAutoencoder(
        repo_path=cfg["dino"]["repo_path"],
        checkpoint_path=cfg["dino"]["checkpoint_path"],
        config_name=cfg["dino"].get("config_name", "train/vit3d_highres"),
        decoder_channels=cfg["decoder"].get("channels", [256, 128, 64, 32]),
    ).to(device)

    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("3DINO encoder must remain frozen for this experiment.")

    optimizer = torch.optim.AdamW(model.decoder.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Train/validation cases: {len(train_ds)}/{len(val_ds)}")
    print(f"Frozen encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Trainable decoder params: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad):,}")
    print(f"DataLoader workers: requested={requested_workers}, effective={workers}, "
          f"CPU affinity={available_cpus}, prefetch_factor={prefetch_factor}")
    print("Input corruption runs inside DataLoader workers; worker corruption time is summed "
          "across workers, while data_wait is wall time seen by the trainer.")

    history_path = log_dir / "history.csv"
    corruption_summary_path = log_dir / "training_corruption_summary.csv"
    validation_detail_path = log_dir / "validation_corruption_details.csv"
    timing_path = log_dir / "epoch_timing.csv"
    corruption_profile_path = log_dir / "corruption_profile.csv"
    fields = ["epoch"] + [f"train_{k}" for k in run_epoch.__annotations__ if False]  # placeholder removed below
    metric_names = ["total", "global_dice_loss", "global_bce", "changed_region_bce", "missing_region_bce",
                    "input_target_dice", "recon_target_dice", "input_recon_dice", "correction_gain"]
    fields = ["epoch"] + [f"train_{k}" for k in metric_names] + [f"validation_{k}" for k in metric_names]

    best = float("inf")
    bad_epochs = 0
    start_epoch = 1
    epochs = int(cfg["training"].get("num_epochs", 200))
    patience = int(cfg["training"].get("early_stopping_patience", 40))
    clean_floor = float(cfg.get("evaluation", {}).get("clean_dice_floor", 0.0))

    resume_path = resolve_resume_checkpoint(args.resume, ckpt_dir)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        hist_best, hist_bad = history_resume_stats(history_path)
        best = float(checkpoint.get("best_validation_total", hist_best))
        if not np.isfinite(best):
            best = float(checkpoint.get("validation_total", float("inf")))
        bad_epochs = int(checkpoint.get("bad_epochs", hist_bad))

        old_impl = str(checkpoint.get("config", {}).get("stochastic_region_dropout", {}).get("implementation_version", "legacy"))
        new_impl = str(cfg.get("stochastic_region_dropout", {}).get("implementation_version", "legacy"))
        if old_impl != new_impl:
            print(
                f"WARNING: corruption implementation changed ({old_impl} -> {new_impl}). "
                "Decoder/optimizer are resumed, but validation-best and early-stopping counters are reset "
                "because the deterministic validation corruptions may differ numerically.",
                flush=True,
            )
            best = float("inf")
            bad_epochs = 0

        print(
            f"Resuming from {resume_path}: completed epoch={completed_epoch}, "
            f"next epoch={start_epoch}, best_validation_total={best:.6f}, bad_epochs={bad_epochs}",
            flush=True,
        )
    else:
        print("Starting a new training run (no resume checkpoint).", flush=True)

    history_exists = history_path.exists() and history_path.stat().st_size > 0 and start_epoch > 1
    with open(history_path, "a" if history_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not history_exists:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            tr, tr_rows, tr_timing = run_epoch(model, train_loader, optimizer, device, cfg, True, epoch)
            va, va_rows, va_timing = run_epoch(model, val_loader, optimizer, device, cfg, False, epoch)

            def summarize(rows):
                numeric_keys = [
                    "changed_voxel_fraction", "input_target_volume_change_pct",
                    "max_displacement_mm", "warp_sigma_vox",
                    "distance_noise_amplitude_mm", "distance_noise_sigma_vox",
                    "input_component_count", "input_largest_component_fraction",
                ]
                summary = {"epoch": epoch, "n": len(rows)}
                branches = {}
                for item in rows:
                    branch = str(item.get("branch", "unknown"))
                    branches[branch] = branches.get(branch, 0) + 1
                summary["branch_counts_json"] = json.dumps(branches, sort_keys=True)

                # Log the realized sign of the physical global-boundary-offset
                # branch. Count requested offsets even if they were rejected to
                # clean, and accepted offsets separately, so direction balance
                # can be audited instead of inferred from the configured 50/50
                # probability.
                global_direction_counts = {}
                accepted_global_direction_counts = {}
                for item in rows:
                    direction = item.get("global_boundary_offset_direction")
                    if direction not in ("inward", "outward"):
                        continue
                    global_direction_counts[direction] = global_direction_counts.get(direction, 0) + 1
                    rejected = int(item.get("global_boundary_offset_rejected_to_clean", 0) or 0)
                    if not rejected:
                        accepted_global_direction_counts[direction] = (
                            accepted_global_direction_counts.get(direction, 0) + 1
                        )
                summary["global_boundary_direction_counts_json"] = json.dumps(
                    global_direction_counts, sort_keys=True
                )
                summary["accepted_global_boundary_direction_counts_json"] = json.dumps(
                    accepted_global_direction_counts, sort_keys=True
                )
                summary["global_boundary_inward_count"] = int(global_direction_counts.get("inward", 0))
                summary["global_boundary_outward_count"] = int(global_direction_counts.get("outward", 0))
                summary["accepted_global_boundary_inward_count"] = int(
                    accepted_global_direction_counts.get("inward", 0)
                )
                summary["accepted_global_boundary_outward_count"] = int(
                    accepted_global_direction_counts.get("outward", 0)
                )
                for key in numeric_keys:
                    vals = [float(item[key]) for item in rows if item.get(key) is not None and np.isfinite(float(item[key]))]
                    summary[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
                    summary[f"{key}_median"] = float(np.median(vals)) if vals else float("nan")
                return summary

            write_corruption_profile(corruption_profile_path, epoch, "train", tr_rows)
            write_corruption_profile(corruption_profile_path, epoch, "validation", va_rows)
            summary = summarize(tr_rows)
            write_header = not corruption_summary_path.exists()
            with open(corruption_summary_path, "a", newline="", encoding="utf-8") as sf:
                sw = csv.DictWriter(sf, fieldnames=list(summary.keys()))
                if write_header:
                    sw.writeheader()
                sw.writerow(summary)

            if va_rows:
                fieldnames = [
                    "epoch", "split", "patient_id", "branch", "accepted_attempt",
                    "changed_voxel_fraction", "input_target_volume_change_pct",
                    "retained_volume_fraction", "max_displacement_mm", "warp_sigma_vox",
                    "control_grid_z", "control_grid_y", "control_grid_x",
                    "distance_noise_amplitude_mm", "distance_noise_sigma_vox",
                    "target_component_count", "input_component_count",
                    "input_largest_component_fraction", "input_euler_characteristic",
                    "global_boundary_offset_direction", "global_boundary_offset_mm",
                    "global_boundary_offset_magnitude_mm",
                    "global_boundary_offset_rejected_to_clean",
                    "global_boundary_offset_volume_change_pct",
                ]
                write_header = not validation_detail_path.exists()
                with open(validation_detail_path, "a", newline="", encoding="utf-8") as vf:
                    vw = csv.DictWriter(vf, fieldnames=fieldnames, extrasaction="ignore")
                    if write_header:
                        vw.writeheader()
                    vw.writerows(va_rows)
            timing_row = {"epoch": epoch}
            for prefix, values in (("train", tr_timing), ("validation", va_timing)):
                timing_row.update({f"{prefix}_{k}": v for k, v in values.items()})
            write_timing_header = not timing_path.exists()
            with open(timing_path, "a", newline="", encoding="utf-8") as tf:
                tw = csv.DictWriter(tf, fieldnames=list(timing_row.keys()))
                if write_timing_header:
                    tw.writeheader()
                tw.writerow(timing_row)
            print("[epoch timing] " + json.dumps(timing_row), flush=True)

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                   **{f"validation_{k}": v for k, v in va.items()}}
            writer.writerow(row)
            f.flush()
            print(json.dumps(row))

            # Primary selection remains validation loss. Correction gain is logged
            # explicitly so identity-mapping behavior is visible during training.
            score = va["total"]
            improved = score < best
            if improved:
                best = score
                bad_epochs = 0
            else:
                bad_epochs += 1
            state = {
                "epoch": epoch,
                "decoder_state_dict": model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_total": score,
                "validation_correction_gain": va["correction_gain"],
                "best_validation_total": best,
                "bad_epochs": bad_epochs,
                "config": cfg,
            }
            torch.save(state, ckpt_dir / "last_decoder.pth")
            if improved:
                torch.save(state, ckpt_dir / "best_decoder.pth")
                print(f"Saved best stochastic-region-dropout decoder: validation total={best:.6f}, "
                      f"correction_gain={va['correction_gain']:.4f}")
            if bad_epochs >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break


if __name__ == "__main__":
    main()
