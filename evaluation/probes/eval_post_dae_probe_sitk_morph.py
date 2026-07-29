#!/usr/bin/env python3
import os
import copy
import re
import random
import argparse
from typing import Dict, List, Tuple, Set

import yaml
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import DataLoader

from dataset.dataset_generator import ProjectionDataset
from dataset.pre_processing import (
    compose_transforms,
    random_shift3d,
)
from network.models import VAE, VQVAE, CVAE, HybridVAEUNet, GMVAE2_3D, VAE_CBAM
from network.training import apply_label_mask, NormalizeImage
from network.losses import calc_recon_loss
from utils.helper import save_as_itk


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def resolve_run_paths(cf: Dict) -> Dict:
    cf = copy.deepcopy(cf)
    base_exp_name = cf["exp_name"]

    alpha = cf.get("alpha", None)
    kld_weight = cf.get("kld_weight", None)
    weight_mode = cf.get("weight_mode", None)

    if isinstance(alpha, list):
        if len(alpha) != 1:
            raise ValueError(f"Expected a single alpha value for evaluation, got {alpha}")
        alpha = alpha[0]

    if isinstance(kld_weight, list):
        if len(kld_weight) != 1:
            raise ValueError(f"Expected a single kld_weight value for evaluation, got {kld_weight}")
        kld_weight = kld_weight[0]

    if isinstance(weight_mode, list):
        if len(weight_mode) != 1:
            raise ValueError(f"Expected a single weight_mode value for evaluation, got {weight_mode}")
        weight_mode = weight_mode[0]

    cf["alpha"] = alpha
    cf["kld_weight"] = kld_weight
    cf["weight_mode"] = weight_mode
    cf["exp_name"] = f"{base_exp_name}_alpha{alpha}_kld{kld_weight}_wm{weight_mode}"
    cf["output_folder"] = os.path.join(cf["save_dir"], cf["exp_name"])
    return cf


def extract_patient_id(path: str) -> str:
    fname = os.path.basename(path)
    m = re.match(r"(?:labels|ct)_patient_(.+)\.nii\.gz$", fname)
    if m is None:
        raise ValueError(f"Could not extract patient id from filename: {fname}")
    return m.group(1)


def get_ct_path_from_label_path(label_path: str) -> str:
    label_path = os.path.normpath(label_path)
    label_dir = os.path.dirname(label_path)
    parent_dir = os.path.dirname(label_dir)

    fname = os.path.basename(label_path)
    if not fname.startswith("labels_patient_"):
        raise ValueError(f"Unexpected label filename: {fname}")

    ct_fname = fname.replace("labels_patient_", "ct_patient_", 1)
    ct_dir = os.path.join(parent_dir, "ct")
    ct_path = os.path.join(ct_dir, ct_fname)

    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found for label file {label_path}: {ct_path}")

    return ct_path


def load_ct_from_label_path(label_path: str) -> np.ndarray:
    ct_path = get_ct_path_from_label_path(label_path)
    ct_img = sitk.ReadImage(ct_path)
    return sitk.GetArrayFromImage(ct_img)


def make_model(cf: Dict, device: torch.device) -> torch.nn.Module:
    """
    Build model for probe evaluation.

    Original version only supported model='ae'. This version supports at least:
      - ae
      - vqvae
      - vae / beta_vae / infovae / paired_vae / paired_vae_label / soft_intro_vae
      - cvae
      - hybrid_vae_unet
      - GMVAE2_3D
      - vae_cbam

    The probe task still assumes binary label reconstruction with logits, which is
    appropriate for your Bronchus VQ-VAE config: out_ch=1, final_activation=Identity.
    """
    model_name = cf["model"]

    if model_name in ["beta_vae", "vae", "infovae", "paired_vae", "paired_vae_label", "ae", "soft_intro_vae"]:
        model = VAE(cf, embedding=False).to(device)
    elif model_name == "vqvae":
        model = VQVAE(cf, embedding=False).to(device)
    elif model_name == "cvae":
        model = CVAE(cf, embedding=False).to(device)
    elif model_name == "hybrid_vae_unet":
        model = HybridVAEUNet(cf, embedding=False).to(device)
    elif model_name == "GMVAE2_3D":
        model = GMVAE2_3D(cf, embedding=False).to(device)
    elif model_name == "vae_cbam":
        model = VAE_CBAM(cf, embedding=False).to(device)
    else:
        raise ValueError(f"Unsupported model for probe evaluation: {model_name}")

    ckpt_dir = os.path.join(cf["output_folder"], "checkpoints")
    load_weights_path = os.path.join(ckpt_dir, cf["ckpt_file_load"])
    print(f"\n> Loading weights from: {load_weights_path}")

    if not os.path.exists(load_weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_weights_path}")

    model.load_state_dict(torch.load(load_weights_path, map_location=device))
    model.eval()
    return model


def get_test_loader(cf: Dict) -> DataLoader:
    test_cf = copy.deepcopy(cf)
    test_cf["train"] = False
    pre_processing = compose_transforms(test_cf)

    ds = ProjectionDataset(
        cf["test_dir"],
        transform=pre_processing,
        cf=test_cf,
    )

    return DataLoader(
        ds,
        batch_size=cf["batch_size"],
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )


def preprocess_clean_target(x_target, cf):
    x_target = NormalizeImage(x_target, cf)
    x_target = apply_label_mask(x_target, cf)
    return (x_target == 1).float()


def preprocess_probe_input(x_in, cf):
    x_in = NormalizeImage(x_in, cf)
    return (x_in == 1).float()


def dice_score_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    denominator = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + eps
    return (2.0 * intersection + eps) / denominator


def hd95_binary(pred: torch.Tensor, target: torch.Tensor, spacing: Tuple[float, float, float]) -> torch.Tensor:
    from surface_distance import metrics

    results = []
    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)

    for b in range(pred_np.shape[0]):
        p = pred_np[b, 0]
        t = target_np[b, 0]

        if (not p.any()) and (not t.any()):
            results.append(0.0)
            continue
        if (not p.any()) or (not t.any()):
            results.append(np.inf)
            continue

        try:
            surf = metrics.compute_surface_distances(t, p, spacing)
            hd95 = metrics.compute_robust_hausdorff(surf, 95.0)
            results.append(float(hd95))
        except Exception as e:
            print(f"HD95 computation failed for batch item {b}: {e}")
            results.append(np.inf)

    return torch.tensor(results, dtype=torch.float32)


def volume_fraction(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().mean(dim=(1, 2, 3, 4))


def volume_diff_abs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    va = a.float().sum(dim=(1, 2, 3, 4))
    vb = b.float().sum(dim=(1, 2, 3, 4))
    return torch.abs(va - vb)


def choose_case_indices(dataset_size: int, ratio: float, seed: int) -> Set[int]:
    if dataset_size <= 0:
        return set()
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"ratio must be in (0,1], got {ratio}")

    rng = random.Random(seed)
    n_keep = max(1, int(round(dataset_size * ratio)))
    n_keep = min(n_keep, dataset_size)
    return set(rng.sample(list(range(dataset_size)), n_keep))


def get_foreground_center(x: torch.Tensor):
    fg = (x[0] > 0).nonzero(as_tuple=False)
    if fg.numel() == 0:
        return None
    center = fg.float().mean(dim=0)
    return tuple(center.round().long().tolist())


def get_foreground_center_and_bbox(x):
    fg = (x[0] > 0).nonzero(as_tuple=False)
    if fg.numel() == 0:
        return None, None, None
    center = fg.float().mean(dim=0)
    mins = fg.min(dim=0).values
    maxs = fg.max(dim=0).values
    return center, mins, maxs


def cut_lower_half(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    center = get_foreground_center(y)
    if center is None:
        return y
    zc, _, _ = center
    y[:, zc:, :, :] = 0
    return y


def cut_upper_half(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    center = get_foreground_center(y)
    if center is None:
        return y
    zc, _, _ = center
    y[:, :zc, :, :] = 0
    return y


def cut_posterior_half(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    center = get_foreground_center(y)
    if center is None:
        return y
    _, yc, _ = center
    y[:, :, yc:, :] = 0
    return y


def _sitk_from_binary_tensor(x: torch.Tensor) -> sitk.Image:
    arr = (x.detach().cpu().numpy()[0] > 0.5).astype(np.uint8)
    return sitk.GetImageFromArray(arr)


def _binary_tensor_from_sitk(img: sitk.Image, device: torch.device) -> torch.Tensor:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = (arr > 0).astype(np.float32)
    return torch.from_numpy(arr[None]).to(device)


def sitk_binary_erode3d(x: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if iterations == 0:
        return (x > 0.5).float()

    radius = max(0, (kernel_size - 1) // 2)
    img = _sitk_from_binary_tensor(x)
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelType(sitk.sitkBall)
    eroder.SetKernelRadius([radius, radius, radius])
    eroder.SetForegroundValue(1)
    eroder.SetBackgroundValue(0)
    eroder.SetBoundaryToForeground(False)

    out = img
    for _ in range(iterations):
        out = eroder.Execute(out)
    return _binary_tensor_from_sitk(out, x.device)


def sitk_binary_dilate3d(x: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if iterations == 0:
        return (x > 0.5).float()

    radius = max(0, (kernel_size - 1) // 2)
    img = _sitk_from_binary_tensor(x)
    dilater = sitk.BinaryDilateImageFilter()
    dilater.SetKernelType(sitk.sitkBall)
    dilater.SetKernelRadius([radius, radius, radius])
    dilater.SetForegroundValue(1)
    dilater.SetBackgroundValue(0)
    dilater.SetBoundaryToForeground(False)

    out = img
    for _ in range(iterations):
        out = dilater.Execute(out)
    return _binary_tensor_from_sitk(out, x.device)


def get_surface_voxels(x: torch.Tensor, kernel_size: int = 3):
    fg = (x > 0.5).float()
    if fg.sum() == 0:
        return None

    eroded = sitk_binary_erode3d(fg, kernel_size=kernel_size, iterations=1)
    surface = (fg > 0.5) & ~(eroded > 0.5)
    coords = surface[0].nonzero(as_tuple=False)
    if coords.numel() == 0:
        coords = fg[0].nonzero(as_tuple=False)
    return coords


def choose_surface_anchor(x: torch.Tensor, anchor_mode: str = "random", kernel_size: int = 3):
    coords = get_surface_voxels(x, kernel_size=kernel_size)
    if coords is None or coords.numel() == 0:
        return None

    if anchor_mode == "random":
        idx = torch.randint(low=0, high=coords.shape[0], size=(1,)).item()
    elif anchor_mode == "extreme_x":
        idx = torch.argmax(coords[:, 2]).item()
    elif anchor_mode == "extreme_z":
        idx = torch.argmax(coords[:, 0]).item()
    else:
        raise ValueError(f"Unknown surface anchor mode: {anchor_mode}")

    return tuple(coords[idx].tolist())


def big_surface_bite(x, frac=(0.5, 0.5, 0.5), anchor_mode="random", surface_kernel=3, min_block=8):
    y = x.clone()
    center, mins, maxs = get_foreground_center_and_bbox(y)
    if center is None:
        return y

    anchor = choose_surface_anchor(y, anchor_mode=anchor_mode, kernel_size=surface_kernel)
    if anchor is None:
        return y

    spans = (maxs - mins + 1).float()
    zc, yc, xc = anchor

    dz = max(min_block, int(round(spans[0].item() * frac[0])))
    dy = max(min_block, int(round(spans[1].item() * frac[1])))
    dx = max(min_block, int(round(spans[2].item() * frac[2])))

    z0 = max(0, zc - dz // 2)
    y0 = max(0, yc - dy // 2)
    x0 = max(0, xc - dx // 2)
    z1 = min(y.shape[1], z0 + dz)
    y1 = min(y.shape[2], y0 + dy)
    x1 = min(y.shape[3], x0 + dx)

    y[:, z0:z1, y0:y1, x0:x1] = 0
    return y


def connected_components_touching_original(merged_mask: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    merged_mask = merged_mask.astype(bool)
    original_mask = original_mask.astype(bool)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, nlab = ndimage.label(merged_mask, structure=structure)

    if nlab == 0:
        return merged_mask

    touching_labels = np.unique(labels[original_mask])
    touching_labels = touching_labels[touching_labels != 0]
    if len(touching_labels) == 0:
        return merged_mask
    return np.isin(labels, touching_labels)


def add_surface_blob(
    x,
    frac=(0.5, 0.5, 0.5),
    anchor_mode="random",
    surface_kernel=3,
    min_block=8,
    inward_fraction=0.35,
    keep_connected=True,
):
    y = x.clone()
    original = (x > 0.5).float()

    center, mins, maxs = get_foreground_center_and_bbox(y)
    if center is None:
        return y

    anchor = choose_surface_anchor(y, anchor_mode=anchor_mode, kernel_size=surface_kernel)
    if anchor is None:
        return y

    anchor_t = torch.tensor(anchor, dtype=torch.float32, device=center.device)
    spans = (maxs - mins + 1).float()
    dz = max(min_block, int(round(spans[0].item() * frac[0])))
    dy = max(min_block, int(round(spans[1].item() * frac[1])))
    dx = max(min_block, int(round(spans[2].item() * frac[2])))

    direction = center - anchor_t
    norm = torch.norm(direction) + 1e-6
    direction = direction / norm

    shift = torch.tensor(
        [
            inward_fraction * dz * direction[0],
            inward_fraction * dy * direction[1],
            inward_fraction * dx * direction[2],
        ],
        dtype=torch.float32,
        device=center.device,
    )

    blob_center = anchor_t + shift
    zc, yc, xc = blob_center.round().long().tolist()

    z0 = max(0, zc - dz // 2)
    y0 = max(0, yc - dy // 2)
    x0 = max(0, xc - dx // 2)
    z1 = min(y.shape[1], z0 + dz)
    y1 = min(y.shape[2], y0 + dy)
    x1 = min(y.shape[3], x0 + dx)

    y[:, z0:z1, y0:y1, x0:x1] = 1.0
    y = (y > 0.5).float()

    if keep_connected:
        original_np = original[0].cpu().numpy().astype(bool)
        y_np = y[0].cpu().numpy().astype(bool)
        connected = connected_components_touching_original(y_np, original_np)
        y = torch.from_numpy(connected[None].astype(np.float32)).to(x.device)

    return y


def corrupt_clean_binary(
    clean_binary: torch.Tensor,
    mode: str,
    shift_vox: int = 3,
    erode_iters: int = 1,
    dilate_iters: int = 1,
    morph_kernel: int = 3,
    strong_shift_vox: int = 5,
    strong_erode_iters: int = 1,
    strong_dilate_iters: int = 1,
    surface_anchor_mode: str = "random",
    surface_kernel: int = 3,
    bite_frac: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    blob_frac: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    bite_min_block: int = 8,
    blob_min_block: int = 8,
) -> torch.Tensor:
    x = clean_binary.clone()
    if mode == "clean":
        return x

    out = []
    for b in range(x.shape[0]):
        xb = x[b]

        if mode == "shifted":
            xb = random_shift3d(xb, max_shift=shift_vox)
        elif mode == "eroded":
            xb = sitk_binary_erode3d(xb, kernel_size=morph_kernel, iterations=erode_iters)
        elif mode == "dilated":
            xb = sitk_binary_dilate3d(xb, kernel_size=morph_kernel, iterations=dilate_iters)
        elif mode == "strong":
            xb = random_shift3d(xb, max_shift=strong_shift_vox)
            xb = sitk_binary_erode3d(xb, kernel_size=morph_kernel, iterations=strong_erode_iters)
            xb = sitk_binary_dilate3d(xb, kernel_size=morph_kernel, iterations=strong_dilate_iters)
        elif mode == "cut_lower_half":
            xb = cut_lower_half(xb)
        elif mode == "cut_upper_half":
            xb = cut_upper_half(xb)
        elif mode == "cut_posterior_half":
            xb = cut_posterior_half(xb)
        elif mode == "big_bite":
            xb = big_surface_bite(xb, frac=bite_frac, anchor_mode=surface_anchor_mode, surface_kernel=surface_kernel, min_block=bite_min_block)
        elif mode == "big_blob":
            xb = add_surface_blob(xb, frac=blob_frac, anchor_mode=surface_anchor_mode, surface_kernel=surface_kernel, min_block=blob_min_block)
        else:
            raise ValueError(f"Unknown corruption mode: {mode}")

        xb = (xb > 0.5).float()
        out.append(xb)

    return torch.stack(out, dim=0)


@torch.no_grad()
def forward_reconstruction(model: torch.nn.Module, x_in: torch.Tensor, cf: Dict) -> torch.Tensor:
    """
    Return reconstruction logits [B,1,D,H,W] for different model families.
    """
    model_name = cf["model"]

    if model_name == "ae":
        x_hat, _, _, _ = model(x_in, deterministic=True)
    elif model_name == "vqvae":
        x_hat, _, _, _, _, _ = model(x_in, deterministic=True)
    elif model_name == "cvae":
        b_size = x_in.shape[0]
        labels = x_in.squeeze(1).long().view(b_size, -1)
        cond = torch.zeros(b_size, cf["out_ch"], device=x_in.device)
        for i in range(b_size):
            counts = torch.bincount(labels[i], minlength=cf["out_ch"]).float()
            denom = counts.sum().clamp_min(1.0)
            cond[i] = counts / denom
        x_hat, _, _, _ = model(x_in, cond, deterministic=True)
    elif model_name == "hybrid_vae_unet":
        _, x_hat, _, _, _ = model(x_in, deterministic=True)
    elif model_name == "GMVAE2_3D":
        x_hat, _, _, _, _, _, _ = model(x_in)
    else:
        # VAE-like models: vae, beta_vae, infovae, vae_cbam, etc.
        x_hat, _, _, _ = model(x_in, deterministic=True)

    return x_hat


@torch.no_grad()
def run_probe_case(model: torch.nn.Module, x_in: torch.Tensor, x_target: torch.Tensor, cf: Dict) -> Dict[str, torch.Tensor]:
    x_hat = forward_reconstruction(model, x_in, cf)

    loss = calc_recon_loss(
        x_hat,
        x_target,
        loss_type=cf["recon_loss"],
        reduction="none",
        weight_mode=cf.get("weight_mode", "uniform"),
        return_per_class=False,
        alpha=cf.get("alpha", 0.5),
        epoch=cf.get("num_epochs", 0),
    )

    prob = torch.sigmoid(x_hat)
    pred = (prob > 0.5).float()
    spacing = tuple(float(s) for s in cf["proj_spacing"])

    return {
        "loss": loss.detach().cpu(),
        "reconstruction": pred.detach().cpu(),
        "dice_recon_vs_target": dice_score_binary(pred, x_target).detach().cpu(),
        "dice_input_vs_recon": dice_score_binary(x_in, pred).detach().cpu(),
        "dice_input_vs_target": dice_score_binary(x_in, x_target).detach().cpu(),
        "hd95_recon_vs_target": hd95_binary(pred, x_target, spacing).detach().cpu(),
        "hd95_input_vs_recon": hd95_binary(x_in, pred, spacing).detach().cpu(),
        "hd95_input_vs_target": hd95_binary(x_in, x_target, spacing).detach().cpu(),
        "volfrac_input": volume_fraction(x_in).detach().cpu(),
        "volfrac_recon": volume_fraction(pred).detach().cpu(),
        "volfrac_target": volume_fraction(x_target).detach().cpu(),
        "vol_diff_recon_target": volume_diff_abs(pred, x_target).detach().cpu(),
        "vol_diff_input_target": volume_diff_abs(x_in, x_target).detach().cpu(),
    }


def save_probe_visualization(clean_input: torch.Tensor, corrupted_input: torch.Tensor, reconstruction: torch.Tensor, patient_id: str, probe_type: str, label_path: str, cf: Dict, vis_dir: str) -> None:
    clean_np = clean_input.squeeze(0).detach().cpu().numpy().astype(np.int16)
    corrupted_np = corrupted_input.squeeze(0).detach().cpu().numpy().astype(np.int16)
    recon_np = reconstruction.squeeze(0).detach().cpu().numpy().astype(np.int16)
    ct_np = load_ct_from_label_path(label_path).astype(np.int16)

    clean_path = os.path.join(vis_dir, f"clean_input_patient_{patient_id}.nii.gz")
    corrupted_path = os.path.join(vis_dir, f"corrupted_input_{probe_type}_patient_{patient_id}.nii.gz")
    recon_path = os.path.join(vis_dir, f"reconstruction_{probe_type}_patient_{patient_id}.nii.gz")
    ct_path = os.path.join(vis_dir, f"original_ct_patient_{patient_id}.nii.gz")

    if not os.path.exists(clean_path):
        save_as_itk(clean_np, clean_path, spacing=cf["proj_spacing"], direction=cf["proj_direction"])
    if not os.path.exists(ct_path):
        save_as_itk(ct_np, ct_path, spacing=cf["proj_spacing"], direction=cf["proj_direction"])

    save_as_itk(corrupted_np, corrupted_path, spacing=cf["proj_spacing"], direction=cf["proj_direction"])
    save_as_itk(recon_np, recon_path, spacing=cf["proj_spacing"], direction=cf["proj_direction"])


def summarize_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("probe_type").agg(
        n=("case_id", "count"),
        loss_mean=("loss", "mean"),
        loss_std=("loss", "std"),
        dice_recon_vs_target_mean=("dice_recon_vs_target", "mean"),
        dice_recon_vs_target_std=("dice_recon_vs_target", "std"),
        dice_input_vs_recon_mean=("dice_input_vs_recon", "mean"),
        dice_input_vs_recon_std=("dice_input_vs_recon", "std"),
        dice_input_vs_target_mean=("dice_input_vs_target", "mean"),
        dice_input_vs_target_std=("dice_input_vs_target", "std"),
        hd95_recon_vs_target_mean=("hd95_recon_vs_target", "mean"),
        hd95_recon_vs_target_std=("hd95_recon_vs_target", "std"),
        hd95_input_vs_recon_mean=("hd95_input_vs_recon", "mean"),
        hd95_input_vs_recon_std=("hd95_input_vs_recon", "std"),
        hd95_input_vs_target_mean=("hd95_input_vs_target", "mean"),
        hd95_input_vs_target_std=("hd95_input_vs_target", "std"),
        volfrac_input_mean=("volfrac_input", "mean"),
        volfrac_recon_mean=("volfrac_recon", "mean"),
        volfrac_target_mean=("volfrac_target", "mean"),
        vol_diff_recon_target_mean=("vol_diff_recon_target", "mean"),
        vol_diff_input_target_mean=("vol_diff_input_target", "mean"),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Post-DAE probe evaluation")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--shift_vox", type=int, default=3, help="Max voxel shift for 'shifted'")
    parser.add_argument("--erode_iters", type=int, default=1, help="Erosion iterations for 'eroded'")
    parser.add_argument("--dilate_iters", type=int, default=1, help="Dilation iterations for 'dilated'")
    parser.add_argument("--morph_kernel", type=int, default=3, help="Morphology kernel size (odd integer; radius=(kernel-1)//2)")
    parser.add_argument("--strong_shift_vox", type=int, default=5, help="Max voxel shift for 'strong'")
    parser.add_argument("--strong_erode_iters", type=int, default=1, help="Erosion iterations for 'strong'")
    parser.add_argument("--strong_dilate_iters", type=int, default=1, help="Dilation iterations for 'strong'")
    parser.add_argument("--surface_anchor_mode", type=str, default="random", choices=["random", "extreme_x", "extreme_z"], help="How to choose surface anchor for big_bite / big_blob")
    parser.add_argument("--surface_kernel", type=int, default=3, help="Kernel size for deriving ROI surface by erosion")
    parser.add_argument("--bite_frac_z", type=float, default=0.5)
    parser.add_argument("--bite_frac_y", type=float, default=0.5)
    parser.add_argument("--bite_frac_x", type=float, default=0.5)
    parser.add_argument("--blob_frac_z", type=float, default=0.5)
    parser.add_argument("--blob_frac_y", type=float, default=0.5)
    parser.add_argument("--blob_frac_x", type=float, default=0.5)
    parser.add_argument("--bite_min_block", type=int, default=8)
    parser.add_argument("--blob_min_block", type=int, default=8)
    parser.add_argument("--save_vis", action="store_true", help="Save NIfTI visualizations for a subset of cases")
    parser.add_argument("--vis_ratio", type=float, default=0.05, help="Fraction of test cases to visualize if --save_vis is used")
    args = parser.parse_args()

    cf = load_config(args.config)
    cf = resolve_run_paths(cf)

    print(f"\nUsing config: {args.config}")
    print(f"Resolved exp_name: {cf['exp_name']}")
    print(f"Resolved output_folder: {cf['output_folder']}")
    print(f"Resolved alpha: {cf['alpha']} ({type(cf['alpha'])})")
    print(f"Resolved kld_weight: {cf['kld_weight']} ({type(cf['kld_weight'])})")
    print(f"Resolved weight_mode: {cf['weight_mode']} ({type(cf['weight_mode'])})")

    if "ori_size" not in cf:
        cf["ori_size"] = cf["proj_size"][::-1]

    device = torch.device(cf["device"])
    model = make_model(cf, device)
    loader = get_test_loader(cf)

    output_dir = os.path.join(cf["output_folder"], "probe_eval")
    os.makedirs(output_dir, exist_ok=True)

    vis_dir = os.path.join(output_dir, "visualization")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    dataset_size = len(loader.dataset)
    vis_case_indices = set()
    if args.save_vis:
        vis_case_indices = choose_case_indices(dataset_size=dataset_size, ratio=args.vis_ratio, seed=cf.get("manual_seed", 88))
        print(f"Saving visualization for {len(vis_case_indices)} / {dataset_size} cases")

    probe_modes = [
        "clean",
        "shifted",
        "eroded",
        "dilated",
        "strong",
        "cut_lower_half",
        "cut_upper_half",
        "cut_posterior_half",
        "big_bite",
        "big_blob",
    ]

    bite_frac = (args.bite_frac_z, args.bite_frac_y, args.bite_frac_x)
    blob_frac = (args.blob_frac_z, args.blob_frac_y, args.blob_frac_x)

    rows: List[Dict] = []
    case_counter = 0

    for batch in loader:
        x_target_raw = batch["target_image"].to(device)
        names = batch["proj_name"]
        x_target = preprocess_clean_target(x_target_raw, cf)

        for probe_type in probe_modes:
            x_in = corrupt_clean_binary(
                x_target,
                mode=probe_type,
                shift_vox=args.shift_vox,
                erode_iters=args.erode_iters,
                dilate_iters=args.dilate_iters,
                morph_kernel=args.morph_kernel,
                strong_shift_vox=args.strong_shift_vox,
                strong_erode_iters=args.strong_erode_iters,
                strong_dilate_iters=args.strong_dilate_iters,
                surface_anchor_mode=args.surface_anchor_mode,
                surface_kernel=args.surface_kernel,
                bite_frac=bite_frac,
                blob_frac=blob_frac,
                bite_min_block=args.bite_min_block,
                blob_min_block=args.blob_min_block,
            )
            x_in = preprocess_probe_input(x_in, cf)
            metrics = run_probe_case(model, x_in, x_target, cf)

            bsz = x_in.shape[0]
            for i in range(bsz):
                global_idx = case_counter + i
                if args.save_vis and global_idx in vis_case_indices:
                    patient_id = extract_patient_id(names[i])
                    save_probe_visualization(
                        clean_input=x_target[i].cpu(),
                        corrupted_input=x_in[i].cpu(),
                        reconstruction=metrics["reconstruction"][i],
                        patient_id=patient_id,
                        probe_type=probe_type,
                        label_path=names[i],
                        cf=cf,
                        vis_dir=vis_dir,
                    )

                rows.append({
                    "case_id": global_idx,
                    "file": os.path.basename(names[i]),
                    "probe_type": probe_type,
                    "shift_vox": args.shift_vox,
                    "erode_iters": args.erode_iters,
                    "dilate_iters": args.dilate_iters,
                    "morph_kernel": args.morph_kernel,
                    "strong_shift_vox": args.strong_shift_vox,
                    "strong_erode_iters": args.strong_erode_iters,
                    "strong_dilate_iters": args.strong_dilate_iters,
                    "surface_anchor_mode": args.surface_anchor_mode,
                    "surface_kernel": args.surface_kernel,
                    "bite_frac_z": args.bite_frac_z,
                    "bite_frac_y": args.bite_frac_y,
                    "bite_frac_x": args.bite_frac_x,
                    "blob_frac_z": args.blob_frac_z,
                    "blob_frac_y": args.blob_frac_y,
                    "blob_frac_x": args.blob_frac_x,
                    "bite_min_block": args.bite_min_block,
                    "blob_min_block": args.blob_min_block,
                    "loss": float(metrics["loss"][i].item()),
                    "dice_recon_vs_target": float(metrics["dice_recon_vs_target"][i].item()),
                    "dice_input_vs_recon": float(metrics["dice_input_vs_recon"][i].item()),
                    "dice_input_vs_target": float(metrics["dice_input_vs_target"][i].item()),
                    "hd95_recon_vs_target": float(metrics["hd95_recon_vs_target"][i].item()),
                    "hd95_input_vs_recon": float(metrics["hd95_input_vs_recon"][i].item()),
                    "hd95_input_vs_target": float(metrics["hd95_input_vs_target"][i].item()),
                    "volfrac_input": float(metrics["volfrac_input"][i].item()),
                    "volfrac_recon": float(metrics["volfrac_recon"][i].item()),
                    "volfrac_target": float(metrics["volfrac_target"][i].item()),
                    "vol_diff_recon_target": float(metrics["vol_diff_recon_target"][i].item()),
                    "vol_diff_input_target": float(metrics["vol_diff_input_target"][i].item()),
                })

        case_counter += x_target.shape[0]

    df = pd.DataFrame(rows)
    summary = summarize_table(df)

    detailed_csv = os.path.join(output_dir, "probe_eval_detailed.csv")
    summary_csv = os.path.join(output_dir, "probe_eval_summary.csv")
    df.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== Probe evaluation summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to: {detailed_csv}")
    print(f"Saved summary results to:  {summary_csv}")
    if args.save_vis:
        print(f"Saved visualizations to:   {vis_dir}")


if __name__ == "__main__":
    main()
