#!/usr/bin/env python3
"""
Smoke test for the pretrained 3DINO-ViT checkpoint.

Tests:
  1. Repository imports and configuration loading.
  2. Pretrained checkpoint loading.
  3. One-volume inference with the official 112^3 input shape.
  4. Global feature extraction.
  5. Spatial patch-token extraction and reshape to [B, C, 7, 7, 7].

Example
-------
cd /home/shaohan/3DINO

python test_3dino_checkpoint_features.py \
    --volume /path/to/ct_patient_1.nii.gz

To test only the checkpoint/model with a synthetic volume:
python test_3dino_checkpoint_features.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_REPO = Path("/home/shaohan/3DINO")
DEFAULT_WEIGHTS = DEFAULT_REPO / "pretrained" / "3dino_vit_weights.pth"
DEFAULT_CONFIG = "train/vit3d_highres"
DEFAULT_INPUT_SIZE = (112, 112, 112)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load 3DINO-ViT and extract global and patch-token features."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
        help=f"3DINO repository path. Default: {DEFAULT_REPO}",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Checkpoint path. Default: {DEFAULT_WEIGHTS}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"3DINO config name. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--volume",
        type=Path,
        default=None,
        help=(
            "Optional NIfTI CT volume (.nii or .nii.gz). "
            "If omitted, a synthetic 112^3 volume is used."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Inference device. The official builder may require CUDA.",
    )
    parser.add_argument(
        "--save-features",
        type=Path,
        default=None,
        help="Optional .pt output path for extracted features.",
    )
    return parser.parse_args()


def validate_paths(repo: Path, weights: Path, volume: Path | None) -> None:
    if not repo.is_dir():
        raise FileNotFoundError(f"3DINO repository not found: {repo}")

    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    if volume is not None and not volume.is_file():
        raise FileNotFoundError(f"Input volume not found: {volume}")


def print_checkpoint_summary(weights: Path) -> None:
    print("\n=== Checkpoint inspection ===")
    print(f"Checkpoint: {weights}")
    print(f"Size: {weights.stat().st_size / (1024 ** 2):.1f} MiB")

    checkpoint = torch.load(weights, map_location="cpu")

    print(f"Loaded Python type: {type(checkpoint).__name__}")
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"Top-level keys ({len(keys)}): {keys[:30]}")
        if len(keys) > 30:
            print("  ...")

        # Give a compact view of likely nested state dictionaries.
        for candidate in ("teacher", "student", "model", "state_dict"):
            value = checkpoint.get(candidate)
            if isinstance(value, dict):
                nested_keys = list(value.keys())
                print(
                    f"Nested '{candidate}' keys ({len(nested_keys)}): "
                    f"{nested_keys[:10]}"
                )
    del checkpoint


def import_3dino(repo: Path):
    # The notebook adds the repository root to sys.path and loads these APIs.
    repo_string = str(repo.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    # Some config paths in the repository are interpreted relative to repo root.
    os.chdir(repo)

    try:
        from dinov2.eval.setup import build_model_for_eval
        from dinov2.configs import load_and_merge_config_3d
    except Exception as exc:
        raise RuntimeError(
            "Could not import 3DINO. Activate the 3DINO environment and run "
            "this script from the cloned repository.\n"
            f"Original import error: {exc}"
        ) from exc

    return build_model_for_eval, load_and_merge_config_3d


def load_model(
    repo: Path,
    weights: Path,
    config_name: str,
    requested_device: str,
) -> tuple[torch.nn.Module, torch.device]:
    build_model_for_eval, load_and_merge_config_3d = import_3dino(repo)

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check that the CUDA-enabled PyTorch build is installed."
        )

    print("\n=== Building 3DINO-ViT ===")
    print(f"Config: {config_name}")
    cfg = load_and_merge_config_3d(config_name)

    # This follows the official notebook interface exactly.
    model = build_model_for_eval(cfg, str(weights))
    model.eval()

    # The official build helper normally places the model on CUDA.
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(requested_device)

    requested = torch.device(requested_device)
    if model_device != requested:
        print(f"Moving model from {model_device} to {requested}.")
        model = model.to(requested)
        model_device = requested

    print(f"Model class: {model.__class__.__name__}")
    print(f"Model device: {model_device}")
    print(
        f"Parameters: {sum(p.numel() for p in model.parameters()):,} total; "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable"
    )

    patch_embed = getattr(model, "patch_embed", None)
    projection = getattr(patch_embed, "proj", None)
    if projection is not None:
        print(f"Patch projection: {projection}")

    return model, model_device


def read_nifti(path: Path) -> np.ndarray:
    """
    Read a NIfTI image as a float32 [D, H, W] array.

    Tries SimpleITK first because the AIQA project already uses it, then nibabel.
    """
    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)  # [D, H, W]
        print(f"Loaded with SimpleITK; spacing xyz: {image.GetSpacing()}")
        return np.asarray(array, dtype=np.float32)
    except ImportError:
        pass

    try:
        import nibabel as nib

        image = nib.load(str(path))
        array_xyz = np.asarray(image.get_fdata(dtype=np.float32))
        # Convert nibabel [X, Y, Z] convention to [D=Z, H=Y, W=X].
        array_zyx = np.transpose(array_xyz, (2, 1, 0))
        print(f"Loaded with nibabel; voxel sizes xyz: {image.header.get_zooms()[:3]}")
        return np.asarray(array_zyx, dtype=np.float32)
    except ImportError as exc:
        raise RuntimeError(
            "Neither SimpleITK nor nibabel is installed. Install one with:\n"
            "  pip install SimpleITK\n"
            "or\n"
            "  pip install nibabel"
        ) from exc


def center_crop_or_pad(
    volume: torch.Tensor,
    output_size: tuple[int, int, int],
    pad_value: float,
) -> torch.Tensor:
    """
    Center crop or pad [1, 1, D, H, W] to output_size.
    """
    if volume.ndim != 5 or volume.shape[:2] != (1, 1):
        raise ValueError(f"Expected [1,1,D,H,W], got {tuple(volume.shape)}")

    # Center crop dimensions that are too large.
    slices = [slice(None), slice(None)]
    for current, target in zip(volume.shape[2:], output_size):
        if current > target:
            start = (current - target) // 2
            slices.append(slice(start, start + target))
        else:
            slices.append(slice(None))
    volume = volume[tuple(slices)]

    # Symmetrically pad dimensions that are too small.
    pads: list[int] = []
    for current, target in reversed(list(zip(volume.shape[2:], output_size))):
        total = max(target - current, 0)
        before = total // 2
        after = total - before
        pads.extend([before, after])

    if any(pads):
        volume = F.pad(volume, pads, mode="constant", value=float(pad_value))

    if tuple(volume.shape[2:]) != output_size:
        raise RuntimeError(
            f"Crop/pad failed: got {tuple(volume.shape[2:])}, expected {output_size}"
        )
    return volume


def percentile_normalize_to_minus_one_one(
    image: torch.Tensor,
    lower_quantile: float = 0.0005,
    upper_quantile: float = 0.9995,
) -> tuple[torch.Tensor, float, float]:
    """
    Follow the official notebook:
      q0.05% -> -1
      q99.95% -> +1
      clip outside [-1, 1]
    """
    image_float = image.float()
    min_val = torch.quantile(image_float, lower_quantile)
    max_val = torch.quantile(image_float, upper_quantile)

    denominator = max_val - min_val
    if denominator.abs().item() < 1e-8:
        raise ValueError(
            "Volume has nearly constant intensity; percentile normalization is undefined."
        )

    normalized = (image_float - min_val) / denominator
    normalized = torch.clamp(normalized * 2.0 - 1.0, -1.0, 1.0)

    return normalized, float(min_val.item()), float(max_val.item())


def prepare_volume(
    volume_path: Path | None,
    device: torch.device,
) -> torch.Tensor:
    print("\n=== Preparing input ===")

    if volume_path is None:
        print("No --volume supplied; using a synthetic random 112^3 volume.")
        image = torch.randn(1, 1, *DEFAULT_INPUT_SIZE, dtype=torch.float32)
    else:
        array = read_nifti(volume_path)
        print(
            f"Original array: shape={array.shape}, dtype={array.dtype}, "
            f"min={np.nanmin(array):.3f}, max={np.nanmax(array):.3f}"
        )

        if not np.isfinite(array).all():
            raise ValueError("Input volume contains NaN or infinite values.")

        image = torch.from_numpy(array)[None, None]  # [1,1,D,H,W]

        # Pad with the low-intensity end of the actual volume.
        pad_value = float(torch.quantile(image.float(), 0.0005).item())
        image = center_crop_or_pad(image, DEFAULT_INPUT_SIZE, pad_value=pad_value)

    image, q_low, q_high = percentile_normalize_to_minus_one_one(image)

    print(f"Input tensor shape: {tuple(image.shape)}")
    print(f"Normalization percentiles: low={q_low:.4f}, high={q_high:.4f}")
    print(
        f"Normalized range: [{image.min().item():.4f}, "
        f"{image.max().item():.4f}]"
    )

    return image.to(device=device, non_blocking=True)


def summarize_feature(name: str, value: Any) -> None:
    if torch.is_tensor(value):
        finite = torch.isfinite(value).all().item()
        print(
            f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device}, finite={finite}, "
            f"mean={value.float().mean().item():.6f}, "
            f"std={value.float().std().item():.6f}"
        )
    else:
        print(f"{name}: {type(value).__name__}")


def extract_features(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> dict[str, torch.Tensor]:
    print("\n=== Feature extraction ===")

    outputs: dict[str, torch.Tensor] = {}

    with torch.inference_mode():
        # Official notebook output: global 1024-dimensional class-token feature.
        global_feature = model(image)
        outputs["global_feature"] = global_feature
        summarize_feature("Global feature", global_feature)

        # Dense features needed for a future reconstruction decoder.
        if not hasattr(model, "forward_features"):
            raise AttributeError(
                "Loaded model does not expose forward_features(); "
                "cannot extract spatial patch tokens."
            )

        feature_dict = model.forward_features(image)
        if not isinstance(feature_dict, dict):
            raise TypeError(
                "Expected model.forward_features() to return a dict, "
                f"got {type(feature_dict).__name__}."
            )

        print(f"forward_features keys: {list(feature_dict.keys())}")
        for key, value in feature_dict.items():
            summarize_feature(f"  {key}", value)

        patch_tokens = feature_dict.get("x_norm_patchtokens")
        class_token = feature_dict.get("x_norm_clstoken")
        register_tokens = feature_dict.get("x_norm_regtokens")

        if patch_tokens is None:
            raise KeyError(
                "forward_features() did not return 'x_norm_patchtokens'. "
                f"Available keys: {list(feature_dict.keys())}"
            )

        outputs["patch_tokens"] = patch_tokens
        if class_token is not None:
            outputs["class_token"] = class_token
        if register_tokens is not None:
            outputs["register_tokens"] = register_tokens

        patch_size = (16, 16, 16)
        patch_embed = getattr(model, "patch_embed", None)
        projection = getattr(patch_embed, "proj", None)
        if projection is not None and hasattr(projection, "kernel_size"):
            kernel_size = tuple(int(v) for v in projection.kernel_size)
            if len(kernel_size) == 3:
                patch_size = kernel_size

        spatial = tuple(int(v) for v in image.shape[2:])
        grid = tuple(size // patch for size, patch in zip(spatial, patch_size))
        expected_tokens = int(np.prod(grid))

        print(f"Detected patch size: {patch_size}")
        print(f"Expected token grid: {grid}; tokens={expected_tokens}")

        if patch_tokens.ndim != 3:
            raise ValueError(
                f"Expected patch tokens [B,N,C], got {tuple(patch_tokens.shape)}"
            )
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Patch-token count {patch_tokens.shape[1]} does not match "
                f"expected grid product {expected_tokens}."
            )

        batch, _, channels = patch_tokens.shape
        feature_grid = (
            patch_tokens.transpose(1, 2)
            .reshape(batch, channels, *grid)
            .contiguous()
        )
        outputs["feature_grid"] = feature_grid
        summarize_feature("Spatial feature grid", feature_grid)

    return outputs


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    volume = args.volume.expanduser().resolve() if args.volume else None

    validate_paths(repo, weights, volume)

    print("=== Environment ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime reported by PyTorch: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print_checkpoint_summary(weights)
    model, device = load_model(repo, weights, args.config, args.device)
    image = prepare_volume(volume, device)
    outputs = extract_features(model, image)

    if args.save_features is not None:
        output_path = args.save_features.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_outputs = {key: value.detach().cpu() for key, value in outputs.items()}
        torch.save(cpu_outputs, output_path)
        print(f"\nSaved features to: {output_path}")

    print("\n=== SUCCESS ===")
    print("Checkpoint loaded and one-volume feature extraction completed.")
    print("Expected principal outputs:")
    print("  global_feature: [1, 1024]")
    print("  patch_tokens:   [1, 343, 1024]")
    print("  feature_grid:   [1, 1024, 7, 7, 7]")


if __name__ == "__main__":
    main()
