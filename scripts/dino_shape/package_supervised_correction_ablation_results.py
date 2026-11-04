#!/usr/bin/env python3
"""Zip an ablation result directory while excluding model/checkpoint payloads."""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile

EXCLUDED_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}
EXCLUDED_DIR_NAMES = {
    "checkpoints", "models", "model", "weights",
    "spatial_qa_head_models", "spatial_qa_head_v2_models",
    "spatial_soft_fusion_qa_models",
    "spatial_qa_tensor_cache",
}


def should_exclude(path: Path, root: Path, output_zip: Path) -> bool:
    try:
        if path.resolve() == output_zip.resolve():
            return True
    except FileNotFoundError:
        pass
    rel = path.relative_to(root)
    if any(part.lower() in EXCLUDED_DIR_NAMES for part in rel.parts[:-1]):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-zip", required=True)
    args = ap.parse_args()
    root = Path(args.input_dir).resolve()
    out = Path(args.output_zip).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    included = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or should_exclude(path, root, out):
                continue
            zf.write(path, arcname=str(path.relative_to(root)))
            included += 1
    print(f"Wrote {out} with {included} files; model/checkpoint payloads excluded")


if __name__ == "__main__":
    main()
