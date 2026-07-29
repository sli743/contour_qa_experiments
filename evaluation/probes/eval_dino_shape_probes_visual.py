#!/usr/bin/env python3
"""Generate a small probe set and export 3DINO reconstructions for 3D Slicer.

Flat output filenames exactly match ProbeContourComparer.py:
  clean_input_patient_<id>.nii.gz
  original_ct_patient_<id>.nii.gz
  corrupted_input_<probe>_patient_<id>.nii.gz
  reconstruction_<probe>_patient_<id>.nii.gz
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml
from scipy import ndimage

from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test", choices=["train","validation","test"])
    p.add_argument("--max_cases", type=int, default=6)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


def pid_from_name(name):
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    for pat in [r"(?:label|mask|seg|proj|ct)_patient_?(.+)$", r"patient_?(.+?)(?:_(?:label|mask|seg|approved|ct))?$"]:
        m = re.search(pat, stem, flags=re.I)
        if m: return m.group(1)
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else stem


def find_ct(ct_dir, pid):
    exact = [p for p in Path(ct_dir).rglob("*.nii.gz") if pid_from_name(p.name) == pid]
    return sorted(exact)[0] if exact else None


def ball(radius):
    r = int(radius)
    zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    return (zz*zz + yy*yy + xx*xx) <= r*r


def shift_mask(mask, axis, voxels):
    delta = [0,0,0]; delta[axis] = int(voxels)
    return ndimage.shift(mask.astype(np.uint8), shift=delta, order=0, mode="constant", cval=0).astype(bool)


def add_blob(mask, radius, rng):
    out = mask.copy()
    boundary = np.argwhere(ndimage.binary_dilation(mask, iterations=max(2, radius)) & ~mask)
    if not len(boundary): return out
    center = boundary[rng.integers(len(boundary))]
    z,y,x = np.indices(mask.shape)
    sphere = (z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2 <= radius**2
    return out | sphere


def add_bite(mask, radius, rng):
    out = mask.copy()
    boundary = np.argwhere(mask & ~ndimage.binary_erosion(mask))
    if not len(boundary): return out
    center = boundary[rng.integers(len(boundary))]
    z,y,x = np.indices(mask.shape)
    sphere = (z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2 <= radius**2
    return out & ~sphere


def directional_cut(mask, axis, fraction, upper):
    out = mask.copy()
    coords = np.argwhere(mask)
    if not len(coords): return out
    lo, hi = coords[:,axis].min(), coords[:,axis].max()
    n = max(1, int(round((hi-lo+1)*fraction)))
    sl = [slice(None)]*3
    sl[axis] = slice(hi-n+1, hi+1) if upper else slice(lo, lo+n)
    out[tuple(sl)] = False
    return out


def probes(mask, rng):
    # Deliberately small pilot set: enough to visually inspect behavior.
    return {
        "erode_1": ndimage.binary_erosion(mask, structure=ball(1), iterations=1),
        "erode_2": ndimage.binary_erosion(mask, structure=ball(1), iterations=2),
        "dilate_1": ndimage.binary_dilation(mask, structure=ball(1), iterations=1),
        "dilate_2": ndimage.binary_dilation(mask, structure=ball(1), iterations=2),
        "shift_x_2": shift_mask(mask, 2, 2),
        "shift_z_2": shift_mask(mask, 0, 2),
        "big_blob_r4": add_blob(mask, 4, rng),
        "big_bite_r4": add_bite(mask, 4, rng),
        "cut_superior_20pct": directional_cut(mask, 0, 0.20, True),
        "cut_inferior_20pct": directional_cut(mask, 0, 0.20, False),
    }


def dice(a,b,eps=1e-8):
    a=a.astype(bool); b=b.astype(bool)
    return (2*np.logical_and(a,b).sum()+eps)/(a.sum()+b.sum()+eps)


def save_like(arr, ref, path, pixel=sitk.sitkUInt8):
    image=sitk.GetImageFromArray(arr)
    image.CopyInformation(ref)
    sitk.WriteImage(sitk.Cast(image,pixel), str(path), True)


def main():
    args=parse_args()
    with open(args.config, "r") as f: cfg=yaml.safe_load(f)
    exp=Path(cfg["output_root"])/cfg["experiment_name"]
    ckpt=Path(args.checkpoint) if args.checkpoint else exp/"checkpoints"/"best_decoder.pth"
    label_dir=Path(cfg["data"][f"{args.split}_label_dir"])
    ct_dir=Path(cfg["data"][f"{args.split}_ct_dir"])
    out=exp/f"probe_visualization_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Frozen3DINOShapeAutoencoder(
        cfg["dino"]["repo_path"], cfg["dino"]["checkpoint_path"],
        cfg["dino"].get("config_name","train/vit3d_highres"),
        cfg["decoder"].get("channels",[256,128,64,32])
    ).to(device)
    state=torch.load(ckpt,map_location="cpu")
    model.decoder.load_state_dict(state["decoder_state_dict"],strict=True)
    model.eval()
    threshold=float(args.threshold if args.threshold is not None else cfg["evaluation"].get("threshold",0.5))
    rng=np.random.default_rng(args.seed)

    rows=[]
    files=sorted(label_dir.rglob("*.nii.gz"))[:args.max_cases]
    for path in files:
        pid=pid_from_name(path.name)
        ref=sitk.ReadImage(str(path))
        clean=(sitk.GetArrayFromImage(ref)>0)
        save_like(clean.astype(np.uint8),ref,out/f"clean_input_patient_{pid}.nii.gz")
        ct=find_ct(ct_dir,pid)
        if ct: shutil.copy2(ct,out/f"original_ct_patient_{pid}.nii.gz")

        for probe_name, corrupted in probes(clean,rng).items():
            corrupted=np.asarray(corrupted,dtype=bool)
            x=torch.from_numpy(corrupted.astype(np.float32))[None,None].to(device)
            with torch.inference_mode():
                logits,_=model(x)
                prob=torch.sigmoid(logits)[0,0].float().cpu().numpy()
            recon=prob>=threshold
            save_like(corrupted.astype(np.uint8),ref,out/f"corrupted_input_{probe_name}_patient_{pid}.nii.gz")
            save_like(recon.astype(np.uint8),ref,out/f"reconstruction_{probe_name}_patient_{pid}.nii.gz")
            save_like(prob.astype(np.float32),ref,out/f"reconstruction_probability_{probe_name}_patient_{pid}.nii.gz",sitk.sitkFloat32)
            rows.append({
                "patient_id":pid,"probe_type":probe_name,
                "dice_input_target":dice(corrupted,clean),
                "dice_input_recon":dice(corrupted,recon),
                "dice_recon_target":dice(recon,clean),
                "input_voxels":int(corrupted.sum()),
                "target_voxels":int(clean.sum()),
                "recon_voxels":int(recon.sum()),
            })
            print(rows[-1])

    with open(exp/f"probe_visual_check_metrics_{args.split}.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Use this directory in Probe Contour Comparer: {out}")


if __name__=="__main__":
    main()
