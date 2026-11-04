#!/usr/bin/env python3
"""Preprocess real autosegmentations on an autoseg-derived ROI-group grid.

The crop center is determined only from raw autosegmentation masks. The approved
contour is resampled *after* that reference grid has been fixed and is used only
as the supervised target / evaluation reference.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import SimpleITK as sitk

GROUP_ROIS = {
    "airway": ("Bronchus", "Carina", "Trachea"),
    "heart_core": ("Heart", "A_Aorta", "A_Pulmonary"),
    "mediastinum_tubes": ("Esophagus", "V_VenaCava_S"),
    "lungs_chest": ("Lung_L", "Lung_R", "Ribs", "NS_Chestwall"),
    "brachial": ("BrachPlx_L", "BrachPlx_R"),
    "spine": ("SpinalCord",),
    "upper_abdomen": ("Liver", "Stomach"),
}
CENTER_ROIS = {
    "airway": ("Bronchus", "Carina", "Trachea"),
    "heart_core": ("Heart", "A_Aorta", "A_Pulmonary"),
    "mediastinum_tubes": ("Esophagus", "V_VenaCava_S"),
    "lungs_chest": ("Lung_L", "Lung_R"),
    "brachial": ("BrachPlx_L", "BrachPlx_R"),
    "spine": ("SpinalCord",),
    "upper_abdomen": ("Liver", "Stomach"),
}

def pid_from_name(name: str) -> Optional[str]:
    m = re.search(r"patient(\d+)_", name)
    return m.group(1) if m else None

def geometry_equal(a: sitk.Image, b: sitk.Image, tol=1e-5) -> bool:
    return (a.GetSize()==b.GetSize() and np.allclose(a.GetSpacing(),b.GetSpacing(),atol=tol)
            and np.allclose(a.GetOrigin(),b.GetOrigin(),atol=tol)
            and np.allclose(a.GetDirection(),b.GetDirection(),atol=tol))

def resample(moving, reference, interpolator, default=0.0, pixel_id=None):
    f=sitk.ResampleImageFilter(); f.SetReferenceImage(reference); f.SetTransform(sitk.Transform())
    f.SetInterpolator(interpolator); f.SetDefaultPixelValue(default)
    if pixel_id is not None: f.SetOutputPixelType(pixel_id)
    return f.Execute(moving)

def binary_on_ct(path: Path, ct: sitk.Image) -> sitk.Image:
    m=sitk.ReadImage(str(path))
    if not geometry_equal(m,ct): m=resample(m,ct,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    arr=(sitk.GetArrayFromImage(m)>0).astype(np.uint8)
    out=sitk.GetImageFromArray(arr); out.CopyInformation(ct); return out

def union_center_physical(masks: Iterable[sitk.Image]) -> Optional[np.ndarray]:
    ref=None; union=None
    for m in masks:
        ref=m; a=sitk.GetArrayFromImage(m)>0; union=a.copy() if union is None else (union|a)
    if ref is None or union is None or not np.any(union): return None
    zyx=np.column_stack(np.where(union)).mean(axis=0)
    xyz=[float(zyx[2]),float(zyx[1]),float(zyx[0])]
    return np.asarray(ref.TransformContinuousIndexToPhysicalPoint(xyz),dtype=float)

def make_reference(center_xyz, size_xyz, spacing_xyz, direction):
    dm=np.asarray(direction,dtype=float).reshape(3,3)
    half=0.5*(np.asarray(size_xyz,dtype=float)-1.0)*np.asarray(spacing_xyz,dtype=float)
    origin=np.asarray(center_xyz,dtype=float)-dm@half
    ref=sitk.Image(tuple(int(v) for v in size_xyz),sitk.sitkFloat32)
    ref.SetSpacing(tuple(float(v) for v in spacing_xyz)); ref.SetDirection(tuple(direction)); ref.SetOrigin(tuple(float(v) for v in origin))
    return ref

def find_template_geometry(existing_root: Path, group: str, roi: str):
    for split in ("training1","validation1","testing1"):
        d=existing_root/group/roi/split/group/roi/"labels"
        files=sorted(d.glob("*.nii.gz")) if d.is_dir() else []
        if files:
            img=sitk.ReadImage(str(files[0])); return img.GetSize(), img.GetSpacing()
    raise FileNotFoundError(f"No existing approved-preprocessed label found for {group}/{roi}")

def write_binary(img: sitk.Image, path: Path):
    arr=(sitk.GetArrayFromImage(img)>0).astype(np.uint8); out=sitk.GetImageFromArray(arr); out.CopyInformation(img)
    path.parent.mkdir(parents=True,exist_ok=True); sitk.WriteImage(out,str(path),True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--roi",required=True); ap.add_argument("--group",required=True,choices=sorted(GROUP_ROIS))
    ap.add_argument("--raw-root",default="/home/shaohan/radformation-data/data")
    ap.add_argument("--existing-approved-root",default="/home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom")
    ap.add_argument("--output-root",default="/scratch/shaohan/contour_QA/supervised_autoseg_correction/preprocessed")
    args=ap.parse_args()
    raw=Path(args.raw_root); out_root=Path(args.output_root); roi=args.roi; group=args.group
    if roi not in GROUP_ROIS[group]: raise ValueError(f"{roi} is not in group {group}")
    size_xyz, spacing_xyz=find_template_geometry(Path(args.existing_approved_root),group,roi)
    rows=[]; skipped=[]
    for ct_path in sorted((raw/"images").glob("patient*_image.nii.gz")):
        pid=pid_from_name(ct_path.name)
        if pid is None: continue
        aut_target=raw/"labels-autoseg"/f"patient{pid}_{roi}_autoseg.nii.gz"
        approved=raw/"labels-approved"/f"patient{pid}_{roi}_approved.nii.gz"
        if not aut_target.is_file() or not approved.is_file():
            skipped.append({"patient_id":pid,"reason":"missing target ROI autoseg or approved"}); continue
        ct=sitk.Cast(sitk.ReadImage(str(ct_path)),sitk.sitkFloat32)
        available={}
        for r in GROUP_ROIS[group]:
            p=raw/"labels-autoseg"/f"patient{pid}_{r}_autoseg.nii.gz"
            if p.is_file(): available[r]=binary_on_ct(p,ct)
        center_names=[r for r in CENTER_ROIS[group] if r in available]
        if not center_names: center_names=list(available)
        center=union_center_physical([available[r] for r in center_names])
        if center is None:
            skipped.append({"patient_id":pid,"reason":"all available autoseg center masks empty"}); continue
        ref=make_reference(center,size_xyz,spacing_xyz,ct.GetDirection())
        ct_out=resample(ct,ref,sitk.sitkLinear,-1024.0,sitk.sitkFloat32)
        aut=resample(binary_on_ct(aut_target,ct),ref,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
        tgt=resample(binary_on_ct(approved,ct),ref,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
        aa=sitk.GetArrayFromImage(aut)>0; ta=sitk.GetArrayFromImage(tgt)>0
        if not aa.any() or not ta.any():
            skipped.append({"patient_id":pid,"reason":f"empty after crop autoseg={int(aa.sum())} approved={int(ta.sum())}"}); continue
        case=out_root/roi/f"patient_{pid}"; case.mkdir(parents=True,exist_ok=True)
        sitk.WriteImage(ct_out,str(case/"ct.nii.gz"),True); write_binary(aut,case/"autoseg.nii.gz"); write_binary(tgt,case/"approved.nii.gz")
        rows.append({"patient_id":pid,"roi":roi,"group":group,"case_dir":str(case),"ct":str(case/"ct.nii.gz"),
                     "autoseg":str(case/"autoseg.nii.gz"),"approved":str(case/"approved.nii.gz"),
                     "center_rois":",".join(center_names),"center_phys_xyz":json.dumps([float(x) for x in center]),
                     "size_xyz":json.dumps(list(size_xyz)),"spacing_xyz":json.dumps(list(spacing_xyz)),
                     "autoseg_voxels":int(aa.sum()),"approved_voxels":int(ta.sum())})
    out_root.mkdir(parents=True,exist_ok=True)
    for name,data in ((f"{roi}_cases.csv",rows),(f"{roi}_skipped.csv",skipped)):
        p=out_root/name
        keys=sorted({k for r in data for k in r}) if data else ["patient_id","reason"]
        with p.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(data)
    print(f"{roi}: wrote {len(rows)} cases; skipped {len(skipped)}")

if __name__=="__main__": main()
