#!/usr/bin/env python3
"""Evaluate one supervised correction fold and save held-out visualizations."""
from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
import yaml
from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder
from evaluation.probes.eval_dino_shape_denoising_probe_sweep import dice_score_binary,hd95_binary,assd_binary,volume_vox
from training.train_supervised_autoseg_correction_decoder import correction_logits

def rows_for(path,split):
    with open(path,newline="") as f:return [r for r in csv.DictReader(f) if r["split"]==split]
def read_binary(path,device):
    img=sitk.ReadImage(path); a=(sitk.GetArrayFromImage(img)>0).astype(np.float32); return torch.from_numpy(a[None,None]).to(device),img
def relvol(a,b): return abs(a-b)/b if b>0 else float("nan")
def write_mask(mask,ref,path):
    a=mask.detach().cpu().numpy().astype(np.uint8); a=np.squeeze(a); img=sitk.GetImageFromArray(a); img.CopyInformation(ref); sitk.WriteImage(img,str(path),True)
def make_model(cfg,device):
    m=Frozen3DINOShapeAutoencoder(cfg["dino"]["repo_path"],cfg["dino"]["checkpoint_path"],cfg["dino"].get("config_name","train/vit3d_highres"),cfg["decoder"].get("channels",[256,128,64,32])).to(device)
    ck=Path(cfg["output_root"])/cfg["experiment_name"]/"checkpoints"/"best_decoder.pth"; state=torch.load(ck,map_location=device)
    m.decoder.load_state_dict(state["decoder_state_dict"]); m._supervised_correction_cfg=cfg; m.eval(); return m,ck
@torch.no_grad()
def eval_rows(model,rows,device,thr,save_root=None):
    out=[]
    for j,r in enumerate(rows,1):
        x,ref=read_binary(r["autoseg"],device); tgt,_=read_binary(r["approved"],device)
        raw_logits,_=model(x); logits,residual=correction_logits(raw_logits,x,model._supervised_correction_cfg); rec=(torch.sigmoid(logits)>=thr).float(); spacing=tuple(float(v) for v in ref.GetSpacing()[::-1])
        dit=float(dice_score_binary(x,tgt)[0]); dir_=float(dice_score_binary(x,rec)[0]); drt=float(dice_score_binary(rec,tgt)[0])
        hit=float(hd95_binary(x,tgt,spacing)[0]); hir=float(hd95_binary(x,rec,spacing)[0]); hrt=float(hd95_binary(rec,tgt,spacing)[0])
        ait=float(assd_binary(x,tgt,spacing)[0]); air=float(assd_binary(x,rec,spacing)[0]); art=float(assd_binary(rec,tgt,spacing)[0])
        vin=float(volume_vox(x)[0]); vr=float(volume_vox(rec)[0]); vt=float(volume_vox(tgt)[0])
        changed=((x>0.5)!=(tgt>0.5)); unchanged=~changed; pred_changed=((rec>0.5)!=(x>0.5))
        true_missing=((tgt>0.5)&(x<0.5)); true_excess=((x>0.5)&(tgt<0.5))
        def frac(num,den):
            d=float(den.sum()); return float(num.sum())/d if d>0 else float("nan")
        correction_fraction=frac(pred_changed, x>0.5)
        unnecessary_fraction=frac(pred_changed & unchanged, unchanged)
        needed_recall=frac(pred_changed & changed, changed)
        missing_recall=frac((rec>0.5)&true_missing, true_missing)
        excess_recall=frac((rec<0.5)&true_excess, true_excess)
        row={"patient_id":r["patient_id"],"split":r["split"],"outer_fold":r["outer_fold"],"cv_bin":r["cv_bin"],
             "true_1_minus_dice":1-dit,"true_hd95_mm":hit,"true_assd_mm":ait,"true_rel_volume_diff":relvol(vin,vt),
             "qa_1_minus_dice":1-dir_,"qa_hd95_mm":hir,"qa_assd_mm":air,"qa_rel_volume_diff":relvol(vin,vr),
             "corrected_1_minus_dice":1-drt,"corrected_hd95_mm":hrt,"corrected_assd_mm":art,"corrected_rel_volume_diff":relvol(vr,vt),
             "dice_gain":drt-dit,"hd95_reduction_mm":hit-hrt,"assd_reduction_mm":ait-art,
             "correction_volume_fraction":correction_fraction,"unnecessary_change_fraction":unnecessary_fraction,
             "needed_correction_recall":needed_recall,"missing_correction_recall":missing_recall,"excess_correction_recall":excess_recall,
             "case_dir":r["case_dir"]}
        out.append(row)
        if save_root is not None:
            case=save_root/f"patient_{r['patient_id']}"; case.mkdir(parents=True,exist_ok=True)
            shutil.copy2(r["ct"],case/"ct.nii.gz"); shutil.copy2(r["autoseg"],case/"autoseg_input.nii.gz"); shutil.copy2(r["approved"],case/"approved.nii.gz")
            write_mask(rec,ref,case/"predicted_approved.nii.gz")
            true_missing=true_missing.float(); true_excess=true_excess.float()
            corr_missing=((rec>0.5)&(x<0.5)).float(); corr_removed=((x>0.5)&(rec<0.5)).float()
            write_mask(true_missing,ref,case/"true_missing.nii.gz"); write_mask(true_excess,ref,case/"true_excess.nii.gz")
            write_mask(corr_missing,ref,case/"correction_added.nii.gz"); write_mask(corr_removed,ref,case/"correction_removed.nii.gz")
            (case/"metrics.json").write_text(json.dumps(row,indent=2))
        print(f"[{j}/{len(rows)}] patient {r['patient_id']} trueASSD={ait:.3f} qaASSD={air:.3f} correctedASSD={art:.3f}")
    return out
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--study-root",default="/scratch/shaohan/contour_QA/supervised_autoseg_correction"); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); device=torch.device(cfg["training"].get("device","cuda:0") if torch.cuda.is_available() else "cpu"); model,ck=make_model(cfg,device); thr=float(cfg.get("evaluation",{}).get("threshold",.5))
    roi=cfg["roi"]; fold=int(cfg["fold"]); root=Path(args.study_root)/"evaluations"/roi/f"fold_{fold}"; root.mkdir(parents=True,exist_ok=True)
    test=eval_rows(model,rows_for(cfg["manifest"],"test"),device,thr,root/"test_cases")
    dev=eval_rows(model,rows_for(cfg["manifest"],"train")+rows_for(cfg["manifest"],"validation"),device,thr,None)
    write_csv(root/"test_metrics.csv",test); write_csv(root/"development_metrics.csv",dev)
    (root/"metadata.json").write_text(json.dumps({"roi":roi,"fold":fold,"checkpoint":str(ck),"threshold":thr,"test_n":len(test),"development_n":len(dev)},indent=2))
if __name__=="__main__":main()
