#!/usr/bin/env python3
"""Train frozen-3DINO contour correction decoder on real autoseg -> approved pairs."""
from __future__ import annotations
import argparse,csv,json,random,shutil,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset,DataLoader
from models.dino_shape.dino_shape_autoencoder import Frozen3DINOShapeAutoencoder

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def read_manifest(path,split):
    with open(path,newline="") as f:return [r for r in csv.DictReader(f) if r["split"]==split]
class PairDataset(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        a=(sitk.GetArrayFromImage(sitk.ReadImage(r["autoseg"]))>0).astype(np.float32)
        t=(sitk.GetArrayFromImage(sitk.ReadImage(r["approved"]))>0).astype(np.float32)
        if a.shape!=t.shape: raise ValueError(f"Shape mismatch patient {r['patient_id']}: {a.shape} vs {t.shape}")
        return {"input":torch.from_numpy(a)[None],"target":torch.from_numpy(t)[None],"patient_id":r["patient_id"]}
def dice(prob,target,eps=1e-6):
    dims=tuple(range(2,target.ndim)); inter=(prob*target).sum(dims); den=prob.sum(dims)+target.sum(dims); return (2*inter+eps)/(den+eps)
def dilate(mask,r): return mask if r<=0 else F.max_pool3d(mask,2*r+1,1,r)
def correction_logits(raw_logits,x,cfg):
    """Map decoder output to final logits; direct mode preserves legacy behavior."""
    corr=cfg.get("correction",{})
    if corr.get("output_mode","direct")!="residual_logits":
        return raw_logits,torch.zeros_like(raw_logits)
    input_mag=float(corr.get("input_logit_magnitude",4.0))
    residual_bound=float(corr.get("residual_logit_bound",6.0))
    input_logits=(2.0*x-1.0)*input_mag
    residual=residual_bound*torch.tanh(raw_logits)
    return input_logits+residual,residual

def loss_fn(raw_logits,target,x,cfg):
    logits,residual=correction_logits(raw_logits,x,cfg)
    prob=torch.sigmoid(logits); gd=1-dice(prob,target).mean(); gb=F.binary_cross_entropy_with_logits(logits,target)
    changed=((target>0.5)!=(x>0.5)).float(); region=dilate(changed,int(cfg["loss"].get("changed_region_dilation_voxels",3)))
    bce_vox=F.binary_cross_entropy_with_logits(logits,target,reduction="none")
    changed_reduction=cfg["loss"].get("changed_region_reduction","weighted_global")
    if changed_reduction=="region_mean":
        cb=(bce_vox*region).sum()/region.sum().clamp_min(1.0)
    elif changed_reduction=="weighted_global":
        weight=1.0+float(cfg["loss"].get("changed_region_voxel_weight",8.0))*region
        cb=F.binary_cross_entropy_with_logits(logits,target,weight=weight)
    else:
        raise ValueError(f"Unknown changed_region_reduction={changed_reduction!r}")
    missing=((target>0.5)&(x<=0.5)).float()
    excess=((x>0.5)&(target<=0.5)).float()
    directional_radius=int(cfg["loss"].get("directional_region_dilation_voxels",cfg["loss"].get("changed_region_dilation_voxels",3)))
    missing_region=dilate(missing,directional_radius)
    excess_region=dilate(excess,directional_radius)
    missing_bce=(bce_vox*missing_region).sum()/missing_region.sum().clamp_min(1.0)
    excess_bce=(bce_vox*excess_region).sum()/excess_region.sum().clamp_min(1.0)
    unchanged=1.0-changed
    denom=unchanged.sum().clamp_min(1.0)
    residual_bound=max(float(cfg.get("correction",{}).get("residual_logit_bound",6.0)),1e-6)
    preserve=((residual.abs()/residual_bound)*unchanged).sum()/denom
    total=(float(cfg["loss"].get("global_dice_weight",.4))*gd
           +float(cfg["loss"].get("global_bce_weight",.1))*gb
           +float(cfg["loss"].get("changed_region_bce_weight",.5))*cb
           +float(cfg["loss"].get("missing_region_bce_weight",0.0))*missing_bce
           +float(cfg["loss"].get("excess_region_bce_weight",0.0))*excess_bce
           +float(cfg["loss"].get("unchanged_residual_l1_weight",0.0))*preserve)
    return total,{"total":total,"global_dice":gd,"global_bce":gb,"changed_bce":cb,"missing_bce":missing_bce,"excess_bce":excess_bce,"preserve_l1":preserve,"changed_fraction":changed.mean()},logits
def paired_affine_augment(x,target,cfg):
    aug=cfg.get("training",{}).get("paired_augmentation",{})
    if not aug.get("enabled",False) or random.random()>float(aug.get("probability",1.0)):
        return x,target
    n,_,d,h,w=x.shape
    deg=float(aug.get("rotation_deg",5.0)); trans=float(aug.get("translation_voxels",6.0))
    smin=float(aug.get("scale_min",0.95)); smax=float(aug.get("scale_max",1.05))
    thetas=[]
    for _ in range(n):
        rx,ry,rz=[np.deg2rad(random.uniform(-deg,deg)) for _ in range(3)]
        cx,sx=np.cos(rx),np.sin(rx); cy,sy=np.cos(ry),np.sin(ry); cz,sz=np.cos(rz),np.sin(rz)
        Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],dtype=np.float32)
        Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],dtype=np.float32)
        Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],dtype=np.float32)
        scale=random.uniform(smin,smax)
        A=(Rz@Ry@Rx)/scale
        tx=random.uniform(-trans,trans)*2.0/max(w-1,1)
        ty=random.uniform(-trans,trans)*2.0/max(h-1,1)
        tz=random.uniform(-trans,trans)*2.0/max(d-1,1)
        theta=np.concatenate([A,np.array([[tx],[ty],[tz]],dtype=np.float32)],axis=1)
        thetas.append(theta)
    theta=torch.as_tensor(np.stack(thetas),device=x.device,dtype=x.dtype)
    grid=F.affine_grid(theta,x.size(),align_corners=False)
    return (F.grid_sample(x,grid,mode="nearest",padding_mode="zeros",align_corners=False),
            F.grid_sample(target,grid,mode="nearest",padding_mode="zeros",align_corners=False))

def amp(device,precision):
    if device.type!="cuda" or precision=="fp32": return nullcontext()
    return torch.autocast("cuda",dtype=torch.bfloat16 if precision=="bf16" else torch.float16)
def run_epoch(model,loader,opt,device,cfg,train):
    model.train(train); accum=int(cfg["training"].get("accumulation_steps",1)); prec=cfg["training"].get("precision","bf16")
    sums={k:0. for k in ["total","global_dice","global_bce","changed_bce","missing_bce","excess_bce","preserve_l1","changed_fraction","input_target_dice","recon_target_dice","correction_gain"]}; n=0
    if train: opt.zero_grad(set_to_none=True)
    for step,b in enumerate(loader):
        x=b["input"].to(device,non_blocking=True); target=b["target"].to(device,non_blocking=True)
        if train: x,target=paired_affine_augment(x,target,cfg)
        with torch.set_grad_enabled(train):
            with amp(device,prec): raw_logits,_=model(x); loss,p,logits=loss_fn(raw_logits,target,x,cfg)
            if train:
                (loss/accum).backward()
                if (step+1)%accum==0 or step+1==len(loader):
                    torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),float(cfg["training"].get("grad_clip",1.0))); opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            rec=(torch.sigmoid(logits)>=float(cfg.get("evaluation",{}).get("threshold",.5))).float(); dit=dice(x,target).mean(); drt=dice(rec,target).mean()
        bs=x.shape[0]; n+=bs
        for k in ["total","global_dice","global_bce","changed_bce","missing_bce","excess_bce","preserve_l1","changed_fraction"]: sums[k]+=float(p[k].detach().float())*bs
        sums["input_target_dice"]+=float(dit)*bs; sums["recon_target_dice"]+=float(drt)*bs; sums["correction_gain"]+=float(drt-dit)*bs
    return {k:v/max(n,1) for k,v in sums.items()}
def save_ckpt(path,model,opt,epoch,best,cfg): torch.save({"decoder_state_dict":model.decoder.state_dict(),"optimizer_state_dict":opt.state_dict(),"epoch":epoch,"best_validation_total":best,"config":cfg},path)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--fresh",action="store_true"); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); seed_all(int(cfg.get("seed",88)))
    exp=Path(cfg["output_root"])/cfg["experiment_name"]; ck=exp/"checkpoints"; logs=exp/"logs"
    if args.fresh and exp.exists(): shutil.rmtree(exp)
    ck.mkdir(parents=True,exist_ok=True); logs.mkdir(parents=True,exist_ok=True); (exp/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    tr=PairDataset(read_manifest(cfg["manifest"],"train")); va=PairDataset(read_manifest(cfg["manifest"],"validation"))
    workers=int(cfg["training"].get("num_workers",4)); gen=torch.Generator().manual_seed(int(cfg.get("seed",88)))
    train_loader=DataLoader(tr,batch_size=int(cfg["training"].get("batch_size",1)),shuffle=True,generator=gen,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    val_loader=DataLoader(va,batch_size=1,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    device=torch.device(cfg["training"].get("device","cuda:0") if torch.cuda.is_available() else "cpu")
    model=Frozen3DINOShapeAutoencoder(cfg["dino"]["repo_path"],cfg["dino"]["checkpoint_path"],cfg["dino"].get("config_name","train/vit3d_highres"),cfg["decoder"].get("channels",[256,128,64,32])).to(device)
    opt=torch.optim.AdamW(model.decoder.parameters(),lr=float(cfg["training"].get("learning_rate",1e-4)),weight_decay=float(cfg["training"].get("weight_decay",1e-4)))
    print(f"{cfg['roi']} fold={cfg['fold']} train={len(tr)} validation={len(va)} test={len(read_manifest(cfg['manifest'],'test'))}")
    hist=logs/"history.csv"; best=float("inf"); best_gain=float("-inf"); bad=0; patience=int(cfg["training"].get("early_stopping_patience",40)); epochs=int(cfg["training"].get("num_epochs",200))
    fields=["epoch"]+[f"train_{k}" for k in run_epoch.__annotations__.keys() if False]
    for epoch in range(1,epochs+1):
        t0=time.time(); tm=run_epoch(model,train_loader,opt,device,cfg,True); vm=run_epoch(model,val_loader,opt,device,cfg,False)
        row={"epoch":epoch,**{f"train_{k}":v for k,v in tm.items()},**{f"validation_{k}":v for k,v in vm.items()},"seconds":time.time()-t0}
        with hist.open("a",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(row));
            if f.tell()==0:w.writeheader()
            w.writerow(row)
        score=vm["total"]; improved=score<best
        if improved: best=score; bad=0; save_ckpt(ck/"best_decoder.pth",model,opt,epoch,best,cfg)
        else: bad+=1
        if vm["correction_gain"]>best_gain:
            best_gain=vm["correction_gain"]; save_ckpt(ck/"best_gain_decoder.pth",model,opt,epoch,best,cfg)
        save_ckpt(ck/"last_decoder.pth",model,opt,epoch,best,cfg)
        print(f"epoch={epoch:03d} train={tm['total']:.5f} val={vm['total']:.5f} train_gain={tm['correction_gain']:.4f} val_gain={vm['correction_gain']:.4f} best={best:.5f} bad={bad}/{patience}",flush=True)
        if bad>=patience: print("Early stopping"); break
if __name__=="__main__":main()
