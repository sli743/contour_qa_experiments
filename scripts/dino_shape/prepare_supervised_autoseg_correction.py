#!/usr/bin/env python3
"""Create patient-wise 5-fold CV manifests/configs for supervised correction QA."""
from __future__ import annotations
import argparse,csv,hashlib,json,random
from pathlib import Path
import yaml

def read_tsv(path):
    with open(path,newline="") as f:return list(csv.DictReader(f,delimiter="\t"))

def read_csv(path):
    with open(path,newline="") as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--all-roi-study-root",default="/scratch/shaohan/contour_QA/all_roi_NSRQA_study")
    ap.add_argument("--study-root",default="/scratch/shaohan/contour_QA/supervised_autoseg_correction")
    ap.add_argument("--seed",type=int,default=88)
    ap.add_argument("--registry",default="configs/dino_shape/all_roi_nsrqa_registry.yaml")
    args=ap.parse_args()
    root=Path(args.study_root); pre=root/"preprocessed"; mani=root/"manifests"; cfgroot=root/"configs"; mani.mkdir(parents=True,exist_ok=True); cfgroot.mkdir(parents=True,exist_ok=True)
    registry=yaml.safe_load(Path(args.registry).read_text())
    active=read_tsv(Path(args.all_roi_study_root)/"manifests"/"active_rois.tsv")
    patient_union=set(); roi_rows={}
    for r in active:
        p=pre/f"{r['roi']}_cases.csv"
        if not p.is_file(): continue
        rows=read_csv(p); roi_rows[r['roi']]=rows; patient_union.update(x['patient_id'] for x in rows)
    ids=sorted(patient_union,key=lambda x:int(x)); rng=random.Random(args.seed); rng.shuffle(ids)
    binmap={pid:i%10 for i,pid in enumerate(ids)}
    with (mani/"global_patient_bins.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["patient_id","cv_bin"]); w.writeheader(); w.writerows([{"patient_id":p,"cv_bin":binmap[p]} for p in sorted(binmap,key=lambda x:int(x))])
    tasks=[]
    for r in active:
        roi=r['roi']; rows=roi_rows.get(roi,[])
        profile_name=r["profile"]; changed_dilation=int(registry["profiles"][profile_name]["changed_region_dilation_voxels"])
        if len(rows)<10: continue
        bypid={x['patient_id']:x for x in rows}; roidir=mani/roi; roidir.mkdir(parents=True,exist_ok=True); cfgdir=cfgroot/roi; cfgdir.mkdir(parents=True,exist_ok=True)
        for fold in range(5):
            test_bins={2*fold,2*fold+1}; val_bin=(2*fold+2)%10
            splitrows=[]
            for pid,row in sorted(bypid.items(),key=lambda kv:int(kv[0])):
                b=binmap[pid]; split="test" if b in test_bins else ("validation" if b==val_bin else "train")
                splitrows.append({**row,"cv_bin":b,"split":split,"outer_fold":fold})
            mp=roidir/f"fold_{fold}.csv"; fields=list(splitrows[0].keys())
            with mp.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(splitrows)
            cfg={"seed":args.seed,"experiment_name":f"supervised_autoseg_correction_{roi}_fold{fold}",
                 "output_root":str(root/"models"/roi),"fold":fold,"roi":roi,"manifest":str(mp),
                 "dino":{"repo_path":"/home/shaohan/3DINO","checkpoint_path":"/home/shaohan/3DINO/pretrained/3dino_vit_weights.pth","config_name":"train/vit3d_highres"},
                 "decoder":{"channels":[256,128,64,32]},
                 "training":{"device":"cuda:0","batch_size":1,"accumulation_steps":4,"num_workers":4,"num_epochs":200,"early_stopping_patience":40,"learning_rate":1e-4,"weight_decay":1e-4,"grad_clip":1.0,"precision":"bf16"},
                 "loss":{"global_dice_weight":0.40,"global_bce_weight":0.10,"changed_region_bce_weight":0.50,"changed_region_voxel_weight":8.0,"changed_region_dilation_voxels":changed_dilation},
                 "evaluation":{"threshold":0.5}}
            cp=cfgdir/f"fold_{fold}.yaml"; cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
            tasks.append({"task_index":len(tasks),"roi":roi,"fold":fold,"config":str(cp),"manifest":str(mp)})
    tp=mani/"training_tasks.tsv"
    with tp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["task_index","roi","fold","config","manifest"],delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(tasks)
    (mani/"preflight.json").write_text(json.dumps({"n_rois":len(roi_rows),"n_patients_global":len(ids),"n_tasks":len(tasks),"cv":"5 outer folds using paired 10-bin tests; 70% train/10% validation/20% test per fold","seed":args.seed},indent=2))
    print(f"Prepared {len(tasks)} fold-training tasks across {len(roi_rows)} ROIs")
if __name__=="__main__":main()
