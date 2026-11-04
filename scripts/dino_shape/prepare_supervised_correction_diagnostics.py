#!/usr/bin/env python3
"""Prepare nested learning-curve, memorization, and paired-augmentation diagnostics."""
from __future__ import annotations
import argparse,csv,copy,random
from pathlib import Path
import yaml

def read_csv(p):
    with open(p,newline="") as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--study-root",default="/scratch/shaohan/contour_QA/supervised_autoseg_correction")
    ap.add_argument("--roi",default="Bronchus"); ap.add_argument("--fold",type=int,default=0)
    ap.add_argument("--seed",type=int,default=88); ap.add_argument("--memorize-n",type=int,default=8)
    args=ap.parse_args(); root=Path(args.study_root)
    base_cfg_path=root/"configs"/args.roi/f"fold_{args.fold}.yaml"
    if not base_cfg_path.is_file(): raise FileNotFoundError(base_cfg_path)
    base=yaml.safe_load(base_cfg_path.read_text()); rows=read_csv(base["manifest"])
    train=[r for r in rows if r["split"]=="train"]; val=[r for r in rows if r["split"]=="validation"]
    if len(train)<args.memorize_n: raise RuntimeError(f"Only {len(train)} training cases; need {args.memorize_n}")
    rng=random.Random(args.seed); order=list(train); rng.shuffle(order)
    droot=root/"diagnostics"/args.roi/f"fold_{args.fold}"; mroot=droot/"manifests"; croot=droot/"configs"; modelroot=droot/"models"
    tasks=[]
    def add(name,tr,va,aug=False,kind="learning_curve",fraction=None):
        out=[]
        for r in tr: q=dict(r); q["split"]="train"; out.append(q)
        for r in va: q=dict(r); q["split"]="validation"; out.append(q)
        mp=mroot/f"{name}.csv"; write_csv(mp,out)
        cfg=copy.deepcopy(base); cfg["manifest"]=str(mp); cfg["output_root"]=str(modelroot); cfg["experiment_name"]=f"diag_{args.roi}_fold{args.fold}_{name}"
        cfg["training"]["paired_augmentation"]={"enabled":bool(aug),"probability":0.8,"rotation_deg":5.0,"translation_voxels":6.0,"scale_min":0.95,"scale_max":1.05}
        cp=croot/f"{name}.yaml"; cp.parent.mkdir(parents=True,exist_ok=True); cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
        tasks.append({"task_index":len(tasks),"experiment":name,"kind":kind,"fraction":"" if fraction is None else fraction,"augmentation":int(aug),"train_n":len(tr),"validation_n":len(va),"config":str(cp)})
    mem=order[:args.memorize_n]; add(f"memorize_{args.memorize_n}",mem,mem,False,"memorization",args.memorize_n/len(train))
    for frac in (0.2,0.4,0.6,0.8,1.0):
        n=max(1,round(len(order)*frac)); add(f"learning_{int(frac*100):03d}",order[:n],val,False,"learning_curve",frac)
    for frac in (0.5,1.0):
        n=max(1,round(len(order)*frac)); add(f"augment_{int(frac*100):03d}",order[:n],val,True,"augmentation",frac)
    tp=droot/"diagnostic_tasks.tsv"; tp.parent.mkdir(parents=True,exist_ok=True)
    with tp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(tasks[0]),delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(tasks)
    print(f"Prepared {len(tasks)} diagnostics for {args.roi} fold {args.fold}")
    print(tp)
if __name__=="__main__":main()
