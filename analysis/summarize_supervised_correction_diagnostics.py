#!/usr/bin/env python3
"""Summarize supervised correction data-vs-capacity diagnostics."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

def read_csv(p):
    with open(p,newline="") as f:return list(csv.DictReader(f))
def fmean(rows,key):
    vals=[float(r[key]) for r in rows if r.get(key,"not") not in ("","nan","NaN")]
    return float(np.mean(vals)) if vals else float("nan")
def fracpos(rows,key):
    vals=[float(r[key]) for r in rows if r.get(key,"") not in ("","nan","NaN")]
    return float(np.mean(np.asarray(vals)>0)) if vals else float("nan")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--diagnostic-root",required=True); args=ap.parse_args(); root=Path(args.diagnostic_root)
    tasks=[]
    with (root/"diagnostic_tasks.tsv").open(newline="") as f: tasks=list(csv.DictReader(f,delimiter="\t"))
    out=[]
    for t in tasks:
        cfg=Path(t["config"]); modeldir=root/"models"/f"diag_{root.parent.name}_fold{root.name.split('_')[-1]}_{t['experiment']}"
        hist=modeldir/"logs"/"history.csv"; ev=root/"evaluation"/t["experiment"]
        if not hist.is_file() or not (ev/"train_metrics.csv").is_file(): continue
        h=read_csv(hist); best=min(h,key=lambda r:float(r["validation_total"])); tr=read_csv(ev/"train_metrics.csv"); va=read_csv(ev/"validation_metrics.csv")
        row={k:t[k] for k in ["experiment","kind","fraction","augmentation","train_n","validation_n"]}
        row.update({"best_epoch":int(best["epoch"]),"best_validation_loss":float(best["validation_total"]),"epoch_train_dice_gain":float(best["train_correction_gain"]),"epoch_validation_dice_gain":float(best["validation_correction_gain"])})
        for split,rows in [("train",tr),("validation",va)]:
            row[f"{split}_dice_gain"]=fmean(rows,"dice_gain"); row[f"{split}_assd_reduction_mm"]=fmean(rows,"assd_reduction_mm"); row[f"{split}_hd95_reduction_mm"]=fmean(rows,"hd95_reduction_mm")
            row[f"{split}_fraction_dice_improved"]=fracpos(rows,"dice_gain"); row[f"{split}_fraction_assd_improved"]=fracpos(rows,"assd_reduction_mm"); row[f"{split}_fraction_hd95_improved"]=fracpos(rows,"hd95_reduction_mm")
        out.append(row)
    if not out: raise RuntimeError("No completed diagnostic outputs found")
    p=root/"diagnostic_summary.csv"
    with p.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for kind,name in [("memorization","memorization_summary.csv"),("learning_curve","learning_curve_summary.csv"),("augmentation","augmentation_summary.csv")]:
        rr=[r for r in out if r["kind"]==kind]
        if rr:
            with (root/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    print(f"Wrote {p}")
    print("\nexperiment\ttrain_n\taug\ttrain_gain\tval_gain\tval_ASSD_reduction\tval_HD95_reduction")
    for r in out: print(f"{r['experiment']}\t{r['train_n']}\t{r['augmentation']}\t{r['train_dice_gain']:.4f}\t{r['validation_dice_gain']:.4f}\t{r['validation_assd_reduction_mm']:.3f}\t{r['validation_hd95_reduction_mm']:.3f}")
if __name__=="__main__":main()
