#!/usr/bin/env python3
"""Evaluate the best diagnostic checkpoint on its train and validation cases."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch,yaml
from evaluation.supervised_correction.eval_supervised_autoseg_correction import make_model,eval_rows,write_csv

def rows_for(path,split):
    with open(path,newline="") as f:return [r for r in csv.DictReader(f) if r["split"]==split]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--output-dir",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); device=torch.device(cfg["training"].get("device","cuda:0") if torch.cuda.is_available() else "cpu")
    model,ck=make_model(cfg,device); thr=float(cfg.get("evaluation",{}).get("threshold",.5)); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr=eval_rows(model,rows_for(cfg["manifest"],"train"),device,thr,None); va=eval_rows(model,rows_for(cfg["manifest"],"validation"),device,thr,None)
    write_csv(out/"train_metrics.csv",tr); write_csv(out/"validation_metrics.csv",va)
    (out/"metadata.json").write_text(json.dumps({"checkpoint":str(ck),"train_n":len(tr),"validation_n":len(va)},indent=2))
if __name__=="__main__":main()
