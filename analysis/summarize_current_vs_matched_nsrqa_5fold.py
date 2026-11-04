#!/usr/bin/env python3
"""Compare current joint-spatial QA with matched previous NSR-QA over five outer folds."""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

TARGETS=("1_minus_dice","hd95","assd")
PCTS=(75,50)
METHOD_CURRENT="current_joint_spatial"
METHOD_NSRQA="previous_nsrqa_matched"


def reg_metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);m=np.isfinite(y)&np.isfinite(p);y=y[m];p=p[m]
    if not len(y): return {k:np.nan for k in ("mae","rmse","r2","pearson","spearman")}
    return {
        "mae":float(mean_absolute_error(y,p)),
        "rmse":float(math.sqrt(mean_squared_error(y,p))),
        "r2":float(r2_score(y,p)) if len(y)>1 else np.nan,
        "pearson":float(pearsonr(y,p).statistic) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else np.nan,
        "spearman":float(spearmanr(y,p).statistic) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else np.nan,
    }


def cls_metrics(y,s,pred):
    y=np.asarray(y,int);s=np.asarray(s,float);pred=np.asarray(pred,int)
    auc=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else np.nan
    ap=float(average_precision_score(y,s)) if np.any(y==1) else np.nan
    tp=np.sum((y==1)&(pred==1));fp=np.sum((y==0)&(pred==1));tn=np.sum((y==0)&(pred==0));fn=np.sum((y==1)&(pred==0))
    return {"roc_auc":auc,"pr_auc":ap,"sensitivity":tp/(tp+fn) if tp+fn else np.nan,"specificity":tn/(tn+fp) if tn+fp else np.nan,
            "ppv":tp/(tp+fp) if tp+fp else np.nan,"npv":tn/(tn+fn) if tn+fn else np.nan,"positive_n":int(np.sum(y==1)),"negative_n":int(np.sum(y==0))}


def read_current(fold_dir:Path,target:str,pct:int):
    a=fold_dir/"spatial_qa_head_analysis"
    pred=pd.read_csv(a/"spatial_qa_test_predictions.csv",dtype={"patient_id":str})
    cs=pd.read_csv(a/"spatial_qa_classification_summary.csv")
    reg=pred[(pred.target_metric==target)&(pred.task=="regression")][["patient_id","true_value","score_or_prediction"]].copy()
    reg.columns=["patient_id","true_value","prediction"]
    cl=pred[(pred.target_metric==target)&(pred.task==f"classification_p{pct}")][["patient_id","true_value","score_or_prediction"]].copy()
    cl.columns=["patient_id","y_true","score"]
    row=cs[(cs.target_metric==target)&(cs.bad_percentile==float(pct))]
    if len(row)!=1: raise RuntimeError(f"Current classification row missing/duplicated: {fold_dir} {target} P{pct}")
    row=row.iloc[0];thr=float(row.development_oof_threshold_at_target_specificity)
    cl["predicted_label"]=(cl.score>=thr).astype(int);cl["threshold"]=thr;cl["true_error_cutoff"]=float(row.true_error_cutoff)
    return reg,cl


def read_nsrqa(fold_dir:Path,target:str,pct:int):
    a=fold_dir/"analysis"
    pred=pd.read_csv(a/"nsrqa_test_predictions.csv",dtype={"patient_id":str})
    cs=pd.read_csv(a/"nsrqa_classification_summary.csv")
    reg=pred[(pred.target_metric==target)&(pred.task=="regression")][["patient_id","true_value","score_or_prediction"]].copy()
    reg.columns=["patient_id","true_value","prediction"]
    cl=pred[(pred.target_metric==target)&(pred.task==f"classification_p{pct}")][["patient_id","true_value","score_or_prediction","operating_threshold","true_error_cutoff"]].copy()
    cl.columns=["patient_id","y_true","score","threshold","true_error_cutoff"]
    cl["predicted_label"]=(cl.score>=cl.threshold).astype(int)
    row=cs[(cs.target_metric==target)&(cs.bad_percentile==float(pct))]
    if len(row)!=1: raise RuntimeError(f"NSR-QA classification row missing/duplicated: {fold_dir} {target} P{pct}")
    return reg,cl


def paired_bootstrap_reg(y,pc,pn,n_boot,seed):
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,len(y),size=len(y)); yy=y[idx]; a=pc[idx];b=pn[idx]
        vals.append((mean_absolute_error(yy,a)-mean_absolute_error(yy,b),
                     spearmanr(yy,a).statistic-spearmanr(yy,b).statistic if np.std(a)>0 and np.std(b)>0 and np.std(yy)>0 else np.nan))
    arr=np.asarray(vals,float)
    return {"delta_mae_ci95_low":float(np.nanpercentile(arr[:,0],2.5)),"delta_mae_ci95_high":float(np.nanpercentile(arr[:,0],97.5)),
            "delta_spearman_ci95_low":float(np.nanpercentile(arr[:,1],2.5)),"delta_spearman_ci95_high":float(np.nanpercentile(arr[:,1],97.5))}


def paired_bootstrap_cls(y,sc,sn,n_boot,seed):
    y=np.asarray(y,int);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    if not len(pos) or not len(neg): return {"delta_auc_ci95_low":np.nan,"delta_auc_ci95_high":np.nan}
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(n_boot):
        idx=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
        vals.append(roc_auc_score(y[idx],sc[idx])-roc_auc_score(y[idx],sn[idx]))
    return {"delta_auc_ci95_low":float(np.percentile(vals,2.5)),"delta_auc_ci95_high":float(np.percentile(vals,97.5))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--roi",required=True)
    ap.add_argument("--variant",default="large_decoder_ft_last2")
    ap.add_argument("--study-root",default="/scratch/shaohan/contour_QA/supervised_autoseg_correction")
    ap.add_argument("--baseline-root",default=None)
    ap.add_argument("--folds",default="0,1,2,3,4")
    ap.add_argument("--n-boot",type=int,default=5000)
    ap.add_argument("--seed",type=int,default=88)
    ap.add_argument("--output-dir",default=None)
    ap.add_argument("--output-zip",default=None)
    args=ap.parse_args()
    root=Path(args.study_root);base=Path(args.baseline_root) if args.baseline_root else root/"matched_previous_nsrqa_5fold"
    folds=[int(x) for x in args.folds.split(',') if x.strip()]
    out=Path(args.output_dir) if args.output_dir else root/"current_vs_previous_nsrqa_5fold"/args.roi/args.variant
    out.mkdir(parents=True,exist_ok=True)

    reg_fold=[];cls_fold=[];reg_preds=[];cls_preds=[]
    for fold in folds:
        current_dir=root/"capacity_finetune_ablation"/args.roi/f"fold_{fold}"/args.variant
        nsr_dir=base/args.roi/f"fold_{fold}"
        for ti,target in enumerate(TARGETS):
            cr,_=read_current(current_dir,target,75); nr,_=read_nsrqa(nsr_dir,target,75)
            m=cr.merge(nr,on="patient_id",suffixes=("_current","_nsrqa"),validate="one_to_one")
            if len(m)!=len(cr) or len(m)!=len(nr): raise RuntimeError(f"Regression patient mismatch fold {fold} {target}")
            if not np.allclose(m.true_value_current,m.true_value_nsrqa,rtol=1e-5,atol=1e-6):
                raise RuntimeError(f"True metric mismatch: preprocessing/metric definitions differ for fold {fold} {target}")
            y=m.true_value_current.to_numpy(float);pc=m.prediction_current.to_numpy(float);pn=m.prediction_nsrqa.to_numpy(float)
            for method,p in ((METHOD_CURRENT,pc),(METHOD_NSRQA,pn)):
                rm=reg_metrics(y,p);reg_fold.append({"roi":args.roi,"fold":fold,"target_metric":target,"method":method,"test_n":len(y),**rm})
            for r in m.itertuples(index=False): reg_preds.append({"roi":args.roi,"fold":fold,"patient_id":r.patient_id,"target_metric":target,"true_value":r.true_value_current,
                                                                  "current_prediction":r.prediction_current,"nsrqa_prediction":r.prediction_nsrqa})

            for pct in PCTS:
                _,cc=read_current(current_dir,target,pct);_,nc=read_nsrqa(nsr_dir,target,pct)
                z=cc.merge(nc,on="patient_id",suffixes=("_current","_nsrqa"),validate="one_to_one")
                if not np.array_equal(z.y_true_current.to_numpy(int),z.y_true_nsrqa.to_numpy(int)):
                    raise RuntimeError(f"Binary-label mismatch fold {fold} {target} P{pct}; cutoffs should be identical")
                yb=z.y_true_current.to_numpy(int)
                for method,s,pred in ((METHOD_CURRENT,z.score_current,z.predicted_label_current),(METHOD_NSRQA,z.score_nsrqa,z.predicted_label_nsrqa)):
                    cm=cls_metrics(yb,np.asarray(s,float),np.asarray(pred,int));cls_fold.append({"roi":args.roi,"fold":fold,"target_metric":target,"bad_percentile":pct,"method":method,"test_n":len(yb),**cm})
                for r in z.itertuples(index=False): cls_preds.append({"roi":args.roi,"fold":fold,"patient_id":r.patient_id,"target_metric":target,"bad_percentile":pct,
                                                                      "y_true":r.y_true_current,"current_score":r.score_current,"nsrqa_score":r.score_nsrqa,
                                                                      "current_predicted_label":r.predicted_label_current,"nsrqa_predicted_label":r.predicted_label_nsrqa})

    rf=pd.DataFrame(reg_fold);cf=pd.DataFrame(cls_fold);rp=pd.DataFrame(reg_preds);cp=pd.DataFrame(cls_preds)
    # Every patient should be held out exactly once within each target.
    for target,g in rp.groupby("target_metric"):
        if g.patient_id.duplicated().any(): raise RuntimeError(f"Duplicate held-out patients across folds for {target}")

    reg_sum=[];cls_sum=[];paired=[]
    for target in TARGETS:
        pg=rp[rp.target_metric==target]
        y=pg.true_value.to_numpy(float);pc=pg.current_prediction.to_numpy(float);pn=pg.nsrqa_prediction.to_numpy(float)
        for method,p in ((METHOD_CURRENT,pc),(METHOD_NSRQA,pn)):
            rm=reg_metrics(y,p);fg=rf[(rf.target_metric==target)&(rf.method==method)]
            reg_sum.append({"roi":args.roi,"target_metric":target,"method":method,"n":len(y),**{f"pooled_{k}":v for k,v in rm.items()},
                            "fold_mae_mean":fg.mae.mean(),"fold_mae_sd":fg.mae.std(ddof=1),"fold_spearman_mean":fg.spearman.mean(),"fold_spearman_sd":fg.spearman.std(ddof=1)})
        rc=reg_metrics(y,pc);rn=reg_metrics(y,pn);boot=paired_bootstrap_reg(y,pc,pn,args.n_boot,args.seed+100*TARGETS.index(target))
        paired.append({"roi":args.roi,"task":"regression","target_metric":target,"bad_percentile":np.nan,
                       "metric":"mae_current_minus_nsrqa","difference":rc["mae"]-rn["mae"],**boot})
        paired.append({"roi":args.roi,"task":"regression","target_metric":target,"bad_percentile":np.nan,
                       "metric":"spearman_current_minus_nsrqa","difference":rc["spearman"]-rn["spearman"],**boot})

        for pct in PCTS:
            pg=cp[(cp.target_metric==target)&(cp.bad_percentile==pct)]
            yb=pg.y_true.to_numpy(int);sc=pg.current_score.to_numpy(float);sn=pg.nsrqa_score.to_numpy(float)
            for method,s,predcol in ((METHOD_CURRENT,sc,"current_predicted_label"),(METHOD_NSRQA,sn,"nsrqa_predicted_label")):
                cm=cls_metrics(yb,s,pg[predcol].to_numpy(int));fg=cf[(cf.target_metric==target)&(cf.bad_percentile==pct)&(cf.method==method)]
                cls_sum.append({"roi":args.roi,"target_metric":target,"bad_percentile":pct,"method":method,"n":len(yb),**{f"pooled_{k}":v for k,v in cm.items()},
                                "fold_roc_auc_mean":fg.roc_auc.mean(),"fold_roc_auc_sd":fg.roc_auc.std(ddof=1),"fold_sensitivity_mean":fg.sensitivity.mean(),"fold_specificity_mean":fg.specificity.mean()})
            boot=paired_bootstrap_cls(yb,sc,sn,args.n_boot,args.seed+1000+10*TARGETS.index(target)+pct)
            paired.append({"roi":args.roi,"task":"classification","target_metric":target,"bad_percentile":pct,"metric":"roc_auc_current_minus_nsrqa",
                           "difference":float(roc_auc_score(yb,sc)-roc_auc_score(yb,sn)) if len(np.unique(yb))==2 else np.nan,**boot})

    rf.to_csv(out/"current_vs_nsrqa_regression_per_fold.csv",index=False);cf.to_csv(out/"current_vs_nsrqa_classification_per_fold.csv",index=False)
    pd.DataFrame(reg_sum).to_csv(out/"current_vs_nsrqa_regression_summary.csv",index=False);pd.DataFrame(cls_sum).to_csv(out/"current_vs_nsrqa_classification_summary.csv",index=False)
    rp.to_csv(out/"current_vs_nsrqa_test_predictions_regression.csv",index=False);cp.to_csv(out/"current_vs_nsrqa_test_predictions_classification.csv",index=False)
    pd.DataFrame(paired).to_csv(out/"current_vs_nsrqa_paired_differences.csv",index=False)
    (out/"comparison_protocol.json").write_text(json.dumps({
        "current_method":"large decoder + last-two-block 3DINO fine-tuning + joint 3-target spatial action QA head",
        "previous_method":"previous stochastic-region-dropout NSR-QA denoising model with frozen 3DINO and input-vs-reconstruction QA metrics",
        "fairness_controls":["same five outer patient folds","same autoseg-derived crop grid","same approved/autoseg masks","same metric implementations","same development-derived P75/P50 bad-case definition","no outer-test fitting"],
        "nsrqa_regression":"corresponding raw NSR-QA reconstruction disagreement linearly calibrated on train+validation",
        "nsrqa_classification":"corresponding raw NSR-QA reconstruction disagreement score",
        "current_classification":"continuous current-method metric prediction as score",
        "paired_bootstrap_difference_sign":"positive AUC/Spearman delta favors current; negative MAE delta favors current",
        "folds":folds,"n_boot":args.n_boot,
    },indent=2))
    zpath=Path(args.output_zip) if args.output_zip else out.parent/f"{args.variant}_current_vs_previous_nsrqa_5fold_results_no_models.zip"
    if zpath.exists(): zpath.unlink()
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file(): zf.write(p,p.relative_to(out))
    print(f"Wrote comparison: {out}\nWrote ZIP: {zpath}")

if __name__=="__main__": main()
