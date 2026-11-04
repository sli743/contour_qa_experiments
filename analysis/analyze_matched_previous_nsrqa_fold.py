#!/usr/bin/env python3
"""Classification/regression QA analysis for one matched previous-NSR-QA fold."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.linear_model import LinearRegression

TARGETS = {
    "1_minus_dice": ("qa_1_minus_dice", "true_1_minus_dice"),
    "hd95": ("qa_hd95_mm", "true_hd95_mm"),
    "assd": ("qa_assd_mm", "true_assd_mm"),
}


def percentile_higher(x, q):
    a = np.asarray(x, float); a = a[np.isfinite(a)]
    if not len(a): return float("nan")
    try: return float(np.percentile(a, q, method="higher"))
    except TypeError: return float(np.percentile(a, q, interpolation="higher"))


def choose_threshold(y, score, target_spec=0.95):
    y=np.asarray(y,int); score=np.asarray(score,float)
    if len(np.unique(y))<2: return float("nan"), float("nan"), float("nan")
    best=None
    for t in np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]:
        pred=score>=t
        tn=np.sum((y==0)&(~pred)); fp=np.sum((y==0)&pred); tp=np.sum((y==1)&pred); fn=np.sum((y==1)&(~pred))
        spec=tn/(tn+fp) if tn+fp else np.nan; sens=tp/(tp+fn) if tp+fn else np.nan
        if np.isfinite(spec) and spec+1e-12>=target_spec:
            cand=(sens,-float(t),float(t),spec)
            if best is None or cand>best: best=cand
    return (best[2],best[0],best[3]) if best else (float("nan"),)*3


def confusion(y, score, threshold):
    if not np.isfinite(threshold): return {k:np.nan for k in ("sensitivity","specificity","ppv","npv")}
    y=np.asarray(y,int); pred=np.asarray(score,float)>=threshold
    tp=np.sum((y==1)&pred); fp=np.sum((y==0)&pred); tn=np.sum((y==0)&(~pred)); fn=np.sum((y==1)&(~pred))
    return {
        "sensitivity":tp/(tp+fn) if tp+fn else np.nan,
        "specificity":tn/(tn+fp) if tn+fp else np.nan,
        "ppv":tp/(tp+fp) if tp+fp else np.nan,
        "npv":tn/(tn+fn) if tn+fn else np.nan,
    }


def reg_metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p); y=y[m];p=p[m]
    if not len(y): return {k:np.nan for k in ("mae","rmse","r2","pearson","spearman")}
    return {
        "mae":float(mean_absolute_error(y,p)),
        "rmse":float(math.sqrt(mean_squared_error(y,p))),
        "r2":float(r2_score(y,p)) if len(y)>1 else np.nan,
        "pearson":float(pearsonr(y,p).statistic) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else np.nan,
        "spearman":float(spearmanr(y,p).statistic) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else np.nan,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metrics-csv",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--target-specificity",type=float,default=0.95)
    ap.add_argument("--bad-percentiles",nargs="+",type=float,default=[75,50])
    args=ap.parse_args()
    df=pd.read_csv(args.metrics_csv,dtype={"patient_id":str})
    dev=df[df.split.isin(["train","validation"])].copy(); test=df[df.split.eq("test")].copy()
    if not len(dev) or not len(test): raise RuntimeError("Need development and test rows")
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    cls_rows=[]; reg_rows=[]; pred_rows=[]

    for metric,(qcol,tcol) in TARGETS.items():
        d=dev[["patient_id",qcol,tcol]].replace([np.inf,-np.inf],np.nan).dropna()
        te=test[["patient_id",qcol,tcol]].replace([np.inf,-np.inf],np.nan).dropna()
        if not len(d) or not len(te): continue

        # Previous NSR-QA regression protocol: one corresponding reconstruction-disagreement
        # metric, linearly calibrated on development, frozen on the outer test set.
        x=d[[qcol]].to_numpy(float); y=d[tcol].to_numpy(float)
        if np.std(x[:,0])>1e-12:
            lr=LinearRegression().fit(x,y)
            pred=lr.predict(te[[qcol]].to_numpy(float))
            slope=float(lr.coef_[0]); intercept=float(lr.intercept_)
        else:
            pred=np.full(len(te),float(np.mean(y))); slope=0.0; intercept=float(np.mean(y))
        rm=reg_metrics(te[tcol].to_numpy(float),pred)
        reg_rows.append({"target_metric":metric,"development_n":len(d),"test_n":len(te),"slope":slope,"intercept":intercept,
                         **{f"test_{k}":v for k,v in rm.items()}})
        for pid,yt,raw,yp in zip(te.patient_id,te[tcol],te[qcol],pred):
            pred_rows.append({"patient_id":pid,"target_metric":metric,"task":"regression","true_value":float(yt),
                              "score_or_prediction":float(yp),"raw_nsrqa_score":float(raw)})

        for pct in args.bad_percentiles:
            cutoff=percentile_higher(d[tcol],pct)
            yd=(d[tcol].to_numpy(float)>cutoff).astype(int); yt=(te[tcol].to_numpy(float)>cutoff).astype(int)
            sd=d[qcol].to_numpy(float); st=te[qcol].to_numpy(float)
            thr,dev_sens,dev_spec=choose_threshold(yd,sd,args.target_specificity)
            auc=float(roc_auc_score(yt,st)) if len(np.unique(yt))==2 else np.nan
            apv=float(average_precision_score(yt,st)) if np.any(yt==1) else np.nan
            cm=confusion(yt,st,thr)
            cls_rows.append({"target_metric":metric,"bad_percentile":float(pct),"true_error_cutoff":cutoff,
                             "development_n":len(d),"test_n":len(te),"test_positive_n":int(yt.sum()),"test_negative_n":int((1-yt).sum()),
                             "test_roc_auc":auc,"test_pr_auc":apv,"development_threshold_at_target_specificity":thr,
                             "development_sensitivity":dev_sens,"development_specificity":dev_spec,
                             "test_sensitivity":cm["sensitivity"],"test_specificity":cm["specificity"],"test_ppv":cm["ppv"],"test_npv":cm["npv"]})
            for pid,label,score,trueval in zip(te.patient_id,yt,st,te[tcol]):
                pred_rows.append({"patient_id":pid,"target_metric":metric,"task":f"classification_p{int(pct)}",
                                  "true_value":int(label),"score_or_prediction":float(score),"true_metric_value":float(trueval),
                                  "true_error_cutoff":cutoff,"operating_threshold":thr})

    pd.DataFrame(cls_rows).to_csv(out/"nsrqa_classification_summary.csv",index=False)
    pd.DataFrame(reg_rows).to_csv(out/"nsrqa_regression_summary.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(out/"nsrqa_test_predictions.csv",index=False)
    (out/"nsrqa_analysis_protocol.json").write_text(json.dumps({
        "baseline":"previous NSR-QA stochastic-region-dropout denoising model, fold/preprocessing matched",
        "classification_score":"corresponding raw input-vs-NSRQA-reconstruction metric",
        "classification_bad_cutoff":"development true error P75 primary; P50 secondary",
        "classification_operating_threshold":f"development raw NSR-QA score threshold targeting >= {args.target_specificity:.2f} specificity",
        "regression":"univariate linear calibration of corresponding NSR-QA metric on train+validation, frozen on test",
        "test_used_for_fitting_or_threshold_selection":False,
    },indent=2))
    print(f"Wrote matched NSR-QA analysis to {out}")

if __name__=="__main__": main()
