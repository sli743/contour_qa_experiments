#!/usr/bin/env python3
"""Summarize 5-fold out-of-fold correction, ranking and high-error detection."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
PAIRS={
 "1_minus_dice":("true_1_minus_dice","qa_1_minus_dice"),
 "hd95_mm":("true_hd95_mm","qa_hd95_mm"),
 "assd_mm":("true_assd_mm","qa_assd_mm"),
 "rel_volume_diff":("true_rel_volume_diff","qa_rel_volume_diff"),
}
def cindex(y,s):
    good=tot=0
    for i in range(len(y)):
      for j in range(i+1,len(y)):
        if y[i]==y[j]: continue
        tot+=1; d=(y[i]-y[j])*(s[i]-s[j]); good+=1 if d>0 else (.5 if d==0 else 0)
    return good/tot if tot else np.nan
def choose_thr(y,s,min_spec=.95):
    vals=np.unique(s[np.isfinite(s)]); best=None
    for t in np.r_[vals,np.inf]:
        pred=s>=t; tn=np.sum((~pred)&(~y)); fp=np.sum(pred&(~y)); tp=np.sum(pred&y); fn=np.sum((~pred)&y)
        spec=tn/(tn+fp) if tn+fp else np.nan; sens=tp/(tp+fn) if tp+fn else np.nan
        if np.isfinite(spec) and spec>=min_spec:
            cand=(sens,-t)
            if best is None or cand>best[0]: best=(cand,float(t),sens,spec)
    return (best[1],best[2],best[3]) if best else (np.nan,np.nan,np.nan)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--roi",required=True); ap.add_argument("--study-root",default="/scratch/shaohan/contour_QA/supervised_autoseg_correction"); args=ap.parse_args()
    root=Path(args.study_root); roi=args.roi; evalroot=root/"evaluations"/roi; out=root/"summary"/roi; out.mkdir(parents=True,exist_ok=True)
    tests=[]; devs={}
    for fold in range(5):
        tp=evalroot/f"fold_{fold}"/"test_metrics.csv"; dp=evalroot/f"fold_{fold}"/"development_metrics.csv"
        if not tp.is_file() or not dp.is_file(): raise FileNotFoundError(f"Missing fold {fold} metrics for {roi}")
        t=pd.read_csv(tp,dtype={"patient_id":str}); d=pd.read_csv(dp,dtype={"patient_id":str}); tests.append(t); devs[fold]=d
    oof=pd.concat(tests,ignore_index=True)
    if oof.patient_id.duplicated().any(): raise RuntimeError("OOF test patients are duplicated")
    oof.to_csv(out/"oof_patient_metrics.csv",index=False)
    correction=[]
    for metric,before,after,improve in [
      ("1_minus_dice","true_1_minus_dice","corrected_1_minus_dice",None),("hd95_mm","true_hd95_mm","corrected_hd95_mm",None),("assd_mm","true_assd_mm","corrected_assd_mm",None),("rel_volume_diff","true_rel_volume_diff","corrected_rel_volume_diff",None)]:
        b=oof[before].to_numpy(float); a=oof[after].to_numpy(float); ok=np.isfinite(b)&np.isfinite(a); b=b[ok]; a=a[ok]
        correction.append({"roi":roi,"metric":metric,"n":len(b),"mean_before":np.mean(b),"mean_after":np.mean(a),"median_before":np.median(b),"median_after":np.median(a),"median_reduction":np.median(b-a),"fraction_improved":np.mean(a<b)})
    pd.DataFrame(correction).to_csv(out/"correction_summary.csv",index=False)
    rank=[]; cls=[]; cases=[]
    for name,(yk,xk) in PAIRS.items():
        y=oof[yk].to_numpy(float); s=oof[xk].to_numpy(float); ok=np.isfinite(y)&np.isfinite(s); yy=y[ok]; ss=s[ok]
        rank.append({"roi":roi,"metric":name,"n":len(yy),"spearman_rho":spearmanr(yy,ss).statistic if len(yy)>2 else np.nan,"kendall_tau":kendalltau(yy,ss).statistic if len(yy)>2 else np.nan,"concordance_index":cindex(yy,ss)})
        all_labels=[]; all_scores=[]; all_pred=[]; all_truth=[]
        for fold in range(5):
            dt=devs[fold]; tt=oof[oof.outer_fold.astype(int)==fold]
            dy=dt[yk].to_numpy(float); ds=dt[xk].to_numpy(float); keep=np.isfinite(dy)&np.isfinite(ds); dy=dy[keep]; ds=ds[keep]
            cutoff=float(np.percentile(dy,75)); dev_bad=dy>cutoff; qthr,_,_=choose_thr(dev_bad,ds,.95)
            ty=tt[yk].to_numpy(float); ts=tt[xk].to_numpy(float); keep=np.isfinite(ty)&np.isfinite(ts); ty=ty[keep]; ts=ts[keep]; bad=ty>cutoff; pred=ts>=qthr if np.isfinite(qthr) else np.zeros_like(bad,dtype=bool)
            all_labels.extend(bad.tolist()); all_scores.extend(ts.tolist()); all_pred.extend(pred.tolist()); all_truth.extend(bad.tolist())
            for pid,tv,sv,bv,pv in zip(tt.loc[keep,"patient_id"],ty,ts,bad,pred): cases.append({"roi":roi,"metric":name,"fold":fold,"patient_id":pid,"true_error":tv,"qa_score":sv,"dev_p75_true_cutoff":cutoff,"dev_qa_threshold_95spec":qthr,"true_high_error":bool(bv),"predicted_high_error":bool(pv)})
        lab=np.asarray(all_labels,dtype=bool); score=np.asarray(all_scores,float); pred=np.asarray(all_pred,dtype=bool)
        auc=roc_auc_score(lab,score) if len(np.unique(lab))==2 else np.nan; apv=average_precision_score(lab,score) if np.any(lab) else np.nan
        tn,fp,fn,tp=confusion_matrix(lab,pred,labels=[False,True]).ravel(); sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan
        cls.append({"roi":roi,"metric":name,"n":len(lab),"n_high_error":int(lab.sum()),"n_other":int((~lab).sum()),"roc_auc":auc,"pr_auc":apv,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"sensitivity_fixed_dev95spec":sens,"specificity_fixed_dev95spec":spec})
    pd.DataFrame(rank).to_csv(out/"ranking_summary.csv",index=False); pd.DataFrame(cls).to_csv(out/"classification_summary.csv",index=False); pd.DataFrame(cases).to_csv(out/"classification_cases.csv",index=False)
    (out/"report.json").write_text(json.dumps({"roi":roi,"primary_qa_metric":"assd_mm","oof_n":len(oof),"interpretation":"All reported patient metrics are outer-fold out-of-fold predictions; each patient's correction was produced by a model that did not train or validate on that patient."},indent=2))
    print(pd.DataFrame(correction).to_string(index=False)); print(pd.DataFrame(rank).to_string(index=False)); print(pd.DataFrame(cls).to_string(index=False))
if __name__=="__main__":main()
