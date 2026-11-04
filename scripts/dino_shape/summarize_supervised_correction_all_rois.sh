#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ALLROI_ROOT="${ALLROI_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
cd "$PROJECT_ROOT"
while IFS=$'\t' read -r idx registry roi rest; do
  [[ "$idx" == "roi_index" ]] && continue
  [[ -d "${STUDY_ROOT}/evaluations/${roi}" ]] || continue
  python analysis/summarize_supervised_autoseg_correction.py --roi "$roi" --study-root "$STUDY_ROOT"
done < "${ALLROI_ROOT}/manifests/active_rois.tsv"
python - <<'PY'
from pathlib import Path
import pandas as pd
root=Path('/scratch/shaohan/contour_QA/supervised_autoseg_correction')
out=root/'summary'; frames={}
for name in ('correction_summary','ranking_summary','classification_summary'):
    xs=[]
    for p in sorted(out.glob(f'*/{name}.csv')):
        try: xs.append(pd.read_csv(p))
        except Exception: pass
    if xs: pd.concat(xs,ignore_index=True).to_csv(out/f'all_roi_{name}.csv',index=False)
print(f'Cross-ROI summaries: {out}')
PY
