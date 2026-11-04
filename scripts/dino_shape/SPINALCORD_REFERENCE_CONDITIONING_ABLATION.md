# SpinalCord reference-conditioning ablation

This experiment tests whether the Month3 deployment gap is caused by loss of
global position when a candidate contour is centered on its own bounding box.
The calibrated V4 corruption distribution and all non-reference-frame training
settings remain fixed.

## Arms

| arm | deployment-matched candidate centering | global spatial context | training FOV jitter |
|---|---:|---:|---:|
| `centered` | yes | no | no |
| `centered_context` | yes | yes | no |
| `centered_jitter` | yes | no | yes |
| `centered_context_jitter` | yes | yes | yes |

Synthetic corruption is applied first in the existing approved-centered,
coverage-safe FOV. The corrupted candidate bbox midpoint is then used to
translate both the corrupted input and approved target together. This mimics
the deployed `autoseg_centered` crop without regenerating the source dataset.
Candidate centering may reproduce genuine deployment clipping and logs that
condition per sample. Jitter is training-only and is rejected when it would
introduce additional approved-target clipping.

The optional 12-D context vector contains crop center and candidate center in
raw-CT normalized continuous-index coordinates, candidate bbox span normalized
by the NSR-QA FOV, and raw-CT span normalized by the NSR-QA FOV. It is injected
through a zero-initialized FiLM layer after the decoder projection; the frozen
3DINO encoder still receives only the binary contour.

## Run

```bash
bash scripts/dino_shape/submit_spinalcord_reference_conditioning_ablation.sh
```

The submitter creates the four configs, launches a four-task GPU training array,
and submits evaluation with an `afterok` dependency.

Default study root:

```text
/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_reference_ablation
```

Primary report:

```text
reports/reference_conditioning_ablation/report.md
```

The report contains all/filtered Month3 P75 AUROC, paired bootstrap AUROC
differences versus `centered`, correction decomposition, clean identity, and a
sentinel-patient table. The default manual acquisition-limited sensitivity
exclusion is patients 95, 129, and 145 and can be changed with
`EXCLUDE_PATIENTS`.

The V4-family diagnostic is rerun under deployment-matched candidate centering
for all four arms and written under `diagnostics/v4_family_correction/`.
