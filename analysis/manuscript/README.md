# NSR-QA manuscript figures

`generate_nsrqa_manuscript_figures.py` is a read-only manuscript plotting utility. It uses existing experiment outputs and does not retrain or re-evaluate a model.

Default outputs are written to:

`/scratch/shaohan/contour_QA/manuscript_figures/nsrqa`

Generated assets:

- Figure 1: NSR-QA training/deployment schematic.
- Figures 4a/4b: separate pooled four-model AUC comparisons for Lung_L and Bronchus.
- Figures 5a/5b: separate proposed-model per-probe ASSD AUC plots for Lung_L and Bronchus.
- Figures 6a–6d: separate ASSD disagreement-versus-severity figures by ROI and corruption family (global vs local).
- Figures 7a/7b: separate asymmetric-model directional-response and reconstruction-accuracy diagnostics.
- Figures 8a/8b: separate Lung_L dilation AUC and high-specificity sensitivity loss-ablation plots.
- Figure 9: real-patient P75 classification AUC heatmap.
- `table3_lungl_dilation_loss_ablation.csv`: source values for manuscript Table 3.
- `figure_source_manifest.csv`: traceability from each figure to its source analysis.

Run interactively (lightweight plotting only):

```bash
bash scripts/dino_shape/run_generate_nsrqa_manuscript_figures.sh
```

or through Slurm:

```bash
sbatch -p hugo --account geoffrey_hugo jobs/generate_nsrqa_manuscript_figures.job
```

All source paths can be overridden with CLI arguments. Use `--skip N` if a source analysis has not yet been generated.

All quantitative figures use enlarged manuscript typography. Multi-panel quantitative figures are intentionally exported as separate files so each panel can be placed at full column/page width without shrinking labels.


## Figure 6 severity-response definition

Figure 6 intentionally follows the same two-criterion formulation used by the pooled ROC analysis instead of pooling unlike severity measures on one panel:

- **Global probes** (erosion, dilation, and directional truncations): x = absolute input-target volume change divided by the ROI-specific global threshold (20% for Lung_L, 10% for Bronchus).
- **Local probes** (`big_bite`, `big_blob`): x = HD95(input,target) divided by the ROI-specific local threshold (10 mm for Lung_L, 4 mm for Bronchus).
- In every panel, x = 1 is therefore exactly the substantial-error boundary used for ROC labeling.
- Shift probes remain excluded, consistent with the principal ROC analysis.

Bronchus main panels use a **family-specific 97.5th-percentile visual zoom** independently on x and y (configurable with `--bronchus-figure6-zoom-quantile`). Off-panel observations remain included in binned medians and all quantitative analyses, are counted on the main panel, and are also shown in automatically saved full-range supplementary companion figures.

The raw plotting rows used for Figure 6 are written to `figure6_severity_plot_data.csv` for traceability.
