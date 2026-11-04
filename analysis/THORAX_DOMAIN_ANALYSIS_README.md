# Thorax Dice/HD95 domain analysis

This analysis compares auto-contour agreement across the recorded thoracic domains and a derived **Normal** group.

## Inputs

1. A **complete** Dice/HD95 metrics CSV containing all patient–ROI observations to be analyzed. Preferred schema:

```text
patient_id,site,roi,metric,value,...
```

2. `OOD_Domain_Assignment_MAD_TOTAL.csv` (or equivalent assignment file) with:

```text
patient_id,site,roi,domain,...
```

The complete metrics CSV defines the analysis universe. A patient–ROI pair not present in the assignment file is labeled `Normal`.

## Domain handling

- `I`: Imaging
- `AV`: Anatomical variation
- `MI`: Metallic implants
- `AC`: Auto-contour error
- `H`: Human/reference-contour issue
- `CA`: Clinically acceptable contour
- `NDI`: Minor edits needed, no expected dosimetric impact
- `LDI`: Major edits needed, likely dosimetric impact

The source file currently uses `GC` for the clinically acceptable category. By default the analysis displays this as `CA`, while preserving the original value in `domain_raw`. Use `--keep_gc_label` to retain `GC` in plots/tables.

Domain assignment is treated as **patient–ROI-level**. If the same patient–ROI has incompatible recorded domains, the default behavior is to write the pair to `domain_assignment_conflicts.csv` and exclude it from primary inference. This avoids silently choosing one assignment.

## Run

```bash
python analysis/analyze_thorax_domain_metrics.py \
  --metrics_csv /path/to/complete_metrics.csv \
  --domain_csv /path/to/OOD_Domain_Assignment_MAD_TOTAL.csv \
  --output_dir /path/to/thorax_domain_analysis
```

If the complete metrics table contains legitimate repeated patient–ROI–metric rows, review them first. The default is to stop. Explicit alternatives are:

```bash
--duplicate_policy median
# or
--duplicate_policy first
```

## Main outputs

The script writes:

- `report.md` — automatic results report, figure descriptions, and interpretation guidance.
- `analysis_dataset.csv` — full analysis table with propagated domain and ROI-normalized severity.
- `domain_assignment_conflicts.csv` — assignments requiring adjudication.
- `descriptive_statistics.csv` — n, median, IQR, range by ROI × metric × domain.
- `kruskal_wallis.csv` — ROI-specific omnibus tests.
- `domain_vs_normal.csv` — domain-versus-Normal tests, effect sizes, and bootstrap CIs.
- `abnormal_domain_pairwise.csv` — comparisons among recorded domains.
- `pooled_severity_domain_vs_normal.csv` — pooled analysis after ROI-specific robust normalization.
- `normalization_baselines.csv` — Normal reference median and scale for each ROI/metric.
- `figures/` — domain-count heatmap, ROI-faceted distributions, pooled severity, Dice-vs-HD95 scatter, and effect-size plot.

## Statistical interpretation

Primary inference is nonparametric. Per ROI/metric, the script uses Kruskal–Wallis and Mann–Whitney tests with Benjamini–Hochberg correction. Cliff's delta quantifies effect size. Median-difference confidence intervals use patient-cluster bootstrap resampling.

Groups smaller than `--min_group_n` (default 5) are retained in descriptive plots/tables but excluded from primary pairwise inference.

The pooled analysis uses a **Normal-referenced ROI-specific robust severity score**. Higher values always mean worse agreement: lower Dice is sign-reversed, while higher HD95 remains positive. This controls for the large baseline differences in raw Dice and HD95 among thoracic structures.

### Important caveat

The poorer-agreement domains (`I`, `AV`, `MI`, `AC`, `H`) were assigned after cases were flagged using Dice/HD95 control limits. Therefore, a strong difference between those domains and `Normal` is partly built into the selection procedure and should not be presented as an independent validation. Differences **among recorded domains**, and differences in how Dice versus HD95 respond within a domain, are more informative about domain-specific failure signatures.
