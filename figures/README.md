# figures/

Scripts that read pre-computed CSVs from `data/` and write PDF figures
to `outputs/`. No Stan calls — these run in seconds to minutes.

## Running all figures

```bash
# From the repo root
for f in figures/fig_*.py; do python "$f"; done
```

## Script index

| Script | Paper figure | Key inputs from data/ | Output in outputs/ |
|--------|-------------|----------------------|--------------------|
| `fig_plate_diagram.py` | Fig S0 — Generative model (Methods) | *(none)* | `figS0_plate_diagram.pdf` |
| `fig_noise_taxonomy.py` | Fig 4 — Noise taxonomy RF vs LR | `paper5_comparison_agg.csv` | `fig_noise_taxonomy.pdf` |
| `fig_two_phase_protocol.py` | Fig 6 — Two-phase LPI-B² protocol | `paper6_two_phase_raw1.5.csv` | `fig_two_phase_protocol.pdf` |
| `fig_strong_vs_weak_capacity.py` | Fig 7 — Full vs Weak architecture | `table_multiarch_def.csv`, `sample_posteriors.csv`, + weak variants | `fig_strong_vs_weak_capacity.pdf` |
| `fig_noise_capacity_trajectories.py` | Fig 8 — Capacity & absorption trajectories | `table_multiarch_def_yc.csv`, `sample_posteriors_yc.csv` | `fig_noise_capacity_trajectories.pdf` |
| `fig_estimator_correlations.py` | Supp Fig S1 — Metric correlation matrix | `sample_posteriors_yc.csv`, `table_multiarch_def_yc.csv`, `arch_summary.csv` | `fig_estimator_correlations.pdf` |
| `fig_threshold_ablation.py` | Supp Fig S2 — Threshold ablation | `table1_noise_robustness.csv`, `table_multiarch_def_yc.csv` | `fig_threshold_ablation.pdf` |
| `fig_posterior_geometry.py` | Supp Fig S3 — Per-sample posterior geometry | `sample_posteriors.csv` | `fig_posterior_geometry.pdf` |
| `fig_ranking_recovery.py` | Fig 11 — Ranking recovery under noise | `postproc_02_classical_vs_lbi.csv`, `table_multiarch_def_*.csv` | `fig_ranking_recovery.pdf` |

## Dependency note

`fig_ranking_recovery.py` requires `data/postproc_02_classical_vs_lbi.csv`.
Run `fig_estimator_correlations.py` first if that file is missing — it
generates the CSV as a side effect.
