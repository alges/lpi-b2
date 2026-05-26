# LPI-B² — Reproducibility Package

Code and pre-computed data to reproduce all experiments and figures in:

> **"Latent Performance Indicator: a Bootstrap-Bayesian framework for  
> architecture-relative evaluation under label noise"**  
> A. Ehrenfeld, A. Egaña, G. Díaz, F. Navarro, J. F. Sánchez-Pérez, G. García-Ros

---

## Two modes of use

| Mode | What it does | Time |
|------|-------------|------|
| **Quick** | Reads pre-computed CSVs in `data/` and regenerates every figure | ~5 minutes |
| **Full**  | Runs Stan sampling from scratch and regenerates everything | ~6–10 hours (CPU) |

For most readers, **quick mode** is sufficient to verify all paper figures.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<org>/lpi-b2-reproducibility.git
cd lpi-b2-reproducibility

# 2a. Install with pip
pip install -r requirements.txt
pip install -e .          # makes the lpi_b2 package importable

# 2b. Alternative: conda / mamba
conda env create -f environment.yml
conda activate lpi-b2
pip install -e .
```

CmdStan (the C++ Stan backend) is installed automatically on first use via
`cmdstanpy`. No manual compilation is needed.

---

## Quick mode — regenerate all figures

Each script in `figures/` reads pre-computed CSVs from `data/` and writes
PDFs to `outputs/`. Run them individually or all at once:

```bash
# All figures at once
for f in figures/fig_*.py; do python "$f"; done

# Or individually (examples)
python figures/fig_plate_diagram.py
python figures/fig_two_phase_protocol.py
python figures/fig_noise_capacity_trajectories.py
```

### Figure → script → data mapping

| Figure in paper | Script | Input CSVs |
|----------------|--------|-----------|
| Fig S0 — Plate diagram (Methods) | `fig_plate_diagram.py` | *(none — pure matplotlib)* |
| Fig 4 — Noise taxonomy | `fig_noise_taxonomy.py` | `paper5_comparison_agg.csv` |
| Fig 6 — Two-phase LPI-B² protocol | `fig_two_phase_protocol.py` | `paper6_two_phase_raw1.5.csv` |
| Fig 7 — Full vs Weak capacity | `fig_strong_vs_weak_capacity.py` | `table_multiarch_def.csv`, `table_multiarch_def_sep1_W.csv`, `sample_posteriors.csv`, `sample_posteriors_sep1_W.csv` |
| Fig 8 — Capacity & noise trajectories | `fig_noise_capacity_trajectories.py` | `table_multiarch_def_yc.csv`, `sample_posteriors_yc.csv` |
| Fig 11 — Ranking recovery | `fig_ranking_recovery.py` | `postproc_02_classical_vs_lbi.csv`, `table_multiarch_def_*.csv` |
| Supp Fig S1 — Estimator correlations | `fig_estimator_correlations.py` | `sample_posteriors_yc.csv`, `table_multiarch_def_yc.csv`, `arch_summary.csv` |
| Supp Fig S2 — Threshold ablation | `fig_threshold_ablation.py` | `table1_noise_robustness.csv`, `table_multiarch_def_yc.csv` |
| Supp Fig S3 — Posterior geometry | `fig_posterior_geometry.py` | `sample_posteriors.csv` |

> **Note on `fig_ranking_recovery.py`:** this script requires
> `data/postproc_02_classical_vs_lbi.csv`. If that file is not present, run
> `figures/fig_estimator_correlations.py` first — it generates that intermediate
> CSV as a side effect.

---

## Full mode — run experiments from scratch

Scripts in `experiments/` run the full Stan sampling pipeline and write
results to `data/` and `outputs/`. After they finish, run the `figures/`
scripts as above.

### Experiment → figure mapping

| Experiment | Script | Stan calls | Est. time | Produces |
|-----------|--------|-----------|-----------|---------|
| Multi-arch capacity grid | `exp_multiarch_capacity.py` | ~20–30 | 3–5 h | `table_multiarch_def*.csv`, `sample_posteriors*.csv` |
| Two-phase auditing protocol | `exp_two_phase_protocol.py` | ~18 | 2–3 h | `paper6_two_phase_raw*.csv` |
| Comparison with Confident Learning | `exp_cleanlab_comparison.py` | ~72 | ~2 h | `exp_cleanlab_results.csv` |
| Prior sensitivity | `exp_prior_sensitivity.py` | 12 | ~1 h | `exp_prior_sensitivity.csv` |
| Asymmetric noise channels | `exp_asymmetric_noise.py` | ~90 | 2–3 h | `exp_asymmetric_noise.csv` |
| Real dataset validation | `exp_real_datasets.py` | ~20 | ~1 h | `exp_real_datasets.csv` |

All experiment scripts support `--dry-run` for a quick smoke-test with one
condition:

```bash
python experiments/exp_multiarch_capacity.py --dry-run
python experiments/exp_cleanlab_comparison.py --dry-run
```

---

## Repository structure

```
lpi-b2-reproducibility/
├── lpi_b2/                  # installable Python package
│   ├── __init__.py
│   ├── estimator.py         # LPIB2Evaluator — main sklearn-compatible class
│   ├── stan_utils.py        # CmdStan auto-install helper
│   └── models/              # Stan model files
│       ├── lpi_b2_bootstrap.stan              # main model (baseline priors)
│       ├── lpi_b2_bootstrap_asym_priors.stan
│       ├── lpi_b2_bootstrap_conc_priors.stan
│       └── lpi_b2_bootstrap_flat_priors.stan
│
├── experiments/             # Stan sampling runs (full mode)
├── figures/                 # figure generation from pre-computed CSVs (quick mode)
├── data/                    # pre-computed CSVs (28 files, ~30 MB)
├── outputs/                 # generated figures land here (git-ignored)
│
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

---

## Using the `LPIB2Evaluator` in your own project

```python
from sklearn.ensemble import RandomForestClassifier
from lpi_b2 import LPIB2Evaluator

evaluator = LPIB2Evaluator(
    base_estimator=RandomForestClassifier(n_estimators=100),
    n_bootstrap=15,
    anchor_rate=0.5,
)
evaluator.evaluate(X_train, y_noisy)

perf = evaluator.get_global_performance()
# {'mu_Se': 0.87, 'mu_Sp': 0.83, 'kappa_Se': 42.1, ...}

audit = evaluator.get_sample_audit()
# DataFrame with columns: posterior_truth_prob, ambiguity_score
```

---

## Citation

```bibtex
@article{ehrenfeld2025lpi,
  title   = {Latent Performance Indicator: a Bootstrap-Bayesian framework for
             architecture-relative evaluation under label noise},
  author  = {Ehrenfeld, Alejandro and Ega{\~n}a, Alvaro and D{\'i}az, Gonzalo
             and Navarro, Felipe and Garc{\'i}a-Ros, Gonzalo},
  journal = {[journal]},
  year    = {2025},
}
```

---

## License

MIT
