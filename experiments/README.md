# experiments/

Scripts that run the full LPI-B² sampling pipeline (Stan MCMC + sklearn).
These are the **full mode** entry points — each one takes at least 1 hour.

For most readers, the pre-computed results in `data/` are sufficient to
regenerate all paper figures using the scripts in `figures/`.

## Running

All scripts accept `--dry-run` to run a single condition and verify the
pipeline works on your machine before committing to the full grid:

```bash
python experiments/exp_multiarch_capacity.py --dry-run   # ~10 min
python experiments/exp_cleanlab_comparison.py --dry-run  # ~5 min
```

Full runs (from the repo root):

```bash
python experiments/exp_multiarch_capacity.py      # ~3-5 h — generates bulk of data/
python experiments/exp_two_phase_protocol.py      # ~2-3 h
python experiments/exp_cleanlab_comparison.py     # ~2 h
python experiments/exp_prior_sensitivity.py       # ~1 h
python experiments/exp_asymmetric_noise.py        # ~2-3 h
python experiments/exp_real_datasets.py           # ~1 h
```

## Script index

| Script | Paper section | Stan calls | Outputs |
|--------|--------------|-----------|---------|
| `exp_multiarch_capacity.py` | §"Latent capacity and consistency", §"Posterior geometry" | 20–30 | `table_multiarch_def*.csv`, `sample_posteriors*.csv` |
| `exp_two_phase_protocol.py` | §"The two-phase auditing protocol" | ~18 | `paper6_two_phase_raw*.csv` |
| `exp_cleanlab_comparison.py` | §"Comparison with Confident Learning" | ~72 | `exp_cleanlab_results.csv` |
| `exp_prior_sensitivity.py` | Supplementary Table S1 | 12 | `exp_prior_sensitivity.csv` |
| `exp_asymmetric_noise.py` | §"Asymmetric noise channels" | ~90 | `exp_asymmetric_noise.csv` |
| `exp_real_datasets.py` | §"Validation on real datasets" | ~20 | `exp_real_datasets.csv` |

## Notes on computational requirements

- Each Stan call uses 4 chains × 16 threads. A 32-core machine can run one
  call in 3–5 minutes.
- Results are written incrementally to `data/` — the scripts are idempotent
  and resume from the last completed condition if interrupted.
- `exp_prior_sensitivity.py` imports utility functions from
  `exp_multiarch_capacity.py`. Run that script first or ensure it is on
  the Python path.
