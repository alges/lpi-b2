"""
exp_prior_sensitivity.py — Prior sensitivity analysis for LPI-B2

Usage:
  python exp_prior_sensitivity.py --dry-run    # 1 seed, baseline only, ~minutes
  python exp_prior_sensitivity.py              # full: 4 variants × 3 seeds = 12 Stan calls

Design:
  Fixed: class_sep=1.5, flip_y=0.10, RF, B=35, anchor_rate=0.8, 3 seeds
  4 variants: baseline, flat, concentrated, asymmetric
  Stan files: lpi_b2_bootstrap.stan, lpi_b2_bootstrap_flat_priors.stan, etc. (in lpi_b2/models/)
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TABLES   = ROOT / 'data'
FIGURES  = ROOT / 'outputs'
POSTPROC = ROOT / 'data'

for d in (TABLES, FIGURES, POSTPROC):
    d.mkdir(parents=True, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / 'experiments'))
try:
    from exp_multiarch_capacity import make_data, cv_metrics, lpi_b2_global, lpi_b2_sample_posteriors
except ImportError as e:
    print(f'[exp_prior_sensitivity] ERROR: cannot import from exp_multiarch_capacity: {e}')
    sys.exit(1)

try:
    from lpi_b2.estimator import LPIB2Evaluator
except ImportError as e:
    print(f'[exp_prior_sensitivity] ERROR: cannot import LPIB2Evaluator: {e}')
    sys.exit(1)

try:
    from cmdstanpy import CmdStanModel
except ImportError as e:
    print(f'[exp_prior_sensitivity] ERROR: cannot import CmdStanModel from cmdstanpy: {e}')
    sys.exit(1)

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 110,
})

ARCH_COLORS = {
    'LR':  '#9467bd',
    'RF':  '#2ca02c',
    'SVM': '#1f77b4',
    'GB':  '#ff7f0e',
    'MLP': '#d62728',
}

TAG = '[exp_prior_sensitivity]'

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_ID = '02'

VARIANTS = [
    {
        'name':      'baseline',
        'stan_file': 'lpi_b2_bootstrap.stan',
        'label':     'Baseline',
    },
    {
        'name':      'flat',
        'stan_file': 'lpi_b2_bootstrap_flat_priors.stan',
        'label':     'Flat',
    },
    {
        'name':      'concentrated',
        'stan_file': 'lpi_b2_bootstrap_conc_priors.stan',
        'label':     'Concentrated',
    },
    {
        'name':      'asymmetric',
        'stan_file': 'lpi_b2_bootstrap_asym_priors.stan',
        'label':     'Asymmetric',
    },
]

SEEDS_FULL    = [0, 1, 2]
SEEDS_DRYRUN  = [0]

CLASS_SEP     = 1.5
FLIP_Y        = 0.10
B_BOOTSTRAP   = 35
ANCHOR_RATE   = 0.8

OUT_CSV       = POSTPROC / 'exp_prior_sensitivity.csv'
OUT_CSV_AGG   = POSTPROC / 'exp_prior_sensitivity_agg.csv'
OUT_FIG       = FIGURES  / 'exp_prior_sensitivity'
STATUS_FILE   = POSTPROC / 'newexp_02_status.md'
PROGRESS_FILE = POSTPROC / 'NEWEXP_PROGRESS.md'

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_convergence(mcmc_fit, params=('mu_Se', 'mu_Sp', 'kappa_Se', 'kappa_Sp')):
    try:
        summ = mcmc_fit.summary()
        pop  = summ[summ.index.isin(list(params))]
        return (
            pop['R_hat'].max() < 1.05 and pop['N_Eff'].min() > 100,
            pop['R_hat'].max(),
            pop['N_Eff'].min(),
        )
    except Exception:
        return False, float('nan'), float('nan')


def write_status(script_id, title, mode, status, conv_str, key_figures, outputs, notes):
    lines = [
        f'# newexp_{script_id} — {title}',
        f'',
        f'**Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'**Mode:** {mode}',
        f'**Status:** {status}',
        f'**Convergence:** {conv_str}',
        f'',
        f'## Key figures',
        *[f'- {f}' for f in key_figures],
        f'',
        f'## Outputs',
        *[f'- {o}' for o in outputs],
        f'',
        f'## Notes',
        *[f'- {n}' for n in notes],
    ]
    STATUS_FILE.write_text('\n'.join(lines))


def append_progress(script_id, mode, status, conv_str, key_line, outputs, notes):
    entry = (
        f'\n## newexp_{script_id} [{datetime.now().strftime("%Y-%m-%d %H:%M")}]\n'
        f'**Mode:** {mode} | **Status:** {status} | **Conv:** {conv_str}\n'
        f'**Key:** {key_line}\n'
        f'**Outputs:** {", ".join(outputs)}\n'
        f'**Notes:** {", ".join(notes)}\n'
    )
    with open(PROGRESS_FILE, 'a') as fh:
        fh.write(entry)


def extract_variant_descriptors(fit, y_noisy):
    """Extract posterior means and fractions from a CmdStanModel fit."""
    mu_Se_draws   = fit.stan_variable('mu_Se')
    mu_Sp_draws   = fit.stan_variable('mu_Sp')
    kappa_Se_draws = fit.stan_variable('kappa_Se')
    kappa_Sp_draws = fit.stan_variable('kappa_Sp')
    kappa_obs_draws = fit.stan_variable('kappa_obs')
    prob_Ti_draws = fit.stan_variable('prob_Ti_pos')   # shape (draws, N)

    mu_Se_mean = float(np.mean(mu_Se_draws))
    mu_Se_q05  = float(np.percentile(mu_Se_draws, 5))
    mu_Se_q95  = float(np.percentile(mu_Se_draws, 95))

    mu_Sp_mean = float(np.mean(mu_Sp_draws))
    mu_Sp_q05  = float(np.percentile(mu_Sp_draws, 5))
    mu_Sp_q95  = float(np.percentile(mu_Sp_draws, 95))

    kappa_Se_mean  = float(np.mean(kappa_Se_draws))
    kappa_Sp_mean  = float(np.mean(kappa_Sp_draws))
    kappa_obs_mean = float(np.mean(kappa_obs_draws))

    # per-sample posterior mean
    p_truth = np.mean(prob_Ti_draws, axis=0)

    # discordance and entropy per sample
    eps         = 1e-9
    discordance = np.abs(y_noisy.astype(float) - p_truth)
    entropy     = -(p_truth * np.log(p_truth + eps) +
                    (1 - p_truth) * np.log(1 - p_truth + eps))

    f_A = float(np.mean(discordance > 0.8))
    f_B = float(np.mean((discordance <= 0.8) & (entropy > 0.8)))
    f_C = 1.0 - f_A - f_B

    return dict(
        mu_Se_mean=mu_Se_mean, mu_Se_q05=mu_Se_q05, mu_Se_q95=mu_Se_q95,
        mu_Sp_mean=mu_Sp_mean, mu_Sp_q05=mu_Sp_q05, mu_Sp_q95=mu_Sp_q95,
        kappa_Se_mean=kappa_Se_mean, kappa_Sp_mean=kappa_Sp_mean,
        kappa_obs_mean=kappa_obs_mean,
        f_A=f_A, f_B=f_B, f_C=f_C,
    )


def load_done_set():
    if not OUT_CSV.exists():
        return set()
    try:
        df = pd.read_csv(OUT_CSV)
        return set(zip(df['prior_variant'], df['seed']))
    except Exception:
        return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dry_run=False):
    mode  = 'dry-run' if dry_run else 'full'
    seeds = SEEDS_DRYRUN if dry_run else SEEDS_FULL
    variants_to_run = VARIANTS[:1] if dry_run else VARIANTS

    # Check variant .stan files exist (skip baseline which is always there)
    blocked = False
    for v in variants_to_run[1:]:
        stan_path = ROOT / 'lpi_b2' / 'models' / v['stan_file']
        if not stan_path.exists():
            print(f'{TAG} BLOCKED: stan file not found: {stan_path}')
            blocked = True
    if blocked and not dry_run:
        print(f'{TAG} BLOCKED: one or more variant Stan files are missing. '
              f'Run dry-run or add the missing .stan files.')
        sys.exit(1)

    done_set = load_done_set()
    print(f'{TAG} mode={mode}, seeds={seeds}, variants={[v["name"] for v in variants_to_run]}')
    print(f'{TAG} already done: {len(done_set)} rows')

    all_rows = []

    for seed in seeds:
        print(f'{TAG} --- seed={seed} ---')
        X, y_noisy, y_clean = make_data(n=1000, sep=CLASS_SEP, flip=FLIP_Y, seed=seed)
        arch = RandomForestClassifier(n_estimators=100, random_state=0)

        # --- Step 1: Baseline (LPIB2Evaluator) ----------------------
        baseline_variant = VARIANTS[0]
        key = (baseline_variant['name'], seed)
        row_baseline = None

        if key in done_set:
            print(f'{TAG}   skip {key} (already done)')
        else:
            print(f'{TAG}   running baseline (seed={seed}) ...')
            ev = LPIB2Evaluator(
                base_estimator=arch,
                n_bootstrap=B_BOOTSTRAP,
                anchor_rate=ANCHOR_RATE,
                stan_file=str(ROOT / 'lpi_b2' / 'models' / baseline_variant['stan_file']),
                stan_chains=4,
                stan_chains_size=1000,
                stan_chains_warmup=250,
            )
            ev.evaluate(X, y_noisy)
            stan_data = ev.stan_data_

            conv_ok, rhat_max, ess_min = check_convergence(ev.mcmc_fit_)
            desc = extract_variant_descriptors(ev.mcmc_fit_, y_noisy)

            row_baseline = dict(
                prior_variant=baseline_variant['name'],
                seed=seed,
                **desc,
                rhat_max=rhat_max,
                ess_min=ess_min,
                convergence_ok=conv_ok,
            )
            all_rows.append(row_baseline)
            _write_row(row_baseline)
            print(f'{TAG}   baseline done: rhat={rhat_max:.3f}, ess={ess_min:.0f}, conv={conv_ok}')

        # --- Step 2: Non-baseline variants (reuse Q matrix) -----------------
        if not dry_run:
            # Need stan_data even if baseline was already done
            if row_baseline is None:
                # Rebuild Q matrix for non-baseline variants
                print(f'{TAG}   rebuilding Q matrix for variant Stan calls (seed={seed})')
                ev = LPIB2Evaluator(
                    base_estimator=arch,
                    n_bootstrap=B_BOOTSTRAP,
                    anchor_rate=ANCHOR_RATE,
                    stan_file=str(ROOT / 'lpi_b2' / 'models' / baseline_variant['stan_file']),
                    stan_chains=4,
                    stan_chains_size=1000,
                    stan_chains_warmup=250,
                )
                ev.evaluate(X, y_noisy)
                stan_data = ev.stan_data_

            for v in variants_to_run[1:]:
                key_v = (v['name'], seed)
                if key_v in done_set:
                    print(f'{TAG}   skip {key_v} (already done)')
                    continue

                stan_path = ROOT / 'lpi_b2' / 'models' / v['stan_file']
                if not stan_path.exists():
                    print(f'{TAG}   WARNING: {stan_path} not found, skipping variant {v["name"]}')
                    continue

                print(f'{TAG}   running variant={v["name"]} (seed={seed}) ...')
                try:
                    variant_model = CmdStanModel(
                        stan_file=str(stan_path),
                        cpp_options={'STAN_THREADS': True},
                    )
                    variant_fit = variant_model.sample(
                        data=stan_data,
                        iter_sampling=1000,
                        iter_warmup=250,
                        chains=4,
                        parallel_chains=4,
                        threads_per_chain=16,
                        seed=31032026,
                    )
                    conv_ok, rhat_max, ess_min = check_convergence(variant_fit)
                    desc = extract_variant_descriptors(variant_fit, y_noisy)

                    row = dict(
                        prior_variant=v['name'],
                        seed=seed,
                        **desc,
                        rhat_max=rhat_max,
                        ess_min=ess_min,
                        convergence_ok=conv_ok,
                    )
                    all_rows.append(row)
                    _write_row(row)
                    print(f'{TAG}   variant={v["name"]} done: rhat={rhat_max:.3f}, ess={ess_min:.0f}, conv={conv_ok}')
                except Exception as exc:
                    print(f'{TAG}   ERROR running variant={v["name"]}: {exc}')

    # ── Aggregate ──────────────────────────────────────────────────────────────
    if OUT_CSV.exists():
        df_all = pd.read_csv(OUT_CSV)
    else:
        df_all = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    if df_all.empty:
        print(f'{TAG} WARNING: no data to aggregate')
        return

    _write_agg(df_all)
    _make_figure(df_all, mode)

    # ── Status / progress ──────────────────────────────────────────────────────
    conv_summary = df_all.groupby('prior_variant')['convergence_ok'].mean()
    conv_str = '; '.join(f'{v}={p:.0%}' for v, p in conv_summary.items())

    baseline_df = df_all[df_all['prior_variant'] == 'baseline']
    key_fig = f'mu_Se={baseline_df["mu_Se_mean"].mean():.3f}, mu_Sp={baseline_df["mu_Sp_mean"].mean():.3f}'

    write_status(
        SCRIPT_ID,
        title='Prior sensitivity analysis',
        mode=mode,
        status='DONE',
        conv_str=conv_str,
        key_figures=[key_fig],
        outputs=[str(OUT_CSV), str(OUT_CSV_AGG), str(OUT_FIG) + '.{pdf,png}'],
        notes=[f'variants={[v["name"] for v in variants_to_run]}', f'seeds={seeds}'],
    )
    append_progress(
        SCRIPT_ID,
        mode=mode,
        status='DONE',
        conv_str=conv_str,
        key_line=key_fig,
        outputs=[OUT_CSV.name, OUT_CSV_AGG.name],
        notes=[f'variants={len(variants_to_run)}, seeds={len(seeds)}'],
    )
    print(f'{TAG} DONE. CSV={OUT_CSV}, AGG={OUT_CSV_AGG}, FIG={OUT_FIG}.pdf/.png')


def _write_row(row):
    df = pd.DataFrame([row])
    df.to_csv(OUT_CSV, mode='a', header=not OUT_CSV.exists(), index=False)


def _write_agg(df_all):
    numeric_cols = [
        'mu_Se_mean', 'mu_Se_q05', 'mu_Se_q95',
        'mu_Sp_mean', 'mu_Sp_q05', 'mu_Sp_q95',
        'kappa_Se_mean', 'kappa_Sp_mean', 'kappa_obs_mean',
        'f_A', 'f_B', 'f_C',
        'rhat_max', 'ess_min',
    ]
    agg_mean = df_all.groupby('prior_variant')[numeric_cols].mean()
    agg_std  = df_all.groupby('prior_variant')[numeric_cols].std()

    rows_agg = []
    # baseline reference
    baseline = agg_mean.loc['baseline'] if 'baseline' in agg_mean.index else None

    for variant in agg_mean.index:
        r = {'prior_variant': variant}
        for c in numeric_cols:
            r[f'{c}_mean'] = agg_mean.loc[variant, c]
            r[f'{c}_std']  = agg_std.loc[variant, c] if variant in agg_std.index else float('nan')

        # relative deltas vs baseline
        if baseline is not None and variant != 'baseline':
            mu_avg_base = (baseline['mu_Se_mean'] + baseline['mu_Sp_mean']) / 2
            mu_avg_var  = (agg_mean.loc[variant, 'mu_Se_mean'] + agg_mean.loc[variant, 'mu_Sp_mean']) / 2
            r['delta_mu_avg_rel'] = (mu_avg_var - mu_avg_base) / (mu_avg_base + 1e-9)

            kappa_avg_base = (baseline['kappa_Se_mean'] + baseline['kappa_Sp_mean']) / 2
            kappa_avg_var  = (agg_mean.loc[variant, 'kappa_Se_mean'] + agg_mean.loc[variant, 'kappa_Sp_mean']) / 2
            r['delta_kappa_avg_rel'] = (kappa_avg_var - kappa_avg_base) / (kappa_avg_base + 1e-9)

            r['delta_f_A_rel'] = (agg_mean.loc[variant, 'f_A'] - baseline['f_A']) / (baseline['f_A'] + 1e-9)
            r['delta_f_C_rel'] = (agg_mean.loc[variant, 'f_C'] - baseline['f_C']) / (baseline['f_C'] + 1e-9)
        else:
            r['delta_mu_avg_rel']    = 0.0
            r['delta_kappa_avg_rel'] = 0.0
            r['delta_f_A_rel']       = 0.0
            r['delta_f_C_rel']       = 0.0

        rows_agg.append(r)

    df_agg = pd.DataFrame(rows_agg)
    df_agg.to_csv(OUT_CSV_AGG, index=False)
    print(f'{TAG} aggregated CSV saved: {OUT_CSV_AGG}')


def _make_figure(df_all, mode):
    """2×2 figure: A=mu_avg bars, B=kappa_avg bars, C=stacked f_A/f_B/f_C, D=relative table."""

    variant_order  = [v['name']  for v in VARIANTS]
    variant_labels = [v['label'] for v in VARIANTS]
    # keep only variants that exist in data
    variant_order  = [v for v in variant_order  if v in df_all['prior_variant'].unique()]
    variant_labels = [VARIANTS[i]['label'] for i, v in enumerate(
        [v['name'] for v in VARIANTS]) if v in variant_order]

    agg_mean = df_all.groupby('prior_variant').mean(numeric_only=True)
    agg_std  = df_all.groupby('prior_variant').std(numeric_only=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Prior sensitivity analysis', fontsize=13, fontweight='bold')

    x = np.arange(len(variant_order))
    bar_w = 0.35
    colors_se = '#7F77DD'
    colors_sp = '#1D9E75'

    # ── Panel A: mu_avg bars ─────────────────────────────────────────────────
    ax = axes[0, 0]
    mu_se = [agg_mean.loc[v, 'mu_Se_mean'] if v in agg_mean.index else 0.0 for v in variant_order]
    mu_sp = [agg_mean.loc[v, 'mu_Sp_mean'] if v in agg_mean.index else 0.0 for v in variant_order]
    mu_se_std = [agg_std.loc[v, 'mu_Se_mean'] if v in agg_std.index else 0.0 for v in variant_order]
    mu_sp_std = [agg_std.loc[v, 'mu_Sp_mean'] if v in agg_std.index else 0.0 for v in variant_order]

    ax.bar(x - bar_w/2, mu_se, bar_w, yerr=mu_se_std, capsize=4,
           color=colors_se, alpha=0.85, label='μ_Se')
    ax.bar(x + bar_w/2, mu_sp, bar_w, yerr=mu_sp_std, capsize=4,
           color=colors_sp, alpha=0.85, label='μ_Sp')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, rotation=15, ha='right')
    ax.set_ylabel('Posterior mean')
    ax.set_title('A: μ_Se and μ_Sp by prior variant')
    ax.set_ylim(0, 1)
    ax.legend()

    # ── Panel B: kappa_avg bars ──────────────────────────────────────────────
    ax = axes[0, 1]
    k_se = [agg_mean.loc[v, 'kappa_Se_mean'] if v in agg_mean.index else 0.0 for v in variant_order]
    k_sp = [agg_mean.loc[v, 'kappa_Sp_mean'] if v in agg_mean.index else 0.0 for v in variant_order]
    k_se_std = [agg_std.loc[v, 'kappa_Se_mean'] if v in agg_std.index else 0.0 for v in variant_order]
    k_sp_std = [agg_std.loc[v, 'kappa_Sp_mean'] if v in agg_std.index else 0.0 for v in variant_order]

    ax.bar(x - bar_w/2, k_se, bar_w, yerr=k_se_std, capsize=4,
           color=colors_se, alpha=0.85, label='κ_Se')
    ax.bar(x + bar_w/2, k_sp, bar_w, yerr=k_sp_std, capsize=4,
           color=colors_sp, alpha=0.85, label='κ_Sp')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, rotation=15, ha='right')
    ax.set_ylabel('Posterior mean')
    ax.set_title('B: κ_Se and κ_Sp by prior variant')
    ax.legend()

    # ── Panel C: stacked f_A / f_B / f_C ────────────────────────────────────
    ax = axes[1, 0]
    f_A = [agg_mean.loc[v, 'f_A'] if v in agg_mean.index else 0.0 for v in variant_order]
    f_B = [agg_mean.loc[v, 'f_B'] if v in agg_mean.index else 0.0 for v in variant_order]
    f_C = [agg_mean.loc[v, 'f_C'] if v in agg_mean.index else 0.0 for v in variant_order]

    ax.bar(x, f_A, bar_w * 2, color='#E24B4A', alpha=0.9, label='Type A (discordant)')
    ax.bar(x, f_B, bar_w * 2, bottom=f_A, color='#EF9F27', alpha=0.9, label='Type B (ambiguous)')
    ax.bar(x, f_C, bar_w * 2,
           bottom=[a + b for a, b in zip(f_A, f_B)],
           color='#B4B2A9', alpha=0.9, label='Type C (concordant)')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, rotation=15, ha='right')
    ax.set_ylabel('Fraction of samples')
    ax.set_title('C: Sample fractions f_A, f_B, f_C by variant')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=9)

    # ── Panel D: relative change table ───────────────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')

    if OUT_CSV_AGG.exists():
        df_agg = pd.read_csv(OUT_CSV_AGG)
        cols_show = ['prior_variant', 'delta_mu_avg_rel', 'delta_kappa_avg_rel',
                     'delta_f_A_rel', 'delta_f_C_rel']
        df_show = df_agg[cols_show].copy()
        df_show.columns = ['Variant', 'Δμ_avg', 'Δκ_avg', 'Δf_A', 'Δf_C']
        # format numeric cols as percentage
        for c in ['Δμ_avg', 'Δκ_avg', 'Δf_A', 'Δf_C']:
            df_show[c] = df_show[c].apply(lambda v: f'{v:+.1%}' if pd.notna(v) else '—')

        table_data = [df_show.columns.tolist()] + df_show.values.tolist()
        tbl = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        ax.set_title('D: Relative changes vs baseline', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'Aggregate CSV not yet available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('D: Relative changes vs baseline', fontsize=12)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out_path = Path(f'{OUT_FIG}.{ext}')
        fig.savefig(out_path, bbox_inches='tight')
        print(f'{TAG} figure saved: {out_path}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prior sensitivity analysis for LPI-B2')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run with 1 seed, baseline only (~minutes)')
    args = parser.parse_args()
    run(dry_run=args.dry_run)
