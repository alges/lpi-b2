"""
exp_asymmetric_noise.py — Asymmetric noise channel analysis for LPI-B2

Usage:
  python exp_asymmetric_noise.py --dry-run    # 1 condition (sym_low, LR, seed=0)
  python exp_asymmetric_noise.py              # full: 90 inferences, ~2.5-3h

Grid: 5 channels × 5 archs × 4 seeds (MLP: 2 seeds) = 90 inferences
Fixed: class_sep=1.5, B=35, anchor_rate=0.5
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from scipy import stats

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TABLES   = ROOT / 'data'
FIGURES  = ROOT / 'outputs'
POSTPROC = ROOT / 'data'

for d in (TABLES, FIGURES, POSTPROC):
    d.mkdir(parents=True, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from experiments_3_multi_arch import make_data, cv_metrics, btd_global, btd_sample_posteriors
except ImportError as e:
    print(f'[exp_asymmetric_noise] ERROR: cannot import from experiments_3_multi_arch: {e}')
    sys.exit(1)

try:
    from lpi_b2.estimator import LPIB2Evaluator
except ImportError as e:
    print(f'[exp_asymmetric_noise] ERROR: cannot import LPIB2Evaluator: {e}')
    sys.exit(1)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, recall_score, f1_score

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

# 5-color qualitative palette for channels
CHANNEL_COLORS = {
    'sym_low':    '#4878cf',
    'sym_high':   '#6acc65',
    'asym_FP':    '#d65f5f',
    'asym_FN':    '#b47cc7',
    'asym_strong':'#c4ad66',
}

TAG       = '[exp_asymmetric_noise]'
SCRIPT_ID = '03'

# ── Configuration ─────────────────────────────────────────────────────────────
ARCHITECTURES = {
    'LR':  LogisticRegression(max_iter=1000, random_state=0),
    'RF':  RandomForestClassifier(n_estimators=100, random_state=0),
    'SVM': SVC(kernel='rbf', probability=True, random_state=0),
    'GB':  GradientBoostingClassifier(n_estimators=100, random_state=0),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0),
}

# Seeds per architecture (MLP uses only 2)
SEEDS_PER_ARCH = {
    'LR':  [0, 1, 2, 3],
    'RF':  [0, 1, 2, 3],
    'SVM': [0, 1, 2, 3],
    'GB':  [0, 1, 2, 3],
    'MLP': [0, 1],
}

CHANNELS = [
    {'name': 'sym_low',    'pi_01': 0.10, 'pi_10': 0.10},
    {'name': 'sym_high',   'pi_01': 0.20, 'pi_10': 0.20},
    {'name': 'asym_FP',    'pi_01': 0.20, 'pi_10': 0.10},
    {'name': 'asym_FN',    'pi_01': 0.10, 'pi_10': 0.20},
    {'name': 'asym_strong','pi_01': 0.30, 'pi_10': 0.05},
]

CLASS_SEP   = 1.5
B_BOOTSTRAP = 35
ANCHOR_RATE = 0.5

OUT_CSV       = POSTPROC / 'exp_asymmetric_noise.csv'
OUT_FIG1      = FIGURES  / 'asymmetry_detection'
OUT_FIG2      = FIGURES  / 'arch_comparison'
STATUS_FILE   = POSTPROC / 'newexp_03_status.md'
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


def inject_asymmetric_noise(y, pi_01, pi_10, rng):
    """Apply asymmetric label noise: P(y=1|T=0)=pi_01, P(y=0|T=1)=pi_10."""
    y_noisy = y.copy()
    mask_0  = (y == 0)
    mask_1  = (y == 1)
    flip_0  = rng.random(mask_0.sum()) < pi_01
    flip_1  = rng.random(mask_1.sum()) < pi_10
    y_noisy[mask_0] = np.where(flip_0, 1, y_noisy[mask_0])
    y_noisy[mask_1] = np.where(flip_1, 0, y_noisy[mask_1])
    return y_noisy


def cv_metrics_simple(arch, X, y, cv=5):
    from sklearn.base import clone
    scoring = {
        'accuracy': 'accuracy',
        'f1':       make_scorer(f1_score, zero_division=0),
        'recall':   make_scorer(recall_score, zero_division=0),
    }
    res = cross_validate(clone(arch), X, y, cv=cv, scoring=scoring)
    return (
        float(np.mean(res['test_accuracy'])),
        float(np.mean(res['test_f1'])),
        float(np.mean(res['test_recall'])),
    )


def load_done_set():
    if not OUT_CSV.exists():
        return set()
    try:
        df = pd.read_csv(OUT_CSV)
        return set(zip(df['channel'], df['arch'], df['seed']))
    except Exception:
        return set()


def write_row(row):
    df = pd.DataFrame([row])
    df.to_csv(OUT_CSV, mode='a', header=not OUT_CSV.exists(), index=False)


# ── Main run ──────────────────────────────────────────────────────────────────

def run(dry_run=False):
    mode = 'dry-run' if dry_run else 'full'

    if dry_run:
        channels_run = [CHANNELS[0]]   # sym_low only
        archs_run    = ['LR']
        seeds_run    = {a: [0] for a in archs_run}
    else:
        channels_run = CHANNELS
        archs_run    = list(ARCHITECTURES.keys())
        seeds_run    = SEEDS_PER_ARCH

    done_set = load_done_set()
    total_planned = sum(
        len(seeds_run.get(a, [0])) * len(channels_run)
        for a in archs_run
    )
    print(f'{TAG} mode={mode}, planned={total_planned} inferences, already_done={len(done_set)}')

    for ch in channels_run:
        for arch_name in archs_run:
            arch      = ARCHITECTURES[arch_name]
            seed_list = seeds_run.get(arch_name, [0]) if not dry_run else [0]

            for seed in seed_list:
                key = (ch['name'], arch_name, seed)
                if key in done_set:
                    print(f'{TAG}   skip {key}')
                    continue

                print(f'{TAG}   channel={ch["name"]} arch={arch_name} seed={seed}')

                # 1. Generate clean data
                X, _, y_clean = make_data(n=1000, sep=CLASS_SEP, flip=0.0, seed=seed)

                # 2. Apply asymmetric noise
                rng    = np.random.default_rng(seed + 20000)
                y_noisy = inject_asymmetric_noise(y_clean, ch['pi_01'], ch['pi_10'], rng)

                # 3. Run BTD
                try:
                    ev = LPIB2Evaluator(
                        base_estimator=arch,
                        n_bootstrap=B_BOOTSTRAP,
                        anchor_rate=ANCHOR_RATE,
                        stan_file=str(ROOT / 'lpi_b2' / 'models' / 'lpi_b2_bootstrap.stan'),
                        stan_chains=4,
                        stan_chains_size=1000,
                        stan_chains_warmup=250,
                    )
                    ev.evaluate(X, y_noisy)
                except Exception as exc:
                    print(f'{TAG}   ERROR in BTD: {exc}')
                    continue

                # 4. Extract descriptors
                fit = ev.mcmc_fit_
                mu_Se_draws  = fit.stan_variable('mu_Se')
                mu_Sp_draws  = fit.stan_variable('mu_Sp')
                kappa_Se_draws = fit.stan_variable('kappa_Se')
                kappa_Sp_draws = fit.stan_variable('kappa_Sp')
                prob_Ti_draws  = fit.stan_variable('prob_Ti_pos')   # (draws, N)

                mu_Se   = float(np.mean(mu_Se_draws))
                mu_Sp   = float(np.mean(mu_Sp_draws))
                mu_avg  = (mu_Se + mu_Sp) / 2
                kappa_Se = float(np.mean(kappa_Se_draws))
                kappa_Sp = float(np.mean(kappa_Sp_draws))

                asym_score = (mu_Sp - mu_Se) / (mu_Sp + mu_Se + 1e-9)

                p_truth     = np.mean(prob_Ti_draws, axis=0)
                eps         = 1e-9
                discordance = np.abs(y_noisy.astype(float) - p_truth)
                entropy     = -(p_truth * np.log(p_truth + eps) +
                                (1 - p_truth) * np.log(1 - p_truth + eps))

                f_A = float(np.mean(discordance > 0.8))
                f_B = float(np.mean((discordance <= 0.8) & (entropy > 0.8)))
                f_C = 1.0 - f_A - f_B

                # class-conditional f_A
                mask_c0 = y_noisy == 0
                mask_c1 = y_noisy == 1
                f_A_class0 = float(np.mean(discordance[mask_c0] > 0.8)) if mask_c0.sum() > 0 else float('nan')
                f_A_class1 = float(np.mean(discordance[mask_c1] > 0.8)) if mask_c1.sum() > 0 else float('nan')

                # 5. CV metrics
                try:
                    cv_acc, cv_f1, cv_recall = cv_metrics_simple(arch, X, y_noisy)
                except Exception:
                    cv_acc = cv_f1 = cv_recall = float('nan')

                # 6. Convergence
                conv_ok, rhat_max, ess_min = check_convergence(fit)

                row = dict(
                    channel=ch['name'],
                    pi_01=ch['pi_01'],
                    pi_10=ch['pi_10'],
                    arch=arch_name,
                    seed=seed,
                    mu_Se=mu_Se,
                    mu_Sp=mu_Sp,
                    mu_avg=mu_avg,
                    asym_score=asym_score,
                    kappa_Se=kappa_Se,
                    kappa_Sp=kappa_Sp,
                    f_A=f_A,
                    f_A_class0=f_A_class0,
                    f_A_class1=f_A_class1,
                    f_B=f_B,
                    f_C=f_C,
                    cv_acc=cv_acc,
                    cv_f1=cv_f1,
                    cv_recall=cv_recall,
                    convergence_ok=conv_ok,
                    rhat_max=rhat_max,
                    ess_min=ess_min,
                )
                write_row(row)
                print(f'{TAG}   done: mu_Se={mu_Se:.3f}, mu_Sp={mu_Sp:.3f}, '
                      f'asym={asym_score:.3f}, conv={conv_ok}')

    # ── Figures ────────────────────────────────────────────────────────────────
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        if not df.empty:
            _make_figure1(df)
            _make_figure2(df)
        else:
            print(f'{TAG} WARNING: CSV empty, skipping figures')
    else:
        print(f'{TAG} WARNING: no CSV found, skipping figures')

    # ── Status ─────────────────────────────────────────────────────────────────
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        conv_rate = df['convergence_ok'].mean() if not df.empty else float('nan')
        conv_str  = f'{conv_rate:.0%} rows converged'
        key_line  = f'rows={len(df)}, asym_score_mean={df["asym_score"].mean():.3f}'
    else:
        conv_str = 'no data'
        key_line = 'no data'

    write_status(
        SCRIPT_ID,
        title='Asymmetric noise channel analysis',
        mode=mode,
        status='DONE',
        conv_str=conv_str,
        key_figures=[
            str(OUT_FIG1) + '.{pdf,png}',
            str(OUT_FIG2) + '.{pdf,png}',
        ],
        outputs=[str(OUT_CSV)],
        notes=[f'channels={len(channels_run)}, archs={archs_run}'],
    )
    append_progress(
        SCRIPT_ID,
        mode=mode,
        status='DONE',
        conv_str=conv_str,
        key_line=key_line,
        outputs=[OUT_CSV.name],
        notes=[f'channels={len(channels_run)}, archs={len(archs_run)}'],
    )
    print(f'{TAG} DONE. CSV={OUT_CSV}')


# ── Figure 1: asymmetry_detection.pdf ────────────────────────────────────────

def _make_figure1(df):
    """
    Panel A: scatter asym_score vs (pi_01 - pi_10), color by arch.
    Panel B: grouped bar f_A_class0 vs f_A_class1 by channel, RF only.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Asymmetry detection', fontsize=13, fontweight='bold')

    # ── Panel A ────────────────────────────────────────────────────────────────
    # aggregate over seeds per (channel, arch)
    grp = df.groupby(['channel', 'arch', 'pi_01', 'pi_10']).agg(
        asym_score=('asym_score', 'mean'),
        pi_01=('pi_01', 'first'),
        pi_10=('pi_10', 'first'),
    ).reset_index(drop=True)

    # rebuild from groupby
    grp2 = df.groupby(['channel', 'arch']).agg(
        asym_score=('asym_score', 'mean'),
        pi_01=('pi_01', 'first'),
        pi_10=('pi_10', 'first'),
    ).reset_index()
    grp2['delta_pi'] = grp2['pi_01'] - grp2['pi_10']

    for arch_name, color in ARCH_COLORS.items():
        sub = grp2[grp2['arch'] == arch_name]
        if sub.empty:
            continue
        ax_a.scatter(sub['delta_pi'], sub['asym_score'],
                     color=color, label=arch_name, s=70, zorder=3, alpha=0.85)
        # linear regression
        if len(sub) >= 2:
            slope, intercept, r_val, p_val, _ = stats.linregress(
                sub['delta_pi'], sub['asym_score']
            )
            x_line = np.linspace(sub['delta_pi'].min(), sub['delta_pi'].max(), 50)
            ax_a.plot(x_line, slope * x_line + intercept, color=color,
                      linewidth=1.5, linestyle='--', alpha=0.7)

    ax_a.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    ax_a.axvline(0, color='grey', linewidth=0.8, linestyle=':')
    ax_a.set_xlabel('π₀₁ − π₁₀  (asymmetry direction)')
    ax_a.set_ylabel('asym_score = (μ_Sp − μ_Se) / (μ_Sp + μ_Se)')
    ax_a.set_title('A: asym_score vs channel asymmetry (per arch)')
    ax_a.legend(title='Arch', fontsize=9)

    # ── Panel B ────────────────────────────────────────────────────────────────
    rf_df = df[df['arch'] == 'RF']
    ch_names = [c['name'] for c in CHANNELS if c['name'] in rf_df['channel'].unique()]

    if rf_df.empty or not ch_names:
        ax_b.text(0.5, 0.5, 'No RF data available', ha='center', va='center',
                  transform=ax_b.transAxes)
        ax_b.set_title('B: f_A by class (RF only)')
    else:
        ch_agg = rf_df.groupby('channel').agg(
            f_A_class0=('f_A_class0', 'mean'),
            f_A_class1=('f_A_class1', 'mean'),
        ).reindex(ch_names)

        x   = np.arange(len(ch_names))
        w   = 0.35
        ax_b.bar(x - w/2, ch_agg['f_A_class0'], w, color='#4878cf', alpha=0.85,
                 label='f_A class 0')
        ax_b.bar(x + w/2, ch_agg['f_A_class1'], w, color='#d65f5f', alpha=0.85,
                 label='f_A class 1')
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(ch_names, rotation=20, ha='right')
        ax_b.set_ylabel('Mean fraction of discordant samples (disc > 0.8)')
        ax_b.set_title('B: f_A (class-conditional) by channel — RF')
        ax_b.legend()

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_FIG1}.{ext}', bbox_inches='tight')
        print(f'{TAG} saved: {OUT_FIG1}.{ext}')
    plt.close(fig)


# ── Figure 2: arch_comparison.pdf ────────────────────────────────────────────

def _make_figure2(df):
    """
    5 panels in 3+2 layout, one per arch.
    X=mu_Se, Y=mu_Sp, each point is a channel (mean over seeds).
    Diagonal mu_Se=mu_Sp in grey dashed.
    """
    archs = [a for a in ARCHITECTURES.keys() if a in df['arch'].unique()]
    n_archs = len(archs)
    if n_archs == 0:
        print(f'{TAG} WARNING: no arches found for figure 2')
        return

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 9), squeeze=False)
    fig.suptitle('Architecture comparison: Se-Sp plane by channel', fontsize=13, fontweight='bold')

    # Flatten active axes and disable the 6th (empty) slot
    ax_flat = [axes[r, c] for r in range(nrows) for c in range(ncols)]
    for extra_ax in ax_flat[n_archs:]:
        extra_ax.set_visible(False)

    # aggregate per (arch, channel)
    agg = df.groupby(['arch', 'channel']).agg(
        mu_Se=('mu_Se', 'mean'),
        mu_Sp=('mu_Sp', 'mean'),
        pi_01=('pi_01', 'first'),
        pi_10=('pi_10', 'first'),
    ).reset_index()

    # common axis range
    all_se = agg['mu_Se']
    all_sp = agg['mu_Sp']
    lo = max(0, min(all_se.min(), all_sp.min()) - 0.05)
    hi = min(1, max(all_se.max(), all_sp.max()) + 0.05)

    for idx, arch_name in enumerate(archs):
        ax  = ax_flat[idx]
        sub = agg[agg['arch'] == arch_name]

        # diagonal
        ax.plot([lo, hi], [lo, hi], color='grey', linestyle='--', linewidth=1, alpha=0.6)

        for _, row_ch in sub.iterrows():
            ch_name = row_ch['channel']
            color   = CHANNEL_COLORS.get(ch_name, '#888888')
            ax.scatter(row_ch['mu_Se'], row_ch['mu_Sp'],
                       color=color, s=90, zorder=3, alpha=0.9)
            ax.annotate(
                ch_name,
                (row_ch['mu_Se'], row_ch['mu_Sp']),
                fontsize=7,
                xytext=(4, 4),
                textcoords='offset points',
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('μ_Se')
        ax.set_ylabel('μ_Sp') if idx == 0 else ax.set_ylabel('')
        ax.set_title(arch_name)

    # legend in the empty 6th slot area
    legend_handles = [
        mpatches.Patch(color=CHANNEL_COLORS.get(c['name'], '#888888'), label=c['name'])
        for c in CHANNELS
    ]
    fig.legend(handles=legend_handles, title='Channel',
               loc='lower right',
               bbox_to_anchor=(0.98, 0.3),
               fontsize=10, title_fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_FIG2}.{ext}', bbox_inches='tight')
        print(f'{TAG} saved: {OUT_FIG2}.{ext}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Asymmetric noise channel analysis for LPI-B2'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Run 1 condition (sym_low, LR, seed=0) only')
    parser.add_argument('--replot', action='store_true',
                        help='Re-generate figures only from existing CSV')
    args = parser.parse_args()

    if args.replot:
        print(f'{TAG} replot mode: loading existing CSV')
        df = pd.read_csv(OUT_CSV)
        _make_figure2(df)
        sys.exit(0)

    run(dry_run=args.dry_run)
