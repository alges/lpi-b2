"""
exp_real_datasets.py — LPI-B2 evaluation on real datasets

Usage:
  python exp_real_datasets.py --dry-run          # 1 condition, ~minutes
  python exp_real_datasets.py                    # reduced mode (default)
  python exp_real_datasets.py --mode full        # full grid, ~6h

Modes:
  dry-run: breast_cancer, flip_y=0.10, LR, 1 seed, B=25
  reduced: datasets=['breast_cancer','phoneme'], flip_y=[0.0,0.10,0.20,0.30],
           archs=['LR','RF','MLP'], seeds=4, B=25
  full:    datasets=['breast_cancer','phoneme','banknote'],
           archs=['LR','RF','SVM','GB','MLP'], seeds=4, B=35
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
from matplotlib.patches import Ellipse
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
try:
    from experiments_3_multi_arch import make_data, cv_metrics, btd_global, btd_sample_posteriors
except ImportError as e:
    print(f'[exp_real_datasets] ERROR: cannot import from experiments_3_multi_arch: {e}')
    sys.exit(1)

try:
    from lpi_b2.estimator import LPIB2Evaluator
except ImportError as e:
    print(f'[exp_real_datasets] ERROR: cannot import LPIB2Evaluator: {e}')
    sys.exit(1)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score, f1_score
from sklearn.datasets import load_breast_cancer

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

# Colormap for flip_y levels
FLIP_COLORS = {
    0.0:  '#4878cf',
    0.10: '#6acc65',
    0.20: '#d65f5f',
    0.30: '#b47cc7',
}

TAG       = '[exp_real_datasets]'
SCRIPT_ID = '04'

# ── Configuration ─────────────────────────────────────────────────────────────
ARCHITECTURES = {
    'LR':  LogisticRegression(max_iter=1000, random_state=0),
    'RF':  RandomForestClassifier(n_estimators=100, random_state=0),
    'SVM': SVC(kernel='rbf', probability=True, random_state=0),
    'GB':  GradientBoostingClassifier(n_estimators=100, random_state=0),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0),
}

FLIP_Y_LEVELS = [0.0, 0.10, 0.20, 0.30]

MODE_CONFIG = {
    'dry-run': {
        'datasets':  ['breast_cancer'],
        'flip_y':    [0.10],
        'archs':     ['LR'],
        'seeds':     1,
        'B':         25,
    },
    'reduced': {
        'datasets':  ['breast_cancer', 'phoneme'],
        'flip_y':    FLIP_Y_LEVELS,
        'archs':     ['LR', 'RF', 'MLP'],
        'seeds':     4,
        'B':         25,
    },
    'full': {
        'datasets':  ['breast_cancer', 'phoneme', 'banknote'],
        'flip_y':    FLIP_Y_LEVELS,
        'archs':     ['LR', 'RF', 'SVM', 'GB', 'MLP'],
        'seeds':     4,
        'B':         35,
    },
}

OUT_CSV       = POSTPROC / 'exp_real_datasets.csv'
OUT_FIG1      = FIGURES  / 'real_capacity_plane'
OUT_FIG2      = FIGURES  / 'real_vs_synthetic'
OUT_FIG3      = FIGURES  / 'recall_inflation_real'
STATUS_FILE   = POSTPROC / 'newexp_04_status.md'
PROGRESS_FILE = POSTPROC / 'NEWEXP_PROGRESS.md'

# ── Dataset loading ───────────────────────────────────────────────────────────

def _load_openml(name, version=1):
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name, version=version, as_frame=False, parser='auto')
    X  = ds.data.astype(float)
    y  = ds.target
    # binarize if needed
    if y.dtype.kind in ('U', 'O', 'S'):
        classes = sorted(set(y))
        y = (y == classes[1]).astype(int)
    elif len(set(y)) > 2:
        median_val = np.median(y)
        y = (y > median_val).astype(int)
    else:
        y = y.astype(int)
        y = (y - y.min()).astype(int)
    return X, y


DATASETS = {
    'breast_cancer': lambda: (lambda d: (d[0], d[1]))(load_breast_cancer(return_X_y=True)),
    'phoneme':       lambda: _load_openml('phoneme', version=1),
    'banknote':      lambda: _load_openml('banknote-authentication', version=1),
}


def load_dataset(name):
    try:
        X, y = DATASETS[name]()
        return X, y
    except Exception as e:
        print(f'{TAG} WARNING: dataset {name} unavailable; skipping — {e}')
        return None, None


def preprocess(X, y):
    """StandardScaler + optional subsample to 2000."""
    X = StandardScaler().fit_transform(X.astype(float))
    y = y.astype(int)
    if len(X) > 2000:
        X, _, y, _ = train_test_split(
            X, y, train_size=2000, stratify=y, random_state=0
        )
    return X, y

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
        return set(zip(df['dataset'], df['arch'], df['flip_y'], df['seed']))
    except Exception:
        return set()


def write_row(row):
    df = pd.DataFrame([row])
    df.to_csv(OUT_CSV, mode='a', header=not OUT_CSV.exists(), index=False)


# ── Main run ──────────────────────────────────────────────────────────────────

def run(mode='reduced'):
    cfg      = MODE_CONFIG[mode]
    datasets = cfg['datasets']
    flip_ys  = cfg['flip_y']
    archs    = cfg['archs']
    n_seeds  = cfg['seeds']
    B        = cfg['B']
    seeds    = list(range(n_seeds))

    done_set = load_done_set()
    total    = len(datasets) * len(flip_ys) * len(archs) * n_seeds
    print(f'{TAG} mode={mode}, planned={total} inferences, already_done={len(done_set)}')

    # Pre-load datasets
    loaded_datasets = {}
    for ds_name in datasets:
        X_raw, y_raw = load_dataset(ds_name)
        if X_raw is None:
            continue
        X_pre, y_pre = preprocess(X_raw, y_raw)
        loaded_datasets[ds_name] = (X_pre, y_pre, X_pre.shape[0], X_pre.shape[1],
                                    float(np.mean(y_pre)))
        print(f'{TAG} loaded {ds_name}: N={X_pre.shape[0]}, d={X_pre.shape[1]}, '
              f'prev={np.mean(y_pre):.3f}')

    for ds_name, (X, y_clean, n_samples, n_features, prevalence) in loaded_datasets.items():
        for flip_y in flip_ys:
            for arch_name in archs:
                arch = ARCHITECTURES[arch_name]
                for seed in seeds:
                    key = (ds_name, arch_name, flip_y, seed)
                    if key in done_set:
                        print(f'{TAG}   skip {key}')
                        continue

                    print(f'{TAG}   ds={ds_name} flip_y={flip_y} arch={arch_name} seed={seed}')

                    # 1. Apply symmetric noise
                    if flip_y > 0:
                        rng    = np.random.default_rng(seed + 30000)
                        mask   = rng.random(len(y_clean)) < flip_y
                        y_noisy = y_clean.copy()
                        y_noisy[mask] = 1 - y_noisy[mask]
                    else:
                        y_noisy = y_clean.copy()

                    # 2. Run BTD
                    try:
                        ev = LPIB2Evaluator(
                            base_estimator=arch,
                            n_bootstrap=B,
                            anchor_rate=0.5,
                            stan_file=str(ROOT / 'lpi_b2' / 'models' / 'lpi_b2_bootstrap.stan'),
                            stan_chains=4,
                            stan_chains_size=1000,
                            stan_chains_warmup=250,
                        )
                        ev.evaluate(X, y_noisy)
                    except Exception as exc:
                        print(f'{TAG}   ERROR in BTD: {exc}')
                        continue

                    # 3. Extract BTD descriptors
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
                    kappa_avg = (kappa_Se + kappa_Sp) / 2

                    p_truth     = np.mean(prob_Ti_draws, axis=0)
                    eps         = 1e-9
                    discordance = np.abs(y_noisy.astype(float) - p_truth)
                    entropy     = -(p_truth * np.log(p_truth + eps) +
                                    (1 - p_truth) * np.log(1 - p_truth + eps))

                    f_A = float(np.mean(discordance > 0.8))
                    f_B = float(np.mean((discordance <= 0.8) & (entropy > 0.8)))
                    f_C = 1.0 - f_A - f_B

                    # 4. CV metrics on noisy labels
                    try:
                        acc_cv, f1_cv, recall_cv = cv_metrics_simple(arch, X, y_noisy)
                    except Exception:
                        acc_cv = f1_cv = recall_cv = float('nan')

                    # 5. recall_inflation: paper definition = recall_cv(y_noisy) - mu_Se
                    recall_inflation = recall_cv - mu_Se

                    # 6. Convergence
                    conv_ok, rhat_max, ess_min = check_convergence(fit)

                    row = dict(
                        dataset=ds_name,
                        n_samples=n_samples,
                        n_features=n_features,
                        prevalence=prevalence,
                        arch=arch_name,
                        flip_y=flip_y,
                        seed=seed,
                        mu_Se=mu_Se,
                        mu_Sp=mu_Sp,
                        mu_avg=mu_avg,
                        kappa_avg=kappa_avg,
                        f_A=f_A,
                        f_B=f_B,
                        f_C=f_C,
                        acc_cv=acc_cv,
                        f1_cv=f1_cv,
                        recall_cv=recall_cv,
                        recall_inflation=recall_inflation,
                        convergence_ok=conv_ok,
                    )
                    write_row(row)
                    print(f'{TAG}   done: mu_Se={mu_Se:.3f}, mu_Sp={mu_Sp:.3f}, '
                          f'mu_avg={mu_avg:.3f}, conv={conv_ok}')

    # ── Figures ────────────────────────────────────────────────────────────────
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        if not df.empty:
            _make_figure1(df, archs, mode)
            _make_figure2(df, archs, mode)
            _make_figure3(df, archs, mode)
        else:
            print(f'{TAG} WARNING: CSV is empty, skipping figures')
    else:
        print(f'{TAG} WARNING: no CSV found, skipping figures')

    # ── Status ─────────────────────────────────────────────────────────────────
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        conv_rate = df['convergence_ok'].mean() if not df.empty else float('nan')
        conv_str  = f'{conv_rate:.0%} converged'
        key_line  = (f'rows={len(df)}, mu_avg_mean={df["mu_avg"].mean():.3f}, '
                     f'recall_inflation_mean={df["recall_inflation"].mean():.3f}')
    else:
        conv_str = 'no data'
        key_line = 'no data'

    write_status(
        SCRIPT_ID,
        title='Real dataset evaluation',
        mode=mode,
        status='DONE',
        conv_str=conv_str,
        key_figures=[
            str(OUT_FIG1) + '.{pdf,png}',
            str(OUT_FIG2) + '.{pdf,png}',
            str(OUT_FIG3) + '.{pdf,png}',
        ],
        outputs=[str(OUT_CSV)],
        notes=[f'datasets={datasets}', f'archs={archs}', f'B={B}'],
    )
    append_progress(
        SCRIPT_ID,
        mode=mode,
        status='DONE',
        conv_str=conv_str,
        key_line=key_line,
        outputs=[OUT_CSV.name],
        notes=[f'datasets={len(datasets)}, archs={len(archs)}, B={B}'],
    )
    print(f'{TAG} DONE. CSV={OUT_CSV}')


# ── Confidence ellipse helper ─────────────────────────────────────────────────

def _confidence_ellipse(x, y, ax, n_std=1.645, **kwargs):
    """Draw a covariance ellipse for a 90% confidence region."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.abs(eigenvalues))
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height,
        angle=angle,
        **kwargs,
    )
    ax.add_patch(ellipse)


# ── Figure 1: real_capacity_plane.pdf ────────────────────────────────────────

def _make_figure1(df, archs, mode):
    """
    3 (or 2) panels, one per dataset.
    mu_Se vs mu_Sp for archs at flip_y=0.20, ellipses 90% over seeds.
    """
    datasets_in_data = df['dataset'].unique()
    n_ds = len(datasets_in_data)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), squeeze=False)
    fig.suptitle('Real data: Se–Sp capacity plane (flip_y=0.20)', fontsize=13, fontweight='bold')

    sub_flip = df[df['flip_y'] == 0.20]

    # common axis limits
    all_se = df['mu_Se']
    all_sp = df['mu_Sp']
    lo = max(0, min(all_se.min(), all_sp.min()) - 0.05)
    hi = min(1, max(all_se.max(), all_sp.max()) + 0.05)

    for col, ds_name in enumerate(datasets_in_data):
        ax     = axes[0, col]
        ds_sub = sub_flip[sub_flip['dataset'] == ds_name]

        # diagonal
        ax.plot([lo, hi], [lo, hi], color='grey', linestyle='--', linewidth=1, alpha=0.6)

        arch_list = [a for a in archs if a in ds_sub['arch'].unique()]
        for arch_name in arch_list:
            color = ARCH_COLORS.get(arch_name, '#888888')
            a_sub = ds_sub[ds_sub['arch'] == arch_name]
            if a_sub.empty:
                continue
            ax.scatter(a_sub['mu_Se'], a_sub['mu_Sp'],
                       color=color, s=60, alpha=0.7, zorder=3, label=arch_name)
            # 90% ellipse if ≥3 seeds
            if len(a_sub) >= 3:
                _confidence_ellipse(
                    a_sub['mu_Se'].values, a_sub['mu_Sp'].values, ax,
                    n_std=1.645, edgecolor=color, facecolor=color,
                    alpha=0.12, linewidth=1.5,
                )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('μ_Se')
        ax.set_ylabel('μ_Sp') if col == 0 else ax.set_ylabel('')
        ax.set_title(ds_name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_FIG1}.{ext}', bbox_inches='tight')
        print(f'{TAG} saved: {OUT_FIG1}.{ext}')
    plt.close(fig)


# ── Figure 2: real_vs_synthetic.pdf ──────────────────────────────────────────

def _make_figure2(df, archs, mode):
    """
    Panels (one per arch in full, 3 in reduced).
    X=flip_y, Y=mu_avg.
    Real dataset mean with shading + synthetic sep=1.5 from table_multiarch_def_yc.csv.
    """
    # load synthetic reference
    synth_path = TABLES / 'table_multiarch_def_yc.csv'
    synth_df   = None
    if synth_path.exists():
        try:
            synth_df = pd.read_csv(synth_path)
            print(f'{TAG} loaded synthetic reference: {synth_path}')
        except Exception as e:
            print(f'{TAG} WARNING: could not load {synth_path}: {e}')
    else:
        print(f'{TAG} WARNING: synthetic CSV not found at {synth_path}, skipping synthetic curves')

    archs_in_data = [a for a in archs if a in df['arch'].unique()]
    n_archs = len(archs_in_data)
    if n_archs == 0:
        return

    fig, axes = plt.subplots(1, n_archs, figsize=(4.5 * n_archs, 5), squeeze=False)
    fig.suptitle('Real vs synthetic: μ_avg vs flip_y', fontsize=13, fontweight='bold')

    flip_ys_sorted = sorted(df['flip_y'].unique())
    datasets_in_data = df['dataset'].unique()

    for col, arch_name in enumerate(archs_in_data):
        ax = axes[0, col]
        arch_df = df[df['arch'] == arch_name]

        # Real data: mean and std over seeds × datasets
        real_means, real_stds = [], []
        for fy in flip_ys_sorted:
            vals = arch_df[arch_df['flip_y'] == fy]['mu_avg']
            real_means.append(vals.mean())
            real_stds.append(vals.std())

        real_means = np.array(real_means)
        real_stds  = np.array(real_stds)
        ax.plot(flip_ys_sorted, real_means, color='#1f77b4', linewidth=2, label='Real data')
        ax.fill_between(
            flip_ys_sorted,
            real_means - real_stds,
            real_means + real_stds,
            alpha=0.2, color='#1f77b4',
        )

        # Synthetic: from table_multiarch_def_yc.csv
        if synth_df is not None and arch_name in synth_df.get('arch', pd.Series()).unique():
            synth_arch = synth_df[synth_df['arch'] == arch_name]
            if 'flip_y' in synth_arch.columns and 'mu_avg' in synth_arch.columns:
                synth_grp = synth_arch.groupby('flip_y')['mu_avg'].mean()
                ax.plot(synth_grp.index, synth_grp.values,
                        color='#d62728', linewidth=1.5, linestyle='--',
                        label='Synthetic (sep=1.5)')

        ax.set_xlabel('flip_y')
        ax.set_ylabel('μ_avg') if col == 0 else ax.set_ylabel('')
        ax.set_title(arch_name)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_FIG2}.{ext}', bbox_inches='tight')
        print(f'{TAG} saved: {OUT_FIG2}.{ext}')
    plt.close(fig)


# ── Figure 3: recall_inflation_real.pdf ──────────────────────────────────────

def _make_figure3(df, archs, mode):
    """
    For each (dataset, arch): recall_inflation vs flip_y.
    One line per (dataset, arch) combination.
    """
    datasets_in_data = df['dataset'].unique()
    archs_in_data    = [a for a in archs if a in df['arch'].unique()]
    n_ds = len(datasets_in_data)
    if n_ds == 0 or not archs_in_data:
        return

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), squeeze=False)
    fig.suptitle('Recall inflation = recall(noisy labels) − recall(clean labels)',
                 fontsize=12, fontweight='bold')

    flip_ys_sorted = sorted(df['flip_y'].unique())

    for col, ds_name in enumerate(datasets_in_data):
        ax = axes[0, col]
        ds_df = df[df['dataset'] == ds_name]

        for arch_name in archs_in_data:
            color   = ARCH_COLORS.get(arch_name, '#888888')
            a_df    = ds_df[ds_df['arch'] == arch_name]
            ri_vals = []
            for fy in flip_ys_sorted:
                vals = a_df[a_df['flip_y'] == fy]['recall_inflation']
                ri_vals.append(vals.mean() if not vals.empty else float('nan'))

            ri_vals = np.array(ri_vals, dtype=float)
            valid   = ~np.isnan(ri_vals)
            if valid.sum() >= 1:
                ax.plot(
                    [flip_ys_sorted[i] for i in range(len(flip_ys_sorted)) if valid[i]],
                    ri_vals[valid],
                    color=color, linewidth=2, marker='o', markersize=5, label=arch_name,
                )

        ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
        ax.set_xlabel('flip_y')
        ax.set_ylabel('Recall inflation') if col == 0 else ax.set_ylabel('')
        ax.set_title(ds_name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_FIG3}.{ext}', bbox_inches='tight')
        print(f'{TAG} saved: {OUT_FIG3}.{ext}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LPI-B2 evaluation on real datasets'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Run 1 condition (~minutes)')
    parser.add_argument('--mode', choices=['reduced', 'full'], default='reduced',
                        help='Run mode: reduced (default) or full (~6h)')
    args = parser.parse_args()

    if args.dry_run:
        run(mode='dry-run')
    else:
        run(mode=args.mode)
