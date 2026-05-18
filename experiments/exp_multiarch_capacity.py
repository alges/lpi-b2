import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.patches import Ellipse
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import (make_scorer, recall_score,
                              precision_score, f1_score, roc_auc_score)

from lpi_b2.estimator import (
    LPIB2Evaluator,
    get_label_noise_audit,
    optimize_noise_threshold,
    get_entropy_audit,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── output dirs ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

(ROOT / "data").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(parents=True, exist_ok=True)

# ── global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PALETTE = {
    "BTD_Se":   "#7F77DD",
    "BTD_Sp":   "#1D9E75",
    "BTD_avg":  "#534AB7",
    "Accuracy": "#D85A30",
    "F1":       "#BA7517",
    "Recall":   "#378ADD",
    "Precision":"#888780",
    "LR":       "#7F77DD",
    "RF":       "#1D9E75",
    "SVM":      "#D85A30",
    "GB":       "#BA7517",
    "MLP":      "#378ADD",
    "TypeA":    "#E24B4A",
    "TypeB":    "#EF9F27",
    "TypeC":    "#B4B2A9",
}

# ── global B ──────────────────────────────────────────────────────────────────
N_BOOTSTRAP = 35

# ── experiment suffix — change here to rename all output files at once ────────
SUFFIX = "_yc"   # _yc = includes y_clean column + flip_y=0.0 level

# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def make_data(n=1000, sep=1.5, flip=0.0, seed=42):
    # Generate clean labels first (flip_y=0), then apply noise manually.
    # This preserves y_clean for NLL/Brier/ECE computation in post-processing.
    X, y_clean = make_classification(
        n_samples=n, n_features=20, n_informative=10, n_redundant=4,
        n_repeated=0, class_sep=sep, flip_y=0.0, random_state=seed,
    )
    if flip > 0.0:
        rng    = np.random.default_rng(seed + 10_000)  # independent RNG for flip
        mask   = rng.random(n) < flip
        y_noisy = y_clean.copy()
        y_noisy[mask] = 1 - y_noisy[mask]
    else:
        y_noisy = y_clean.copy()
    return X, y_noisy, y_clean


def btd_fit(X, y, anchor_rate=0.5, chains=4, chain_size=1000):
    ev = LPIB2Evaluator(
        base_estimator=RandomForestClassifier(n_estimators=100, random_state=0),
        n_bootstrap=N_BOOTSTRAP,
        anchor_rate=anchor_rate,
    )
    ev.evaluate(X, y)
    return ev


def btd_fit_arch(arch, X, y, anchor_rate=0.5):
    ev = LPIB2Evaluator(
        base_estimator=arch,
        n_bootstrap=N_BOOTSTRAP,
        anchor_rate=anchor_rate,
    )
    ev.evaluate(X, y)
    return ev


def cv_metrics(arch, X, y, cv=5):
    scoring = {
        "accuracy":  "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score),
        "f1":        make_scorer(f1_score),
    }
    res = cross_validate(arch, X, y, cv=cv, scoring=scoring)
    return {k: (np.mean(res[f"test_{k}"]), np.std(res[f"test_{k}"])) for k in scoring}


def lpi_b2_global(ev):
    """Return posterior mean ± std for key BTD parameters, plus µ_avg."""
    fit = ev.mcmc_fit_
    out = {}
    for v in ["mu_Se", "mu_Sp", "kappa_Se", "kappa_Sp", "kappa_obs"]:
        s = fit.stan_variable(v)
        out[v] = (float(np.mean(s)), float(np.std(s)))
    avg = (fit.stan_variable("mu_Se") + fit.stan_variable("mu_Sp")) / 2
    out["mu_avg"] = (float(np.mean(avg)), float(np.std(avg)))
    avg_k = (fit.stan_variable("kappa_Se") + fit.stan_variable("kappa_Sp")) / 2
    out["kappa_avg"] = (float(np.mean(avg_k)), float(np.std(avg_k)))
    return out


def lpi_b2_sample_posteriors(ev, y, y_clean, flip, arch_name, seed):
    """
    Extract per-sample posterior quantities from a fitted LPIB2Evaluator.

    Returns a DataFrame with one row per sample and columns:
        flip_y, arch, seed, sample_idx,
        p_truth, ambiguity, discordance, entropy, y_noisy, y_clean
    y_clean contains the true labels before noise injection, enabling
    NLL, Brier score, and ECE computation in post-processing.
    """
    # posterior mean P(Ti = 1 | data) for every sample — shape (N,)
    p_truth = np.mean(ev.mcmc_fit_.stan_variable("prob_Ti_pos"), axis=0)

    eps = 1e-9
    ambiguity   = 1.0 - 2.0 * np.abs(p_truth - 0.5)
    discordance = np.abs(y.astype(float) - p_truth)
    entropy     = -(
        p_truth       * np.log2(p_truth       + eps) +
        (1 - p_truth) * np.log2(1 - p_truth   + eps)
    )

    N = len(p_truth)
    return pd.DataFrame({
        "flip_y":      flip,
        "arch":        arch_name,
        "seed":        seed,
        "sample_idx":  np.arange(N),
        "p_truth":     p_truth,
        "ambiguity":   ambiguity,
        "discordance": discordance,
        "entropy":     entropy,
        "y_noisy":     y.astype(int),
        "y_clean":     y_clean.astype(int),
    })


def btd_kappa_obs_draws(ev, flip, arch_name, seed):
    """
    Extract the full posterior draw vector for kappa_obs.

    Returns a DataFrame with one row per MCMC draw and columns:
        flip_y, arch, seed, kappa_obs
    """
    draws = ev.mcmc_fit_.stan_variable("kappa_obs")   # shape (total_draws,)
    return pd.DataFrame({
        "flip_y":    flip,
        "arch":      arch_name,
        "seed":      seed,
        "kappa_obs": draws,
    })


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING — four new figures
# ═════════════════════════════════════════════════════════════════════════════

# ── Figure A ─────────────────────────────────────────────────────────────────
def plot_kappa_obs_distributions(kappa_df, flip_levels, arch_order, out_path):
    """
    Ridge / overlapping KDE plot of kappa_obs posterior draws.

    Layout: 1 row × len(flip_levels) columns.
    Each panel: one KDE per architecture, coloured by PALETTE, with
    a dashed vertical line at each distribution's median.
    All seeds are pooled within each (flip_y, arch) cell, so the full
    posterior variability (across draws AND seeds) is visible.
    """
    n_flip = len(flip_levels)
    fig, axes = plt.subplots(1, n_flip, figsize=(5 * n_flip, 4.0), sharey=False)
    if n_flip == 1:
        axes = [axes]

    for ax, flip in zip(axes, flip_levels):
        sub = kappa_df[kappa_df["flip_y"] == flip]
        for name in arch_order:
            vals = sub[sub["arch"] == name]["kappa_obs"].values
            if len(vals) == 0:
                continue
            c = PALETTE[name]
            # KDE via seaborn
            sns.kdeplot(vals, ax=ax, color=c, fill=True, alpha=0.20,
                        linewidth=1.4, label=name, bw_adjust=0.6)
            ax.axvline(np.median(vals), color=c, linestyle="--",
                       linewidth=0.9, alpha=0.75)

        ax.set_title(f"flip_y = {flip:.2f}", fontsize=10)
        ax.set_xlabel(r"$\kappa_{\mathrm{obs}}$ (posterior draws)")
        ax.set_ylabel("Density" if ax is axes[0] else "")

    handles = [mpatches.Patch(color=PALETTE[n], label=n) for n in arch_order]
    axes[-1].legend(handles=handles, fontsize=8, loc="upper right",
                    framealpha=0.6)
    fig.suptitle(
        r"Posterior distribution of $\kappa_{\mathrm{obs}}$ by architecture and noise level",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out_path.name}")


# ── Figure B ─────────────────────────────────────────────────────────────────
def plot_p_truth_grid(sample_df, flip_levels, arch_order, out_path):
    """
    Grid of histograms (arch × flip_y).

    Each cell shows one semi-transparent histogram per seed so the
    inter-seed variance is visible. A vertical reference line marks p = 0.5.
    Colour encodes seed (sequential palette within each cell).
    """
    n_arch = len(arch_order)
    n_flip = len(flip_levels)
    seed_colors = sns.color_palette("Blues_d", n_colors=sample_df["seed"].nunique())

    fig, axes = plt.subplots(
        n_arch, n_flip,
        figsize=(3.8 * n_flip, 2.8 * n_arch),
        sharex=True, sharey=False,
    )

    for r, arch in enumerate(arch_order):
        for c, flip in enumerate(flip_levels):
            ax = axes[r, c]
            sub = sample_df[(sample_df["arch"] == arch) &
                            (sample_df["flip_y"] == flip)]
            for s, sc in zip(sorted(sub["seed"].unique()), seed_colors):
                vals = sub[sub["seed"] == s]["p_truth"].values
                ax.hist(vals, bins=30, range=(0, 1),
                        color=sc, alpha=0.45, density=True, linewidth=0)

            ax.axvline(0.5, color="#333333", linestyle=":", linewidth=0.9)
            if r == 0:
                ax.set_title(f"flip = {flip:.2f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(arch, fontsize=9, rotation=0,
                               labelpad=38, va="center")
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=7)

    # common x-label
    fig.text(0.5, -0.01, r"$p_{\mathrm{truth}}$ = P(Ti = 1 | data)",
             ha="center", fontsize=10)
    fig.suptitle(
        r"Distribution of $p_{\mathrm{truth}}$ per architecture × noise level"
        "\n(colours = seeds)",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out_path.name}")


# ── Figure C ─────────────────────────────────────────────────────────────────
def plot_violin_metrics(sample_df, flip_levels, arch_order, out_path):
    """
    Three-panel vertical layout: one violin plot per metric
    (ambiguity, discordance, entropy).

    X axis: flip_y level (treated as categorical).
    Violins coloured by architecture and split horizontally inside each
    flip_y tick so all architectures are visible side by side.
    Median markers are drawn explicitly.
    """
    metrics = [
        ("ambiguity",   "Ambiguity  (1 − 2|p − 0.5|)"),
        ("discordance", "Discordance  |y − p_truth|"),
        ("entropy",     "Shannon Entropy  (bits)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    flip_labels = [f"{f:.2f}" for f in flip_levels]

    for ax, (metric, ylabel) in zip(axes, metrics):
        # Build a tidy sub-frame pooling all seeds
        frames = []
        for flip in flip_levels:
            for arch in arch_order:
                sub = sample_df[
                    (sample_df["flip_y"] == flip) &
                    (sample_df["arch"]   == arch)
                ]
                tmp = sub[["seed", metric]].copy()
                tmp["flip_label"] = f"{flip:.2f}"
                tmp["arch"]       = arch
                frames.append(tmp)
        plot_df = pd.concat(frames, ignore_index=True)

        sns.violinplot(
            data=plot_df,
            x="flip_label", y=metric, hue="arch",
            hue_order=arch_order,
            palette={n: PALETTE[n] for n in arch_order},
            order=flip_labels,
            inner="quart",        # shows quartile lines inside violin
            density_norm="width",
            linewidth=0.7,
            alpha=0.75,
            ax=ax,
            legend=(ax is axes[0]),   # legend only on the first panel
        )

        ax.set_xlabel("flip_y" if ax is axes[-1] else "")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(title="arch", fontsize=7, title_fontsize=8,
                      loc="upper right", framealpha=0.6,
                      ncol=len(arch_order))

    fig.suptitle(
        "Sample-level posterior metrics by architecture and noise level\n"
        "(all seeds pooled within each cell)",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out_path.name}")


# ── Figure D ─────────────────────────────────────────────────────────────────
def plot_scatter_ptruth_entropy(sample_df, flip_levels, arch_order,
                                out_path, subsample=500, rng_seed=0):
    """
    Scatter p_truth vs entropy, coloured by discordance.

    Grid: flip_y columns × arch rows.
    Each cell plots `subsample` randomly drawn points from all seeds combined.
    Colour = discordance (RdYlGn_r colormap).
    Reference lines: p_truth = 0.5 (vertical) and entropy = 1.0 (horizontal).
    """
    n_arch = len(arch_order)
    n_flip = len(flip_levels)
    rng    = np.random.default_rng(rng_seed)

    fig, axes = plt.subplots(
        n_arch, n_flip,
        figsize=(3.5 * n_flip, 2.8 * n_arch),
        sharex=True, sharey=True,
    )

    # shared colour normalisation across the whole dataset
    vmin = sample_df["discordance"].quantile(0.02)
    vmax = sample_df["discordance"].quantile(0.98)
    cmap = cm.get_cmap("RdYlGn_r")

    for r, arch in enumerate(arch_order):
        for c, flip in enumerate(flip_levels):
            ax = axes[r, c]
            sub = sample_df[
                (sample_df["arch"]   == arch) &
                (sample_df["flip_y"] == flip)
            ]
            # random subsample across all seeds
            n_draw = min(subsample, len(sub))
            idx    = rng.choice(len(sub), size=n_draw, replace=False)
            pts    = sub.iloc[idx]

            sc = ax.scatter(
                pts["p_truth"], pts["entropy"],
                c=pts["discordance"],
                cmap=cmap, vmin=vmin, vmax=vmax,
                s=8, alpha=0.65, linewidths=0,
            )

            # reference lines
            ax.axvline(0.5,  color="#444", linestyle=":",  linewidth=0.8)
            ax.axhline(1.0,  color="#444", linestyle="--", linewidth=0.8)

            if r == 0:
                ax.set_title(f"flip = {flip:.2f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(arch, fontsize=9, rotation=0,
                               labelpad=38, va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=7)

    # shared colour bar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("discordance  |y − p_truth|", fontsize=9)

    # shared axis labels
    fig.text(0.45, -0.01, r"$p_{\mathrm{truth}}$", ha="center", fontsize=10)
    fig.text(-0.01, 0.5,  "Entropy (bits)",
             va="center", ha="center", rotation=90, fontsize=10)

    fig.suptitle(
        r"$p_{\mathrm{truth}}$ vs Entropy coloured by Discordance"
        "\n(arch × noise level; random subsample per cell)",
        y=1.02, fontsize=11,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# ARCHITECTURES
# ═════════════════════════════════════════════════════════════════════════════

ARCHITECTURES = {
    "LR":  LogisticRegression(max_iter=1000, random_state=0),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=0),
    "SVM": SVC(kernel="rbf", probability=True, random_state=0),
    "GB":  GradientBoostingClassifier(n_estimators=100, random_state=0),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                         random_state=0),
}

ARCHITECTURES_WEAK = {
    "LR":  LogisticRegression(
               C=0.001,
               max_iter=200,
               solver="liblinear",
               random_state=0,
           ),
    "RF":  RandomForestClassifier(
               n_estimators=8,
               max_depth=2,
               max_features=2,
               min_samples_leaf=50,
               random_state=0,
           ),
    "SVM": SVC(
               kernel="linear",
               C=0.01,
               probability=True,
               random_state=0,
           ),
    "GB":  GradientBoostingClassifier(
               n_estimators=5,
               max_depth=1,
               learning_rate=0.5,
               subsample=0.4,
               random_state=0,
           ),
    "MLP": MLPClassifier(
               hidden_layer_sizes=(16, 8),
               max_iter=50,
               learning_rate_init=0.5,
               momentum=0.0,
               random_state=0,
           ),
}

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT — Multi-architecture benchmarking + noise parameters
# ═════════════════════════════════════════════════════════════════════════════

def exp3_multiarch(flip_levels=(0.0, 0.10, 0.20), sep=1.0, n_seeds=5):
    print(f"\n=== Experiment 3: Multi-architecture (B={N_BOOTSTRAP}) ===")

    arch_order = list(ARCHITECTURES.keys())

    # accumulation lists — written to CSV once after all loops complete
    records        = []   # one row per (flip, arch, seed) — global params
    sample_records = []   # one row per (flip, arch, seed, sample) — per-sample posteriors
    kappa_records  = []   # one row per (flip, arch, seed, mcmc_draw) — kappa_obs draws

    for flip in flip_levels:
        for name, arch in ARCHITECTURES.items():
            print(f"  {name}, flip_y={flip:.2f}")
            for seed in range(n_seeds):
                X, y, y_clean = make_data(sep=sep, flip=flip, seed=seed)
                ev     = btd_fit_arch(arch, X, y)
                btd    = lpi_b2_global(ev)
                cv     = cv_metrics(arch, X, y)

                # ── global record (unchanged) ─────────────────────────────
                records.append({
                    "arch": name, "flip_y": flip, "seed": seed,
                    "mu_Se":        btd["mu_Se"][0],
                    "mu_Se_std":    btd["mu_Se"][1],
                    "mu_Sp":        btd["mu_Sp"][0],
                    "mu_Sp_std":    btd["mu_Sp"][1],
                    "mu_avg":       btd["mu_avg"][0],
                    "kappa_Se":     btd["kappa_Se"][0],
                    "kappa_Se_std": btd["kappa_Se"][1],
                    "kappa_Sp":     btd["kappa_Sp"][0],
                    "kappa_Sp_std": btd["kappa_Sp"][1],
                    "kappa_avg":    btd["kappa_avg"][0],
                    "kappa_avg_std":btd["kappa_avg"][1],
                    "kappa_obs":    btd["kappa_obs"][0],
                    "kappa_obs_std":btd["kappa_obs"][1],
                    "cv_recall":    cv["recall"][0],
                    "cv_f1":        cv["f1"][0],
                    "cv_acc":       cv["accuracy"][0],
                })

                # ── per-sample posteriors ─────────────────────────────────
                sample_records.append(
                    lpi_b2_sample_posteriors(ev, y, y_clean, flip, name, seed)
                )

                # ── kappa_obs MCMC draws ──────────────────────────────────
                kappa_records.append(
                    btd_kappa_obs_draws(ev, flip, name, seed)
                )

    # ── build DataFrames ──────────────────────────────────────────────────────
    df        = pd.DataFrame(records)
    sample_df = pd.concat(sample_records, ignore_index=True)
    kappa_df  = pd.concat(kappa_records,  ignore_index=True)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    df.to_csv(ROOT / "data" / f"table_multiarch_def{SUFFIX}.csv", index=False)
    print(f"  → saved table_multiarch_def{SUFFIX}.csv")

    sample_df.to_csv(ROOT / "data" / f"sample_posteriors{SUFFIX}.csv", index=False)
    print(f"  → saved sample_posteriors{SUFFIX}.csv  ({len(sample_df):,} rows)")

    kappa_df.to_csv(ROOT / "data" / f"kappa_obs_draws{SUFFIX}.csv", index=False)
    print(f"  → saved kappa_obs_draws{SUFFIX}.csv  ({len(kappa_df):,} rows)")

    # ── original Figure 1: µ–κ plane (unchanged) ─────────────────────────────
    fig, axes = plt.subplots(
        1, len(flip_levels),
        figsize=(5 * len(flip_levels), 4.5), sharey=True,
    )
    for ax, flip in zip(axes, flip_levels):
        sub = df[df["flip_y"] == flip]
        for name in arch_order:
            g  = sub[sub["arch"] == name]
            mx = g["mu_avg"].mean()
            my = g["kappa_avg"].mean()
            sx = g["mu_avg"].std()
            sy = g["kappa_avg"].std()
            c  = PALETTE[name]
            ell = Ellipse(
                (mx, my),
                width=2 * 1.645 * sx, height=2 * 1.645 * sy,
                angle=0, color=c, alpha=0.2, lw=0,
            )
            ax.add_patch(ell)
            ax.scatter(mx, my, color=c, s=60, zorder=5)
            ax.annotate(name, (mx + 0.002, my + 0.3), color=c, fontsize=8)
        ax.set_xlabel("µ_avg = (µ_Se + µ_Sp)/2")
        ax.set_title(f"flip_y = {flip:.2f}")
        ax.set_xlim(0.4, 1.0)
    axes[0].set_ylabel("κ_avg = (κ_Se + κ_Sp)/2")
    handles = [mpatches.Patch(color=PALETTE[n], label=n) for n in arch_order]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle(
        "Balanced Accuracy – Stability plane by architecture and noise level",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / f"fig_Bal_Acc_Stability_plane{SUFFIX}.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved fig_Bal_Acc_Stability_plane{SUFFIX}.pdf")

    # ── Figure A: kappa_obs posterior distributions ───────────────────────────
    plot_kappa_obs_distributions(
        kappa_df, flip_levels, arch_order,
        ROOT / "outputs" / f"fig_kappa_obs_distributions{SUFFIX}.pdf",
    )

    # ── Figure B: p_truth histogram grid ─────────────────────────────────────
    plot_p_truth_grid(
        sample_df, flip_levels, arch_order,
        ROOT / "outputs" / f"fig_p_truth_by_arch_flip{SUFFIX}.pdf",
    )

    # ── Figure C: violin plots for ambiguity / discordance / entropy ──────────
    plot_violin_metrics(
        sample_df, flip_levels, arch_order,
        ROOT / "outputs" / f"fig_sample_metrics_violin{SUFFIX}.pdf",
    )

    # ── Figure D: p_truth vs entropy scatter coloured by discordance ──────────
    plot_scatter_ptruth_entropy(
        sample_df, flip_levels, arch_order,
        ROOT / "outputs" / f"fig_scatter_ptruth_entropy{SUFFIX}.pdf",
        subsample=500,
    )

    return df, sample_df, kappa_df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    FLIP_LEVELS = [0.0, 0.10, 0.30, 0.50]

    df3, sample_df3, kappa_df3 = exp3_multiarch(
        flip_levels=FLIP_LEVELS, sep=1.0, n_seeds=5
    )

    print("\nAll results saved to ./latex/results/")