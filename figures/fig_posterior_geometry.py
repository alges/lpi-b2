"""
fig_posterior_geometry.py
Generates fig_posterior_geometry.pdf — per-sample posterior geometry
(p̂_i vs entropy coloured by discordance) across architectures and
noise levels.

Paper: Supplementary Figure S3 / Section "Posterior geometry and noise honesty".
Inputs (quick mode): data/sample_posteriors.csv
Output: outputs/fig_posterior_geometry.pdf

Run:
    python figures/fig_posterior_geometry.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT     = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

POSTERIORS_CSV = DATA / "sample_posteriors.csv"
if not POSTERIORS_CSV.exists():
    raise FileNotFoundError(POSTERIORS_CSV)

sample_df  = pd.read_csv(POSTERIORS_CSV)
arch_order = ["LR", "RF", "SVM", "GB", "MLP"]
flip_levels = sorted(sample_df["flip_y"].unique())

PALETTE = {
    "LR":  "#7F77DD",
    "RF":  "#1D9E75",
    "SVM": "#D85A30",
    "GB":  "#BA7517",
    "MLP": "#378ADD",
}

rng       = np.random.default_rng(0)
subsample = 500

n_arch = len(arch_order)
n_flip = len(flip_levels)

fig, axes = plt.subplots(
    n_arch, n_flip,
    figsize=(3.5 * n_flip, 2.1 * n_arch),
    sharex=True, sharey=True,
)

vmin = sample_df["discordance"].quantile(0.02)
vmax = sample_df["discordance"].quantile(0.98)
cmap = cm.get_cmap("RdYlGn_r")

for r, arch in enumerate(arch_order):
    for c, flip in enumerate(flip_levels):
        ax  = axes[r, c]
        sub = sample_df[
            (sample_df["arch"]   == arch) &
            (sample_df["flip_y"] == flip)
        ]
        n_draw = min(subsample, len(sub))
        idx    = rng.choice(len(sub), size=n_draw, replace=False)
        pts    = sub.iloc[idx]

        sc = ax.scatter(
            pts["p_truth"], pts["entropy"],
            c=pts["discordance"],
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=16, alpha=0.65, linewidths=0,
        )

        ax.axvline(0.5, color="#444", linestyle=":",  linewidth=0.8)
        ax.axhline(1.0, color="#444", linestyle="--", linewidth=0.8)

        if r == 0:
            ax.set_title(f"flip = {flip:.2f}", fontsize=9)
        if c == 0:
            ax.set_ylabel(arch, fontsize=9, rotation=0, labelpad=38, va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=7)

fig.subplots_adjust(
    right=0.88, hspace=0.175, wspace=0.125, top=0.984, bottom=0.055,
)

cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r"discordance $\delta_i$", fontsize=9)

fig.text(0.45, -0.008, r"$\hat{p}_i$", ha="center", fontsize=10)
fig.text(-0.01, 0.5, "Entropy (bits)",
         va="center", ha="center", rotation=90, fontsize=10)

fig.suptitle(
    r"$\hat{p}_i$ vs Entropy coloured by Discordance"
    "\n(arch × noise level; random subsample per cell)",
    y=1.05, fontsize=11,
)

out = OUT / "fig_posterior_geometry.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
