"""
fig_noise_capacity_trajectories.py
Generates fig_noise_capacity_trajectories.pdf — capacity (µ̂_avg) and
noise absorption fraction (f_C) trajectories across architectures and
noise levels.

Paper: Figure 8 / Section "Posterior geometry and noise honesty".
Inputs (quick mode):
    data/table_multiarch_def_yc.csv
    data/sample_posteriors_yc.csv
Output: outputs/fig_noise_capacity_trajectories.pdf

Run:
    python figures/fig_noise_capacity_trajectories.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT     = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                     "legend.fontsize": 10, "figure.dpi": 110})

ARCH_COLORS = {"LR": "#9467bd", "RF": "#2ca02c", "SVM": "#1f77b4",
               "GB": "#ff7f0e", "MLP": "#d62728"}
ARCHS = list(ARCH_COLORS)

tbl_yc  = pd.read_csv(DATA / "table_multiarch_def_yc.csv")
post_yc = pd.read_csv(DATA / "sample_posteriors_yc.csv",
                      usecols=["arch", "seed", "flip_y", "discordance", "entropy"])

post_yc["typeC"] = (post_yc["discordance"] <= 0.8) & (post_yc["entropy"] <= 0.8)
frac_C_df = post_yc.groupby(["arch", "flip_y"])["typeC"].mean().reset_index()
frac_C_df.columns = ["arch", "flip_y", "frac_C"]

arch_agg = tbl_yc.groupby(["arch", "flip_y"]).agg(
    mu_avg_mean=("mu_avg", "mean"),
    mu_avg_std=("mu_avg", "std"),
).reset_index().merge(frac_C_df, on=["arch", "flip_y"])

fig8, axes8 = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

for arch in ARCHS:
    s = arch_agg[arch_agg["arch"] == arch].sort_values("flip_y")
    if s.empty:
        continue
    color = ARCH_COLORS[arch]
    phis  = s["flip_y"].values

    axes8[0].plot(phis, s["mu_avg_mean"].values, color=color, lw=1.8,
                  marker="o", ms=7, label=arch)
    axes8[0].fill_between(phis,
                          s["mu_avg_mean"].values - s["mu_avg_std"].values,
                          s["mu_avg_mean"].values + s["mu_avg_std"].values,
                          color=color, alpha=0.12)

    axes8[1].plot(phis, s["frac_C"].values, color=color, lw=1.8,
                  marker="o", ms=7, label=arch)

axes8[0].set_ylabel("µ̂_avg (capacity)")
axes8[0].set_title("µ̂_avg trajectory under label noise")
axes8[0].legend(ncol=3, fontsize=9)
axes8[0].set_ylim(0.5, 1.0)
axes8[0].tick_params(labelbottom=False)

fc_min   = arch_agg["frac_C"].min()
ylim_bot = round(max(0.0, fc_min - 0.05) / 0.05) * 0.05

axes8[1].axhline(0.95, color="grey", ls="-", lw=1.2, alpha=0.6,
                 label="f_C = 0.95 (near-total absorption)")
axes8[1].set_xlabel("Label noise φ")
axes8[1].set_ylabel("f_C (fraction Type C)")
axes8[1].set_title("f_C trajectory — noise absorption fraction")
axes8[1].legend(ncol=2, fontsize=9)
axes8[1].set_ylim(ylim_bot, 1.05)

fig8.suptitle("Capacity and noise absorption trajectories", fontsize=12)
plt.tight_layout()

out = OUT / "fig_noise_capacity_trajectories.pdf"
fig8.savefig(out, bbox_inches="tight")
fig8.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=110)
plt.close()
print(f"Saved: {out}")
