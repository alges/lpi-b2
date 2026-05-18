"""
fig_strong_vs_weak_capacity.py
Generates fig_strong_vs_weak_capacity.pdf — full vs. weak architecture
comparison: latent performance and noise absorption at φ = 0.10.

Paper: Figure 7 / Section "Capacity and modes of consistency".
Inputs (quick mode):
    data/table_multiarch_def.csv
    data/table_multiarch_def_sep1_W.csv
    data/sample_posteriors.csv
    data/sample_posteriors_sep1_W.csv
Output: outputs/fig_strong_vs_weak_capacity.pdf

Run:
    python figures/fig_strong_vs_weak_capacity.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data"
OUT    = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ARCH_ORDER   = ["LR", "RF", "SVM", "GB", "MLP"]
NOISE_LABELS = {0.1: "φ = 0.10", 0.3: "φ = 0.30", 0.5: "φ = 0.50"}
TAU_DISC     = 0.8
TAU_ENTROPY  = 0.8

PALETTE = {
    "LR":  "#7F77DD",
    "RF":  "#1D9E75",
    "SVM": "#D85A30",
    "GB":  "#BA7517",
    "MLP": "#378ADD",
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})


def load_posteriors(path):
    df = pd.read_csv(path)
    is_A = df["discordance"] > TAU_DISC
    is_B = (~is_A) & (df["entropy"] > TAU_ENTROPY)
    is_C = (~is_A) & (~is_B)
    df["type_A"] = is_A.astype(int)
    df["type_B"] = is_B.astype(int)
    df["type_C"] = is_C.astype(int)
    return df


def taxonomy_fractions(post_df):
    grp  = post_df.groupby(["arch", "flip_y"])
    frac = grp[["type_A", "type_B", "type_C"]].mean().reset_index()
    frac.columns = ["arch", "flip_y", "frac_A", "frac_B", "frac_C"]
    return frac


df_full   = pd.read_csv(DATA / "table_multiarch_def.csv")
df_weak   = pd.read_csv(DATA / "table_multiarch_def_sep1_W.csv")
post_full = load_posteriors(DATA / "sample_posteriors.csv")
post_weak = load_posteriors(DATA / "sample_posteriors_sep1_W.csv")

tax_full = taxonomy_fractions(post_full)
tax_weak = taxonomy_fractions(post_weak)

phi = 0.1

rows_full, rows_weak = [], []
for arch in ARCH_ORDER:
    gf = df_full[(df_full["arch"] == arch) & (df_full["flip_y"] == phi)]
    gw = df_weak[(df_weak["arch"] == arch) & (df_weak["flip_y"] == phi)]
    tf = tax_full[(tax_full["arch"] == arch) & (tax_full["flip_y"] == phi)]
    tw = tax_weak[(tax_weak["arch"] == arch) & (tax_weak["flip_y"] == phi)]
    rows_full.append({
        "arch": arch,
        "mu_avg": gf["mu_avg"].mean(),
        "mu_avg_std": gf["mu_avg"].std() if len(gf) > 1 else 0,
        "frac_C": tf["frac_C"].mean() if not tf.empty else np.nan,
    })
    rows_weak.append({
        "arch": arch,
        "mu_avg": gw["mu_avg"].mean(),
        "mu_avg_std": gw["mu_avg"].std() if len(gw) > 1 else 0,
        "frac_C": tw["frac_C"].mean() if not tw.empty else np.nan,
    })

df_f = pd.DataFrame(rows_full)
df_w = pd.DataFrame(rows_weak)

x     = np.arange(len(ARCH_ORDER))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.bar(x - width / 2, df_f["mu_avg"], width,
        color=[PALETTE[a] for a in ARCH_ORDER],
        alpha=0.85, label="Full arch",
        yerr=df_f["mu_avg_std"], capsize=3)
ax1.bar(x + width / 2, df_w["mu_avg"], width,
        color=[PALETTE[a] for a in ARCH_ORDER],
        alpha=0.40, label="Weak arch",
        yerr=df_w["mu_avg_std"], capsize=3, hatch="//")

ax2.plot(x - width / 2, df_f["frac_C"],
         "D--", color="#555", markersize=6, lw=1.4, label="frac_C (full)")
ax2.plot(x + width / 2, df_w["frac_C"],
         "D:", color="#aaa", markersize=6, lw=1.4, label="frac_C (weak)")

ax1.set_xticks(x)
ax1.set_xticklabels(ARCH_ORDER, fontsize=10)
ax1.set_ylabel("µ_avg  (latent performance)", fontsize=10)
ax1.set_ylim(0, 1.05)
ax2.set_ylabel("Type C fraction  (structural absorption)", fontsize=10)
ax2.set_ylim(0, 1.05)

ax1.set_title(
    f"Full vs Weak Architecture: latent performance and noise absorption\n"
    f"({NOISE_LABELS[phi]}, sep = 1.0; hatched bars = weak version)",
    fontsize=10,
)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

fig.tight_layout()
out = OUT / "fig_strong_vs_weak_capacity.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
