"""
fig_noise_taxonomy.py
Generates fig_noise_taxonomy.pdf — label corruption taxonomy (Type A / B / C)
for RF vs LR evaluators across separability levels and noise rates.

Paper: Figure 4 / Section "Label noise taxonomy and estimation".
Inputs (quick mode): data/paper5_comparison_agg.csv
Output: outputs/fig_noise_taxonomy.pdf

Run:
    python figures/fig_noise_taxonomy.py
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT     = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

COMP_CSV = DATA / "paper5_comparison_agg.csv"
if not COMP_CSV.exists():
    raise FileNotFoundError(COMP_CSV)

df_comp = pd.read_csv(COMP_CSV)
if "sep" not in df_comp.columns and "class_sep" in df_comp.columns:
    df_comp = df_comp.rename(columns={"class_sep": "sep"})

PALETTE = {
    "TypeA": "#E24B4A",
    "TypeB": "#EF9F27",
    "TypeC": "#B4B2A9",
}

SEP_LEVELS  = [0.5, 1.0, 1.5, 2.0]
FLIP_LEVELS = [0.05, 0.15, 0.30]

FS_BASE    = 12
FS_TITLE   = 12
FS_YLABEL  = 11
FS_XTICK   = 10
FS_ANNOT   = 9
FS_LEGEND  = 11
FS_SUPTITLE = 13

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         FS_BASE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

n_flip = len(FLIP_LEVELS)
n_sep  = len(SEP_LEVELS)
x      = np.arange(n_sep)
w      = 0.38

fig, axes = plt.subplots(1, n_flip, figsize=(5.5 * n_flip, 6.5), sharey=True)

for col, flip in enumerate(FLIP_LEVELS):
    ax  = axes[col]
    sub = df_comp[np.isclose(df_comp["flip_y"], flip)]

    rf = sub[sub["arch"] == "RF"].sort_values("sep")
    lr = sub[sub["arch"] == "LR"].sort_values("sep")

    ax.bar(x - w / 2, rf["frac_C_mean"], w, color=PALETTE["TypeC"], label="Type C (RF)")
    ax.bar(x - w / 2, rf["frac_B_mean"], w,
           bottom=rf["frac_C_mean"], color=PALETTE["TypeB"])
    ax.bar(x - w / 2, rf["frac_A_mean"], w,
           bottom=rf["frac_C_mean"] + rf["frac_B_mean"], color=PALETTE["TypeA"])

    lc = lr["frac_C_mean"].values
    lb = lr["frac_B_mean"].values
    la = lr["frac_A_mean"].values

    ax.bar(x + w / 2, lc, w, color=PALETTE["TypeC"],
           hatch="///", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, lb, w, bottom=lc,
           color=PALETTE["TypeB"], hatch="///", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, la, w, bottom=lc + lb,
           color=PALETTE["TypeA"], hatch="///", edgecolor="white", linewidth=0.5)

    for xi, (la_v, lc_v, lb_v) in enumerate(zip(la, lc, lb)):
        top = lc_v + lb_v + la_v
        ax.text(xi + w / 2, top + 0.015, f"{la_v:.3f}",
                ha="center", va="bottom", fontsize=FS_ANNOT,
                color="#8a1a1a", fontweight="500")

    ax.set_xticks(x)
    ax.set_xticklabels([f"sep={s}" for s in SEP_LEVELS],
                       rotation=30, ha="right", fontsize=FS_XTICK)
    ax.set_title(fr"$\varphi = {flip:.2f}$", fontsize=FS_TITLE)
    ax.set_ylim(0, 1.12)
    if col == 0:
        ax.set_ylabel("Fraction of samples", fontsize=FS_YLABEL)

legend_handles = [
    mpatches.Patch(color=PALETTE["TypeA"],
                   label=r"Type A — Explicit noise  ($\delta_i > \tau$)"),
    mpatches.Patch(color=PALETTE["TypeB"],
                   label=r"Type B — Aleatoric uncertainty  ($H_i > \tau_H$)"),
    mpatches.Patch(color=PALETTE["TypeC"],
                   label="Type C — Structural absorption / clean"),
    mpatches.Patch(facecolor="#ccc", label="Solid = RF   |   Hatched = LR"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=FS_LEGEND, framealpha=0.9, edgecolor="#ccc",
           bbox_to_anchor=(0.5, -0.13))

fig.suptitle(
    "Noise taxonomy: RF vs. LR evaluator  ($B = 50$,  $\\rho = 0.8$)\n"
    "Switching from RF to LR substantially reduces Type C absorption"
    " and exposes Type A noise",
    fontsize=FS_SUPTITLE, weight="500", y=1.04,
)
fig.tight_layout()

out = OUT / "fig_noise_taxonomy.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
