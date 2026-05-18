"""
fig_plate_diagram.py
Generates figS0_plate_diagram.pdf — plate diagram of the LPI-B²
hierarchical generative model.

Paper: Methods section / Supplementary Figure S0.
Inputs: none (pure matplotlib drawing).
Output: outputs/figS0_plate_diagram.pdf

Run:
    python figures/fig_plate_diagram.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "cm",
})

C_OBS     = "#534AB7"
C_LATENT  = "#FFFFFF"
C_PRIOR   = "#B4B2A9"
C_PLATE_J = "#1D9E75"
C_PLATE_I = "#7F77DD"
C_EDGE    = "#222222"
C_TEXT    = "#222222"

FS_PRIOR_LABEL = 11.5
FS_EQ          = 11.0
FS_EQ_TITLE    = 12.0
FS_LEGEND      = 10.5
FS_NODE        = 11

fig, ax = plt.subplots(figsize=(12.0, 5.8))
ax.set_xlim(0, 13)
ax.set_ylim(1.4, 7.6)
ax.set_aspect("equal")
ax.axis("off")


def node_circle(ax, xy, label, observed=False, radius=0.36, fontsize=FS_NODE):
    fc = C_OBS if observed else C_LATENT
    tc = "white" if observed else C_TEXT
    ax.add_patch(Circle(xy, radius, facecolor=fc, edgecolor=C_EDGE,
                        linewidth=1.4, zorder=4))
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=fontsize, color=tc, zorder=5)


def node_prior(ax, xy, label, label_offset=(0.28, 0.0)):
    ax.add_patch(Circle(xy, 0.18, facecolor=C_PRIOR, edgecolor=C_EDGE,
                        linewidth=0.9, zorder=4))
    ax.text(xy[0] + label_offset[0], xy[1] + label_offset[1], label,
            ha="left", va="center", fontsize=FS_PRIOR_LABEL, color=C_TEXT, zorder=5)


def arrow(ax, p_from, p_to, r_from=0.36, r_to=0.36):
    p_from, p_to = np.array(p_from), np.array(p_to)
    d = p_to - p_from
    length = np.linalg.norm(d)
    d_hat = d / length
    start = p_from + d_hat * r_from
    end   = p_to   - d_hat * r_to
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                lw=1.2, mutation_scale=12),
                zorder=3)


def plate(ax, xy, w, h, label, color, label_corner="bl"):
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch(xy, w, h,
                          boxstyle="square,pad=0.05",
                          facecolor="none", edgecolor=color,
                          linewidth=1.6, zorder=1)
    ax.add_patch(rect)
    pad = 0.12
    if label_corner == "bl":
        lx, ly = xy[0] + pad, xy[1] + pad
        ha, va = "left", "bottom"
    elif label_corner == "br":
        lx, ly = xy[0] + w - pad, xy[1] + pad
        ha, va = "right", "bottom"
    elif label_corner == "tl":
        lx, ly = xy[0] + pad, xy[1] + h - pad
        ha, va = "left", "top"
    else:
        lx, ly = xy[0] + w - pad, xy[1] + h - pad
        ha, va = "right", "top"
    ax.text(lx, ly, label, ha=ha, va=va,
            fontsize=10.5, color=color, style="italic", zorder=2)


P_pT   = (1.40, 6.80)
P_muSe = (2.80, 6.80)
P_kSe  = (3.90, 6.80)
P_muSp = (5.10, 6.80)
P_kSp  = (6.20, 6.80)
P_kobs = (8.40, 6.80)

P_Sej  = (3.20, 5.00)
P_Spj  = (5.80, 5.00)
P_Ti   = (1.40, 3.00)
P_Qij  = (4.30, 3.00)

plate(ax, xy=(2.10, 1.80), w=5.80, h=4.00,
      label="j = 1, …, B", color=C_PLATE_J, label_corner="br")
plate(ax, xy=(0.55, 1.80), w=4.70, h=2.40,
      label="i = 1, …, N", color=C_PLATE_I, label_corner="bl")

node_prior(ax, P_pT,   r"$p_T$")
node_prior(ax, P_muSe, r"$\mu_{Se}$")
node_prior(ax, P_kSe,  r"$\kappa_{Se}$")
node_prior(ax, P_muSp, r"$\mu_{Sp}$")
node_prior(ax, P_kSp,  r"$\kappa_{Sp}$")
node_prior(ax, P_kobs, r"$\kappa_{\mathrm{obs}}$")

node_circle(ax, P_Sej, r"$Se_j$")
node_circle(ax, P_Spj, r"$Sp_j$")
node_circle(ax, P_Ti,  r"$T_i$")
node_circle(ax, P_Qij, r"$Q_{ij}$", observed=True)

arrow(ax, P_muSe, P_Sej, r_from=0.18, r_to=0.36)
arrow(ax, P_kSe,  P_Sej, r_from=0.18, r_to=0.36)
arrow(ax, P_muSp, P_Spj, r_from=0.18, r_to=0.36)
arrow(ax, P_kSp,  P_Spj, r_from=0.18, r_to=0.36)
arrow(ax, P_pT,   P_Ti,  r_from=0.18, r_to=0.36)
arrow(ax, P_Sej,  P_Qij)
arrow(ax, P_Spj,  P_Qij)
arrow(ax, P_Ti,   P_Qij)
arrow(ax, P_kobs, P_Qij, r_from=0.18, r_to=0.36)

ax.text(4.50, 7.42,
        r"LPI-B$^2$: hierarchical generative model",
        ha="center", va="center", fontsize=14, weight="bold")

LX, LY = 9.90, 7.00
ax.text(LX + 0.22, LY + 0.10, "Legend",
        ha="left", va="center", fontsize=FS_LEGEND + 1, weight="bold")
ax.add_patch(Circle((LX, LY - 0.55), 0.18, facecolor=C_LATENT,
                    edgecolor=C_EDGE, linewidth=1.2))
ax.text(LX + 0.32, LY - 0.55, "latent variable",
        ha="left", va="center", fontsize=FS_LEGEND)
ax.add_patch(Circle((LX, LY - 1.15), 0.18, facecolor=C_OBS,
                    edgecolor=C_EDGE, linewidth=1.2))
ax.text(LX + 0.32, LY - 1.15, "observed",
        ha="left", va="center", fontsize=FS_LEGEND)
ax.add_patch(Circle((LX, LY - 1.75), 0.13, facecolor=C_PRIOR,
                    edgecolor=C_EDGE, linewidth=0.9))
ax.text(LX + 0.32, LY - 1.75, "prior /\nhyperparameter",
        ha="left", va="center", fontsize=FS_LEGEND)

EX, EY = 9.90, 3.90
ax.text(EX, EY + 0.40, "Generative process",
        ha="left", va="bottom",
        fontsize=FS_EQ_TITLE, weight="bold", color=C_PLATE_J)
eqs = [
    r"$Se_j \sim \mathrm{Beta}(\mu_{Se}\kappa_{Se},\,(1{-}\mu_{Se})\kappa_{Se})$",
    r"$Sp_j \sim \mathrm{Beta}(\mu_{Sp}\kappa_{Sp},\,(1{-}\mu_{Sp})\kappa_{Sp})$",
    r"$T_i \sim \mathrm{Bernoulli}(p_T)$",
    r"$Q_{ij}\!\mid\!T_i\!=\!1 \sim \mathrm{Beta}(Se_j\kappa_{\mathrm{obs}},\,(1{-}Se_j)\kappa_{\mathrm{obs}})$",
    r"$Q_{ij}\!\mid\!T_i\!=\!0 \sim \mathrm{Beta}((1{-}Sp_j)\kappa_{\mathrm{obs}},\,Sp_j\kappa_{\mathrm{obs}})$",
]
for k, eq in enumerate(eqs):
    ax.text(EX, EY - 0.02 - 0.42 * k, eq,
            ha="left", va="top", fontsize=FS_EQ, color=C_TEXT)

plt.tight_layout(pad=0.3)
out_path = OUT / "figS0_plate_diagram.pdf"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
