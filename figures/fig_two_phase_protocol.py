"""
fig_two_phase_protocol.py
Generates fig_two_phase_protocol.pdf — the two-phase LPI-B² auditing protocol.

Paper: Figure 6 / Section "The two-phase auditing protocol".
Inputs (quick mode): data/paper6_two_phase_raw1.5.csv
Output: outputs/fig_two_phase_protocol.pdf

Run:
    python figures/fig_two_phase_protocol.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PALETTE = {
    "oracle":  "#2C2C2A",
    "noisy":   "#D85A30",
    "clean":   "#1D9E75",
}

csv_path = DATA / "paper6_two_phase_raw1.5.csv"
df_raw = pd.read_csv(csv_path)

valid = df_raw[df_raw["collapse"] == False].dropna(subset=["mu_Se_clean"])
flip_arr = np.array(sorted(valid["flip_y"].unique()))

def curve(col):
    g = valid.groupby("flip_y")[col]
    return g.mean().values, g.std().values

m_or, s_or = curve("mu_Se_oracle")
m_no, s_no = curve("mu_Se_noisy")
m_cl, s_cl = curve("mu_Se_clean")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(flip_arr, m_or, color=PALETTE["oracle"], lw=2.5,
        ls="-", marker="D", markersize=5,
        label=r"$\hat{\mu}_{Se}^{\mathrm{oracle}}$ (RF, $\varphi=0$)")
ax.fill_between(flip_arr, m_or - s_or, m_or + s_or,
                alpha=0.10, color=PALETTE["oracle"])

ax.plot(flip_arr, m_no, color=PALETTE["noisy"], lw=2.0,
        ls="--", marker="o", markersize=5,
        label=r"$\hat{\mu}_{Se}^{\mathrm{noisy}}$ (RF on noisy data)")
ax.fill_between(flip_arr, m_no - s_no, m_no + s_no,
                alpha=0.12, color=PALETTE["noisy"])

ax.plot(flip_arr, m_cl, color=PALETTE["clean"], lw=2.5,
        ls="-", marker="s", markersize=5,
        label=r"$\hat{\mu}_{Se}^{\mathrm{clean}}$ (RF after LPI-B$^2$ cleaning)")
ax.fill_between(flip_arr, m_cl - s_cl, m_cl + s_cl,
                alpha=0.15, color=PALETTE["clean"])

ax.set_xlabel("True noise level $\\varphi$", fontsize=11)
ax.set_ylabel("$\\hat{\\mu}_{Se}$", fontsize=11)
ax.set_title(
    "Two-phase LPI-B$^2$ protocol: architectural sensitivity under noise\n"
    "class\\_sep $= 1.5$,  $B=50$,  $\\rho=0.8$  |  "
    "Shaded bands $= \\pm 1$ std over 6 seeds",
    fontsize=9.5
)
ax.legend(fontsize=9.5, loc="lower left")
fig.tight_layout()

output_path = OUT / "fig_two_phase_protocol.pdf"
fig.savefig(output_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {output_path}")
