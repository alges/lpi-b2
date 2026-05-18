"""
fig_estimator_correlations.py
Generates fig_estimator_correlations.pdf — Pearson correlation matrix
comparing classical scalar metrics (accuracy, NLL, Brier, ECE, pred_var)
against LPI-B² descriptors (µ̂_avg, κ̂_avg).

Also saves data/postproc_02_classical_vs_lpi.csv, which is required by
fig_ranking_recovery.py.

Paper: Supplementary Figure S1 / Section "Comparison with Confident Learning".
Inputs (quick mode):
    data/sample_posteriors_yc.csv
    data/table_multiarch_def_yc.csv
    data/arch_summary.csv
Output:
    outputs/fig_estimator_correlations.pdf
    data/postproc_02_classical_vs_lbi.csv  (intermediate, used by fig_ranking_recovery)

Run:
    python figures/fig_estimator_correlations.py
"""
import sys
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
TAG = "[fig_estimator_correlations]"
ARCH_COLORS   = {"LR": "#9467bd", "RF": "#2ca02c", "SVM": "#1f77b4", "GB": "#ff7f0e", "MLP": "#d62728"}
NOISE_MARKERS = {0.0: "D", 0.1: "o", 0.3: "s", 0.5: "^"}

NEEDED = [DATA / "sample_posteriors_yc.csv",
          DATA / "table_multiarch_def_yc.csv",
          DATA / "arch_summary.csv"]
for p in NEEDED:
    if not p.exists():
        print(f"{TAG} ERROR: missing {p}"); sys.exit(1)

post     = pd.read_csv(DATA / "sample_posteriors_yc.csv")
tbl      = pd.read_csv(DATA / "table_multiarch_def_yc.csv")
arch_sum = pd.read_csv(DATA / "arch_summary.csv")
arch_sum["kappa_avg_mean"] = (arch_sum["kappa_Se_mean"] + arch_sum["kappa_Sp_mean"]) / 2


def compute_ece(p_pred, y_true, n_bins=15):
    confidence = np.maximum(p_pred, 1 - p_pred)
    predicted  = (p_pred >= 0.5).astype(int)
    correct    = (predicted == y_true).astype(float)
    bin_edges  = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(p_pred)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        acc_bin  = correct[mask].mean()
        conf_bin = confidence[mask].mean()
        ece += mask.sum() / n * abs(acc_bin - conf_bin)
    return ece


ens_rows = []
for (arch, seed, flip_y), grp in post.groupby(["arch", "seed", "flip_y"]):
    p       = grp["p_truth"].values
    y_clean = grp["y_clean"].values
    eps     = 1e-7
    acc_ens  = np.mean((p >= 0.5).astype(int) == y_clean)
    nll      = -np.mean(y_clean * np.log(p + eps) + (1 - y_clean) * np.log(1 - p + eps))
    brier    = np.mean((p - y_clean) ** 2)
    ece      = compute_ece(p, y_clean)
    pred_var = np.mean(p * (1 - p))
    mean_entr= np.mean(grp["entropy"].values)
    mean_disc= np.mean(grp["discordance"].values)
    frac_A   = np.mean(grp["discordance"].values > 0.8)
    ens_rows.append({"arch": arch, "seed": seed, "flip_y": flip_y,
                     "acc_ens": acc_ens, "nll": nll, "brier": brier,
                     "ece": ece, "pred_var": pred_var,
                     "mean_entropy": mean_entr, "mean_discordance": mean_disc,
                     "frac_A_proxy": frac_A})

ens = pd.DataFrame(ens_rows)
tbl["kappa_avg"] = (tbl["kappa_Se"] + tbl["kappa_Sp"]) / 2
merged = ens.merge(
    tbl[["arch", "seed", "flip_y", "cv_acc", "cv_recall", "cv_f1",
         "mu_avg", "kappa_avg", "mu_Se", "mu_Sp", "kappa_Se", "kappa_Sp"]],
    on=["arch", "seed", "flip_y"], how="inner")

out_csv = DATA / "postproc_02_classical_vs_lbi.csv"
merged.to_csv(out_csv, index=False)
print(f"{TAG} Saved intermediate CSV: {out_csv}")

agg = merged.groupby(["arch", "flip_y"]).agg(
    acc_ens_mean=("acc_ens", "mean"),      acc_ens_std=("acc_ens", "std"),
    nll_mean=("nll", "mean"),
    brier_mean=("brier", "mean"),
    ece_mean=("ece", "mean"),
    pred_var_mean=("pred_var", "mean"),
    mu_avg_mean=("mu_avg", "mean"),        mu_avg_std=("mu_avg", "std"),
    kappa_avg_mean=("kappa_avg", "mean"),  kappa_avg_std=("kappa_avg", "std"),
).reset_index()

metric_cols = ["acc_ens", "nll", "brier", "ece", "pred_var", "mu_avg", "kappa_avg"]
labels_map  = ["Accuracy", "NLL", "Brier", "ECE", "Pred.Var", "µ̂_avg", "κ̂_avg"]
corr_mat    = merged[metric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_mat.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(metric_cols))); ax.set_xticklabels(labels_map, rotation=45, ha="right")
ax.set_yticks(range(len(metric_cols))); ax.set_yticklabels(labels_map)
for i in range(len(metric_cols)):
    for j in range(len(metric_cols)):
        val = corr_mat.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color="white" if abs(val) > 0.6 else "black")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Correlation matrix: classical metrics vs LPI-B² descriptors\n"
             "(all arches × noise levels × seeds)")
plt.tight_layout()

out = OUT / "fig_estimator_correlations.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close()
print(f"{TAG} Saved: {out}")
