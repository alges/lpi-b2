"""
exp_two_phase_protocol.py
================================
Paper Experiment 6 — The complete two-phase evaluation protocol.

This is the experiment that demonstrates that the LPI-B2 auditing
chain is actionable: detecting noisy samples and removing them produces
a measurably better estimate of the target architecture's true capability.

Protocol
--------
For each (flip_y, seed) condition:

  Phase 0 — Oracle baseline (phi=0 reference):
    Run BTD with RF on a clean dataset (flip_y=0, same seed).
    → mu_Se_oracle: what RF reports when labels are clean.

  Phase 1 — Audit (LR evaluator):
    Run BTD with LR on the noisy dataset.
    Identify Type A samples: discordance delta_i > tau = 0.8.
    → type_A_indices: candidate mislabelled samples.
    → E1_LR: detected noise fraction.

  Phase 2a — Evaluation on noisy data (RF evaluator):
    Run BTD with RF on the original noisy dataset.
    → mu_Se_noisy: what RF reports before cleaning.

  Phase 2b — Evaluation on cleaned data (RF evaluator):
    Remove Type A samples identified in Phase 1.
    Run BTD with RF on the residual dataset.
    → mu_Se_clean: what RF reports after cleaning.

Hypothesis
----------
  mu_Se_clean > mu_Se_noisy for phi > 0, and
  mu_Se_clean is closer to mu_Se_oracle than mu_Se_noisy is,
  particularly in the range phi in [0.10, 0.25] where LR recall is
  high and false positives are manageable.

Grid
----
  flip_y    : {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
  class_sep : 1.5  (fixed — best regime for LR auditing)
  seeds     : 6 per condition

Total BTD runs: 7 flip levels × 6 seeds × 3 phases = 126 Stan calls.
At ~30s per call: ~63 minutes on a machine with parallel_chains=4.

Outputs
-------
  results/tables/paper6_two_phase_raw.csv
      Per-seed raw results: mu_Se_oracle, mu_Se_noisy, mu_Se_clean,
      mu_Sp versions, n_type_A removed, E1_LR, fraction_removed.

  results/tables/paper6_two_phase_agg.csv
      Aggregated means and stds per flip_y level.

  results/figures/fig_paper6_two_phase_main.pdf
      MAIN FIGURE: three curves over phi axis:
        - mu_Se_oracle  (RF on clean data, reference)
        - mu_Se_noisy   (RF on noisy data)
        - mu_Se_clean   (RF after LR cleaning)
      Shows mu_Se_clean converging toward mu_Se_oracle.

  results/figures/fig_paper6_improvement.pdf
      Improvement delta = mu_Se_clean - mu_Se_noisy over phi.
      Positive values confirm the cleaning helps.
      Overlaid with E1_LR to show that improvement tracks detection quality.

  results/figures/fig_paper6_removal_stats.pdf
      Fraction of samples removed (Type A) and precision proxy
      (fraction of removed samples that were genuinely noisy).

Usage
-----
  python exp_two_phase_protocol.py

Dependencies: lpi_b2/ package (pip install -e .)
must be in the same directory or on PYTHONPATH.

Notes
-----
- If a Phase 1 LR run produces a collapse (E1 >= 0.99), the seed is
  flagged and excluded from aggregation. The raw CSV still records it.
- Phase 2b uses all remaining samples after Type A removal. If too few
  samples remain (< 200), the seed is skipped and a warning is printed.
- anchor_rate for Phase 2b is recomputed as rho=0.8 of the cleaned N.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from lpi_b2.estimator import LPIB2Evaluator

warnings.filterwarnings("ignore")

# ── output dirs ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

(ROOT / "data").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(parents=True, exist_ok=True)

# ── plot style (matching existing scripts) ────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PALETTE = {
    "oracle":  "#2C2C2A",
    "noisy":   "#D85A30",
    "clean":   "#1D9E75",
    "delta":   "#7F77DD",
    "E1_LR":   "#BA7517",
    "TypeA":   "#E24B4A",
}

# ── experiment settings ───────────────────────────────────────────────────────
FLIP_LEVELS   = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CLASS_SEP     = 1.0
N_SEEDS       = 6
N_BOOTSTRAP   = 50
ANCHOR_RATE   = 0.8

# Phase 1 LR settings
LR_MAX_ITER   = 1000

# Taxonomy thresholds
TAU_DISC      = 0.8
TAU_ENTROPY   = 0.8

# Safety: skip Phase 2b if cleaned dataset is too small
MIN_CLEAN_N   = 200

# Collapse detection
COLLAPSE_THRESHOLD = 0.99


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_data(n=1000, sep=1.5, flip=0.0, seed=42):
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=10, n_redundant=4,
        n_repeated=0, class_sep=sep, flip_y=flip, random_state=seed,
    )
    return X, y


def run_btd(X, y, base_clf, anchor_rate=ANCHOR_RATE,
            n_bootstrap=N_BOOTSTRAP):
    """Run BTD and return the fitted evaluator."""
    ev = LPIB2Evaluator(
        base_estimator=base_clf,
        n_bootstrap=n_bootstrap,
        anchor_rate=anchor_rate,
    )
    ev.evaluate(X, y)
    return ev


def posterior_signals(ev, y):
    """Extract delta_i, H_i from fitted evaluator."""
    p   = np.mean(ev.mcmc_fit_.stan_variable("prob_Ti_pos"), axis=0)
    eps = 1e-9
    delta = np.abs(y - p)
    H     = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))
    return delta, H, p


def E1_gcr_plateau(delta):
    """GCR plateau estimator (same implementation as experiments_3_5y6b.py)."""
    tau_grid = np.linspace(0.05, 0.95, 100)
    gcr      = np.array([np.mean(delta > t) for t in tau_grid])
    grad     = np.abs(np.diff(gcr))
    thr      = 0.01 * gcr[0] if gcr[0] > 0 else 1e-4
    idx      = np.where(grad < thr)[0]
    return float(gcr[idx[0]] if len(idx) else gcr[-1])


def get_mu_Se_Sp(ev):
    """Return posterior mean of mu_Se and mu_Sp."""
    return (
        float(np.mean(ev.mcmc_fit_.stan_variable("mu_Se"))),
        float(np.mean(ev.mcmc_fit_.stan_variable("mu_Sp"))),
        float(np.std(ev.mcmc_fit_.stan_variable("mu_Se"))),
        float(np.std(ev.mcmc_fit_.stan_variable("mu_Sp"))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(flip_levels=FLIP_LEVELS, sep=CLASS_SEP, n_seeds=N_SEEDS):
    print(f"\n=== Paper Experiment 6: Two-phase protocol "
          f"(sep={sep}, B={N_BOOTSTRAP}) ===")
    records = []

    # Base classifiers (created once, cloned internally by BTD)
    lr_clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=0)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)

    total = len(flip_levels) * n_seeds
    done  = 0

    for flip in flip_levels:
        for seed in range(n_seeds):
            done += 1
            print(f"  [{done:3d}/{total}] flip={flip:.2f}, seed={seed}", flush=True)

            # ── data ──────────────────────────────────────────────────────────
            X_noisy, y_noisy = make_data(sep=sep, flip=flip, seed=seed)
            X_clean_ref, y_clean_ref = make_data(sep=sep, flip=0.0, seed=seed)
            # Note: same seed → same feature structure, different labels

            record = {
                "flip_y": flip, "seed": seed,
                "N_original": len(y_noisy),
            }

            # ── Phase 0: Oracle (RF on clean labels) ─────────────────────────
            print("    Phase 0: Oracle RF (phi=0)...", end=" ", flush=True)
            ev_oracle = run_btd(X_clean_ref, y_clean_ref, rf_clf)
            mu_Se_oracle, mu_Sp_oracle, std_Se_oracle, _ = get_mu_Se_Sp(ev_oracle)
            record.update({
                "mu_Se_oracle": mu_Se_oracle,
                "mu_Sp_oracle": mu_Sp_oracle,
                "std_Se_oracle": std_Se_oracle,
            })
            print(f"mu_Se={mu_Se_oracle:.3f}", flush=True)

            # ── Phase 1: Audit with LR ────────────────────────────────────────
            print("    Phase 1: Audit LR...", end=" ", flush=True)
            ev_lr = run_btd(X_noisy, y_noisy, lr_clf)
            delta_lr, H_lr, p_lr = posterior_signals(ev_lr, y_noisy)
            E1_lr = E1_gcr_plateau(delta_lr)

            # Collapse check
            collapse = E1_lr >= COLLAPSE_THRESHOLD
            record["E1_LR"]    = E1_lr
            record["collapse"] = collapse

            # Identify Type A: discordance > TAU_DISC
            type_A_mask = delta_lr > TAU_DISC
            n_type_A    = type_A_mask.sum()
            record["n_type_A"]          = n_type_A
            record["fraction_removed"]  = n_type_A / len(y_noisy)

            # Type B and C fractions for reference
            type_B_mask = (H_lr > TAU_ENTROPY) & ~type_A_mask
            record["frac_A_LR"] = type_A_mask.mean()
            record["frac_B_LR"] = type_B_mask.mean()
            record["frac_C_LR"] = (~type_A_mask & ~type_B_mask).mean()

            if collapse:
                print(f"COLLAPSE (E1={E1_lr:.3f}), skipping phases 2a/2b",
                      flush=True)
                for k in ["mu_Se_noisy", "mu_Sp_noisy", "std_Se_noisy",
                          "mu_Se_clean", "mu_Sp_clean", "std_Se_clean",
                          "N_clean", "skip_reason"]:
                    record[k] = np.nan
                record["skip_reason"] = "collapse"
                records.append(record)
                continue

            print(f"E1={E1_lr:.3f}, Type A removed={n_type_A}", flush=True)

            # ── Phase 2a: RF on noisy dataset ─────────────────────────────────
            print("    Phase 2a: RF on noisy...", end=" ", flush=True)
            ev_noisy = run_btd(X_noisy, y_noisy, rf_clf)
            mu_Se_noisy, mu_Sp_noisy, std_Se_noisy, _ = get_mu_Se_Sp(ev_noisy)
            record.update({
                "mu_Se_noisy":  mu_Se_noisy,
                "mu_Sp_noisy":  mu_Sp_noisy,
                "std_Se_noisy": std_Se_noisy,
            })
            print(f"mu_Se={mu_Se_noisy:.3f}", flush=True)

            # ── Phase 2b: RF on cleaned dataset ──────────────────────────────
            clean_mask = ~type_A_mask
            X_clean    = X_noisy[clean_mask]
            y_clean    = y_noisy[clean_mask]
            N_clean    = len(y_clean)
            record["N_clean"] = N_clean

            if N_clean < MIN_CLEAN_N:
                print(f"    Phase 2b: SKIPPED (N_clean={N_clean} < {MIN_CLEAN_N})",
                      flush=True)
                record["mu_Se_clean"]  = np.nan
                record["mu_Sp_clean"]  = np.nan
                record["std_Se_clean"] = np.nan
                record["skip_reason"]  = f"N_clean={N_clean}"
                records.append(record)
                continue

            print(f"    Phase 2b: RF on cleaned (N={N_clean})...",
                  end=" ", flush=True)
            ev_clean = run_btd(X_clean, y_clean, rf_clf,
                               anchor_rate=ANCHOR_RATE,
                               n_bootstrap=N_BOOTSTRAP)
            mu_Se_clean, mu_Sp_clean, std_Se_clean, _ = get_mu_Se_Sp(ev_clean)
            record.update({
                "mu_Se_clean":  mu_Se_clean,
                "mu_Sp_clean":  mu_Sp_clean,
                "std_Se_clean": std_Se_clean,
                "skip_reason":  None,
            })
            print(f"mu_Se={mu_Se_clean:.3f}  "
                  f"[Δ = {mu_Se_clean - mu_Se_noisy:+.3f}]", flush=True)

            records.append(record)

    df = pd.DataFrame(records)
    csv_path = ROOT / "data" / f"paper6_two_phase_raw{sep}.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  → saved {csv_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(df):
    """Aggregate over seeds; exclude collapses and skipped seeds."""
    valid = df[df["collapse"] == False].copy()   # noqa: E712
    valid = valid.dropna(subset=["mu_Se_clean"])

    cols = ["mu_Se_oracle", "mu_Se_noisy", "mu_Se_clean",
            "mu_Sp_oracle", "mu_Sp_noisy", "mu_Sp_clean",
            "E1_LR", "fraction_removed", "N_clean"]

    agg = valid.groupby("flip_y")[cols].agg(["mean", "std"]).round(4)

    # Compute improvement and gap reduction
    agg_flat = valid.groupby("flip_y")[cols].mean()
    agg_flat["delta_mu_Se"] = (agg_flat["mu_Se_clean"]
                               - agg_flat["mu_Se_noisy"])
    agg_flat["gap_oracle_noisy"] = (agg_flat["mu_Se_oracle"]
                                    - agg_flat["mu_Se_noisy"])
    agg_flat["gap_oracle_clean"] = (agg_flat["mu_Se_oracle"]
                                    - agg_flat["mu_Se_clean"])
    agg_flat["gap_reduction_pct"] = (
        100 * (agg_flat["gap_oracle_noisy"] - agg_flat["gap_oracle_clean"])
        / agg_flat["gap_oracle_noisy"].replace(0, np.nan)
    )

    n_valid = valid.groupby("flip_y").size().rename("n_valid")
    n_coll  = df[df["collapse"] == True].groupby("flip_y").size().rename("n_collapse")   # noqa: E712
    agg_flat = agg_flat.join(n_valid).join(n_coll).fillna({"n_collapse": 0})

    csv_path = ROOT / "data" / "paper6_two_phase_agg.csv"
    agg_flat.to_csv(csv_path, float_format="%.4f")
    print(f"  → saved {csv_path}")

    return agg_flat, agg


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig_main(df_raw, agg_flat, output_path):
    """
    Main figure: three mu_Se curves over phi.
    oracle (black solid), noisy (orange dashed), clean (green solid).
    Shaded bands = ±1 std over seeds.
    """
    valid = df_raw[df_raw["collapse"] == False].dropna(  # noqa: E712
        subset=["mu_Se_clean"])

    flip_arr = np.array(sorted(valid["flip_y"].unique()))

    def curve(col):
        g = valid.groupby("flip_y")[col]
        return g.mean().values, g.std().values

    m_or, s_or   = curve("mu_Se_oracle")
    m_no, s_no   = curve("mu_Se_noisy")
    m_cl, s_cl   = curve("mu_Se_clean")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(flip_arr, m_or, color=PALETTE["oracle"], lw=2.5,
            ls="-", marker="D", markersize=5, label=r"$\hat{\mu}_{Se}^{\mathrm{oracle}}$ (RF, $\varphi=0$)")
    ax.fill_between(flip_arr, m_or - s_or, m_or + s_or,
                    alpha=0.10, color=PALETTE["oracle"])

    ax.plot(flip_arr, m_no, color=PALETTE["noisy"], lw=2.0,
            ls="--", marker="o", markersize=5, label=r"$\hat{\mu}_{Se}^{\mathrm{noisy}}$ (RF on noisy data)")
    ax.fill_between(flip_arr, m_no - s_no, m_no + s_no,
                    alpha=0.12, color=PALETTE["noisy"])

    ax.plot(flip_arr, m_cl, color=PALETTE["clean"], lw=2.5,
            ls="-", marker="s", markersize=5, label=r"$\hat{\mu}_{Se}^{\mathrm{clean}}$ (RF after LR cleaning)")
    ax.fill_between(flip_arr, m_cl - s_cl, m_cl + s_cl,
                    alpha=0.15, color=PALETTE["clean"])

    ax.set_xlabel("True noise level $\\varphi$", fontsize=11)
    ax.set_ylabel("$\\hat{\\mu}_{Se}$", fontsize=11)
    ax.set_title(
        "Two-phase LPI\\_B$^2$ protocol: architectural sensitivity under noise\n"
        "class\\_sep $= 1.5$,  $B=50$,  $\\rho=0.8$  |  "
        "Shaded bands $= \\pm 1$ std over 6 seeds",
        fontsize=9.5
    )
    ax.legend(fontsize=9.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {output_path}")


def fig_improvement(df_raw, output_path):
    """
    Delta = mu_Se_clean - mu_Se_noisy over phi, with E1_LR on secondary axis.
    Positive delta confirms cleaning helps.
    """
    valid = df_raw[df_raw["collapse"] == False].dropna(  # noqa: E712
        subset=["mu_Se_clean"])

    flip_arr = np.array(sorted(valid["flip_y"].unique()))

    delta_mean = valid.groupby("flip_y").apply(
        lambda g: (g["mu_Se_clean"] - g["mu_Se_noisy"]).mean()).values
    delta_std  = valid.groupby("flip_y").apply(
        lambda g: (g["mu_Se_clean"] - g["mu_Se_noisy"]).std()).values
    e1_mean    = valid.groupby("flip_y")["E1_LR"].mean().values

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    ax1.bar(flip_arr, delta_mean, width=0.025,
            color=PALETTE["delta"], alpha=0.75,
            label=r"$\hat{\mu}_{Se}^{\mathrm{clean}} - \hat{\mu}_{Se}^{\mathrm{noisy}}$")
    ax1.errorbar(flip_arr, delta_mean, yerr=delta_std,
                 fmt="none", color=PALETTE["delta"], capsize=4, lw=1.5)
    ax1.axhline(0, color="black", lw=0.8, ls=":")
    ax1.set_xlabel("True noise level $\\varphi$", fontsize=10)
    ax1.set_ylabel("$\\Delta\\hat{\\mu}_{Se}$ (clean $-$ noisy)", fontsize=10,
                   color=PALETTE["delta"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["delta"])

    ax2.plot(flip_arr, e1_mean, color=PALETTE["E1_LR"], lw=1.8,
             marker="o", markersize=4, ls="--",
             label=r"$\hat{\phi}_{E1}^{\mathrm{LR}}$ (detected noise)")
    ax2.set_ylabel("$\\hat{\\phi}_{E1}^{\\mathrm{LR}}$", fontsize=10,
                   color=PALETTE["E1_LR"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["E1_LR"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title(
        "Improvement in $\\hat{\\mu}_{Se}$ from two-phase cleaning\n"
        "class\\_sep $= 1.5$  |  positive $\\Delta$ = cleaning improved the estimate",
        fontsize=9.5
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {output_path}")


def fig_removal_stats(df_raw, output_path):
    """
    Fraction of samples removed (Type A) and composition of what was removed.
    Secondary plot: gap to oracle before and after cleaning.
    """
    valid = df_raw[df_raw["collapse"] == False].dropna(  # noqa: E712
        subset=["mu_Se_clean"])

    flip_arr   = np.array(sorted(valid["flip_y"].unique()))
    frac_removed = valid.groupby("flip_y")["fraction_removed"].mean().values
    gap_noisy    = (valid.groupby("flip_y")
                    .apply(lambda g: (g["mu_Se_oracle"] - g["mu_Se_noisy"]).mean())
                    .values)
    gap_clean    = (valid.groupby("flip_y")
                    .apply(lambda g: (g["mu_Se_oracle"] - g["mu_Se_clean"]).mean())
                    .values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: fraction removed vs phi and vs true phi
    ax1.bar(flip_arr, frac_removed, width=0.025,
            color=PALETTE["TypeA"], alpha=0.8,
            label="Fraction removed (Type A)")
    ax1.plot(flip_arr, flip_arr, "k--", lw=1, alpha=0.5,
             label="True $\\varphi$ (reference)")
    ax1.set_xlabel("True noise level $\\varphi$", fontsize=10)
    ax1.set_ylabel("Fraction of samples removed", fontsize=10)
    ax1.set_title("Samples removed by LR audit (Type A)\nvs. true noise level",
                  fontsize=9.5)
    ax1.legend(fontsize=8.5)

    # Right: oracle gap before and after cleaning
    w = 0.012
    ax2.bar(flip_arr - w, gap_noisy, width=w * 1.8,
            color=PALETTE["noisy"], alpha=0.8,
            label="Gap: oracle $-$ noisy")
    ax2.bar(flip_arr + w, gap_clean, width=w * 1.8,
            color=PALETTE["clean"], alpha=0.8,
            label="Gap: oracle $-$ clean")
    ax2.axhline(0, color="black", lw=0.7, ls=":")
    ax2.set_xlabel("True noise level $\\varphi$", fontsize=10)
    ax2.set_ylabel("$\\hat{\\mu}_{Se}^{\\mathrm{oracle}} - \\hat{\\mu}_{Se}$",
                   fontsize=10)
    ax2.set_title("Gap to oracle: before vs. after cleaning\n"
                  "Smaller gap = estimate closer to true architecture capability",
                  fontsize=9.5)
    ax2.legend(fontsize=8.5)

    fig.suptitle(
        "Two-phase protocol: removal statistics and oracle gap reduction\n"
        "class\\_sep $= 1.5$,  $B=50$",
        fontsize=10, weight="500"
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Paper Experiment 6: two-phase LPI-B2 protocol."
    )
    parser.add_argument("--figures-only", action="store_true",
                        help="Skip Stan runs; load existing raw CSV and plot only.")
    parser.add_argument("--raw-csv",
                        default="results/tables/paper6_two_phase_raw.csv",
                        help="Path to existing raw CSV (used with --figures-only).")
    parser.add_argument("--sep",  type=float, default=CLASS_SEP)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    if args.figures_only:
        print(f"=== Figures-only mode: loading {args.raw_csv} ===")
        df_raw = pd.read_csv(args.raw_csv)
    else:
        #for sep_val in [1.0, 2.0]:
        run_experiment(
            flip_levels=FLIP_LEVELS,
            sep=2.0,
            n_seeds=args.seeds,
        )

    # agg_flat, _ = aggregate(df_raw)

    # print("\n  Aggregated results:")
    # print(agg_flat[["mu_Se_oracle", "mu_Se_noisy", "mu_Se_clean",
    #                  "delta_mu_Se", "gap_reduction_pct",
    #                  "E1_LR", "fraction_removed",
    #                  "n_valid", "n_collapse"]].to_string())

    # print("\n  Generating figures...")
    # fig_main(
    #     df_raw, agg_flat,
    #     ROOT / "outputs" / "fig_paper6_two_phase_main.pdf"
    # )
    # fig_improvement(
    #     df_raw,
    #     ROOT / "outputs" / "fig_paper6_improvement.pdf"
    # )
    # fig_removal_stats(
    #     df_raw,
    #     ROOT / "outputs" / "fig_paper6_removal_stats.pdf"
    # )

    print("\nDone. All outputs in results/")
