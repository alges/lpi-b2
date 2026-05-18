"""
exp_cleanlab_comparison.py
Compares LPI-B2-LR noise estimation against Cleanlab across a 3x4x6 grid
(class_sep × flip_y × seed = 72 conditions).

Usage:
  python exp_cleanlab_comparison.py --dry-run    # 1 condition, ~minutes
  python exp_cleanlab_comparison.py              # full grid, ~hours (~2h with BTD re-runs)

Output files (all in POSTPROC = ROOT/latex/results/postproc/):
  exp_cleanlab_results.csv          incremental, idempotent
  exp_cleanlab_status.md            overwritten each run
  NEWEXP_PROGRESS.md             append-only log

Figure (in FIGURES = ROOT/latex/results/figures/):
  newexp_01_comparison.pdf

Requirements (beyond the project's existing dependencies):
  pip install cleanlab>=2.5
"""

# ---------------------------------------------------------------------------
# Stdlib / third-party imports
# ---------------------------------------------------------------------------
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, jaccard_score,
)

# ---------------------------------------------------------------------------
# Cleanlab import — fail early with a helpful message
# ---------------------------------------------------------------------------
try:
    import cleanlab  # noqa: F401
    from cleanlab.filter import find_label_issues
    try:
        from cleanlab.count import num_label_issues as _cl_count_issues
        _CLEANLAB_COUNT_API = "num_label_issues"
    except ImportError:
        _cl_count_issues = None
        _CLEANLAB_COUNT_API = "unavailable"
    _CLEANLAB_OK = True
except ImportError:
    print(
        "\n[exp_cleanlab_comparison] ERROR: cleanlab is not installed.\n"
        "  Install with:  pip install cleanlab>=2.5\n"
        "  Cleanlab docs: https://docs.cleanlab.ai/\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lpi_b2.estimator import LPIB2Evaluator, get_label_noise_audit  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TABLES   = ROOT / "data"
FIGURES  = ROOT / "outputs"
POSTPROC = ROOT / "data"

TABLE5_CSV        = TABLES / "table5_noise_estimators_v2_LR_B50.csv"
SAMPLE_POST_CSV   = TABLES / "sample_posteriors_yc.csv"
RESULTS_CSV       = POSTPROC / "exp_cleanlab_results.csv"
STATUS_MD         = POSTPROC / "exp_cleanlab_status.md"
PROGRESS_MD       = POSTPROC / "NEWEXP_PROGRESS.md"
FIGURE_PATH       = FIGURES  / "newexp_01_comparison.pdf"

TAG = "[exp_cleanlab_comparison]"

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
CLASS_SEPS = [1.0, 1.5, 2.0]
FLIP_YS    = [0.10, 0.15, 0.20, 0.30]
SEEDS      = [0, 1, 2, 3, 4, 5]

# Colour / marker maps for figure
COLOR_MAP  = {"LPI-B2":      "#1f77b4", "cleanlab": "#ff7f0e"}
MARKER_MAP = {1.0: "o",  1.5: "s",  2.0: "^"}

# Result columns
RESULT_COLS = [
    "class_sep", "flip_y", "seed",
    "phi_true",
    "phi_btd_lr", "phi_cl",
    "bias_btd",   "bias_cl",
    "btd_precision", "btd_recall", "btd_f1", "btd_auc",
    "cl_precision",  "cl_recall",  "cl_f1",  "cl_auc",
    "jaccard_btd_cl",
    "btd_rhat_max", "btd_neff_min", "btd_converged",
    "btd_source",   # "cache_table5" | "cache_posteriors" | "fresh"
]

# ---------------------------------------------------------------------------
# Data generation (matches exp_multiarch_capacity.py make_data exactly)
# ---------------------------------------------------------------------------

def make_data(n: int = 1000, sep: float = 1.5, flip: float = 0.0, seed: int = 42):
    """Generate (X, y_noisy, y_clean) using the same logic as experiments_3_multi_arch."""
    X, y_clean = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_repeated=0,
        class_sep=sep,
        flip_y=0.0,
        random_state=seed,
    )
    if flip > 0.0:
        rng = np.random.default_rng(seed + 10_000)
        mask = rng.random(n) < flip
        y_noisy = y_clean.copy()
        y_noisy[mask] = 1 - y_noisy[mask]
    else:
        y_noisy = y_clean.copy()
    return X, y_noisy, y_clean


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def check_convergence(mcmc_fit, params=("mu_Se", "mu_Sp", "kappa_Se", "kappa_Sp")):
    """
    Returns (converged: bool, rhat_max: float, neff_min: float).
    Considers converged if R_hat < 1.05 and N_Eff > 100 for all listed params.
    """
    try:
        summ = mcmc_fit.summary()
        pop  = summ[summ.index.isin(list(params))]
        rhat_max = float(pop["R_hat"].max())
        neff_min = float(pop["N_Eff"].min())
        return rhat_max < 1.05 and neff_min > 100, rhat_max, neff_min
    except Exception:
        return False, float("nan"), float("nan")


# ---------------------------------------------------------------------------
# LPI-B2-LR fresh run
# ---------------------------------------------------------------------------

def run_btd_lr_fresh(X, y_noisy):
    """Run LPI-B2-LR with n_bootstrap=35, anchor_rate=0.8 (LR base estimator)."""
    ev = LPIB2Evaluator(
        LogisticRegression(max_iter=1000),
        n_bootstrap=35,
        anchor_rate=0.8,
    )
    ev.evaluate(X, y_noisy)
    return ev


# ---------------------------------------------------------------------------
# Cleanlab helpers
# ---------------------------------------------------------------------------

def run_cleanlab(X, y_noisy):
    """
    Returns (issue_mask: np.ndarray[bool], pred_probs: np.ndarray).

    Uses cleanlab v2.5+ API:
      find_label_issues(labels, pred_probs, return_indices_ranked_by='self_confidence')
    Returns a boolean mask of the same length as y_noisy.

    Falls back to an alternative if the API has changed between versions.
    """
    lr = LogisticRegression(max_iter=1000)

    # Cross-validated predicted probabilities (required by cleanlab)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_probs = cross_val_predict(lr, X, y_noisy, cv=cv, method="predict_proba")

    # Ensure probabilities are valid
    pred_probs = np.clip(pred_probs, 1e-6, 1 - 1e-6)

    try:
        # Primary API (cleanlab >= 2.5)
        issue_indices = find_label_issues(
            labels=y_noisy,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
        )
        issue_mask = np.zeros(len(y_noisy), dtype=bool)
        issue_mask[issue_indices] = True
    except TypeError:
        # Alternative: older versions may not accept return_indices_ranked_by
        # or may return a boolean array directly
        try:
            result = find_label_issues(labels=y_noisy, pred_probs=pred_probs)
            if isinstance(result, np.ndarray) and result.dtype == bool:
                issue_mask = result
            else:
                # Treat as indices
                issue_mask = np.zeros(len(y_noisy), dtype=bool)
                issue_mask[result] = True
        except Exception as exc:
            raise RuntimeError(
                f"cleanlab.filter.find_label_issues() failed with: {exc}\n"
                "Please check the cleanlab API for your installed version."
            ) from exc

    return issue_mask, pred_probs


def phi_cl_estimate(issue_mask):
    """Global noise rate estimate from cleanlab: fraction of flagged samples."""
    return float(np.mean(issue_mask))


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

def _load_table5():
    """Load table5_noise_estimators_v2_LR_B50.csv once; return None if missing."""
    if not TABLE5_CSV.exists():
        print(f"{TAG} WARN: {TABLE5_CSV} not found — BTD results will always run fresh.")
        return None
    df = pd.read_csv(TABLE5_CSV)
    # Normalise types for lookup
    df["class_sep"] = df["class_sep"].astype(float)
    df["flip_y"]    = df["flip_y"].astype(float)
    df["seed"]      = df["seed"].astype(int)
    return df


def _load_sample_posteriors():
    """Load sample_posteriors_yc.csv once; return None if missing."""
    if not SAMPLE_POST_CSV.exists():
        print(f"{TAG} WARN: {SAMPLE_POST_CSV} not found — per-sample posteriors always fresh.")
        return None
    df = pd.read_csv(SAMPLE_POST_CSV)
    df["flip_y"] = df["flip_y"].astype(float)
    df["seed"]   = df["seed"].astype(int)
    # This file only covers class_sep=1.0; there is no class_sep column.
    df["class_sep"] = 1.0
    return df


# ---------------------------------------------------------------------------
# Per-condition computation
# ---------------------------------------------------------------------------

def compute_condition(
    class_sep: float,
    flip_y: float,
    seed: int,
    table5_df,
    sample_post_df,
) -> dict:
    """
    Run one condition (class_sep, flip_y, seed).
    Returns a dict with all RESULT_COLS values.
    """
    X, y_noisy, y_clean = make_data(n=1000, sep=class_sep, flip=flip_y, seed=seed)
    true_flip_mask = (y_clean != y_noisy)
    phi_true = float(np.mean(true_flip_mask))

    # ── LPI-B2-LR: try caches first ────────────────────────────────────────────
    phi_btd_lr    = None
    btd_mask      = None
    btd_converged = False
    btd_rhat_max  = float("nan")
    btd_neff_min  = float("nan")
    btd_source    = "fresh"

    # Cache 1: table5 (global phi only — no per-sample data)
    if table5_df is not None:
        row = table5_df[
            (table5_df["class_sep"] == class_sep)
            & (np.isclose(table5_df["flip_y"], flip_y))
            & (table5_df["seed"] == seed)
        ]
        if not row.empty:
            phi_btd_lr = float(row.iloc[0]["E1"])
            btd_source = "cache_table5"

    # Cache 2: sample_posteriors_yc (per-sample, only class_sep=1.0)
    if sample_post_df is not None and class_sep == 1.0:
        sp = sample_post_df[
            (np.isclose(sample_post_df["flip_y"], flip_y))
            & (sample_post_df["seed"] == seed)
            & (sample_post_df["arch"] == "LR")
        ].sort_values("sample_idx")
        if not sp.empty and len(sp) == len(y_noisy):
            p_truth = sp["p_truth"].values
            btd_mask = (np.abs(y_noisy - p_truth) > 0.8)
            if phi_btd_lr is None:
                phi_btd_lr = float(np.mean(btd_mask))
            btd_converged = True          # cached posterior assumed converged
            btd_source = "cache_posteriors"

    # Fallback: run LPI-B2-LR fresh
    if phi_btd_lr is None or btd_mask is None:
        print(
            f"{TAG} WARN: LPI-B2-LR not cached for "
            f"(class_sep={class_sep}, flip_y={flip_y}, seed={seed}) — running fresh."
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = run_btd_lr_fresh(X, y_noisy)

        btd_converged, btd_rhat_max, btd_neff_min = check_convergence(ev.mcmc_fit_)

        p_truth = np.mean(ev.mcmc_fit_.stan_variable("prob_Ti_pos"), axis=0)
        btd_mask = (np.abs(y_noisy - p_truth) > 0.8)
        phi_btd_lr = float(np.mean(btd_mask))
        btd_source = "fresh"

    # ── Cleanlab (always fresh) ──────────────────────────────────────────────
    cl_mask, _ = run_cleanlab(X, y_noisy)
    phi_cl = phi_cl_estimate(cl_mask)

    # ── Metrics ─────────────────────────────────────────────────────────────
    def _safe_metrics(pred_mask):
        """Precision/Recall/F1/AUC against true_flip_mask."""
        if pred_mask.sum() == 0:
            return 0.0, 0.0, 0.0, 0.5
        prec = precision_score(true_flip_mask, pred_mask, zero_division=0)
        rec  = recall_score(true_flip_mask, pred_mask, zero_division=0)
        f1   = f1_score(true_flip_mask, pred_mask, zero_division=0)
        try:
            auc = roc_auc_score(true_flip_mask, pred_mask.astype(float))
        except ValueError:
            auc = float("nan")
        return float(prec), float(rec), float(f1), float(auc)

    btd_prec, btd_rec, btd_f1, btd_auc = _safe_metrics(btd_mask)
    cl_prec,  cl_rec,  cl_f1,  cl_auc  = _safe_metrics(cl_mask)

    # Jaccard similarity between the two flagged sets
    try:
        jaccard = float(jaccard_score(btd_mask, cl_mask))
    except ValueError:
        jaccard = float("nan")

    return {
        "class_sep":      class_sep,
        "flip_y":         flip_y,
        "seed":           seed,
        "phi_true":       phi_true,
        "phi_btd_lr":     phi_btd_lr,
        "phi_cl":         phi_cl,
        "bias_btd":       phi_btd_lr - phi_true,
        "bias_cl":        phi_cl     - phi_true,
        "btd_precision":  btd_prec,
        "btd_recall":     btd_rec,
        "btd_f1":         btd_f1,
        "btd_auc":        btd_auc,
        "cl_precision":   cl_prec,
        "cl_recall":      cl_rec,
        "cl_f1":          cl_f1,
        "cl_auc":         cl_auc,
        "jaccard_btd_cl": jaccard,
        "btd_rhat_max":   btd_rhat_max,
        "btd_neff_min":   btd_neff_min,
        "btd_converged":  btd_converged,
        "btd_source":     btd_source,
    }


# ---------------------------------------------------------------------------
# Incremental CSV helpers
# ---------------------------------------------------------------------------

def _load_existing_results() -> pd.DataFrame:
    """Load results CSV if it exists; else return empty DataFrame."""
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=RESULT_COLS)


def _row_exists(df: pd.DataFrame, class_sep: float, flip_y: float, seed: int) -> bool:
    if df.empty:
        return False
    return bool(
        (
            (df["class_sep"] == class_sep)
            & (np.isclose(df["flip_y"], flip_y))
            & (df["seed"] == seed)
        ).any()
    )


def _append_row(row: dict) -> None:
    """Append one row to the results CSV (creates file + header if needed)."""
    write_header = not RESULTS_CSV.exists()
    new_df = pd.DataFrame([row], columns=RESULT_COLS)
    new_df.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame) -> None:
    """
    Three-panel comparison figure.

    Panel A: scatter phi_true vs phi_hat (BTD and cleanlab), marker by class_sep,
             colour by method, diagonal reference line.
    Panel B: F1 by method, grouped bar by flip_y, mean±std over seeds × class_sep.
    Panel C: Bias vs phi_true for both methods with ±1 sigma bands.
    """
    if df.empty:
        print(f"{TAG} Figure skipped — no results yet.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("LPI-B2-LR vs. Cleanlab: noise estimation comparison", fontsize=13)

    # ── Panel A: scatter phi_true vs phi_hat ────────────────────────────────
    ax = axes[0]
    ax.set_title("A — Global noise rate estimation")
    lims = [0, df[["phi_btd_lr", "phi_cl", "phi_true"]].max().max() * 1.1]
    ax.plot(lims, lims, color="grey", lw=1, ls="--", zorder=0, label="Ideal")

    for method, col, color in [("LPI-B2", "phi_btd_lr", COLOR_MAP["LPI-B2"]),
                                ("cleanlab", "phi_cl", COLOR_MAP["cleanlab"])]:
        for sep, marker in MARKER_MAP.items():
            sub = df[np.isclose(df["class_sep"], sep)]
            if sub.empty:
                continue
            ax.scatter(
                sub["phi_true"], sub[col],
                color=color, marker=marker,
                s=40, alpha=0.7,
                label=f"{method} sep={sep}" if color == COLOR_MAP["LPI-B2"] else "_nolegend_",
            )

    # Build a compact legend: two colour patches + three marker patches
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP["LPI-B2"],
               markersize=8, label="LPI-B2-LR"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP["cleanlab"],
               markersize=8, label="cleanlab"),
        Line2D([0], [0], marker="o", color="grey", markersize=8, label="sep=1.0"),
        Line2D([0], [0], marker="s", color="grey", markersize=8, label="sep=1.5"),
        Line2D([0], [0], marker="^", color="grey", markersize=8, label="sep=2.0"),
        Line2D([0], [0], color="grey", lw=1, ls="--", label="Ideal"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left")
    ax.set_xlabel(r"$\phi_{\rm true}$")
    ax.set_ylabel(r"$\hat\phi$")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # ── Panel B: grouped bar F1 by flip_y ──────────────────────────────────
    ax = axes[1]
    ax.set_title("B — F1 score by flip rate")
    flip_vals = sorted(df["flip_y"].unique())
    x = np.arange(len(flip_vals))
    width = 0.35

    btd_means, btd_stds = [], []
    cl_means,  cl_stds  = [], []
    for fy in flip_vals:
        sub = df[np.isclose(df["flip_y"], fy)]
        btd_means.append(sub["btd_f1"].mean())
        btd_stds.append(sub["btd_f1"].std())
        cl_means.append(sub["cl_f1"].mean())
        cl_stds.append(sub["cl_f1"].std())

    ax.bar(x - width / 2, btd_means, width, yerr=btd_stds,
           color=COLOR_MAP["LPI-B2"],      label="LPI-B2-LR",   capsize=4, alpha=0.85)
    ax.bar(x + width / 2, cl_means,  width, yerr=cl_stds,
           color=COLOR_MAP["cleanlab"], label="cleanlab", capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fy:.2f}" for fy in flip_vals])
    ax.set_xlabel(r"$\phi_{\rm true}$ (flip rate)")
    ax.set_ylabel("F1 (mean ± std over seeds × class_sep)")
    ax.set_ylim(0, 1)
    ax.legend()

    # ── Panel C: bias vs phi_true with ±1 sigma bands ───────────────────────
    ax = axes[2]
    ax.set_title("C — Bias vs. true noise rate")
    ax.axhline(0, color="grey", lw=1, ls="--")

    for method, bias_col, color in [("LPI-B2-LR",   "bias_btd", COLOR_MAP["LPI-B2"]),
                                     ("cleanlab", "bias_cl",  COLOR_MAP["cleanlab"])]:
        grouped = df.groupby("phi_true")[bias_col]
        phi_vals = sorted(df["phi_true"].unique())
        means = [grouped.get_group(p).mean() if p in grouped.groups else np.nan
                 for p in phi_vals]
        stds  = [grouped.get_group(p).std()  if p in grouped.groups else np.nan
                 for p in phi_vals]
        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)

        ax.plot(phi_vals, means, color=color, lw=2, label=method)
        ax.fill_between(phi_vals,
                        means - stds,
                        means + stds,
                        color=color, alpha=0.2)

    ax.set_xlabel(r"$\phi_{\rm true}$")
    ax.set_ylabel(r"$\hat\phi - \phi_{\rm true}$ (bias)")
    ax.legend()

    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"{TAG} Figure saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Status / progress writers
# ---------------------------------------------------------------------------

def _write_status(
    n_done: int,
    n_total: int,
    n_fresh: int,
    n_cached: int,
    start_ts: str,
    errors: list[str],
) -> None:
    """Overwrite exp_cleanlab_status.md with current run summary."""
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# newexp_01 run status",
        "",
        f"- Run started : {start_ts}",
        f"- Last update : {now}",
        f"- Conditions  : {n_done} / {n_total} completed",
        f"- BTD source  : {n_cached} cached, {n_fresh} run fresh",
        f"- Errors      : {len(errors)}",
        "",
    ]
    if errors:
        lines += ["## Errors", ""]
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")
    STATUS_MD.write_text("\n".join(lines), encoding="utf-8")


def _append_progress(msg: str) -> None:
    """Append one line to NEWEXP_PROGRESS.md."""
    with PROGRESS_MD.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only 1 condition (first cell of the grid) for a quick smoke test.",
    )
    args = parser.parse_args()

    start_ts = datetime.now().isoformat(timespec="seconds")
    print(f"{TAG} Started at {start_ts}")
    print(f"{TAG} cleanlab API: find_label_issues — count API: {_CLEANLAB_COUNT_API}")

    # Build full grid
    conditions = [
        (sep, fy, seed)
        for sep in CLASS_SEPS
        for fy  in FLIP_YS
        for seed in SEEDS
    ]
    if args.dry_run:
        conditions = conditions[:1]
        print(f"{TAG} --dry-run: only 1 condition will be processed.")

    n_total  = len(conditions)
    n_done   = 0
    n_fresh  = 0
    n_cached = 0
    errors: list[str] = []

    # Load caches
    table5_df      = _load_table5()
    sample_post_df = _load_sample_posteriors()

    # Load existing results (for idempotency)
    existing = _load_existing_results()
    print(f"{TAG} Grid: {n_total} conditions. Already done: {len(existing)}.")

    _append_progress(f"\n## Run {start_ts}  (dry_run={args.dry_run})")

    for idx, (sep, fy, seed) in enumerate(conditions, 1):
        # Idempotency check
        if _row_exists(existing, sep, fy, seed):
            print(f"{TAG} [{idx}/{n_total}] SKIP (already computed): sep={sep} flip={fy} seed={seed}")
            n_done += 1
            continue

        print(f"{TAG} [{idx}/{n_total}] Computing: class_sep={sep} flip_y={fy} seed={seed}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = compute_condition(sep, fy, seed, table5_df, sample_post_df)
        except Exception as exc:
            err_msg = f"sep={sep} flip={fy} seed={seed}: {exc}"
            print(f"{TAG} ERROR: {err_msg}", file=sys.stderr)
            errors.append(err_msg)
            _append_progress(f"  ERROR: {err_msg}")
            _write_status(n_done, n_total, n_fresh, n_cached, start_ts, errors)
            continue

        # Track source stats
        if row["btd_source"] == "fresh":
            n_fresh += 1
        else:
            n_cached += 1

        # Append to CSV immediately (incremental)
        _append_row(row)
        # Also update in-memory existing to prevent duplicate appends in same run
        existing = pd.concat([existing, pd.DataFrame([row], columns=RESULT_COLS)],
                              ignore_index=True)

        n_done += 1
        _append_progress(
            f"  OK sep={sep} flip={fy} seed={seed} "
            f"phi_true={row['phi_true']:.3f} "
            f"phi_btd={row['phi_btd_lr']:.3f} "
            f"phi_cl={row['phi_cl']:.3f} "
            f"src={row['btd_source']}"
        )

        print(
            f"{TAG}   phi_true={row['phi_true']:.3f}  "
            f"phi_btd={row['phi_btd_lr']:.3f}  phi_cl={row['phi_cl']:.3f}  "
            f"btd_f1={row['btd_f1']:.3f}  cl_f1={row['cl_f1']:.3f}  "
            f"src={row['btd_source']}"
        )

        # Update status after each row
        _write_status(n_done, n_total, n_fresh, n_cached, start_ts, errors)

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{TAG} === Done === {n_done}/{n_total} conditions, {n_fresh} fresh BTD runs.")
    if errors:
        print(f"{TAG} {len(errors)} error(s) — see {STATUS_MD}")

    # Reload full results for figure
    if RESULTS_CSV.exists():
        df_all = pd.read_csv(RESULTS_CSV)
        print(f"{TAG} Results CSV: {len(df_all)} rows  →  {RESULTS_CSV}")
        make_figure(df_all)
    else:
        print(f"{TAG} No results CSV found — figure skipped.")

    _write_status(n_done, n_total, n_fresh, n_cached, start_ts, errors)
    _append_progress(
        f"  FINISHED: {n_done}/{n_total} done, {n_fresh} fresh, {len(errors)} errors"
    )

    end_ts = datetime.now().isoformat(timespec="seconds")
    print(f"{TAG} Finished at {end_ts}")


if __name__ == "__main__":
    main()
