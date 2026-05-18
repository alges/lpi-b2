import os as _os
from .stan_utils import ensure_cmdstan_installed
ensure_cmdstan_installed()

from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y
from sklearn.utils import resample

_DEFAULT_STAN = _os.path.join(_os.path.dirname(__file__), "models", "lpi_b2_bootstrap.stan")


class LPIB2Evaluator(BaseEstimator):
    """LPI-B² evaluator: bootstrap training + Bayesian latent-truth inference.

    Parameters
    ----------
    base_estimator : sklearn estimator
        Binary classifier to bootstrap.
    n_bootstrap : int
        Number of bootstrap replicates (B).
    anchor_rate : float
        Fraction of samples whose labels are revealed to the Stan model as
        anchors (0 < anchor_rate <= 1).
    stan_file : str
        Path to the Stan model file. Defaults to the bundled
        lpi_b2_bootstrap.stan inside this package.
    stan_chains, stan_chains_size, stan_chains_warmup : int
        MCMC configuration passed to CmdStanPy.
    """

    def __init__(
        self,
        base_estimator,
        n_bootstrap=15,
        anchor_rate=0.5,
        stan_file=_DEFAULT_STAN,
        stan_chains=4,
        stan_chains_size=1000,
        stan_chains_warmup=250,
    ):
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.anchor_rate = anchor_rate
        self.stan_file = stan_file
        self.stan_chains = stan_chains
        self.stan_chains_size = stan_chains_size
        self.stan_chains_warmup = stan_chains_warmup

    def evaluate(self, X, y):
        """Run bootstrap training and Bayesian inference.

        Populates ``self.mcmc_fit_``, ``self.Q_matrix_``, and
        ``self.stan_data_``.
        """
        X, y = check_X_y(X, y)
        N, B = X.shape[0], self.n_bootstrap
        self.Q_matrix_ = np.zeros((N, B))
        self.models_ = []

        for j in range(B):
            X_b, y_b = resample(X, y, random_state=j)
            model = clone(self.base_estimator)
            model.fit(X_b, y_b)
            self.models_.append(model)

            if hasattr(model, "predict_proba"):
                self.Q_matrix_[:, j] = model.predict_proba(X)[:, 1]
            else:
                self.Q_matrix_[:, j] = model.predict(X)

        # clip to valid Beta domain
        self.Q_matrix_ = np.clip(self.Q_matrix_, 1e-4, 1 - 1e-4)

        y_anchored = y.astype(int).copy()
        if self.anchor_rate < 1.0:
            n_anchors = int(N * self.anchor_rate)
            non_anchor_indices = np.random.choice(
                np.arange(N), size=(N - n_anchors), replace=False
            )
            y_anchored[non_anchor_indices] = -1

        self.stan_data_ = {
            "N": N,
            "B": B,
            "Q": self.Q_matrix_,
            "GS": y_anchored,  # 0 / 1 for anchors, -1 for latent
        }

        stan_model = CmdStanModel(
            stan_file=self.stan_file, cpp_options={"STAN_THREADS": True}
        )
        self.mcmc_fit_ = stan_model.sample(
            data=self.stan_data_,
            iter_sampling=self.stan_chains_size,
            iter_warmup=self.stan_chains_warmup,
            chains=self.stan_chains,
            parallel_chains=self.stan_chains,
            threads_per_chain=16,
            seed=31032026,
        )
        return self

    def _check_if_evaluated(self):
        if not hasattr(self, "mcmc_fit_"):
            raise RuntimeError(
                "The evaluator has not been run yet. Call .evaluate(X, y) first."
            )

    def get_global_performance(self):
        """Return posterior mean of the five structural parameters."""
        self._check_if_evaluated()
        vars_ = ["mu_Se", "mu_Sp", "kappa_Se", "kappa_Sp", "kappa_obs"]
        return {v: np.mean(self.mcmc_fit_.stan_variable(v)) for v in vars_}

    def get_sample_audit(self):
        """Return per-sample posterior truth probability and ambiguity score."""
        self._check_if_evaluated()
        p_truth = np.mean(self.mcmc_fit_.stan_variable("prob_Ti_pos"), axis=0)
        ambiguity = 1 - 2 * np.abs(p_truth - 0.5)
        return pd.DataFrame(
            {"posterior_truth_prob": p_truth, "ambiguity_score": ambiguity}
        )


def get_label_noise_audit(evaluator, y_noisy, tau=0.8):
    """Characterise label noise by comparing provided labels to latent truth.

    Returns a DataFrame with per-sample scores and the Global Corruption Rate.
    """
    p_truth = np.mean(evaluator.mcmc_fit_.stan_variable("prob_Ti_pos"), axis=0)
    discordance = np.abs(y_noisy - p_truth)
    is_corrupted = discordance > tau
    gcr = np.mean(is_corrupted)
    audit_df = pd.DataFrame(
        {
            "provided_label": y_noisy,
            "posterior_truth_prob": p_truth,
            "discordance_score": discordance,
            "is_potential_noise": is_corrupted,
        }
    )
    return audit_df, gcr


def optimize_noise_threshold(evaluator, y_noisy, true_noise_level):
    """Sweep tau values and return the one minimising estimation error."""
    thresholds = np.linspace(0.3, 0.9, 20)
    p_truth = np.mean(evaluator.mcmc_fit_.stan_variable("prob_Ti_pos"), axis=0)
    discordance = np.abs(y_noisy - p_truth)
    audit_results = [
        {"threshold": tau, "est_noise": np.mean(discordance > tau),
         "error": abs(np.mean(discordance > tau) - true_noise_level)}
        for tau in thresholds
    ]
    sweep_df = pd.DataFrame(audit_results)
    best_tau = sweep_df.loc[sweep_df["error"].idxmin(), "threshold"]
    return sweep_df, best_tau


def get_entropy_audit(evaluator, tau_entropy=0.8):
    """Return per-sample Shannon entropy of the latent truth posterior."""
    p = np.mean(evaluator.mcmc_fit_.stan_variable("prob_Ti_pos"), axis=0)
    eps = 1e-9
    entropy = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))
    return pd.DataFrame(
        {
            "posterior_prob": p,
            "entropy": entropy,
            "is_high_entropy": entropy > tau_entropy,
        }
    )
