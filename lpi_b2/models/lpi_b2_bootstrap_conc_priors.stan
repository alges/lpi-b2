functions {
  real partial_sum_lpmf(array[] int GS_slice, int start, int end,
                        matrix Q, real p_T,
                        vector Se, vector Sp, real kappa_obs) {
    real lp = 0;
    for (i in 1:size(GS_slice)) {
      int idx = start + i - 1;
      real lp_T1 = bernoulli_lpmf(1 | p_T)
             + beta_lpdf(Q[idx]' | Se * kappa_obs, (1 - Se) * kappa_obs);
      real lp_T0 = bernoulli_lpmf(0 | p_T)
             + beta_lpdf(Q[idx]' | (1 - Sp) * kappa_obs, Sp * kappa_obs);
      if (GS_slice[i] == 1) {
        lp += lp_T1;
      } else if (GS_slice[i] == 0) {
        lp += lp_T0;
      } else {
        lp += log_sum_exp(lp_T1, lp_T0);
      }
    }
    return lp;
  }
}

data {
  int<lower=1> N;
  int<lower=1> B;
  matrix[N, B] Q;
  array[N] int<lower=-1, upper=1> GS;
}

parameters {
  real<lower=0.01, upper=0.99> p_T;
  real<lower=0.1, upper=0.99> mu_Se;
  real<lower=0.1, upper=0.99> mu_Sp;
  real<lower=1, upper=500> kappa_Se;
  real<lower=1, upper=500> kappa_Sp;
  real<lower=1, upper=100> kappa_obs;
  vector<lower=0.01, upper=0.99>[B] Se;
  vector<lower=0.01, upper=0.99>[B] Sp;
}

model {
  p_T      ~ beta(2, 2);
  mu_Se    ~ beta(5, 5);
  mu_Sp    ~ beta(5, 5);
  kappa_Se ~ exponential(0.05);
  kappa_Sp ~ exponential(0.05);
  kappa_obs ~ exponential(0.05);
  Se ~ beta(mu_Se * kappa_Se, (1 - mu_Se) * kappa_Se);
  Sp ~ beta(mu_Sp * kappa_Sp, (1 - mu_Sp) * kappa_Sp);
  target += reduce_sum(partial_sum_lpmf, GS, 1, Q, p_T, Se, Sp, kappa_obs);
}

generated quantities {
  vector[N] prob_Ti_pos;
  for (i in 1:N) {
    real lp_T1 = bernoulli_lpmf(1 | p_T)
             + beta_lpdf(Q[i]' | Se * kappa_obs, (1 - Se) * kappa_obs);
    real lp_T0 = bernoulli_lpmf(0 | p_T)
             + beta_lpdf(Q[i]' | (1 - Sp) * kappa_obs, Sp * kappa_obs);
    prob_Ti_pos[i] = exp(lp_T1 - log_sum_exp(lp_T1, lp_T0));
  }
}