functions {
#include gpbasisfun_functions.stan
}
data {
  int<lower=1> N;      // number of observations
  vector[N] x;         // univariate covariate
  vector[N] y;         // target variable
  array[N] int day_of_week;  // 
        
  real<lower=0> c_f1;  // factor c to determine the boundary value L
  int<lower=1> M_f1;   // number of basis functions for smooth function
  int<lower=1> J_f2;   // number of cos and sin functions for periodic
  real<lower=0> c_g3;  // factor c to determine the boundary value L
  int<lower=1> M_g3;   // number of basis functions for smooth function
}
transformed data {
  // Normalize data
  real xmean = mean(x);
  real ymean = mean(y);
  real xsd = sd(x);
  real ysd = sd(y);
  vector[N] xn = (x - xmean)/xsd;
  vector[N] yn = (y - ymean)/ysd;
  real xmax = max(x);
  vector[N] x1 = x/xmax;
  // Basis functions for f1
  real L_f1 = c_f1*max(xn);
  matrix[N,M_f1] PHI_f1 = PHI(N, M_f1, L_f1, xn);
  // Basis functions for f2
  real period_year = 365.25/xsd;
  matrix[N,2*J_f2] PHI_f2 = PHI_periodic(N, J_f2, 2*pi()/period_year, xn);
  // Basis functions for g3
  real L_g3= c_g3*max(xn);
  matrix[N,M_g3] PHI_g3 = PHI(N, M_g3, L_g3, xn);
  // Concatenated basis functions for f1 and f2
  matrix[N,M_f1+2*J_f2] PHI_f = append_col(PHI_f1, PHI_f2);
}
parameters {
  vector[M_f1] beta_f1;         // the basis functions coefficients for f1
  vector[2*J_f2] beta_f2;       // the basis functions coefficients for f2
  vector[6] beta_f3;            // day of week effect
  vector[M_g3] beta_g3;         // the basis functions coefficients for g3
  real<lower=0> lengthscale_f1; //
  real<lower=0> lengthscale_f2; //
  real<lower=0> lengthscale_g3; //
  real<lower=0> sigma_f1;       // scale of f1
  real<lower=0> sigma_f2;       // scale of f2
  real<lower=0> sigma_g3;       // scale of g3
  real<lower=0> sigma;          // residual scale
}
model {
  // spectral densities for f1, f2, and g3
  vector[M_f1] diagSPD_f1 = diagSPD_EQ(sigma_f1, lengthscale_f1, L_f1, M_f1);
  vector[2*J_f2] diagSPD_f2 = diagSPD_periodic(sigma_f2, lengthscale_f2, J_f2);
  vector[M_g3] diagSPD_g3 = diagSPD_EQ(sigma_g3, lengthscale_g3, L_g3, M_g3);
  // day of week effect with increasing magnitude (Monday set to 0)
  vector[7] f_day_of_week = append_row(0, beta_f3);
  vector[N] g3 = PHI_g3 * (diagSPD_g3 .* beta_g3);
  vector[N] intercept = 0.0 + exp(g3).*f_day_of_week[day_of_week];
  // priors
  beta_f1 ~ normal(0, 1);
  beta_f2 ~ normal(0, 1);
  beta_f3 ~ normal(0, 1);
  beta_g3 ~ normal(0, 1);
  lengthscale_f1 ~ lognormal(log(700/xsd), 1);
  lengthscale_f2 ~ normal(0, .1);
  lengthscale_g3 ~ lognormal(log(7000/xsd), 1);
  sigma_f1 ~ normal(0, 1);
  sigma_f2 ~ normal(0, 1);
  sigma_g3 ~ normal(0, 0.1);
  sigma ~ normal(0, 0.5);
  // model
  yn ~ normal_id_glm(PHI_f,
		     intercept,
		     append_row(diagSPD_f1 .* beta_f1, diagSPD_f2 .* beta_f2),
		     sigma);
}
generated quantities {
  vector[N] f1;
  vector[N] f2;
  vector[N] f3;
  vector[N] f;
  vector[7] f_day_of_week = append_row(0, beta_f3);
  vector[N] log_lik;
  {
    // spectral densities
    vector[M_f1] diagSPD_f1 = diagSPD_EQ(sigma_f1, lengthscale_f1, L_f1, M_f1);
    vector[2*J_f2] diagSPD_f2 = diagSPD_periodic(sigma_f2, lengthscale_f2, J_f2);
    vector[M_g3] diagSPD_g3 = diagSPD_EQ(sigma_g3, lengthscale_g3, L_g3, M_g3);
    // day of week effect with increasing magnitude (Monday set to 0)
    vector[N] g3 = PHI_g3 * (diagSPD_g3 .* beta_g3);
    // functions scaled back to original scale
    f1 = (0.0 + PHI_f1 * (diagSPD_f1 .* beta_f1))*ysd;
    f2 = (PHI_f2 * (diagSPD_f2 .* beta_f2))*ysd;
    f3 = exp(g3).*f_day_of_week[day_of_week]*ysd;
    f_day_of_week = append_row(0, beta_f3)*ysd;
    f = f1 + f2 + f3 + ymean;
    // log_liks for loo
    for (n in 1:N) log_lik[n] = normal_lpdf(y[n] | f[n], sigma*ysd);
  }
}
