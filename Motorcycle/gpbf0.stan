functions {
#include gpbasisfun_functions.stan
}
data {
  int<lower=1> N;      // number of observations
  vector[N] x;         // univariate covariate
        
  real<lower=0> c_f;   // factor c to determine the boundary value L
  int<lower=1> M_f;    // number of basis functions
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f
}
transformed data {
  // Normalize data
  real xmean = mean(x);
  real xsd = sd(x);
  vector[N] xn = (x - xmean)/xsd;
  // Basis functions for f
  real L_f = c_f*max(xn);
}
generated quantities {
  // Basis functions for f
  matrix[N,M_f] PHI_f = PHI(N, M_f, L_f, xn);
  // spectral densities
  vector[M_f] diagSPD_EQ_f = diagSPD_EQ(sigma_f, lengthscale_f, L_f, M_f);
  vector[M_f] diagSPD_Matern32_f = diagSPD_Matern32(sigma_f, lengthscale_f, L_f, M_f);
}
