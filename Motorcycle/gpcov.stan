/* functions { */
/* #include gpbasisfun_functions.stan */
/* } */
data {
  int<lower=1> N;      // number of observations
  vector[N] x;         // univariate covariate
  vector[N] y;         // target variable
}
transformed data {
  // Normalize data
  real xmean = mean(x);
  real ymean = mean(y);
  real xsd = sd(x);
  real ysd = sd(y);
  real xn[N] = to_array_1d((x - xmean)/xsd);
  vector[N] yn = (y - ymean)/ysd;
}
parameters {
  real intercept;              // 
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f
  real<lower=0> lengthscale_g; // lengthscale of f
  real<lower=0> sigma_g;       // scale of f
  vector[N] z_f;
  vector[N] z_g;
}
model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, rep_vector(1e-6, N)));
  matrix[N, N] K_g = gp_exp_quad_cov(xn, sigma_g, lengthscale_g);
  matrix[N, N] L_g = cholesky_decompose(add_diag(K_g, rep_vector(1e-6, N)));
  // priors
  intercept ~ normal(0, 1);
  z_f ~ std_normal();
  z_g ~ std_normal();
  lengthscale_f ~ lognormal(log(.3), .2);
  lengthscale_g ~ lognormal(log(.5), .2);
  sigma_f ~ normal(0, .5);
  sigma_g ~ normal(0, .5);
  // model
  yn ~ normal(intercept + L_f * z_f, exp(L_g * z_g));
}
generated quantities {
  vector[N] f;
  vector[N] sigma;
  {
    // covariances and Cholesky decompositions
    matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
    matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, rep_vector(1e-6, N)));
    matrix[N, N] K_g = gp_exp_quad_cov(xn, sigma_g, lengthscale_g);
    matrix[N, N] L_g = cholesky_decompose(add_diag(K_g, rep_vector(1e-6, N)));
    // function scaled back to original scale
    f = (intercept + L_f * z_f)*ysd + ymean;
    sigma = exp(L_g * z_g)*ysd;
  }
}
