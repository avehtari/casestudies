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
  array[N] real xn = to_array_1d((x - xmean)/xsd);
  vector[N] yn = (y - ymean)/ysd;
  real sigma_intercept = 0.1;
  vector[N] jitter = rep_vector(1e-9, N);
}
parameters {
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f
  real<lower=0> lengthscale_g; // lengthscale of g
  real<lower=0> sigma_g;       // scale of g
  vector[N] z_f;
  vector[N] z_g;
}
model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f)+
                     sigma_intercept;
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));
  matrix[N, N] K_g = gp_exp_quad_cov(xn, sigma_g, lengthscale_g)+
                     sigma_intercept;
  matrix[N, N] L_g = cholesky_decompose(add_diag(K_g, jitter));
  // priors
  z_f ~ std_normal();
  z_g ~ std_normal();
  lengthscale_f ~ lognormal(log(.3), .2);
  lengthscale_g ~ lognormal(log(.5), .2);
  sigma_f ~ normal(0, .5);
  sigma_g ~ normal(0, .5);
  // model
  yn ~ normal(L_f * z_f, exp(L_g * z_g));
}
generated quantities {
  vector[N] f;
  vector[N] sigma;
  {
    // covariances and Cholesky decompositions
    matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f)+
                       sigma_intercept;
    matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));
    matrix[N, N] K_g = gp_exp_quad_cov(xn, sigma_g, lengthscale_g)+
                       sigma_intercept;
    matrix[N, N] L_g = cholesky_decompose(add_diag(K_g, jitter));
    // function scaled back to the original scale
    f = (L_f * z_f)*ysd + ymean;
    sigma = exp(L_g * z_g)*ysd;
  }
}
