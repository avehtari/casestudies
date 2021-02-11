functions {
  vector gp_pred_rng(real[] x2,
                     vector y1,
                     real[] x1,
                     real sigma_f,
                     real lengthscale_f,
                     real sigma,
                     real jitter) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] v_pred;
      vector[N2] f2_mu;
      matrix[N2, N2] cov_f2;
      matrix[N1, N1] K;
      K = gp_exp_quad_cov(x1, sigma_f, lengthscale_f);
      for (n in 1:N1)
        K[n, n] = K[n,n] + square(sigma);
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_x2 = gp_exp_quad_cov(x1, x2, sigma_f, lengthscale_f);
      f2_mu = (k_x1_x2' * K_div_y1);
      v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      cov_f2 = gp_exp_quad_cov(x2, sigma_f, lengthscale_f) - v_pred' * v_pred;

      f2 = multi_normal_rng(f2_mu, add_diag(cov_f2, rep_vector(jitter, N2)));
    }
    return f2;
  }
}
data {
  int<lower=1> N;      // number of observations
  vector[N] x;         // univariate covariate
  vector[N] y;         // target variable
  int<lower=1> N2;     // number of test points
  vector[N2] x2;       // univariate test points
}
transformed data {
  // Normalize data
  real xmean = mean(x);
  real ymean = mean(y);
  real xsd = sd(x);
  real ysd = sd(y);
  real xn[N] = to_array_1d((x - xmean)/xsd);
  real x2n[N2] = to_array_1d((x2 - xmean)/xsd);
  vector[N] yn = (y - ymean)/ysd;
  real sigma_intercept = 0.1;
  vector[N] zeros = rep_vector(0, N);
}
parameters {
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f
  real<lower=0> sigman;         // noise sigma
}
model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f)+
                     sigma_intercept;
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, sigman));
  // priors
  lengthscale_f ~ normal(0, 1);
  sigma_f ~ normal(0, 1);
  sigman ~ normal(0, 1);
  // model
  yn ~ multi_normal_cholesky(zeros, L_f);
}
generated quantities {
  // function scaled back to the original scale
  vector[N2] f = gp_pred_rng(x2n, yn, xn, sigma_f, lengthscale_f, sigman, 1e-9)*ysd + ymean;
  real sigma = sigman*ysd;
}
