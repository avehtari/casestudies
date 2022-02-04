// generated with brms 2.14.5
functions {
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  int Kc = K - 1;
  matrix[N, Kc] X0 = X[,2:K];  // X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
  }
}
parameters {
  vector[Kc] b;  // population-level effects
  real<offset=-dot_product(means_X, b)> b_Intercept;  // temporary intercept for centered predictors
  real<lower=0> sigma;  // residual SD
}
transformed parameters {
  real Intercept = b_Intercept + dot_product(means_X, b);
}
model {
  // likelihood including all constants
  if (!prior_only) {
    target += normal_id_glm_lpdf(Y | X0, b_Intercept, b, sigma);
  }
  // priors including all constants
  target += student_t_lpdf(Intercept | 3, 9.4, 2.5);
  target += student_t_lpdf(sigma | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
}
