// Poisson regression
data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> y[N];
  matrix[N,M] x;
}
parameters {
  real alpha;
  vector[M] beta;
}
model {
  alpha ~ normal(0,10);
  beta ~ normal(0,10);
  y ~ poisson_log_glm(x, alpha, beta);
}
