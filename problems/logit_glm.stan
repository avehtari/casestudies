// logistic regression
data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0,upper=1> y[N];
  matrix[N,M] x;
}
parameters {
  real alpha;
  vector[M] beta;
}
model {
  y ~ bernoulli_logit_glm(x, alpha, beta);
}
