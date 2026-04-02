// Binomial model with beta(1,1) prior
data {
  int<lower=0> N;              // number of experiments
  int<lower=0> y;              // number of successes
}
parameters {
  real<lower=0,upper=1> theta; // probability of success in range (0,1)
}
model {
  theta ~ beta(1, 1);          // prior
  y ~ binomial(N, theta);      // data model
}
