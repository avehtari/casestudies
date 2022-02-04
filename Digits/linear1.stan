// Gaussian linear model with adjustable priors
data {
  int<lower=0> N; // number of data points
  vector[N] x;    // covariate / predictor
  vector[N] y;    // target
  real pmualpha;  // prior mean for alpha
  real psalpha;   // prior std for alpha
  real pmubeta;   // prior mean for beta
  real psbeta;    // prior std for beta
  real pssigma;   // prior std for half-normal prior for sigma
}
parameters {
  real alpha;          // intercept
  real beta;           // slope
  real<lower=0> sigma; // standard deviation is constrained to be positive
}
model {
  alpha ~ normal(pmualpha, psalpha);  // prior
  beta ~ normal(pmubeta, psbeta);     // prior
  sigma ~ normal(0, pssigma);         // as sigma is constrained to be positive,
                                      // this is same as half-normal prior
  y ~ normal(alpha + beta*x, sigma);  // observation model / likelihood
}
