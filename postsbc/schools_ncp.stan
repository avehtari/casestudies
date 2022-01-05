data {
  int <lower=0> J; // number of schools
  real y[J]; // estimated treatment
  real<lower=0> sigma[J]; // std of estimated effect
}
parameters {
  vector[J] theta_trans; // transformation of theta
  real mu; // hyper-parameter of mean
  real<lower=0> tau; // hyper-parameter of sd
}
transformed parameters{
  vector[J] theta;
  // original theta
  theta=theta_trans*tau+mu;
}
model {
  target += normal_lpdf(theta_trans | 0, 1);
  target += normal_lpdf(y | theta , sigma);
  target += normal_lpdf(mu | 0, 5); // a non-informative prior
  target += cauchy_lpdf(tau | 0, 5);
}
generated quantities {
  real yrep[J] = normal_rng(theta, sigma);
}
