// Cauchy
parameters {
  vector[40] beta;
}
model {
  beta ~ cauchy(0, 1);
}
