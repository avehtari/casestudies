// Disc golf putting model with 2D angular uncertainty and distance uncertainty
data {
  int N_obs;                         // number of observed distances
  int N_mis;                         // number of missing distances
  vector[N_obs] x_obs;               // distance in feet
  vector[N_mis] x_lower;             // distance in feet lower bound
  vector[N_mis] x_upper;             // distance in feet upper bound
  array[N_obs + N_mis] int throws;   // number of attempts
  array[N_obs + N_mis] int throwins; // number of successes
  real R;                            // target radius (half of basket diameter)
  int N_pred;                        // number of prediction distances
  vector[N_pred] x_pred;             // distance in feet for predictions
}
transformed data {
  int N = N_obs + N_mis;             // total number of observations
}
parameters {
  real<lower=0> sigma_base;          // scale for base probability
  real<lower=0> sigma_intercept;     // scale for angle and distance uncertainty
  // change in scale for angle and distance uncertainty
  real<lower=0> sigma_slope;
  // distance threshold for change in scale
  real<lower=0,upper=65> slope_threshold;
  // distances with known lower and upper bound
  vector<lower=x_lower,upper=x_upper>[N_mis] x_mis;
}
transformed parameters {
  // all observations
  vector[N] x = append_row(x_obs, x_mis);
  // probability of success
  vector[N] p;
  for (n in 1:N) {
    // base probability
    real p_base = Phi(1 ./ sigma_base);
    // common sigma for angle and distance uncertainties
    real sigma = sqrt(square(sigma_intercept) + square(sigma_slope .* max([x[n]-slope_threshold,0.0])));
    // maximum allowable angle at distance x
    real threshold_angle = asin(R ./ x[n]);
    // probability of correct angle
    real p_angle = rayleigh_cdf(threshold_angle | sigma);
    // probability of correct distance
    real p_distance = normal_cdf(1.0 | 0.0, sigma * x[n]);
    // probability of success
    p[n] = p_base * p_angle * p_distance;
  }
  jacobian += 1;
}
model {
  // weakly informative priors
  sigma_base ~ normal(0, 1);
  sigma_intercept ~ normal(0, 1);
  sigma_slope ~ normal(0, 1);
  slope_threshold ~ normal(30, 10);

  // p(x_mis) \propto (x_mis)^2
  target += 2*log(x_mis);

  // data model  
  throwins ~ binomial(throws, p);
}
generated quantities {
  // predicted probabilities at different distances
  vector[N_pred] p_pred;
  for (n in 1:N_pred) {
    // base probability
    real p_base = Phi(1 ./ sigma_base);
    // common sigma for angle and distance uncertainties
    real sigma = sqrt(square(sigma_intercept) + square(sigma_slope .* max([x_pred[n]-slope_threshold,0.0])));
    // maximum allowable angle at each distance
    real threshold_angle_pred = asin(R ./ x_pred[n]);
    // probability of correct angle
    real p_angle = rayleigh_cdf(threshold_angle_pred | sigma);
    // probability of correct distance
    real p_distance = normal_cdf(1.0 | 0.0, sigma * x_pred[n]);
    // probability of success
    p_pred[n] = p_base * p_angle * p_distance;
  }
}
