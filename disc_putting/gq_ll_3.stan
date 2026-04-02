functions {
  // function used to integrate over the unknown distance
  real integrand(real x, real notused1, array[] real theta,
               array[] real X, array[] int data_n) {
    real sigma_base = theta[1];
    real sigma_intercept = theta[2];
    real sigma_slope = theta[3];
    real slope_threshold = theta[4];
    real R = X[1];
    real x_lower = X[2];
    real x_upper = X[3];
    int throws_n = data_n[1];
    int throwins_n = data_n[2];
    // base probability
    real p_base = Phi(1 ./ sigma_base);
    // common sigma for angle and distance uncertainties
    real sigma = sqrt(square(sigma_intercept) + square(sigma_slope .* max([x-slope_threshold,0.0])));
    // maximum allowable angle at distance x
    real threshold_angle = asin(R ./ x);
    // probability of correct angle
    real p_angle = rayleigh_cdf(threshold_angle | sigma);
    // probability of correct distance
    real p_distance = normal_cdf(1.0 | 0.0, sigma * x);
    // probability of success
    real p_n = p_base * p_angle * p_distance;
    // normalization term of the distance prior
    real logZ = log(pow(x_upper,3)/3.0 - pow(x_lower,3)/3.0);
    // return prior times likelihood
    return exp(2*log(x) - logZ + binomial_lpmf(throwins_n | throws_n, p_n));
  }
}
data {
  int N_obs;                         // number of observed distances
  int N_mis;                         // number of missing distances
  int N_players;
  vector[N_obs] x_obs;               // distance in feet
  vector[N_mis] x_lower;             // distance in feet lower bound
  vector[N_mis] x_upper;             // distance in feet upper bound
  array[N_obs + N_mis] int throws;   // number of attempts
  array[N_obs + N_mis] int throwins; // number of successes
  array[N_obs + N_mis] int player;   // player index
  real R;                            // target radius (half of basket diameter)
  int N_pred;                        // number of prediction distances
  vector[N_pred] x_pred;             // distance in feet for predictions
}
transformed data {
  int N = N_obs + N_mis;             // total number of observations
  int K = 3;                         // number of player-specific parameters
}
parameters {
  // distance threshold for change in scale
  real<lower=0,upper=65> slope_threshold;
  // population-level parameters (on log scale for positive constraint)
  vector[K] mu;                     // population means
  vector<lower=0>[K] tau;           // population SDs
  cholesky_factor_corr[K] L_Omega;  // Cholesky factor of correlation matrix
  // latent player effects
  matrix[K, N_players] z;
  // distances with known lower and upper bound
  vector<lower=x_lower,upper=x_upper>[N_mis] x_mis;
}
transformed parameters {
  // all observations
  vector[N] x = append_row(x_obs, x_mis);

 // player-specific parameters (on log scale)
  matrix[N_players, K] eta;

  // transform from non-centered to centered parameterization
  // eta = mu + diag(tau) * L_Omega * z
  eta = (diag_pre_multiply(tau, L_Omega) * z)';
  for (pl in 1:N_players) {
    eta[pl] = eta[pl] + mu';
  }

  // player-specific parameters on natural scale
  vector<lower=0>[N_players] sigma_base;
  vector<lower=0>[N_players] sigma_intercept;
  vector<lower=0>[N_players] sigma_slope;
  for (pl in 1:N_players) {
    sigma_base[pl] = exp(eta[pl, 1]);
    sigma_intercept[pl] = exp(eta[pl, 2]);
    sigma_slope[pl] = exp(eta[pl, 3]);
  }

  // probability of success
  vector[N] p;
  for (n in 1:N) {
    int pl = player[n];
    // base probability
    real p_base = Phi(1 ./ sigma_base[pl]);
    // common sigma for angle and distance uncertainties
    real sigma = sqrt(square(sigma_intercept[pl]) + square(sigma_slope[pl] .* max([x[n]-slope_threshold,0.0])));
    // maximum allowable angle at distance x
    real threshold_angle = asin(R ./ x[n]);
    // probability of correct angle
    real p_angle = rayleigh_cdf(threshold_angle | sigma);
    // probability of correct distance
    real p_distance = normal_cdf(1.0 | 0.0, sigma * x[n]);
    // probability of success
    p[n] = p_base * p_angle * p_distance;
  }
}
generated quantities {
  // log likelihood
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = binomial_lpmf(throwins[n] | throws[n], p[n]);
  }
  // log of integrated likelihood
  vector[N] log_liki;
  log_liki[1:N_obs] = log_lik[1:N_obs];
  for (n in 1:N_mis) {
    int pl = player[N_obs+n];
    // 1D quadrature integration to integrate out the missing distance
    log_liki[N_obs+n] = log(integrate_1d(integrand,
                              x_lower[n],
                              x_upper[n],
                              {sigma_base[pl], sigma_intercept[pl], sigma_slope[pl], slope_threshold},
			      {R, x_lower[n], x_upper[n]},
			      append_array({throws[N_obs+n]}, {throwins[N_obs+n]}),
			      5e-2));
  }
}
