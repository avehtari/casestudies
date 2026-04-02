// Disc golf putting model with 2D angular uncertainty and distance uncertainty
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
model {
  // weakly informative priors
  slope_threshold ~ normal(30, 10);
  mu ~ normal(-4, 2);               // prior for mean log sigmas
  tau ~ normal(0, 0.5);             // population SDs
  L_Omega ~ lkj_corr_cholesky(K);   // prior on Cholesky of correlation matrix
  // latent parameters
  to_vector(z) ~ std_normal();

  // p(x_mis) \propto (x_mis)^2
  target += 2*log(x_mis);

  // data model
  throwins ~ binomial(throws, p);
}
generated quantities {
  // predicted probabilities at different distances for different players
  matrix[N_players, N_pred] p_pred;
  for (pl in 1:N_players) {
    for (n in 1:N_pred) {
      // base probability
      real p_base = Phi(1 ./ sigma_base[pl]);
      // common sigma for angle and distance uncertainties
      real sigma = sqrt(square(sigma_intercept[pl]) + square(sigma_slope[pl] .* max([x_pred[n]-slope_threshold,0.0])));
      // maximum allowable angle at each distance
      real threshold_angle_pred = asin(R ./ x_pred[n]);
      // probability of correct angle
      real p_angle = rayleigh_cdf(threshold_angle_pred | sigma);
      // probability of correct distance
      real p_distance = normal_cdf(1.0 | 0.0, sigma * x_pred[n]);
      // probability of success
      p_pred[pl, n] = p_base * p_angle * p_distance;
    }
  }
}
