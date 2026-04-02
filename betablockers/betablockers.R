#' ---
#' title: "Beta blocker cross-validation demo"
#' author: "Aki Vehtari"
#' date: 2018-01-10
#' date-modified: today
#' date-format: iso
#' format:
#'   html:
#'     number-sections: true
#'     code-copy: true
#'     code-download: true
#'     code-tools: true
#' bibliography: ../casestudies.bib
#' ---
#'
#' # Introduction
#'
#' This notebook demonstrates a simple model we trust (no model
#' misspecification). In this case, cross-validation (or other model
#' selection appraoch) is not needed, and we can get better accuracy
#' looking at the posterior of quantity of interest directly.
#'
#+ setup, include = FALSE
knitr::opts_chunk$set(
  cache = FALSE,
  message = FALSE,
  error = FALSE,
  warning = FALSE,
  comment = NA,
  out.width = '95%'
)

#' **Load packages**
#| cache: FALSE
library("rprojroot")
root<-has_file(".casestudies-root")$make_fix_file()
library(tidyr)
library(rstanarm)
library(loo)
library(ggplot2)
theme_set(bayesplot::theme_default(base_family = "sans"))
library(ggridges)
library(bridgesampling)

#' # Comparison of two groups with Binomial
#'
#' An experiment was performed to estimate the effect of beta-blockers
#' on mortality of cardiac patients [the example is from @BDA3, Ch
#' 3]. A group of patients were randomly assigned to treatment and
#' control groups:
#'
#' - out of 674 patients receiving the control, 39 died
#' - out of 680 receiving the treatment, 22 died
#'
#' Data, where `grp2` is a dummy variable that captures the difference
#' of the intercepts in the first and the second group.
d_bin2 <- data.frame(N = c(674, 680),
                     y = c(39,22),
                     grp2 = c(0,1))

#' ## Analysis of the observed data
#'
#' To analyse whether the treatment is useful, we can use Binomial
#' model for both groups and compute odds-ratio.
#| label: fit_bin2
#| results: hide
fit_bin2 <- stan_glm(y/N ~ grp2,
                     family = binomial(),
                     data = d_bin2,
                     weights = N,
                     refresh = 0)

#' In general we recommend showing the full posterior of the quantity
#' of interest, which in this case is the odds ratio.
samples_bin2 <- rstan::extract(fit_bin2$stanfit)
theta1 <- plogis(samples_bin2$alpha)
theta2 <- plogis(samples_bin2$alpha + samples_bin2$beta)
oddsratio <- (theta2/(1-theta2))/(theta1/(1-theta1))
ggplot() +
  coord_cartesian(expand = c(bottom = FALSE)) +
  geom_histogram(aes(oddsratio), bins = 50, fill = "grey", color = "darkgrey") +
  labs(y = "") +
  scale_y_continuous(breaks = NULL) +
  theme_sub_axis_y(line = element_blank())

#' We can compute the probability that odds-ratio is less than 1:
print(mean(oddsratio<1),2)

#' This posterior distribution of the odds-ratio (or some
#' transformation of it) is the simplest and the most accurate way to
#' analyse the effectiveness of the treatment. In this case, there is
#' high probability that the treatment is effective and relatively
#' big. Additional observations would be helpful to reduce the
#' uncertainty.
#'
#' ## Simulation experiment
#'
#' Although we recommend showing the full posterior, the probability
#' that oddsratio < 1 can be a useful summary. Simulation experiment
#' `binom_odds_comparison.R` runs 100 simulations with simulated data
#' with varying oddsratio (0.1,...,1.0) and computes for each run the
#' probability that oddsratio<1. The following figures show the
#' variation in the results.
#'
#' Variation in probability that oddsratio<1 when true oddsratio is
#' varied.
load(root("betablockers","binom_test_densities.RData"))
ggplot(betaprobs_densities, aes(x = values, y = ind, height = scaled)) + 
  geom_density_ridges(stat = "identity", scale=0.6)

#' We see that for small treatment effects, just by chance we can
#' observe data that hve varying information about the latent
#' treatment effect.
#'
#' # Cross-validation
#'
#' Sometimes it is better to focus on observable space (we can't
#' observe $\theta$ or odds-ratio directly, but we can observe $y$).
#' For example, in case of many collinear covariates, it can be
#' difficult to interpret the posterior directly in the same way we
#' can do in this simple example. In such cases, we may investigate
#' the difference in the predictive performance.
#'
#' In leave-one-out cross-validation, model is fitted $n$ times with
#' each observation left out at time in fitting and used to evaluate
#' the predictive performance. This corresponds to using the already
#' seen observations as pseudo Monte Carlo samples from the future
#' data distribution, with the leave-trick used to avoid double use of
#' data. With the often used log-score we get
#' $$\mathrm{LOO} = \frac{1}{n} \sum_{i=1}^n \log {p(y_i|x_i,D_{-i},M_k)}.$$
#'
#' Basic cross-validation makes only assumption that the future data
#' comes from the same distribution as the observed data (weghted
#' cross-validation can be used to handle moderate data shifts), but
#' doesn't make any model assumption about that distribution. This sis
#' useful when we don't trust any model (the models might include good
#' enough models, but we just don't know if that is the case).
#'
#' Next we demonstrate one of the weaknesses of cross-validation (same
#' holds for WAIC etc.).
#'
#' ## Analysis of the observed data
#'
#' To use leave-one-out where "one" refers to an individual patient,
#' we need to change the model formulation a bit. In the above model
#' formulation, the individual observations have been aggregated to
#' group observations and running `loo(fit_bin2)` would try to leave
#' one group completely. In case of having more groups, this could be
#' what we want, but in case of just two groups it is unlikely. Thus,
#' in the following we switch to a Bernoulli model with each
#' individual as it's own observation.
d_bin2b <- data.frame(y = c(rep(1,39), rep(0,674-39), rep(1,22), rep(0,680-22)),
                      grp2 = c(rep(0, 674), rep(1, 680)))
#| label: fit_bin2b
#| results: hide
fit_bin2b <- stan_glm(y ~ grp2,
                      family = binomial(),
                      data = d_bin2b,
                      seed=180202538,
                      refresh=0)

#' We fit also a "null" model which doesn't use the group variable and
#' thus has common parameter for both groups.
#| label: fit_bin2bnull
#| results: hide
fit_bin2bnull <- stan_glm(y ~ 1,
                          family = binomial(),
                          data = d_bin2b,
                          seed=180202538,
                          refresh=0)

#' We can then use cross-validation to compare whether adding the
#' treatment variable improves predictive performance. We use fast
#' Pareto smoothed importance sampling leave-one-out cross-validation
#' [PSIS-LOO; @Vehtari+etal:PSIS-LOO:2017].
(loo_bin2 <- loo(fit_bin2b))
(loo_bin2null <- loo(fit_bin2bnull))

#' All Pareto $k<0.5$ and we can trust PSIS-LOO computation
#' [@Vehtari+etal:PSIS-LOO:2017; @Vehtari+etal:PSIS:2022].
#'
#' We make a pairwise comparison.
loo_compare(loo_bin2null, loo_bin2)

#' `elpd_diff` is small compared to `diff_se`, and thus
#' cross-validation is uncertain whether estimating the treatment
#' effect improves the predictive performance. The difference is so
#' small that the normal approximation for quantifying the uncertainty
#' in the difference is not reliable
#' [@Sivula-Magnusson-Vehtari:2025]. To put this in perspective, we
#' have $N_1=674$ and $N_2=680$, and 5.8% and 3.2% deaths, which is
#' too weak information for cross-validation. Although the difference
#' in predictive performance for individuals is small, the posterior
#' of the treatment effect can still be informative.
#'
#' ## Simulation experiment
#'
#' Simulation experiment `binom_odds_comparison.R` runs 100
#' simulations with simulated data with varying oddsratio
#' (0.1,...,1.0) and computes LOO comparison for each run.
#'
#' Variation in LOO comparison when true oddsratio is varied.
ggplot(looprobs_densities, aes(x = values, y = ind, height = scaled)) + 
  geom_density_ridges(stat = "identity", scale=0.6)

#' We see that using the posterior distribution from the model is more
#' efficient to detect the effect, but cross-validation will detect it
#' eventually too. The difference here comes that cross-validation
#' doesn't trust the model, compares the model predictions to the
#' "future data" using very weak assumption about the future, which
#' leads to higher variance of the estimates. The weak assumption
#' about the future is also the cross-validation strength as we'll see
#' in another notebook.
#'
#' # Reference predictive approach
#'
#' We can also do predictive performance estimates using stronger
#' assumption about the future. A reference predictive estimate with
#' log-score can be computed as
#' $$ \mathrm{elpd}_{\mathrm{ref}} = \int
#' p(\tilde{y}|D,M_*) \log p(\tilde{y}|D,M_k) d\tilde{y}, $$
#' where $M_*$ is a reference model we trust. Using a reference model
#' to assess the other models corresponds to $M$-completed case
#' [@Vehtari+Ojanen:2012], where the true model is replaced with a
#' model we trust to be close enough to the true model. The reference
#' model approch has smaller variance than cross-validation, but it is
#' biased towards the reference model, which means that the reference
#' model should be carefully checked to not be in conflict with the
#' observed data, and the the reference model approch provides the
#' best predictive performance estimate for the reference model
#' itself. Here we illustrate the reference model approach so that ech
#' $p(\tilde{y}|D,M_k)$- is the usual posterior predictive
#' distribution. Even better would be to use projection approach,
#' which is demonstrated in other notebooks. See more about the
#' decision theoretical justification of the reference and projection
#' approaches in Section 3.3 of the review by
#' @Vehtari+Ojanen:2012, and experimental results by
#' @Piironen+Vehtari:2017a.
#'
#' ## Simulation experiment
#'
#' The next figure shows the results from the same simulation study
#' using a reference predictive approach with the `fit_bin2` model
#' used as the reference.
ggplot(refprobs_densities, aes(x = values, y = ind, height = scaled)) + 
  geom_density_ridges(stat = "identity", scale=0.6)

#' We can see better accuracy than for cross-validation. We also see,
#' especially when there is no treatment effect that the reference
#' model approach is favoring the reference model itself.
#'
#' The similar and even bigger improvement in the model selection
#' performance is observed in projection predictive variable selection
#' [@Piironen+Vehtari:2017a; @Piironen+etal:projpred:2020;
#' @McLatchie+etal:2023:projpred_workflow] implemented in [`projpred`
#' package](https://cran.r-project.org/package=projpred).
#'
#' # Marginal likelihood
#'
#' As comparison we include marginal likelihood based approach to
#' compute the posterior probabilities for the null model (treatment
#' effect is zero) and the model with unknown treatment effect. As the
#' data and models are very simple, we may assume that the model is
#' well specified. Marginal likelihoods and relative posterior
#' probabilities can be sensitive to the selected prior on the bigger
#' model. Here we simply use the same `rstanarm` default prior as for
#' the above examples. Marginal likelihoods are computed using the
#' default bridge sampling approach implemented in `bridge_sampling`
#' package.
#'
#' ## Analysis of the observed data
#| label: fit_bin2_bridge
#| results: hide
# rerun models with diagnostic file required by bridge_sampler
fit_bin2 <- stan_glm(y/N ~ grp2,
                     family = binomial(),
                     data = d_bin2,
                     weights = N,
                     refresh=0,
                     diagnostic_file = file.path(tempdir(), "df.csv"))
(ml_bin2 <- bridge_sampler(fit_bin2, silent=TRUE))
fit_bin2null <- stan_glm(y/N ~ 1,
                         family = binomial(),
                         data = d_bin2,
                         weights = N,
                         refresh=0,
                         diagnostic_file = file.path(tempdir(), "df.csv"))
(ml_bin2null <- bridge_sampler(fit_bin2null, silent=TRUE))
print(post_prob(ml_bin2, ml_bin2null), digits=2)

#' Posterior probability computed from the marginal likelihoods is
#' indecisive.
#'
#' ## Simulation experiment
#'
#' We repeat the simulation with marginal likelihood approach.
ggplot(bfprobs_densities, aes(x = values, y = ind, height = scaled)) + 
  geom_density_ridges(stat = "identity", scale=0.6)

#' We can see that marginal likelihood based approach favors more
#' strongly null model for smaller treatment effects, requires a
#' bigger effect than the other approaches to not favor the null
#' model, but given big enough effect is more decisive on non-null
#' model than cross-validation.
#'
#' # References {.unnumbered}
#'
#' ::: {#refs}
#' :::
#'
#' # Licenses {.unnumbered}
#'
#' * Code &copy; 2018-2026, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2018-2026, Aki Vehtari, licensed under CC-BY-NC 4.0.
#' * Part of the code copied from [rstanarm_demo.Rmd](https://github.com/avehtari/BDA_R_demos/blob/master/demos_rstan/rstanarm_demo.Rmd) written by Aki Vehtari and Markus Paasiniemi
#'
