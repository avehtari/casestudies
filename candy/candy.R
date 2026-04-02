#' ---
#' title: "Bayesian variable selection for candy ranking data"
#' author: "Aki Vehtari"
#' date: 2018-02-27
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
#' This notebook was inspired by Joshua Loftus' two blog posts [Model
#' selection bias invalidates significance
#' tests](https://joshualoftus.com/posts/2020-12-22-model-selection-bias-invalidates-significance-tests/model-selection-bias-invalidates-significance-tests.html)
#' and [A conditional approach to inference after model
#' selection](https://joshualoftus.com/posts/2020-12-22-a-conditional-approach-to-inference-after-model-selection/a-conditional-approach-to-inference-after-model-selection.html).
#'
#' In this notebook we illustrate Bayesian inference for model
#' selection, including PSIS-LOO [@Vehtari+etal:PSIS-LOO:2017] and
#' projection predictive approach
#' [@McLatchie+etal:2023:projpred_workflow;
#' @Piironen+etal:projpred:2020; @Piironen+Vehtari:2017a] which makes
#' decision theoretically justified inference after model selection.
#'
#' # Setup  {.unnumbered}
#'
#+ setup, include = FALSE
knitr::opts_chunk$set(
  cache=FALSE,
  message=FALSE,
  error=FALSE,
  warning=FALSE,
  comment=NA,
  out.width='95%'
)

#' **Load packages**
#| cache: FALSE
library("rprojroot")
root <- has_file(".casestudies-root")$make_fix_file()
library(brms)
options(brms.backend = "cmdstanr")
options(mc.cores = 4)
library(loo)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size = 14))
library(ggdist)
library(posterior)
library(projpred)
library(fivethirtyeight)
library(dplyr)
SEED <- 150702646

#' # Data
#'
#' We use candy rankings data from fivethirtyeight package. Dataset
#' was originally used in [a fivethirtyeight
#' story](http://fivethirtyeight.com/features/the-ultimate-halloween-candy-power-ranking/).
data(candy_rankings)
candy_rankings <- candy_rankings |>
  select(!competitorname) |>
  mutate_if(is.logical, as.numeric)
prednames <- candy_rankings |>
  select(!winpercent) |>
  colnames()
glimpse(candy_rankings)

#' # Random data
#'
#' We start first analysing a random "null" data set, where winpercent
#' has been replaced with random draws from a normal distribution
#' (with same mean and standar deviation) so that covariates do not
#' have any predictive information.
set.seed(SEED)
N <- nrow(candy_rankings)
candy_random <- candy_rankings |>
  mutate(random = rnorm(N, mean(winpercent), sd(winpercent))) |>
  select(!winpercent)

#' Doing variable selection we are anyway assuming that some of the
#' variables are not relevant, and thus it is sensible to use priors
#' which assume some of the covariate effects are close to zero. We
#' use R2D2 prior [@Zhang+etal:2022:R2D2] which provides adaptive
#' shrinkage of regression coefficients.
#| label: fit_random
#| results: hide
fit_random <- brm(random ~ .,
                  data = candy_random,
                  prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3)),
                  seed = SEED,
                  silent = 2,
                  refresh = 0)

#' Let's look at the summary:
summary(fit_random)

#' We didn't get divergences, Rhat's are less than 1.01 and
#' bulk/tail ESS are useful.
draws_random <- as_draws_df(fit_random) |>
  subset_draws(variable = paste0("b_", prednames)) |>
  set_variables(variable = prednames)
mcmc_areas(draws_random, prob_outer = .95)

#' All 95% posterior intervals are overlapping 0, the R2D2 prior
#' makes the posteriors concentrate near 0, but there is some
#' uncertainty.
#'
#' We can easily test whether any of the covariates are useful by
#' using cross-validation to compare to a null model,
#| label: fit0
#| results: hide
fit0 <- brm(random ~ 1,
            data = candy_random,
            seed = SEED,
            silent = 2,
            refresh = 0)

(loo_random <- loo(fit_random))
(loo0 <- loo(fit0))
loo_compare(loo0, loo_random)

#' Based on cross-validation covariates together do not contain any
#' useful information, and there is no need to continue with variable
#' selection. This step of checking whether the full model has any
#' predictive power is often ignored especially when non-Bayesian
#' methods are used. If loo (or AIC as Joshua Loftus demonstrated)
#' would be used for stepwise variable selection it is possible that
#' selection process over a large number of models overfits to the
#' data.
#'
#' To illustrate the robustness of projpred, we make the projective
#' predictive variable selection using the previous model for "null"
#' data. A fast leave-one-out cross-validation approach
#' [@Vehtari+etal:PSIS-LOO:2017] is used to choose the model size. As
#' the number of observations is large compared to the number of
#' covariates, we estimate the performance using LOO-CV only along the
#' search path (`validate_search=FALSE`), as we may assume that the
#' overfitting in search is negligible (see more about this in
#' @McLatchie+etal:2023:projpred_workflow).
#| label: fit_random_cv
#| results: hide
fit_random_cv <- cv_varsel(fit_random,
                           method = "forward",
                           cv_method = "loo",
                           validate_search = FALSE)

#' We can now look at the estimated predictive performance of smaller
#' models compared to the full model.
plot(fit_random_cv, stats = c("elpd"))

#' As the estimated predictive performance is not going much above the
#' reference model performance, we know that the use of option
#' `validate_search=FALSE` was safe (see more in
#' @McLatchie+etal:2023:projpred_workflow).
#'
#' And we get a LOO based recommendation for the model size to choose
(nv <- suggest_size(fit_random_cv, alpha = 0.1))

#' We see that projpred agrees that no variables have useful
#' information.
#'
#' Next we form the projected posterior for the chosen model.
proj_random <- project(fit_random_cv, nterms = nv, ns = 4000)
projdraws_random <- as_draws_df(proj_random)
round(colMeans(as.matrix(projdraws_random)), 1)

#' This looks good as the true values for "null" data are
#' intercept=`r mean(candy_rankings$winpercent)`,
#' sigma=`r sd(candy_rankings$winpercent)`.
#'
#' # Original data
#'
#' Next we repeat the above analysis with original target variable
#' winpercent.
#| label: fit_candy
#| results: hide
fit_candy <- brm(winpercent ~ .,
                 data = candy_rankings,
                 prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3)),
                 seed = SEED,
                 silent = 2,
                 refresh = 0)

#' Let's look at the summary.
summary(fit_candy)
#' We didn't get divergences, Rhat's are less than 1.01 and
#' bulk/tail ESS are useful.
#' 
#' Posterior predictive checking looks good enough
pp_check(fit_candy) +
  scale_x_continuous(breaks = seq(10, 90, by = 10)) +
  theme(axis.line.y = element_blank())

#' We can examine posterior marginals.
draws_candy <- as_draws_df(fit_candy) |>
  subset_draws(variable = paste0("b_", prednames)) |>
  set_variables(variable = prednames)
mcmc_areas(draws_candy, prob_outer = .95)

#' 95% posterior interval for `chocolateTRUE` is not overlapping 0,
#' so maybe there is something useful here.
#'
#' In case of collinear variables it is possible that marginal
#' posteriors overlap 0, but the covariates can still be useful for
#' prediction. With many variables it will be difficult to analyse
#' joint posterior to see which variables are jointly relevant. We can
#' easily test whether any of the covariates are useful by using
#' cross-validation to compare to a null model,
#| label: fit0_candy
#| results: hide
fit0 <- brm(winpercent ~ 1,
            data = candy_rankings,
            seed = SEED,
            silent = 2,
            refresh = 0)

(loo_candy <- loo(fit_candy))
(loo0 <- loo(fit0))
loo_compare(loo0, loo_candy)

#' Based on cross-validation covariates together do contain useful
#' information. If we need just the predictions we can stop here, but
#' if we want to learn more about the relevance of the covariates we
#' can continue with variable selection.
#'
#' We make the projective predictive variable selection using the
#' previous model. A fast leave-one-out cross-validation approach is
#' used to choose the model size.
#| label: fit_candy_cvv
#| results: hide
fit_candy_cvv <- cv_varsel(fit_candy,
                          method = "forward",
                          cv_method = "loo",
                          validate_search = TRUE)

#' We can now look at the estimated predictive performance of smaller
#' models compared to the full model.
plot(fit_candy_cvv, stats = c("elpd","R2"), deltas = "mixed")

#' Only one variable seems to be needed to get the same performance
#' as the full model.
#'
#' And we get a LOO based recommendation for the model size to choose
(nsel <- suggest_size(fit_candy_cvv, alpha = 0.1))
(vsel <- ranking(fit_candy_cvv, nterms_max = nsel)$fulldata)

#' projpred recommends to use just one variable.
#'
#' Next we form the projected posterior for the chosen model.
#| label: proj_candy
#| results: hide
proj_candy <- project(fit_candy_cvv, nterms = nsel)

#' We plot the marginals of projected posteriors
projdraws <- as_draws_df(proj_candy) |>
  subset_draws(variable = paste0("b_", vsel)) |>
  set_variables(variable = vsel)
mcmc_areas(projdraws,
           prob_outer = 0.99,
           area_method = "scaled height")

#' In our loo and projpred analysis, we find the `chocolateTRUE` to
#' have predictive information. Other variables may have predictive
#' power, too, but conditionally on `chocolateTRUE` other variables do
#' not provide enough additional information to improve predictive
#' performance.

preds <- proj_predict(proj_candy) |>
  as_draws_rvars(preds)
candy_rankings |>
  as_tibble() |>
  mutate(preds = preds) |>
  ggplot(aes(x = winpercent, ydist = preds)) +
  stat_pointinterval(alpha = 0.3, color = "steelblue") +
  geom_abline(linetype = "dashed", alpha = 0.5) +
  labs(x = "Win percent", y = "Prediction")

#' # References {.unnumbered}
#'
#' ::: {#refs}
#' :::
#'
#' # Licenses {.unnumbered}
#'
#' * Code &copy; 2017-2018, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2017-2018, Aki Vehtari, licensed under CC-BY-NC 4.0.
#'
