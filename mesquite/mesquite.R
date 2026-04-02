#' ---
#' title: "Collinear demo with mesquite bushes"
#' author: "Aki Vehtari"
#' date: 2018-01-16
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
#' This notebook demonstrates collinearity in multipredictor
#' regression. Example of predicting the yields of mesquite bushes
#' comes from [Gelman and Hill
#' (2007)](http://www.stat.columbia.edu/~gelman/arm/). The outcome
#' variable is the total weight (in grams) of photosynthetic material
#' as derived from actual harvesting of the bush. The predictor
#' variables are:
#'
#' - diam1: diameter of the canopy (the leafy area of the bush)
#' in meters, measured along the longer axis of the bush
#' - diam2: canopy diameter measured along the shorter axis
#' - canopy height: height of the canopy
#' - total height: total height of the bush
#' - density: plant unit density (# of primary stems per plant unit)
#' - group: group of measurements (0 for the first group, 1 for the
#'   second group)
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
library(dplyr)
library(rstanarm)
library(arm)
options(mc.cores = 4)
library(loo)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size = 14))
library(ggdist)
library(GGally)
library(projpred)
library(posterior)

#' # Data
#' Available in [mesquite.dat](./mesquite.dat)
mesquite <- read.table(root("mesquite", "mesquite.dat"), header=TRUE) |>
  mutate_if(is.character, as.factor) |>
  mutate_if(is.factor, as.numeric) |>
  tibble() |>
  mutate(Group = as.factor(Group))  
#' Additional transformed variables
mesquite <- mesquite |>
  mutate(CanVol = Diam1 * Diam2 * CanHt,
         CanAre = Diam1 * Diam2,
         CanSha = Diam1 / Diam2)
summary(mesquite)

#' Plot data
ggpairs(mesquite,
        diag = list(continuous = "densityDiag", discrete = "barDiag"),
        lower = list(combo = "points"),
        upper = list(combo = "blank", continuous = "blank"),
        axisLabels = "none")

#' It may be reasonable to fit on the logarithmic scale, so that
#' effects are multiplicative rather than additive (we'll return to
#' checking this assumption in another notebook).
#'
#' # Maximum likelihood estimate
#'
#' We first illustrate the problem with maximum likelihood estimate
lm1 <- lm(formula = log(LeafWt) ~ log(CanVol) + log(CanAre) +
             log(CanSha) + log(TotHt) + log(Dens) + Group,
           data = mesquite)
display(lm1)

#' GroupMCD seems to be only variable which has coefficient far away
#' from zero. Let's try making a model with just the group variable.
lm2 <- lm(formula = log(LeafWt) ~ Group, data = mesquite)
display(lm2)

#' $R^2$ dropped a lot, so it seems that other variables
#' are useful even if estimated effects and their standard errors
#' indicate that they are not relevant. There are approaches for
#' maximum likelihood estimated models to investigate this, but we'll
#' switch now to Bayesian inference using
#' [`rstanarm`](https://cran.r-project.org/package=rstanarm).
#'
#' # Bayesian inference
#'
#' The corresponding `rstanarm` model fit using `stan_glm`
#| label: fitg
#| results: hide
fitg <- stan_glm(formula = log(LeafWt) ~ log(CanVol) + log(CanAre) +
                   log(CanSha) + log(TotHt) + log(Dens) + Group,
                 data = mesquite,
                 refresh = 0)

#' Print summary for some diagnostics.
summary(fitg)

#' Rhats and n_effs are good (see, e.g., [RStan
#' workflow](http://mc-stan.org/users/documentation/case-studies/rstan_workflow.html)),
#' but QR transformation usually makes sampling work even better
#' (see, [The QR Decomposition For Regression
#' Models](http://mc-stan.org/users/documentation/case-studies/qr_regression.html))
#'
#' Print summary for some diagnostics.
summary(fitg)

#' Use of QR decomposition improved sampling efficiency (actually we
#' get superefficient sampling, ie better than independent sampling)
#' and we continue with this model.
#'
#' Instead of looking at the tables, it's easier to look at plots
#| fig-height: 4
#| fig-width: 8
mcmc_areas(as.matrix(fitg), prob = .5, prob_outer = .95)

#' All 95% posterior intervals except for GroupMCD are overlapping 0
#' and it seems we have serious collinearity problem.
#'
#' Looking at the pairwise posteriors we can see high correlations
#' especially between log(CanVol) and log(CanAre).
mcmc_pairs(as.matrix(fitg),
           pars = c("log(CanVol)", "log(CanAre)", "log(CanSha)",
                    "log(TotHt)", "log(Dens)"))

#' If look more carefully on of the subplots, we see that although
#' marginal posterior intervals overlap 0, some pairwise joint
#' posteriors are not overlapping 0. Let's look more carefully the
#' joint posterior of log(CanVol) and log(CanAre).
mcmc_scatter(as.matrix(fitg), pars = c("log(CanVol)", "log(CanAre)")) +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0)

#' From the joint posterior scatter plot, we can see that 0 is far
#' away from the typical set.
#'
#' In case of even more variables with some being relevant and some
#' irrelevant, it will be difficult to analyse joint posterior to see
#' which variables are jointly relevant. We can easily test whether
#' any of the covariates are useful by using cross-validation to
#' compare to a null model,
fitg0 <- update(fitg, formula = log(LeafWt) ~ 1, QR = FALSE)

#' We compute leave-one-out cross-validation elpd's using PSIS-LOO
#' [@Vehtari+etal:PSIS-LOO:2017]
(loog <- loo(fitg))
(loog0 <- loo(fitg0))

#' And then we can compare the models.
loo_compare(loog0, loog)

#' Based on cross-validation covariates together contain significant
#' information to improve predictions.
#'
#' We might want to choose some variables 1) because we don't want to
#' observe all the variables in the future (e.g. due to the
#' measurement cost), or 2) we want to most relevant variables which
#' we define here as a minimal set of variables which can provide
#' similar predictions to the full model.
#'
#' LOO can be used for model selection, but we don't recommend it for
#' variable selection as discussed by @Piironen+Vehtari:2017a and @McLatchie+etal:2025:projpred_workflow. The
#' reason for not using LOO in variable selection is that the
#' selection process uses the data twice, and in case of large number
#' variable combinations the selection process overfits and can
#' produce really bad models. Using the usual posterior inference
#' given the selected variables ignores that the selected variables
#' are conditional on the selection process and simply setting some
#' variables to 0 ignores the uncertainty related to their relevance.
#'
#' @Piironen+Vehtari:2017a and @McLatchie+etal:2025:projpred_workflow
#' also show that a projection predictive
#' approach can be used to make a model reduction, that is, choosing
#' a smaller model with some coefficients set to 0. The projection
#' predictive approach solves the problem how to do inference after
#' the selection. The solution is to project the full model posterior
#' to the restricted subspace. See more by
#' @McLatchie+etal:2025:projpred_workflow, @Pavone+etal:2022, and
#' @Piironen+etal:projpred:2020.
#'
#' We make the projective predictive variable selection using the
#' previous full model. A fast leave-one-out cross-validation approach
#' [@Vehtari+etal:PSIS-LOO:2017] is used to choose the model size.
#| label: fitg_cvvs
#| results: hide
#| cache: true
fitg_cvvs <- cv_varsel(fitg,
                       method = "forward",
                       cv_method = "LOO",
                       validate_search = TRUE)

#' We can now look at the estimated predictive performance of smaller
#' models compared to the full model.
plot(fitg_cvvs, stats = c("elpd", "R2"), deltas = "mixed")

#' We get a LOO-CV based recommendation for the model size to
#' choose.
(nsel <- suggest_size(fitg_cvvs, alpha = 0.1))
(vsel <- ranking(fitg_cvvs, nterms_max = nsel)$fulldata)

#' We see that `r nsel` variables is enough to get the same predictive
#' accuracy as with all variables.
#'
#' Next we form the projected posterior for the chosen model.
projg <- project(fitg_cvvs, nv = nsel, ns = 4000)
projdraws <- as_draws_df(projg) |>
  rename_variables(Group = Group2) |>
  subset_draws(variable = vsel) |>
  set_variables(variable = vsel)

#' The marginals of projected posteriors look like this
mcmc_areas(projdraws,
           prob_outer = 0.99,
           area_method = "scaled height")

#' We can also look at the predictive intervals of the selected
#' submodel predictions
preds <- proj_predict(projg) |>
  as_draws_rvars(preds)
mesquite |>
  as_tibble() |>
  mutate(preds = preds) |>
  ggplot(aes(x = log(LeafWt), ydist = preds)) +
  stat_pointinterval(alpha = 0.3, color = "steelblue") +
  geom_abline() +
  labs(x = "log(leaf-weight)", y = "Prediction")
  
#' # References {.unnumbered}
#'
#' ::: {#refs}
#' :::
#'
#' # Licenses {.unnumbered}
#'
#' * Code &copy; 2018-2026, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2018-2026, Aki Vehtari, licensed under CC-BY-NC 4.0.
#'
