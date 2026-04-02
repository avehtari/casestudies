#' ---
#' title: "Cross-validation for hierarchical models"
#' author: "Aki Vehtari"
#' date: 2019-03-11
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
#' In this case study, we demonstrate different cross-validation
#' variants for hierarchical/multilevel models using **brms**.
#'
#+ setup, include = FALSE
knitr::opts_chunk$set(
  cache=TRUE,
  message=FALSE,
  error=FALSE,
  warning=FALSE,
  comment=NA
)

#' **Load packages**
#| cache: FALSE
library("rprojroot")
root <- has_file(".casestudies-root")$make_fix_file()
library("loo")
library("brms")
options(brms.backend="cmdstanr", mc.cores = 4)
library("ggplot2")
library("bayesplot")
theme_set(bayesplot::theme_default(base_family = "sans", base_size = 14))

#' # Rats data
#'
#' Throughout, we will use a simple grouped dataset in [rats.Rdata](./rats.Rdata).
#' The example data
#' is taken from Section 6 of @Gelfand+etal:1990, and concerns 30
#' young rats whose weights were measured weekly for five consecutive
#' weeks.
load(root("rats", "rats.Rdata"))
df_rats <- with(rats,
                data.frame(age = x,
                           age_c = x - xbar,
                           weight = y,
                           rat = rat))
N <- rats$Npts

#' **Plot data**
pr <- ggplot(data = df_rats, aes(x = age, y = weight)) +
  geom_line(aes(group = rat), color = "black", linewidth = 0.1) +
  geom_point(color = "black", size = 2) +
  labs(x = "Age (days)", y = "Weight (g)", title = "Rats data")
pr

#' Just by looking at the data, it seems that the rat growth could
#' be modelled with a linear model (up to an age of 36 days).
#' Individual intercepts are likely and possibly also individual
#' slopes.
#'
#' # Models
#'
#' We are going to compare three models: one with population effect
#' only, another with an additional varying intercept term, and a
#' third one with both varying intercept and slope terms.
#'
#' **Pooled linear model**
#| label: fit_1
#| results: hide
fit_1 <- brm(weight ~ age,
             data = df_rats,
             save_pars = save_pars(all = TRUE),
             silent = 2,
             refresh = 0)

#' **Linear model with hierarchical intercept**
#| label: fit_2
#| results: hide
fit_2 <- brm(weight ~ age + (1 | rat),
             data = df_rats,
             save_pars = save_pars(all = TRUE),
             silent = 2,
             refresh = 0)

#' **Linear model with hierarchical intercept and slope**
#| label: fit_3
#| results: hide
fit_3 <- brm(weight ~ age + (age | rat),
             data = df_rats,
             save_pars = save_pars(all = TRUE),
             silent = 2,
             refresh = 0)

#' # Leave-one-out cross-validation
#'
#' In leave-one-out cross-validation (LOO), one observation is left
#' out at a time and predicted given all the other observations.
pr1 <- pr +
  geom_point(data = df_rats[69,], color = "red", size = 5, shape = 1) +
  ggtitle("Leave-one-out")
pr1

#' This is useful and valid if we are interested in model fit in
#' general. We would use the model to predict random missing data, or
#' if we were comparing different conditional observation models.
#'
#' The `loo` package offers a fast Pareto smoothed importance sampling
#' approximation of LOO
#' [@Vehtari+etal:PSIS-LOO:2017;@Vehtari+etal:PSIS:2024]
fit_1 <- fit_1 |> add_criterion(criterion = "loo")
fit_2 <- fit_2 |> add_criterion(criterion = "loo")
fit_3 <- fit_3 |> add_criterion(criterion = "loo")

#' We get warnings about high Pareto k values.
#' As there are only 5 observations per rat, and the hierarchical
#' model has 2 rat-specific parameters, some of the observations are
#' highly influential and PSIS-LOO is not able to give a reliable
#' estimate (if PSIS-LOO fails, WAIC fails, too, but a failure of
#' WAIC is more difficult to diagnose [@Vehtari+etal:PSIS-LOO:2017])
#'
#' We can run exact LOO-CV for the failing folds using `reloo`.
fit_2 <- fit_2 |> add_criterion(criterion = "loo", reloo = TRUE, overwrite = TRUE)
fit_3 <- fit_3 |> add_criterion(criterion = "loo", reloo = TRUE, overwrite = TRUE)

#' We see that PSIS-LOO-estimated `elpd_loo` for model 3 was too
#' optimistic by 2.6 points. Furthermore, its SE was also
#' underestimated.
#'
#' We can now safely do the model comparison:
loo_compare(fit_1, fit_2, fit_3)

#' Model 3 is better than models 1 and 2. Knowing all the other
#' observations except one, it is beneficial to have individual
#' intercept and slope terms.
#'
#' # K-fold cross-validation
#'
#' In K-fold cross-validation the data is divided into K blocks. By
#' using different ways to divide the data, we can target for
#' different prediction tasks or assess different model parts.
#'
#' ## Random K-fold approximation of LOO
#'
#' Sometimes it is possible that a very large number of PSIS-LOO
#' folds fail. In this case, performing exact LOO-CV for all of these
#' observations would take too long. We can approximate LOO
#' cross-validation running K-fold-CV with completely random division
#' of data and then looking at the individual CV predictions.
#'
#' The helper function `kfold_split_random` can be used to form such
#' a random division. We generate random divisions with K=10 and
#' K=30. `kfold` function could do random splits, too, but this
#' way we can use the same random splitting for all models, which
#' makes the mode comparison to have smaller variance.
cv10rfolds <- kfold_split_random(K = 10, N = N)
cv30rfolds <- kfold_split_random(K = 30, N = N)

#' Let's illustrate the first of the 30 folds:
prr <- pr +
  geom_point(data = df_rats[cv30rfolds == 1,], color = "red", size = 5, shape = 1) +
  ggtitle("Random kfold approximation of LOO")
prr

#' We use the `kfold` function for K-fold cross-validation. We specify
#' the folds explicitly, so that the same folds are used for all
#' models.
#| label: kfold-random
cv10r_1 <- kfold(fit_1, K = 10, folds = cv10rfolds)
cv10r_2 <- kfold(fit_2, K = 10, folds = cv10rfolds)
cv10r_3 <- kfold(fit_3, K = 10, folds = cv10rfolds)
cv30r_1 <- kfold(fit_1, K = 30, folds = cv30rfolds)
cv30r_2 <- kfold(fit_2, K = 30, folds = cv30rfolds)
cv30r_3 <- kfold(fit_3, K = 30, folds = cv30rfolds)

#' Compare models
loo_compare(cv10r_1, cv10r_2, cv10r_3)
loo_compare(cv30r_1, cv30r_2, cv30r_3)

#' The results are similar to LOO, and the differences depend on the
#' random division of the data in folds.
#'
#' ## Stratified K-fold approximation of LOO
#'
#' The random split might just by chance leave out more than one
#' observation from one rat, which would not be good for
#' approximating LOO in case of hierarchical models. We can further
#' improve K-fold-CV by using stratified resampling which ensures
#' that the relative category frequencies are approximately
#' preserved. In this case, it means that from each rat only up to
#' one observation is left out per fold.
#'
#' The helper function `kfold_split_stratified` can be used to form a
#' stratified division.
cv10sfolds <- kfold_split_stratified(K = 10, x = df_rats$rat)
cv30sfolds <- kfold_split_stratified(K = 30, x = df_rats$rat)

#' Let's illustrate the first of the 30 folds:
prs <- pr +
  geom_point(data = df_rats[cv30sfolds == 1,], color = "red", size = 5, shape = 1) +
  ggtitle("Stratified K-fold approximation of LOO")
prs

#' We use the `kfold` function for K-fold cross-validation.
#| label: kfold-stratified
cv10s_1 <- kfold(fit_1, K = 10, folds = cv10sfolds)
cv10s_2 <- kfold(fit_2, K = 10, folds = cv10sfolds)
cv10s_3 <- kfold(fit_3, K = 10, folds = cv10sfolds)
cv30s_1 <- kfold(fit_1, K = 30, folds = cv30sfolds)
cv30s_2 <- kfold(fit_2, K = 30, folds = cv30sfolds)
cv30s_3 <- kfold(fit_3, K = 30, folds = cv30sfolds)

#' Compare models
loo_compare(cv10s_1, cv10s_2, cv10s_3)
loo_compare(cv30s_1, cv30s_2, cv30s_3)

#' The results are similar to LOO. For hierarchical models, the results
#' with K=10 and K=30 are closer to each other than in case of
#' complete random division, as the stratified division balances the
#' data division and reduces randomness.
#'
#' ## Grouped K-fold for leave-one-group-out
#'
#' K-fold cross-validation can also be used for leave-one-group-out
#' cross-validation (LOGO-CV). In our example, each group could
#' represent all observations from a particular rat. LOGO-CV is
#' useful if the future prediction task would be to predict growth
#' curves for new rats, or if we are interested in primarily
#' assessing the hierarchical part of the model.
#'
#' In theory, PSIS could be used to also approximate LOGO-CV.
#' However, in hierarchical models, each group has its own set of
#' parameters and the posterior of those parameters tends to change a
#' lot if all observations in that group are removed, which likely
#' leads to failure of importance sampling. For certain models,
#' quadrature methods could be used to compute integrated
#' (marginalized) importance sampling, see e.g. [Roaches case
#' study](https://users.aalto.fi/~ave/modelselection/roaches.html#poisson-model-with-varying-intercept-and-integrated-loo)
#' and paper by @Merkle+Furr+Rabe-Hesketh:2019.
#'
#' The helper function `kfold_split_grouped` can be used to form a
#' grouped division. With K=30 we thus perform leave-one-rat-out CV.
#' With K=10 we get faster computation by leaving out 3 rats at a
#' time, but the results are likely to be similar to K=30. 
cv10gfolds <- kfold_split_grouped(K = 10, x = df_rats$rat)
cv30gfolds <- kfold_split_grouped(K = 30, x = df_rats$rat)

#' Let's illustrate the first of the 30 folds:
prg <- pr +
  geom_point(data = df_rats[cv30gfolds == 1,], color = "red", size = 5, shape = 1) +
  ggtitle("Leave-one-rat-out")
prg

#' We use the `kfold` function for K-fold cross-validation. First with
#' we compute pointwise log-scores, that is, even when we leave out
#' whole groups, we consider predicting left out observations
#' independently.
#| label: kfold-group
cv10g_1 <- kfold(fit_1, K = 10, folds = cv10gfolds)
cv10g_2 <- kfold(fit_2, K = 10, folds = cv10gfolds)
cv10g_3 <- kfold(fit_3, K = 10, folds = cv10gfolds)
cv30g_1 <- kfold(fit_1, K = 30, folds = cv30gfolds)
cv30g_2 <- kfold(fit_2, K = 30, folds = cv30gfolds)
cv30g_3 <- kfold(fit_3, K = 30, folds = cv30gfolds)

#' Compare models
loo_compare(cv10g_1, cv10g_2, cv10g_3)
loo_compare(cv30g_1, cv30g_2, cv30g_3)

#' The results are very different from those obtained by LOO. The
#' order of the models is the same, but the differences are much
#' smaller. As there is no rat-specific covariate information, there
#' is not much difference between predicting with the population
#' curve and a normal response distribution with large scale
#' (`fit_1`) or predicting with uncertain individual curves and a
#' normal response distribution with a small scale (`fit_2` and
#' `fit_3`).
#'
#' When doing leave-one-group-out cross-validation, it is often better
#' to use joint log-scores instead of pointwise log-scores.
#| label: kfold-group-joint
cv30gj_1 <- kfold(fit_1, K = 30, folds = cv30gfolds, joint = "fold")
cv30gj_2 <- kfold(fit_2, K = 30, folds = cv30gfolds, joint = "fold")
cv30gj_3 <- kfold(fit_3, K = 30, folds = cv30gfolds, joint = "fold")

#' Compare models
loo_compare(cv30gj_1, cv30gj_2, cv30gj_3)

#' The results are very different from those obtained by leaving out
#' groups and using pointwise log-score. The comparison shows again a
#' clear difference between the models.
#'
#' Above we used predfefined folds variable `cv30gfolds`. When we have
#' 30 rats, the data division using 30 groups is deterministic and we
#' don't need predefined explicit folds. If the model includes the
#' grouping term as models 2 and 3 do, we could use the more
#' convenient arguments
#| eval: false
cv30g_2 <- kfold(fit_2, group = "rat", joint = "group")
cv30g_3 <- kfold(fit_3, group = "rat", joint = "group")

#' # Alternative models for the prediction given initial weight
#'
#' If in the future we would like to predict growth curves after we
#' have measured the birth weight, we can create new models with the
#' first weight as a covariate.
#'
#' **Create dataframe**
df_rats2 <- with(rats,
                data.frame(age = x[x > 8],
                           age_c = x[x > 8] - 25.5,
                           weight = y[x > 8],
                           rat = rat[x > 8],
                           initweight_c = rep(y[x == 8], 4) - mean(y[x == 8])))

#' ## Models
#'
#' **Simple linear model**
#| label: fit2_1
#| results: hide
fit2_1 <- brm(weight ~ initweight_c + age_c,
              data = df_rats2,
              silent = 2,
              refresh = 0)

#' **Linear model with hierarchical intercept**
#| label: fit2_2
#| results: hide
fit2_2 <- brm(weight ~ initweight_c + age_c + (1 | rat),
              data = df_rats2,
              silent = 2,
              refresh = 0)

#' **Linear model with hierarchical intercept and slope**
#| label: fit2_3
#| results: hide
fit2_3 <- brm(weight ~ initweight_c + age_c + (age_c | rat),
              data = df_rats2,
              silent = 2,
              refresh = 0)

#' ## Grouped K-fold for prediction given initial weight
#'
#' The helper function `kfold_split_grouped` can be used to form a
#' grouped division.
cv30g2folds <- kfold_split_grouped(K = 30, x = df_rats2$rat)

#' We use the `kfold` function for K-fold cross-validation.
#| label: kfold-group-joint-2
cv30gj2_1 <- kfold(fit2_1, K = 30, folds = cv30g2folds, joint = "fold")
cv30gj2_2 <- kfold(fit2_2, K = 30, folds = cv30g2folds, joint = "fold")
cv30gj2_3 <- kfold(fit2_3, K = 30, folds = cv30g2folds, joint = "fold")

#' Compare models
loo_compare(cv30gj2_1, cv30gj2_2, cv30gj2_3)

#' Model 3 is the best, although there is smaller relative difference
#' to model 2.
#'
#' # Conclusion
#'
#' In all comparisons shown in this case study, model 3 was the best,
#' followed by model 2, while model 1 clearly performed the worst.
#' However, depending on the particular cross-validation approach,
#' the differences between models varied.
#'
#' # References {.unnumbered}
#'
#' ::: {#refs}
#' :::
#'
#' # Licenses {.unnumbered}
#'
#' * Code &copy; 2019-2026, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2019-2026, Aki Vehtari, licensed under CC-BY-NC 4.0.
#'
