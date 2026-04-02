#' ---
#' title: "Bayesian variable selection for red wine quality ranking data"
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
#' This notebook was inspired by Eric Novik's slides "Deconstructing
#' Stan Manual Part 1: Linear". The idea is to demonstrate how easy
#' it is to do good variable selection with
#' [`brms`](https://cran.r-project.org/package=brms),
#' [`loo`](https://cran.r-project.org/package=loo), and
#' [`projpred`](https://cran.r-project.org/package=projpred).
#'
#' In this notebook we illustrate Bayesian inference for model
#' selection, including PSIS-LOO [@Vehtari+etal:PSIS-LOO:2017] and
#' projection predictive approach
#' [@McLatchie+etal:2023:projpred_workflow;
#' @Piironen+etal:projpred:2020; @Piironen+Vehtari:2017a] which makes
#' decision theoretically justified inference after model selection.
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
library(brms)
options(brms.backend = "cmdstanr")
options(mc.cores = 4)
library(loo)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size = 14))
library(ggdist)
library(ggforce)
library(posterior)
library(projpred)
SEED <- 170701694

#' # Wine quality data
#'
#' We use [Wine quality data set from UCI Machine Learning
#' repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
wine <- read.delim(root("winequality-red", "winequality-red.csv"), sep = ";") |>
  distinct()
(p <- ncol(wine))
prednames <- names(wine)[1:(p-1)]
glimpse(wine)

#' We scale the covariates so that when looking at the marginal
#' posteriors for the effects they are on the same scale.
wine_scaled <- as.data.frame(scale(wine))

#' # Fit regression model
#'
#' We use the `brms` package with R2D2 prior
#' [@Zhang+etal:2022:R2D2] which provides adaptive shrinkage of
#' regression coefficients.
#| label: fitg
#| results: hide
#| cache: true
fitg <- brm(quality ~ .,
            data = wine_scaled,
            prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3)),
            seed = SEED,
            silent = 2,
            refresh = 0)

#' Let's look at the summary:
summary(fitg)

#' Next do posterior predictive checking
pp_check(fitg)

#' Looking at this, we remember that the data are discrete quality
#' rankings, and it would be better to use ordinal model.
#| label: fito
#| results: hide
#| cache: true
fito <- brm(ordered(quality) ~ .,
            family = cumulative("logit"),
            data = wine_scaled,
            prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3)),
            seed = SEED,
            silent = 2,
            refresh = 0)

#' Let's look at the summary:
summary(fitg)

#' Although in general we can't directly compare continuous and
#' discrete data models, when the target is integers, we can do direct
#' comparison as explained in [Nabiximols case
#' study](https://users.aalto.fi/~ave/casestudies/Nabiximols/nabiximols.html).
#' We use fast Pareto smoothed importance sampling leave-one-out
#' cross-validation [@Vehtari+etal:PSIS-LOO:2017]
loo_compare(loo(fitg), loo(fito))

#' Ordinal model has much better predictive performance.
#'
#' Ordinal model is flexible enough, that posterior predictive checks
#' are unlikely to see any issues as explained in [Recommendations for
#' visual predictive checks in Bayesian
#' workflow](https://teemusailynoja.github.io/visual-predictive-checks/)
#' [@Sailynoja-Johnson-Martin-etal:2025]. For example, the following
#' bar plot is usually useless.
pp_check(fito, type="bars")

#' We can now examine posterior marginals.
drawso <- as_draws_df(fito) |>
  subset_draws(variable = paste0('b_', prednames)) |>
  set_variables(variable = prednames)
mcmc_areas(drawso, prob_outer = .95)

#' Several 95% posterior intervals are not overlapping 0, so maybe
#' there is something useful here.
#'
#' # Projection predictive variable selection
#'
#' We make the projective predictive variable selection
#' [@Piironen+etal:projpred:2020; @Piironen+Vehtari:2017a] using
#' `projpred` package. A fast PSIS-LOO
#' [@Vehtari+etal:PSIS-LOO:2017] is used to choose the model size. As
#' the number of observations is large compared to the number of
#' covariates, we estimate the performance using LOO-CV only along the
#' search path (`validate_search=FALSE`), as we may assume that the
#' overfitting in search is negligible (see more about this in
#' @McLatchie+etal:2023:projpred_workflow).
#'
#' For ordinal models we can use either latent projection approach
#' [@Catalina-Burkner-Vehtari:2021] or augmented-data projection
#' [@Weber-Glass-Vehtari:2025]. The augmented-data projection is more
#' accurate, but much slower than latent projection. It is a good idea
#' to first use latent projection and augmented-data projection can
#' then be run with smaller `nterms_max` chosen based on the latent
#' projection result.
#'
#' We first use the latent projection.
#| label: fito_cv_latent
#| results: hide
#| cache: true
fito_cv_latent <- cv_varsel(fito,
                      latent = TRUE,
                      method = "forward",
                      cv_method = "loo",
                      validate_search = FALSE)

#' We look at the estimated predictive performance of smaller
#' models compared to the full model.
plot(fito_cv_latent, stats = c("elpd"), delta = TRUE)

#' We then repat using augmented-data projection and `nterms_max=5`.
#| label: fito_cv
#| results: hide
#| cache: true
fito_cv <- cv_varsel(fito,
                     nterms_max = 5,
                     method = "forward",
                     cv_method = "loo",
                     validate_search = FALSE)

#' In this case, there is no difference in the predictor ordering, but 
plot(fito_cv, stats = c("elpd"), delta = TRUE)

#' Three or four variables seem to be needed to get the same
#' performance as the full model. As the estimated predictive
#' performance is not going much above the reference model
#' performance, we know that the use of option
#' `validate_search=FALSE` was safe (see more in
#' @McLatchie+etal:2023:projpred_workflow).
#'
#' We can get a loo-cv based recommendation for the model size to
#' choose.
(nsel <- suggest_size(fito_cv, alpha = 0.1))
(vsel <- ranking(fito_cv, nterms_max = nsel)$fulldata)

#' projpred recommends to use four variables: alcohol,
#' volatile.acidity, sulphates, and chlorides.
#'
#' ## Projected posterior
#'
#' Next we form the projected posterior for the chosen model. This
#' projected model can be used in the future to make predictions by
#' using only the selected variables.
#| label: projo
#| results: hide
projo <- project(fito_cv, nterms = nsel, ndraws = 400)

#' The marginals of projected posteriors look like this.
projdraws <- as_draws_df(projo) |>
  subset_draws(variable = paste0("b_", vsel)) |>
  set_variables(variable = vsel)
mcmc_areas(projdraws,
           prob_outer = 0.99,
           area_method = "scaled height")

#' ## Predicted qualities
#'
#' We can examine how well we can actually predict the wine
#' quality based on these predictors.
#| label: preds
preds <- proj_predict(projo) + 2

#' We can compare the predictive means and observed qualities.
pred_mean <- colMeans(preds)
ggplot(data.frame(observed = factor(wine$quality),
                  predicted_mean = pred_mean),
       aes(x = observed, y = predicted_mean)) +
  geom_swarm(color = "steelblue") +
  annotate("segment", x = 1, xend = 6, y = 3, yend = 8,
           linetype = "dashed", color = "gray50") +
  labs(x = "Observed quality", y = "Posterior predictive mean") +
  scale_y_continuous(breaks = 3:8, limits = c(3, 8))

#' Next we compute the average predicted probability for each
#' quality category, grouped by the observed quality. This shows the
#' full predictive distribution rather than just the mean.
qlevels <- sort(unique(wine$quality))
prob_df <- do.call(rbind, lapply(seq_along(wine$quality), function(i) {
  probs <- table(factor(preds[, i], levels = qlevels)) / nrow(preds)
  data.frame(observed = wine$quality[i],
             predicted = as.integer(names(probs)),
             prob = as.numeric(probs))
}))
prob_avg <- aggregate(prob ~ observed + predicted,
                      data = prob_df,
                      FUN = mean)
ggplot(prob_avg, aes(x = factor(observed), y = factor(predicted),
                     fill = prob)) +
  geom_tile() +
  geom_text(aes(label = round(prob, 2)), size = 4) +
  scale_fill_gradient(low = "white", high = "#2166AC",
                      name = "Probability") +
  labs(x = "Observed quality", y = "Predicted quality") +
  coord_equal() +
  annotate("segment", x = .5, xend = 6.5, y = .5, yend = 6.5,
           linetype = "dashed", alpha = 0.3)

#' We can predict something, but there is plenty of unexplained
#' variation, which makes sense considering the available predictors.
#' The model distinguishes well between low (5) and high (7–8)
#' quality wines, but there is substantial overlap in the middle
#' categories.
#'
#' ## Predicted probabilities
#'
#' Instead of discrete posterior predictive draws, we can use
#' `proj_linpred` with `transform=TRUE` to obtain predictive
#' probabilities for each ranking. This provides smoother
#' and more informative summaries.
ppreds <- proj_linpred(projo, transform = TRUE)$pred

#' `ppreds$pred` is a 3D array with dimensions (draws × observations
#' × categories). We average over projected draws to get the mean
#' predicted probability for each observation and category.
qlevels <- 3:8
mean_probs <- apply(ppreds, c(2, 3), mean)
colnames(mean_probs) <- qlevels

#' We compute the expected quality as the probability-weighted mean of
#' the quality levels. Compared to the posterior predictive mean from
#' discrete draws, this gives a smoother prediction (although in this
#' case, there is not much difference.
exp_quality <- as.numeric(mean_probs %*% qlevels)
ggplot(data.frame(observed = factor(wine$quality),
                  expected = exp_quality),
       aes(x = observed, y = expected)) +
  geom_swarm(color = "steelblue") +
  annotate("segment", x = 1, xend = 6, y = 3, yend = 8,
           linetype = "dashed", color = "gray50") +
  labs(x = "Observed quality",
       y = "Expected quality (probability-weighted)") +
  scale_y_continuous(breaks = 3:8, limits = c(3, 8))

#' We can also compute the average predicted probability for each
#' quality category grouped by observed quality.  We show four
#' different visualizations. First, a plot with tiles resembling
#' a confusion matrix, but showing probabilities.
prob_by_obs <- do.call(rbind, lapply(qlevels, function(q) {
  idx <- wine$quality == q
  if (sum(idx) == 0) return(NULL)
  data.frame(observed = q,
             predicted = qlevels,
             prob = colMeans(mean_probs[idx, ]))
}))
ggplot(prob_by_obs, aes(x = factor(observed), y = factor(predicted),
                        fill = prob)) +
  geom_tile() +
  geom_text(aes(label = if_else(prob>0.01,sprintf("%.2f", prob),"")), size = 4) +
  scale_fill_gradient(low = "white", high = "#2166AC",
                      name = "Probability") +
  labs(x = "Observed quality", y = "Predicted quality") +
  coord_equal()

#' Second, where the tiles have areas proportional to the probabilities.
ggplot(prob_by_obs, aes(x = factor(observed), y = factor(predicted),
                        width = 1.1*sqrt(prob), height = 1.1*sqrt(prob),
                        fill = prob)) +
  geom_tile() +
  geom_text(aes(label = if_else(prob>0.15,sprintf("%.2f", prob),"")), size = 4) +
  scale_fill_gradient(low = "white", high = "#2166AC",
                      name = "Probability") +
  labs(x = "Observed quality", y = "Predicted quality") +
  coord_equal()

#' Third, using circles instead of tiles.
ggplot(prob_by_obs, aes(x0 = factor(observed), y0 = factor(predicted),
                        x = factor(observed), y = factor(predicted),
                        fill = prob, r = 0.6*sqrt(prob))) +
  ggforce::geom_circle() +
  geom_text(aes(label = if_else(prob>0.14,sprintf("%.2f", prob),"")), size = 4) +
  scale_fill_gradient(low = "white", high = "#2166AC",
                      name = "Probability") +
  labs(x = "Observed quality", y = "Predicted quality") +
  coord_equal()

#' Finally, stacked bars grouped by observed quality.
ggplot(prob_by_obs, aes(x = factor(observed), y = prob,
                        fill = factor(predicted))) +
  geom_col(position = position_stack(reverse = TRUE)) +
  scale_fill_brewer(palette = "RdYlBu", direction = -1,
                    name = "Predicted\nquality") +
  guides(fill = guide_legend(reverse = TRUE)) +
  labs(x = "Observed quality", y = "Average predicted probability") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.02)))

#' The probability-based visualizations confirm that the model can
#' separate low from high quality wines, but assigns substantial
#' probability to neighboring categories, reflecting high uncertainty
#' in the predictions.
#'
#' # References {.unnumbered}
#'
#' ::: {#refs}
#' :::
#'
#' # Licenses {.unnumbered}
#'
#' * Code &copy; 2017-2026, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2017-2026, Aki Vehtari, licensed under CC-BY-NC 4.0.
