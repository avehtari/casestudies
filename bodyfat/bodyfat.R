#' ---
#' title: "Projection predictive variable selection – A review and recommendations for the practicing statistician"
#' author: "Aki Vehtari"
#' date: 2018-03-06
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
#' This notebook was inspired by an article by @Heinze+etal:bodyfat.
#' They provide "an overview of various available variable selection
#' methods that are based on significance or information criteria,
#' penalized likelihood, the change-in-estimate criterion, background
#' knowledge, or combinations thereof." I agree that they provide
#' sensible recommendations and warnings for those methods. Similar
#' recommendations and warnings hold for information criterion and
#' naive cross-validation based variable selection in Bayesian
#' framework as demonstrated by @Piironen+Vehtari:2017a.
#'
#' @Piironen+Vehtari:2017a and @Pavone+etal:2022 demonstrate also the superior stability of
#' projection predictive variable selection (see specially figures 4
#' and 10). In this notebook I demonstrate the projection predictive
#' variable selection method as presented by
#' @Piironen+etal:projpred:2020 and
#' @McLatchie+etal:2025:projpred_workflow, and implemented in R
#' package
#' [`projpred`](https://cran.r-project.org/package=projpred). I use
#' the same body fat data as used in Section 3.3 of the article by
#' @Heinze+etal:bodyfat. The dataset with the background information
#' is available
#' [here](https://ww2.amstat.org/publications/jse/v4n1/datasets.johnson.html)
#' but @Heinze+etal:bodyfat have made some data cleaning and I have
#' used the same data and some bits of the code they provide in the
#' supplementary material. There still are some strange values like
#' the one person with zero fat percentage, but I didn't do additional
#' cleaning.
#'
#' This notebook was initially created in 2018, but was later extended
#' to an article "Using reference models in variable selection" by
#' @Pavone+etal:2022.
#'
#' The excellent performance of the projection predictive variable
#' selection comes from following parts
#'
#' 1. Bayesian inference using priors and integration over all the
#'    uncertainties makes it easy to get good predictive performance
#'    with all variables included in the model
#'    [@Piironen+Vehtari:RHS:2017; @Piironen+Vehtari:ISPC:2018;
#'    @Piironen+etal:projpred:2020]
#' 2. Projection of the information from the full model to a smaller
#'    model is able to include information and uncertainty from the
#'    left out variables (while conditioning of the smaller model to
#'    data would ignore left out variables)
#'    [@Piironen+etal:projpred:2020; @Pavone+etal:2022].
#' 3. During the search through the model space comparing the
#'    predictive distributions of projected smaller models to the
#'    predictive distribution of the full model reduces greatly the
#'    variance in model comparisons [@Piironen+Vehtari:2017a].
#' 4. Even with greatly reduced variance in model comparison, the
#'    selection process slightly overfits to the data, but we can
#'    cross-validate this effect using the fast Pareto smoothed
#'    importance sampling leave-one-out cross-validation
#'    [@Vehtari+etal:PSIS-LOO:2017; @Vehtari+etal:PSIS:2024]
#'
#' Excellent performance of projection predictive variable selection
#' compared to other Bayesian variable selection methods was presented
#' by @Piironen+Vehtari:2017a. @Piironen+etal:projpred:2020 present
#' further improvements such as improved model size selection and
#' several options to make the approach faster for larger number of
#' variables or bigger data sets. @Vehtari+Ojanen:2012 present
#' theoretical justification for projection predictive model selection
#' and inference after selection.
#'
#' Note that if the goal is only the prediction no variable selection
#' is needed. The projection predictive variable selection can be used
#' to learn which are the most useful variables for making predictions
#' and potentially reduce the future measurement costs. In the
#' bodyfat example, most of the measurements have time cost and there
#' is a benefit of finding the smallest set of variables to be used in
#' the future for the predictions.
#'
#' This notebook presents a linear regression example, but the
#' projection predictive approach can be used also, for example, with
#' generalized linear models [@Piironen+Vehtari:2017a;
#' @Piironen+etal:projpred:2020], Gaussian processes
#' [@Piironen+Vehtari:GP-projection:2016], and generalized linear and
#' additive multilevel models
#' [@Catalina+etal:projpredgamms:2022].
#'
#+ setup, include = FALSE
knitr::opts_chunk$set(
  cache=FALSE,
  message=FALSE,
  error=FALSE,
  warning=FALSE,
  comment=NA,
  out.width="95%"
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
library(projpred)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size = 14))
library(ggdist)
library(posterior)
library(corrplot)
library(knitr)
SEED <- 1513306866

#' # Bodyfat data
#'
#' Load data from [bodyfat.txt](./bodyfat.txt) and scale it.
#' @Heinze+etal:bodyfat used unscaled data,
#' but we scale it for easier comparison of the effect sizes. In
#' theory this scaling should not have detectable difference in the
#' predictions and I did run the results also without scaling and
#' there is no detectable difference in practice.
bodyfat <- read.table(root("bodyfat", "bodyfat.txt"), header = TRUE, sep = ";") |>
  as_tibble() |>
  select(!c(case,brozek,density,weight_kg,height_cm)) |>
  filter(siri > 0) |>
  mutate(across(-siri, ~ as.numeric(scale(.x))))

#' Predictor names
prednames <- bodyfat |>
  select(!siri) |>
  colnames()

#' Plot correlation structure
corrplot(cor(bodyfat))

#' # Regression model with R2D2 prior
#'
#' We fit a model with all predictors. With the default flat or
#' independent normal priors, the prior would have a lot of posterior
#' mass for $R^2$ near 1. We use weakly informative R2D2 prior
#' @Zhang+etal:2022:R2D2 to include prior assumption that the
#' proportion of explained variance $R^2$ is unlikely to be very
#' close to 1. (Previously this case study used regularized horseshoe
#' prior @Piironen+Vehtari:RHS:2017, but for moderate number of
#' predictors we now favor R2D2 prior).
#| label: fitr
#| results: hide
fitr <- brm(siri ~ .,
            data = bodyfat,
            prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3)),
            seed = SEED,
            refresh = 0,
            control = list(adapt_delta = 0.9))

summary(fitr)

#' Make graphical posterior predictive check to check that the model
#' predictions make sense.
pp_check(fitr)

#' Kernel density estimate for the data and posterior predictive
#' replicates are similar. Although data do not have any negative $y$
#' values, the kernel density estimate is smoothing over to the
#' negative side. The predictive distribution of the model is not
#' constrained to be positive. We modify the model to include
#' truncation above 0 (log-normal model is commonly used for positive
#' targets, but in this case the distribution really is closer to
#' normal than log-normal distribution).
#| label: fitrt
#| results: hide
fitrt <- brm(siri | trunc(lb=0) ~ .,
             data = bodyfat,
             prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3)),
             seed = SEED,
             refresh = 0,
             control = list(adapt_delta = 0.9))

#' Now we don't get negative values in prediction.
pp_check(fitrt)

#' LOO-$R^2$ indicates that the model can explain big part of the
#' total variation.
loo_R2(fitrt) |> round(2)

#' Plot marginal posterior of the coefficients.
drawsrt <- as_draws_df(fitrt) |>
  subset_draws(variable = paste0('b_', prednames)) |>
  set_variables(variable = prednames)
drawsrt |> mcmc_areas()

#' We can see that the posterior of abdomen coefficient is far away
#' from zero and wrist coefficient is slightly away from zero, but
#' it's not as clear what other variables should be included. `weight`
#' has wide marginal overlapping zero, which hints potentially
#' relevant variable with correlation in joint posterior.
#'
#' Looking at the marginals has the problem that correlating variables
#' may have marginal posteriors overlapping zero while joint posterior
#' typical set does not include zero. Compare, for example, marginals
#' of `height` and `weight` above to their joint distribution below.
drawsrt |>
  subset(variable = c("height", "weight")) |>
  mcmc_scatter() +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0)

#' Projection predictive variable selection is easily made with
#' `cv_varsel` function, which also computes an LOO-CV estimate of
#' the predictive performance for the best models with certain number
#' of variables. @Heinze+etal:bodyfat "consider abdomen and height as
#' two central IVs [independent variables] for estimating body fat
#' proportion, and will not subject these two to variable selection."
#' We subject all variables to selection. We start with doing fast
#' LOO-CV only for the full data search.
#| results: hide
#| cache: true
fitrt_vs <- cv_varsel(fitrt, validate_search = FALSE)

#' Plot the estimated predictive performance of smaller models
#' compared to the full model.
plot(fitrt_vs, stats = c("R2"), deltas = "mixed")

#' Including abdomen, weight and wrist provide similar performance as
#' including all predictors. As the RMSE estimates don't get below the
#' full model performance, there is no overfitting in the selection
#' process, and we could continue with these variables. This would be
#' handy as the cross-validation over search paths is more time
#' consuming. However, we do here cross-validation over search paths
#' to show the stability of the selection process. To save time we do
#' the search only up to 4 included terms.
#| results: hide
#| cache: true
fitrt_cvvs <- cv_varsel(fitrt,
                        nterms_max = 4,
                        validate_search = TRUE)

#| results: hide
plot(fitrt_cvvs, stats = c("R2"), deltas = "mixed")

#' We see that in 98% LOO-folds the best 3 predictor model includes
#' `abdomen`, `weight` and `wrist`, and the performance is similar to
#' the full model performance. We can also get a suggestion for the
#' model size which is 3.
(nsel <- suggest_size(fitrt_cvvs, alpha = 0.1))
(vsel <- ranking(fitrt_cvvs, nterms_max = nsel)$fulldata)

#' As comparison, the model selected by @Heinze+etal:bodyfat had seven
#' variables `height` (fixed), `abdomen` (fixed), `wrist`, `age`,
#' `neck`, `forearm`, and `chest`.
#'
#' We can also examine the projected posterior.
#| results: hide
projrt <- project(fitrt_cvvs, nterms = nsel, ns = 4000)
projdraws <- as_draws_df(projrt) |>
  subset_draws(variable = paste0('b_', vsel)) |>
  set_variables(variable = vsel)

#' The marginals of projected posteriors look like this
mcmc_areas(projdraws,
           prob_outer = 0.99,
           area_method = "scaled height")

#' `projpred` is able to select a smaller set of variables which have
#' very similar predictive performance as the full model.
#' @Pavone+etal:2022 report also simulation results running `projpred`
#' and other methods many times using bootstrapped data sets.
#'
#' We can also look at the predictive intervals of the selected
#' submodel predictions
preds <- proj_predict(projrt) |>
  as_draws_rvars(preds)
bodyfat |>
  as_tibble() |>
  mutate(preds = preds) |>
  ggplot(aes(x = siri, ydist = preds)) +
  stat_pointinterval(alpha = 0.3, color = "steelblue") +
  geom_abline(linetype = "dashed", alpha = 0.5) +
  labs(x = "siri", y = "Prediction")

#' Currently, the `proj_predict()` is not including the truncation and
#' some predictive intervals go below zero.
#' 
#' # More predictors
#'
#' @Heinze+etal:bodyfat also write "In routine work, however, it is
#' not known a priori which covariates should be included in a model,
#' and often we are confronted with the number of candidate variables
#' in the range 10-30. This number is often too large to be
#' considered in a statistical model." I strongly disagree with this
#' as there are many statistical models working with more than million
#' candidate variables (see, e.g., @Peltola+etal:finite:2012). As the
#' bodyfat dataset proved to be quite easy in that sense that maximum
#' likelihood performed well compared to Bayesian approach,
#' @Pavone+etal:2022 report also simulation results with extra 87
#' completely irrelevant predictors.
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
#'
