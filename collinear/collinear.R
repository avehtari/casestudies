#' ---
#' title: "Bayesian version of Does model averaging make sense?"
#' author: "Aki Vehtari"
#' date: 2017-08-09
#' date-modified: today
#' date-format: iso
#' format:
#'   html:
#'     number-sections: true
#'     code-copy: true
#'     code-download: true
#'     code-tools: true
#' bibliography: ../casestudies.bib
#' csl: ../harvard-cite-them-right.csl
#' ---
#'
#' # Introduction
#'
#' This notebook was inspired by Andrew Tyre's blog post [Does model
#' averaging make
#' sense?](https://atyre2.github.io/2017/06/16/rebutting_cade.html).
#' Tyre discusses problems in current statistical practices in
#' ecology, focusing in multi-collinearity, model averaging and
#' measuring the relative importance of variables. Tyre's post is
#' commenting a paper [Model averaging and muddled multimodel
#' inferences](http://onlinelibrary.wiley.com/doi/10.1890/14-1639.1/full.)
#' In his blog post he uses maximum likelihood and AIC_c. Here we
#' provide a Bayesian approach for handling multicollinearity, model
#' averaging and measuring relative importance of variables using
#' packages
#' [`rstanarm`](https://cran.r-project.org/package=rstanarm),
#' [`bayesplot`](https://cran.r-project.org/package=bayesplot),
#' [`loo`](https://cran.r-project.org/package=loo) and
#' [`projpred`](https://cran.r-project.org/package=projpred). We
#' demonstrate the benefits of Bayesian posterior analysis [@BDA3] and
#' projection predictive approach
#' [@McLatchie+etal:2023:projpred_workflow;
#' @Piironen+etal:projpred:2020; @Piironen+Vehtari:2017a].
#'
#+ setup, include = FALSE
knitr::opts_chunk$set(cache=FALSE, message=FALSE, error=FALSE, warning=FALSE, comment=NA, out.width='95%')

#' **Load packages**
#| cache: FALSE
library(tibble)
library(rstanarm)
options(mc.cores = 1)
library(loo)
library(ggplot2)
library(GGally)
library(bayesplot)
theme_set(bayesplot::theme_default())
library(projpred)
SEED=87

#' # Data
#'
#' We generate the data used previously to illustrate
#' multi-collinearity problems.
N <- 200
set.seed(SEED)
sim_grouse <- tibble(
  pos_tot = runif(N, min = 0.8 ,max = 1.0),
  urban_tot = pmin(runif(N, min = 0.0, max = 0.02), 1.0 - pos_tot),
  neg_tot = (1.0 - pmin(pos_tot + urban_tot, 1)),
  x1 = pmax(pos_tot - runif(N, min = 0.05, max = 0.30), 0),
  x3 = pmax(neg_tot - runif(N, min = 0.0, max = 0.10), 0),
  x2 = pmax(pos_tot - x1 - x3/2, 0),
  x4 = pmax(1 - x1 - x2 - x3 - urban_tot, 0),
  y = rpois(N, exp(-5.8 + 6.3 * x1 + 15.2 * x2))) 

ggpairs(sim_grouse,
        diag = list(continuous = "densityDiag", discrete = "barDiag"),
        lower = list(combo = "points"),
        upper = list(combo = "blank", continuous = "blank"),
        axisLabels = "none")

#' Tyre writes:
#' 
#' > So there is a near perfect negative correlation between the
#' > things sage grouse like and the things they don't like, although it
#' > gets less bad when considering the individual covariates.
#'
#' # Bayesian inference
#'
#' From this point onwards we switch to Bayesian approach. The
#' [rstanarm
#' package](https://cran.r-project.org/package=rstanarm) provides
#' `stan_glm` function which accepts same arguments as `glm`, but
#' makes full Bayesian inference using Stan
#' ([mc-stan.org](https://mc-stan.org)). By default a weakly
#' informative Gaussian prior is used for weights.
#| label: fitg
#| results: hide
fitg <- stan_glm(y ~ x1 + x2 + x3 + x4,
                 family = poisson(),
                 data = sim_grouse,
                 na.action = na.fail,
                 seed = SEED,
                 refresh = 0)

#' Let's look at the summary:
summary(fitg)

#' We didn't get divergences, Rhat's are less than 1.01 and n_eff's
#' are useful (see, e.g., [RStan
#' workflow](http://mc-stan.org/users/documentation/case-studies/rstan_workflow.html)).
#' However, when we know that covariats are correlating we can get
#' even better performance by using QR decomposition (see, [The QR
#' Decomposition For Regression
#' Models](http://mc-stan.org/users/documentation/case-studies/qr_regression.html)).
#| label: fitg-QR
#| results: hide
fitg <- stan_glm(y ~ x1 + x2 + x3 + x4,
                 family=poisson(),
                 QR=TRUE,
                 data = sim_grouse,
                 na.action = na.fail,
                 seed = SEED,
                 refresh = 0)

#' Let's look at the summary and plot:
summary(fitg)

#' Use of QR decomposition greatly improved sampling efficiency and
#' we continue with this model.
mcmc_areas(as.matrix(fitg), prob_outer = .99)

#' All 95% posterior intervals are overlapping 0 and it seems we have
#' the same collinearity problem as with maximum likelihood estimates.
#'
#' Looking at the pairwise posteriors we can see high correlations
mcmc_pairs(as.matrix(fitg), pars = c("x1", "x2", "x3", "x4"))

#' If look more carefully on of the subplots, we see that although
#' marginal posterior intervals overlap 0, the joint posterior is not
#' overlapping 0.
mcmc_scatter(as.matrix(fitg), pars = c("x1", "x2")) +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0)

#' Based on the joint distributions all the variables would be
#' relevant. To make predictions we don't need to make variable
#' selection, we just integrate over the uncertainty (kind of
#' continuous model averaging).
#'
#' In case of even more variables with some being relevant and some
#' irrelevant, it will be difficult to analyse joint posterior to see
#' which variables are jointly relevant. We can easily test whether
#' any of the covariates are useful by using cross-validation to
#' compare to a null model,
#| label: fitg0
fitg0 <- stan_glm(y ~ 1,
                  family=poisson(),
                  data = sim_grouse,
                  na.action = na.fail,
                  seed=SEED,
                  refresh=0)

(loog <- loo(fitg))
(loog0 <- loo(fitg0))
loo_compare(loog0, loog)

#' Based on cross-validation covariates together contain significant
#' information to improve predictions.
#'
#' # Variable selection
#'
#' We might want to choose some variables 1) because we don't want to
#' observe all the variables in the future (e.g. due to the
#' measurement cost), or 2) we want to most relevant variables which
#' we define here as a minimal set of variables which can provide
#' similar predictions to the full model.
#'
#' Tyre used AIC_c to estimate the model performance. In Bayesian
#' setting we could use Bayesian cross-validation or WAIC, but we
#' don't recommend them for variable selection as discussed by
#' @Piironen+Vehtari:2017a. The reason for not using Bayesian CV or
#' WAIC is that the selection process uses the data twice, and in case
#' of large number variable combinations the selection process
#' overfits and can produce really bad models. Using the usual
#' posterior inference given the selected variables ignores that the
#' selected variables are conditonal on the selection process and
#' simply setting some variables to 0 ignores the uncertainty related
#' to their relevance.
#'
#' @Piironen+Vehtari:2017a also show that a projection predictive
#' approach can be used to make a model reduction, that is, choosing a
#' smaller model with some coefficients set to 0. The projection
#' predictive approach solves the problem how to do inference after
#' the selection. The solution is to project the full model posterior
#' to the restricted subspace. See more by
#' @Piironen+etal:projpred:2020 and
#' @McLatchie+etal:2023:projpred_workflow.
#'
#' We make the projective predictive variable selection using the
#' previous full model. A fast leave-one-out cross-validation approach
#' [@Vehtari+etal:PSIS-LOO:2017] is used to choose the model size. As
#' the number of observations is large compared to the number of
#' covariates, we estimate the performance using LOO-CV only along the
#' search path (`validate_search=FALSE`), as we may assume that the
#' overfitting in search is negligible (see more about this in
#' @McLatchie+etal:2023:projpred_workflow).
#| label: cvvs
#| results: hide
cvvs <- cv_varsel(fitg,
                  method = "forward",
                  cv_method = "loo",
                  validate_search = FALSE)

#' We can now look at the estimated predictive performance of smaller
#' models compared to the full model.
plot(cvvs, stats = c("elpd", "rmse"))

#' As the estimated predictive performance is not going much above the
#' reference model performance, we know that the use of option
#' `validate_search=FALSE` was safe (see more in
#' @McLatchie+etal:2023:projpred_workflow).
#'
#' And we get a LOO based recommendation for the model size to choose
(nsel <- suggest_size(cvvs))
(vsel <- solution_terms(cvvs)[1:nsel])

#' We see that `r nsel` variables is enough to get the same predictive
#' accuracy as with all 4 variables.
#'
#' Next we form the projected posterior for the chosen model.
projg <- project(cvvs, nv = nsel, ns = 4000)
projdraws <- as.matrix(projg)
round(colMeans(projdraws),1)
round(posterior_interval(projdraws),1)

#' This looks good as the true values are intercept=-5.8, x2=15.2,
#' x1=6.3.
mcmc_areas(projdraws, pars=c("(Intercept)", vsel))

#' Even if we started with a model which had due to a collinearity
#' difficult to interpret posterior, the projected posterior is able to
#' match closely the true values. The necessary information was in the
#' full model and with the projection we were able to form the
#' projected posterior which we should use if x3 and x4 are set to 0.
#'
#' Back to the Tyre's question "Does model averaging make sense?". If
#' we are interested just in good predictions we can do continuous
#' model averaging by using suitable priors and by integrating over
#' the posterior. If we are interested in predictions, then we don't
#' first average weights (ie posterior mean), but use all weight values
#' to compute predictions and do the averaging of the predictions. All
#' this is automatic in Bayesian framework.
#'
#' Tyre also commented on the problems of measuring variable
#' importance. The projection predictive approach above is derived
#' using decision theory and is very helpful for measuring relevancy
#' and choosing relevant variables. Tyre did not comment about the
#' inference after selection although it is also known problem in
#' variable selection. The projection predictive approach above solves
#' that problem, too.
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
#' * Code for generating the data by [Andrew Tyre](https://atyre2.github.io/2017/06/16/rebutting_cade.html)
#'
