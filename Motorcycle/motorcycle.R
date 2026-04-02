#' ---
#' title: "Gaussian process demonstration with Stan"
#' author: "Aki Vehtari"
#' date: 2021-01-28
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
#' This case study demonstrates covariance matrix and basis function
#' implementation of Gaussian process model in Stan.
#'
#' The Stan code for the covariance matrix approach is based on [Stan
#' User’s Guide Chapter on Gaussian
#' Processes](https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html)
#' by Stan Development Team (2026). Good reading material on Gaussian processes
#' is Gaussian Processes for Machine Learning by @Rasmussen-Williams:2006.
#'
#' The basics of the Hilbert space basis function approximation is
#' based on "Practical Hilbert space approximate Bayesian Gaussian
#' processes for probabilistic programming" by @Riutort-Mayol:2023:HSGP.
#'
#' Data are measurements of head acceleration in a simulated
#' motorcycle accident, used to test crash helmets. We use dataset
#' from MASS R package [@Venables-Ripley:2002], which mentions
#' @Silverman:1985 as the original source.
#'
#' Data are modelled first with normal distribution having Gaussian process
#' prior on mean:
#' $$
#' y \sim \operatorname{normal}(f(x), \sigma)\\
#' f \sim GP(0, K_1)\\
#' \sigma \sim \operatorname{normal}^{+}(0, 1),
#' $$
#' and then with normal distribution having Gaussian process
#' prior on mean and log standard deviation:
#' $$
#' y \sim \operatorname{normal}(f(x), \exp(g(x))\\
#' f \sim GP(0, K_1)\\
#' g \sim GP(0, K_2).
#' $$
#'
#' We compare different approximate posterior inference methods:
#' optimization, Laplace, Pathfinder [@Zhang+etal:2022:pathfinder],
#' HMC/NUTS Markov chain Monte Carlo (MCMC; @Hoffman+Gelman:2014:nuts,
#' @StanTeam:2026), and automatic differentiation variational
#'   inference (ADVI; @Kucukelbir-Tran-Ranganath-etal:2017)
#' 
#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA, cache=FALSE)

#' **Load packages**
#| cache: FALSE
library(rprojroot)
root <- has_file(".Bayesian-Workflow-root")$make_fix_file()
library(cmdstanr) 
options(stanc.allow_optimizations = TRUE, mc.cores = 4)
# CmdStanR output directory makes Quarto cache to work
dir.create(root("Motorcycle", "stan_output"), showWarnings = FALSE)
options(cmdstanr_output_dir = root("Motorcycle", "stan_output"))
#library(posterior)
devtools::load_all("~/proj/posterior")
library(loo)
library(tidybayes)
options(pillar.neg = FALSE, pillar.subtle=FALSE, pillar.sigfig=2)
library(tidyr) 
library(dplyr) 
library(ggplot2)
library(ggrepel)
library(patchwork)
library(latex2exp)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size=16))
set1 <- RColorBrewer::brewer.pal(7, "Set1")
library(tictoc)
mytoc <- \() {toc(func.toc = \(tic, toc, msg) {
  sprintf("%s took %s sec", msg, as.character(signif(toc-tic, 2))) })}
SEED <- 48927 # set random seed for reproducibility

print_stan_file <- function(file) {
  code <- readLines(file)
  if (isTRUE(getOption("knitr.in.progress")) &
        identical(knitr::opts_current$get("results"), "asis")) {
    # In render: emit as-is so Pandoc/Quarto does syntax highlighting
    block <- paste0("```stan", "\n", paste(code, collapse = "\n"), "\n", "```")
    knitr::asis_output(block)
  } else {
    writeLines(code)
  }
}

#' # Simulated motorcycle accident data
#' 
#' Load data
data(mcycle, package = "MASS")
head(mcycle)

#' Plot data
plot_mcycle <- ggplot(mcycle, aes(x = times, y = accel)) +
  geom_point() +
  labs(x = "Time (ms)", y = "Acceleration (g)")
plot_mcycle

#' # Homoskedastic GP with covariance matrices
#'
#' We start with a simpler homoskedastic residual Gaussian process
#' model (residual variance $\sigma$ does not depend on $x$).
#' $$
#' y \sim \operatorname{normal}(f(x), \sigma)\\
#' f \sim GP(0, K_1)\\
#' \sigma \sim \operatorname{normal}^{+}(0, 1).
#' $$
#' In this case the latent values $f$ can be integrated out
#' analytically, and only the covariance function lengthscale and
#' magnitude and residual scale parameters need to be handled
#' numerically.
#' 

#' ## Model code
file_gpcovf <- "gpcovf.stan"
#| results: asis
print_stan_file(file_gpcovf)

#' Compile Stan model
#| label: model_gpcovf
#| results: hide
model_gpcovf <- cmdstan_model(stan_file = file_gpcovf)

#' Data to be passed to Stan
standata_gpcovf <- list(x = mcycle$times, 
                        x2 = mcycle$times, 
                        y = mcycle$accel, 
                        N = length(mcycle$times), 
                        N2 = length(mcycle$times))

#' ## Optimize and find MAP estimate in the unconstrained space (jacobian=TRUE)
#'
#' As the lengthscale, magnitude, and residual scale parameters are
#' constrained to be positive, Stan does the inference in transformed
#' unconstrained space (for positivity constraint using
#' logarithm). For proper Bayesian inference, we need to take into
#' account the Jacobian of the transformation. For historical reasons,
#' Stan optimization does not include Jacobian by default. See
#' [Laplace method and Jacobian of parameter
#' transformation](https://users.aalto.fi/~ave/casestudies/Jacobian/jacobian.html)
#' for more information.
tic("Find MAP estimate for homoskedastic GP with covariance matrices")
#| label: opt_gpcovf
#| results: hide
opt_gpcovf <- model_gpcovf$optimize(data = standata_gpcovf, 
                                    jacobian = TRUE, 
                                    init = 0.01, 
                                    algorithm = "bfgs")
#'
mytoc()

#' Check whether parameters have reasonable values
odraws_gpcovf <- as_draws_rvars(opt_gpcovf$draws())
subset(odraws_gpcovf, variable = c("sigma_", "lengthscale_", "sigma"), regex = TRUE)

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(odraws_gpcovf$f), 
         sigma = mean(odraws_gpcovf$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' The model fit given optimized parameters, looks reasonable
#' considering the use of homoskedastic residual model. Optimization
#' is likely to be fine as we have many more observations (133) than
#' parameters (3).
#' 
#' ## Sample from the Laplace approximation (normal at the mode in the unconstrained space)
#'
#' As we have many more observations (133) than parameters (3), it is
#' likely that the posterior is close to normal. Laplace approximation
#' finds the mode, computes Hessian at the mode, and form a normal
#' approximation with mean at the mode and covariance matrix computed
#' from the Hessian. Stan's Laplace method returns draws from the
#' normal approximation, which can be used as usual posterior draws.
tic("Sample from the Laplace approximation for homoskedastic GP with covariance matrices")
#| label: lap_gpcovf
#| results: hide
lap_gpcovf <- model_gpcovf$laplace(data = standata_gpcovf, 
                                   mode = opt_gpcovf, 
                                   draws = 1000)
#'
mytoc()

#' Check whether parameters have reasonable values
ldraws_gpcovf <- as_draws_rvars(lap_gpcovf$draws())
summarise_draws(subset(ldraws_gpcovf, 
                       variable = c("sigma_", "lengthscale_", "sigma"), 
                       regex = TRUE), 
                default_summary_measures())

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(ldraws_gpcovf$f), 
         sigma = mean(ldraws_gpcovf$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' We'll later compare these to MCMC result.

#' ## Sample from the Pathfinder approximation
#'
#' Pathfinder starts many optimizations each using L-BFGS
#' algorithm. L-BFGS produces as a side product a normal approximation
#' of the posterior. Pathfinder uses stochastic Kullback-Leibler
#' divergence to select the best normal approximation along the
#' path. The best normal approximations from each path form a mixture
#' of normals which is used as importance sampling proposal
#' distribution to get approximate posterior draws. Pathfinder works
#' better than Laplace method if the posterior is highly skewed, but
#' is still limited by using normal approximation. Pathfinder can be
#' useful for initializing MCMC.
tic("Sample from the Pathfinder approximation for homoskedastic GP with covariance matrices")
#| label: pth_gpcovf
pth_gpcovf <- model_gpcovf$pathfinder(data = standata_gpcovf, 
                                      init = 0.01, 
                                      num_paths = 10, 
                                      single_path_draws = 200, 
                                      history_size = 100, 
                                      max_lbfgs_iters = 100,
                                      refresh = 0)
#'
mytoc()

#' Check whether parameters have reasonable values
pdraws_gpcovf <- as_draws_rvars(pth_gpcovf$draws())
summarise_draws(subset(pdraws_gpcovf, 
                       variable = c("sigma_", "lengthscale_", "sigma"), 
                       regex = TRUE), 
                default_summary_measures())

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(pdraws_gpcovf$f), 
         sigma = mean(pdraws_gpcovf$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' We'll later compare these to MCMC result.
#' 
#' ## Sample using HMC/NUTS
#'
#' We use multinomial No-U-Turn sampler variant of Hamiltonian Monte
#' Carlo algorithm as implemented in Stan.  We sample fewer than
#' default number of iterations to keep sampling faster, and this is
#' enough for the demonstration.
tic("MCMC sample from the posterior of homoskedastic GP with covariance matrices")
#| label: fit_gpcovf
#| results: hide
#| cache: true
fit_gpcovf <- model_gpcovf$sample(data = standata_gpcovf, 
                                  iter_warmup = 500, 
                                  iter_sampling = 250, 
                                  refresh = 100)
#'
mytoc()

#' Check whether parameters have reasonable values
draws_gpcovf <- as_draws_rvars(fit_gpcovf$draws())
summarise_draws(subset(draws_gpcovf, 
                       variable = c("sigma_", "lengthscale_", "sigma"), 
                       regex = TRUE))

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(draws_gpcovf$f), 
         sigma = mean(draws_gpcovf$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' The model fit given integrated parameters looks quite similar to the
#' optimized one.
#' 
#' Compare the posterior draws to the optimized parameters and to
#' Laplace and Pathfinder approximations
odraws_gpcovf <- as_draws_df(opt_gpcovf$draws())
p1 <- draws_gpcovf |>
  as_draws_df() |>
  ggplot(aes(x = lengthscale_f, y = sigma_f)) +
  geom_point(color = set1[2]) +
  lims(x = c(0.21, 0.62), y = c(0.5, 2.8)) +
  geom_point(data = odraws_gpcovf,
             color = set1[1],
             size = 5) +
  annotate("text",
           label = "Posterior draws",
           x = median(draws_gpcovf$lengthscale_f), 
           y = max(draws_gpcovf$sigma_f) + 0.1, 
           hjust = 0.5,
           color = set1[2],
           size = 5) +
  annotate("text",
           label = "Optimized",
           x = odraws_gpcovf$lengthscale_f + 0.01, 
           y = odraws_gpcovf$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 5)

p2 <- ldraws_gpcovf |>
  as_draws_df() |>
  ggplot(aes(x = lengthscale_f, y = sigma_f)) +
  geom_point(color = set1[2]) +
  lims(x = c(0.21, 0.62), y = c(0.5, 2.8)) +
  geom_point(data = odraws_gpcovf, color = set1[1], size = 5) +
  annotate("text",
           label = "Laplace draws",
           x = median(draws_gpcovf$lengthscale_f), 
           y = max(draws_gpcovf$sigma_f) + 0.1, 
           hjust = 0.5,
           color = set1[2],
           size = 5) +
  annotate("text",
           label = "Optimized",
           x = odraws_gpcovf$lengthscale_f + 0.01, 
           y = odraws_gpcovf$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 5)

p3 <- pdraws_gpcovf |>
  as_draws_df() |>
  ggplot(aes(x = lengthscale_f, y = sigma_f)) +
  geom_jitter(color = set1[2]) +
  lims(x = c(0.21, 0.62), y = c(0.5, 2.8)) +
  geom_point(data = odraws_gpcovf, color = set1[1], size = 5) +
  annotate("text",
           label = "Pathfinder draws",
           x = median(draws_gpcovf$lengthscale_f), 
           y = max(draws_gpcovf$sigma_f) + 0.1, 
           hjust = 0.5,
           color = set1[2],
           size = 5) +
  annotate("text",
           label = "Optimized",
           x = odraws_gpcovf$lengthscale_f + 0.01, 
           y = odraws_gpcovf$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 5)

#| out-width: 100%
p1 + p2 + p3 + plot_layout(axes = "collect")

#' The optimization result is in the middle of the posterior and quite
#' well representative of the low dimensional posterior (in higher
#' dimensions the mean or mode of the posterior is not likely to be
#' representative). Laplace and Pathfinder approximations resemble the
#' MCMC result.
#' 
#' Compare optimized and posterior predictive distributions
odraws_gpcovf <- as_draws_rvars(opt_gpcovf$draws())
mcycle |>
  mutate(Ef = mean(draws_gpcovf$f), 
         sigma = mean(draws_gpcovf$sigma), 
         Efo = mean(odraws_gpcovf$f), 
         sigmao = mean(odraws_gpcovf$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Efo), color = set1[2]) +
  geom_line(aes(y = Efo- 2*sigmao), color = set1[2], linetype = "dashed") +
  geom_line(aes(y = Efo+ 2*sigmao), color = set1[2], linetype = "dashed")

#' Optimization based predictive distribution is close to posterior
#' predictive distribution.  In general GP covariance function and
#' observation model parameters can be quite safely optimized if there
#' are only a few of them and thus marginal posterior is low
#' dimensional and the number of observations is relatively high.
#' 
#' Compare Laplace approximated and posterior predictive distributions
ldraws_gpcovf <- as_draws_rvars(lap_gpcovf$draws())
mcycle |>
  mutate(Ef = mean(draws_gpcovf$f), 
         sigma = mean(draws_gpcovf$sigma), 
         Efo = mean(ldraws_gpcovf$f), 
         sigmao = mean(ldraws_gpcovf$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Efo), color = set1[2]) +
  geom_line(aes(y = Efo- 2*sigmao), color = set1[2], linetype = "dashed") +
  geom_line(aes(y = Efo+ 2*sigmao), color = set1[2], linetype = "dashed")

#' There is no visible difference between Laplace approximated and
#' MCMC posterior predictive distributions.
#' 
#' Compare Pathfinder approximated and posterior predictive distributions
pdraws_gpcovf <- as_draws_rvars(pth_gpcovf$draws())
mcycle |>
  mutate(Ef = mean(draws_gpcovf$f), 
         sigma = mean(draws_gpcovf$sigma), 
         Efo = mean(pdraws_gpcovf$f), 
         sigmao = mean(pdraws_gpcovf$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Efo), color = set1[2]) +
  geom_line(aes(y = Efo- 2*sigmao), color = set1[2], linetype = "dashed") +
  geom_line(aes(y = Efo+ 2*sigmao), color = set1[2], linetype = "dashed")

#' There is no visible difference between Pathfinder approximated and
#' MCMC posterior predictive distributions.
#' 
#' ## 10% of data
#'
#' To demonstrate that the optimization is not always safe, we next use
#' only (about) 10% of the data (14 observations) for model fitting.
#' 
#' Data to be passed to Stan
mcycle_10p <- mcycle[seq(1, 133, by = 10), ]
standata_10p <- list(x = mcycle_10p$times, 
                     x2 = mcycle$times, 
                     y = mcycle_10p$accel, 
                     N = length(mcycle_10p$times), 
                     N2 = length(mcycle$times))

#' Optimize and find MAP estimate
#| label: opt_10p
#| results: hide
opt_10p <- model_gpcovf$optimize(data = standata_10p, 
                                 init = 0.1, 
                                 jacobian = TRUE, 
                                 algorithm = "lbfgs")

#' Check whether parameters have reasonable values
odraws_10p <- as_draws_rvars(opt_10p$draws())
subset(odraws_10p, variable = c("sigma_", "lengthscale_", "sigma"), regex = TRUE)

#' Compare the model to the data
mcycle_10p |>
  ggplot(aes(x = times, y = accel)) +
  geom_point() +
  labs(x = "Time (ms)", y = "Acceleration (g)") +
  geom_line(data = mcycle,
            aes(x = times, y = mean(odraws_10p$f)),
            color = set1[1]) +
  geom_line(data = mcycle,
            aes(x = times, y = mean(odraws_10p$f- 2*odraws_10p$sigma)),
            color = set1[1], 
            linetype = "dashed") +
  geom_line(data = mcycle,
            aes(x = times, y = mean(odraws_10p$f + 2*odraws_10p$sigma)),
            color = set1[1], 
            linetype = "dashed")

#' The model fit is clearly over-fitted and over-confident.
#' 
#' ## Sample from the Laplace approximation (normal at the mode in the unconstrained space)
#'
#| label: lap_10p
#| results: hide
lap_10p <- model_gpcovf$laplace(data = standata_10p, 
                                mode = opt_10p, 
                                draws = 1000)

#' Check whether parameters have reasonable values
ldraws_10p <- as_draws_rvars(lap_10p$draws())
summarise_draws(subset(ldraws_10p, 
                       variable = c("sigma_", "lengthscale_", "sigma"), 
                       regex = TRUE), 
                default_summary_measures())

#' Compare the model to the data
mcycle_10p |>
  ggplot(aes(x = times, y = accel)) +
  geom_point() +
  labs(x = "Time (ms)", y = "Acceleration (g)") +
  geom_line(data = mcycle, aes(x = times, y = mean(ldraws_10p$f)), color = set1[1]) +
  geom_line(data = mcycle, aes(x = times, y = mean(ldraws_10p$f- 2*ldraws_10p$sigma)), color = set1[1], 
            linetype = "dashed") +
  geom_line(data = mcycle, aes(x = times, y = mean(ldraws_10p$f + 2*ldraws_10p$sigma)), color = set1[1], 
            linetype = "dashed")

#' As the optimization is over-fitted, normal approximation at the
#' mode is unlikely to be great, but the fitted function is slightly
#' smoother as we average over the draws from the normal
#' approximation.
#' 
#' ## Sample from the Pathfinder approximation
#| label: pth_10p
pth_10p <- model_gpcovf$pathfinder(data = standata_10p, 
                                   init = 0.1,
                                   num_paths = 20, 
                                   single_path_draws = 100, 
                                   history_size = 100, 
                                   max_lbfgs_iters = 100,
                                   refresh = 0)

#' Check whether parameters have reasonable values
pdraws_10p <- as_draws_rvars(pth_10p$draws())
pdraws_10p <- pdraws_10p |>
    mutate_variables(lw = lp__-lp_approx__, 
                     w = exp(lw-max(lw)), 
                     ws = pareto_smooth(w, tail = "right", r_eff = 1))
pdraws_10p <- pdraws_10p |>
  weight_draws(weights = extract_variable(pdraws_10p, "ws"), log = FALSE) |>
  resample_draws()
summarise_draws(subset(pdraws_10p, 
                       variable = c("sigma_", "lengthscale_", "sigma"), 
                       regex = TRUE), 
                default_summary_measures())

#' Compare the model to the data
mcycle_10p |>
  ggplot(aes(x = times, y = accel)) +
  geom_point() +
  labs(x = "Time (ms)", y = "Acceleration (g)") +
  geom_line(data = mcycle, aes(x = times, y = mean(pdraws_10p$f)), color = set1[1]) +
  geom_line(data = mcycle, aes(x = times, y = mean(pdraws_10p$f- 2*pdraws_10p$sigma)), color = set1[1], 
            linetype = "dashed") +
  geom_line(data = mcycle, aes(x = times, y = mean(pdraws_10p$f + 2*pdraws_10p$sigma)), color = set1[1], 
            linetype = "dashed")

#' Pathfinder approximation is not necessarily centered at the mode,
#' and the posterior prediction is now smoother without clear
#' overfitting.
#'
#' ## Sample using HMC/NUTS
#| label: fit_10p
#| results: hide
#| cache: true
fit_10p <- model_gpcovf$sample(data = standata_10p, 
                               iter_warmup = 1000, 
                               iter_sampling = 1000, 
                               adapt_delta = 0.95, 
                               refresh = 100)

#' Check whether parameters have reasonable values
draws_10p <- as_draws_rvars(fit_10p$draws())
summarise_draws(subset(draws_10p, variable = c("sigma_", "lengthscale_", "sigma"), 
                       regex = TRUE))

#' Compare the model to the data
mcycle_10p |>
  ggplot(aes(x = times, y = accel)) +
  geom_point() +
  labs(x = "Time (ms)", y = "Acceleration (g)") +
  geom_line(data = mcycle, aes(x = times, y = mean(draws_10p$f)), color = set1[1]) +
  geom_line(data = mcycle, aes(x = times, y = mean(draws_10p$f- 2*draws_10p$sigma)), color = set1[1], 
            linetype = "dashed") +
  geom_line(data = mcycle, aes(x = times, y = mean(draws_10p$f + 2*draws_10p$sigma)), color = set1[1], 
            linetype = "dashed")

#' The posterior predictive distribution is quite smooth and
#' shows the uncertainty due to having only a small number of
#' observations.
#' 
#' Compare the posterior draws to the optimized parameters
odraws_10p <- as_draws_df(opt_10p$draws())
p1 <- draws_10p |>
  thin_draws(thin = 4) |>
  as_draws_df() |>
  ggplot(aes(x = sigma, y = sigma_f)) +
  geom_point(color = set1[2]) +
  lims(x = c(0, 90), y = c(0, 3.3)) +
  geom_point(data = odraws_10p, color = set1[1], size = 4) +
  annotate("text",
           label = "Posterior draws",
           x = min(draws_10p$sigma), 
           y = max(draws_10p$sigma_f) + 0.1, 
           hjust = 0,
           color = set1[2],
           size = 5) +
  annotate("text",
           label = "Optimized",
           x = odraws_10p$sigma+ 0.01, 
           y = odraws_10p$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 5)

#' Compare the posterior draws to the optimized parameters
ldraws_10p <- as_draws_df(lap_10p$draws())
p2 <- ldraws_10p |>
  as_draws_df() |>
  ggplot(aes(x = sigma, y = sigma_f)) +
  geom_point(color = set1[2]) +
  lims(x = c(0, 90), y = c(0, 3.3)) +
  geom_point(data = odraws_10p, color = set1[1], size = 4) +
  annotate("text",
           label = "Laplace draws",
           x = min(draws_10p$sigma), 
           y = max(draws_10p$sigma_f) + 0.1, 
           hjust = 0,
           color = set1[2],
           size = 5) +
  annotate("text", x = odraws_10p$sigma+ 0.01, 
           label = "Optimized",
           y = odraws_10p$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 5)

#' Compare the posterior draws to the optimized parameters
pdraws_10p <- as_draws_df(pdraws_10p)
p3 <- pdraws_10p |>
  as_draws_df() |>
  ggplot(aes(x = sigma, y = sigma_f)) +
  geom_jitter(color = set1[2]) +
  lims(x = c(0, 90), y = c(0, 3.3)) +
  geom_point(data = odraws_10p, color = set1[1], size = 4) +
  annotate("text",
           label = "Pathfinder draws",
           x = min(draws_10p$sigma), 
           y = max(draws_10p$sigma_f) + 0.1, 
           hjust = 0,
           color = set1[2],
           size = 5) +
  annotate("text",
           label = "Optimized",
           x = odraws_10p$sigma+ 0.01, 
           y = odraws_10p$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 5)

#| out-width: 100%
p1 + p2 + p3 + plot_layout(axes = "collect")

#' The optimization result is in the edge of the posterior close to
#' zero residual scale. While there are posterior draws close to zero,
#' integrating over the wide posterior takes into account the
#' uncertainty and the predictions thus are more uncertain,
#' too. Approximate integration is helpful, too. As the Laplace method
#' makes the normal approximation in the unconstrained space, the
#' approximate distribution in the constrained space can be skewed. 
#' 
#' Compare optimized and posterior predictive distributions
odraws_10p <- as_draws_rvars(opt_10p$draws())
mcycle |>
  mutate(Ef = mean(draws_10p$f), 
         sigma = mean(draws_10p$sigma), 
         Efo = mean(odraws_10p$f), 
         sigmao = mean(odraws_10p$sigma)) |>
  ggplot(aes(x = times, y = accel)) +
  geom_point() +
  labs(x = "Time (ms)", y = "Acceleration (g)") +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Efo), color = set1[2]) +
  geom_line(aes(y = Efo- 2*sigmao), color = set1[2], linetype = "dashed") +
  geom_line(aes(y = Efo+ 2*sigmao), color = set1[2], linetype = "dashed")

#' The figure shows the model prediction given 10% of data, but also
#' the full data as test data. The optimized model is over-fitted and
#' overconfident. Even if the homoskedastic residual is wrong here, 
#' the posterior predictive interval covers most of the observation
#' (and in case of good calibration should not cover them all).
#' 
#' Compare Laplace approximated and posterior predictive distributions
ldraws_10p <- as_draws_rvars(lap_10p$draws())
mcycle |>
  mutate(Ef = mean(draws_10p$f), 
         sigma = mean(draws_10p$sigma), 
         Efo = mean(ldraws_10p$f), 
         sigmao = mean(ldraws_10p$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Efo), color = set1[2]) +
  geom_line(aes(y = Efo- 2*sigmao), color = set1[2], linetype = "dashed") +
  geom_line(aes(y = Efo+ 2*sigmao), color = set1[2], linetype = "dashed")

#' Laplace approximated predictive distribution is narrower than MCMC
#' based predictive distribution.
#' 

#' Compare Pathfinder approximated and posterior predictive distributions
pdraws_10p <- as_draws_rvars(pdraws_10p)
mcycle |>
  mutate(Ef = mean(draws_10p$f), 
         sigma = mean(draws_10p$sigma), 
         Efo = mean(pdraws_10p$f), 
         sigmao = mean(pdraws_10p$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Efo), color = set1[2]) +
  geom_line(aes(y = Efo- 2*sigmao), color = set1[2], linetype = "dashed") +
  geom_line(aes(y = Efo+ 2*sigmao), color = set1[2], linetype = "dashed")

#' Pathfinder approximated predictive distribution is just slightly
#' narrower than MCMC based predictive distribution. 
#' 
#' # Heteroskedastic GP with covariance matrices
#'
#' We next make a model with heteroskedastic residual model using
#' Gaussian process prior also for the logarithm of the residual
#' scale:
#' $$
#' y \sim \operatorname{normal}(f(x), \exp(g(x))\\
#' f \sim GP(0, K_1)\\
#' g \sim GP(0, K_2).
#' $$
#'
#' Now there is no analytical solution as GP prior through the
#' exponential function is not a conjugate prior. In this case we
#' present the latent values of $f$ and $g$ explicitly and sample from the
#' joint posterior of the covariance function parameters, and the
#' latent values. It would be possible also to use Laplace, 
#' variational inference, or expectation propagation to integrate over
#' the latent values, but that is another story.
#' 
#' ## Model code
file_gpcovfg <- "gpcovfg.stan"
#| results: asis
print_stan_file(file_gpcovfg)

#' Compile Stan model
#| results: hide
model_gpcovfg <- cmdstan_model(stan_file = file_gpcovfg, 
                               compile_model_methods = TRUE, 
                               force_recompile = TRUE)


#' Data to be passed to Stan
standata_gpcovfg <- list(x = mcycle$times, 
                         y = mcycle$accel, 
                         N = length(mcycle$times))

#' ## Optimize and find MAP estimate
tic("Find MAP estimate for heteroskedastic GP with covariance matrices")
#| label: opt_gpcovfg
#| results: hide
opt_gpcovfg <- model_gpcovfg$optimize(data = standata_gpcovfg, 
                                      jacobian = TRUE, 
                                      init = 0, 
                                      algorithm = "bfgs")
#+
mytoc()

#' Check whether parameters have reasonable values
odraws_gpcovfg <- as_draws_rvars(opt_gpcovfg$draws())
subset(odraws_gpcovfg, variable = c("sigma_", "lengthscale_"), regex = TRUE)

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(odraws_gpcovfg$f), 
         sigma = mean(odraws_gpcovfg$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' The optimization overfits, as we are now optimizing the joint
#' posterior of 2 covariance function parameters and 2 x 133 latent
#' values while having only 133 observations.
#'
#' ## Sample from the Laplace approximation (normal at the mode in the unconstrained space)
tic("Sample from the Laplace approximation for heteroskedastic GP with covariance matrices")
#| label: lap_gpcovfg
#| results: hide
lap_gpcovfg <- model_gpcovfg$laplace(data = standata_gpcovfg, 
                                     mode = opt_gpcovfg, 
                                     draws = 1000)
#+
mytoc()

#' Check whether parameters have reasonable values
ldraws_gpcovfg <- as_draws_rvars(lap_gpcovfg$draws())
summarise_draws(subset(ldraws_gpcovfg, 
                       variable = c("sigma_", "lengthscale_"), 
                       regex = TRUE), 
                default_summary_measures())

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(ldraws_gpcovfg$f), 
         sigma = mean(ldraws_gpcovfg$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' Drawing from the normal approximation doesn't help, if the mode is
#' far from sensible values.
#' 
#' ## Sample from the Pathfinder approximation
tic("Sample from the Pathfinder approximation for heteroskedastic GP with covariance matrices")
#| label: pth_gpcovfg
#| cache: true
pth_gpcovfg <- model_gpcovfg$pathfinder(data = standata_gpcovfg, 
                                        init = 0.2, 
                                        num_paths = 20, 
                                        single_path_draws = 200, 
                                        history_size = 100, 
                                        max_lbfgs_iters = 100,
                                        refresh = 0)
#+
mytoc()

#' The Pathfinder diagnostic message shows that Pareto k value
#' [@Vehtari+etal:PSIS:2024] is very high (common diagnostic threshold
#' is 0.7).

#' Check whether parameters have reasonable values
pdraws_gpcovfg <- as_draws_rvars(pth_gpcovfg$draws())
summarise_draws(subset(pdraws_gpcovfg, 
                       variable = c("sigma_", "lengthscale_"), 
                       regex = TRUE), 
                default_summary_measures())

#' By default Stan Pathfinder method does Pareto smoothed importance
#' re-sampling in the end. If the diagnostic Pareto-k value is very
#' high, it is likely that only a draws dominate. We can check the
#' number of unique draws
length(unique(extract_variable(pdraws_gpcovfg, "lp__")))

#' In this case, there are
#' `r length(unique(extract_variable(pdraws_gpcovfg, "lp__")))`
#' unique draws.

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(pdraws_gpcovfg$f), 
         sigma = mean(pdraws_gpcovfg$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' As indicated by very high Pareto-k values, Pathfinder approximation
#' is not accurate, but it is better than optimization and Laplace.
#' 
#' ## Sample using dynamic HMC
#'
#'  Although Pathfinder approximation is not accurate, the results are
#'  closer to where the most of posterior mass is than random
#'  initialization, and thus Pathfinder draws can be used as initial
#'  values for MCMC, which can improve the convergence speed.
tic("MCMC sample from the posterior of heteroskedastic GP with covariance matrices")
#| label: fit_gpcovfg
#| results: hide
#| cache: true
fit_gpcovfg <- model_gpcovfg$sample(data = standata_gpcovfg, 
                                    iter_warmup = 100, 
                                    iter_sampling = 200, 
                                    refresh = 20, 
                                    init = pdraws_gpcovfg)
#+
mytoc()

#' Check whether parameters have reasonable values
draws_gpcovfg <- as_draws_rvars(fit_gpcovfg$draws())
summarise_draws(subset(draws_gpcovfg,
                       variable = c("sigma_", "lengthscale_"), 
                       regex = TRUE))

#' We also see that MCMC diagnostics rhat and ESSs are fine,

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(draws_gpcovfg$f), 
         sigma = mean(draws_gpcovfg$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' The MCMC integration works well and the model fit looks good.
#' 
#' Plot posterior draws and posterior mean of the mean function
draws_gpcovfg |>
  spread_rvars(f[i]) |>
  unnest_rvars() |>
  mutate(time = mcycle$times[i]) |>
  ggplot(aes(x = time, y = f, group = .draw)) +
  geom_line(color = set1[2], alpha = 0.1) +
  geom_point(data = mcycle, mapping = aes(x = times, y = accel), inherit.aes = FALSE) +
  geom_line(data = mcycle, mapping = aes(x = times, y = mean(draws_gpcovfg$f)), 
            inherit.aes = FALSE, color = set1[1], size = 1) +
  labs(x = "Time (ms)", y = "Acceleration (g)")

#' We can also plot the posterior draws of the latent functions, which
#' is a good reminder that individual draws are more wiggly than the
#' average of the draws, and thus show better also the uncertainty, 
#' for example, in the edge of the data.
#' 
#' Compare the posterior draws to the optimized parameters
odraws_gpcovfg <- as_draws_df(opt_gpcovfg$draws())
draws_gpcovfg |>
  as_draws_df() |>
  ggplot(aes(x = lengthscale_f, y = sigma_f)) +
  geom_point(color = set1[2]) +
  geom_point(data = as_draws_df(pdraws_gpcovfg), color = set1[3]) +
  lims(x = c(0.1, 0.7), y = c(0.5, 2.2)) +
  geom_point(data = odraws_gpcovfg, color = set1[1], size = 4) +
  annotate("text",
           label = "Pathfinder draws",
           x = median(pdraws_gpcovfg$lengthscale_f), 
           y = max(pdraws_gpcovfg$sigma_f) + 0.1, 
           hjust = 0.5,
           color = set1[3],
           size = 6) +
  annotate("text",
           label = "Posterior draws",
           x = median(draws_gpcovfg$lengthscale_f), 
           y = max(draws_gpcovfg$sigma_f) + 0.1, 
           hjust = 0.5,
           color = set1[2],
           size = 6) +
  annotate("text",
           label = "Optimized",
           x = odraws_gpcovfg$lengthscale_f + 0.01, 
           y = odraws_gpcovfg$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 6)

#' Optimization result is far from being representative of the
#' posterior. The Pathfinder draws are also clearly not overlapping
#' MCMC draws, but in GPs the ratio of sigma_f and lengthscale_f is
#' the important one, and that is close.
#' 

#' # Heteroskedastic GP with Hilbert basis functions
#'
#' The covariance matrix approach requires computation of Cholesky of
#' the covariance matrix which has time cost $O(n^3)$ and this needs
#' to be done every time the parameters change, which in case of MCMC
#' can be quite many times and thus the computation time can be
#' significant when n grows. One way to speed up the computation in
#' low dimensional covariate case is to use basis function
#' approximation which changes the GP to a linear model. Here we use
#' Hilbert space basis functions. With increasing number of basis
#' functions and factor c, the approximation error can be made
#' arbitrarily small. Sufficient accuracy and significant saving in
#' the computation speed is often achieved with a relatively small
#' number of basis functions.
#' 
#' ## Illustrate the basis functions
#' 
#' Code
filebf0 <- "gpbf0.stan"
#| results: asis
print_stan_file(filebf0)
#' The model code includes Hilbert space basis function helpers
#| results: asis
print_stan_file("gpbasisfun_functions.stan")

#' Compile basis function generation code
#| label: modelbf0
#| results: hide
modelbf0 <- cmdstan_model(stan_file = filebf0,
                          include_paths = ".")

#' Data to be passed to Stan
standatabf0 <- list(x = seq(0, 1, length.out = 100), 
                    N = 100, 
                    c_f = 3, # factor c of basis functions for GP for f1
                    M_f = 160,  # number of basis functions for GP for f1
                    sigma_f = 1, 
                    lengthscale_f = 1) 
#' Generate basis functions
#| results: hide
fixbf0 <- modelbf0$sample(data = standatabf0,
                          fixed_param = TRUE, 
                          chains = 1,
                          iter = 1,
                          iter_sampling = 1)
#' There is certainly easier way to do this, but this is what I came up quickly
q<-subset(fixbf0$draws(), variable = "PHI_f") |>
  as_draws_matrix() |>
  as.numeric()|>
  matrix(nrow = 100, ncol = 160)|>
  as.data.frame()
id <- rownames(q)
q <- cbind(x = as.numeric(id), q)
q <- q |>
  pivot_longer(!x, 
               names_to = "ind", 
               names_transform = list(ind = readr::parse_number), 
               values_to = "f")|>
  mutate(x = x/100)

#' Plot the first 6 basis functions. These are just sine and cosine
#' functions with different frequencies and truncated to a pre-defined
#' box.
q |>
  dplyr::filter(ind <= 6) |>
  ggplot(aes(x = x, y = f, group = ind, color = factor(ind))) +
  geom_line() +
  geom_text_repel(data = dplyr::filter(q, ind <= 6 & x == 0.01),
                  aes(x = -0.01, y = f, label = ind), 
                  direction = "y") +
  geom_text_repel(data = dplyr::filter(q, ind <= 6 & x == 1),
                  aes(x = 1.02, y = f, label = ind), 
                  direction = "y") +
  theme(legend.position = "none")

#' The first 8 spectral densities for exponentiated quadratic
#' covariance function with sigma_f = 1 and lengthscale_f = 1. These
#' spectral densities give a prior weight for each basis
#' function. Bigger weights on the smoother basis functions thus imply
#' a prior on function space favoring smoother functions.
spd_EQ <- as.matrix(fixbf0$draws(variable = "diagSPD_EQ_f"))
round(spd_EQ[1:12], 2)

#' The first 8 spectral densities for Matérn-3/2 covariance function
#' with sigma_f = 1 and lengthscale_f = 1. The spectral density values go
#' down much slower than for the exponentiated quadratic covariance
#' function, which is natural as Matérn-3/2 is less smooth.
spd_Matern32 <- as.matrix(fixbf0$draws(variable = "diagSPD_Matern32_f"))
round(spd_Matern32[1:12], 2)

#' Plot 4 random draws from the prior on function space with
#' exponentiated quadratic covariance function and sigma_f = 1 and
#' lengthscale_f = 1. The basis function approximation is just a linear
#' model, with the basis functions weighted by the spectral densities
#' depending on the sigma_f and lengthscale_f, and the prior for the
#' linear model coefficients is simply independent normal(0, 1).
set.seed(365)
qr <- bind_rows(lapply(1:4, function(i) {
  q |>
    mutate(r = rep(rnorm(160), times = 100), fr = f*r*spd_EQ[ind]) |>
    group_by(x) |>
    summarise(f = sum(fr)) |>
    mutate(ind = i) }))
qr |>
  ggplot(aes(x = x, y = f, group = ind, color = factor(ind))) +
  geom_line() +
  geom_text_repel(data = dplyr::filter(qr, x == 0.01),
                  aes(x = -0.01, y = f, label = ind), 
                  direction = "y") +
  geom_text_repel(data = dplyr::filter(qr, x == 1),
                  aes(x = 1.02, y = f, label = ind), 
                  direction = "y") +
  theme(legend.position = "none")

#' Plot 4 random draws from the prior on function space with
#' Matérn-3/2 covariance function and sigma_f = 1 and
#' lengthscale_f = 1. The same random number generator seed was used so
#' that you can compare this plot to the above one. Matérn-3/2 had
#' more prior mass on higher frequencies and the prior draws are more
#' wiggly.
set.seed(365)
qr <- bind_rows(lapply(1:4, function(i) {
  q |>
    mutate(r = rep(rnorm(160), times = 100), fr = f*r*spd_Matern32[ind]) |>
    group_by(x) |>
    summarise(f = sum(fr)) |>
    mutate(ind = i) }))
qr |>
  ggplot(aes(x = x, y = f, group = ind, color = factor(ind))) +
  geom_line() +
  geom_text_repel(data = dplyr::filter(qr, x == 0.01), aes(x = -0.01, y = f, label = ind), 
                  direction = "y") +
  geom_text_repel(data = dplyr::filter(qr, x == 1), aes(x = 1.02, y = f, label = ind), 
                  direction = "y") +
  theme(legend.position = "none")

#' Let's do the same with lengthscale_f = 0.3
standatabf0 <- list(x = seq(0, 1, length.out = 100), 
                    N = 100, 
                    c_f = 1.5, # factor c of basis functions for GP for f1
                    M_f = 160,  # number of basis functions for GP for f1
                    sigma_f = 1, 
                    lengthscale_f = 0.3) 
fixbf0 <- modelbf0$sample(data = standatabf0,
                          fixed_param = TRUE, 
                          chains = 1,
                          iter = 1,
                          iter_sampling = 1)
#' The basis functions are exactly the same, and only the spectral
#' densities have changed. Now the weight doesn't drop as fast for
#' the more wiggly basis functions.
spd_EQ <- as.matrix(fixbf0$draws(variable = "diagSPD_EQ_f"))
round(spd_EQ[1:15], 2)
spd_Matern32 <- as.matrix(fixbf0$draws(variable = "diagSPD_Matern32_f"))
round(spd_Matern32[1:15], 2)

#' Plot 4 random draws from the prior on function space with
#' exponentiated quadratic covariance function and sigma_f = 1 and
#' lengthscale_f = 0.3. The random functions from the prior are now more
#' wiggly. The same random number generator seed was used so that you
#' can compare this plot to the above one. Above the prior draw number
#' 2 looks like a decreasing slope. Here the prior draw number 2 still
#' has downward trend, but is more wiggly. The same random draw from
#' the coefficient space produces a wigglier function as the spectral
#' densities go down slower for the more wiggly basis functions.
set.seed(365)
qr <- bind_rows(lapply(1:4, function(i) {
  q |>
    mutate(r = rep(rnorm(160), times = 100), fr = f*r*spd_EQ[ind]) |>
    group_by(x) |>
    summarise(f = sum(fr)) |>
    mutate(ind = i) }))
qr |>
  ggplot(aes(x = x, y = f, group = ind, color = factor(ind))) +
  geom_line() +
  geom_text_repel(data = dplyr::filter(qr, x == 0.01), aes(x = -0.01, y = f, label = ind), 
                  direction = "y") +
  geom_text_repel(data = dplyr::filter(qr, x == 1), aes(x = 1.02, y = f, label = ind), 
                  direction = "y") +
  theme(legend.position = "none")

#' Plot 4 random draws from the prior on function space with
#' Matérn-3/2 covariance function and sigma_f = 1 and
#' lengthscale_f = 0.3. The prior draws are more wiggly than with
#' exponentiated quadratic.
set.seed(365)
qr <- bind_rows(lapply(1:4, function(i) {
  q |>
    mutate(r = rep(rnorm(160), times = 100), fr = f*r*spd_Matern32[ind]) |>
    group_by(x) |>
    summarise(f = sum(fr)) |>
    mutate(ind = i) }))
qr |>
  ggplot(aes(x = x, y = f, group = ind, color = factor(ind))) +
  geom_line() +
  geom_text_repel(data = dplyr::filter(qr, x == 0.01), aes(x = -0.01, y = f, label = ind), 
                  direction = "y") +
  geom_text_repel(data = dplyr::filter(qr, x == 1), aes(x = 1.02, y = f, label = ind), 
                  direction = "y") +
  theme(legend.position = "none")

#' ## GP with basis functions for f
#' 
#' Model code
file_gpbff <- "gpbff.stan"
#| results: asis
print_stan_file(file_gpbff)

#' Compile Stan model
#| label: model_gpbff
#| results: hide
model_gpbff <- cmdstan_model(stan_file = file_gpbff,
                             include_paths = ".",
                             stanc_options = list("O1"))

#' Data to be passed to Stan
standata_gpbff <- list(x = mcycle$times, 
                        y = mcycle$accel, 
                        N = length(mcycle$times), 
                        c_f = 1.5, # factor c of basis functions for GP for f1
                        M_f = 40,  # number of basis functions for GP for f1
                        c_g = 1.5, # factor c of basis functions for GP for g3
                        M_g = 40)  # number of basis functions for GP for g3

#' ## Optimize and find MAP estimate
tic("Find MAP estimate for homoskedastic GP with basis functions")
#| label: opt_gpbff
#| results: hide
opt_gpbff <- model_gpbff$optimize(data = standata_gpbff, 
                                  init = 0.1,
                                  algorithm = "bfgs")
#+
mytoc()

#' Check whether parameters have reasonable values
odraws_gpbff <- as_draws_rvars(opt_gpbff$draws())
subset(odraws_gpbff, variable = c("intercept", "sigma_", "lengthscale_"), 
       regex = TRUE)

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(odraws_gpbff$f), 
         sigma = mean(odraws_gpbff$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")


#' The optimization is not that bad. We are now optimizing the joint
#' posterior of 1 covariance function parameters and 40 basis
#' function coefficients.
#'
#' As MCMC is very fast for this model, we skip showing Laplace and
#' Pathfinder approximations.
#' 
#' ## Sample using dynamic HMC
#| label: fit_gpbff
#| results: hide
#| cache: true
fit_gpbff <- model_gpbff$sample(data = standata_gpbff, 
                                iter_warmup = 500,
                                iter_sampling = 500,
                                refresh = 100, 
                                adapt_delta = 0.9)

#' Check whether parameters have reasonable values
draws_gpbff <- as_draws_rvars(fit_gpbff$draws())
summarise_draws(subset(draws_gpbff, 
                       variable = c("intercept", "sigma_", "lengthscale_"), 
                       regex = TRUE))

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(draws_gpbff$f), 
         sigma = mean(draws_gpbff$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' The MCMC integration works well and the model fit looks good. The model fit
#' is clearly more smooth than with optimization.
#' 
#' Plot posterior draws and posterior mean of the mean function
draws_gpbff |>
  thin_draws(thin = 5) |>
  spread_rvars(f[i]) |>
  unnest_rvars() |>
  mutate(time = mcycle$times[i]) |>
  ggplot(aes(x = time, y = f, group = .draw)) +
  geom_line(color = set1[2], alpha = 0.1) +
  geom_point(data = mcycle, mapping = aes(x = times, y = accel), inherit.aes = FALSE) +
  geom_line(data = mcycle, mapping = aes(x = times, y = mean(draws_gpbff$f)), 
            inherit.aes = FALSE, color = set1[1], size = 1) +
  labs(x = "Time (ms)", y = "Acceleration (g)")

#' We can also plot the posterior draws of the latent functions, which
#' is a good reminder that individual draws are more wiggly than the
#' average of the draws, and thus show better also the uncertainty, 
#' for example, in the edge of the data.
#' 
#' Compare the posterior draws to the optimized parameters
odraws_gpbff <- as_draws_df(opt_gpbff$draws())
draws_gpbff |>
  thin_draws(thin = 5) |>
  as_draws_df() |>
  ggplot(aes(x = lengthscale_f, y = sigma_f)) +
  geom_point(color = set1[2]) +
  geom_point(data = odraws_gpbff, color = set1[1], size = 4) +
  annotate("text",
           label = "Posterior draws",
           x = median(draws_gpbff$lengthscale_f), 
           y = max(draws_gpbff$sigma_f) + 0.1, 
           hjust = 0.5,
           color = set1[2],
           size = 6) +
  annotate("text",
           label = "Optimized",
           x = odraws_gpbff$lengthscale_f + 0.01, 
           y = odraws_gpbff$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 6)

#' Optimization result is far from being representative of the
#' posterior.
#' 

#' ## GP with basis functions for f and g
#' 
#' Model code
file_gpbffg <- "gpbffg.stan"
#| results: asis
print_stan_file(file_gpbffg)

#' Compile Stan model
#| label: model_gpbffg
#| results: hide
model_gpbffg <- cmdstan_model(stan_file = file_gpbffg,
                              include_paths = ".",
                              stanc_options = list("O1"))

#' Data to be passed to Stan
standata_gpbffg <- list(x = mcycle$times, 
                        y = mcycle$accel, 
                        N = length(mcycle$times), 
                        c_f = 1.5, # factor c of basis functions for GP for f1
                        M_f = 40,  # number of basis functions for GP for f1
                        c_g = 1.5, # factor c of basis functions for GP for g3
                        M_g = 40)  # number of basis functions for GP for g3

#' ## Optimize and find MAP estimate
tic("Find MAP estimate for heteroskedastic GP with basis functions")
#| label: opt_gpbffg
#| results: hide
opt_gpbffg <- model_gpbffg$optimize(data = standata_gpbffg, 
                                    init = 0.1,
                                    algorithm = "bfgs")
#+
mytoc()

#' Check whether parameters have reasonable values
odraws_gpbffg <- as_draws_rvars(opt_gpbffg$draws())
subset(odraws_gpbffg, variable = c("intercept", "sigma_", "lengthscale_"), 
       regex = TRUE)

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(odraws_gpbffg$f), 
         sigma = mean(odraws_gpbffg$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' The optimization overfits, as we are now optimizing the joint
#' posterior of 2 covariance function parameters and 2 x 40 basis
#' function coefficients.
#'
#' ## Sample from the Pathfinder approximation
tic("Sample from the Pathfinder approximation for heteroskedastic GP with basis functions")
#| label: pth_gpbffg
#| cache: true
pth_gpbffg <- model_gpbffg$pathfinder(data = standata_gpbffg, 
                                      init = 0.01, 
                                      num_paths = 20,
                                      single_path_draws = 200, 
                                      history_size = 50,
                                      max_lbfgs_iters = 100,
                                      refresh = 0)
#+
mytoc()
                                  
#' Check whether parameters have reasonable values
pdraws_gpbffg <- as_draws_rvars(pth_gpbffg$draws())
summarise_draws(subset(pdraws_gpbffg, 
                       variable = c("sigma_", "lengthscale_"), 
                       regex = TRUE), 
                default_summary_measures())

#' As diagnostic Pareto-k value was very high, it is likely there are
#' only a few unique draws.
length(unique(extract_variable(pdraws_gpbffg, "lp__")))

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(pdraws_gpbffg$f), 
         sigma = mean(pdraws_gpbffg$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' Pathfinder result looks quite good and just slightly overfitted.
#' 
#' ## Sample using dynamic HMC
#| label: fit_gpbffg
#| results: hide
#| cache: true
fit_gpbffg <- model_gpbffg$sample(data = standata_gpbffg, 
                                  iter_warmup = 500,
                                  iter_sampling = 500,
                                  refresh = 100, 
                                  adapt_delta = 0.9,
                                  init = pdraws_gpbffg)

#' Check whether parameters have reasonable values
draws_gpbffg <- as_draws_rvars(fit_gpbffg$draws())
summarise_draws(subset(draws_gpbffg, 
                       variable = c("intercept", "sigma_", "lengthscale_"), 
                       regex = TRUE))

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(draws_gpbffg$f), 
         sigma = mean(draws_gpbffg$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' The MCMC integration works well and the model fit looks
#' good. Sampling is much faster than when using the covariance matrix
#' based computation.
#' 
#' Plot posterior draws and posterior mean of the mean function
draws_gpbffg |>
  thin_draws(thin = 5) |>
  spread_rvars(f[i]) |>
  unnest_rvars() |>
  mutate(time = mcycle$times[i]) |>
  ggplot(aes(x = time, y = f, group = .draw)) +
  geom_line(color = set1[2], alpha = 0.1) +
  geom_point(data = mcycle, mapping = aes(x = times, y = accel), inherit.aes = FALSE) +
  geom_line(data = mcycle, mapping = aes(x = times, y = mean(draws_gpbffg$f)), 
            inherit.aes = FALSE, color = set1[1], size = 1) +
  labs(x = "Time (ms)", y = "Acceleration (g)")

#' We can also plot the posterior draws of the latent functions, which
#' is a good reminder that individual draws are more wiggly than the
#' average of the draws, and thus show better also the uncertainty, 
#' for example, in the edge of the data.
#' 
#' Compare the posterior draws to the optimized parameters
odraws_gpbffg <- as_draws_df(opt_gpbffg$draws())
draws_gpbffg |>
  thin_draws(thin = 4) |>
  as_draws_df() |>
  ggplot(aes(x = lengthscale_f, y = sigma_f)) +
  geom_point(color = set1[2]) +
  geom_point(data = as_draws_df(pdraws_gpbffg), color = set1[3]) +
  geom_point(data = odraws_gpbffg, color = set1[1], size = 4) +
  annotate("text",
           label = "Pathfinder draws",
           x = median(pdraws_gpbffg$lengthscale_f), 
           y = max(pdraws_gpbffg$sigma_f) + 0.01, 
           hjust = 0.5,
           color = set1[3],
           size = 6) +
  annotate("text",
           label = "Posterior draws",
           x = median(draws_gpbffg$lengthscale_f), 
           y = max(thin_draws(draws_gpbffg, 4)$sigma_f) + 0.01, 
           hjust = 0.5,
           color = set1[2],
           size = 6) +
  annotate("text",
           label = "Optimized",
           x = odraws_gpbffg$lengthscale_f + 0.01, 
           y = odraws_gpbffg$sigma_f, 
           hjust = 0,
           color = set1[1],
           size = 6)

#' Optimization and Pathfinder results are far from being representative of the
#' posterior.
#' 

#' ## Model comparison
#' 
#' Looking at the plots comparing model predictions and data, it is
#' quite obvious in this case that the heteroskedastic model is better
#' for these data. In cases when it is not as clear, we can use
#' leave-one-out cross-validation comparison. Here we compare
#' homoskedastic and heteroskedastic models.

loobff <- fit_gpbff$loo()
loobffg <- fit_gpbffg$loo()
loo_compare(list(homoskedastic = loobff, heteroskedastic = loobffg))

#' Heteroskedastic model has clearly much higher elpd estimate.
#'
#' We can plot also the difference in the pointwise elpd values (log scores)
#' so that we see in which parts the heteroskedastic model is better

data.frame(time = mcycle$times, 
           elpd_diff = loobffg$pointwise[, 'elpd_loo']-loobff$pointwise[, 'elpd_loo']) |>
  ggplot(aes(x = time, y = elpd_diff)) +
  geom_point(color = set1[2]) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Time (ms)", y = TeX("elpd of heteroskedastic GP is higher $\\rightarrow$"))

#' ## ADVI
#'
#' Stan has also automatic differentiation variational inference
#' (ADVI) algorithm, which is slightly outdated. There exists faster
#' and more robust algorithms, but they are not available in
#' Stan. Running ADVI long enough gives anyway results that can give
#' some idea on the accuracy of stochastic optimization variational
#' inference. Pathfinder uses deterministic optimization and chooses
#' among the normal approximations along the optimization path, while
#' ADVI uses stochastic optimization to optimize the normal
#' approximation parameters. The default is to use normal
#' approximation with diagonal covariance matrix (meanfield approach),
#' which in case of hierarchical models may lead to overestimation of
#' the variance [@Yao-Vehtari-Simpson-Gelman:2018].
#'
#' When used carefully for selected models, parameters,
#' parameterization, and approximate distribution, variational
#' inference can be useful and fast. The following example
#' illustrates, why it can also fail when applied in black box style.
#' 
#' Run auto-differentiated variational inference (ADVI) with meanfield
#' normal approximation, and in the end, sample from the
#' approximation.
#| label: vi_gpbffg
#| results: hide
vi_gpbffg <- model_gpbffg$variational(data = standata_gpbffg, 
                                      init = 0.01,
                                      tol_rel_obj = 1e-4,
                                      iter = 1e5,
                                      refresh = 1000,
                                      seed = 2678)

#' Check whether parameters have reasonable values
vidraws_gpbffg <- as_draws_rvars(vi_gpbffg$draws())
summarise_draws(subset(vidraws_gpbffg, 
                       variable = c("intercept", "sigma_", "lengthscale_"), 
                       regex = TRUE))

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(vidraws_gpbffg$f), 
         sigma = mean(vidraws_gpbffg$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' ADVI inference is catching the mean function well, and some of the
#' varying noise variance, but clearly overestimating the noise
#' variance in the early part.
#' 
#' Plot posterior draws and posterior mean of the mean function
vidraws_gpbffg |>
  thin_draws(thin = 5) |>
  spread_rvars(f[i]) |>
  unnest_rvars() |>
  mutate(time = mcycle$times[i]) |>
  ggplot(aes(x = time, y = f, group = .draw)) +
  geom_line(color = set1[2], alpha = 0.1) +
  geom_point(data = mcycle, mapping = aes(x = times, y = accel), inherit.aes = FALSE) +
  geom_line(data = mcycle, mapping = aes(x = times, y = mean(vidraws_gpbffg$f)), 
            inherit.aes = FALSE, color = set1[1], size = 1) +
  labs(x = "Time (ms)", y = "Acceleration (g)")

#' We can also plot the posterior draws of the latent functions, which
#' is a good reminder that individual draws are more wiggly than the
#' average of the draws, and thus show better also the uncertainty, 
#' for example, in the edge of the data.
odraws_gpbffg <- as_draws_df(opt_gpbffg$draws())
draws_gpbffg |>
  thin_draws(thin = 5) |>
  as_draws_df() |>
  ggplot(aes(x = `f[1]`, y = log(`sigma[1]`))) +
  geom_point(color = set1[2]) +
  geom_point(data = as_draws_df(pdraws_gpbffg), color = set1[3]) +
  geom_point(data = as_draws_df(vidraws_gpbffg), color = set1[4]) +
  geom_point(data = odraws_gpbffg, color = set1[1], size = 4) +
  annotate("text",
           label = "ADVI draws",
           x = median(vidraws_gpbffg$f[1]) + 1.3, 
           y = max(log(vidraws_gpbffg$sigma[1])) + 0.2, 
           hjust = 0,
           color = set1[4],
           size = 6) +
  annotate("text",
           label = "Pathfinder draws",
           x = max(pdraws_gpbffg$f[1]) + 1.5, 
           y = median(log(pdraws_gpbffg$sigma[1])) + 0.1, 
           hjust = 0,
           color = set1[3],
           size = 6) +
  annotate("text",
           label = "MCMC draws",
           x = median(draws_gpbffg$f[1]) + 1, 
           y = min(log(draws_gpbffg$sigma[1])) - 0.1, 
           hjust = 0,
           color = set1[2],
           size = 6) +
  annotate("text",
           label = "Optimized",
           x = odraws_gpbffg$`f[1]` + 1, 
           y = log(odraws_gpbffg$`sigma[1]`), 
           hjust = 0,
           color = set1[1],
           size = 6) +
  labs(y = "g[1]")

#' Compare the draws from the variational approximation to the MCMC
#' draws and optimized parameters. This time show f[1] and g[1] to
#' illustrate the challenging funnel shape. Although the inference
#' happens in the space of beta_f and beta_g, f[1] and g[1] are linear
#' projection of beta_f and beta_g, and thus the funnel is causing the
#' problems for ADVI. Full rank normal approximation would not be able
#' to help here. Pathfinder works better than ADVI.

#' # Heteroskedastic GP with Matérn covariance function and Hilbert basis functions 
#' 
#' Exponentiated quadratic is sometimes considered to be too smooth as
#' all the derivatives are continuous. For comparison we use Matérn-3/2
#' covariance. The Hilbert space basis functions are the same and only
#' the spectral density values change (that is different basis
#' functions have a different weighting).
#' 
#' ## Model code
file_gpbffg2 <- "gpbffg_matern.stan"
#| results: asis
print_stan_file(file_gpbffg2)

#' Compile Stan model
#| label: model_gpbffg2
#| results: hide
model_gpbffg2 <- cmdstan_model(stan_file = file_gpbffg2,
                               include_paths = ".")

#' Data to be passed to Stan
standata_gpbffg2 <- list(x = mcycle$times, 
                        y = mcycle$accel, 
                        N = length(mcycle$times), 
                        c_f = 1.5, # factor c of basis functions for GP for f1
                        M_f = 160,  # number of basis functions for GP for f1
                        c_g = 1.5, # factor c of basis functions for GP for g3
                        M_g = 160)  # number of basis functions for GP for g3

#' ## Sample using dynamic HMC
#| label: fit_gpbffg2
#| results: hide
#| cache: true
fit_gpbffg2 <- model_gpbffg2$sample(data = standata_gpbffg2, 
                                    iter_warmup = 500,
                                    iter_sampling = 500, 
                                    adapt_delta = 0.9)

#' Check whether parameters have reasonable values
draws_gpbffg2 <- as_draws_rvars(fit_gpbffg2$draws())
summarise_draws(subset(draws_gpbffg2, 
                       variable = c("intercept", "sigma_", "lengthscale_"), 
                       regex = TRUE))

#' Compare the model to the data
mcycle |>
  mutate(Ef = mean(draws_gpbffg2$f), 
         sigma = mean(draws_gpbffg2$sigma)) |>
  (\(d) plot_mcycle %+% d)() +
  geom_line(aes(y = Ef), color = set1[1]) +
  geom_line(aes(y = Ef - 2*sigma), color = set1[1], linetype = "dashed") +
  geom_line(aes(y = Ef + 2*sigma), color = set1[1], linetype = "dashed")

#' The MCMC integration works well and the model fit looks good.
#' 
#' Plot posterior draws and posterior mean of the mean function
draws_gpbffg2 |>
  thin_draws(thin = 5) |>
  spread_rvars(f[i]) |>
  unnest_rvars() |>
  mutate(time = mcycle$times[i]) |>
  ggplot(aes(x = time, y = f, group = .draw)) +
  geom_line(color = set1[2], alpha = 0.1) +
  geom_point(data = mcycle, mapping = aes(x = times, y = accel), inherit.aes = FALSE) +
  geom_line(data = mcycle, mapping = aes(x = times, y = mean(draws_gpbffg2$f)), 
            inherit.aes = FALSE, color = set1[1], size = 1) +
  labs(x = "Time (ms)", y = "Acceleration (g)")

#' We see that when using Matérn-3/2 covariance instead of the
#' exponentiated quadratic, the model fit is more wiggly.
#' 
#' # References {.unnumbered}
#'
#' <div id="refs"></div>
#'
#' # Licenses {.unnumbered}
#' 
#' * Code &copy; 2021--2026, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2021--2026, Aki Vehtari, licensed under CC-BY-NC 4.0.
#' 
