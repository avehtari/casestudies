#' ---
#' title: "Laplace method and Jacobian of parameter transformation"
#' author: "Aki Vehtari"
#' date: 2021-01-21
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
#' Stan has Laplace approximation algorithm (see [Stan
#' Reference
#' Manual](https://mc-stan.org/docs/reference-manual/laplace-approximation.html)).
#' Specificlly Stan makes the normal approximation in the
#' unconstrained space, samples from the approximation, transforms the
#' sample to the constrained space, and returns the sample. The method
#' has option `jacobian` that can be used to select whether the
#' Jacobian adjustment is included or not.
#'
#' This case study provides visual illustration of Jacobian adjustment
#' for a parameter transformation, why it is needed for the Laplace
#' approximation, and effect of `jacobian` option in Stan `log_prob`
#' and `log_prob_grad` functions. This notebook intentionally doesn't
#' go in the mathematical details of measure and probability theory.
#'
#+ setup, include = FALSE
knitr::opts_chunk$set(message = FALSE, error = FALSE, warning = FALSE, comment = NA, cache = FALSE)

#' **Load packages**
#| cache: FALSE
library("rprojroot")
root<-has_file(".casestudies-root")$make_fix_file()
library(tidyr) 
library(dplyr) 
library(cmdstanr) 
options(mc.cores = 1)
library(posterior)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size = 16))
theme_remove_yaxis <- theme(axis.text.y = element_blank(),
                            axis.line.y = element_blank(),
                            axis.ticks.y = element_blank())
set1 <- RColorBrewer::brewer.pal(7, "Set1")
library(latex2exp)
library(patchwork)
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

#' # Example model and posterior
#' 
#' For the illustration Binomial model is used with observed data
#' $N = 10, y = 9$.
data_bin <- list(N = 10, y = 9)

#' As Beta(1,1) prior is used the posterior is Beta(9+1,1+1), but for
#' illustration we also use Stan to find the mode of the posterior,
#' sample from the posterior, and compare different posterior density
#' values that we can ask Stan to compute.
code_binom <- root("Jacobian", "binom.stan")
#| output: asis
print_stan_file(code_binom)

#' We use `compile_model_methods = TRUE` to be able to access
#' `log_prob()` method later.
model_bin <- cmdstan_model(stan_file = code_binom,
                           compile_model_methods = TRUE,
                           force_recompile = TRUE)

#' Default MCMC sampling (as this is an easy posterior we skip showing
#' the results for the convergence diagnostics).
#| results: hide
fit_bin <- model_bin$sample(data = data_bin,
                            seed = SEED,
                            refresh = 0)

#' Default optimization finds the maximum of the posterior.
opt_bin <- model_bin$optimize(data = data_bin)

#' The following plot shows the exact posterior (black) and grey
#' vertical line shows the MAP, that is, posterior mode in the
#' constrained space. Stan optimizing finds correctly the mode.
df_bin <- data.frame(theta = plogis(seq(-6, 7, length.out = 200))) |>
  mutate(pdfbeta = dbeta(theta, 9+1, 1+1))
plot_theta <- ggplot(data = df_bin, aes(x = theta, y = pdfbeta)) +
  coord_cartesian(expand = c(bottom = FALSE)) +
  geom_line() +
  geom_vline(xintercept = (opt_bin$draws()[1, "theta"]), color = "gray", linetype="dashed") +
  labs(x = TeX(r"($\theta$)"), y = "") +
  theme_remove_yaxis +
  annotate("text", x = 0.81, y = 3.3, label = TeX(r"($p(\theta|y)$)"),
           color = 1, size = 5, hjust = 1) +
  annotate("text", x = 0.89, y = 4.3, label = "mode",
           color = "gray", size = 5, hjust = 1)
plot_theta 

#' # Posterior log_prob
#' 
#' The CmdStanR fit object `fit_bin` provides also access to log_prob
#' and log_prob_grad functions. The documentation of log_prob says
#'
#'     Using model's log_prob and grad_log_prob take values from the
#'     unconstrained space of model parameters and (by default) return
#'     values in the same space.
#'
#' And one of the options say
#' 
#'     jacobian_adjustment: (logical) Whether to include the log-density
#'     adjustments from un/constraining variables.
#'
#' The functions accepts also `jacobian`. We can compute the exact
#' posterior density values in grid using Stan and log_prob with
#' `jacobian = FALSE`. We create a helper function.
fit_pdf <- function(th, fit) {
  exp(fit$log_prob(fit$unconstrain_variables(list(theta = th)), 
                   jacobian = FALSE))
}

df_bin <- df_bin |>
  mutate(pdfstan = sapply(theta, fit_pdf, fit_bin))
plot_theta +
  geom_line(data = df_bin, aes(y = pdfstan), color = set1[1]) +
  annotate("text", x = 0.85, y = .2, label = TeX(r"($q(\theta|y)$ = exp(lp__))"),
           color = set1[1], size = 5, hjust = 1)

#' 
#' The pdf from Stan is much lower than the true posterior, because it
#' is unnormalized posterior as in general computing the normalization
#' term is non-trivial. In this case the true posterior has analytic
#' solution for the normalizing constant
#' $$
#' \frac{\Gamma(N+2)}{\Gamma(y+1)\Gamma(N-y+1)} = 110
#' $$
#' and we get exact match by multiplying the density returned by
#' Stan by 110.
df_bin <- df_bin |>
  mutate(pdfstan = sapply(theta, fit_pdf, fit_bin)*110)
plot_theta +
  geom_line(data = df_bin, aes(y = pdfstan), color = set1[1]) +
  annotate("text", x = 0.79, y = 2.9, label = TeX(r"($q(\theta|y)\cdot 110$)"),
           color = set1[1], size = 5, hjust = 1)

#' Thus if someone cares about the posterior mode in the constrained
#' space they need `jacobian = FALSE`.
#' 
#' Side note: the normalizing constant is not needed for MCMC and not
#' needed when estimating various expectations using MCMC draws, but
#' is used here for illustration.
#'
#' # Constraint and parameter transformation
#' 
#' In this example, theta is constrained to be between 0 and 1
#' ```
#'  real<lower = 0, upper = 1> theta; // probability of success in range (0, 1)
#' ```
#' To avoid problems with constraints in optimization and MCMC, Stan
#' switches under the hood to unconstrained parameterization using
#' logit transformation
#' $$
#' \operatorname{logit}(\theta) = \log\left(\frac{\theta}{1-\theta}\right)
#' $$.
#'
#' In the above `fit_pdf` function we used Stan's `unconstrain_pars`
#' function to transform theta to logit(theta). In R we can also
#' define logit function as the inverse of the logistic function.
logit <- qlogis

#' We now switch looking at the distributions in the unconstrained
#' space. First we look at what happens if we replace $\theta$ with
#' $\operatorname{logit}(\theta)$, but do not take into acoount the
#' Jacobian.
df_bin |>
  mutate(pdfstan = sapply(theta, fit_pdf, fit_bin)*110) |>
  ggplot(aes(x = logit(theta), y = pdfbeta)) +
  coord_cartesian(expand = c(bottom = FALSE)) +
  geom_line() +
  geom_line(aes(y = pdfstan), color = set1[1]) +
  geom_vline(xintercept = logit(opt_bin$draws()[1, "theta"]), color = "gray", linetype="dashed") +
  xlim(-3.1, 7) +
  labs(x = TeX(r"($\logit(\theta)$)"), y = "") +
  theme_remove_yaxis +
  annotate("text", x = logit(0.81), y = 3.3, label = TeX(r"($p(\theta|y)$)"),
           color = 1, size = 5, hjust = 1) +
  annotate("text", x = logit(0.89), y = 4.3, label = "mode",
           color = "gray", size = 5, hjust = 1) +
  annotate("text", x = logit(0.79), y = 2.9, label = TeX(r"($q(\theta|y)\cdot 110$)"),
           color = set1[1], size = 5, hjust = 1)

#' The above plot shows now the function that Stan optimizing
#' optimizes in the unconstrained space, and the MAP is the logit of
#' the MAP in the constrained space. Thus if someone cares about the
#' posterior mode in the constrained, but is doing the optimization in
#' unconstrained space they still need `jacobian = FALSE`.
#'
#' # Parameter transformation and Jacobian adjustment
#' 
#' That function shown above is not the posterior of `logit(theta)`. As
#' the transformation is non-linear we need to take into account the
#' distortion caused by the transform. The density must be multiplied by a
#' Jacobian adjustment equal to the absolute determinant of the
#' Jacobian of the transform. See more in [Stan User's
#' Guide](https://mc-stan.org/docs/2_26/stan-users-guide/changes-of-variables.html).
#' The Jacobian for lower and upper bounded scalar is given in [Stan
#' Reference
#' Manual](https://mc-stan.org/docs/2_25/reference-manual/logit-transform-jacobian-section.html), and for (0, 1)-bounded it is
#' $$
#' \theta (1-\theta).
#' $$
#'
#' Stan can do this transformation for us when we call log_prob with
#' `jacobian = TRUE`
fit_pdf_jacobian <- function(th, fit) {
  exp(fit$log_prob(fit$unconstrain_variables(list(theta = th)), 
                   jacobian = TRUE))
}

#' We compare the true adjusted posterior density in logit(theta)
#' space to non-adjusted density function. For visualization purposes
#' we scale the functions to have the same maximum, so they are not
#' normalized distributions.
df_bin <- df_bin |>
  mutate(pdfstan_nonadjusted = sapply(theta, fit_pdf, fit_bin), 
         pdfstan_jacobian = sapply(theta, fit_pdf_jacobian, fit_bin))
# store the common part
plot_logit <- ggplot(data = df_bin,
                     aes(x = logit(theta),
                         y = pdfstan_nonadjusted/max(pdfstan_nonadjusted))) +
  coord_cartesian(expand = c(bottom = FALSE)) +
  geom_line(color = set1[1]) +
  geom_line(aes(y = pdfstan_jacobian/max(pdfstan_jacobian)), color = set1[2]) +
  xlim(-3.1, 7) +
  labs(x = TeX(r"($\logit(\theta)$)"), y = "") +
  theme_remove_yaxis +
  annotate("text", x = 1.15, y = 0.95, label = TeX("jacobian = TRUE"),
           color = set1[2], size = 5, hjust = 1) +
  annotate("text", x = 1.2, y = 0.9,
           label = TeX(r"($q(\logit(\theta)|y)  = q(\theta|y)\theta(1-\theta)$)"),
           color = set1[2], size = 5, hjust = 1) +
  annotate("text", x = 2.9, y = 0.95, label = "jacobian = FALSE",
           color = set1[1], size = 5, hjust = 0) +
  annotate("text", x = 2.9, y = 0.9,
           label = TeX(r"($q(\theta|y) \neq q(\logit(\theta)|y)$)"),
           color = set1[1], size = 5, hjust = 0)
plot_logit

#' Stan MCMC samples from the blue distribution with
#' `jacobian = TRUE`. The mode of that distribution is different from
#' the mode of `jacobian = FALSE`. In general the mode is not invariant
#' to transformations.
#'
#' # Comparing quantiles
#'
#' We can further illustrate the transformation, by examining what
#' happens to quantiles.  We start with uniform distribution and look
#' at the quantiles $0.05, 0.1, \ldots, 0.95$. The probability mass
#' between consequtive quantiles is 5%. When we transform these
#' quantiles with logit transformation we get the following values:
logit(seq(0.05, .95, by=0.05)) |> round(2)

#' We see that these quantiles after logit transformation are not
#' uniformly spaced. To still have 5% probability mass between the
#' quantiles the probability density can't be uniform. The Jacobian of
#' the transformation $\theta (1-\theta)$ tells us how the density
#' needs to be adjusted. Uniform distribution for $\theta$ transforms
#' to a distribution proportional to $\theta (1-\theta)$. The
#' following figure shows the quantiles in $\theta$-space and in
#' $\operatorname{logit}(\theta)$-space.

p1 <- data.frame(theta=c(0, 1), pdfunif=c(1, 1)) |>
  ggplot(aes(x = theta, y = pdfunif)) +
  coord_cartesian(expand = c(bottom = FALSE)) +
  scale_x_continuous(breaks=seq(0, 1, by=0.1)) +
  geom_line() +
  labs(x = TeX(r"($\theta$)"), y = "") +
  theme_remove_yaxis +
  geom_segment(data = data.frame(x=seq(0, 1, by=0.05), 
                                 xend=seq(0, 1, by=0.05), 
                                 y=rep(0, 21), 
                                 yend=rep(1, 21)), 
               aes(x=x, y=y, xend=xend, yend=yend), 
               alpha=0.3)
p2 <- data.frame(theta = plogis(seq(-6, 6, length.out = 100))) |>
  mutate(pdf = theta*(1-theta)) |>
  ggplot(aes(x = qlogis(theta), y = pdf)) +
  coord_cartesian(expand = c(bottom = FALSE)) +
  scale_x_continuous(breaks=seq(-6, 6, by=2)) +
  geom_line() +
  labs(x = TeX(r"($\logit(\theta)$)"), y = "") +
  theme_remove_yaxis +
  geom_segment(data = data.frame(x=qlogis(seq(0.05, .95, by=0.05)), 
                                 xend=qlogis(seq(0.05, .95, by=0.05)), 
                                 y=rep(0, 19), 
                                 yend=seq(0.05, .95, by=0.05)*(1-seq(0.05, .95, by=0.05))), 
               aes(x=x, y=y, xend=xend, yend=yend), 
               alpha=0.3)
p1 / p2

#' The next plot shows quantiles of our posterior distribution. We see
#' that the relative heights of areas between quantile lines have
#' changed according to the Jacobian adjustment.
p1 <- ggplot(data = df_bin, aes(x = theta, y = pdfbeta)) +
  geom_line() +
  labs(x = TeX(r"($\theta$)"), y = "") +
  theme_remove_yaxis +
  annotate("text", x = 0.8, y = 3.3, label = TeX(r"($p(\theta|y)$)"),
           color = 1, size = 5, hjust = 1) +
  geom_segment(data = data.frame(x=qbeta(seq(0.1, 0.9, by=0.1), 9+1, 1+1), 
                                 xend=qbeta(seq(0.1, 0.9, by=0.1), 9+1, 1+1), 
                                 y=rep(0, 9), 
                                 yend=dbeta(qbeta(seq(0.1, 0.9, by=0.1), 9+1, 1+1), 9+1, 1+1)), 
               aes(x=x, y=y, xend=xend, yend=yend), 
               alpha=0.3)
p2 <- ggplot(data = df_bin, aes(x = logit(theta), y = pdfstan_nonadjusted/max(pdfstan_nonadjusted))) +
  geom_line(aes(y = pdfstan_jacobian/max(pdfstan_jacobian)), color = set1[2]) +
  xlim(-6, 6) +
  labs(x = TeX(r"($\logit(\theta)$)"), y = "") +
  theme_remove_yaxis +
  annotate("text", x = 1.1, y = 0.9, label = TeX(r"($q(\logit(\theta)|y)$)"), color = set1[2], size = 5, hjust = 1) +
  geom_segment(data = data.frame(x=logit(qbeta(seq(0.1, 0.9, by=0.1), 9+1, 1+1)), 
                                 xend=logit(qbeta(seq(0.1, 0.9, by=0.1), 9+1, 1+1)), 
                                 y=rep(0, 9), 
                                 yend=sapply(qbeta(seq(0.1, 0.9, by=0.1), 9+1, 1+1),
                                             fit_pdf_jacobian, fit_bin)/max(df_bin$pdfstan_jacobian)), 
               aes(x=x, y=y, xend=xend, yend=yend), 
               alpha=0.3)
p1 / p2

#' # Wrong normal approximation
#' 
#' Stan optimizing/optimize finds the mode of `jacobian = FALSE` (in
#' some intefaces `jacobian = FALSE`). rstanarm has had an option to do
#' normal approximation at the mode `jacobian = FALSE` in the
#' unconstrained space by computing the Hessian of `jacobian = FALSE`
#' and then sampling independent draws from that normal
#' distribution. We can do the same in CmdStanR with `$laplace()`
#' method and option `jacobian = FALSE`, but this is the wrong thing to
#' do.
lap_bin <- model_bin$laplace(data = data_bin, jacobian = FALSE, 
                             seed = SEED, draws = 4000, refresh = 0)
lap_draws = lap_bin$draws(format = "df")

#' We add the current normal approximation to the plot with dashed line.
lap_draws <- lap_draws |>
  mutate(logp = lp__-max(lp__), 
         logg = lp_approx__-max(lp_approx__))
plot_logit2 <- plot_logit +
  geom_line(data = lap_draws,
            aes(x = logit(theta), y = exp(logg)),
            color = set1[1],
            linetype = "dashed")
plot_logit2

#' The problem is that this normal approximation is quite different
#' from the true posterior (with `jacobian = TRUE`).
#'
#' # Normal approximation
#' 
#' Recently the normal approximation method was implemented in Stan
#' itself with the name `laplace`. This approximation uses by default
#' `jacobian = TRUE`. We can use Laplace approximation with CmdStanR
#' method `$laplace()`, which by default is using he option
#' `jacobian = TRUE`, which is the correct thing to do.
lap_bin2 <- model_bin$laplace(data = data_bin, seed = SEED, draws = 4000, refresh = 0)
lap_draws2 <- lap_bin2$draws(format = "df")

#' We add the second normal approximation to the plot dashed line.
lap_draws2 <- lap_draws2 |>
  mutate(logp = lp__-max(lp__), 
         logg = lp_approx__-max(lp_approx__))
plot_logit3 <- plot_logit2 + 
  geom_line(data = lap_draws2, aes(x = logit(theta), y = exp(logg)),
            color = set1[2], linetype = "dashed")
plot_logit3

#' # Transforming draws to the constrained space
#' 
#' The draws from the normal approximation (shown with rug lines in op
#' and bottom) can be easily transformed back to the constrained
#' space, and illustrated with kernel density estimates. Before that
#' we plot the kernel density estimates of the draws in the
#' unconstrained space to show that this etimate is reasonable.
plot_logit4 <- plot_logit3 +
  geom_rug(data = lap_draws[1:400, ], aes(x = logit(theta), y = 0),
           color = set1[1], alpha = 0.2, sides = "t") +
  geom_rug(data = lap_draws2[1:400, ], aes(x = logit(theta), y = 0),
           color = set1[2], alpha = 0.2, sides = "b") +
  geom_density(data = lap_draws, aes(x = logit(theta), after_stat(scaled)), adjust = 2,
               color = set1[1], linetype = "dotdash") +
  geom_density(data = lap_draws2, aes(x = logit(theta), after_stat(scaled)), adjust = 2,
               color = set1[2], linetype = "dotdash")
plot_logit4

#' When we plot kernel density estimates of the logit transformed
#' draws (draws shown with rug lines in top and bottom) in the
#' constrained space, it's clear which draws approximate better the
#' true posterior (black line)
ggplot(data = df_bin, aes(x = (theta), y = pdfbeta/max(pdfbeta))) +
  geom_line() +
  geom_density(data = lap_draws, aes(x = (theta), after_stat(scaled)), adjust = 1,
               color = set1[1], linetype = "dotdash") +
  geom_density(data = lap_draws2, aes(x = (theta), after_stat(scaled)), adjust = 1,
               color = set1[2], linetype = "dotdash") +
  geom_rug(data = lap_draws[1:400, ], aes(x = (theta), y = 0),
           color = set1[1], alpha = 0.2, sides = "t") +
  geom_rug(data = lap_draws2[1:400, ], aes(x = (theta), y = 0),
           color = set1[2], alpha = 0.2, sides = "b") +
  labs(x = TeX(r"($\theta$)"), y = "") +
  theme_remove_yaxis

#' # Importance resampling
#' 
#' `$laplace()` method returns also unormalized target log density
#' (`lp__`) and normal approximation log density (`lp_approx__`) for
#' the draws from the normal approximation. These can be used to
#' compute importance ratios `exp(lp__-lp_approax__)` which can be
#' used to do importance resampling. We can do the importance
#' resampling using the posterior package.
lap_draws2is <- lap_draws2 |>
  weight_draws(lap_draws2$lp__-lap_draws2$lp_approx__, log = TRUE) |>
  resample_draws()

#' The kernel density estimate using importance resampled draws is
#' even close to the true distribution.
ggplot(data = df_bin, aes(x = (theta), y = pdfbeta/max(pdfbeta))) +
  geom_line() +
  labs(x = TeX(r"($\theta$)"), y = "") +
  theme_remove_yaxis +
  geom_density(data = lap_draws2is, aes(x = (theta), after_stat(scaled)), adjust = 1,
               color = set1[2], linetype = "dotdash") +
  geom_rug(data = lap_draws2is[1:400, ], aes(x = (theta), y = 0),
           color = set1[2], alpha = 0.2, sides = "b")

#' # Discussion
#' 
#' The normal approximation and importance resampling did work quite
#' well in this simple one dimensional case, but in general the normal
#' approximation for importance sampling works well only in quite low
#' dimensional settings or when the posterior in the unconstrained
#' space is very close to normal.
#' 
#' # Licenses {.unnumbered}
#' 
#' * Code &copy; 2021--2026, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2021--2026, Aki Vehtari, licensed under CC-BY-NC 4.0.
#' 
