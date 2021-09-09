#' ---
#' title: "Bayesian workflow book - Kilpisjärvi"
#' author: "Gelman, Vehtari, Simpson, et al"
#' date: "First version 2020-12-05. Last modified `r format(Sys.Date())`."
#' output:
#'   html_document:
#'     theme: readable
#'     toc: true
#'     toc_depth: 2
#'     toc_float: true
#'     code_download: true
#' ---

#' Workflow for checking how many MCMC iterations are needed, and how
#' many digits to report in the posterior summary results.
#' 
#' We analyse the trend in summer months average temperature 1952-2013
#' at Kilpisjärvi in northwestern Finnish Lapland (about 69°03'N, 20°50'E).
#' 
#' -------------
#' 

#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA)
# switch this to TRUE to save figures in separate files
savefigs <- FALSE

#' #### Load packages
library("rprojroot")
root<-has_file(".Workflow-Examples-root")$make_fix_file()
library(tidyr) 
library(dplyr) 
library(cmdstanr)
library(posterior)
options(pillar.negative = FALSE)
library(lemon)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))
SEED <- 48927 # set random seed for reproducability

#' ## Load and plot data
#' 
#' Load Kilpisjärvi summer month average temperatures 1952-2013:
data_kilpis <- read.delim(root("Kilpisjarvi/data","kilpisjarvi-summer-temp.csv"), sep = ";")
data_lin <-list(N = nrow(data_kilpis),
             x = data_kilpis$year,
             xpred = 2016,
             y = data_kilpis[,5])
#' 
#' Plot the data
ggplot() +
  geom_point(aes(x, y), data = data.frame(data_lin), size = 1) +
  labs(y = 'Summer temp. @Kilpisjärvi', x= "Year")

#' ## Gaussian linear model
#'
#' To analyse whether the average summer month temperature is rising,
#' we use a linear model with Gaussian model for the unexplained
#' variation.
#' 
#' ### Stan model
#' 
#' The following Stan code centers the covariate to reduce posterior
#' dependency of slope and coefficient parameters. It also makes it
#' easier to define the prior on average temperature in the center of
#' the time range (instead defining prior for temperature at year 0).
code_lin <- root("Kilpisjarvi", "linear.stan")
writeLines(readLines(code_lin))
#'
#'
#' ### Prior parameter values for weakly informative priors
data_lin_priors <- c(list(
    pmualpha_c = 10,     # prior mean for average temperature
    psalpha = 10,        # weakly informative
    pmubeta = 0,         # a priori incr. and decr. as likely
    psbeta = 0.1/3,   # avg temp prob does does not incr. more than a degree per 10 years:  setting this to +/-3 sd's
    pssigma = 1),        # setting sd of total variation in summer average temperatures to 1 degree implies that +/- 3 sd's is +/-3 degrees: 
  data_lin)

#' ### Run Stan
#+ results='hide'
mod_lin <- cmdstan_model(stan_file = code_lin)
fit_lin <- mod_lin$sample(data = data_lin_priors, seed = SEED)

#' ### Convergence diagnostics
#' There were no convergence issues reported by sampling. We can also
#' explicitly call CmsStan inference diagnostics:
fit_lin$cmdstan_diagnose()

#' And we can also look at the values of Rhats end ESS's:
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2
draws <- as_draws_rvars(fit_lin$draws())
summarize_draws(draws)

#' At this point it's sufficient that diagnostics are ok
#' (cmdstan_diagnose says "no problems detected") and effective sample
#' sizes are large enough (>400) that we can assume the diagnostics to
#' be reliable.
#'
#' Normally we would also do posterior predictive checking and residual
#' plots, but now we focus on checking how many MCMC iterations are
#' needed and how many digits to report in posterior summary results.
#' 

#' ## How many digits to report
#' 
#' We want to report posterior summaries for the slope, that is the
#' increase in average summer temperature, and for the probability
#' that the slope is positive.

#' We start looking at the mean and 90% interval for the slope parameter beta
#+ render=lemon_print, digits=3
draws %>%
  subset_draws("beta") %>%
  summarize_draws(mean, ~quantile(.x, probs = c(0.05, 0.95))) 

#' These values correspond to the temperature increase per year, but
#' to improve readability we switch looking at the temperature
#' increase per 100 years. At the same time we also add an indicator
#' variable for positivity of beta.
#+ render=lemon_print, digits=3
draws <- draws %>%
  mutate_variables(beta100 = 100*beta,
                   betapos = beta>0)

#' Let's look at the mean and 90% interval for expected temperature
#' increase per 100 years.
#+ render=lemon_print, digits=3
draws %>%
  subset_draws("beta100") %>%
  summarize_draws(mean, ~quantile(.x, probs = c(0.05, 0.95)))

#' By default `summarize_draws` shows 3 significant digits. But
#' considering the width of the 90% interval, practically meaningful
#' accuracy would be here to report posterior mean as 2.0 and the
#' posterior interval as [0.7 , 3.2]. It might be even better to round
#' more, and report that the increase is estimated to be 1 to 3 degrees
#' per century (81% probability), or 0 to 4 degrees per century (99%
#' probability).
#'
#' Now that we have an estimate for the posterior uncertainty of the
#' slope parameter, we can check whether we have enough many iterations for
#' the desired reporting accuracy. We can estimate the Monte Carlo standard
#' error that takes into account the quantity of interest, and the
#' effective sample size of MCMC draws.
#'
#+ render=lemon_print, digits=3
draws %>%
  subset_draws("beta100") %>%
  summarize_draws(mcse_mean, ~mcse_quantile(.x, probs = c(0.05, 0.95)))

#' For the mean MCSE is about 0.01 and for 5% and 95% quantiles about
#' 0.03, which indicates that we have already obtained enough many
#' iterations for stable estimates with one decimal value accuracy
#' (MCSE estimates assume that the earlier convergence diagnostics did
#' not indicate convergence or mixing issues). These MCSE estimates
#' illustrate also the fact that usually tail quantiles have lower
#' accuracy than the posterior mean.
#'
#' If the MCSE would indicate more iterations would be needed for
#' desired accuracy, we would expect that running four times more
#' iterations would halve the MCSEs.
#'
#' We can also report the probability that the temperature change is
#' positive.
#+ render=lemon_print, digits=3
draws %>%
  subset_draws("betapos") %>%
  summarize_draws("mean", mcse = mcse_mean)

#' Again the MCSE indicates we have enough MCMC iterations for
#' practically meaningful reporting of saying that the probability
#' that the temperature is increasing is larger than 99%. There is not
#' much practical difference to reporting that the probability is
#' 99.4% and to estimate that third digit accurately would require 64
#' times more iterations. For this simple problem, sampling that many
#' iterations would not be time consuming, but we might also instead
#' consider to obtain more data to verify that the summer temperature
#' in northern Finland has been increasing since 1952.
#'
#' 
#' ## Summary of workflow for how many digits to report
#'
#' 1. Run inference with some default number of iterations
#' 2. Check convergence diagnostics for all parameters
#' 3. Check that ESS is big enough for reliable convergence
#'    diagnostics for all parameters
#' 4. Look at the posterior for quantities of interest and decide how
#'    many significant digits is reasonable taking into account the
#'    posterior uncertainty (using SD, MAD, or tail quantiles)
#' 5. Check that MCSE is small enough for the desired accuracy of
#'    reporting the posterior summaries for the quantities of
#'    interest.
#'    - If the accuracy is not sufficient, report less digits or run more iterations.
#'    - Halving MCSE requires quadrupling the number of iterations.
#'    - Different quantities of interest have different MCSE and may require different
#'      number of iterations for the desired accuracy.
