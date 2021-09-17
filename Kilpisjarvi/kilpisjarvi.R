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
SEED <- 48927 # set random seed for reproducibility

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
  labs(y = 'Summer temperature\n at Kilpisjärvi', x= "Year")

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

#' ## Run inference for some number of iterations
#+ results='hide'
mod_lin <- cmdstan_model(stan_file = code_lin)
fit_lin <- mod_lin$sample(data = data_lin_priors, seed = SEED)

#' ## Run convergence diagnostics
#' There were no convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_lin$cmdstan_diagnose()

#' And we can also look at the values of Rhats end ESS's:
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
draws <- as_draws_rvars(fit_lin$draws())
summarize_draws(draws)

#' Compute posterior draws for the linear fit
draws$mu <- draws$alpha_c+draws$beta*(data_lin$x-mean(data_lin$x))
#' Plot the linear fit with 90% posterior interval
data.frame(x = data_lin$x,
           y = data_lin$y,
           Emu = mean(draws$mu),
           q05 = quantile(draws$mu, 0.05),
           q95 = quantile(draws$mu, 0.95)) %>% 
  ggplot() +
  geom_ribbon(aes(x=x, ymin=q05, ymax=q95), fill='grey90') +
  geom_line(aes(x=x, y=Emu, )) +
  geom_point(aes(x, y), size = 1) +
  labs(y = 'Summer temperature\n at Kilpisjärvi (°C)', x= "Year")+
  guides(linetype = "none")
if (savefigs) ggsave(root("Kilpisjarvi","kilpisjarvi_fit.pdf"),
                     width=6, height=3)

#' At this point it's sufficient that diagnostics are OK
#' (cmdstan_diagnose says "no problems detected") and effective sample
#' sizes are large enough (>400) that we can assume the diagnostics to
#' be reliable.
#'
#' Normally we would also do posterior predictive checking and residual
#' plots, but now we focus on checking how many MCMC iterations are
#' needed and how many digits to report in posterior summary results.
#' 

#' ## How many digits to report based on posterior uncertainty
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

#' The number of digits shown in R varies depending on the print
#' method and options for the specific object and above we see 3
#' decimal digits. Depending on the print method, We may also see
#' more digits, for example:
mean(draws$beta)
quantile(draws$beta, probs=c(0.05,0.95))

#' The more digits are shown, the more of them are likely to be
#' unnecessary clutter distracting the reader from the important
#' message.  Considering the width of the 90% interval, practically
#' meaningful accuracy would be here to report posterior mean as 2.0
#' and the posterior interval as [0.7 , 3.2]. Depending on the
#' context, it might be even better to round more, and report that the
#' increase is estimated to be 1 to 3 degrees per century (81%
#' probability), or 0 to 4 degrees per century (99%
#' probability). There is no need to stick to reporting 90% interval.
#'
#' ## How many digits to report based on Monte Carlo standard error
#'
#' Now that we have an estimate for the posterior uncertainty of the
#' slope parameter, we can check whether we have enough many
#' iterations for the desired reporting accuracy. Markov chain Monte
#' Carlo method used by Stan to make the posterior estimates is
#' stochastic. If we repeat the computation with different random
#' number generator seed, we get slightly different estimates.
#+ results='hide'
estimates <- t(sapply(1:10, function(i) {
  mod_lin$sample(data = data_lin_priors, seed = SEED+i, refresh = 0,
                 show_messages = FALSE)$draws() %>%
    mutate_variables(beta100 = 100*beta) %>%
    subset_draws("beta100") %>%
      summarize_draws(mean, ~quantile(.x, probs = c(0.05, 0.95))) %>%
      select(-variable)})) %>% unlist() %>% matrix(nrow=10)
colnames(estimates) <- c("mean" ,"5%","95%")
#+ render=lemon_print, digits=3
as_tibble(estimates)

#' We see that for mean the third digit is varying and the rounded
#' value is between 1.9 and 2.0. For 5% quantile even the first
#' significant digit is sometimes varying and the rounded value would
#' vary between 0.6 and 0.7.  For 95% quantile the second digit is
#' varying and the rounded values would vary between 3.2 and
#' 3.3. Based on this, it would be OK to report the mean as 1.9 or 2
#' and 90% interval as [0.7, 3.2] as based on the first Monte Carlo
#' estimate. Considering the scale, the minor variation in the last
#' digit is not affecting the interpretation of the
#' results. Alternatively the courser ranges [1, 3] or [0, 4] could be
#' reported as discussed above.
#' 
#' Instead of repeating the estimation many times we can estimate the
#' accuracy of the original sampling by computing Monte Carlo standard
#' error that takes into account the quantity of interest, and the
#' effective sample size of MCMC draws.
#'
#+ render=lemon_print, digits=3
draws %>%
  subset_draws("beta100") %>%
  summarize_draws(mcse_mean, ~mcse_quantile(.x, probs = c(0.05, 0.95)))

#' For the mean MCSE is about 0.01 and for 5% and 95% quantiles about
#' 0.03. If we multiply these by 2, the likely range of variation due
#' to Monte Carlo is $\pm\!$ 0.02 for mean and $\pm\!$ 0.07 for 5% and 95%
#' quantiles. From this we can interpret that it's unlikely there
#' would be variation in th report estimate for the mean if it is
#' reported as 2.0. For 5% and 95% quantiles there can be variation in
#' the first decimal digit, but that difference would in many cases be
#' not meaningful. We could run more iterations to get the MCSE for
#' quantiles down to something like 0.01, which would require about 10
#' times more iterations. Assuming the posterior has finite variance
#' and MCMC is mixing well, we expect that running four times more
#' iterations would halve the MCSEs.
#'
#' As discussed before it might be anyway better to report the
#' posterior uncertainty interval with less digits and if we would
#' report either 81% interval [1, 3] or 99% interval [0, 4], we would
#' already have sufficient accuracy for the number of shown
#' digits. These MCSE estimates illustrate the fact that usually tail
#' quantiles have lower accuracy than the posterior mean.
#'
#'
#' We can also report the probability that the temperature change is
#' positive.
#+ render=lemon_print, digits=3
draws %>%
  subset_draws("betapos") %>%
  summarize_draws("mean", mcse = mcse_mean)

#' The MCSE indicates we have enough MCMC iterations for practically
#' meaningful reporting of saying that the probability that the
#' temperature is increasing is larger than 99%. There is not much
#' practical difference to reporting that the probability is 99.3% and
#' to estimate that third digit accurately would require 64 times more
#' iterations. For this simple problem, sampling that many iterations
#' would not be time consuming, but we might also instead consider to
#' obtain more data to verify that the summer temperature in northern
#' Finland has been increasing since 1952.
#'
