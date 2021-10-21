#' ---
#' title: "Illustration of simple problematic posteriors"
#' author: "Aki Vehtari"
#' date: "First version 2021-06-10. Last modified `r format(Sys.Date())`."
#' output:
#'   html_document:
#'     theme: readable
#'     toc: true
#'     toc_depth: 2
#'     toc_float: true
#'     code_download: true
#' ---


#' Demonstration of simple problematic distributions and how to
#' interpret the diagnostics.
#' 

#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA, cache=FALSE)

#' #### Load packages
library("rprojroot")
root<-has_file(".Workflow-Examples-root")$make_fix_file()
library(cmdstanr) 
library(posterior)
options(pillar.neg = FALSE, pillar.subtle=FALSE, pillar.sigfig=2)
library(lemon)
library(tidyr) 
library(dplyr) 
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size=16))
set1 <- RColorBrewer::brewer.pal(7, "Set1")
SEED <- 48927 # set random seed for reproducibility

#'
#' ## Improper posterior
#' 
#' Unbounded likelihood without proper prior leading to improper
#' posterior
#'
#' ### Data
#'
#' Univariate continous x, binary y, and the two classes are
#' completely separable, which leads to unbounded likelihood.
set.seed(SEED+4)
M=1;
N=10;
x=matrix(sort(rnorm(N)),ncol=M)
y=rep(c(0,1), each=N/2)
data_logit <-list(M = M, N = N, x = x, y = y)
ggplot() +
  geom_point(aes(x, y), data = data.frame(data_logit), size = 3)+
  scale_y_continuous(breaks=c(0,1))

#'
#' ### Model
#'
#' A simple Bernoulli regression (where we have forgotten to include priors)
code_logit <- root("problems", "logit_glm.stan")
writeLines(readLines(code_logit))
#' Sample
mod_logit <- cmdstan_model(stan_file = code_logit)
fit_logit <- mod_logit$sample(data = data_logit, seed = SEED, refresh = 0)

#'
#' ### Convergence diagnostics
#' 
#' There are convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_logit$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values
draws <- as_draws_rvars(fit_logit$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plot
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta"))

#' Quite clear case
#' 
#' ### A fixed model with proper priors
#'
#' A simple Bernoulli regression with proper prior
code_logit2 <- root("problems", "logit_glm2.stan")
writeLines(readLines(code_logit2))
#' Sample
mod_logit2 <- cmdstan_model(stan_file = code_logit2)
fit_logit2 <- mod_logit2$sample(data = data_logit, seed = SEED, refresh = 0)

#'
#' ### Convergence diagnostics
#' 
#' There were no convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_logit2$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values, which in this case all look good.
draws <- as_draws_rvars(fit_logit2$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plot
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta"))

#' No problems
#' 

#'
#' ## A model with unused parameter
#' 
#' A simple Bernoulli regression with proper prior (but we have
#' forgotten to remove unused parameter declaration)
code_logit3 <- root("problems", "logit_glm3.stan")
writeLines(readLines(code_logit3))
#' Sample
mod_logit3 <- cmdstan_model(stan_file = code_logit3)
fit_logit3 <- mod_logit3$sample(data = data_logit, seed = SEED, refresh = 0)

#'
#' ### Convergence diagnostics
#' 
#' There are convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_logit3$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values
draws <- as_draws_rvars(fit_logit3$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plots
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta","gamma"))

#' A case where trace plot is actually useful
mcmc_trace(as_draws_array(draws), pars=c("gamma"))

#'
#' ## A posterior with two parameters competing
#'
#' ### Data
#'
#' We add another column to the previous data matrix. Sometimes data
#' matrix is augmented with a column of 1's, to present the intercept
#' effect, but in this case that is redundant as our model has
#' explicit intercept term `alpha`.
M=2;
N=1000;
x=matrix(c(rep(1,N),sort(rnorm(N))),ncol=M)
y=((x[,1]+rnorm(N)/2)>0)+0
data_logit4 <-list(M = M, N = N, x = x, y = y)

#' We use the previous Bernoulli regression model with proper priors.
code_logit2 <- root("problems", "logit_glm2.stan")
writeLines(readLines(code_logit2))
#' Sample
mod_logit4 <- cmdstan_model(stan_file = code_logit2)
fit_logit4 <- mod_logit4$sample(data = data_logit4, seed = SEED, refresh = 0)

#' The computation time per chain with the original x with just one
#' column was less than 0.1s per chain. Now the computation time per
#' chain is several seconds, which is suspicious.

#'
#' ### Convergence diagnostics
#' 
#' There were no convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_logit4$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values, which in this case are ok,
#' but ESS's are lower than what we would expect from Stan for such a
#' lower dimensional problem.
draws <- as_draws_rvars(fit_logit4$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plots
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta[1]","beta[2]"))

#' And there it is: `alpha` and `beta[1]` are super-correlated.
#' We can compute the correlation.
cor(as_draws_matrix(draws)[,c("alpha","beta[1]")])[1,2]

#' The correlation close to 1 can happen also from other reasons (see
#' the next example), but one possibility is that parameters have
#' similar role in the model. Here the reason is the constant column
#' in x, which we put there for the demonstration purposes. We may
#' have constant column also if the predictor matrix is augmented
#' unnecessarily with the intercept predictor, or if the observed data
#' or subdata used in the specific analysis just happens to have only
#' one unique value.
#' 

#'
#' ## A posterior with very high correlation
#'
#' The data are Kilpisjärvi summer month temperatures 1952-2013.
data_kilpis <- read.delim(root("problems","kilpisjarvi-summer-temp.csv"), sep = ";")
data_lin <-list(M=1,
                N = nrow(data_kilpis),
                x = matrix(data_kilpis$year, ncol=1),
                y = data_kilpis[,5])

#' Plot the data
ggplot() +
  geom_point(aes(x, y), data = data.frame(data_lin), size = 1) +
  labs(y = 'Summer temp. @Kilpisjärvi', x= "Year") +
  guides(linetype = "none")

#' We use a linear model
code_lin <- root("problems", "linear_glm_kilpis.stan")
writeLines(readLines(code_lin))

#' Run Stan
mod_lin <- cmdstan_model(stan_file = code_lin)
fit_lin <- mod_lin$sample(data = data_lin, seed = SEED, refresh = 0)

#' Stan gives a warning: There were X transitions after warmup that exceeded the maximum treedepth. 
#' 
#' We can check other diagnostics as follows
fit_lin$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values, which are fine.
draws <- as_draws_rvars(fit_lin$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plots
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta"))

#' And there it is: `alpha` and `beta` are super-correlated.  Here the
#' reason is that the x values are in the range 1952--2013, and the
#' inrecept alpha denotes the temperature at year 0 which is very far
#' away from the data. If the intercept alpha changes, the slope beta
#' needs to change too. The high correlation makes the inference
#' slower, and we can make it faster by centering x.
#'
#' Here we simply subtract 1982.5 from the year, so that the mean of x
#' is 0. We could also include the centering and back transformation
#' to Stan code.
data_lin <-list(M=1,
                N = nrow(data_kilpis),
                x = matrix(data_kilpis$year-1982.5, ncol=1),
                y = data_kilpis[,5])

fit_lin <- mod_lin$sample(data = data_lin, seed = SEED, refresh = 0)

#' Now treedepth exceedence warnings.
#' 
#' We can check other diagnostics as follows
fit_lin$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values, which are now even better.
draws <- as_draws_rvars(fit_lin$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plots
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta"))

#'
#' The posterior dependency has disappeared and the interpretation of
#' alpha is the average temperature over all the observed years.
#' 

#'
#' ## A bimodal posterior
#'
#' A toy example of bimodal distribution. Bimodal distributions can
#' arise from many reasons as in mixture models or models with
#' non-log-concave likelihoods or priors (ie with thick tails).
#' 
#' ### Data
#'
#' Bimodally distributed data
N=20
y=c(rnorm(N/2, mean=-5, sd=1),rnorm(N/2, mean=5, sd=1));
data_tt <-list(N = N, y = y)

#' Student's t model
code_tt <- root("problems", "student.stan")
writeLines(readLines(code_tt))
#' Sample
mod_tt <- cmdstan_model(stan_file = code_tt)
fit_tt <- mod_tt$sample(data = data_tt, seed = SEED, refresh = 0)

#'
#' ### Convergence diagnostics
#' 
#' There are convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_tt$cmdstan_diagnose()

#' High Rhat and very low ESS

#' We check $\widehat{R}$ end ESS values.
draws <- as_draws_rvars(fit_tt$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Histogram shows two modes
mcmc_hist(as_draws_array(draws), pars=c("mu"))

#' Trace plot shows that the chains are not mixing between the modes
mcmc_trace(as_draws_array(draws), pars=c("mu"))

#' ### Easy bimodal posterior
#'
#' The same example, but with this data, the modes are close enough
#' that it's easy for MCMC to jump from one mode to another.
N=20
y=c(rnorm(N/2, mean=-3, sd=1),rnorm(N/2, mean=3, sd=1));
data_tt <-list(N = N, y = y)

fit_tt <- mod_tt$sample(data = data_tt, seed = SEED, refresh = 0)

#' ### Convergence diagnostics
#' 
#' There were no convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_tt$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values, which in this case all look good.
draws <- as_draws_rvars(fit_tt$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Two modes are visible
mcmc_hist(as_draws_array(draws), pars=c("mu"))

#' Trace plot is not very useful. It shows the chains are jumping
#' between modes, but it's difficult to see whether the jumps happen
#' often enough and chains are mixing well.
mcmc_trace(as_draws_array(draws), pars=c("mu"))

#' Rank histogram plot
mcmc_rank_hist(as_draws_array(draws), pars=c("mu"))

#' Rank ECDF plot
#'
#' **Add here**

#'
#' ## Initial value issues
#'
#' MCMC requires some initial values. By default Stan generates them
#' randomly from [-2,2] (in unconstrained space). Sometimes these
#' initial values can be dbad and cause numerical issues. Computers,
#' in general, use finite number of bits to present numbers and with
#' very small or large numbers, there can be problems of presenting
#' them or there can be significant loss of accuracy.
#'
#' The data is generated from a Poisson regression model. The Poisson
#' intensity parameter has to be positive and usually the latent
#' linear predictor is exponentiated to be positive (the
#' exponentiation can also be justified by multiplicative effects on
#' Poisson intensity).
set.seed(SEED)
M=1;
N=20;
x=1e3*matrix(c(sort(rnorm(N))),ncol=M)
y=rpois(N,exp(1e-3*x[,1]))
data_pois <-list(M = M, N = N, x = x, y = y)
ggplot() +
  geom_point(aes(x, y), data = data.frame(data_pois), size = 3)

#' Poisson regression model with proper priors
code_pois <- root("problems", "pois_glm.stan")
writeLines(readLines(code_pois))
#' Sample
mod_pois <- cmdstan_model(stan_file = code_pois)
fit_pois <- mod_pois$sample(data = data_pois, seed = SEED, refresh = 0)

#' We get a lot of warnings. Uh, they show in console, but not in the notebook!
#'
#'```
#' Chain 4 Rejecting initial value:
#' Chain 4   Log probability evaluates to log(0), i.e. negative infinity.
#' Chain 4   Stan can't start sampling from this initial value.
#'```

#'
#' ### Convergence diagnostics
#' 
#' There are convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_pois$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values
draws <- as_draws_rvars(fit_pois$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plot
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta"))

#' The reason for the issue is that the initial values for `beta` is
#' sampled from [-2, 2] and x has some large values. If the initial
#' value for `beta` is higher than about 0.3 or lower than -0.4, some
#' of the values of exp(alpha + beta * x) will overflow to `Inf`.
#'
#' In this case the problem is alleviated by scaling the x

data_pois <-list(M = M, N = N, x = x/1e3, y = y)
ggplot() +
  geom_point(aes(x, y), data = data.frame(data_pois), size = 3)

#' Poisson regression model with proper priors
code_pois <- root("problems", "pois_glm.stan")
writeLines(readLines(code_pois))
#' Sample
mod_pois <- cmdstan_model(stan_file = code_pois)
fit_pois <- mod_pois$sample(data = data_pois, seed = SEED, refresh = 0)

#' We get a lot of warnings
#'
#' Chain 4 Rejecting initial value:
#' Chain 4   Log probability evaluates to log(0), i.e. negative infinity.
#' Chain 4   Stan can't start sampling from this initial value.
#'

#'
#' ### Convergence diagnostics
#' 
#' There are convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_pois$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values
draws <- as_draws_rvars(fit_pois$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plot
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta"))

#' Everything works fine.
#'
#' It can be sometimes difficult to find a good initial values.
#'
#' If the initial value warning comes only once, it is possible that
#' MCMC was able to escape the bad region and rest of the inference is
#' ok.
#'
#' We expect Pathfinder to help with initial values.
#' 

#' ## Thick tailed posterior
#'
#' A simple Bernoulli regression with proper but thick tailed prior (Cauchy)
code_logit4 <- root("problems", "logit_glm4.stan")
writeLines(readLines(code_logit4))
#' Sample
mod_logit4 <- cmdstan_model(stan_file = code_logit4)
fit_logit4 <- mod_logit4$sample(data = data_logit, seed = SEED, refresh = 0)

#'
#' ### Convergence diagnostics
#' 
#' There were no convergence issues reported by sampling. We can also
#' explicitly call CmdStan inference diagnostics:
fit_logit4$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values, which in this case all look good.
draws <- as_draws_rvars(fit_logit4$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' ESSs are not that big, but not really alarming

#' Plot
mcmc_pairs(as_draws_array(draws), pars=c("alpha","beta"))

#' Tail is long

#' Rank-plots
mcmc_rank_hist(as_draws_array(draws), pars=c("alpha"))

#' Histograms are not really uniform.

#'
#' ## Variance parameter that is not constrained to be positive
#'
#' Demonstration what happens if we forget <lower=0> from a parameter
#' that has to be positive.
#' 
#' ### Data
#'
#' We simulated x and y independently from normal distributions. As
#' N=8 is small, there will be lot of uncertainty about the parameters
#' including the scale sigma.
M=1;
N=8;
set.seed(SEED)
x=matrix(rnorm(N),ncol=M)
y=rnorm(N)/10
data_lin <-list(M = M, N = N, x = x, y = y)

#' We use linear regression model with proper priors.
code_lin <- root("problems", "linear_glm.stan")
writeLines(readLines(code_lin))
#' Sample
mod_lin <- cmdstan_model(stan_file = code_lin)
fit_lin <- mod_lin$sample(data = data_lin, seed = SEED, refresh = 0)

#' We get many times the following warnings
#'```
#' Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#' Chain 4 Exception: normal_id_glm_lpdf: Scale vector is -0.747476, but must be positive finite! (in '/tmp/RtmprEP4gg/model-7caa12ce8e405.stan', line 16, column 2 to column 43)
#' Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#' Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#'```
#'
#' Sometimes these happen in early phase, even if the model has been
#' correctly defined, but now we have too many of them, which
#' indicates the samples is trying to jump to infeasible values, which
#' here means the negative scale parameter values. Many rejections may
#' lead to biased estimates.
#'
#' ### Convergence diagnostics
#' 
#' There are some divergences reported, which is not necessarily
#' alarming, but in this case are likely to be related to sub-optimal
#' stepsize adaptation due to the many rejections. We can also
#' explicitly call CmdStan inference diagnostics:
fit_lin$cmdstan_diagnose()

#' We check $\widehat{R}$ end ESS values, which in this case are ok.
draws <- as_draws_rvars(fit_lin$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws)

#' Plots
mcmc_hist(as_draws_array(draws), pars=c("sigma"))+xlim(c(0,0.31))

#' Fixed model inlcudes <lower=0> constraint for sigma.
code_lin2 <- root("problems", "linear_glm2.stan")
writeLines(readLines(code_lin2))
#' Sample
mod_lin2 <- cmdstan_model(stan_file = code_lin2)
fit_lin2 <- mod_lin2$sample(data = data_lin, seed = SEED, refresh = 0)

#' No sampling warnings

#' We check $\widehat{R}$ end ESS values, whic are ok.
draws2 <- as_draws_rvars(fit_lin2$draws())
#+ render=lemon_print, digits=c(0,2,2,2,2,2,2,2,0,0)
summarize_draws(draws2)

#' Plots
mcmc_hist(as_draws_array(draws), pars=c("sigma"))+xlim(c(0,0.31))

#'
#' In this specific case the bias is negliglible, but the sampling
#' with the proper constraint is more efficient.
#' 
