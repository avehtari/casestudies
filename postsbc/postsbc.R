#' ---
#' title: "Simulation Based Calibration Conditional on Observed Data"
#' author: "Aki Vehtari"
#' date: "First version 2022-01-03. Last modified `r format(Sys.Date())`."
#' output:
#'   html_document:
#'     theme: readable
#'     toc: true
#'     toc_depth: 3
#'     toc_float: true
#'     code_download: true
#' bibliography: postsbc.bib
#' csl: harvard-cite-them-right.csl
#' ---
#'
#' # Introduction
#'
#' Simulation based calibration (SBC) can be used to validate
#' inference algorithms by repeated inference given repeated simulated
#' data from a generative model. The original and commonly used
#' approach generated the generative model parameters from prior, and
#' thus the approach is testing whether inference works for simulated
#' data generated with parameter values plausible under that
#' prior. This is natural and desirable when we want to test whether
#' the inference works for for many different types of data sets we
#' might observe. After observing data, we are interested whether the
#' inference did work conditional on that data, and the posterior
#' given the observed data is often much more concentrated than the
#' prior. Here we demonstrate how SBC can be used conditionally on
#' observed data.
#' 
#' ## Simulation based calibration
#' 
#' @Cook+Gelman+Rubin:2006 proposed a simulation-based calibration
#' method for validating Bayesian inference software. The idea is
#' based on the fact we can factor the joint distribution of data
#' $y$ and parameters $\theta$ in two ways
#' $$
#'   \pi(y,\theta) = \pi(y|\theta)\pi(\theta) = \pi(\theta|y)\pi(y).
#' $$
#' By considering $\theta'$ and $\theta''$ the joint distribution is 
#' $$
#'   \pi(y,\theta',\theta'') = \pi(y)\pi(\theta'|y)\pi(\theta''|y),
#' $$
#' and it's easy to see that $\theta'$ and $\theta''$ have the same
#' distribution conditionally on $y$. If we write the joint
#' distribution in an alternative way
#' $$
#'   \pi(y,\theta',\theta'') = \pi(\theta')\pi(y|\theta')\pi(\theta''|y),
#' $$
#' $\theta'$ and $\theta''$ still have the same distribution
#' conditionally on $y$. We can sample from the joint distribution
#' $\pi(y,\theta',\theta'')$ by first sampling from $\pi(\theta')$ and
#' $\pi(y|\theta')$, which is usually easy for generative models. The
#' last step is to sample from the conditional $\pi(\theta|y)$, which
#' is usually not trivial and instead, for example, a Markov chain
#' Monte Carlo algorithm is used. We can validate the algorithm and
#' its implementation used to sample from $\pi(\theta''|y)$ by
#' checking that the samples obtained have the same distribution as
#' $\theta'$ (conditionally on $y)$.
#' 
#' @Cook+Gelman+Rubin:2006 operationalize the approach by drawing
#' $\theta'_i$ from $\pi(\theta')$, generating data $y_i \sim
#' \pi(y_i|\theta'_i)$ and then using the algorithm to be validated to
#' draw a sample $\theta''_1,\ldots,\theta''_S \sim
#' \pi(\theta''|y_i)$. If the algorithm and its implementation are
#' correct, then $\theta'_i,\theta''_1,\ldots,\theta''_S$ conditional
#' on $y_i$ are draws from the same
#' distribution. @Cook+Gelman+Rubin:2006 propose to compute
#' empirical PIT valued for $\theta'_i$ that they show to be uniformly
#' distributed given $S \to \infty$. The process is repeated for
#' $i=1,\ldots,N$ and $N$ empirical PIT values are used for testing.
#' @Cook+Gelman+Rubin:2006 propose to use $\chi^2$-test for
#' the inverse of the normal CDF of the empirical PIT values. However,
#' with finite $S$ this approach doesn't correctly take into account
#' the discreteness or the effect of correlated sample from Markov
#' chain @Gelman:correction.
#' 
#' By thinning $\theta_1^{''},\ldots,\theta_S^{''}$ to be
#' approximately independent, the uniformity of empirical PIT values
#' can be tested with the approach presented in
#' Säilynoja+Buerkner+Vehtari:2020.
#'
#' ## Simulation based calibration conditional on observed data
#' 
#' When using weakly informative priors, most of the prior mass can be
#' in the region of the parameter space where the inference works
#' well, but a small amount of prior mass can also be in the regions
#' where inference is likely to fail. We could update the prior to
#' avoid such regions, but if the posterior given observed data is
#' concentrated far from the problematic regions we could instead
#' focus on assessing whether the inference works around the
#' posterior.
#'
#' Considering that given the sequential update rule in Bayesian
#' approach, an old posterior can be a new prior and we can naturally
#' consider SBC conditional on the observed data $y_{\mathrm{obs}}$.
#' Thus, we operationalize the approach by drawing $\theta'_i$ from
#' $\pi(\theta' | y_{\mathrm{obs}})$, generating new data $y_i \sim
#' \pi(y_i|\theta'_i)$ and then using the algorithm to be validated to
#' draw a sample $\theta''_1,\ldots,\theta''_S \sim
#' \pi(\theta'' | y_i, y_{\mathrm{obs}})$. If the algorithm and its
#' implementation are correct, then
#' $\theta'_i,\theta''_1,\ldots,\theta''_S$ conditional on $y_i$ and
#' $y_{\mathrm{obs}}$ are draws from the same distribution.
#'
#' - In prior SBC, the prior formulated so that it is easy to draw
#'   exactly from the prior (and we assume no mistakes are made when
#'   generating draws from the prior), and exact prior draws are
#'   compared to draws obtained by the approximate inference algorithm
#'   (the approach can also detect mistakes in the model code used in
#'   the inference).
#' - If we would be able to get exact draws from the posterior, we
#'   could directly compare these exact draws to any approximate
#'   inference result, and posterior SBC is not needed.
#' - In posterior SBC, we are comparing the same algorithm drawing
#'   from the original posterior and the updated posterior. If the
#'   algorithm is not consistent when observing more data, SBC may be
#'   able to detect this (with finite number of SBC iterations and
#'   finite sample sizes, we may miss small discrepancies).
#' - It is possible that inference could work given the observed data,
#'   but not anymore with additional data. The differences in the
#'   shape of the posterior given the observed data and the new
#'   posteriors in posterior SBC are likely to be smaller than the
#'   differences in the shape of different posteriors in the prior SBC
#'   approach.
#'
#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA, cache=FALSE)
#+ load_packages, echo=FALSE
library(cmdstanr)
library(posterior)
library(dplyr)
library(latex2exp)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))

#' # normal(mu, 1) model
#'
#' We start by illustrating the idea of prior SBC and posterior SBC
#' using a simple $\mathrm{normal}(\mu, 1)$ model. Given prior draws
#' of $\mu$, we generate data sets with $10$ observations.
#' 
#' ## Illustration of prior SBC
#'
#' Prior parameters
mu0=0
tau0=10
#' Plot the prior
p0 = ggplot(data = data.frame(x = c(-40, 40)), aes(x)) +
  stat_function(fun = dnorm, n = 101, args = list(mean = mu0, sd = tau0), linetype='dashed') +
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  ylab('') 
p0

#' Observation model scale
sigma=1
#' Number of observations
N=10

#' Run simulations
pp = p0
for (i in 1:10) {
  set.seed(1000+i)
  # draw from the prior
  mug=rnorm(1, mean=mu0, sd=tau0)
  # generate data from the predictive distribution given the parameter value sampled from the prior
  yg=rnorm(N, mean=mug, sd=sigma)
  # posterior
  ybar=mean(yg)
  mup=(mu0/tau0^2+N*ybar/sigma^2)/(1/tau0^2+N/sigma^2)
  taup=sqrt(1/(1/tau0^2+N/sigma^2))
  # draw from the posterior
  mupg=rnorm(1, mean=mup, sd=taup)
  # add prior draw, posterior, and posterior draw to the plot
  pp = pp +
    stat_function(fun = function(...) dnorm(...)/33, n = 1001, args = list(mean = mup, sd = taup), alpha=0.3) +
    annotate(geom = "point", x=mug, y=0, color='red', alpha=0.9)+
    annotate(geom = "point", x=mupg, y=0, color='blue', alpha=0.9)
}
pp

#' The typical feature of prior SBC is visible, that is, the
#' conditional posteriors are much more narrow than the prior
#' distribution. We illustrate later how this affects how the prior
#' and conditional posterior draws should be compared.
#' 
#' ## Illustration of posterior SBC
#'
#' We assume we have observed 10 observations with mean $8.6$. Given
#' posterior draws of $\mu$ given $y_\mathrm{obs}$, we generate
#' additional data sets with $10$ observations each. In posterior SBC,
#' we run the inference given original $y_\mathrm{obs}$ and new
#' additional data.
#' 
#' Prior
mu0=0
tau0=10
#' Observation model scale
sigma=1
#' Number of observations
N=10
#' Observed data mean
ybar=8.6
#' Posterior given the observed data
mup=(mu0/tau0^2+N*ybar/sigma^2)/(1/tau0^2+N/sigma^2)
taup=sqrt(1/(1/tau0^2+N/sigma^2))
#' Plot the posterior
pp1 = ggplot(data = data.frame(x = c(7.4, 9.7)), aes(x)) +
  stat_function(fun = dnorm, n = 101, args = list(mean = mup, sd = taup), linetype='dashed') +
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  ylab('') 
pp1

#' Run simulations 
pp2 = pp1
for (i in 1:10) {
  set.seed(1000+i)
  # draw from the posterior
  mug2=rnorm(1, mean=mup, sd=taup)
  # generate data from the predictive distribution given the parameter value sampled from the posterior
  yg2=rnorm(N, mean=mug2, sd=sigma)
  # second posterior
  ybar2=mean(yg2)
  mup2=(mup/taup^2+N*ybar2/sigma^2)/(1/taup^2+N/sigma^2)
  taup2=sqrt(1/(1/taup^2+N/sigma^2))
  # draw from the second posterior
  mupg2=rnorm(1, mean=mup2, sd=taup2)
  # add posterior draw, second posterior, and draw from the second posterior to the plot
  pp2 = pp2 +
    stat_function(fun = function(...) dnorm(...)/2, n = 1001, args = list(mean = mup2, sd = taup2), alpha=0.3) +
    annotate(geom = "point", x=mug2, y=0, color='red', alpha=0.9)+
    annotate(geom = "point", x=mupg2, y=0, color='blue', alpha=0.9)
}
pp2

#' We see the typical feature of posterior SBC, that is, the
#' conditional posteriors are only slightly more narrow than the
#' original posterior distribution (factor of $\sqrt{2}). We
#' illustrate later how this affects how the posterior and conditional
#' posterior draws should be compared.
#'
#' ## Prior SBC
#'
#' We now illustrate the behavior of prior SBC given a correct and
#' incorrect conditional posterior inference.
#' 
#' ### Correct posterior
mugs = mups = pits = numeric()
pp = p0
for (i in 1:1000) {
  set.seed(1000+i)
  mug=rnorm(1, mean=mu0, sd=tau0)
  mugs[i]=mug[1]
  yg=rnorm(N, mean=mug, sd=sigma)
  ybar=mean(yg)
  mup=(mu0/tau0^2+N*ybar/sigma^2)/(1/tau0^2+N/sigma^2)
  taup=sqrt(1/(1/tau0^2+N/sigma^2))
  mupg=rnorm(1000, mean=mup, sd=taup)
  mups[i]=mupg[1]
  pits[i]=mean(mups<mug)
}
df=data.frame(mugs,mups,pits)

p2=ggplot(data=df,aes(x=mugs,y=mups))+
  geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  labs(x=TeX('$\\mu\' \\sim p(\\mu)$'),y=TeX('$\\mu\'\' \\sim p(\\mu | y_i)$'))
plims = range(c(ggplot_build(p2)$layout$panel_params[[1]]$x.range,
                 ggplot_build(p2)$layout$panel_params[[1]]$y.range))
p2+lims(x=plims,y=plims)
#'
#' Prior draws and conditional posterior draws are highly correlated.
#'
ggplot(data=df,aes(x=sort(mugs),y=sort(mups)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  labs(x=TeX('sorted $\\mu\' \\sim p(\\mu)$'),y=TeX('sorted $\\mu\'\' \\sim p(\\mu | y_i)$'))
#'
#' QQ-plot also looks good.
#' 
ggplot(data=df,aes(x=seq(0,1,length.out=1000),y=sort(pits)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' ECDF of probability integral transformation (PIT) looks also good
#' as it should.
#' 

#' ### Incorrect posterior
#'
#' Here we have incorrect inference, so that posterior scale formula
#' is missing the square root (a mistake we actually first made).
#' 
mugs = mups = pits = numeric()
pp = p0
for (i in 1:1000) {
  set.seed(1000+i)
  mug=rnorm(1, mean=mu0, sd=tau0)
  mugs[i]=mug[1]
  yg=rnorm(N, mean=mug, sd=sigma)
  ybar=mean(yg)
  mup=(mu0/tau0^2+N*ybar/sigma^2)/(1/tau0^2+N/sigma^2)
  # correct
  # taup=sqrt(1/(1/tau0^2+N/sigma^2))
  # wrong
  taup=(1/(1/tau0^2+N/sigma^2))
  mupg=rnorm(1000, mean=mup, sd=taup)
  mups[i]=mupg[1]
  pits[i]=mean(mupg<mug)
}
df=data.frame(mugs,mups,pits)

p2=ggplot(data=df,aes(x=mugs,y=mups))+
  geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  labs(x=TeX('$\\mu\' \\sim p(\\mu)$'),y=TeX('$\\mu\'\' \\sim p(\\mu | y_i)$'))
plims = range(c(ggplot_build(p2)$layout$panel_params[[1]]$x.range,
                 ggplot_build(p2)$layout$panel_params[[1]]$y.range))
p2+lims(x=plims,y=plims)
#'
#' As the conditional posteriors are very narrow, the draws from the
#' conditional posteriors are highly correlated with the prior draws
#' and we can't see anything being wrong.
#' 
ggplot(data=df,aes(x=sort(mugs),y=sort(mups)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  labs(x=TeX('sorted $\\mu\' \\sim p(\\mu)$'),y=TeX('sorted $\\mu\'\' \\sim p(\\mu | y_i)$'))
#'
#' QQ-plot also looks good.
#' 
ggplot(data=df,aes(x=seq(0,1,length.out=1000),y=sort(pits)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' PIT plot looks terrible. The conditional posteriors are too narrow,
#' and thus PIT values are not uniformly distributed.
#' 

#' ## Posterior SBC
#'
#' We now illustrate the behavior of posterior SBC given a correct and
#' incorrect conditional posterior inference.
#' 
#' ### Correct posterior
#' 
#' Observed data mean
ybar=8.6
#' Posterior given the observed data
mup=(mu0/tau0^2+N*ybar/sigma^2)/(1/tau0^2+N/sigma^2)
taup=sqrt(1/(1/tau0^2+N/sigma^2))
#' Run simulations 
mug2s = mup2s = pit2s = numeric()
pp2 = pp1
for (i in 1:1000) {
  set.seed(1000+i)
  # draw from the posterior
  mug2=rnorm(1, mean=mup, sd=taup)
  mug2s[i]=mug2
  # generate data from the predictive distribution given the parameter value sampled from the posterior
  yg2=rnorm(N, mean=mug2, sd=sigma)
  # second posterior
  ybar2=mean(yg2)
  mup2=(mup/taup^2+N*ybar2/sigma^2)/(1/taup^2+N/sigma^2)
  taup2=sqrt(1/(1/taup^2+N/sigma^2))
  # draw from the second posterior
  mupg2=rnorm(1000, mean=mup2, sd=taup2)
  mup2s[i]=mupg2[1]
  pit2s[i]=mean(mupg2<mug2)
}
df=data.frame(mug2s,mup2s,pit2s)

p2=ggplot(data=df,aes(x=mug2s,y=mup2s))+
  geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  labs(x=TeX('$\\mu\' \\sim p(\\mu | y_{\\mathrm{obs}})$'),y=TeX('$\\mu\'\' \\sim p(\\mu | y_i, y_{\\mathrm{obs}})$'))
plims = range(c(ggplot_build(p2)$layout$panel_params[[1]]$x.range,
                 ggplot_build(p2)$layout$panel_params[[1]]$y.range))
p2+lims(x=plims,y=plims)
#'
#' Posterior draws and conditional posterior draws are weakly correlated.
#' 

ggplot(data=df,aes(x=sort(mug2s),y=sort(mup2s)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  labs(x=TeX('sorted $\\mu\' \\sim p(\\mu | y_{\\mathrm{obs}})$'),y=TeX('sorted $\\mu\'\' \\sim p(\\mu | y_i, y_{\\mathrm{obs}})$'))
#'
#' QQ-plot also looks good.
#' 
ggplot(data=df,aes(x=seq(0,1,length.out=1000),y=sort(pit2s)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' ECDF of probability integral transformation (PIT) looks also good
#' as it should.
#' 
#' ### Incorrect posterior 1
#' 
#' Here we have incorrect inference, so that posterior scale formula
#' is missing the square root.
#' 
#' Observed data mean
ybar=8.6
#' Posterior given the observed data
mup=(mu0/tau0^2+N*ybar/sigma^2)/(1/tau0^2+N/sigma^2)
taup=(1/(1/tau0^2+N/sigma^2))
#' Run simulations 
mug2s = mup2s = pit2s = numeric()
pp2 = pp1
for (i in 1:1000) {
  set.seed(1000+i)
  # draw from the posterior
  mug2=rnorm(1, mean=mup, sd=taup)
  mug2s[i]=mug2
  # generate data from the predictive distribution given the parameter value sampled from the posterior
  yg2=rnorm(N, mean=mug2, sd=sigma)
  # second posterior
  ybar2=mean(yg2)
  mup2=(mup/taup^2+N*ybar2/sigma^2)/(1/taup^2+N/sigma^2)
  taup2=(1/(1/taup^2+N/sigma^2))
  # draw from the second posterior
  mupg2=rnorm(1000, mean=mup2, sd=taup2)
  mup2s[i]=mupg2[1]
  pit2s[i]=mean(mupg2<mug2)
}
df=data.frame(mug2s,mup2s,pit2s)

p2 = ggplot(data=df,aes(x=mug2s,y=mup2s))+
  geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  labs(x=TeX('$\\mu\' \\sim p(\\mu | y_{\\mathrm{obs}})$'),y=TeX('$\\mu\'\' \\sim p(\\mu | y_i, y_{\\mathrm{obs}})$'))
plims <- range(c(ggplot_build(p2)$layout$panel_params[[1]]$x.range,
                 ggplot_build(p2)$layout$panel_params[[1]]$y.range))
p2+lims(x=plims,y=plims)
#'
#' Posterior draws and conditional posterior draws are weakly correlated, but 
#' the conditional posterior draws have much smaller variability.
#' 
ggplot(data=df,aes(x=sort(mug2s),y=sort(mup2s)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  labs(x=TeX('sorted $\\mu\' \\sim p(\\mu | y_{\\mathrm{obs}})$'),y=TeX('sorted $\\mu\'\' \\sim p(\\mu | y_i, y_{\\mathrm{obs}})$'))
#'
#' QQ-plot also shows that the conditional posterior draws have much
#' smaller variability.
#' 
ggplot(data=df,aes(x=seq(0,1,length.out=1000),y=sort(pit2s)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' PIT plot looks also terrible. The conditional posteriors are too
#' narrow, and thus PIT values are not uniformly distributed.
#' 

#' ### Incorrect posterior 2
#'
#' Here we're again underestimating the posterior variance, but less
#' than in the first incorrect inference example. Now we compute the
#' variance as 80% from the true posterior variance.
#' 
#' Observed data mean
ybar=8.6
#' Posterior given the observed data
mup=(mu0/tau0^2+N*ybar/sigma^2)/(1/tau0^2+N/sigma^2)
taup=0.8*sqrt(1/(1/tau0^2+N/sigma^2))
#' Run simulations 
mug2s = mup2s = pit2s = numeric()
pp2 = pp1
for (i in 1:1000) {
  set.seed(1000+i)
  # draw from the posterior
  mug2=rnorm(1, mean=mup, sd=taup)
  mug2s[i]=mug2
  # generate data from the predictive distribution given the parameter value sampled from the posterior
  yg2=rnorm(N, mean=mug2, sd=sigma)
  # second posterior
  ybar2=mean(yg2)
  mup2=(mup/taup^2+N*ybar2/sigma^2)/(1/taup^2+N/sigma^2)
  taup2=0.8*sqrt(1/(1/taup^2+N/sigma^2))
  # draw from the second posterior
  mupg2=rnorm(1000, mean=mup2, sd=taup2)
  mup2s[i]=mupg2[1]
  pit2s[i]=mean(mupg2<mug2)
}
df=data.frame(mug2s,mup2s,pit2s)

p2=ggplot(data=df,aes(x=mug2s,y=mup2s))+
  geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  labs(x=TeX('$\\mu\' \\sim p(\\mu | y_{\\mathrm{obs}})$'),y=TeX('$\\mu\'\' \\sim p(\\mu | y_i, y_{\\mathrm{obs}})$'))
plims <- range(c(ggplot_build(p2)$layout$panel_params[[1]]$x.range,
                 ggplot_build(p2)$layout$panel_params[[1]]$y.range))
p2+lims(x=plims,y=plims)
#'
#' Posterior draws and conditional posterior draws are weakly
#' correlated. It is difficult to see any discrepancy from this plot.
#' 
ggplot(data=df,aes(x=sort(mug2s),y=sort(mup2s)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  labs(x=TeX('sorted $\\mu\' \\sim p(\\mu | y_{\\mathrm{obs}})$'),y=TeX('sorted $\\mu\'\' \\sim p(\\mu | y_i, y_{\\mathrm{obs}})$'))
#'
#' QQ-plot indicates problems at tails. The conditional posterior
#' seems to be slightly too narrow.
#' 
ggplot(data=df,aes(x=seq(0,1,length.out=1000),y=sort(pit2s)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' PIT plot confirms the suspicious. The conditional posteriors are
#' too narrow, and thus PIT values are not uniformly distributed.
#' 
#' # 8-schools
#'
#' Next we illustrate the posterior SBC in case of a hierarchical
#' model where a certain parameterization can lead to a funnel shaped
#' posterior that is difficult to sample with fixed step size
#' (dynamic) Hamiltonian Monte Carlo.
#' 
#' 8-schools data
dat = list(J=8, y=c(28,8,-3,7,-1,1,18,12), sigma=c(15,10,16,11,9,11,10,18))

#' ## Non-centered parameterization - dynamic HMC
#'
#' For 8-schools data and model, it is known that non-centered
#' parameterization produces a posterior that is relatively easy to
#' sample with fixed step size dynamic HMC. We expect that posterior
#' SBC doesn't detect any problems.
#' 
#' 8-schools model with non-centered parameterization
mod_ncp = cmdstan_model(stan_file = 'schools_ncp.stan')

#' sample from the posterior given the observed data
#+ warning=FALSE
out = capture.output(
  fit_ncp <- mod_ncp$sample(data=dat, refresh=0, show_messages=FALSE, seed=0))
draws_ncp = as_draws_rvars(thin_draws(fit_ncp$draws(),20))
tau_ncp = as_draws_matrix(subset_draws(draws_ncp, variable="tau"))
#' draws from the posterior predictive distribution
yrep_ncp = as_draws_matrix(subset_draws(draws_ncp, variable="yrep"))

#' 200 iterations of posterior SBC
pitp_ncp = taup_ncp = numeric()
#+ warning=FALSE
for (j in 1:200) {
  # combine the original data with posterior predictive data
  datp = list(J = 2*dat$J,
              y = c(dat$y, yrep_ncp[j,]),
              sigma = rep(dat$sigma, 2))
  # sample from the second posterior
  out = capture.output(
    fitp <- mod_ncp$sample(data=datp, refresh=0, show_messages = FALSE, seed=j))
  drawsp = as_draws_rvars(fitp$draws())
  # one draw from the second posterior
  taup_ncp[j] = as_draws_matrix(drawsp$tau)[1]
  # PIT value
  pitp_ncp[j] = mean(drawsp$tau < as.vector(tau_ncp[j]))
}

df=data.frame(tau_ncp,taup_ncp,pitp_ncp)

ggplot(data=df,aes(x=tau_ncp,y=taup_ncp))+geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  scale_x_log10()+scale_y_log10()+
  labs(x=TeX('$\\tau\' \\sim p(\\tau | y_{\\mathrm{obs}})$'),y=TeX('$\\tau\'\' \\sim p(\\tau | y_i, y_{\\mathrm{obs}})$'))
#'
#' Posterior draws and conditional posterior draws are weakly correlated.
#' 
ggplot(data=df,aes(x=sort(tau_ncp),y=sort(taup_ncp)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  scale_x_log10()+scale_y_log10()+
  labs(x=TeX('sorted $\\tau\' \\sim p(\\tau | y_{\\mathrm{obs}})$'),y=TeX('sorted $\\tau\'\' \\sim p(\\tau | y_i, y_{\\mathrm{obs}})$'))
#'
#' QQ-plot looks good.
#' 

ggplot(data=df,aes(x=seq(0,1,length.out=200),y=sort(pitp_ncp)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' ECDF of probability integral transformation (PIT) looks also good
#' as we expected.
#' 
#' ## Centered parameterization - dynamic HMC
#'
#' For 8-schools data and model, it is known that centered
#' parameterization produces a posterior that has strong funnel shape
#' and with fixed step size (dynamic) HMC is unable to explore the
#' narrow part of the funnel. The HMC specific and generic MCMC
#' diagnostics indicate these problems, and thus posterior SBC is not
#' necessary here, but as a well known example 8-schools centered
#' parameterization works as a useful illustration.
#' 
#' 8-schools model with centered parameterization model
mod_cp = cmdstan_model(stan_file = 'schools_cp.stan')

#' sample from the posterior given the observed data
out = capture.output(
  fit_cp <- mod_cp$sample(data=dat, refresh=0, show_messages=FALSE, seed=0))
draws_cp = as_draws_rvars(thin_draws(fit_cp$draws(),20))
tau_cp = as_draws_matrix(subset_draws(draws_cp, variable="tau"))
#' draws from the posterior predictive distribution
yrep_cp = as_draws_matrix(subset_draws(draws_cp, variable="yrep"))

#' 200 iterations of posterior SBC
pitp_cp = taup_cp = numeric()
#+ warning=FALSE
for (j in 1:200) {
  # combine the original data with posterior predictive data
  datp = list(J = 2*dat$J,
              y = c(dat$y, yrep_cp[j,]),
              sigma = rep(dat$sigma, 2))
  # sample from the second posterior
  out = capture.output(
    fitp <- mod_cp$sample(data=datp, refresh=0, show_messages = FALSE, seed=j))
  drawsp = as_draws_rvars(fitp$draws())
  # one draw from the second posterior
  taup_cp[j] = as_draws_matrix(drawsp$tau)[1]
  # PIT value
  pitp_cp[j] = mean(drawsp$tau < as.vector(tau_cp[j]))
}

df=data.frame(tau_cp,taup_cp,pitp_cp)

p2=ggplot(data=df,aes(x=tau_cp,y=taup_cp))+
  geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  scale_x_log10()+
  scale_y_log10()+
  labs(x=TeX('$\\tau\' \\sim p(\\tau | y_{\\mathrm{obs}})$'),y=TeX('$\\tau\'\' \\sim p(\\tau | y_i, y_{\\mathrm{obs}})$'))
plims = range(c(ggplot_build(p2)$layout$panel_params[[1]]$x.range,
                 ggplot_build(p2)$layout$panel_params[[1]]$y.range))
p2+scale_x_log10(limits=10^plims)+
  scale_y_log10(limits=10^plims)
#'
#' Posterior draws and conditional posterior draws are weakly
#' correlated. It is difficult to see any discrepancy in this plot.
#' 
ggplot(data=df,aes(x=sort(tau_cp),y=sort(taup_cp)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  scale_x_log10()+
  scale_y_log10()+
  labs(x=TeX('sorted $\\tau\' \\sim p(\\tau | y_{\\mathrm{obs}})$'),y=TeX('sorted $\\tau\'\' \\sim p(\\tau | y_i, y_{\\mathrm{obs}})$'))
#'
#' QQ-plot reveals clear discrepancy in small values of $\tau$. Here
#' we do get sometimes much smaller conditional posterior draws than
#' the smallest original posterior draws, which indicates that the
#' inference for the original posterior is failing.
#' 
ggplot(data=df,aes(x=seq(0,1,length.out=200),y=sort(pitp_cp)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' PIT plot doesn't show the discrepancy that clearly, although there
#' is some suspicion in the small values.
#' 
#' ## Non-centered parameterization - ADVI
#'
#' Automatic differentiation variational inference uses normal
#' approximation. @Yao+Vehtari+Simpson+Gelman:2018 demonstrate that
#' (given enough computation time) it works reasonably for the
#' non-centered parameterization.
#' 
#' 8-schools model with non-centered parameterization
mod_ncp = cmdstan_model(stan_file = 'schools_ncp.stan')

#' sample from the posterior given the observed data
#+ warning=FALSE
out = capture.output(
  fit_ncpv <- mod_ncp$variational(data=dat, refresh=0, seed=0, tol_rel_obj=1e-4, iter=1e5))
draws_ncpv = as_draws_rvars(thin_draws(fit_ncpv$draws(),5))
tau_ncpv = as_draws_matrix(subset_draws(draws_ncpv, variable="tau"))
#' draws from the posterior predictive distribution
yrep_ncpv = as_draws_matrix(subset_draws(draws_ncpv, variable="yrep"))

#' 200 iterations of posterior SBC
pitp_ncpv = taup_ncpv = numeric()
#+ warning=FALSE
for (j in 1:200) {
  # combine the original data with posterior predictive data
  datp = list(J = 2*dat$J,
              y = c(dat$y, yrep_ncpv[j,]),
              sigma = rep(dat$sigma, 2))
  # sample from the second posterior
  out = capture.output(
    fitp <- mod_ncp$variational(data=datp, refresh=0, seed=j, tol_rel_obj=1e-4, iter=1e5))
  drawsp = as_draws_rvars(fitp$draws())
  # one draw from the second posterior
  taup_ncpv[j] = as_draws_matrix(drawsp$tau)[1]
  # PIT value
  pitp_ncpv[j] = mean(drawsp$tau < as.vector(tau_ncpv[j]))
}

df=data.frame(tau_ncpv,taup_ncpv,pitp_ncpv)

p2=ggplot(data=df,aes(x=tau_ncpv,y=taup_ncpv))+
  geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  scale_x_log10()+
  scale_y_log10()+
  labs(x=TeX('$\\tau\' \\sim p(\\tau | y_{\\mathrm{obs}})$'),y=TeX('$\\tau\'\' \\sim p(\\tau | y_i, y_{\\mathrm{obs}})$'))
plims = range(c(ggplot_build(p2)$layout$panel_params[[1]]$x.range,
                 ggplot_build(p2)$layout$panel_params[[1]]$y.range))
p2+scale_x_log10(limits=10^plims)+
  scale_y_log10(limits=10^plims)
#'
#' Posterior draws and conditional posterior draws are weakly
#' correlated. It is difficult to see any discrepancy in this plot.
#' 
ggplot(data=df,aes(x=sort(tau_ncpv),y=sort(taup_ncpv)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  scale_x_log10()+
  scale_y_log10()+
  labs(x=TeX('sorted $\\tau\' \\sim p(\\tau | y_{\\mathrm{obs}})$'),y=TeX('sorted $\\tau\'\' \\sim p(\\tau | y_i, y_{\\mathrm{obs}})$'))
#'
#' QQ-plot indicates that the original posterior is likely to be
#' narrower than the true posterior.
#' 
ggplot(data=df,aes(x=seq(0,1,length.out=200),y=sort(pitp_ncpv)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' PIT plot indicates problems in the extreme left tail.
#' 
#' ## Centered parameterization - ADVI
#' 
#' 8-schools model with centered parameterization model
mod_cp = cmdstan_model(stan_file = 'schools_cp.stan')

#' sample from the posterior given the observed data
out = capture.output(
  fit_cpv <- mod_cp$variational(data=dat, refresh=0, seed=0, tol_rel_obj=1e-4, iter=1e5))
draws_cpv = as_draws_rvars(thin_draws(fit_cpv$draws(),5))
tau_cpv = as_draws_matrix(subset_draws(draws_cpv, variable="tau"))
#' draws from the posterior predictive distribution
yrep_cpv = as_draws_matrix(subset_draws(draws_cpv, variable="yrep"))

#' 200 iterations of posterior SBC
pitp_cpv = taup_cpv = numeric()
#+ warning=FALSE
for (j in 1:200) {
  # combine the original data with posterior predictive data
  datp = list(J = 2*dat$J,
              y = c(dat$y, yrep_cpv[j,]),
              sigma = rep(dat$sigma, 2))
  # sample from the second posterior
  out = capture.output(
    fitp <- mod_cp$variational(data=datp, refresh=0, seed=200+j, tol_rel_obj=1e-4, iter=1e5))
  drawsp = as_draws_rvars(fitp$draws())
  # one draw from the second posterior
  taup_cpv[j] = as_draws_matrix(drawsp$tau)[1]
  # PIT value
  pitp_cpv[j] = mean(drawsp$tau < as.vector(tau_cpv[j]))
}

df=data.frame(tau_cpv,taup_cpv,pitp_cpv)

p2=ggplot(data=df,aes(x=tau_cpv,y=taup_cpv))+
  geom_point(alpha=0.5,color='blue')+
  geom_abline()+
  scale_x_log10()+
  scale_y_log10()+
  labs(x=TeX('$\\tau\' \\sim p(\\tau | y_{\\mathrm{obs}})$'),y=TeX('$\\tau\'\' \\sim p(\\tau | y_i, y_{\\mathrm{obs}})$'))
plims = range(c(ggplot_build(p2)$layout$panel_params[[1]]$x.range,
                 ggplot_build(p2)$layout$panel_params[[1]]$y.range))
p2+scale_x_log10(limits=10^plims)+
  scale_y_log10(limits=10^plims)
#'
#' Posterior draws and conditional posterior draws are weakly
#' correlated. It is difficult to see any discrepancy in this plot.
#' 
ggplot(data=df,aes(x=sort(tau_cpv),y=sort(taup_cpv)))+
  geom_point(alpha=.3,color='blue')+
  geom_abline()+
  scale_x_log10()+
  scale_y_log10()+
  labs(x=TeX('sorted $\\tau\' \\sim p(\\tau | y_{\\mathrm{obs}})$'),y=TeX('sorted $\\tau\'\' \\sim p(\\tau | y_i, y_{\\mathrm{obs}})$'))
#'
#' QQ-plot has some structure, but no clear indication of the
#' problems.
#' 
ggplot(data=df,aes(x=seq(0,1,length.out=200),y=sort(pitp_cpv)))+
  geom_line(color='blue',size=2)+
  geom_abline()+
  labs(x='Uniform',y='PIT')
#'
#' PIT plot shows clearly that the posterior variance is underestimated.
#'
#' ## Comparison of approximations
#'
#' After seeing the diagnostics, we compare all posterior
#' approximations and the conditional posteriors.
#' 
rtau=as_draws_df(rvar(cbind(ncp=as.vector(tau_ncp),pncp=taup_ncp,
                            cp=as.vector(tau_cp),pcp=taup_cp,
                            ncpv=as.vector(tau_ncpv),pncpv=taup_ncpv,
                            cpv=as.vector(tau_cpv),pcpv=taup_cpv)))
rtau<-rename_variables(rtau,
                       "Non-centered HMC"='x[ncp]',
                       "Non-centered HMC-SBC"='x[pncp]',
                       "Centered HMC"='x[cp]',
                       "Centered HMC-SBC"='x[pcp]',
                       "Non-centered ADVI"='x[ncpv]',
                       "Non-centered ADVI-SBC"='x[pncpv]',
                       "Centered ADVI"='x[cpv]',
                       "Centered ADVI-SBC"='x[pcpv]')
mcmc_areas(as_draws_matrix(log10(rtau)))+
  scale_x_continuous(breaks=c(-1,0,1),labels=c('0.1','1.0','10.0'))+
  xlab(TeX('$\\tau$'))+
  theme(axis.line.y=element_blank(),
        axis.ticks.y=element_blank())
#'
#' We see that
#' 
#' - non-centered HMC matches non-centered HMC-SBC
#' - centered HMC is missing smaller values of tau, which is revealed
#'   by centered HMC-SBC
#' - non-centered ADVI is close to non-centered HMC-SBC. The ADVI
#'   normal approximation has most of the mass where the true
#'   posterior (based on non-centered HMC), but the normal
#'   approximation is missing the skewness of the true posterior and
#'   this was not indicated by the posterior SBC.
#' - Centered ADVI looks similar to centered HMC-SBC. The ADVI normal
#'   approximation is very different from the true posterior (based on
#'   non-centered HMC), and the posterior SBC did indicate severe
#'   underestimation of the posterior variance.
#'
#' <br />
#' 
#' # References {.unnumbered}
#' 
#' <div id="refs"></div>
#' 
#' # Licenses {.unnumbered}
#' 
#' * Code &copy; 2022, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2022, Aki Vehtari, licensed under CC-BY-NC 4.0.
#' 
#' # Original Computing Environment {.unnumbered}
#' 
sessionInfo()
#' 
#' <br />
