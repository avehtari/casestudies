#' ---
#' title: "Laplace method and Jacobian of parameter transformation"
#' author: "[Aki Vehtari](https://users.aalto.fi/~ave/)"
#' date: "First version 2021-01-21. Last modified `r format(Sys.Date())`."
#' output:
#'   html_document:
#'     theme: readable
#'     toc: true
#'     toc_depth: 2
#'     toc_float: true
#'     code_download: true
#' ---
#' 


#' ## Introduction
#' 
#' Stan recently got Laplace approximation algorithm (see [Stan
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

#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA, cache=FALSE)

#' #### Load packages
library("rprojroot")
root<-has_file(".casestudies-root")$make_fix_file()
library(tidyr) 
library(dplyr) 
library(cmdstanr) 
options(mc.cores = 1)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size=16))
theme_remove_yaxis <- theme(axis.text.y = element_blank(),
                            axis.line.y=element_blank(),
                            axis.ticks.y=element_blank())
set1 <- RColorBrewer::brewer.pal(7, "Set1")
library(latex2exp)
library(posterior)
SEED <- 48927 # set random seed for reproducibility

#' ## Example model and posterior
#' 
#' For the illustration Binomial model is used with observed data
#' $N=10, y=9$.
data_bin <- list(N = 10, y = 9)

#' As Beta(1,1) prior is used the posterior is Beta(9+1,1+1), but for
#' illustration we also use Stan to find the mode of the posterior,
#' sample from the posterior, and compare different posterior density
#' values that we can ask Stan to compute. We use
#' `compile_model_methods=TRUE` to be able to access `log_prob()`
#' method later.
code_binom <- root("Jacobian","binom.stan")
writeLines(readLines(code_binom))
model_bin <- cmdstan_model(stan_file = code_binom,
                           compile_model_methods=TRUE, force_recompile=TRUE)

#' Default MCMC sampling (as this is an easy posterior we skip showing
#' the results for the convergence diagnostics).
fit_bin <- model_bin$sample(data = data_bin, seed = SEED, refresh=0)

#' Default optimization finds the maximum of the posterior.
opt_bin <- model_bin$optimize(data = data_bin, seed = SEED)

#' The following plot shows the exact posterior (black) and grey
#' vertical line shows the MAP, that is, posterior mode in the
#' constrained space. Stan optimizing finds correctly the mode.
df_bin <- data.frame(theta=plogis(seq(-4,6,length.out=100))) |>
  mutate(pdfbeta=dbeta(theta,9+1,1+1))
ggplot(data=df_bin,aes(x=theta,y=pdfbeta))+
  geom_line()+
  geom_vline(xintercept=(opt_bin$draws()[1,'theta']),color="gray")+
  labs(x=TeX(r'($\theta$)'),y='')+
  theme_remove_yaxis+
  annotate("text", x=0.81,y=3.3,label=TeX(r'($p(\theta|y)$)'),color=1,size=5,hjust=1)+
  annotate("text", x=0.89,y=4.3,label='mode',color="gray",size=5,hjust=1)

#' ## Posterior log_prob
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
#' `jacobian=FALSE`. We create a helper function.
fit_pdf <- function(th, fit) {
  exp(fit$log_prob(fit$unconstrain_variables(list(theta=th)),
                   jacobian=FALSE))
}

df_bin |>
  mutate(pdfstan=sapply(theta, fit_pdf, fit_bin)) |>
  ggplot(aes(x=theta,y=pdfbeta))+
  geom_line()+
  geom_line(aes(y=pdfstan),color=set1[1])+
  geom_vline(xintercept=(opt_bin$draws()[1,'theta']),color="gray")+
  labs(x=TeX(r'($\theta$)'),y='')+
  theme_remove_yaxis+
  annotate("text", x=0.81,y=3.3,label=TeX(r'($p(\theta|y)$)'),color=1,size=5,hjust=1)+
  annotate("text", x=0.89,y=4.3,label='mode',color="gray",size=5,hjust=1)+
  annotate("text", x=0.85,y=.2,label=TeX(r'($q(\theta|y)$=exp(lp__))'),color=set1[1],size=5,hjust=1)

#' 
#' The pdf from Stan is much lower than the true posterior, because it
#' is unnormalized posterior as in general computing the normalization
,#' term is non-trivial. In this case the true posterior has analytic
#' solution for the normalizing constant
#' $$
#' \frac{\Gamma(N+2)}{\Gamma(y+1)\Gamma(N-y+1)}=110
#' $$
#' and we get exact match by multiplying the density returned by
#' Stan by 110.
df_bin |>
  mutate(pdfstan=sapply(theta, fit_pdf, fit_bin)*110) |>
  ggplot(aes(x=theta,y=pdfbeta))+
  geom_line()+
  geom_line(aes(y=pdfstan),color=set1[1])+
  geom_vline(xintercept=(opt_bin$draws()[1,'theta']),color="gray")+
  labs(x=TeX(r'($\theta$)'),y='')+
  theme_remove_yaxis+
  annotate("text", x=0.81,y=3.3,label=TeX(r'($p(\theta|y)$)'),color=1,size=5,hjust=1)+
  annotate("text", x=0.89,y=4.3,label='mode',color="gray",size=5,hjust=1)+
  annotate("text", x=0.79,y=2.9,label=TeX(r'($q(\theta|y)\cdot 110$)'),color=set1[1],size=5,hjust=1)

#' Thus if someone cares about the posterior mode in the constrained
#' space they need `jacobian=FALSE`.
#' 
#' Side note: the normalizing constant is not needed for MCMC and not
#' needed when estimating various expectations using MCMC draws, but
#' is used here for illustration.
#'
#' ## Constraint and parameter transformation
#' 
#' In this example, theta is constrained to be between 0 and 1
#' ```
#'  real<lower=0,upper=1> theta; // probability of success in range (0,1)
#' ```
#' To avoid problems with constraints in optimization and MCMC, Stan
#' switches under the hood to unconstrained parameterization using
#' logit transformation
#' $$
#' \mbox{logit}(\theta) = \log\left(\frac{\theta}{1-\theta}\right)
#' $$.
#'
#' In the above `fit_pdf` function we used Stan's `unconstrain_pars`
#' function to transform theta to logit(theta). In R we can also
#' define logit function as the inverse of the logistic function.
logit <- qlogis

#' We now switch looking at the distributions in the unconstrained space.
df_bin |>
  mutate(pdfstan=sapply(theta, fit_pdf, fit_bin)*110) |>
  ggplot(aes(x=logit(theta),y=pdfbeta))+
  geom_line()+
  geom_line(aes(y=pdfstan),color=set1[1])+
  geom_vline(xintercept=logit(opt_bin$draws()[1,'theta']),color="gray")+
  xlim(-2.9,6.5)+
  labs(x=TeX(r'($\logit(\theta)$)'),y='')+
  theme_remove_yaxis+
  annotate("text", x=logit(0.81),y=3.3,label=TeX(r'($p(\theta|y)$)'),color=1,size=5,hjust=1)+
  annotate("text", x=logit(0.89),y=4.3,label='mode',color="gray",size=5,hjust=1)+
  annotate("text", x=logit(0.79),y=2.9,label=TeX(r'($q(\theta|y)\cdot 110$)'),color=set1[1],size=5,hjust=1)

#' The above plot shows now the function that Stan optimizing
#' optimizes in the unconstrained space, and the MAP is the logit of
#' the MAP in the constrained space. Thus if someone cares about the
#' posterior mode in the constrained, but is doing the optimization in
#' unconstrained space they still need `jacobian=FALSE`.
#'
#' ## Parameter transformation and Jacobian adjustment
#' 
#' That function shown above is not the posterior of `logit(theta)`. As
#' the transformation is non-linear we need to take into account the
#' distortion caused by the transform. The density must be multiplied by a
#' Jacobian adjustment equal to the absolute determinant of the
#' Jacobian of the transform. See more in [Stan User's
#' Guide](https://mc-stan.org/docs/2_26/stan-users-guide/changes-of-variables.html).
#' The Jacobian for lower and upper bounded scalar is given in [Stan
#' Reference
#' Manual](https://mc-stan.org/docs/2_25/reference-manual/logit-transform-jacobian-section.html), and for (0,1)-bounded it is
#' $$
#' \theta (1-\theta).
#' $$
#'
#' Stan can do this transformation for us when we call log_prob with
#' `jacobian=TRUE`
fit_pdf_jacobian <- function(th, fit) {
  exp(fit$log_prob(fit$unconstrain_variables(list(theta=th)),
                   jacobian=TRUE))
}

#' We compare the true adjusted posterior density in logit(theta)
#' space to non-adjusted density function. For visualization purposes
#' we scale the functions to have the same maximum, so they are not
#' normalized distributions.
df_bin <- df_bin |>
  mutate(pdfstan_nonadjusted=sapply(theta, fit_pdf, fit_bin),
         pdfstan_jacobian=sapply(theta, fit_pdf_jacobian, fit_bin))
ggplot(data=df_bin,aes(x=logit(theta),y=pdfstan_nonadjusted/max(pdfstan_nonadjusted)))+
  geom_line(color=set1[1])+
  geom_line(aes(y=pdfstan_jacobian/max(pdfstan_jacobian)),color=set1[2])+
  xlim(-2.9,6.5)+
  labs(x=TeX(r'($\logit(\theta)$)'),y='')+
  theme_remove_yaxis+
  annotate("text", x=1.15,y=0.95,label=TeX('jacobian=TRUE'),color=set1[2],size=5,hjust=1)+
  annotate("text", x=1.15,y=0.9,label=TeX(r'($q(\logit(\theta)|y)  = q(\theta|y)\theta(1-\theta)$)'),color=set1[2],size=5,hjust=1)+
  annotate("text", x=2.9,y=0.95,label="jacobian=FALSE",color=set1[1],size=5,hjust=0)+
  annotate("text", x=2.9,y=0.9,label=TeX(r'($q(\theta|y) \neq q(\logit(\theta)|y)$)'),color=set1[1],size=5,hjust=0)

#' Stan MCMC samples from the blue distribution with
#' `jacobian=TRUE`. The mode of that distribution is different from
#' the mode of `jacobian=FALSE`. In general the mode is not invariant
#' to transformations.
#'
#' ## Wrong normal approximation
#' 
#' Stan optimizing/optimize finds the mode of `jacobian=FALSE` (in
#' some intefaces `jacobian=FALSE`). rstanarm has had an option to do
#' normal approximation at the mode `jacobian=FALSE` in the
#' unconstrained space by computing the Hessian of `jacobian=FALSE`
#' and then sampling independent draws from that normal
#' distribution. We can do the same in CmdStanR with `$laplace()`
#' method and option `jacobian=FALSE`, but this is the wrong thing to
#' do.
lap_bin <- model_bin$laplace(data = data_bin, jacobian=FALSE,
                             seed = SEED, draws=4000, refresh=0)
lap_draws = lap_bin$draws(format = "df")

#' We add the current normal approximation to the plot with dashed line.
lap_draws <- lap_draws |>
  mutate(logp=lp__-max(lp__),
         logg=lp_approx__-max(lp_approx__))
ggplot(data=df_bin,aes(x=logit(theta),y=pdfstan_nonadjusted/max(pdfstan_nonadjusted)))+
  geom_line(color=set1[1])+
  geom_line(aes(y=pdfstan_jacobian/max(pdfstan_jacobian)),color=set1[2])+
  geom_line(data=lap_draws,aes(x=logit(theta),y=exp(logg)),color=set1[1],linetype="dashed")+
  xlim(-2.9,6.5)+
  labs(x=TeX(r'($\logit(\theta)$)'),y='')+
  theme_remove_yaxis+
  annotate("text", x=1.15,y=0.95,label=TeX('jacobian=TRUE'),color=set1[2],size=5,hjust=1)+
  annotate("text", x=1.15,y=0.9,label=TeX(r'($q(\logit(\theta)|y) = q(\theta|y)\theta(1-\theta)$)'),color=set1[2],size=5,hjust=1)+
  annotate("text", x=2.9,y=0.95,label="jacobian=FALSE",color=set1[1],size=5,hjust=0)+
  annotate("text", x=2.9,y=0.9,label=TeX(r'($q(\theta|y) \neq q(\logit(\theta)|y)$)'),color=set1[1],size=5,hjust=0)

#' The problem is that this normal approximation is quite different
#' from the true posterior (with `jacobian=TRUE`).
#'
#' ## Normal approximation
#' 
#' Recently the normal approximation method was implemented in Stan
#' itself with the name `laplace`. This approximation uses by default
#' `jacobian=TRUE`. We can use Laplace approximation with CmdStanR
#' method `$laplace()`, which by default is using he option
#' `jacobian=TRUE`, which is the correct thing to do.
lap_bin2 <- model_bin$laplace(data=data_bin, seed=SEED, draws=4000, refresh=0)
lap_draws2 <- lap_bin2$draws(format = "df")

#' We add the second normal approximation to the plot dashed line.
lap_draws2 <- lap_draws2 |>
  mutate(logp=lp__-max(lp__),
         logg=lp_approx__-max(lp_approx__))
ggplot(data=df_bin,aes(x=logit(theta),y=pdfstan_nonadjusted/max(pdfstan_nonadjusted)))+
  geom_line(color=set1[1])+
  geom_line(aes(y=pdfstan_jacobian/max(pdfstan_jacobian)),color=set1[2])+
  geom_line(data=lap_draws,aes(x=logit(theta),y=exp(logg)),color=set1[1],linetype="dashed")+
  geom_line(data=lap_draws2,aes(x=logit(theta),y=exp(logg)),color=set1[2],linetype="dashed")+
  xlim(-2.9,6.5)+
  labs(x=TeX(r'($\logit(\theta)$)'),y='')+
  theme_remove_yaxis+
  annotate("text", x=1.15,y=0.95,label=TeX('jacobian=TRUE'),color=set1[2],size=5,hjust=1)+
  annotate("text", x=1.15,y=0.9,label=TeX(r'($q(\logit(\theta)|y) = q(\theta|y)\theta(1-\theta)$)'),color=set1[2],size=5,hjust=1)+
  annotate("text", x=2.9,y=0.95,label="jacobian=FALSE",color=set1[1],size=5,hjust=0)+
  annotate("text", x=2.9,y=0.9,label=TeX(r'($q(\theta|y) \neq q(\logit(\theta)|y)$)'),color=set1[1],size=5,hjust=0)

#' ## Transforming draws to the constrained space
#' 
#' The draws from the normal approximation (shown with rug lines in op
#' and bottom) can be easily transformed back to the constrained
#' space, and illustrated with kernel density estimates. Before that
#' we plot the kernel density estimates of the draws in the
#' unconstrained space to show that this etimate is reasonable.
ggplot(data=df_bin,aes(x=logit(theta),y=pdfstan_nonadjusted/max(pdfstan_nonadjusted)))+
  geom_line(color=set1[1])+
  geom_line(aes(y=pdfstan_jacobian/max(pdfstan_jacobian)),color=set1[2])+
  geom_line(data=lap_draws,aes(x=logit(theta),y=exp(logg)),color=set1[1],linetype="dashed")+
  geom_line(data=lap_draws2,aes(x=logit(theta),y=exp(logg)),color=set1[2],linetype="dashed")+
  geom_rug(data=lap_draws[1:400,],aes(x=logit(theta),y=0), color=set1[1], alpha=0.2, sides='t')+
  geom_rug(data=lap_draws2[1:400,],aes(x=logit(theta),y=0), color=set1[2], alpha=0.2, sides='b')+
  geom_density(data=lap_draws,aes(x=logit(theta),after_stat(scaled)),adjust=2,color=set1[1],linetype="dotdash")+
  geom_density(data=lap_draws2,aes(x=logit(theta),after_stat(scaled)),adjust=2,color=set1[2],linetype="dotdash")+
  xlim(-2.9,6.5)+
  labs(x=TeX(r'($\logit(\theta)$)'),y='')+
  theme_remove_yaxis+
  annotate("text", x=1.15,y=0.95,label=TeX('jacobian=TRUE'),color=set1[2],size=5,hjust=1)+
  annotate("text", x=1.15,y=0.9,label=TeX(r'($q(\logit(\theta)|y) = q(\theta|y)\theta(1-\theta)$)'),color=set1[2],size=5,hjust=1)+
  annotate("text", x=2.9,y=0.95,label="jacobian=FALSE",color=set1[1],size=5,hjust=0)+
  annotate("text", x=2.9,y=0.9,label=TeX(r'($q(\theta|y) \neq q(\logit(\theta)|y)$)'),color=set1[1],size=5,hjust=0)

#' When we plot kernel density estimates of the logit transformed
#' draws (draws shown with rug lines in top and bottom) in the
#' constrained space, it's clear which draws approximate better the
#' true posterior (black line)
ggplot(data=df_bin,aes(x=(theta),y=pdfbeta/max(pdfbeta)))+
  geom_line()+
  geom_density(data=lap_draws,aes(x=(theta),after_stat(scaled)),adjust=1,color=set1[1],linetype="dotdash")+
  geom_density(data=lap_draws2,aes(x=(theta),after_stat(scaled)),adjust=1,color=set1[2],linetype="dotdash")+
  geom_rug(data=lap_draws[1:400,],aes(x=(theta),y=0), color=set1[1], alpha=0.2, sides='t')+
  geom_rug(data=lap_draws2[1:400,],aes(x=(theta),y=0), color=set1[2], alpha=0.2, sides='b')+
  labs(x=TeX(r'($\theta$)'),y='')+
  theme_remove_yaxis

#' ## Importance resampling
#' 
#' `$laplace()` method returns also unormalized target log density
#' (`lp__`) and normal approximation log density (`lp_approx__`) for
#' the draws from the normal approximation. These can be used to
#' compute importance ratios `exp(lp__-lp_approax__)` which can be
#' used to do importance resampling. We can do the importance
#' resampling using the posterior package.
lap_draws2is <- lap_draws2 |>
  weight_draws(lap_draws2$lp__-lap_draws2$lp_approx__, log=TRUE) |>
  resample_draws()

#' The kernel density estimate using importance resampled draws is
#' even close to the true distribution.
ggplot(data=df_bin,aes(x=(theta),y=pdfbeta/max(pdfbeta)))+
  geom_line()+
  labs(x=TeX(r'($\theta$)'),y='')+
  theme_remove_yaxis+
  geom_density(data=lap_draws2is,aes(x=(theta),after_stat(scaled)),adjust=1,color=set1[2],linetype="dotdash")+
  geom_rug(data=lap_draws2is[1:400,],aes(x=(theta),y=0), color=set1[2], alpha=0.2, sides='b')

#' ## Discussion
#' 
#' The normal approximation and importance resampling did work quite
#' well in this simple one dimensional case, but in general the normal
#' approximation for importance sampling works well only in quite low
#' dimensional settings or when the posterior in the unconstrained
#' space is very close to normal.
#' 
