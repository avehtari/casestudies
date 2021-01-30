#' ---
#' title: "Visual illustration of Jacobian of parameter transformation"
#' author: "Aki Vehtari"
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
#' This case study provides visual illustration of Jacobian adjustment
#' for parameter transformations and effect of `adjust_transform`
#' option in Stan `log_prob` and `log_prob_grad` functions that are
#' accesible in some interfaces. This notebook intentionally doesn't
#' go in the mathematical details of measure and probability theory.
#'

#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA, cache=FALSE)

#' #### Load packages
library(tidyr) 
library(dplyr) 
library(rstan) 
rstan_options(auto_write = TRUE)
options(mc.cores = 1)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size=16))
set1 <- RColorBrewer::brewer.pal(7, "Set1")
library(posterior)
SEED <- 48927 # set random seed for reproducibility

#' ## Example model and posterior

#' For the illustration Binomial model is used with observed data
#' $N=10, y=1$.
data_bin <- list(N = 10, y = 9)

#' As Beta(1,1) prior is used the posterior is Beta(9+1,1+1), but for
#' illustration we also use Stan to find the mode of the posterior,
#' sample from the posterior, and compare different posterior density
#' values that we can ask Stan to compute.
code_binom <- "binom.stan"
model_bin <- stan_model(file = code_binom)

#' Default MCMC sampling (as this is an easy posterior we skip showing
#' the results for the convergence diagnostics).
fit_bin<-sampling(model_bin, data = data_bin, seed = SEED, refresh=0)

#' Default optimization finds the maximum of the posterior.
opt_bin<-optimizing(model_bin, data = data_bin, seed = SEED)

#' The following plot shows the exact posterior (black) and grey
#' vertical line shows the MAP, that is, posterior mode in the
#' constrained space. Stan optimizing finds correctly the mode.
df<-data.frame(theta=plogis(seq(-4,6,length.out=100)))%>%
  mutate(pdfbeta=dbeta(theta,9+1,1+1))
ggplot(data=df,aes(x=theta,y=pdfbeta))+
  geom_line()+
  geom_vline(xintercept=(opt_bin$par),color="gray")+
  labs(x='theta',y='p')

#' ## Posterior log_prob
#' 
#' The Stan model object `fit_bin` provides also access to log_prob
#' and log_prob_grad functions. The documentation of log_prob says
#'
#'     Using model's log_prob and grad_log_prob take values from the
#'     unconstrained space of model parameters and (by default) return
#'     values in the same space.
#'
#' And one of the options say
#' 
#' adjust_transform: Logical to indicate whether to adjust the log density
#'          since Stan transforms parameters to unconstrained space if it
#'          is in constrained space. Set to FALSE to make the function
#'          return the same values as Stan's ‘lp__’ output.
#'
#' We can compute the exact posterior density values in grid using
#' Stan and log_prob with adjust_transform=FALSE (strangely it returns
#' 0, unless gradient=TRUE).
fit_pdf <-function(th, fit) { exp(log_prob(fit,
                                           unconstrain_pars(fit,
                                                            list(theta=th)),
                                           adjust_transform=FALSE,
                                           gradient=TRUE)) }

df %>%
  mutate(pdfstan=sapply(theta, fit_pdf, fit_bin)) %>%
  ggplot(aes(x=theta,y=pdfbeta))+
  geom_line()+
  geom_line(aes(y=pdfstan),color=set1[1])+
  geom_vline(xintercept=(opt_bin$par),color="gray")+
  labs(x='theta',y='p')
#' 
#' The pdf from Stan is much lower than the true posterior, because it
#' is unnormalized posterior as in general computing the normalization
#' term is non-trivial. In this case the true posterior has analytic
#' solution for the normalizing constant
#' $$
#' \frac{\Gamma(N+2)}{\Gamma(y+1)\Gamma(N-y+1)}=110
#' $$
#' and we get exact match by multiplying the density returned by
#' Stan by 110.
df %>%
  mutate(pdfstan=sapply(theta, fit_pdf, fit_bin)*110) %>%
  ggplot(aes(x=theta,y=pdfbeta))+
  geom_line()+
  geom_line(aes(y=pdfstan),color=set1[1])+
  geom_vline(xintercept=(opt_bin$par),color="gray")+
  labs(x='theta',y='p')

#' Thus if someone cares about the posterior mode in the constrained
#' space they need adjust_transform = FALSE.
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
#' In the above `fit_bin` function we used Stan's `unconstrain_pars`
#' function to transform theta to logit(theta). In R we can also
#' define logit function as the inverse of the logistic function.
logit<-qlogis

#' We now switch looking at the distributions in the unconstrained space.
df %>%
  mutate(pdfstan=sapply(theta, fit_pdf, fit_bin)*110) %>%
  ggplot(aes(x=logit(theta),y=pdfbeta))+
  geom_line()+
  geom_line(aes(y=pdfstan),color=set1[1])+
  geom_vline(xintercept=logit(opt_bin$par),color="gray")+
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  labs(x='logit(theta)',y='p')
#' The above plot shows now the function that Stan optimizing
#' optimizes in the unconstrained space, and the MAP is the logit of
#' the MAP in the constrained space. Thus if someone cares about the
#' posterior mode in the constrained, but is doing the optimization in
#' unconstrained space they still need adjust_transform = FALSE.
#'
#' ## Parameter transformation and Jacobian adjustment
#' 
#' That function shown above is not the posterior of `logit(theta)`. As
#' the transformation is non-linear we need to take into account the
#' distortion caused by the transform. The density must be scaled by a
#' Jacobian adjustment equal to the absolute determinant of the
#' Jacobian of the transform. See more in [Stan User's
#' Guide](https://mc-stan.org/docs/2_26/stan-users-guide/changes-of-variables.html)
#' and see the Jacobian for lower and upper bounded scalar in [Stan
#' Reference
#' Manual](https://mc-stan.org/docs/2_25/reference-manual/logit-transform-jacobian-section.html)
#'
#' Stan can do this transformation for us when we call log_prob with
#' `adjust_transform=TRUE`
fit_pdf_adjust <-function(th, fit) { exp(log_prob(fit,
                                                  unconstrain_pars(fit,
                                                            list(theta=th)),
                                           adjust_transform=TRUE,
                                           gradient=TRUE)) }

#' We compare the true adjusted posterior density in logit(theta)
#' space to non-adjusted density function. For visualization purposes
#' we scale the functions to have the same maximum, so they are not
#' normalized distributions.
df <-df %>%
  mutate(pdfstan_nonadjusted=sapply(theta, fit_pdf, fit_bin),
         pdfstan_adjusted=sapply(theta, fit_pdf_adjust, fit_bin))
ggplot(data=df,aes(x=logit(theta),y=pdfstan_nonadjusted/max(pdfstan_nonadjusted)))+
  geom_line(color=set1[1])+
  geom_line(aes(y=pdfstan_adjusted/max(pdfstan_adjusted)),color=set1[2])+
  geom_vline(xintercept=logit(opt_bin$par),color="gray")+
  xlim(-2,6)+
  labs(x='logit(theta)',y='')+
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  annotate("text", x=0.8,y=0.7,label="adjust_transform\n=TRUE",color=set1[2],size=5,hjust=1)+
  annotate("text", x=3.5,y=0.7,label="adjust_transform\n=FALSE",color=set1[1],size=5,hjust=0)

#' Stan MCMC samples from the blue distribution with
#' adjust_transform=TRUE. The mode of that distribution is different
#' from the mode of adjust_transform=FALSE. In general mode is not
#' invariant to transformations.
#'
#' ## Normal approximation 1
#' 
#' Currently when using optimizing rstanarm finds the mode of
#' adjust_transform=FALSE and makes normal approximation at that point
#' in the unconstrained space by computing the Hessian of
#' adjust_transform=FALSE and then samples independent draws from that
#' normal distribution. We can do the same in rstan. We also add `
#' importance_resampling=TRUE` to get some densities computed.
opt_bin<-optimizing(model_bin, data = data_bin, seed = SEED,
                    draws=4000, importance_resampling=TRUE)

#' We add the current normal approximation to the plot with dashed line.
dfo<-data.frame(theta=opt_bin$theta_tilde,
                logp=opt_bin$log_p-max(opt_bin$log_p),
                logg=opt_bin$log_g-max(opt_bin$log_g))
ggplot(data=df,aes(x=logit(theta),y=pdfstan_nonadjusted/max(pdfstan_nonadjusted)))+
  geom_line(color=set1[1])+
  geom_line(aes(y=pdfstan_adjusted/max(pdfstan_adjusted)),color=set1[2])+
  geom_vline(xintercept=logit(opt_bin$par),color="gray")+
  xlim(-2,6)+
  labs(x='logit(theta)',y='')+
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  annotate("text", x=0.8,y=0.7,label="adjust_transform\n=TRUE",color=set1[2],size=5,hjust=1)+
  annotate("text", x=3.5,y=0.7,label="adjust_transform\n=FALSE",color=set1[1],size=5,hjust=0)+
  geom_line(data=dfo,aes(x=logit(theta),y=exp(logg)),color=set1[1],linetype="dashed")

#' The problem is that this normal approximation is quite different
#' from the true posterior (with adjust_transform=TRUE).
#'
#' ## Normal approximation 2
#' 
#' The following optimizing uses a modified code that instead finds
#' the maximum of the density with adjust_transform=TRUE and makes the
#' normal approximation there.
opt_bin2<-optimizing(model_bin, data=data_bin, seed=SEED,
                     draws=4000, importance_resampling=2)

#' We add the second normal approximation to the plot dashed line.
dfo2<-data.frame(theta=opt_bin2$theta_tilde,
                logp=opt_bin2$log_p-max(opt_bin2$log_p),
                logg=opt_bin2$log_g-max(opt_bin2$log_g))
ggplot(data=df,aes(x=logit(theta),y=pdfstan_nonadjusted/max(pdfstan_nonadjusted)))+
  geom_line(color=set1[1])+
  geom_line(aes(y=pdfstan_adjusted/max(pdfstan_adjusted)),color=set1[2])+
  geom_vline(xintercept=logit(opt_bin$par),color="gray")+
  xlim(-2,6)+
  labs(x='logit(theta)',y='')+
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  annotate("text", x=0.8,y=0.7,label="adjust_transform\n=TRUE",color=set1[2],size=5,hjust=1)+
  annotate("text", x=3.5,y=0.7,label="adjust_transform\n=FALSE",color=set1[1],size=5,hjust=0)+
  geom_line(data=dfo,aes(x=logit(theta),y=exp(logg)),color=set1[1],linetype="dashed")+
  geom_line(data=dfo2,aes(x=logit(theta),y=exp(logg)),color=set1[2],linetype="dashed")

#' ## Transforming draws to the constrained space
#' 
#' The draws from the normal approximation can be easily transformed
#' back to the constrained space, but before that we plot kernel
#' density estimates of the draws in the unconstrained space.
ggplot(data=df,aes(x=logit(theta),y=pdfstan_nonadjusted/max(pdfstan_nonadjusted)))+
  geom_line(color=set1[1])+
  geom_line(aes(y=pdfstan_adjusted/max(pdfstan_adjusted)),color=set1[2])+
  geom_vline(xintercept=logit(opt_bin$par),color="gray")+
  xlim(-2,6)+
  labs(x='logit(theta)',y='')+
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  annotate("text", x=0.8,y=0.7,label="adjust_transform\n=TRUE",color=set1[2],size=5,hjust=1)+
  annotate("text", x=3.5,y=0.7,label="adjust_transform\n=FALSE",color=set1[1],size=5,hjust=0)+
  geom_line(data=dfo,aes(x=logit(theta),y=exp(logg)),color=set1[1],linetype="dashed")+
  geom_line(data=dfo2,aes(x=logit(theta),y=exp(logg)),color=set1[2],linetype="dashed")+
  geom_density(data=dfo,aes(x=logit(theta),after_stat(scaled)),adjust=2,color=set1[1],linetype="dotdash")+
  geom_density(data=dfo2,aes(x=logit(theta),after_stat(scaled)),adjust=2,color=set1[2],linetype="dotdash")

#' When we plot kernel density estimates of the logistic transformed
#' draws in the constrained space, it's clear which draws approximate
#' better the true posterior (black line)
ggplot(data=df,aes(x=(theta),y=pdfbeta/max(pdfbeta)))+
  geom_line()+
  geom_vline(xintercept=(opt_bin$par),color="gray")+
  labs(x='theta',y='')+
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  geom_density(data=dfo,aes(x=(theta),after_stat(scaled)),adjust=1,color=set1[1],linetype="dotdash")+
  geom_density(data=dfo2,aes(x=(theta),after_stat(scaled)),adjust=1,color=set1[2],linetype="dotdash")

#' ## Importance resampling
#' 
#' When using option importance_resampling=TRUE (or 2) rstan computes
#' also for each draw `log_p` which is log posterior density (with
#' adjust_transform=TRUE) and `log_g` which is log normal
#' density. These can be used to compute importance ratios
#' exp(log_p-log_g) which can be used to do importance
#' resampling. rstanarm does the importance resampling when
#' importance_resampling=TRUE, while rstan has this option only to
#' support rstanarm. We can do the importance resampling using the
#' posterior package.
dfo2is<-as_draws_df(opt_bin2$theta_tilde)%>%
  weight_draws(opt_bin2$log_p-opt_bin2$log_g,log=TRUE)%>%
  resample_draws()

#' The kernel density estimate using importance resampled draws is
#' even close to the true distribution.
ggplot(data=df,aes(x=(theta),y=pdfbeta/max(pdfbeta)))+
  geom_line()+
  geom_vline(xintercept=(opt_bin$par),color="gray")+
  labs(x='theta',y='')+
  theme(axis.text.y = element_blank(),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank())+
  geom_density(data=dfo2is,aes(x=(theta),after_stat(scaled)),adjust=1,color=set1[2],linetype="dotdash")

#' ## Discussion
#' 
#' The normal approximation and importance resampling did work quite
#' well in this simple one dimensional case, but in general the normal
#' approximation works well as an importance sampling proposal only in
#' quite low dimensional settings or when the posterior in the
#' unconstrained space is close to normal. Thus this feature in Stan
#' hasn't been much advertised.
#' 

