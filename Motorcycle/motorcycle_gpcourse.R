#' ---
#' title: "Gaussian process demonstration with Stan"
#' author: "Aki Vehtari"
#' date: "First version 2021-01-28. Last modified `r format(Sys.Date())`."
#' output:
#'   html_document:
#'     theme: readable
#'     toc: true
#'     toc_depth: 2
#'     toc_float: true
#'     code_download: true
#' ---


#' Demonstration of covariance matrix and basis function
#' implementation of Gaussian process model in Stan.
#'
#' The basics of the covariance matrix approach is based on the Chapter
#' 10 of Stan User’s Guide, Version 2.26 by Stan Development Team
#' (2021). https://mc-stan.org/docs/stan-users-guide/
#'
#' The basics of the Hilbert space basis function approximation is
#' based on Riutort-Mayol, Bürkner, Andersen, Solin, and Vehtari
#' (2020). Practical Hilbert space approximate Bayesian Gaussian
#' processes for probabilistic programming. https://arxiv.org/abs/2004.11408
#'
#' ## Motorcycle
#' 
#' Data are measurements of head acceleration in a simulated
#' motorcycle accident, used to test crash helmets.
#'
#' Data are modelled first with normal distribution having Gaussian process
#' prior on mean:
#' $$
#' y \sim \mbox{normal}(f(x), \sigma)\\
#' f \sim GP(0, K_1)\\
#' \sigma \sim \mbox{normal}^{+}(0, 1),
#' $$
#' and then with normal distribution having Gaussian process
#' prior on mean and log standard deviation:
#' $$
#' y \sim \mbox{normal}(f(x), \exp(g(x))\\
#' f \sim GP(0, K_1)\\
#' g \sim GP(0, K_2).
#' $$
#'

#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA, cache=FALSE)

#' #### Load packages
library(cmdstanr) 
library(posterior)
library(tidybayes)
options(pillar.neg = FALSE, pillar.subtle=FALSE, pillar.sigfig=2)
library(tidyr) 
library(dplyr) 
library(ggplot2)
library(ggrepel)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size=16))
set1 <- RColorBrewer::brewer.pal(7, "Set1")
SEED <- 48927 # set random seed for reproducibility

#' ## Motorcycle accident acceleration data
#' 
#' Load data
data(mcycle, package="MASS")
head(mcycle)

#' Plot data
mcycle %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")

#' ## GP model with homoskedastic residual
#'
#' We start with a simpler homoskedastic residual Gaussian process model
#' $$
#' y \sim \mbox{normal}(f(x), \sigma)\\
#' f \sim GP(0, K_1)\\
#' \sigma \sim normal^{+}(0, 1),
#' $$
#' that has analytic marginal likelihood for the covariance function
#' and residual scale parameters.
#' 

#' Model code
file_gpcovf <- "gpcovf.stan"
writeLines(readLines(file_gpcovf))

#' Compile Stan model
model_gpcovf <- cmdstan_model(stan_file = file_gpcovf)

#' Data to be passed to Stan
standata_gpcovf <- list(x=mcycle$times,
                        x2=mcycle$times,
                        y=mcycle$accel,
                        N=length(mcycle$times),
                        N2=length(mcycle$times))

#' Optimize and find MAP estimate
#+ opt_gpcovf, results='hide'
opt_gpcovf <- model_gpcovf$optimize(data=standata_gpcovf,
                                    init=0.1, algorithm='bfgs')

#' Check whether parameters have reasonable values
odraws_gpcovf <- as_draws_rvars(opt_gpcovf$draws())
subset(odraws_gpcovf, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE)

#' Compare the model to the data
mcycle %>%
  mutate(Ef=mean(odraws_gpcovf$f),
         sigma=mean(odraws_gpcovf$sigma)) %>%  
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The model fit given optimized parameters, looks reasonable
#' considering the use of homoskedastic residual model.
#' 
#' Sample using dynamic HMC
#+ fit_gpcovf, results='hide'
fit_gpcovf <- model_gpcovf$sample(data=standata_gpcovf,
                                  iter_warmup=500, iter_sampling=500,
                                  chains=4, parallel_chains=4, refresh=100)

#' Check whether parameters have reasonable values
draws_gpcovf <- as_draws_rvars(fit_gpcovf$draws())
summarise_draws(subset(draws_gpcovf,
                       variable=c('sigma_','lengthscale_','sigma'),
                       regex=TRUE))

#' Compare the model to the data
mcycle %>%
  mutate(Ef=mean(draws_gpcovf$f),
         sigma=mean(draws_gpcovf$sigma)) %>%  
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The model fit given integrated parameters looks similar to the
#' optimized one.
#' 
#' Compare the posterior draws to the optimized parameters
odraws_gpcovf <- as_draws_df(opt_gpcovf$draws())
draws_gpcovf %>%
  as_draws_df() %>%
  ggplot(aes(x=lengthscale_f,y=sigma_f))+
  geom_point(color=set1[2])+
  geom_point(data=odraws_gpcovf,color=set1[1],size=4)+
  annotate("text",x=median(draws_gpcovf$lengthscale_f),
           y=max(draws_gpcovf$sigma_f)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=odraws_gpcovf$lengthscale_f+0.01,
           y=odraws_gpcovf$sigma_f,
           label='Optimized',hjust=0,color=set1[1],size=6)

#' The optimization result is in the middle of the posterior and quite
#' well representative of the low dimensional posterior (in higher
#' dimensions the mean or mode of the posterior is not likely to be
#' representative).
#' 
#' Compare optimized and posterior predictive distributions
odraws_gpcovf <- as_draws_rvars(opt_gpcovf$draws())
mcycle %>%
  mutate(Ef=mean(draws_gpcovf$f),
         sigma=mean(draws_gpcovf$sigma),
         Efo=mean(odraws_gpcovf$f),
         sigmao=mean(odraws_gpcovf$sigma)) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Efo), color=set1[2])+
  geom_line(aes(y=Efo-2*sigmao), color=set1[2],linetype="dashed")+
  geom_line(aes(y=Efo+2*sigmao), color=set1[2],linetype="dashed")

#' The model predictions are very similar, and in general GP
#' covariance function and observation model parameters can be quite
#' safely optimized if there are only a few of them and thus marginal
#' posterior is low dimensional and the number of observations is
#' relatively high.
#' 
#' ### 10% of data
#'
#' To demonstrate that the optimization is not always safe, we use
#' only 10% of the data for model fitting.
#' 
#' Data to be passed to Stan
mcycle_10p <- mcycle[seq(1,133,by=10),]
standata_10p <- list(x=mcycle_10p$times,
                     x2=mcycle$times,
                     y=mcycle_10p$accel,
                     N=length(mcycle_10p$times),
                     N2=length(mcycle$times))

#' Optimize and find MAP estimate
#+ opt_10p, results='hide'
opt_10p <- model_gpcovf$optimize(data=standata_10p, init=0.1,
                                 algorithm='lbfgs')

#' Check whether parameters have reasonable values
odraws_10p <- as_draws_rvars(opt_10p$draws())
subset(odraws_10p, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE)

#' Compare the model to the data
mcycle_10p %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(data=mcycle,aes(x=times,y=mean(odraws_10p$f)), color=set1[1])+
  geom_line(data=mcycle,aes(x=times,y=mean(odraws_10p$f-2*odraws_10p$sigma)), color=set1[1],
            linetype="dashed")+
  geom_line(data=mcycle,aes(x=times,y=mean(odraws_10p$f+2*odraws_10p$sigma)), color=set1[1],
            linetype="dashed")

#' The model fit is clearly over-fitted and over-confident.
#' 
#' Sample using dynamic HMC
#+ fit_10p, results='hide'
fit_10p <- model_gpcovf$sample(data=standata_10p,
                               iter_warmup=1000, iter_sampling=1000,
                               adapt_delta=0.95,
                               chains=4, parallel_chains=4, refresh=100)

#' Check whether parameters have reasonable values
draws_10p <- as_draws_rvars(fit_10p$draws())
summarise_draws(subset(draws_10p, variable=c('sigma_','lengthscale_','sigma'),
                       regex=TRUE))

#' Compare the model to the data
mcycle_10p %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(data=mcycle,aes(x=times,y=mean(draws_10p$f)), color=set1[1])+
  geom_line(data=mcycle,aes(x=times,y=mean(draws_10p$f-2*draws_10p$sigma)), color=set1[1],
            linetype="dashed")+
  geom_line(data=mcycle,aes(x=times,y=mean(draws_10p$f+2*draws_10p$sigma)), color=set1[1],
            linetype="dashed")

#' The posterior predictive distribution is much more conservative and
#' shows the uncertainty due to having only a small number of
#' observations.
#' 
#' Compare the posterior draws to the optimized parameters
odraws_10p <- as_draws_df(opt_10p$draws())
draws_10p %>%
  thin_draws(thin=5) %>%
  as_draws_df() %>%
  ggplot(aes(x=sigma,y=sigma_f))+
  geom_point(color=set1[2])+
  geom_point(data=odraws_10p,color=set1[1],size=4)+
  annotate("text",x=median(draws_10p$sigma),
           y=max(draws_10p$sigma_f)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=odraws_10p$sigma+0.01,
           y=odraws_10p$sigma_f,
           label='Optimized',hjust=0,color=set1[1],size=6)

#' The optimization result is in the edge of the posterior close to
#' zero residual scale. While there are posterior draws close to zero,
#' integrating over the wide posterior takes into account the
#' uncertainty and the predictions thus are more uncertain, too.
#' 
#' Compare optimized and posterior predictive distributions
odraws_10p <- as_draws_rvars(opt_10p$draws())
mcycle %>%
  mutate(Ef=mean(draws_10p$f),
         sigma=mean(draws_10p$sigma),
         Efo=mean(odraws_10p$f),
         sigmao=mean(odraws_10p$sigma)) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Efo), color=set1[2])+
  geom_line(aes(y=Efo-2*sigmao), color=set1[2],linetype="dashed")+
  geom_line(aes(y=Efo+2*sigmao), color=set1[2],linetype="dashed")

#' The figure shows the model prediction given 10% of data, but also
#' the full data as test data. The optimized model is over-fitted and
#' overconfident. Even if the homoskedastic residual is wrong here,
#' the posterior predictive interval covers most of the observation
#' (and in case of good calibration should not cover them all).
#' 

#' ## GP with covariance matrices
#'
#' We next make a model with heteroskedastic residual model using
#' Gaussian process prior also for the logarithm of the residual
#' scale:
#' $$
#' y \sim \mbox{normal}(f(x), \exp(g(x))\\
#' f \sim GP(0, K_1)\\
#' g \sim GP(0, K_2).
#' $$
#'
#' Now there is no analytical solution as GP prior through the
#' exponential function is not a conjugate prior. In this case we
#' present the latent values of f and g explicitly and sample from the
#' joint posterior of the covariance function parameters, and the
#' latent values. It would be possible also to use Laplace,
#' variational inference, or expectation propagation to integrate over
#' the latent values, but that is another story.
#' 
#' Model code
file_gpcovfg <- "gpcovfg.stan"
writeLines(readLines(file_gpcovfg))

#' Compile Stan model
model_gpcovfg <- cmdstan_model(stan_file = file_gpcovfg)

#' Data to be passed to Stan
standata_gpcovfg <- list(x=mcycle$times,
                         y=mcycle$accel,
                         N=length(mcycle$times))

#' Optimize and find MAP estimate
#+ opt_gpcovfg, results='hide'
opt_gpcovfg <- model_gpcovfg$optimize(data=standata_gpcovfg,
                                      init=0.1, algorithm='bfgs')

#' Check whether parameters have reasonable values
odraws_gpcovfg <- as_draws_rvars(opt_gpcovfg$draws())
subset(odraws_gpcovfg, variable=c('sigma_','lengthscale_'), regex=TRUE)

#' Compare the model to the data
mcycle %>%
  mutate(Ef = mean(odraws_gpcovfg$f),
         sigma = mean(odraws_gpcovfg$sigma)) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The optimization overfits, as we are now optimizing the joint
#' posterior of 2 covariance function parameters and 2 x 133 latent
#' values.
#' 
#' Sample using dynamic HMC
#+ fit_gpcovfg, results='hide'
fit_gpcovfg <- model_gpcovfg$sample(data=standata_gpcovfg,
                                    iter_warmup=100, iter_sampling=200,
                                    chains=4, parallel_chains=4, refresh=20)

#' Check whether parameters have reasonable values
draws_gpcovfg <- as_draws_rvars(fit_gpcovfg$draws())
summarise_draws(subset(draws_gpcovfg, variable=c('sigma_','lengthscale_'),
                       regex=TRUE))

#' Compare the model to the data
mcycle %>%
  mutate(Ef = mean(draws_gpcovfg$f),
         sigma = mean(draws_gpcovfg$sigma)) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The MCMC integration works well and the model fit looks good.
#' 
#' Plot posterior draws and posterior mean of the mean function
draws_gpcovfg %>%
  spread_rvars(f[i]) %>%
  unnest_rvars() %>%
  mutate(time=mcycle$times[i]) %>%
  ggplot(aes(x=time, y=f, group = .draw)) +
  geom_line(color=set1[2], alpha = 0.1) +
  geom_point(data=mcycle, mapping=aes(x=times,y=accel), inherit.aes=FALSE) +
  geom_line(data=mcycle, mapping=aes(x=times,y=mean(draws_gpcovfg$f)),
            inherit.aes=FALSE, color=set1[1], size=1)+
  labs(x="Time (ms)", y="Acceleration (g)")

#' We can also plot the posterior draws of the latent functions, which
#' is a good reminder that individual draws are more wiggly than the
#' average of the draws, and thus show better also the uncertainty,
#' for example, in the edge of the data.
#' 
#' Compare the posterior draws to the optimized parameters
odraws_gpcovfg <- as_draws_df(opt_gpcovfg$draws())
draws_gpcovfg %>%
  as_draws_df() %>%
  ggplot(aes(x=lengthscale_f,y=sigma_f))+
  geom_point(color=set1[2])+
  geom_point(data=odraws_gpcovfg,color=set1[1],size=4)+
  annotate("text",x=median(draws_gpcovfg$lengthscale_f),
           y=max(draws_gpcovfg$sigma_f)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=odraws_gpcovfg$lengthscale_f+0.01,
           y=odraws_gpcovfg$sigma_f,
           label='Optimized',hjust=0,color=set1[1],size=6)

#' Optimization result is far from being representative of the
#' posterior.
#' 

#' ## GP model with Hilbert basis functions
#'
#' The covariance matrix approach requires computation of Cholesky of
#' the covariance matrix which has time cost O(n^3) and this is needs
#' to be done every time the parameters change, which in case of MCMC
#' can be quite many times and thus the computation time can be
#' significant when n grows. One way to speed up the computation in
#' low dimensional covariate case is to use basis function
#' approximation which changes the GP to a linear model. Here we use
#' Hilbert space basis functions. With infinite number of basis
#' functions, the approach is exact, but sufficient accuracy and
#' significant saving in the computation speed is often achieveved
#' with a relatively small number of basis functions.
#' 
#' Code for illustrating the basis functions
#' Model code
filebf0 <- "gpbf0.stan"
writeLines(readLines(filebf0))
#' The model code includes Hilbert space basis function helpers
writeLines(readLines("gpbasisfun_functions.stan"))

#' Compile basis function generation code
modelbf0 <- cmdstan_model(stan_file = filebf0, include_paths = ".")

#' Data to be passed to Stan
standatabf0 <- list(x=seq(0,1,length.out=100),
                    N=100,
                    c_f=1.5, # factor c of basis functions for GP for f1
                    M_f=40,  # number of basis functions for GP for f1
                    sigma_f=1,
                    lengthscale_f=1) 
#' Generate basis functions
fixbf0 <- modelbf0$sample(data=standatabf0, fixed_param=TRUE,
                          iter=1, iter_sampling=1)
#' There is certainly easier way to do this, but this is what I came up quickly
q<-subset(fixbf0$draws(), variable="PHI_f") %>%
  as_draws_matrix() %>%
  as.numeric()%>%
  matrix(nrow=100,ncol=40)%>%
  as.data.frame()
id <- rownames(q)
q <- cbind(x=as.numeric(id), q)
q <- q %>%
  pivot_longer(!x,
               names_to="ind",
               names_transform = list(ind = readr::parse_number),
               values_to="f")%>%
  mutate(x=x/100)

#' Plot the first 6 basis functions. These are just sine and cosine
#' functions with different frequencies and truncated to a pre-defined
#' box.
q %>%
  filter(ind<=6) %>%
  ggplot(aes(x=x, y=f, group=ind, color=factor(ind))) +
  geom_line()+
  geom_text_repel(data=filter(q, ind<=6 & x==0.01),aes(x=-0.01,y=f,label=ind),
                  direction="y")+
  geom_text_repel(data=filter(q, ind<=6 & x==1),aes(x=1.02,y=f,label=ind),
                  direction="y")+
  theme(legend.position="none")
ggsave('gp_basis_functions.pdf',width=4,height=3)

#' The first 8 spectral densities with sigma_f=1 and
#' lengthscale_f=1. These spectral densities give a prior weight for
#' each basis function. Bigger weights on the smoother basis functions
#' thus imply a prior on function space favoring smoother functions.
spd <- as.matrix(fixbf0$draws(variable='diagSPD_f'))
round(spd[1:8],2)

#' Plot 4 random draws from the prior on function space with sigma_f=1
#' and lengthscale_f=1. The basis function approximation is just a
#' linear model, with the basis functions weighted by the spectral
#' densities depending on the sigma_f and lengthscale_f, and the prior
#' for the linear model coefficients is simply independent
#' normal(0,1).
set.seed(365)
qr <- bind_rows(lapply(1:4, function(i) {
  q %>%
    mutate(r=rep(rnorm(40),times=100),fr=f*r*spd[ind]) %>%
    group_by(x) %>%
    summarise(f=sum(fr)) %>%
    mutate(ind=i) }))
qr %>%
  ggplot(aes(x=x, y=f, group=ind, color=factor(ind))) +
  geom_line()+
  geom_text_repel(data=filter(qr, x==0.01),aes(x=-0.01,y=f,label=ind),
                  direction="y")+
  geom_text_repel(data=filter(qr, x==1),aes(x=1.02,y=f,label=ind),
                  direction="y")+
  theme(legend.position="none")
ggsave('gp_prior_draws_l1.pdf',width=4,height=3)

#' Let's do the same with lengthscale_f=0.3
standatabf0 <- list(x=seq(0,1,length.out=100),
                    N=100,
                    c_f=1.5, # factor c of basis functions for GP for f1
                    M_f=40,  # number of basis functions for GP for f1
                    sigma_f=1,
                    lengthscale_f=0.3) 
fixbf0 <- modelbf0$sample(data=standatabf0, fixed_param=TRUE,
                          iter=1, iter_sampling=1)
#' The basis functions are exactly the same, and only the spectral
#' densities have changed. Now the weight doesn't drop as fast for
#' the more wiggly basis functions.
spd <- as.matrix(fixbf0$draws(variable='diagSPD_f'))
round(spd[1:8],2)

#' Plot 4 random draws from the prior on function space with sigma_f=1
#' and lengthscale_f=0.3. The random functions from the prior are now
#' more wiggly. The same random number generator seed was used so that
#' you can compare this plot to the above one. Above the prior draw
#' number 2 looks like a decreasing slope. Here the prior draw number
#' 2 still has downward trend, but is more wiggly. The same random
#' draw from the coefficient space produces a wigglier function as the
#' spectral densities go down slower for the more wiggly basis
#' functions.
set.seed(365)
qr <- bind_rows(lapply(1:4, function(i) {
  q %>%
    mutate(r=rep(rnorm(40),times=100),fr=f*r*spd[ind]) %>%
    group_by(x) %>%
    summarise(f=sum(fr)) %>%
    mutate(ind=i) }))
qr %>%
  ggplot(aes(x=x, y=f, group=ind, color=factor(ind))) +
  geom_line()+
  geom_text_repel(data=filter(qr, x==0.01),aes(x=-0.01,y=f,label=ind),
                  direction="y")+
  geom_text_repel(data=filter(qr, x==1),aes(x=1.02,y=f,label=ind),
                  direction="y")+
  theme(legend.position="none")
ggsave('gp_prior_draws_l03.pdf',width=4,height=3)

#' And now the actual model using GP basis functions for f and g
#' 
#' Model code
file_gpbffg <- "gpbffg.stan"
writeLines(readLines(file_gpbffg))

#' Compile Stan model
model_gpbffg <- cmdstan_model(stan_file = file_gpbffg, include_paths = ".")

#' Data to be passed to Stan
standata_gpbffg <- list(x=mcycle$times,
                        y=mcycle$accel,
                        N=length(mcycle$times),
                        c_f=1.5, # factor c of basis functions for GP for f1
                        M_f=40,  # number of basis functions for GP for f1
                        c_g=1.5, # factor c of basis functions for GP for g3
                        M_g=40)  # number of basis functions for GP for g3

#' Optimize and find MAP estimate
#+ opt_gpbffg, results='hide'
opt_gpbffg <- model_gpbffg$optimize(data=standata_gpbffg,
                                    init=0.1, algorithm='bfgs')

#' Check whether parameters have reasonable values
odraws_gpbffg <- as_draws_rvars(opt_gpbffg$draws())
subset(odraws_gpbffg, variable=c('intercept','sigma_','lengthscale_'),
       regex=TRUE)

#' Compare the model to the data
mcycle %>%
  mutate(Ef=mean(odraws_gpbffg$f),
         sigma=mean(odraws_gpbffg$sigma)) %>%  
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")


#' The optimization overfits, as we are now optimizing the joint
#' posterior of 2 covariance function parameters and 2 x 40 basis
#' function co-efficients.
#' 
#' Sample using dynamic HMC
#+ fit_gpbffg, results='hide'
fit_gpbffg <- model_gpbffg$sample(data=standata_gpbffg,
                                  iter_warmup=500, iter_sampling=500, refresh=100,
                                  chains=4, parallel_chains=4, adapt_delta=0.9)

#' Check whether parameters have reasonable values
draws_gpbffg <- as_draws_rvars(fit_gpbffg$draws())
summarise_draws(subset(draws_gpbffg,
                       variable=c('intercept','sigma_','lengthscale_'),
                       regex=TRUE))

#' Compare the model to the data
mcycle %>%
  mutate(Ef=mean(draws_gpbffg$f),
         sigma=mean(draws_gpbffg$sigma)) %>%  
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The MCMC integration works well and the model fit looks good.
#' 
#' Plot posterior draws and posterior mean of the mean function
draws_gpbffg %>%
  thin_draws(thin=5) %>%
  spread_rvars(f[i]) %>%
  unnest_rvars() %>%
  mutate(time=mcycle$times[i]) %>%
  ggplot(aes(x=time, y=f, group = .draw)) +
  geom_line(color=set1[2], alpha = 0.1) +
  geom_point(data=mcycle, mapping=aes(x=times,y=accel), inherit.aes=FALSE)+
  geom_line(data=mcycle, mapping=aes(x=times,y=mean(draws_gpbffg$f)),
            inherit.aes=FALSE, color=set1[1], size=1)+
  labs(x="Time (ms)", y="Acceleration (g)")

#' We can also plot the posterior draws of the latent functions, which
#' is a good reminder that individual draws are more wiggly than the
#' average of the draws, and thus show better also the uncertainty,
#' for example, in the edge of the data.
#' 
#' Compare the posterior draws to the optimized parameters
odraws_gpbffg <- as_draws_df(opt_gpbffg$draws())
draws_gpbffg %>%
  thin_draws(thin=5) %>%
  as_draws_df() %>%
  ggplot(aes(x=lengthscale_f,y=sigma_f))+
  geom_point(color=set1[2])+
  geom_point(data=odraws_gpbffg,color=set1[1],size=4)+
  annotate("text",x=median(draws_gpbffg$lengthscale_f),
           y=max(draws_gpbffg$sigma_f)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=odraws_gpbffg$lengthscale_f+0.01,
           y=odraws_gpbffg$sigma_f,
           label='Optimized',hjust=0,color=set1[1],size=6)

#' Optimization result is far from being representative of the
#' posterior.
#' 

#' ## GP model with Hilbert basis functions and Matern covariance
#' 
#' Exponentiated quadratic is sometimes considered to be too smooth as
#' all the derivatives are continuos. For comparison we use Matern
#' covariance. The Hilbert space basis functions are the same and only
#' the spectral density values change (that is different basis
#' functions have a different weighting).
#' 
#' Model code
file_gpbffg2 <- "gpbffg_matern.stan"
writeLines(readLines(file_gpbffg2))

#' Compile Stan model
model_gpbffg2 <- cmdstan_model(stan_file = file_gpbffg2, include_paths = ".")

#' Data to be passed to Stan
standata_gpbffg2 <- list(x=mcycle$times,
                        y=mcycle$accel,
                        N=length(mcycle$times),
                        c_f=1.5, # factor c of basis functions for GP for f1
                        M_f=40,  # number of basis functions for GP for f1
                        c_g=1.5, # factor c of basis functions for GP for g3
                        M_g=40)  # number of basis functions for GP for g3

#' Sample using dynamic HMC
#+ fit_gpbffg2, results='hide'
fit_gpbffg2 <- model_gpbffg2$sample(data=standata_gpbffg2,
                                  iter_warmup=500, iter_sampling=500,
                                  chains=4, parallel_chains=4, adapt_delta=0.9)

#' Check whether parameters have reasonable values
draws_gpbffg2 <- as_draws_rvars(fit_gpbffg2$draws())
summarise_draws(subset(draws_gpbffg2,
                       variable=c('intercept','sigma_','lengthscale_'),
                       regex=TRUE))

#' Compare the model to the data
mcycle %>%
  mutate(Ef=mean(draws_gpbffg2$f),
         sigma=mean(draws_gpbffg2$sigma)) %>%  
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The MCMC integration works well and the model fit looks good.
#' 
#' Plot posterior draws and posterior mean of the mean function
draws_gpbffg2 %>%
  thin_draws(thin=5) %>%
  spread_rvars(f[i]) %>%
  unnest_rvars() %>%
  mutate(time=mcycle$times[i]) %>%
  ggplot(aes(x=time, y=f, group = .draw)) +
  geom_line(color=set1[2], alpha = 0.1) +
  geom_point(data=mcycle, mapping=aes(x=times,y=accel), inherit.aes=FALSE)+
  geom_line(data=mcycle, mapping=aes(x=times,y=mean(draws_gpbffg2$f)),
            inherit.aes=FALSE, color=set1[1], size=1)+
  labs(x="Time (ms)", y="Acceleration (g)")

#' We see that when using Matern covariance instead of the
#' exponentiated quadratic, the model fit is more wigggly.
#' 
