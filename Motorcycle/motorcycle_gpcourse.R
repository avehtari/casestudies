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
options(pillar.neg = FALSE, pillar.subtle=FALSE, pillar.sigfig=2)
library(tidyr) 
library(dplyr) 
library(ggplot2)
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
odraws_gpcovf <- opt_gpcovf$draws()
subset(odraws_gpcovf, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE)

#' Compare the model to the data
Ef <- as.numeric(subset(odraws_gpcovf, variable='f'))
sigma <- as.numeric(subset(odraws_gpcovf, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
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
                                  chains=4, parallel_chains=2, refresh=100)

#' Check whether parameters have reasonable values
draws_gpcovf <- fit_gpcovf$draws()
summarise_draws(subset(draws_gpcovf,
                       variable=c('sigma_','lengthscale_','sigma'),
                       regex=TRUE))

#' Compare the model to the data
draws_gpcovf_m <- as_draws_matrix(draws_gpcovf)
Ef <- colMeans(subset(draws_gpcovf_m, variable='f'))
sigma <- colMeans(subset(draws_gpcovf_m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
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
optim<-as_draws_df(subset(odraws_gpcovf,
                          variable=c('sigma_f','lengthscale_f')))
drawsdf<-as_draws_df(subset(draws_gpcovf,
                            variable=c('sigma_f','lengthscale_f')))%>%
  thin_draws(thin=5)
drawsdf%>%
  ggplot(aes(x=lengthscale_f,y=sigma_f))+
  geom_point(color=set1[2])+
  geom_point(data=optim,color=set1[1],size=4)+
  annotate("text",x=median(drawsdf$lengthscale_f),y=max(drawsdf$sigma_f)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=optim$lengthscale_f+0.01,y=optim$sigma_f,
           label='Optimized',hjust=0,color=set1[1],size=6)

#' The optimization result is in the middle of the posterior and quite
#' well representative of the low dimensional posterior (in higher
#' dimensions the mean or mode of the posterior is not likely to be
#' representative).
#' 
#' Compare optimized and posterior predictive distributions
Efo <- as.numeric(subset(odraws_gpcovf, variable='f'))
sigmao <- as.numeric(subset(odraws_gpcovf, variable='sigma'))
draws_gpcovf_m <- as_draws_matrix(draws_gpcovf)
Ef <- colMeans(subset(draws_gpcovf_m, variable='f'))
sigma <- colMeans(subset(draws_gpcovf_m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma,Ef=Efo,sigma=sigmao)
cbind(mcycle,pred) %>%
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
odraws_10p <- opt_10p$draws()
subset(odraws_10p, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE)

#' Compare the model to the data
Ef <- as.numeric(subset(odraws_10p, variable='f'))
sigma <- as.numeric(subset(odraws_10p, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
ggplot(data=mcycle_10p,aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(data=mcycle,aes(x=times,y=Ef), color=set1[1])+
  geom_line(data=mcycle,aes(x=times,y=Ef-2*sigma), color=set1[1],
            linetype="dashed")+
  geom_line(data=mcycle,aes(x=times,y=Ef+2*sigma), color=set1[1],
            linetype="dashed")

#' The model fit is clearly over-fitted and over-confident.
#' 
#' Sample using dynamic HMC
#+ fit_10p, results='hide'
fit_10p <- model_gpcovf$sample(data=standata_10p,
                               iter_warmup=1000, iter_sampling=1000,
                               adapt_delta=0.95,
                               chains=4, parallel_chains=2, refresh=100)

#' Check whether parameters have reasonable values
draws_10p <- fit_10p$draws()
summarise_draws(subset(draws_10p, variable=c('sigma_','lengthscale_','sigma'),
                       regex=TRUE))

#' Compare the model to the data
draws_10p_m <- as_draws_matrix(draws_10p)
Ef <- colMeans(subset(draws_10p_m, variable='f'))
sigma <- colMeans(subset(draws_10p_m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
ggplot(data=mcycle_10p,aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(data=mcycle,aes(x=times,y=Ef), color=set1[1])+
  geom_line(data=mcycle,aes(x=times,y=Ef-2*sigma), color=set1[1],
            linetype="dashed")+
  geom_line(data=mcycle,aes(x=times,y=Ef+2*sigma), color=set1[1],
            linetype="dashed")

#' The posterior predictive distribution is much more conservative and
#' shows the uncertainty due to having only a small number of
#' observations.
#' 
#' Compare the posterior draws to the optimized parameters
optim<-as_draws_df(subset(odraws_10p,
                          variable=c('sigma','sigma_f','lengthscale_f')))
drawsdf<-as_draws_df(subset(draws_10p,
                            variable=c('sigma','sigma_f','lengthscale_f')))%>%
  thin_draws(thin=5)
drawsdf%>%
  ggplot(aes(x=lengthscale_f,y=sigma))+
  geom_point(color=set1[2])+
  geom_point(data=optim,color=set1[1],size=4)+
  annotate("text",x=median((drawsdf$lengthscale_f)),y=max((drawsdf$sigma))+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=(optim$lengthscale_f+0.01),y=(optim$sigma),
           label='Optimized',hjust=0,color=set1[1],size=6)

#' The optimization result is in the edge of the posterior close to
#' zero residual scale. While there are posterior draws close to zero,
#' integrating over the wide posterior takes into account the
#' uncertainty and the predictions thus are more uncertain, too.
#' 
#' Compare optimized and posterior predictive distributions
Efo <- as.numeric(subset(odraws_10p, variable='f'))
sigmao <- as.numeric(subset(odraws_10p, variable='sigma'))
Ef <- colMeans(subset(draws_10p_m, variable='f'))
sigma <- colMeans(subset(draws_10p_m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma,Ef=Efo,sigma=sigmao)
cbind(mcycle,pred) %>%
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
odraws_gpcovfg <- opt_gpcovfg$draws()
subset(odraws_gpcovfg, variable=c('sigma_','lengthscale_'), regex=TRUE)

#' Compare the model to the data
Ef <- as.numeric(subset(odraws_gpcovfg, variable='f'))
sigma <- as.numeric(subset(odraws_gpcovfg, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
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
                                    iter_warmup=100, iter_sampling=100,
                                    chains=4, parallel_chains=2, refresh=10)

#' Check whether parameters have reasonable values
draws_gpcovfg <- fit_gpcovfg$draws()
summarise_draws(subset(draws_gpcovfg, variable=c('sigma_','lengthscale_'),
                       regex=TRUE))

#' Compare the model to the data
draws_gpcovfg_m <- as_draws_matrix(draws_gpcovfg)
Ef <- colMeans(subset(draws_gpcovfg_m, variable='f'))
sigma <- colMeans(subset(draws_gpcovfg_m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The MCMC integration works well and the model fit looks good.
#' 
#' Plot posterior draws and posterior mean of the mean function
subset(draws_gpcovfg, variable="f") %>%
  as_draws_df() %>%
  pivot_longer(!starts_with("."),
               names_to="ind",
               names_transform = list(ind = readr::parse_number),
               values_to="mu") %>%
  mutate(time=mcycle$times[ind])%>%
  ggplot(aes(time, mu, group = .draw)) +
  geom_line(color=set1[2], alpha = 0.1) +
  geom_point(data=mcycle, mapping=aes(x=times,y=accel), inherit.aes=FALSE)+
  geom_line(data=cbind(mcycle,pred), mapping=aes(x=times,y=Ef),
            inherit.aes=FALSE, color=set1[1], size=1)+
  labs(x="Time (ms)", y="Acceleration (g)")

#' We can also plot the posterior draws of the latent functions, which
#' is a good reminder that individual draws are more wiggly than the
#' average of the draws, and thus show better also the uncertainty,
#' for example, in the edge of the data.
#' 
#' Compare the posterior draws to the optimized parameters
optim<-as_draws_df(subset(odraws_gpcovfg,
                          variable=c('sigma_g','lengthscale_g')))
drawsdf<-as_draws_df(subset(draws_gpcovfg,
                            variable=c('sigma_g','lengthscale_g')))
drawsdf%>%
  ggplot(aes(x=lengthscale_g,y=sigma_g))+
  geom_point(color=set1[2])+
  geom_point(data=optim,color=set1[1],size=4)+
  annotate("text",x=median(drawsdf$lengthscale_g),y=max(drawsdf$sigma_g)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=optim$lengthscale_g+0.01,y=optim$sigma_g,
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
#' Hilbert space basis functions.
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

#' Plot 10 first basis functions
q %>%
  filter(ind<=10) %>%
  ggplot(aes(x=x, y=f, group=ind)) +
  geom_line() +
  facet_grid(rows=vars(ind))

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
odraws_gpbffg <- opt_gpbffg$draws()
subset(odraws_gpbffg, variable=c('intercept','sigma_','lengthscale_'),
       regex=TRUE)

#' Compare the model to the data
Ef <- as.numeric(subset(odraws_gpbffg, variable='f'))
sigma <- as.numeric(subset(odraws_gpbffg, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
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
                                  iter_warmup=500, iter_sampling=500,
                                  chains=4, parallel_chains=2, adapt_delta=0.9)

#' Check whether parameters have reasonable values
draws_gpbffg <- fit_gpbffg$draws()
summarise_draws(subset(draws_gpbffg,
                       variable=c('intercept','sigma_','lengthscale_'),
                       regex=TRUE))

#' Compare the model to the data
draws_gpbffg_m <- as_draws_matrix(draws_gpbffg)
Ef <- colMeans(subset(draws_gpbffg_m, variable='f'))
sigma <- colMeans(subset(draws_gpbffg_m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The MCMC integration works well and the model fit looks good.
#' 
#' Plot posterior draws and posterior mean of the mean function
subset(draws_gpbffg, variable="f") %>%
  thin_draws(thin=5)%>%
  as_draws_df() %>%
  pivot_longer(!starts_with("."),
               names_to="ind",
               names_transform = list(ind = readr::parse_number),
               values_to="mu") %>%
  mutate(time=mcycle$times[ind])%>%
  ggplot(aes(time, mu, group = .draw)) +
  geom_line(color=set1[2], alpha = 0.1) +
  geom_point(data=mcycle, mapping=aes(x=times,y=accel), inherit.aes=FALSE)+
  geom_line(data=cbind(mcycle,pred), mapping=aes(x=times,y=Ef),
            inherit.aes=FALSE, color=set1[1], size=1)+
  labs(x="Time (ms)", y="Acceleration (g)")

#' We can also plot the posterior draws of the latent functions, which
#' is a good reminder that individual draws are more wiggly than the
#' average of the draws, and thus show better also the uncertainty,
#' for example, in the edge of the data.
#' 
#' Compare the posterior draws to the optimized parameters
optim<-as_draws_df(subset(odraws_gpbffg,
                          variable=c('sigma_g','lengthscale_g')))
drawsdf<-as_draws_df(subset(draws_gpbffg,
                            variable=c('sigma_g','lengthscale_g')))
drawsdf%>%
  ggplot(aes(x=lengthscale_g,y=sigma_g))+
  geom_point(color=set1[2])+
  geom_point(data=optim,color=set1[1],size=4)+
  annotate("text",x=median(drawsdf$lengthscale_g),y=max(drawsdf$sigma_g)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=optim$lengthscale_g+0.01,y=optim$sigma_g,
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
                                  chains=4, parallel_chains=2, adapt_delta=0.9)

#' Check whether parameters have reasonable values
draws_gpbffg2 <- fit_gpbffg2$draws()
summarise_draws(subset(draws_gpbffg2,
                       variable=c('intercept','sigma_','lengthscale_'),
                       regex=TRUE))

#' Compare the model to the data
draws_gpbffg2_m <- as_draws_matrix(draws_gpbffg2)
Ef <- colMeans(subset(draws_gpbffg2_m, variable='f'))
sigma <- colMeans(subset(draws_gpbffg2_m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' The MCMC integration works well and the model fit looks good.
#' 
#' Plot posterior draws and posterior mean of the mean function
subset(draws_gpbffg2, variable="f") %>%
  thin_draws(thin=5)%>%
  as_draws_df() %>%
  pivot_longer(!starts_with("."),
               names_to="ind",
               names_transform = list(ind = readr::parse_number),
               values_to="mu") %>%
  mutate(time=mcycle$times[ind])%>%
  ggplot(aes(time, mu, group = .draw)) +
  geom_line(color=set1[2], alpha = 0.1) +
  geom_point(data=mcycle, mapping=aes(x=times,y=accel), inherit.aes=FALSE)+
  geom_line(data=cbind(mcycle,pred), mapping=aes(x=times,y=Ef),
            inherit.aes=FALSE, color=set1[1], size=1)+
  labs(x="Time (ms)", y="Acceleration (g)")

#' We see that when using Matern covariance instead of the
#' exponentiated quadratic, the model fit is more wigggly.
#' 
