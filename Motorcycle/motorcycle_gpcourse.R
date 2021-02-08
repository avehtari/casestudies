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
#' Data are modelled with normal distribution having Gaussian process
#' prior on mean and log standard deviation:
#' $$
#' y \sim \mbox{normal}(\mu(x), \exp(\eta(x))\\
#' \mu \sim GP(0, K_1)\\
#' \eta \sim GP(0, K_2)
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
SEED <- 48927 # set random seed for reproducability

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

#' ## GP model with homoskedastic noise

#' Model code
file0 <- "gpcov0.stan"
writeLines(readLines(file0))

#' Compile Stan model
model0 <- cmdstan_model(stan_file = file0)

#' Data to be passed to Stan
standata0 <- list(x=mcycle$times,
                  x2=mcycle$times,
                  y=mcycle$accel,
                  N=length(mcycle$times),
                  N2=length(mcycle$times))

#' Optimize and find MAP estimate
#+ opt0, results='hide'
opt0 <- model0$optimize(data=standata0, init=0.1, algorithm='bfgs')

#' Check whether parameters have reasonable values
odraws0 <- opt0$draws()
subset(odraws0, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE)

#' Compare the model to the data
Ef <- as.numeric(subset(odraws0, variable='f'))
sigma <- as.numeric(subset(odraws0, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' Sample using dynamic HMC
#+ fit0, results='hide'
fit0 <- model0$sample(data=standata0, iter_warmup=500, iter_sampling=500,
                      chains=4, parallel_chains=2, refresh=100)

#' Check whether parameters have reasonable values
draws0 <- fit0$draws()
summarise_draws(subset(draws0, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))

#' Compare the model to the data
draws0m <- as_draws_matrix(draws0)
Ef <- colMeans(subset(draws0m, variable='f'))
sigma <- colMeans(subset(draws0m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")


#' Compare the posterior draws to the optimized parameters
optim<-as_draws_df(subset(odraws0, variable=c('sigma_f','lengthscale_f')))
drawsdf<-as_draws_df(subset(draws0, variable=c('sigma_f','lengthscale_f')))%>%thin_draws(thin=5)
drawsdf%>%
  ggplot(aes(x=lengthscale_f,y=sigma_f))+
  geom_point(color=set1[2])+
  geom_point(data=optim,color=set1[1],size=4)+
  annotate("text",x=median(drawsdf$lengthscale_f),y=max(drawsdf$sigma_f)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=optim$lengthscale_f+0.01,y=optim$sigma_f,
           label='Optimized',hjust=0,color=set1[1],size=6)


#' Compare optimized and posterior predictive distributions
Efo <- as.numeric(subset(odraws0, variable='f'))
sigmao <- as.numeric(subset(odraws0, variable='sigma'))
draws0m <- as_draws_matrix(draws0)
Ef <- colMeans(subset(draws0m, variable='f'))
sigma <- colMeans(subset(draws0m, variable='sigma'))
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

#' ### 10% of data
#' 
#' Data to be passed to Stan
mcycle00 <- mcycle[seq(1,133,by=10),]
standata00 <- list(x=mcycle00$times,
                   x2=mcycle$times,
                   y=mcycle00$accel,
                   N=length(mcycle00$times),
                   N2=length(mcycle$times))

#' Optimize and find MAP estimate
#+ opt00, results='hide'
opt00 <- model0$optimize(data=standata00, init=0.1, algorithm='lbfgs')

#' Check whether parameters have reasonable values
odraws00 <- opt00$draws()
subset(odraws00, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE)

#' Compare the model to the data
Ef <- as.numeric(subset(odraws00, variable='f'))
sigma <- as.numeric(subset(odraws00, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
  ggplot(data=mcycle00,aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(data=mcycle,aes(x=times,y=Ef), color=set1[1])+
  geom_line(data=mcycle,aes(x=times,y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(data=mcycle,aes(x=times,y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' Sample using dynamic HMC
#+ fit00, results='hide'
fit00 <- model0$sample(data=standata00, iter_warmup=1000, iter_sampling=1000,
                       adapt_delta=0.95,
                       chains=4, parallel_chains=2, refresh=100)

#' Check whether parameters have reasonable values
draws00 <- fit00$draws()
summarise_draws(subset(draws00, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))

#' Compare the model to the data
draws00m <- as_draws_matrix(draws00)
Ef <- colMeans(subset(draws00m, variable='f'))
sigma <- colMeans(subset(draws00m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
  ggplot(data=mcycle00,aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(data=mcycle,aes(x=times,y=Ef), color=set1[1])+
  geom_line(data=mcycle,aes(x=times,y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(data=mcycle,aes(x=times,y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' Compare the posterior draws to the optimized parameters
optim<-as_draws_df(subset(odraws00, variable=c('sigma','sigma_f','lengthscale_f')))
drawsdf<-as_draws_df(subset(draws00, variable=c('sigma','sigma_f','lengthscale_f')))%>%thin_draws(thin=5)
drawsdf%>%
  ggplot(aes(x=lengthscale_f,y=sigma))+
  geom_point(color=set1[2])+
  geom_point(data=optim,color=set1[1],size=4)+
  annotate("text",x=median((drawsdf$lengthscale_f)),y=max((drawsdf$sigma))+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=(optim$lengthscale_f+0.01),y=(optim$sigma),
           label='Optimized',hjust=0,color=set1[1],size=6)

#' Compare optimized and posterior predictive distributions
Efo <- as.numeric(subset(odraws00, variable='f'))
sigmao <- as.numeric(subset(odraws00, variable='sigma'))
Ef <- colMeans(subset(draws00m, variable='f'))
sigma <- colMeans(subset(draws00m, variable='sigma'))
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


#' ## GP with covariance matrices
#' 
#' Model code
file2 <- "gpcov.stan"
writeLines(readLines(file2))

#' Compile Stan model
model2 <- cmdstan_model(stan_file = file2)

#' Data to be passed to Stan
standata2 <- list(x=mcycle$times,
                  y=mcycle$accel,
                  N=length(mcycle$times))

#' Optimize and find MAP estimate
#+ opt2, results='hide'
opt2 <- model2$optimize(data=standata2, init=0.1, algorithm='bfgs')

#' Check whether parameters have reasonable values
odraws2 <- opt2$draws()
subset(odraws2, variable=c('sigma_','lengthscale_'), regex=TRUE)

#' Compare the model to the data
Ef <- as.numeric(subset(odraws2, variable='f'))
sigma <- as.numeric(subset(odraws2, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' Sample using dynamic HMC
#+ fit2, results='hide'
fit2 <- model2$sample(data=standata2, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=2, refresh=10)

#' Check whether parameters have reasonable values
draws2 <- fit2$draws()
summarise_draws(subset(draws2, variable=c('sigma_','lengthscale_'), regex=TRUE))

#' Compare the model to the data
draws2m <- as_draws_matrix(draws2)
Ef <- colMeans(subset(draws2m, variable='f'))
sigma <- colMeans(subset(draws2m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' Plot posterior draws and posterior mean of the mean function
subset(draws2, variable="f") %>%
  as_draws_df() %>%
  pivot_longer(!starts_with("."),
               names_to="ind",
               names_transform = list(ind = readr::parse_number),
               values_to="mu") %>%
  mutate(time=mcycle$times[ind])%>%
  ggplot(aes(time, mu, group = .draw)) +
  geom_line(color=set1[2], alpha = 0.1) +
  geom_point(data=mcycle, mapping=aes(x=times,y=accel), inherit.aes=FALSE)+
  geom_line(data=cbind(mcycle,pred), mapping=aes(x=times,y=Ef), inherit.aes=FALSE, color=set1[1], size=1)

#' Compare the posterior draws to the optimized parameters
optim<-as_draws_df(subset(odraws2, variable=c('sigma_g','lengthscale_g')))
drawsdf<-as_draws_df(subset(draws2, variable=c('sigma_g','lengthscale_g')))
drawsdf%>%
  ggplot(aes(x=lengthscale_g,y=sigma_g))+
  geom_point(color=set1[2])+
  geom_point(data=optim,color=set1[1],size=4)+
  annotate("text",x=median(drawsdf$lengthscale_g),y=max(drawsdf$sigma_g)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=optim$lengthscale_g+0.01,y=optim$sigma_g,
           label='Optimized',hjust=0,color=set1[1],size=6)



#' ## GP model with Hilbert basis functions
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
fixbf0 <- modelbf0$sample(data=standatabf0, fixed_param=TRUE, iter=1, iter_sampling=1)
#' There is certainly easier way to manipulate,
#' but this is what I came up quickly
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

#' Model code
file1 <- "gpbf1.stan"
writeLines(readLines(file1))

#' Compile Stan model
model1 <- cmdstan_model(stan_file = file1, include_paths = ".")

#' Data to be passed to Stan
standata1 <- list(x=mcycle$times,
                  y=mcycle$accel,
                  N=length(mcycle$times),
                  c_f=1.5, # factor c of basis functions for GP for f1
                  M_f=40,  # number of basis functions for GP for f1
                  c_g=1.5, # factor c of basis functions for GP for g3
                  M_g=40)  # number of basis functions for GP for g3

#' Optimize and find MAP estimate
#+ opt1, results='hide'
opt1 <- model1$optimize(data=standata1, init=0.1, algorithm='bfgs')

#' Check whether parameters have reasonable values
odraws1 <- opt1$draws()
subset(odraws1, variable=c('intercept','sigma_','lengthscale_'), regex=TRUE)

#' Compare the model to the data
Ef <- as.numeric(subset(odraws1, variable='f'))
sigma <- as.numeric(subset(odraws1, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")


#' Sample using dynamic HMC
#+ fit1, results='hide'
fit1 <- model1$sample(data=standata1, iter_warmup=500, iter_sampling=500,
                      chains=4, parallel_chains=2, adapt_delta=0.9)

#' Check whether parameters have reasonable values
draws1 <- fit1$draws()
summarise_draws(subset(draws1, variable=c('intercept','sigma_','lengthscale_'), regex=TRUE))

#' Compare the model to the data
draws1m <- as_draws_matrix(draws1)
Ef <- colMeans(subset(draws1m, variable='f'))
sigma <- colMeans(subset(draws1m, variable='sigma'))
pred<-data.frame(Ef=Ef,sigma=sigma)
cbind(mcycle,pred) %>%
  ggplot(aes(x=times,y=accel))+
  geom_point()+
  labs(x="Time (ms)", y="Acceleration (g)")+
  geom_line(aes(y=Ef), color=set1[1])+
  geom_line(aes(y=Ef-2*sigma), color=set1[1],linetype="dashed")+
  geom_line(aes(y=Ef+2*sigma), color=set1[1],linetype="dashed")

#' Plot posterior draws and posterior mean of the mean function
subset(draws1, variable="f") %>%
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
  geom_line(data=cbind(mcycle,pred), mapping=aes(x=times,y=Ef), inherit.aes=FALSE, color=set1[1], size=1)

#' Compare the posterior draws to the optimized parameters
optim<-as_draws_df(subset(odraws1, variable=c('sigma_g','lengthscale_g')))
optim<-as_draws_df(subset(odraws1, variable=c('sigma_g','lengthscale_g')))
drawsdf<-as_draws_df(subset(draws1, variable=c('sigma_g','lengthscale_g')))
drawsdf%>%
  ggplot(aes(x=lengthscale_g,y=sigma_g))+
  geom_point(color=set1[2])+
  geom_point(data=optim,color=set1[1],size=4)+
  annotate("text",x=median(drawsdf$lengthscale_g),y=max(drawsdf$sigma_g)+0.1,
           label='Posterior draws',hjust=0.5,color=set1[2],size=6)+
  annotate("text",x=optim$lengthscale_g+0.01,y=optim$sigma_g,
           label='Optimized',hjust=0,color=set1[1],size=6)

