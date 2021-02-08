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
#' $K_1$ and $K_2$ are exponentiated quadratic covariance functions.
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

#' ## GP model with Hilbert basis functions
#'
#' Model code
file1 <- "gpbf1.stan"
writeLines(readLines(file1))
#' The model code includes Hilbert space basis function helpers
writeLines(readLines("gpbasisfun_functions.stan"))

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
  geom_line(data=cbind(mcycle,pred), mapping=aes(x=times,y=Ef), inherit.aes=FALSE, color=set1[1], size=1)+
  labs(x="Time (ms)", y="Acceleration (g)")

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
  geom_line(data=cbind(mcycle,pred), mapping=aes(x=times,y=Ef), inherit.aes=FALSE, color=set1[1], size=1) +
  labs(x="Time (ms)", y="Acceleration (g)")
