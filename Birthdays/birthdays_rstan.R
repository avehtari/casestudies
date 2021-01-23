#' ---
#' title: "Bayesian workflow book - Birthdays"
#' author: "Gelman, Vehtari, Simpson, et al"
#' date: "First version 2020-12-28. Last modified `r format(Sys.Date())`."
#' output:
#'   html_document:
#'     theme: readable
#'     toc: true
#'     toc_depth: 2
#'     toc_float: true
#'     code_download: true
#' ---

#' Workflow for iterative building of time series model.
#'
#' We analyse the number of births per day in USA 1969-1988 using
#' Gaussian process time series model with several model components.
#'
#' -------------
#' 

#' - switch to log10
#' - switch plotting with 100 as baseline
#' 

#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA)
# switch this to TRUE to save figures in separate files
savefigs <- FALSE

#' #### Load packages
library("rprojroot")
root<-has_file(".Workflow-Examples-root")$make_fix_file()
library(tidyverse)
library(rstan)
rstan_options(auto_write = TRUE)
library(posterior)
options(pillar.negative = FALSE)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))
library(patchwork)
set1 <- RColorBrewer::brewer.pal(7, "Set1")
SEED <- 48927 # set random seed for reproducability
library(mvtnorm)

#' ## Load and plot data
#' 
#' Load birthds per day in USA 1959-1988:
data <- read_csv(root("Birthdays/data", "births_usa_1969.csv"))

#' Add date type column for plotting
data <- data %>%
  mutate(date = as.Date("1968-12-31") + id,
         births_relative100 = births/mean(births)*100)

#' Plot all births
data %>%
  ggplot(aes(x=date, y=births)) + geom_point(color=set1[2]) +
  labs(x="Date", y="Births per day")

#' Plot all births as relative to mean
data %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative births per day")

#' Plot mean per day of year
data %>%
  group_by(day_of_year) %>%
  summarise(meanbirths=mean(births_relative100)) %>%
  ggplot(aes(x=as.Date("1959-12-31")+day_of_year, y=meanbirths)) + geom_point(color=set1[2]) +
  geom_hline(yintercept=100, color='gray') +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  labs(x="Day of year", y="Relative births per day of year")

#' Plot mean per day of week
data %>%
  group_by(day_of_week) %>%
  summarise(meanbirths=mean(births_relative100)) %>%
  ggplot(aes(x=day_of_week, y=meanbirths)) + geom_point(color=set1[2], size=4) +
  geom_hline(yintercept=100, color='gray') +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  labs(x="Day of week", y="Births per day of week")

#' Compile Stan model
model1 <- stan_model(file = root("Birthdays", "gpbf1rstan.stan"))

#' Data to be passed to Stan
c_f1 <- 1.5  # factor c for a smooth trend f1
M_f1 <- 10   # number of basis functions for a smooth trend f1
standata1 <- list(x=data$id, y=log(data$births_relative100), N=length(data$id),
                  M_f1= M_f1, c_f1= c_f1)

#' Optimizing is faster than sampling
opt1 <- optimizing(model1, data=standata1, verbose=TRUE, as_vector=FALSE,
                   algorithm='BFGS', tol_rel_obj=1, tol_rel_grad=1)

opt1is <- optimizing(model1, data=standata1, verbose=TRUE, as_vector=FALSE,
                   algorithm='BFGS', tol_rel_obj=1, tol_rel_grad=1,
                   draws=4000, importance_resampling=TRUE)

opt1is2 <- optimizing(model1, data=standata1, verbose=TRUE, as_vector=FALSE,
                   algorithm='BFGS', tol_rel_obj=1, tol_rel_grad=1,
                   draws=4000, importance_resampling=2)

opt1is3 <- optimizing(model1, data=standata1, verbose=TRUE, as_vector=FALSE,
                   algorithm='BFGS', tol_rel_obj=1, tol_rel_grad=1,
                   draws=4000, importance_resampling=7)

#' Check whether hyperparameters have reasonable values
odraws1 <- as_draws_matrix(opt1$theta_tilde)
odraws1is1 <- as_draws_matrix(opt1is$theta_tilde)
odraws1is2 <- as_draws_matrix(opt1is2$theta_tilde)

odraws1is3 <- as_draws_matrix(opt1is3$theta_tilde)

summarise_draws(subset(odraws1is, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE),
                'mean', ~quantile(.x, probs = c(0.05, 0.95)))
summarise_draws(subset(odraws1is2, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE),
                'mean', ~quantile(.x, probs = c(0.05, 0.95)))
summarise_draws(subset(odraws1is3, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE),
                'mean', ~quantile(.x, probs = c(0.05, 0.95)))

odraws1is <- as_draws_matrix(opt1is$theta_tilde) %>% weight_draws(weights=opt1is$log_p-opt1is$log_g, log=TRUE) %>% resample_draws()
odraws1is2 <- as_draws_matrix(opt1is2$theta_tilde) %>% weight_draws(weights=opt1is2$log_p-opt1is2$log_g, log=TRUE) %>% resample_draws()

odraws1is3 <- as_draws_matrix(opt1is3$theta_tilde) %>% weight_draws(weights=opt1is3$log_p-opt1is3$log_g, log=TRUE) %>% resample_draws()

summarise_draws(subset(odraws1is, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE),
                'mean', ~quantile(.x, probs = c(0.05, 0.95)))
summarise_draws(subset(odraws1is2, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE),
                'mean', ~quantile(.x, probs = c(0.05, 0.95)))

(loo::psis(opt1is3$log_p-opt1is3$log_g,r_eff=NA))

#' Compare the modeled function to data
oEf <- exp(apply(subset(odraws1, variable='f'), 2, median))
data %>%
  mutate(oEf = oEf) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=oEf), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")

#' Sample short chains
fit1 <- sampling(model1, data= standata1, iter=2000,  warmup=1000, chains=4, cores=2)

#' Check whether hyperparameters have reasonable values
draws1 <- as_draws_df(fit1)
summarise_draws(subset(draws1, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE))

summarise_draws(subset(draws1, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE), 'mean', ~quantile(.x, probs = c(0.05, 0.95)))



#' Trace plot shows multimodality
mcmc_trace(fit1, regex_pars=c('intercept','alpha','rho','sigma'))

#' Start with optimization result
fit1 <- sampling(model1, data= standata1, iter=200,  warmup=100, chains=4, cores=4,
                 init=function() { opt1$par })

#' Check whether hyperparameters have reasonable values
draws1 <- as_draws_df(fit1)
summarise_draws(subset(draws1, variable=c('intercept','alpha','rho','sigma')))

#' Trace plot shows just slow mixing
mcmc_trace(fit1, regex_pars=c('intercept','alpha','rho','sigma'))

#' Compare the modeled function to data
draws1 <- as_draws_matrix(fit1)
Ef <- exp(apply(subset(draws1, variable='f'), 2, median))
data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")

#' Compare the result from short sampling to optimizing
data %>%
  mutate(Ef = Ef,
         oEf = oEf) %>%
  ggplot(aes(x=Ef, y=oEf)) + geom_point(color=set1[2]) +
  geom_abline() +
  labs(x="Ef from short Markov chain", y="Ef from optimizing")

#' Compile Stan model 2
model2 <- stan_model(file = root("Birthdays", "gpbf2.stan"))

#' Data to be passed to Stan
c_f1 <- 1.5  # factor c for a smooth trend f1
M_f1 <- 30   # number of basis functions for a smooth trend f1
J_f2 <- 10   # number of cos and sin functions for periodic f2
standata2 <- list(x=data$id, y=log(data$births_relative100), N=length(data$id),
                  M_f1=M_f1, c_f1=c_f1, J_f2=J_f2) 

#' Optimizing is faster than sampling
opt2 <- optimizing(model2, data= standata2, verbose=TRUE, as_vector=FALSE,
                   algorithm='BFGS', tol_rel_obj=1, tol_rel_grad=1,
                   init=0, draws=4000, importance_resampling=TRUE)

opt2 <- optimizing(model2, data= standata2, verbose=TRUE, as_vector=FALSE,
                   algorithm='BFGS', tol_rel_obj=1, tol_rel_grad=1,
                   init=0)

odraws2 <- as_draws_matrix(opt2$theta_tilde)
subset(odraws2, variable=c('intercept','alpha','rho','sigma'))

#' Compare the modeled function to data
Ef <- exp(as.numeric(subset(odraws2, variable='f')))
Ef1 <- as.numeric(subset(odraws2, variable='f1'))
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- as.numeric(subset(odraws2, variable='f2'))
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))

pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf

pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf1

pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1959-12-31")+day_of_year, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day of year")
pf2

pf / (pf1 + pf2)

#' Sample short chains
fit2 <- sampling(model2, data=standata2, iter=200,  warmup=100, chains=4, cores=4)

#' Check whether hyperparameters have reasonable values
draws2 <- as_draws_df(fit2)
summarise_draws(subset(draws2, variable=c('intercept','alpha','rho','sigma')))

#' Trace plot shows multimodality
mcmc_trace(fit2, regex_pars=c('intercept','alpha','rho','sigma'))

#' Start with optimization result
fit2 <- sampling(model2, data=standata2, iter=200,  warmup=100, chains=4, cores=4,
                 init=function() { opt2$par })

#' Check whether hyperparameters have reasonable values
draws2 <- as_draws_df(fit2)
summarise_draws(subset(draws2, variable=c('intercept','alpha','rho','sigma')))

#' Trace plot shows multimodality
mcmc_trace(fit2, regex_pars=c('intercept','alpha','rho','sigma'))


#' Compare the modeled function to data
draws2 <- as_draws_matrix(fit2)
Ef <- exp(apply(subset(draws2, variable='f'), 2, median))
Ef1 <- apply(subset(draws2, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws2, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))

pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf

pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf1

pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1959-12-31")+day_of_year, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day of year")
pf2

pf / (pf1 + pf2)

#' Model 3
model3 <- stan_model(file = root("Birthdays", "gpbf3.stan"))

#' Data to be passed to Stan
c_f1 <- 1.5  # factor c for a smooth trend f1
M_f1 <- 30   # number of basis functions for a smooth trend f1
J_f2 <- 10   # number of cos and sin functions for periodic f2
standata3 <- list(x=data$id, y=log(data$births_relative100), N=length(data$id),
                  M_f1=M_f1, c_f1=c_f1, J_f2=J_f2, day_of_week=data$day_of_week)

#' Optimizing is faster than sampling
opt3 <- optimizing(model3, data=standata3, verbose=TRUE, as_vector=FALSE,
                   algorithm='BFGS', tol_rel_obj=1, tol_rel_grad=1, init=0)

odraws3 <- as_draws_matrix(opt3$theta_tilde)
subset(odraws3, variable=c('intercept0','alpha','rho','sigma'))
subset(odraws3, variable=c('beta_f3'))

#' Compare the modeled function to data
Ef <- exp(as.numeric(subset(odraws3, variable='f')))
Ef1 <- as.numeric(subset(odraws3, variable='f1'))
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- as.numeric(subset(odraws3, variable='f2'))
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- as.numeric(subset(odraws3, variable='f_day_of_week'))
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))

pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf

pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf1

pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1959-12-31")+day_of_year, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day of year")
pf2

pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day of week")
pf3

(pf + pf1) / (pf2 + pf3)

#' Sample short chains
fit3 <- sampling(model3, data=standata3, iter=200,  warmup=100, chains=4, cores=4,
                 init=function() { opt3$par })

#' Check whether hyperparameters have reasonable values
draws3 <- as_draws_df(fit3)
summarise_draws(subset(draws3, variable=c('intercept0','alpha','rho','sigma')))
summarise_draws(subset(draws3, variable=c('beta_f3')))

#' Compare the modeled function to data
draws3 <- as_draws_matrix(fit3)
Ef <- exp(apply(subset(draws3, variable='f'), 2, median))
Ef1 <- apply(subset(draws3, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws3, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws3, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))

pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Births per day")
pf

pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf1

pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1959-12-31")+day_of_year, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day of year")
pf2

pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day of week")
pf3

(pf + pf1) / (pf2 + pf3)


#' Model 4
model4 <- stan_model(file = root("Birthdays", "gpbf4.stan"))

#' Data to be passed to Stan
c_f1 <- 1.5  # factor c for a smooth trend f1
M_f1 <- 30   # number of basis functions for a smooth trend f1
J_f2 <- 10   # number of cos and sin functions for periodic f2
c_f3 <- 1.5  # factor c for a smooth trend f3
M_f3 <- 10   # number of basis functions for a smooth trend f3
standata4 <- list(x=data$id, y=log(data$births_relative100), N=length(data$id),
                  M_f1=M_f1, c_f1=c_f1, J_f2=J_f2, M_f3=M_f3, c_f3=c_f3, day_of_week=data$day_of_week) 

#' Optimizing is faster than sampling
opt4 <- optimizing(model4, data=standata4, verbose=TRUE, as_vector=FALSE,
                   algorithm='BFGS', tol_rel_obj=1, tol_rel_grad=1, init=0)

odraws4 <- as_draws_matrix(opt4$theta_tilde)
subset(odraws4, variable=c('intercept0','alpha','rho','sigma'))
subset(odraws4, variable=c('beta_f3'))

#' Compare the modeled function to data
Ef <- exp(as.numeric(subset(odraws4, variable='f')))
Ef1 <- as.numeric(subset(odraws4, variable='f1'))
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- as.numeric(subset(odraws4, variable='f2'))
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- as.numeric(subset(odraws4, variable='f_day_of_week'))
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef3 <- as.numeric(subset(odraws4, variable='f3'))
Ef3 <- exp(Ef3 - mean(Ef3) + mean(log(data$births_relative100)))

pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf

pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  labs(x="Date", y="Births per day")
pf1

pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1959-12-31")+day_of_year, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day of year")
pf2

pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day of week")
pf3

pf3b <- data %>%
  mutate(Ef3 = Ef3) %>%
  ggplot(aes(x=date, y=births_relative100/Ef1/Ef2*100*100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Births per day")
pf3b

(pf + pf1) / (pf2 + (pf3/pf3b))

#' Model 5

mod1c <- cmdstan_model(stan_file = root("Birthdays", "gpbf1.stan"), include_paths = root("Birthdays"))
opt1c <- mod1c$optimize(data = standata1, algorithm='bfgs')
fit1c <- mod1c$sample(data = standata1, iter_warmup=100, iter_sampling=100, chains=4, parallel_chains=4,
                      init=function() { as_draws_list(opt1c$draws()) })

draws1c <- fit1c$draws()
summarise_draws(subset(draws1c, variable=c('intercept','alpha','rho','sigma')))
mcmc_trace(draws1c, regex_pars=c('intercept','alpha','rho','sigma'))

> 
```
```{r}
fit1c$summary()
```
Plot the histogram of the posterior draws
```{r}
