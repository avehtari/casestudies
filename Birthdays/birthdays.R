#' ---
#' title: "Bayesian workflow book - Birthdays"
#' author: "Gelman, Vehtari, Simpson, et al"
#' date: "First version 2020-12-28. Last modified `r format(Sys.Date())`."
#' output:
#'   html_document:
#'     theme: readable
#'     toc: true
#'     toc_depth: 3
#'     toc_float: true
#'     code_download: true
#' ---

#' Workflow for iterative building of a time series model.
#'
#' We analyse the relative number of births per day in USA 1969-1988
#' using Gaussian process time series model with several model
#' components that can explain the long term, seasonal, weekly, day of
#' year, and special floatind day variation.
#'
#' Stan model codes are available in [the corresponding git repo](https://github.com/avehtari/casestudies/tree/master/Birthdays)
#'
#' -------------
#' 

#+ setup, include=FALSE
knitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA, cache=FALSE)
# switch this to TRUE to save figures in separate files
savefigs <- FALSE

#' #### Load packages
library("rprojroot")
root<-has_file(".Workflow-Examples-root")$make_fix_file()
library(tidyverse)
library(cmdstanr)
library(posterior)
options(pillar.neg = FALSE, pillar.subtle=FALSE, pillar.sigfig=2)
library(loo)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))
library(patchwork)
set1 <- RColorBrewer::brewer.pal(7, "Set1")

#' ## Load and plot data
#' 
#' Load birthdays per day in USA 1969-1988:
data <- read_csv(root("Birthdays/data", "births_usa_1969.csv"))

#' Add date type column for plotting
data <- data %>%
  mutate(date = as.Date("1968-12-31") + id,
         births_relative100 = births/mean(births)*100)

#' ### Plot all births
#'
#' We can see slow variation in trend, yearly pattern, and especially
#' in the later years spread to lower and higher values.
data %>%
  ggplot(aes(x=date, y=births)) + geom_point(color=set1[2]) +
  labs(x="Date", y="Relative number of births")

#' ### Plot all births as relative to mean
#'
#' To make the interpretation we switch to examine the relative
#' change, with the mean level denoted with 100.
data %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative births per day")

#' ### Plot mean per day of year
#'
#' We can see the generic pattern in yearly seasonal trend simply by
#' averaging over each day of year (day_of_year has numbers from 1 to
#' 366 every year with leap day being 60 and 1st March 61 also on
#' non-leap-years).
data %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100)) %>%
  ggplot(aes(x=as.Date("1986-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2]) +
  geom_hline(yintercept=100, color='gray') +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  labs(x="Day of year", y="Relative births per day of year")

#' ### Plot mean per day of week
#' 
#' We can see the generic pattern in weekly trend simply by averaging
#' over each day of week.
data %>%
  group_by(day_of_week) %>%
  summarise(meanbirths=mean(births_relative100)) %>%
  ggplot(aes(x=day_of_week, y=meanbirths)) +
  geom_point(color=set1[2], size=4) +
  geom_hline(yintercept=100, color='gray') +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  labs(x="Day of week", y="Relative number of births of week")

#' ## Previous analyses
#' 
#' We have analysed the same data before in BDA3 and thus had idea of
#' what kind of model to use. For BDA3 we used GPstuff software which
#' is Gaussian process specific software for Matlab and Octave. As
#' Stan has aimed to be very generic it can be slower than specialized
#' software for some specific models such as Gaussian processes, but
#' Stan provides more flexibility in the model definition.
#'
#' Riutort-Mayol et al (2020) demonstrate Hilbert space approximate
#' basis function approximation of Gaussian processes also for the
#' same birthday data. In the experiments the inference was slower
#' than expected raising suspicion of inefficient model code or bad
#' posterior shape due to bad model specification.
#'
#' ## Workflow for quick iterative model building
#'
#' Even we have general idea for the model (slow trend, seasonal
#' trend, weekday effect, etc), adding them all at once to the model
#' makes the model complex and difficult to debug and solve the
#' computational problems. It is thus natural to build the model
#' gradually and check that each addition works before adding the next
#' model component. During this iterative model building we want the
#' inference to be fast, but it doesn't need to be very accurate as
#' long as qualitatively the new model is reasonable. For quick
#' testing and iterative model building we can use optimization and
#' shorter MCMC chains that we would not recommend for the final
#' inference.  Furthermore, in this specific example, the new
#' additions are qualitatively so clear improvements that there is no
#' need for quantitative model comparison whether the additions are
#' ``significant'' (see also Navarro, 2019) and there is no danger of
#' overfitting. Although there is one part of the model where the data
#' is weakly informative and the prior choices seem to matter and
#' we'll get back to this and consequences later. Overall we build
#' tens of different models but illustrate here only the main line.
#' 

#' ## Models for relative number of birthdays
#'
#' As the relative number of births is positive it's natural to model
#' the logarithm value. The generic form of the models is
#' $$
#' y \sim \mbox{normal}(f(x), \sigma),
#' $$
#' where $f$ is different and gradually more complex function
#' conditional on $x$ that includes running day number, day of year,
#' day of week and eventually some special floating US bank holidays.
#'
#' ### Model 1: Slow trend
#'
#' The model 1 is just the slow trend over the years using Hilbert
#' space basis function approximated Gaussian process
#' $$
#' f = \mbox{intercept} + f_1\\
#' \mbox{intercept} \sim \mbox{normal}(0,1)\\
#' f_1 \sim \mbox{GP}(0,K_1)
#' $$
#' where GP has exponentiated quadratic covariance function.
#' 
#' In this phase the code from Riutort-Mayol et al.(2020) was cleaned
#' and written to be more efficient, but only the one GP component was
#' included to make the testing easier. Although the code was made
#' more efficient, the aim wasn't to make it the fastest possible as
#' the later model changes may have bigger effect on the performance
#' (it's good o avoid premature optimization). We also use quite small
#' number of basis functions to make the code run faster, and only
#' later examine more carefully whether the number of basis function
#' is sufficient compared to the posterior of the length scale (see,
#' Riutort-Mayol et al., 2020).
#'

#' Compile Stan model [gpbf1.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf1.stan) which includes [gpbasisfun_functions.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions.stan)
#+ model1, results='hide'
model1 <- cmdstan_model(stan_file = root("Birthdays", "gpbf1.stan"),
                        include_paths = root("Birthdays"))
#' Data to be passed to Stan
standata1 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=10)  # number of basis functions for GP for f1

#' As the basis function approximation and priors restrict the
#' complexity of GP, we can safely use optimization to get a very
#' quick initial result to check that the model code is computing what
#' we intended. As there are only 14 parameters and 7305 observations
#' it's likely that the posterior is close to normal (in unconstrained
#' space). In this case the optimization takes less than one second
#' while MCMC sampling with default options would have taken several
#' minutes. Although this result can be useful in a quick workflow,
#' the result should not be used as the final result.
#+ opt1, results='hide'
opt1 <- model1$optimize(data = standata1, init=0, algorithm='bfgs')
#' Check whether parameters have reasonable values
odraws1 <- opt1$draws()
subset(odraws1, variable=c('intercept','sigma_f1','lengthscale_f1','sigma'))

#' Compare the model to the data
oEf <- exp(as.numeric(subset(odraws1, variable='f')))
data %>%
  mutate(oEf = oEf) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=oEf), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")

#' After we get the model working using optimization we can compare
#' the result to using short MCMC chains which will also provide us
#' additional information on speed of different code implementations
#' for the same model. We intentionally use just 1/10th length from
#' the usual recommendation, as during the iterative model building a
#' rough results are sufficient. When testing the code we initially
#' used just one chain, but at this point running four chains with
#' four core CPU doesn't add much to the wall clock time, but gives
#' more information of how easy it is sample from the posterior and
#' can reveal if there are multiple modes. Although the result from
#' short chains can be useful in a quick workflow, the result should
#' not be used as th final result.
#+ fit1, results='hide'
fit1 <- model1$sample(data=standata1, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4, seed=3891)

#' Depending on the random seed and luck, we sometimes observed that
#' some of the chains got stuck in different modes. We could see this
#' in high Rhat and low ESS diagnostic values.
draws1 <- fit1$draws()
summarise_draws(subset(draws1, variable=c('intercept','sigma_f1','lengthscale_f1','sigma')))
#' Examining the trace plots shows the multimodality clearly.
mcmc_trace(draws1, regex_pars=c('intercept','sigma_f1','lengthscale_f1','sigma'))

#' In this case it was easy to figure out that some of the chains got
#' stuck in qualitatively much worse modes. We don't in general
#' recommend to start from the mode as the mode is not usually
#' representative point in hierarchical model posterior or in high
#' dimensional posterior, but we can use this again to speed up the
#' iterative model building as long as we check that the optimization
#' result is sensible and later do more careful inference. Although
#' the result from short chains can be useful in a quick workflow, the
#' result should not be used as the final result.
init1 <- sapply(c('intercept','sigma_f1','lengthscale_f1','beta_f1','sigma'),
                function(variable) {as.numeric(subset(odraws1, variable=variable))})
#+ fit1init, results='hide'
fit1 <- model1$sample(data=standata1, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=function() { init1 })

#' We now observe better Rhat and ESS diagnostic values, although due
#' to very short chains they are not yet perfect. We are likely to
#' also observe Hamiltonian Monte Carlo divergences and treedepth
#' exceedences in dynamic building of the Hamiltonian trajectory,
#' but there is no need to worry about those as long as the
#' model results are qualitatively sensible as these computational
#' issues can also go away when the model itself is improved. In all
#' the following short MCMC samplings we get some or many divergences
#' and usually very large number of treedepth exceedences. Divergences
#' indicate possible bias and should be eventually investigated
#' carefully. Treedepth exceedences indicate strong posterior
#' dependencies and slow mixing and sometimes the posterior can be
#' much improved by changing the parameterization or priors, but as
#' the treedepth exceedences don't indicate bias there is no need for
#' more careful analysis if the resulting ESS and MCSE values are good
#' for the purpose in hand.  We'll come back later to more careful
#' analysis of the final models.
draws1 <- fit1$draws()
summarise_draws(subset(draws1, variable=c('intercept','sigma_f1','lengthscale_f1','sigma')))
#' Trace plot shows slow mixing but no multimodality.
mcmc_trace(draws1, regex_pars=c('intercept','sigma_f1','lengthscale_f1','sigma'))

#' The model result from short MCMC chains looks very similar to the
#' optimization result.
draws1 <- as_draws_matrix(draws1)
Ef <- exp(apply(subset(draws1, variable='f'), 2, median))
data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")

#' If we compare the result from short sampling to optimizing, we
#' don't see practical difference in the predictions (although we see
#' later more differences between optimization and MCMC).
data %>%
  mutate(Ef = Ef,
         oEf = oEf) %>%
  ggplot(aes(x=Ef, y=oEf)) + geom_point(color=set1[2]) +
  geom_abline() +
  labs(x="Ef from short Markov chain", y="Ef from optimizing")

#' ### Model 2: Slow trend + yearly seasonal trend
#' 
#' The model 2 adds yearly seasonal trend using GP with periodic
#' covariance function.
#' $$
#' f = \mbox{intercept} + f_1 + f_2 \\
#' \mbox{intercept} \sim \mbox{normal}(0,1)\\
#' f_1 \sim \mbox{GP}(0,K_1)\\
#' f_2 \sim \mbox{GP}(0,K_2)
#' $$
#' where the first GP uses the exponentiated quadratic covariance
#' function, and the second one a periodic covariance function. Most
#' years have 365 calendar days and every four years (during the data
#' range) there are 366 days, and thus we simplify and use period of
#' 365.25 for the periodic component,
#'
#' The first version of model 2 with the added periodic component
#' following from Riutort-Mayol (2020) turned out be very slow. With
#' the default MCMC options the inference would have taken hours, but
#' with the short chains it was possible to infer that something has
#' to be wrong. The model output was sensible, but diagnostics
#' indicated very slow mixing. By more careful examination of the
#' model it turned out that the periodic component was including
#' another intercept term and with two intercept terms their sum was
#' well informed by the data, but individually they were not well
#' informed and thus the posteriors were wide, which lead to very slow
#' mixing. This bad model is not shown here, but the optimization,
#' short MCMC chains and sampling diagnostic tools were crucial for
#' fast experimentation and solving the problem.
#' 
#' Compile Stan model 2 (the fixed version) [gpbf2.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf2.stan)
#+ model2, results='hide'
model2 <- cmdstan_model(stan_file = root("Birthdays", "gpbf2.stan"),
                        include_paths = root("Birthdays"))

#' Data to be passed to Stan
standata2 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=10,  # number of basis functions for GP for f1
                  J_f2=20)  # number of basis functions for periodic f2

#' Optimizing is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ opt2, results='hide'
opt2 <- model2$optimize(data=standata2, init=0, algorithm='bfgs')

#' Check whether parameters have reasonable values
odraws2 <- opt2$draws()
subset(odraws2, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE)

#' Compare the model to the data
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
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf / (pf1 + pf2)

#' Sample short chains using the optimization result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
init2 <- sapply(c('intercept','lengthscale_f1','lengthscale_f2','sigma_f1','sigma_f2','sigma','beta_f1','beta_f2'),
                function(variable) {as.numeric(subset(odraws2, variable=variable))})
#+ fit2, results='hide'
fit2 <- model2$sample(data=standata2, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=function() { init2 })

#' Check whether parameters have reasonable values
draws2 <- fit2$draws()
summarise_draws(subset(draws2, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE))

#' Compare the model to the data
draws2 <- as_draws_matrix(draws2)
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
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf / (pf1 + pf2)

#' Seasonal component has reasonable fit to the data.
#' 

#' ### Model 3: Slow trend + yearly seasonal trend + day of week
#'
#' Based on the quick plotting of the data above, day of week has a
#' clear effect and there are less babies born on Saturday and
#' Sunday. This can be taken into account with simple additive
#' coefficients. We fix the effect of Monday to 0 and have additional
#' coefficients for other weekdays.
#' $$
#' f = \mbox{intercept} + f_1 + f_2 + \beta_{\mbox{day of week}} \\
#' \mbox{intercept} \sim \mbox{normal}(0,1)\\
#' f_1 \sim \mbox{GP}(0,K_1)\\
#' f_2 \sim \mbox{GP}(0,K_2)\\
#' \beta_{\mbox{day of week}} = 0 \quad \mbox{if day of week is Monday}\\
#' \beta_{\mbox{day of week}} \sim \mbox{normal}(0,1) \quad \mbox{if day of week is not Monday}
#' $$
#' 
#' Compile Stan model 3 [gpbf3.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf3.stan)
#+ model3, results='hide'
model3 <- cmdstan_model(stan_file = root("Birthdays", "gpbf3.stan"),
                        include_paths = root("Birthdays"))

#' Data to be passed to Stan
standata3 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=10,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  day_of_week=data$day_of_week)

#' Optimizing is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ opt3, results='hide'
opt3 <- model3$optimize(data=standata3, init=0, algorithm='bfgs')

#' Check whether parameters have reasonable values
odraws3 <- opt3$draws()
subset(odraws3, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE)
subset(odraws3, variable=c('beta_f3'))

#' Compare the model to the data
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
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
(pf + pf1) / (pf2 + pf3)

#' Sample short chains using the optimization result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
init3 <- sapply(c('intercept0','lengthscale_f1','lengthscale_f2','sigma_f1','sigma_f2','sigma',
                  'beta_f1','beta_f2','beta_f3'),
                function(variable) {as.numeric(subset(odraws3, variable=variable))})
#+ fit3, results='hide'
fit3 <- model3$sample(data=standata3, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=function() { init3 })

#' Check whether parameters have reasonable values
draws3 <- fit3$draws()
summarise_draws(subset(draws3, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE))
summarise_draws(subset(draws3, variable=c('beta_f3')))

#' Compare the model to the data
draws3 <- as_draws_matrix(draws3)
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
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
(pf + pf1) / (pf2 + pf3)

#' Weekday effects are easy to estimate as there are about thousand
#' observations per weekday.
#' 

#' ### Model 4: long term smooth + seasonal + weekday with increasing magnitude
#'
#' Looking at the time series of whole data we see the dots
#' representing the daily values forming three branches that are
#' getting further away from each other. In previous analysis (BDA3)
#' we also had a model component allowing gradually changing effect
#' for day of week and did observe that the effect of Saturday and
#' Sunday did get stronger in time. The next model includes time
#' dependent magnitude component for the day of week effect.
#' $$
#' f = \mbox{intercept} + f_1 + f_2 + \exp(g_3)\beta_{\mbox{day of week}} \\
#' \mbox{intercept} \sim \mbox{normal}(0,1)\\
#' f_1 \sim \mbox{GP}(0,K_1)\\
#' f_2 \sim \mbox{GP}(0,K_2)\\
#' g_3 \sim \mbox{GP}(0,K_3)\\
#' \beta_{\mbox{day of week}} = 0 \quad \mbox{if day of week is Monday}\\
#' \beta_{\mbox{day of week}} \sim \mbox{normal}(0,1) \quad \mbox{if day of week is not Monday}
#' $$
#' The magnitude of the weekday effect is modelled with $\exp(g_3)$,
#' where $g_3$ has GP prior with zero mean and exponentiated quadratic
#' covariance function.
#' 
#' Compile Stan model 4 [gpbf4.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf4.stan)
#+ model4, results='hide'
model4 <- cmdstan_model(stan_file = root("Birthdays", "gpbf4.stan"),
                        include_paths = root("Birthdays"))

#' Data to be passed to Stan
standata4 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=10,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  c_g3=1.5, # factor c of basis functions for GP for g3
                  M_g3=5,   # number of basis functions for GP for g3
                  day_of_week=data$day_of_week) 

#' As we have increased the complexity of the model, the mode starts
#' to be less and less representative of the posterior. We still use
#' the optimization to check that code returns something reasonable
#' and as initial values for MCMC, but we now stop the optimization
#' early. By adding `tol_obj=10` argument, the optimization stops when
#' the change in the log posterior density is less than 10, which is
#' likely to happened before reaching the mode.
#+ opt4, results='hide'
opt4 <- model4$optimize(data=standata4, init=0, algorithm='bfgs', tol_obj=10)

#' Check whether parameters have reasonable values
odraws4 <- opt4$draws()
subset(odraws4, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE)
subset(odraws4, variable=c('beta_f3'))

#' Compare the model to the data
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
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
pf3b <- data %>%
  mutate(Ef3 = Ef3) %>%
  ggplot(aes(x=date, y=births_relative100/Ef1/Ef2*100*100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
(pf + pf1) / (pf2 + pf3b)

#' Sample short chains using the early stopped optimization result as
#' initial values (although the result from short chains can be useful
#' in a quick workflow, the result should not be used as the final
#' result).
init4 <- sapply(c('intercept0','lengthscale_f1','lengthscale_f2','lengthscale_g3',
                  'sigma_f1','sigma_f2','sigma_g3','sigma',
                  'beta_f1','beta_f2','beta_f3','beta_g3'),
                function(variable) {as.numeric(subset(odraws4, variable=variable))})
#+ fit4, results='hide'
fit4 <- model4$sample(data=standata4, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=function() { init4 }, refresh=10)

#' Check whether parameters have reasonable values
draws4 <- fit4$draws()
summarise_draws(subset(draws4, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE))
summarise_draws(subset(draws4, variable=c('beta_f3')))

#' Compare the model to the data
draws4 <- as_draws_matrix(draws4)
Ef <- exp(apply(subset(draws4, variable='f'), 2, median))
Ef1 <- apply(subset(draws4, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws4, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws4, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef3 <- apply(subset(draws4, variable='f3'), 2, median)
Ef3 <- exp(Ef3 - mean(Ef3) + mean(log(data$births_relative100)))
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
pf3b <- data %>%
  mutate(Ef3 = Ef3) %>%
  ggplot(aes(x=date, y=births_relative100/Ef1/Ef2*100*100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
(pf + pf1) / (pf2 + pf3b)

#' The model fits well the different branches visible in plotted daily
#' relative number of births, that is, it is able to model the
#' increasing weekend effect.
#' 

#' ### Model 5: long term smooth + seasonal + weekday with time dependent magnitude + day of year RHS
#'
#' The next component to add is day of year effect. Many bank holidays
#' are every year on the same day of year and there might be also
#' other special days that are favored or disfavored.
#'
#' $$
#' f = \mbox{intercept} + f_1 + f_2 + \exp(g_3)\beta_{\mbox{day of week}} + \beta_{\mbox{day of year}}\\
#' \mbox{intercept} \sim \mbox{normal}(0,1)\\
#' f_1 \sim \mbox{GP}(0,K_1)\\
#' f_2 \sim \mbox{GP}(0,K_2)\\
#' g_3 \sim \mbox{GP}(0,K_3)\\
#' \beta_{\mbox{day of week}} = 0 \quad \mbox{if day of week is Monday}\\
#' \beta_{\mbox{day of week}} \sim \mbox{normal}(0,1) \quad \mbox{if day of week is not Monday}\\
#' \beta_{\mbox{day of year}} \sim RHS(0,0.1)
#' $$
#' As we assume that only some days of year are special, we use
#' regularized horseshoe (RHS) prior for day of year effects.
#'
#' At this point the optimization didn't produce reasonable result as
#' earlier and sampling turned out to be very slow. We assumed the
#' optimization fails because there were so many more parameters with
#' hierarchical prior. As even the short chain sampling would have
#' taken more than hour, it would have been time consuming to further
#' to test the model. As part of the quick iterative model building it
#' was better to give up on this model for a moment.
#'
#' Compile Stan model 5 [gpbf5.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf5.stan)
#+ model5, results='hide'
model5 <- cmdstan_model(stan_file = root("Birthdays", "gpbf5.stan"),
                        include_paths = root("Birthdays"))

#' Data to be passed to Stan
standata5 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=10,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  c_g3=1.5, # factor c of basis functions for GP for g3
                  M_g3=5,   # number of basis functions for GP for g3
                  scale_global=0.1, # gloval scale for RHS prior
                  day_of_week=data$day_of_week,
                  day_of_year=data$day_of_year2) # 1st March = 61 every year

#' Optimizing is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ opt5, results='hide'
opt5 <- model5$optimize(data=standata5, init=0, algorithm='lbfgs',
                        history=100, tol_obj=10)

#' Check whether parameters have reasonable values
odraws5 <- opt5$draws()
subset(odraws5, variable=c('intercept0','sigma_','lengthscale_','sigma'), regex=TRUE)
subset(odraws5, variable=c('beta_f3'))
Ef4 <- as.numeric(subset(odraws5, variable='beta_f4'))*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")

#' Compare the model to the data
Ef <- exp(as.numeric(subset(odraws5, variable='f')))
Ef1 <- as.numeric(subset(odraws5, variable='f1'))
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- as.numeric(subset(odraws5, variable='f2'))
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- as.numeric(subset(odraws5, variable='f_day_of_week'))
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef4 <- as.numeric(subset(odraws5, variable='beta_f4'))*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
pf2b <-data.frame(x=as.Date("1959-12-31")+1:366, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
(pf + pf1) / (pf2 + pf3) / (pf2b)

#' The quick model fit for model 5 is not good, but as the sampling
#' was very slow it wasn't easy to figure out what is going wrong.
#' 

#' ### Model 6: long term smooth + seasonal + weekday + day of year
#'
#' To simplify the analysis of the day of year effect and make the
#' inference during the exploration faster, we drop the time dependent
#' day of week effect and RHS for a moment and use normal prior for
#' the day of year effect.
#'
#' $$
#' f = \mbox{intercept} + f_1 + f_2 + \exp(g_3)\beta_{\mbox{day of week}} + \beta_{\mbox{day of year}}\\
#' \mbox{intercept} \sim \mbox{normal}(0,1)\\
#' f_1 \sim \mbox{GP}(0,K_1)\\
#' f_2 \sim \mbox{GP}(0,K_2)\\
#' g_3 \sim \mbox{GP}(0,K_3)\\
#' \beta_{\mbox{day of week}} = 0 \quad \mbox{if day of week is Monday}\\
#' \beta_{\mbox{day of week}} \sim \mbox{normal}(0,1) \quad \mbox{if day of week is not Monday}\\
#' \beta_{\mbox{day of year}} \sim \mbox{normal}(0,0.1)
#' $$
#' 
#' Compile Stan model 6 [gpbf6.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf6.stan)
#+ model6, results='hide'
model6 <- cmdstan_model(stan_file = root("Birthdays", "gpbf6.stan"),
                        include_paths = root("Birthdays"))

#' Data to be passed to Stan
standata6 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=10, # number of basis functions for GP for f1
                  J_f2=20, # number of basis functions for periodic f2
                  day_of_week=data$day_of_week,
                  day_of_year=data$day_of_year2) # 1st March = 61 every year

#' Optimizing is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ opt6, results='hide'
opt6 <- model6$optimize(data=standata6, init=0, algorithm='lbfgs',
                        history=100, tol_obj=10)

#' Check whether parameters have reasonable values
odraws6 <- opt6$draws()
subset(odraws6, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE)
subset(odraws6, variable=c('beta_f3'))
Ef4 <- as.numeric(subset(odraws6, variable='beta_f4'))*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
#' We recognize some familiar structure in the day of year effect and
#' proceed to sampling.

#' Sample short chains using the early stopped optimization result as
#' initial values (although the result from short chains can be useful
#' in a quick workflow, the result should not be used as the final
#' result).
init6 <- sapply(c('intercept0','lengthscale_f1','lengthscale_f2',
                  'sigma_f1','sigma_f2','sigma_f4','sigma',
                  'beta_f1','beta_f2','beta_f3','beta_f4'),
                function(variable) {as.numeric(subset(odraws6, variable=variable))})
#+ fit6, results='hide'
fit6 <- model6$sample(data=standata6, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=function() { init6 })

#' Check whether parameters have reasonable values
draws6 <- fit6$draws()
summarise_draws(subset(draws6, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE))
summarise_draws(subset(draws6, variable=c('beta_f3')))
draws6 <- as_draws_matrix(draws6)
Ef4 <- apply(subset(draws6, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")

#' Compare the model to the data
draws6 <- as_draws_matrix(draws6)
Ef <- exp(apply(subset(draws6, variable='f'), 2, median))
Ef1 <- apply(subset(draws6, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws6, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws6, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef4 <- apply(subset(draws6, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year") +
  annotate("text",x=as.Date("1988-01-01"),y=Ef4[1]-1,label="New year") +
  annotate("text",x=as.Date("1988-02-14"),y=Ef4[45]+1.5,label="Valentine's day") +
  annotate("text",x=as.Date("1988-02-29"),y=Ef4[60]-2.5,label="Leap day") +
  annotate("text",x=as.Date("1988-04-01"),y=Ef4[92]-1.5,label="April 1st") + 
  annotate("text",x=as.Date("1988-07-04"),y=Ef4[186]-1.5,label="Independence day") +
  annotate("text",x=as.Date("1988-10-31"),y=Ef4[305]-1.5,label="Halloween") + 
  annotate("text",x=as.Date("1988-12-24"),y=Ef4[360]-1.5,label="Christmas") +
  geom_point(data=f13,aes(x=date,y=y), size=3, shape=1)
(pf + pf1) / (pf2 + pf3) / pf2b

#' The short sampling result looks reasonable and thus the problem is
#' not in adding the day of year effect itself.  In the bottom plot,
#' the circles mark 13th day of each month. Results look similar to
#' our previous analyses , so it seems the day or year effect model
#' component is working as it should, but there was some problem with
#' our RHS implementation. As there is more variation in the day of
#' year effects than we would hope, we did some additional experiments
#' with different priors for the day of year effect (double
#' exponential, Cauchy and Student's t with unknown degrees of freedom
#' as models 6b, 6c, 6d), but decided it's better to add other
#' components before investing that part more thoroughly.
#' 

#' ### Model 7: long term smooth + seasonal + weekday + day of year normal + floating special days
#'
#' We can see in the model 6 results that day of year effects have
#' some dips in the relative number of births that are spread over a
#' week. From previous analyse we know these correspond to holidays
#' that are not on a specific day of year, but are for example on the
#' last Monday of May. We call these floating special days and include
#' Memorial day (last Monday of May), Labor day (first Monday of
#' September, and we include also the following Tuesday), and
#' Thanksgiving (fourth Thursday of November, and we include also the
#' following Friday).
#'
#' Compile Stan model 7 [gpbf7.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf7.stan)
#+ model7, results='hide'
model7 <- cmdstan_model(stan_file = root("Birthdays", "gpbf7.stan"),
                        include_paths = root("Birthdays"))

#' Floating special days
# Memorial day
memorial_days <- with(data,which(month==5&day_of_week==1&day>=25))
# Labor day
labor_days <- with(data,which(month==9&day_of_week==1&day<=7))
labor_days <- c(labor_days, labor_days+1)
# Thanksgiving
thanksgiving_days <- with(data,which(month==11&day_of_week==4&day>=22&day<=28))
thanksgiving_days <- c(thanksgiving_days, thanksgiving_days+1)

#' Data to be passed to Stan
standata7 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=10,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  day_of_week=data$day_of_week,
                  day_of_year=data$day_of_year2, # 1st March = 61 every year
                  memorial_days=memorial_days,
                  labor_days=labor_days,
                  thanksgiving_days=thanksgiving_days)

#' Optimizing is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ opt7, results='hide'
opt7 <- model7$optimize(data=standata7, init=0, algorithm='lbfgs',
                        history=100, tol_obj=10)

#' Check whether parameters have reasonable values
odraws7 <- opt7$draws()
subset(odraws7, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE)
subset(odraws7, variable=c('beta_f3'))
Ef4 <- as.numeric(subset(odraws7, variable='beta_f4'))*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")

#' Sample short chains using the early stopped optimization result as
#' initial values (although the result from short chains can be useful
#' in a quick workflow, the result should not be used as the final
#' result).
init7 <- sapply(c('intercept0','lengthscale_f1','lengthscale_f2',
                  'sigma_f1','sigma_f2','sigma_f4','sigma',
                  'beta_f1','beta_f2','beta_f3','beta_f4','beta_f5'),
                function(variable) {as.numeric(subset(odraws7, variable=variable))})
#+ fit7, results='hide'
fit7 <- model7$sample(data=standata7, iter_warmup=100, iter_sampling=100, chains=4, parallel_chains=4,
                      init=function() { init7 }, refresh=10)

#' Check whether parameters have reasonable values
draws7 <- fit7$draws()
summarise_draws(subset(draws7, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE))
summarise_draws(subset(draws7, variable=c('beta_f3')))

#' Compare the model to the data
draws7 <- as_draws_matrix(draws7)
Ef <- exp(apply(subset(draws7, variable='f'), 2, median))
Ef1 <- apply(subset(draws7, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws7, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws7, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef4 <- apply(subset(draws7, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
Efloats <- apply(subset(draws7, variable='beta_f5'), 2, median)*sd(log(data$births_relative100))
Efloats <- exp(Efloats)*100
floats1988<-c(memorial_days[20], labor_days[c(20,40)], thanksgiving_days[c(20,40)])-6939
Ef4float <- Ef4
Ef4float[floats1988] <- Ef4float[floats1988]*Efloats[c(1,2,2,3,3)]/100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)

pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year") +
  annotate("text",x=as.Date("1988-01-01"),y=Ef4float[1]-1,label="New year") +
  annotate("text",x=as.Date("1988-02-14"),y=Ef4float[45]+1.5,label="Valentine's day") +
  annotate("text",x=as.Date("1988-02-29"),y=Ef4float[60]-2.5,label="Leap day") +
  annotate("text",x=as.Date("1988-04-01"),y=Ef4float[92]-1.5,label="April 1st") + 
  annotate("text",x=as.Date("1988-07-04"),y=Ef4float[186]-1.5,label="Independence day") +
  annotate("text",x=as.Date("1988-10-31"),y=Ef4float[305]-1.5,label="Halloween") + 
  annotate("text",x=as.Date("1988-12-24"),y=Ef4float[360]-2,label="Christmas") +
  annotate("text",x=as.Date("1988-05-30"),y=Ef4float[151]-1.5,label="Memorial day") +
  annotate("text",x=as.Date("1988-09-05"),y=Ef4float[249]-1.5,label="Labor day") + 
  annotate("text",x=as.Date("1988-11-24"),y=Ef4float[329]-1,label="Thanksgiving")+
  geom_point(data=f13,aes(x=date,y=y), size=3, shape=1)
(pf + pf1) / (pf2 + pf3) / (pf2b)

#' The day of year and floating special day effects are shown for year
#' 1988 (which is also a leap year) and the results seem reasonable.
#' 

#' ### Model 8: long term smooth + seasonal + weekday with time dependent magnitude + day of year + special
#'
#' As the day of year and floating day effects work well, we'll add
#' the time dependent day of week effect back to the model.
#' 
#' Compile Stan model 8 [gpbf8.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf8.stan)
#+ model8, results='hide'
model8 <- cmdstan_model(stan_file = root("Birthdays", "gpbf8.stan"),
                        include_paths = root("Birthdays"))

#' Floating special days
# Memorial day
memorial_days <- with(data,which(month==5&day_of_week==1&day>=25))
# Labor day
labor_days <- with(data,which(month==9&day_of_week==1&day<=7))
labor_days <- c(labor_days, labor_days+1)
# Thanksgiving
thanksgiving_days <- with(data,which(month==11&day_of_week==4&day>=22&day<=28))
thanksgiving_days <- c(thanksgiving_days, thanksgiving_days+1)

#' Data to be passed to Stan
standata8 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=10,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  c_g3=1.5, # factor c of basis functions for GP for g3
                  M_g3=5,   # number of basis functions for GP for g3
                  day_of_week=data$day_of_week,
                  day_of_year=data$day_of_year2, # 1st March = 61 every year
                  memorial_days=memorial_days,
                  labor_days=labor_days,
                  thanksgiving_days=thanksgiving_days)

#' Optimizing is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ opt8, results='hide'
opt8 <- model8$optimize(data=standata8, init=0.1, algorithm='lbfgs',
                        history=100, tol_obj=10)

#' Check whether parameters have reasonable values
odraws8 <- opt8$draws()
subset(odraws8, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE)
subset(odraws8, variable=c('beta_f3'))
Ef4 <- as.numeric(subset(odraws8, variable='beta_f4'))*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")

#' Compare the model to the data
Ef <- exp(as.numeric(subset(odraws8, variable='f')))
Ef1 <- as.numeric(subset(odraws8, variable='f1'))
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- as.numeric(subset(odraws8, variable='f2'))
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- as.numeric(subset(odraws8, variable='f_day_of_week'))
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef3 <- as.numeric(subset(odraws8, variable='f3'))
Ef3 <- exp(Ef3 - mean(Ef3) + mean(log(data$births_relative100)))
Ef4 <- as.numeric(subset(odraws8, variable='beta_f4'))*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
Efloats <- as.numeric(subset(odraws8, variable='beta_f5'))*sd(log(data$births_relative100))
Efloats <- exp(Efloats)*100
floats1988<-c(memorial_days[20], labor_days[c(20,40)], thanksgiving_days[c(20,40)])-6939
Ef4float <- Ef4
Ef4float[floats1988] <- Ef4float[floats1988]*Efloats[c(1,2,2,3,3)]/100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef), color=set1[1], alpha=0.2) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
N=length(data$id)
pf3b <- data %>%
  mutate(Ef3 = Ef3*Ef1/100) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births") +
  annotate("text",x=as.Date("1989-08-01"),y=(Ef3*Ef1/100)[c((N-5):(N-4), N, N-6)],label=c("Mon","Tue","Sat","Sun"))
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year") +
  annotate("text",x=as.Date("1988-01-01"),y=Ef4float[1]-1,label="New year") +
  annotate("text",x=as.Date("1988-02-14"),y=Ef4float[45]+1.5,label="Valentine's day") +
  annotate("text",x=as.Date("1988-02-29"),y=Ef4float[60]-2.5,label="Leap day") +
  annotate("text",x=as.Date("1988-04-01"),y=Ef4float[92]-1.5,label="April 1st") + 
  annotate("text",x=as.Date("1988-07-04"),y=Ef4float[186]-1.5,label="Independence day") +
  annotate("text",x=as.Date("1988-10-31"),y=Ef4float[305]-1.5,label="Halloween") + 
  annotate("text",x=as.Date("1988-12-24"),y=Ef4float[360]-2,label="Christmas") +
  annotate("text",x=as.Date("1988-05-30"),y=Ef4float[151]-2,label="Memorial day") +
  annotate("text",x=as.Date("1988-09-05"),y=Ef4float[249]-1.5,label="Labor day") + 
  annotate("text",x=as.Date("1988-11-24"),y=Ef4float[329]-1,label="Thanksgiving")+
  geom_point(data=f13,aes(x=date,y=y), size=3, shape=1)
(pf + pf1) / (pf2 + pf3b) / (pf2b)

#' Sample short chains using the early stopped optimization result as
#' initial values (although the result from short chains can be useful
#' in a quick workflow, the result should not be used as the final
#' result).
init8 <- sapply(c('intercept0','lengthscale_f1','lengthscale_f2','lengthscale_g3',
                  'sigma_f1','sigma_f2','sigma_g3','sigma_f4','sigma',
                  'beta_f1','beta_f2','beta_f3','beta_g3','beta_f4','beta_f5'),
                function(variable) {as.numeric(subset(odraws8, variable=variable))})
#+ fit8, results='hide'
fit8 <- model8$sample(data=standata8, iter_warmup=100, iter_sampling=100, chains=4, parallel_chains=4,
                      init=function() { init8 }, refresh=10)

#' Check whether parameters have reasonable values
draws8 <- fit8$draws()
summarise_draws(subset(draws8, variable=c('intercept','sigma_','lengthscale_','sigma'), regex=TRUE))
summarise_draws(subset(draws8, variable=c('beta_f3')))

#' Compare the model to the data
draws8 <- as_draws_matrix(draws8)
Ef <- exp(apply(subset(draws8, variable='f'), 2, median))
Ef1 <- apply(subset(draws8, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws8, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws8, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef3 <- apply(subset(draws8, variable='f3'), 2, median)
Ef3 <- exp(Ef3 - mean(Ef3) + mean(log(data$births_relative100)))
Ef4 <- apply(subset(draws8, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
Efloats <- apply(subset(draws8, variable='beta_f5'), 2, median)*sd(log(data$births_relative100))
Efloats <- exp(Efloats)*100
floats1988<-c(memorial_days[20], labor_days[c(20,40)], thanksgiving_days[c(20,40)])-6939
Ef4float <- Ef4
Ef4float[floats1988] <- Ef4float[floats1988]*Efloats[c(1,2,2,3,3)]/100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef), color=set1[1], alpha=0.2) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
N=length(data$id)
pf3b <- data %>%
  mutate(Ef3 = Ef3*Ef1/100) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births") +
  annotate("text",x=as.Date("1989-08-01"),y=(Ef3*Ef1/100)[c((N-5):(N-4), N, N-6)],label=c("Mon","Tue","Sat","Sun"))
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year") +
  annotate("text",x=as.Date("1988-01-01"),y=Ef4float[1]-1,label="New year") +
  annotate("text",x=as.Date("1988-02-14"),y=Ef4float[45]+1.5,label="Valentine's day") +
  annotate("text",x=as.Date("1988-02-29"),y=Ef4float[60]-2.5,label="Leap day") +
  annotate("text",x=as.Date("1988-04-01"),y=Ef4float[92]-1.5,label="April 1st") + 
  annotate("text",x=as.Date("1988-07-04"),y=Ef4float[186]-1.5,label="Independence day") +
  annotate("text",x=as.Date("1988-10-31"),y=Ef4float[305]-1.5,label="Halloween") + 
  annotate("text",x=as.Date("1988-12-24"),y=Ef4float[360]-2,label="Christmas") +
  annotate("text",x=as.Date("1988-05-30"),y=Ef4float[151]-2,label="Memorial day") +
  annotate("text",x=as.Date("1988-09-05"),y=Ef4float[249]-1.5,label="Labor day") + 
  annotate("text",x=as.Date("1988-11-24"),y=Ef4float[329]-1,label="Thanksgiving")+
  geom_point(data=f13,aes(x=date,y=y), size=3, shape=1)
(pf + pf1) / (pf2 + pf3b) / (pf2b)

#' The inference for the model works fine, which hints that our RHS
#' implementation for the model 5 was wrong or had very difficult
#' posterior. Before testing RHS again, we'll test with an easier to
#' implement Student's $t$ prior whether long tailed prior for day of
#' year effect is reasonable. These experiments help also to find out
#' whether the day of year effect is sensitive to the prior choice.
#'
#'
#' ### Model 8+t_nu: day of year effect with Student's t prior
#' 
#' Compile Stan model 8 + t_nu [gpbf8tnu.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf8tnu.stan)
#+ model8tnu, results='hide'
model8tnu <- cmdstan_model(stan_file = root("Birthdays", "gpbf8tnu.stan"),
                           include_paths = root("Birthdays"))

#' Optimizing is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ opt8tnu, results='hide'
opt8tnu <- model8tnu$optimize(data=standata8, init=0.1, algorithm='lbfgs',
                              history=100, tol_obj=10)
odraws8tnu <- opt8tnu$draws()

#' Sample short chains using the early stopped optimization result as
#' initial values (although the result from short chains can be useful
#' in a quick workflow, the result should not be used as the final
#' result).
init8tnu <- sapply(c('intercept0','lengthscale_f1','lengthscale_f2','lengthscale_g3',
                  'sigma_f1','sigma_f2','sigma_g3','sigma_f4','nu_f4','sigma',
                  'beta_f1','beta_f2','beta_f3','beta_g3','beta_f4','beta_f5'),
                function(variable) {as.numeric(subset(odraws8tnu, variable=variable))})
#+ fit8tnu, results='hide'
fit8tnu <- model8tnu$sample(data=standata8, iter_warmup=100, iter_sampling=100,
                            chains=4, parallel_chains=4,
                      init=function() { init8tnu }, refresh=10)

#' Check whether parameters have reasonable values
draws8tnu <- fit8tnu$draws()
summarise_draws(subset(draws8tnu, variable=c('intercept','sigma_','lengthscale_','sigma','nu_'), regex=TRUE))
#' Posterior of degrees of freedom `nu_f4` is very close to 1, and
#' thus the distribution is very close to Cauchy. This is strong
#' evidence that the distribution of day of year effects is far from
#' normal, even if Cauchy would not be the correct distribution.

#' Compare the model to the data
draws8 <- as_draws_matrix(draws8tnu)
Ef <- exp(apply(subset(draws8, variable='f'), 2, median))
Ef1 <- apply(subset(draws8, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws8, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws8, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef3 <- apply(subset(draws8, variable='f3'), 2, median)
Ef3 <- exp(Ef3 - mean(Ef3) + mean(log(data$births_relative100)))
Ef4 <- apply(subset(draws8, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
Efloats <- apply(subset(draws8, variable='beta_f5'), 2, median)*sd(log(data$births_relative100))
Efloats <- exp(Efloats)*100
floats1988<-c(memorial_days[20], labor_days[c(20,40)], thanksgiving_days[c(20,40)])-6939
Ef4float <- Ef4
Ef4float[floats1988] <- Ef4float[floats1988]*Efloats[c(1,2,2,3,3)]/100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef), color=set1[1], alpha=0.2) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
N=length(data$id)
pf3b <- data %>%
  mutate(Ef3 = Ef3*Ef1/100) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births") +
  annotate("text",x=as.Date("1989-08-01"),y=(Ef3*Ef1/100)[c((N-5):(N-4), N, N-6)],label=c("Mon","Tue","Sat","Sun"))
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)

pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year") +
  annotate("text",x=as.Date("1988-01-01"),y=Ef4float[1]-1,label="New year") +
  annotate("text",x=as.Date("1988-02-14"),y=Ef4float[45]+1.5,label="Valentine's day") +
  annotate("text",x=as.Date("1988-02-29"),y=Ef4float[60]-2.5,label="Leap day") +
  annotate("text",x=as.Date("1988-04-01"),y=Ef4float[92]-1.5,label="April 1st") + 
  annotate("text",x=as.Date("1988-07-04"),y=Ef4float[186]-1.5,label="Independence day") +
  annotate("text",x=as.Date("1988-10-31"),y=Ef4float[305]-1.5,label="Halloween") + 
  annotate("text",x=as.Date("1988-12-24"),y=Ef4float[360]-2,label="Christmas") +
  annotate("text",x=as.Date("1988-05-30"),y=Ef4float[151]-2,label="Memorial day") +
  annotate("text",x=as.Date("1988-09-05"),y=Ef4float[249]-1.5,label="Labor day") + 
  annotate("text",x=as.Date("1988-11-24"),y=Ef4float[329]-1,label="Thanksgiving")+
  geom_point(data=f13,aes(x=date,y=y), size=3, shape=1)
(pf + pf1) / (pf2 + pf3b) / (pf2b)

#' The other effects seem to be quite similar as with the previous
#' model, but the day of year effects are clearly different with most
#' days having non-detectable effect. There are also effects that
#' seemed to be quite clear in normal prior model such as 13th day of
#' month effect, which is not visible anymore. As the posterior of
#' degrees of freedom `t_nu` was concentrated close to 1, it's likely
#' that the normal prior for day of year effect can't be the best. So
#' far we hadn't used model comparison such as leave-one-out
#' cross-validation (LOO-CV) as each added component had qualitatively
#' big and reasonable effect. Now as day of year effect is sensitive
#' to prior choice, but it's not clear how much better Cauchy prior
#' distribution is we use LOO-CV to compare the models.
loo8 <- fit8$loo()
loo8tnu <- fit8tnu$loo()
loo_compare(list(`Model 8 normal`=loo8,`Model 8 Student\'s t`=loo8tnu))
#' As we could have expected based on the posterior of `nu_f4`
#' Student's t prior on day of year effects is better. As low degrees
#' of freedom indicate a thick tailed distribution for day of year
#' effect is needed, we decided to test again RHS prior.
#' 

#' ### Model 8+RHS: day of year effect with RHS prior
#'
#' Model 5 had RHS prior but the problem was that optimization result
#' wasn't even close to sensible and MCMC was very slow. Given the
#' other models we now know that the problem is not in adding day of
#' year effect or combining it with time dependent magnitude for the
#' day of week effect. It was easier now to focus on figuring out the
#' problem in RHS. Since RHS is presented as a scale mixture of
#' normals involving hierarchical prior, it is common to use
#' non-centered parameterization for RHS prior. Non-centered
#' parameterization is useful when the information from the likelihood
#' is weak and the prior dependency dominates in the posterior
#' dependency. RHS is often used when there are less observations than
#' unknowns. In this problem each unknown (one day of year effect) is
#' informed by several observations from different years, and then it
#' might be that the centered parameterization is better. And this
#' turned out to be true and the inference for model 8 with centered
#' parameterization RHS prior on day of year effect worked much better
#' than for model 5.  (In Stan it was easy to test switch from
#' non-centered to centered parameterization by removing the multplier
#' from one of the parameter declarations).
#'
#' Compile Stan model 8 + RHS [gpbf8rhs.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf8rhs.stan)
#+ model8rhs, results='hide'
model8rhs <- cmdstan_model(stan_file = root("Birthdays", "gpbf8rhs.stan"),
                           include_paths = root("Birthdays"))

#' Add a global scale for RHS prior
standata8 <- c(standata8,
               scale_global=0.1) # global scale for RHS prior

#' Optimizing is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ opt8rhs, results='hide'
opt8rhs <- model8rhs$optimize(data=standata8, init=0.1, algorithm='lbfgs',
                              history=100, tol_obj=10)
odraws8rhs <- opt8rhs$draws()

#' Sample short chains using the optimization result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
init8rhs <- sapply(c('intercept0','lengthscale_f1','lengthscale_f2','lengthscale_g3',
                  'sigma_f1','sigma_f2','sigma_g3','sigma_f4','sigma',
                  'beta_f1','beta_f2','beta_f3','beta_g3','beta_f4','beta_f5',
                  'tau_f4','lambda_f4','caux_f4'),
                function(variable) {as.numeric(subset(odraws8rhs, variable=variable))})
#+ fit8rhs, results='hide'
fit8rhs <- model8rhs$sample(data=standata8, iter_warmup=100, iter_sampling=100, chains=4, parallel_chains=4,
                            init=function() { init8rhs }, refresh=10)

#' Check whether parameters have reasonable values
draws8rhs <- fit8rhs$draws()
summarise_draws(subset(draws8rhs, variable=c('intercept','sigma_','lengthscale_','sigma','nu_'), regex=TRUE))

#' Compare the model to the data
draws8 <- as_draws_matrix(draws8rhs)
Ef <- exp(apply(subset(draws8, variable='f'), 2, median))
Ef1 <- apply(subset(draws8, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws8, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws8, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef3 <- apply(subset(draws8, variable='f3'), 2, median)
Ef3 <- exp(Ef3 - mean(Ef3) + mean(log(data$births_relative100)))
Ef4 <- apply(subset(draws8, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
Efloats <- apply(subset(draws8, variable='beta_f5'), 2, median)*sd(log(data$births_relative100))
Efloats <- exp(Efloats)*100
floats1988<-c(memorial_days[20], labor_days[c(20,40)], thanksgiving_days[c(20,40)])-6939
Ef4float <- Ef4
Ef4float[floats1988] <- Ef4float[floats1988]*Efloats[c(1,2,2,3,3)]/100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef), color=set1[1], alpha=0.2) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
N=length(data$id)
pf3b <- data %>%
  mutate(Ef3 = Ef3*Ef1/100) %>%
  ggplot(aes(x=date, y=births_relative100)) + geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births") +
  annotate("text",x=as.Date("1989-08-01"),y=(Ef3*Ef1/100)[c((N-5):(N-4), N, N-6)],label=c("Mon","Tue","Sat","Sun"))
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)

pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) + geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year") +
  annotate("text",x=as.Date("1988-01-01"),y=Ef4float[1]-1,label="New year") +
  annotate("text",x=as.Date("1988-02-14"),y=Ef4float[45]+1.5,label="Valentine's day") +
  annotate("text",x=as.Date("1988-02-29"),y=Ef4float[60]-2.5,label="Leap day") +
  annotate("text",x=as.Date("1988-04-01"),y=Ef4float[92]-1.5,label="April 1st") + 
  annotate("text",x=as.Date("1988-07-04"),y=Ef4float[186]-1.5,label="Independence day") +
  annotate("text",x=as.Date("1988-10-31"),y=Ef4float[305]-1.5,label="Halloween") + 
  annotate("text",x=as.Date("1988-12-24"),y=Ef4float[360]-2,label="Christmas") +
  annotate("text",x=as.Date("1988-05-30"),y=Ef4float[151]-2,label="Memorial day") +
  annotate("text",x=as.Date("1988-09-05"),y=Ef4float[249]-1.5,label="Labor day") + 
  annotate("text",x=as.Date("1988-11-24"),y=Ef4float[329]-1,label="Thanksgiving")+
  geom_point(data=f13,aes(x=date,y=y), size=3, shape=1)
(pf + pf1) / (pf2 + pf3b) / (pf2b)

#' Visually we get quite similar result as with Cauchy prior. When we
#' compare the models with LOO-CV, Cauchy is favored instead of RHS.
loo8rhs<-fit8rhs$loo()
loo_compare(list(`Model 8 Students t`=loo8tnu,`Model 8 RHS`=loo8rhs))
#' If we look at the LOO-stacking model weights, the predictive
#' performance can be improved by combining Cauchy and RHS priors on
#' day of year effect which indicates that neither of them is very
#' close to true distribution.
loo_model_weights(list(`Model 8 Students t`=loo8tnu,`Model 8 RHS`=loo8rhs))

#' ### Further improvements for the day of year effect
#' 
#' It's unlikely that day of year effect would be unstructured with
#' some distribution like Cauchy, and thus instead of trying to find a
#' prior distribution that would improve LOO-CV, it would make more
#' sense to further add structural information. For example, it would
#' be possible to add more known special days and take into account
#' that a special day effect and weekend effect probably are not
#' additive. Furthermore if there are less births during some day, the
#' births need to happen some other day and it can be assumed that
#' there would be corresponding excess of births before of after a
#' bank holiday. This ringing around days with less births is not
#' simple as it is also affected whether the previous and following
#' days are weekend days. This all gets more complicated than we want
#' to include in this case study, but the reader can see how the
#' similar gradual model building could be made by adding additional
#' components. Eventually it is likely that there starts to be worry
#' of overfitting, but integration over the unknown alleviates that
#' and looking at the predictive performance estimates such LOO-CV can
#' help to decide when the additional model components don't improve
#' the predictive performance or can't be well identified.
#'
#' ### Quantitative predictive performance for the series of models
#' 
#' We didn't use LOO-CV until in the end, as the qualitative
#' differences between models were very convincing. We can use LOO-CV
#' to check how big the difference in the predictive performance are
#' and if the differences are big, we know that model averaging that
#' would take into account the uncertainty would give weights close to
#' zero for all but the most elaborate models.
loo1<-fit1$loo()
loo2<-fit2$loo()
loo3<-fit3$loo()
loo4<-fit4$loo()
loo6<-fit6$loo()
loo7<-fit7$loo()
loo_compare(list(`Model 1`=loo1,`Model 2`=loo2,`Model 3`=loo3,`Model 4`=loo4,`Model 6`=loo6,`Model 7`=loo7,`Model 8 + t_nu`=loo8tnu))

#' ### Residual analysis
#' 
#' We can get further ideas for how to improve the model also by
#' looking at the residuals.
draws8 <- as_draws_matrix(draws8tnu)
Ef <- exp(apply(subset(draws8, variable='f'), 2, median))
data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=log(births_relative100/Ef))) + geom_point(color=set1[2]) +
  geom_hline(yintercept=0, color='gray') +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme(panel.grid.major.x=element_line(color='gray',size=1))
#' We can see some structure, specifically in years 1969--1978 the
#' residual has negative peak in the middle of the year, while in years
#' 1981--1988 the residual has positive peak in the middle of the
#' year. This kind of pattern appears as we use the same seasonal
#' effect for all years, but the magnitude of seasonal effect is
#' changing. It would be possible to modify the model to include
#' gradually changing seasonal effect, but leave it out from this case
#' study. 
#'
#' The best model so far explains already 94% of the variance (LOO-R2).
draws8 <- as_draws_matrix(draws8tnu)
f <- exp(subset(draws8, variable='f'))
loo8tnu <- fit8tnu$loo(save_psis=TRUE)
Efloo <- E_loo(f, psis_object=loo8tnu$psis_object)$value
LOOR2 <- 1-var(log(data$births_relative100/Efloo))/var(log(data$births_relative100))
print(LOOR2, digits=2)
#' As it seems we could still improve by adding more structure and
#' time varying seasonal effect, it seems the variability in the
#' number of births from day to day is quite well predictable. Of
#' course big part of the variation is due to planned induced births
#' and c-sections, and thus hospitals do already control the number of
#' births per day and there is no really practical use for the
#' result. However there are plenty of similar time series, for
#' example, in consumer behavior that are affected by special days.
#'
#' ### More accurate inference
#' 
#' During all the iterative model building we favored optimization and
#' short MCMC chains. In the end we also run with higher adapt_delta
#' to reduce the probability of divergences, higher maximum treedepth
#' to ensure higher effective sample size per iteration (ESS per
#' second doesn't necessarily improve), and run much longer chains,
#' but didn't see practical differences in plots or LOO-CV values. As
#' running these longer chains can take hours they are not run as part
#' of this notebook. An example of how to reduce probability of
#' divergences and increase maximum treedepth is shown below (there is
#' rarely need to increase adapt_delta larger than 0.95 and if there
#' are still divergences with adapt_delta equal to 0.99, the posterior
#' has serious problems and it should be considered whether
#' re-parameterization, better data or more informative priors could
#' help).
## fit8tnu <- model8tnu$sample(data=standata8, chains=4, parallel_chains=4,
##                             adapt_delta=0.95, max_treedepth=15)
