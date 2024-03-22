#' ---
#' title: "Birthdays workflow example"
#' author: "Aki Vehtari"
#' date: "First version 2020-12-28. Last modified `r format(Sys.Date())`."
#' output:
#'   html_document:
#'     theme: readable
#'     toc: true
#'     toc_depth: 3
#'     toc_float: true
#'     code_download: true
#' bibliography: ../casestudies.bib
#' csl: ../harvard-cite-them-right.csl
#' link-citations: yes
#' ---

#' Workflow example for iterative building of a time series model.
#'
#' We analyse the relative number of births per day in USA 1969-1988
#' using Gaussian process time series model with several model
#' components that can explain the long term, seasonal, weekly, day of
#' year, and special floating day variation. We use Pahtfinder
#' algorithm [@Zhang+etal:2022:pathfinder] to quickly check that model
#' code produces something reasonable and to initialize MCMC sampling.
#'
#' At the time of writing this 2023-11-16, Stan's multi-Pathfinder
#' implementation had an underflow bug, and to go around it, multiple
#' calls to single-Pathfinder are used here. Furthermore, this allows
#' us to use resampling without replacement to get unique
#' initialization values.
#'
#' Stan model codes are available in [the corresponding git repo](https://github.com/avehtari/casestudies/tree/master/Birthdays)
#'
#' -------------
#' 

#+ setup, include=FALSE
nitr::opts_chunk$set(message=FALSE, error=FALSE, warning=FALSE, comment=NA, cache=FALSE)
# switch this to TRUE to save figures in separate files
savefigs <- FALSE

#' #### Load packages
library("rprojroot")
root<-has_file(".Workflow-Examples-root")$make_fix_file()
library(tidyverse)
library(tictoc)
mytoc <- \() {toc(func.toc=\(tic,toc,msg) { sprintf("%s took %s sec",msg,as.character(signif(toc-tic,2))) })}
library(cmdstanr)
options(stanc.allow_optimizations = TRUE)
library(posterior)
options(pillar.neg = FALSE, pillar.subtle=FALSE, pillar.sigfig=2)
library(loo)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))
library(patchwork)
set1 <- RColorBrewer::brewer.pal(7, "Set1")
#' Use English for names of weekdays and months
Sys.setlocale("LC_TIME", "en_GB.UTF-8")

#' Function to form a list of list of initial values from a draws object.
#' Something like this will be eventually available in `cmdstanr` package.
as_inits <- function(draws, variable=NULL, ndraws=4) {
  ndraws <- min(ndraws(draws),ndraws)
  if (is.null(draws)) {variable = variables(draws)}
  inits <- lapply(1:ndraws,
                  function(drawid) {
                    sapply(variable,
                           function(var) {
                             as.numeric(subset_draws(draws, variable=var, draw=drawid))
                           })
                  })
  if (ndraws==1) { inits[[1]] } else { inits }
}

#' Function to form a list of list of initial values from a Pathfinder object.
#' Something like this will be eventually available in `cmdstanr` package.
create_inits <- function(pthfs, variables=NULL, ndraws=4) {
  if (is.list(pthfs)) {
    pthf <- pthfs[[1]]
    draws <- do.call(bind_draws, c(lapply(pthfs, as_draws), along='draw'))
  } else {
    pthf <- pthfs
    draws <- pthf$draws()
  }
  draws <- draws |>
    mutate_variables(lw=lp__-lp_approx__,
                     w=exp(lw-max(lw)),
                     ws=pareto_smooth(w, tail='right', r_eff=1)$x)
  if (is.null(variables)) {
    variables <- names(pthf$variable_skeleton(transformed_parameters = FALSE,
                                              generated_quantities = FALSE))
  }
  draws |>
    weight_draws(weights=extract_variable(draws,"ws"), log=FALSE) |>
    resample_draws(ndraws=ndraws, method = "simple_no_replace") |>
    as_inits(variable=variables, ndraws=ndraws)
}

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
  ggplot(aes(x=date, y=births)) +
  geom_point(color=set1[2]) +
  labs(x="Date", y="Relative number of births")

#' ### Plot all births as relative to mean
#'
#' To make the interpretation we switch to examine the relative
#' change, with the mean level denoted with 100.
data %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2]) +
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
#' We had analysed the same data before (see @BDA3) and thus we
#' already had an idea of what kind of model to use. Previously we
#' used GPstuff software which is Gaussian process specific software
#' for Matlab and Octave. As Stan has aimed to be very generic it can
#' be slower than specialized software for some specific models such
#' as Gaussian processes, but Stan provides more flexibility in the
#' model definition.
#'
#' @Riutort-Mayol:2023:HSGP demonstrate Hilbert space approximate
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
#' testing and iterative model building we can use optimization,
#' Pathfinder [@Zhang+etal:2022:pathfinder] and shorter MCMC chains
#' that we would not recommend for the final inference.  Furthermore,
#' in this specific example, the new additions are qualitatively so
#' clear improvements that there is no need for quantitative model
#' comparison whether the additions are ``significant'' (see also
#' @Navarro:2019:between) and there is no danger of overfitting (see
#' also @McLatchie+Vehtari:2023). Although there is one part of the
#' model where the data is weakly informative and the prior choices
#' seem to matter and we'll get back to this and consequences
#' later. Overall we build tens of different models, but illustrate
#' here only the main line.
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
#' In this phase, the code by @Riutort-Mayol:2023:HSGP was cleaned
#' and written to be more efficient, but only the one GP component was
#' included to make the testing easier. Although the code was made
#' more efficient, the aim wasn't to make it the fastest possible, as
#' the later model changes may have bigger effect on the performance
#' (it's good to avoid premature optimization). We also use quite small
#' number of basis functions to make the code run faster, and only
#' later examine more carefully whether the number of basis function
#' is sufficient compared to the posterior of the length scale (see,
#' @Riutort-Mayol:2023:HSGP).
#'

#' Compile Stan model [gpbf1.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf1.stan) which includes [gpbasisfun_functions1.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions1.stan). The option `compile_model_methods=TRUE` is used to be able get the parameter names from the compiled model, to improve creation of initial values for MCMC.
#+ model1, results='hide'
tic("Compilation of model 1")
model1 <- cmdstan_model(stan_file = root("Birthdays", "gpbf1.stan"),
                        include_paths = root("Birthdays"),
                        compile_model_methods=TRUE, force_recompile=TRUE)
#+
mytoc()

#' Data to be passed to Stan
standata1 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=20)  # number of basis functions for GP for f1

#' In this simplest model with just one GP, and as the basis function
#' approximation and priors restrict the complexity of GP, we can
#' safely use optimization to find maximum a posteriori (MAP) estimate
#' get a very quick initial result to check that the model code is
#' computing what we intended (e.g. no NaN, Infs, or non-sensical
#' results). As there are only 14 parameters and 7305 observations
#' it's likely that the posterior in the unconstrained parameter space
#' is close to normal. To obain the correct mode in the unconstrained
#' space, we need to call Stan optimizer with option `jacobian=TRUE`
#' (see [Laplace and Jacobian case
#' study](https://users.aalto.fi/~ave/casestudies/Jacobian/jacobian.html)
#' for illustration). Initialization at 0 in unconstrained space is
#' good for most GP models. In this case the optimization takes less
#' than one second while MCMC sampling with default options would have
#' taken several minutes. Although this result can be useful in a
#' quick workflow, the result should not be used as the final result.
#' #+ opt1, results='hide'
tic('Finding MAP for model 1 with optimization')
opt1 <- model1$optimize(data = standata1, init=0, algorithm='bfgs',
                        jacobian=TRUE)
#+
mytoc()

#' Check whether parameters have reasonable values
odraws1 <- opt1$draws()
subset(odraws1, variable=c('intercept','sigma_f1','lengthscale_f1','sigma'))

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

#' We can obtaina a bit more information by making a normal
#' approximation at the mode in the unconstrained parameter space. As
#' Laplace ws the first one to use this, the method is commonly called
#' Laplace method. Stan samples from the normal approximation in the
#' unconstrained space and transforms the obtained draws to the
#' constrained space. Stan's Laplace method uses `jacobian=TRUE` by
#' default. As we did already optimize, we can pass the optimization
#' result to the Laplace method. With additional 2s we get 400
#' approximate draws.
#+ lap1, results='hide'
tic('Sampling from Laplace approximation of model 1 posterior')
lap1 <- model1$laplace(data = standata1, mode=opt1, draws=400)
#+
mytoc()

#' Check whether parameters have reasonable values. With Laplace method, we get
#' also some information about the uncertainty in the posterior.
ldraws1 <- lap1$draws()
summarise_draws(subset(ldraws1, variable=c('intercept','sigma_f1','lengthscale_f1','sigma')),
                default_summary_measures())

#' At the moment, the Laplace method doesn't automatically run
#' diagnostic to assess the quality of the normal approximation, but
#' we can do it manually by checking the Pareto-$\hat{k}$ diagnostic for
#' the importance sampling weights if the normal approximation would be used as
#' a proposal distribution (see [Vehtari et al., 2022}(https://arxiv.org/abs/1507.02646v8)).
ldraws1 |>
  mutate_variables(lw = lp__-lp_approx__, w=exp(lw-max(lw))) |>
  subset_draws(variable="w") |>
  summarise_draws(pareto_khat, .args = list(tail='right', extra_diags = TRUE))

#' Here `khat` is larger than 0.7 indicating that importance sampling
#' even with Pareto smoothing is not able to provide accurate
#' adjustment. `min_ss` indicates how many draws would be needed to
#' get an accurate importance weighting adjustment, and in this that
#' number is impractically big. Even the Laplace approximation can be
#' useful, this diagnostic shows that we would eventually want to run
#' MCMC for more accurate inference.
#' 

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
#' not be used as the final result.
#+ fit1, results='hide'
tic('MCMC sampling from model 1 posterior')
fit1 <- model1$sample(data=standata1, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4, seed=3896)
#+
mytoc()

#' Depending on the random seed and luck, we sometimes observed that
#' some of the chains got stuck in different modes. We could see this
#' in high Rhat and low ESS diagnostic values. When updating this
#' case study, we didn't see multimodality with a few different
#' seeds, but you can see such an example in [Illustration of simple
#' problematic posteriors case study](https://users.aalto.fi/~ave/casestudies/Problems/problems.html)
#' 

#' We can reduce the possibility of getting stuck in minor modes and
#' improve the warmup by using Pathfinder algorithm. Pathfinder runs
#' several optimizations, but chooses a normal approximation along the
#' optimization path that minimizes ``exlusive''-Kullback-Leibler
#' distance from the approximation to the target posterior. Pathfinder
#' is better than Laplace for highly skewed and funnel like posteriors
#' which are typical for hierarchical model. We get 400 draws from
#' 10 Pathfinder runs.
#' pth1, results='hide'
tic('Sampling from Pathfinder approximation of model 1 posterior')
# Multi-Pathfinder is broken
#pth1 <- model1$pathfinder(data = standata1, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=100, max_lbfgs_iters=100)
pth1s=list()
for (i in 1:10) {
  pth1s[[i]] <- model1$pathfinder(data = standata1, init=0.1, num_paths=1, single_path_draws=40,
                                  history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Pathfinder provides automatically Pareto-$\hat{k}$ diagnostic which
#' is high, indicating the normal approximation is not good. The
#' Pathfinder draws do have reasonable values and we get also some
#' information about the uncertainty in the posterior (as with Laplace
#' method). We use `default_summary_measures()` as the MCMC diagnostics
#' are not useful for Pathfinder draws.
#pdraws1 <- pth1$draws()
pdraws1 <- do.call(bind_draws, c(lapply(pth1s, as_draws), along='draw'))
summarise_draws(subset(pdraws1, variable=c('intercept','sigma_f1','lengthscale_f1','sigma')),
                default_summary_measures())

#' The Pathfinder draws are likely to be closer to where most of the
#' posterior mass is than the default Stan initialization using
#' uniform random draws from -2 to 2 (in unconstrained space). 
#init1 <- create_inits(pth1)
init1 <- create_inits(pth1s)
#+ fit1init, results='hide'
tic('MCMC sampling from model 1 posterior with Pathfinder initialization')
fit1 <- model1$sample(data=standata1, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=init1)
#+
mytoc()

#' In many of the following short MCMC samplings we get some or many
#' divergences and usually very large number of treedepth
#' exceedences. Divergences indicate possible bias and should be
#' eventually investigated carefully. Treedepth exceedences indicate
#' strong posterior dependencies and slow mixing and sometimes the
#' posterior can be much improved by changing the parameterization or
#' priors, but as the treedepth exceedences don't indicate bias there
#' is no need for more careful analysis if the resulting ESS and MCSE
#' values are good for the purpose in hand.  We'll come back later to
#' more careful analysis of the final models.
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")

#' If we compare the result from short sampling to optimizing, we
#' don't see practical difference in the predictions (although we see
#' later more differences between optimization and MCMC).
data %>%
  mutate(Ef = Ef,
         oEf = oEf) %>%
  ggplot(aes(x=Ef, y=oEf)) +
  geom_point(color=set1[2]) +
  geom_abline() +
  labs(x="Ef from short Markov chain", y="Ef from optimizing")

#' After the first version of this notebook, [Nikolas Siccha examined
#' more carefully the posterior
#' correlations](https://github.com/nsiccha/birthday) and noticed
#' strong correlation between intercept and the first basis
#' function. Stan's dynamic HMC is so efficient that the inference is
#' succesful anyway. Nikolas suggested removing the intercept
#' term. The intercept term is not necessarily needed as the data has
#' been centered. We test a model without the explicit intercept term.
#'
#' Compile Stan model [gpbf1b.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf1b.stan)
#+ model1b, results='hide'
model1b <- cmdstan_model(stan_file = root("Birthdays", "gpbf1b.stan"),
                         include_paths = root("Birthdays"),
                         compile_model_methods=TRUE, force_recompile=TRUE)

#' First run Pathfinder
#' pth1b, results='hide'
tic('Sampling from Pathfinder approximation of model 1b posterior')
#pth1b <- model1b$pathfinder(data = standata1, init=0.1, num_paths=10, single_path_draws=40,
#                            history_size=50, max_lbfgs_iters=100)
pth1bs=list()
for (i in 1:10) {
  pth1bs[[i]] <- model1b$pathfinder(data = standata1, init=0.1, num_paths=1, single_path_draws=40,
                                    history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' We sample using the Pathfinder initialization.
#init1b <- create_inits(pth1b)
init1b <- create_inits(pth1bs)
#+ fit1b, results='hide'
tic('MCMC sampling from model 1b posterior with Pathfinder initialization')
fit1b <- model1b$sample(data=standata1, iter_warmup=100, iter_sampling=100,
                        chains=4, parallel_chains=4,
                        init=init1b)
#+
mytoc()

#' The sampling is even faster, indicating that the strong posterior
#' correlation in the first model was causing troubles for the
#' adaptation in the short warmup.
draws1b <- fit1b$draws()
summarise_draws(subset(draws1b, variable=c('sigma_f1','lengthscale_f1','sigma')))
#' Examining the trace plots don't show multimodality
mcmc_trace(draws1b, regex_pars=c('sigma_f1','lengthscale_f1','sigma'))

#' We drop global intercept from the rest of the models, but continue
#' using Pathfinder to initialize the sampling.
#' 
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
#' following Riutort-Mayol:2023:HSGP turned out be very slow. With
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
                        include_paths = root("Birthdays"),
                         compile_model_methods=TRUE, force_recompile=TRUE)

#' Data to be passed to Stan
standata2 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=20,  # number of basis functions for GP for f1
                  J_f2=20)  # number of basis functions for periodic f2

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth2, results='hide'
tic('Sampling from Pathfinder approximation of model 2 posterior')
#pth2 <- model2$pathfinder(data = standata2, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=50, max_lbfgs_iters=100)
pth2s=list()
for (i in 1:10) {
  pth2s[[i]] <- model2$pathfinder(data = standata2, init=0.1, num_paths=1, single_path_draws=40,
                                  history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Pareto-$\hat{k}$ is even higher, but the Pathfinder draws are
#' likely to be useful for quick analysis and initialization of MCMC
#' sampling.

#' Check whether parameters have reasonable values
#pdraws2 <- pth2$draws()
pdraws2 <- do.call(bind_draws, c(lapply(pth2s, as_draws), along='draw'))
summarise_draws(subset(pdraws2, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE),
                default_summary_measures())

#' Compare the model to the data
draws2 <- as_draws_matrix(pdraws2)
Ef <- exp(apply(subset(draws2, variable='f'), 2, median))
Ef1 <- apply(subset(draws2, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws2, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf / (pf1 + pf2)

#' Even Pareto-$\hat{k}$ indicated that Pathfinder approximation was
#' not good enough to be reliably adjusted using importance sampling,
#' the draws produce sensible model predictions.
#' 
#' Sample short chains using the Pathfinder result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init2 <- create_inits(pth2)
init2 <- create_inits(pth2s)
#+ fit2, results='hide'
tic('MCMC sampling from model 2 posterior with Pathfinder initialization')
fit2 <- model2$sample(data=standata2, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=init2)
#+
mytoc()

#' While Pathfinder took about 12s, sampling with short chains is
#' taking over 70s, and we see a clear benefit in being able to obtain
#' approximate Pathfinder results faster.
#' 

#' Check whether parameters have reasonable values
draws2 <- fit2$draws()
summarise_draws(subset(draws2, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))

#' Compare the model to the data
draws2 <- as_draws_matrix(draws2)
Ef <- exp(apply(subset(draws2, variable='f'), 2, median))
Ef1 <- apply(subset(draws2, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws2, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
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
                        include_paths = root("Birthdays"),
                        compile_model_methods=TRUE, force_recompile=TRUE)

#' Data to be passed to Stan
standata3 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=20,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  day_of_week=data$day_of_week)

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth3, results='hide'
tic('Sampling from Pathfinder approximation of model 3 posterior')
#pth3 <- model3$pathfinder(data = standata3, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=50, max_lbfgs_iters=100)
pth3s=list()
for (i in 1:10) {
  pth3s[[i]] <- model3$pathfinder(data = standata3, init=0.1, num_paths=1, single_path_draws=40,
                                  history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Check whether parameters have reasonable values
#pdraws3 <- pth3$draws()
pdraws3 <- do.call(bind_draws, c(lapply(pth3s, as_draws), along='draw'))
summarise_draws(subset(pdraws3, variable=c('sigma_','lengthscale_','sigma', 'beta_f3'), regex=TRUE),
                default_summary_measures())

#' Compare the model to the data
draws3 <- as_draws_matrix(pdraws3)
Ef <- exp(apply(subset(draws3, variable='f'), 2, median))
Ef1 <- apply(subset(draws3, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws3, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws3, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
(pf + pf1) / (pf2 + pf3)

#' Sample short chains using the Pathfinder result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init3 <- create_inits(pth3)
init3 <- create_inits(pth3s)
#+ fit3, results='hide'
tic('MCMC sampling from model 3 posterior with Pathfinder initialization')
fit3 <- model3$sample(data=standata3, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=init3)
#+
mytoc()

#' Check whether parameters have reasonable values
draws3 <- fit3$draws()
summarise_draws(subset(draws3, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
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
#' getting further away from each other. In previous analysis [@BDA3]
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
                        include_paths = root("Birthdays"),
                        compile_model_methods=TRUE, force_recompile=TRUE)

#' Data to be passed to Stan
standata4 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=20,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  c_g3=1.5, # factor c of basis functions for GP for g3
                  M_g3=5,   # number of basis functions for GP for g3
                  day_of_week=data$day_of_week) 

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth4, results='hide'
tic('Sampling from Pathfinder approximation of model 4 posterior')
#pth4 <- model4$pathfinder(data = standata4, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=50, max_lbfgs_iters=100)
pth4s=list()
for (i in 1:10) {
  pth4s[[i]] <- model4$pathfinder(data = standata4, init=0.1, num_paths=1, single_path_draws=40,
                                  history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Check whether parameters have reasonable values
#pdraws4 <- pth4$draws()
pdraws4 <- do.call(bind_draws, c(lapply(pth4s, as_draws), along='draw'))
summarise_draws(subset(pdraws4, variable=c('sigma_','lengthscale_','sigma', 'beta_f3'), regex=TRUE),
                default_summary_measures())

#' Compare the model to the data
draws4 <- as_draws_matrix(pdraws4)
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
pf3b <- data %>%
  mutate(Ef3 = Ef3) %>%
  ggplot(aes(x=date, y=births_relative100/Ef1/Ef2*100*100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
(pf + pf1) / (pf2 + pf3b)

#' Sample short chains using the Pathfinder result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init4 <- create_inits(pth4)
init4 <- create_inits(pth4s)
#+ fit4, results='hide'
tic('MCMC sampling from model 4 posterior with Pathfinder initialization')
fit4 <- model4$sample(data=standata4, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=init4)
#+
mytoc()

#' Check whether parameters have reasonable values
draws4 <- fit4$draws()
summarise_draws(subset(draws4, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
pf3b <- data %>%
  mutate(Ef3 = Ef3) %>%
  ggplot(aes(x=date, y=births_relative100/Ef1/Ef2*100*100)) +
  geom_point(color=set1[2], alpha=0.2) +
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
#' regularized horseshoe (RHS) prior [@Piironen+Vehtari:2017:rhs] for
#' day of year effects.
#'
#' At this point the optimization didn't produce reasonable result as
#' earlier and sampling turned out to be very slow. We assumed the
#' optimization fails because there were so many more parameters with
#' hierarchical prior. As even the short chain sampling would have
#' taken more than hour, it would have been time consuming to further
#' to test the model. As part of the quick iterative model building it
#' was better to give up on this model for a moment. When revisiting
#' this case study and adding Pathfinder approximation, it produced
#' much better results and using it to initialize MCMC, the sampling
#' took only 5 minutes.
#'
#' Compile Stan model 5 [gpbf5.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf5.stan)
#+ model5, results='hide'
model5 <- cmdstan_model(stan_file = root("Birthdays", "gpbf5.stan"),
                        include_paths = root("Birthdays"),
                        compile_model_methods=TRUE, force_recompile=TRUE)

#' Data to be passed to Stan
standata5 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=20,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  c_g3=1.5, # factor c of basis functions for GP for g3
                  M_g3=5,   # number of basis functions for GP for g3
                  scale_global=0.1, # gloval scale for RHS prior
                  day_of_week=data$day_of_week,
                  day_of_year=data$day_of_year2) # 1st March = 61 every year

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth5, results='hide'
tic('Sampling from Pathfinder approximation of model 5 posterior')
#pth5 <- model5$pathfinder(data = standata5, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=50, max_lbfgs_iters=100)
pth5s=list()
for (i in 1:10) {
  pth5s[[i]] <- model5$pathfinder(data = standata5, init=0.1, num_paths=1, single_path_draws=40,
                                  history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Check whether parameters have reasonable values
#pdraws5 <- pth5$draws()
pdraws5 <- do.call(bind_draws, c(lapply(pth5s, as_draws), along='draw'))
summarise_draws(subset(pdraws5, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE),
                default_summary_measures())
summarise_draws(subset(pdraws5, variable=c('beta_f3')),
                default_summary_measures())

draws5 <- as_draws_matrix(pdraws5)
Ef4 <- apply(subset(draws5, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")

draws5 <- as_draws_matrix(pdraws5)
Ef <- exp(apply(subset(draws5, variable='f'), 2, median))
Ef1 <- apply(subset(draws5, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws5, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws5, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef4 <- apply(subset(draws5, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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

#' The quick model fit looks reasoanble for a quick fit.
#'
#' Sample short chains using the Pathfinder result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init5 <- create_inits(pth5)
init5 <- create_inits(pth5s)
#+ fit5, results='hide'
tic('MCMC sampling from model 5 posterior with Pathfinder initialization')
fit5 <- model5$sample(data=standata5, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=init5)
#+
mytoc()

#' Before using Pathfinder to initialize sampling, the sampling took
#' longer than my patience, and the sampler result was not included in
#' the case study. With Pathfinder initialization, the sampler finishd
#' in 5 mins, but reported 100% of maximum treedepths which indicates
#' very strong posterior dependencies.
#' 

#' Check whether parameters have reasonable values
draws5 <- fit5$draws()
summarise_draws(subset(draws5, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))
summarise_draws(subset(draws5, variable=c('beta_f3')))

draws5 <- as_draws_matrix(pdraws5)
Ef4 <- apply(subset(draws5, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")

draws5 <- as_draws_matrix(draws5)
Ef <- exp(apply(subset(draws5, variable='f'), 2, median))
Ef1 <- apply(subset(draws5, variable='f1'), 2, median)
Ef1 <- exp(Ef1 - mean(Ef1) + mean(log(data$births_relative100)))
Ef2 <- apply(subset(draws5, variable='f2'), 2, median)
Ef2 <- exp(Ef2 - mean(Ef2) + mean(log(data$births_relative100)))
Ef_day_of_week <- apply(subset(draws5, variable='f_day_of_week'), 2, median)
Ef_day_of_week <- exp(Ef_day_of_week - mean(Ef_day_of_week) + mean(log(data$births_relative100)))
Ef4 <- apply(subset(draws5, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
pf <- data %>%
  mutate(Ef = Ef) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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

#' The plot looks quite good.

#' ### Model 6: long term smooth + seasonal + weekday + day of year
#'
#' To simplify the analysis of the day of year effect and make the
#' inference during the exploration faster, we drop the time dependent
#' day of week effect and RHS for a moment and use normal prior for
#' the day of year effect.
#'
#' $$
#' f = \mbox{intercept} + f_1 + f_2 + \beta_{\mbox{day of week}} + \beta_{\mbox{day of year}}\\
#' \mbox{intercept} \sim \mbox{normal}(0,1)\\
#' f_1 \sim \mbox{GP}(0,K_1)\\
#' f_2 \sim \mbox{GP}(0,K_2)\\
#' \beta_{\mbox{day of week}} = 0 \quad \mbox{if day of week is Monday}\\
#' \beta_{\mbox{day of week}} \sim \mbox{normal}(0,1) \quad \mbox{if day of week is not Monday}\\
#' \beta_{\mbox{day of year}} \sim \mbox{normal}(0,0.1)
#' $$
#' 
#' Compile Stan model 6 [gpbf6.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf6.stan)
#+ model6, results='hide'
model6 <- cmdstan_model(stan_file = root("Birthdays", "gpbf6.stan"),
                        include_paths = root("Birthdays"),
                        compile_model_methods=TRUE, force_recompile=TRUE)

#' Data to be passed to Stan
standata6 <- list(x=data$id,
                  y=log(data$births_relative100),
                  N=length(data$id),
                  c_f1=1.5, # factor c of basis functions for GP for f1
                  M_f1=20, # number of basis functions for GP for f1
                  J_f2=20, # number of basis functions for periodic f2
                  day_of_week=data$day_of_week,
                  day_of_year=data$day_of_year2) # 1st March = 61 every year

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth6, results='hide'
tic('Sampling from Pathfinder approximation of model 6 posterior')
#pth6 <- model6$pathfinder(data = standata6, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=50, max_lbfgs_iters=100)
pth6s=list()
for (i in 1:10) {
  pth6s[[i]] <- model6$pathfinder(data = standata6, init=0.1, num_paths=1, single_path_draws=40,
                                  history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Check whether parameters have reasonable values
#pdraws6 <- pth6$draws()
pdraws6 <- do.call(bind_draws, c(lapply(pth6s, as_draws), along='draw'))
summarise_draws(subset(pdraws6, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE),
                default_summary_measures())
summarise_draws(subset(pdraws6, variable=c('beta_f3')),
                default_summary_measures())

draws6 <- as_draws_matrix(pdraws6)
Ef4 <- apply(subset(draws6, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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

#' We recognize some familiar structure in the day of year effect and
#' proceed to sampling.
#' 
#' Sample short chains using the Pathfinder result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init6 <- create_inits(pth6)
init6 <- create_inits(pth6s)
#+ fit6, results='hide'
tic('MCMC sampling from model 6 posterior with Pathfinder initialization')
fit6 <- model6$sample(data=standata6, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=init6)
#+
mytoc()

#' Check whether parameters have reasonable values
draws6 <- fit6$draws()
summarise_draws(subset(draws6, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))
summarise_draws(subset(draws6, variable=c('beta_f3')))

draws6 <- as_draws_matrix(draws6)
Ef4 <- apply(subset(draws6, variable='beta_f4'), 2, median)*sd(log(data$births_relative100))
Ef4 <- exp(Ef4)*100
data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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
#' our previous analyses [@BDA3], so it seems the day or year effect model
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
                        include_paths = root("Birthdays"),
                        compile_model_methods=TRUE, force_recompile=TRUE)

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
                  M_f1=20,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  day_of_week=data$day_of_week,
                  day_of_year=data$day_of_year2, # 1st March = 61 every year
                  memorial_days=memorial_days,
                  labor_days=labor_days,
                  thanksgiving_days=thanksgiving_days)

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth7, results='hide'
tic('Sampling from Pathfinder approximation of model 7 posterior')
#pth7 <- model7$pathfinder(data = standata7, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=50, max_lbfgs_iters=100)
pth7s=list()
for (i in 1:10) {
  pth7s[[i]] <- model7$pathfinder(data = standata7, init=0.1, num_paths=1, single_path_draws=40,
                                  history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Check whether parameters have reasonable values
#pdraws7 <- pth7$draws()
pdraws7 <- do.call(bind_draws, c(lapply(pth7s, as_draws), along='draw'))
summarise_draws(subset(pdraws7, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE),
                default_summary_measures())
summarise_draws(subset(pdraws7, variable=c('beta_f3')),
                default_summary_measures())

#' Compare the model to the data
draws7 <- as_draws_matrix(pdraws7)
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)

pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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

#' Sample short chains using the Pathfinder result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init7 <- create_inits(pth7)
init7 <- create_inits(pth7s)
#+ fit7, results='hide'
tic('MCMC sampling from model 7 posterior with Pathfinder initialization')
fit7 <- model7$sample(data=standata7, iter_warmup=100, iter_sampling=100,
                      chains=4, parallel_chains=4,
                      init=init7, refresh=10)
#+
mytoc()

#' Check whether parameters have reasonable values
draws7 <- fit7$draws()
summarise_draws(subset(draws7, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef), color=set1[1], alpha=0.75) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)

pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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
                        include_paths = root("Birthdays"),
                        compile_model_methods=TRUE, force_recompile=TRUE)

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
                  M_f1=20,  # number of basis functions for GP for f1
                  J_f2=20,  # number of basis functions for periodic f2
                  c_g3=1.5, # factor c of basis functions for GP for g3
                  M_g3=5,   # number of basis functions for GP for g3
                  day_of_week=data$day_of_week,
                  day_of_year=data$day_of_year2, # 1st March = 61 every year
                  memorial_days=memorial_days,
                  labor_days=labor_days,
                  thanksgiving_days=thanksgiving_days)

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth8, results='hide'
tic('Sampling from Pathfinder approximation of model 8 posterior')
pth8s=list()
for (i in 1:10) {
  pth8s[[i]] <- model8$pathfinder(data = standata8, init=0.1, num_paths=1, single_path_draws=40,
                                  history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Check whether parameters have reasonable values
#pdraws8 <- pth8$draws()
pdraws8 <- do.call(bind_draws, c(lapply(pth8s, as_draws), along='draw'))
summarise_draws(subset(pdraws8, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE),
                default_summary_measures())
summarise_draws(subset(pdraws8, variable=c('beta_f3')),
                default_summary_measures())

#' Compare the model to the data
draws8 <- as_draws_matrix(pdraws8)
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef), color=set1[1], alpha=0.2) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
N=length(data$id)
pf3b <- data %>%
  mutate(Ef3 = Ef3*Ef1/100) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births") +
  annotate("text",x=as.Date("1989-08-01"),y=(Ef3*Ef1/100)[c((N-5):(N-4), N, N-6)],label=c("Mon","Tue","Sat","Sun"))
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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

#' Sample short chains using the Pathfinder result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init8 <- create_inits(pth8)
init8 <- create_inits(pth8s)
#+ fit8, results='hide'
tic('MCMC sampling from model 8 posterior with Pathfinder initialization')
fit8 <- model8$sample(data=standata8, iter_warmup=100, iter_sampling=100, chains=4, parallel_chains=4,
                      init=init8, refresh=10)
#+
mytoc()

#' Check whether parameters have reasonable values
draws8 <- fit8$draws()
summarise_draws(subset(draws8, variable=c('sigma_','lengthscale_','sigma'), regex=TRUE))
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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef), color=set1[1], alpha=0.2) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
N=length(data$id)
pf3b <- data %>%
  mutate(Ef3 = Ef3*Ef1/100) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births") +
  annotate("text",x=as.Date("1989-08-01"),y=(Ef3*Ef1/100)[c((N-5):(N-4), N, N-6)],label=c("Mon","Tue","Sat","Sun"))
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)
pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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
#' implementation for the model 5 had challenging posterior. Before
#' testing RHS again, we'll test with an easier to implement Student's
#' $t$ prior whether long tailed prior for day of year effect is
#' reasonable. These experiments help also to find out whether the day
#' of year effect is sensitive to the prior choice.
#'
#'
#' ### Model 8+t_nu: day of year effect with Student's t prior
#' 
#' Compile Stan model 8 + t_nu [gpbf8tnu.stan](https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbf8tnu.stan)
#+ model8tnu, results='hide'
model8tnu <- cmdstan_model(stan_file = root("Birthdays", "gpbf8tnu.stan"),
                           include_paths = root("Birthdays"),
                           compile_model_methods=TRUE, force_recompile=TRUE)

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth8tnu, results='hide'
tic('Sampling from Pathfinder approximation of model 8tnu posterior')
#pth8tnu <- model8tnu$pathfinder(data = standata8, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=50, max_lbfgs_iters=100)
pth8tnus=list()
for (i in 1:10) {
  pth8tnus[[i]] <- model8tnu$pathfinder(data = standata8, init=0.1, num_paths=1, single_path_draws=40,
                                        history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Sample short chains using the Pathfinder result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init8tnu <- create_inits(pth8tnu)
init8tnu <- create_inits(pth8tnus)
#+ fit8tnu, results='hide'
tic('MCMC sampling from model 8tnu posterior with Pathfinder initialization')
fit8tnu <- model8tnu$sample(data=standata8, iter_warmup=100, iter_sampling=100,
                            chains=4, parallel_chains=4,
                            init=init8tnu, refresh=10)
#+
mytoc()

#' Check whether parameters have reasonable values
draws8tnu <- fit8tnu$draws()
summarise_draws(subset(draws8tnu, variable=c('intercept','sigma_','lengthscale_','sigma','nu_'), regex=TRUE))
#' Posterior of degrees of freedom `nu_f4` is very close to 0.5, and
#' thus the distribution has thicker tails than Cauchy. This is strong
#' evidence that the distribution of day of year effects is far from
#' normal.

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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef), color=set1[1], alpha=0.2) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
N=length(data$id)
pf3b <- data %>%
  mutate(Ef3 = Ef3*Ef1/100) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births") +
  annotate("text",x=as.Date("1989-08-01"),y=(Ef3*Ef1/100)[c((N-5):(N-4), N, N-6)],label=c("Mon","Tue","Sat","Sun"))
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)

pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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
#' cross-validation (LOO-CV, @Vehtari+Gelman+Gabry:2017:psisloo) as
#' each added component had qualitatively big and reasonable
#' effect. Now as day of year effect is sensitive to prior choice, but
#' it's not clear how much better $t_\nu$ prior distribution is we use
#' LOO-CV to compare the models.
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
                           include_paths = root("Birthdays"),
                           compile_model_methods=TRUE, force_recompile=TRUE)

#' Add a global scale for RHS prior
standata8 <- c(standata8,
               scale_global=0.1) # global scale for RHS prior

#' Pathfinder is faster than sampling (although this result can be
#' useful in a quick workflow, the result should not be used as the
#' final result).
#+ pth8rhs, results='hide'
tic('Sampling from Pathfinder approximation of model 8rhs posterior')
#pth8rhs <- model8rhs$pathfinder(data = standata8, init=0.1, num_paths=10, single_path_draws=40,
#                          history_size=50, max_lbfgs_iters=100, draws=800)
pth8rhss=list()
for (i in 1:10) {
  pth8rhss[[i]] <- model8rhs$pathfinder(data = standata8, init=0.1, num_paths=1, single_path_draws=40,
                                        history_size=100, max_lbfgs_iters=100)
}
#+
mytoc()

#' Sample short chains using the optimization result as initial values
#' (although the result from short chains can be useful in a quick
#' workflow, the result should not be used as the final result).
#init8rhs <- create_inits(pth8rhs)
init8rhs <- create_inits(pth8rhss)
#+ fit8rhs, results='hide'
tic('MCMC sampling from model 8rhs posterior with Pathfinder initialization')
fit8rhs <- model8rhs$sample(data=standata8, iter_warmup=100, iter_sampling=100,
                            chains=4, parallel_chains=4,
                            init=init8rhs, refresh=10)
#+
mytoc()

#' Check whether parameters have reasonable values
draws8rhs <- fit8rhs$draws()
summarise_draws(subset(draws8rhs, variable=c('sigma_','lengthscale_','sigma','nu_'), regex=TRUE))

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
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef), color=set1[1], alpha=0.2) +
  labs(x="Date", y="Relative number of births")
pf1 <- data %>%
  mutate(Ef1 = Ef1) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_line(aes(y=Ef1), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births")
pf2 <- data %>%
  mutate(Ef2 = Ef2) %>%
  group_by(day_of_year2) %>%
  summarise(meanbirths=mean(births_relative100), meanEf2=mean(Ef2)) %>%
  ggplot(aes(x=as.Date("1987-12-31")+day_of_year2, y=meanbirths)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  geom_line(aes(y=meanEf2), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of year")
pf3 <- ggplot(data=data, aes(x=day_of_week, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  scale_x_continuous(breaks = 1:7, labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun')) +
  geom_line(data=data.frame(x=1:7,y=Ef_day_of_week), aes(x=x, y=Ef_day_of_week), color=set1[1]) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births of week")
N=length(data$id)
pf3b <- data %>%
  mutate(Ef3 = Ef3*Ef1/100) %>%
  ggplot(aes(x=date, y=births_relative100)) +
  geom_point(color=set1[2], alpha=0.2) +
  geom_point(aes(y=Ef3), color=set1[1], size=0.1) +
  geom_hline(yintercept=100, color='gray') +
  labs(x="Date", y="Relative number of births") +
  annotate("text",x=as.Date("1989-08-01"),y=(Ef3*Ef1/100)[c((N-5):(N-4), N, N-6)],label=c("Mon","Tue","Sat","Sun"))
f13 <- data %>% filter(year==1988)%>%select(day,date)%>%mutate(y=Ef4float)%>%filter(day==13)

pf2b <-data.frame(x=as.Date("1988-01-01")+0:365, y=Ef4float) %>%
  ggplot(aes(x=x,y=y)) +
  geom_line(color=set1[1]) +
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

#' Visually we get quite similar result as with $t_\nu$ prior. When we
#' compare the models with LOO-CV [@Vehtari+Gelman+Gabry:2017:psisloo],
#' there is not much difference between these priors.
loo8rhs<-fit8rhs$loo()
loo_compare(list(`Model 8 Students t`=loo8tnu,`Model 8 RHS`=loo8rhs))

#' ### Further improvements for the day of year effect
#' 
#' It's unlikely that day of year effect would be unstructured with
#' some distribution like RHS, and thus instead of trying to find a
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
#' We didn't use LOO-CV [@Vehtari+Gelman+Gabry:2017:psisloo] until in
#' the end, as the qualitative differences between models were very
#' convincing. We can use LOO-CV to check how big the difference in
#' the predictive performance are and if the differences are big, we
#' know that model averaging that would take into account the
#' uncertainty would give weights close to zero for all but the most
#' elaborate models.
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
  ggplot(aes(x=date, y=log(births_relative100/Ef))) +
  geom_point(color=set1[2]) +
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
