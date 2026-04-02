#' ---
#' title: "Friends model checking case study"
#' author: "Aki Vehtari"
#' date: 2025-08-26
#' date-modified: today
#' date-format: iso
#' format:
#'   html:
#'     number-sections: true
#'     code-copy: true
#'     code-download: true
#'     code-tools: true
#' bibliography: ../casestudies.bib
#' ---
#' 
#' # Setup  {.unnumbered}
#' 
#+ setup, include=FALSE
knitr::opts_chunk$set(cache=FALSE, message=FALSE, error=FALSE, warning=FALSE, comment=NA, out.width='95%')

#' 
#' **Load packages**
#| code-fold: true
#| cache: FALSE
library(dplyr)
library(tidyr)
library(reliabilitydiag)
library(loo)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size=16))
library(brms)
options(brms.backend = "cmdstanr", mc.cores = 4)
library(marginaleffects)
library(patchwork)
library(ggdist)
library(khroma)

#' 
#' # Introduction
#'
#' This case study was inspired by [a case study by Julia M. Rohrer
#' and Vincent
#' Arel-Bundock](https://j-rohrer.github.io/marginal-psych/examples/relationship.html)
#' illustrating the use of [`marginaleffects` R
#' package](https://marginaleffects.com/).
#'
#' From their case study:
#'
#' > It’s a common complaint that people who enter a relationship
#'   start to neglect their friends. Here, we are going to use this to
#'   motivate an associational research question: Do people in
#'   romantic relationships, on average, assign less importance to
#'   their friends?  To address this question, we analyze data that
#'   were collected in the context of a diary study on satisfaction
#'   with various aspects of life (Rohrer et al., 2024). In this
#'   study, 482 people reported whether they were in a romantic
#'   relationship of any kind (partner) and also the extent to which
#'   they considered their friendships important
#'   (friendship_importance) on a scale from 1 (not important at all)
#'   to 5 (very important).
#'
#' In this cases study, I build corresponding Bayesian models and
#' illustrate model checking, comparison, and interpretation. I did
#' check convergence diagnostics and sufficient effective sample sizes
#' [@Vehtari-Gelman-Simpson-etal:2021] for all MCMC inferences, but do
#' not show them explicitly here.
#' 
#' # Read and clean the data
friends <- read.csv("start.csv")
friends <- friends |>
  filter(sex != 3 &      # exclude 5 people who reported a gender distinct from male/female
           age < 60) |>  # exclude people over the age of 60
  select(IMP_friends_Start, age, partner_any, sex) |>
  rename(partner = partner_any,
         gender = sex,
         friendship_importance = IMP_friends_Start) |>
  drop_na() |> 
  mutate(gender = factor(gender, levels = c(1, 2), labels = c("female", "male")),
         partner = factor(partner, levels = c(0, 1), labels = c("no", "yes")),
         age_scaled = age/60) # normalize for easier prior definition
head(friends) |> print(digits = 2)

#' # Normal data model
#'
#' Normal model is commonly used even we know the target is not
#' normally distributed or continuous. It can be computationally
#' convenient especially in initial data exploration in the early part
#' of modeling workflow. 
#'
#' ## Linear model with age
#
#' Using scaled age, makes the prior definition easier especially for
#' the later models with more predictor terms. For this simplest model,
#' we could use any wide prior or even flat prior for the only coefficient,
#' but R2D2 prior [@Zhang+etal:2022:R2D2; @Aguilar-Burkner:2023] has
#' a benefit that the prior is set on $R^2$ and when
#' we later add more model components, by using the same R2D2 prior all
#' the models have similar prior predictive distribution.
#| results: hide
#| cache: true
fit_age_lin <- brm(friendship_importance ~ age_scaled,
               data = friends,
               prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3, cons_D2 = 1/3), class = b),
               control = list(adapt_delta = 0.99),
               refresh = 0) |>
  add_criterion(criterion="loo")

#' ## Each year of age as its own category
#'
#' The original model had formula `~ as.factor(age)`, but for Bayesians
#' it is more natural to use hierarchical model where the prior scale
#' for the category specific intercepts is also inferred.
#| results: hide
#| cache: true
fit_age_cat <- brm(friendship_importance ~ 1 + (1 | age), data = friends,
               refresh = 0) |>
  add_criterion(criterion="loo")

#' ## Spline model with age
#'
#' The original model used `bs(age, df = 4)`, but `s()` is better as
#' it uses regularized thin plate splines and we don't need to choose
#' degrees of freedoms. We can use R2D2 prior also for spline
#' magnitude parameter.
#| results: hide
#| cache: true
fit_age_smooth <- brm(friendship_importance ~ s(age_scaled),
                      data = friends,
                      prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3, cons_D2 = 1/3), class = sds),
                      control = list(adapt_delta = 0.99),
                      refresh = 0) |>
  add_criterion(criterion = "loo")

#' ## Spline and interactions with age, gender, and partner
#'
#' The original case study included the following formula:
#' ```
#' friendship_importance ~ bs(age, df = 4) + gender + partner +
#'            bs(age, df = 4):gender + partner:gender + bs(age, df = 4):partner
#' ```
#' 
#' As splines didn't seem to be helpful, we also build a corresponding
#' model with linear terms and the same interactions.
#| results: hide
#| cache: true
fit_i <- brm(friendship_importance ~ age_scaled + age_scaled:gender + age_scaled:partner +
               gender*partner,
             prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3, cons_D2 = 1/3), class = b),
             data = friends,
             control = list(adapt_delta = 0.99),
             refresh = 0) |>
  add_criterion(criterion = "loo")

#' As before for the spline version we use s() instead of bs().
#| results: hide
#| cache: true
fit_is <- brm(friendship_importance ~ s(age_scaled) + gender*partner +
                s(age_scaled, by = gender) + s(age_scaled, by = partner),
              prior = c(prior(R2D2(mean_R2 = 1/3, prec_R2 = 3, cons_D2 = 1/3, main = TRUE), class = b),
                        prior(R2D2(main = FALSE), class = sds)),
              data = friends,
              control = list(adapt_delta = 0.95),
              refresh = 0) |>
  add_criterion(criterion = "loo")

#' ## Comparison of predictive performance
#'
#' We already have computed above the expected log predictive densities
#' using fast Pareto smoothed importance sampling leave-one-out
#' cross-validation [@Vehtari+Gelman+Gabry:2017:psisloo; @Vehtari+etal:2024:PSIS],
#' and use those for comparing the models.

loo_compare(fit_age_lin, fit_age_cat, fit_age_smooth,
            model_names = c("Normal linear",
                            "Normal categorical",
                            "Normal spline"))

#' Categorical is clearly the worst. Linear and spline model are
#' practically equally good.

loo_compare(fit_age_smooth, fit_i, fit_is,
            model_names = c("Normal spline with age only",
                            "Normal linear with interactions",
                            "Normal spline with interactions"))

#' There is no predictive performance benefit from the additional predictors.
#'
#' ## Predictive model checking
#'
#' We compare model predictions to the data to diagnose possible model
#' misspecification.
#' 
#' The default posterior predictive checking plot uses kernel density
#' estimate, which is not good for discrete data, but still shows that
#' the data distribution is more skewed than normal.
pp_check(fit_age_smooth)

#' Histogram plot makes it clear that the normal distribution does not
#' resemble the data.
pp_check(fit_age_smooth, type = "hist", ndraws=1) +
  scale_x_continuous(breaks = 1:7)

#' We can discretize the normal model by rounding the predictions to
#' nearest integer and use bars plot. Bars plot is often useless for
#' ordinal models as shown later also in this case study. In this
#' case, the normal model is so rigid that even the bar plot can show
#' the misspecification.
ppc_bars(y = friends$friendship_importance,
         yrep = round(posterior_predict(fit_age_smooth))) +
  scale_x_continuous(breaks = 1:7) +
  coord_cartesian(xlim=c(.5,7.5))

#' # Ordinal model
#' 
#' At this point let's switch to a better data model and use actual
#' ordinal model. We build linear and spline models without and with
#' interactions. In this case study, I don't build any polynomial
#' models as there is no domain specific theoretical justification to
#' use them for this data, and as generic non-linear models they are
#' inferior to regularized thin plate splines.
#'
#' We can use R2D2 prior with ordinal model, too. The implementation
#' in `brms` is not defining prior on $R^2$, but it is still taking
#' care that the prior predictive variance in latent space is not
#' increasing when we add more model compare. @Yanchenko:2025
#' discusses how to specify prior on pseudo-$R^2$ for ordinal models.
#' 
#' ## Latent linear model without interactions
#| results: hide
#| cache: true
fit_ord <- brm(friendship_importance ~ age_scaled + gender + partner,
                 family = cumulative(),
                 prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3, cons_D2 = 1/3), class = b),
               data = friends,
               control = list(adapt_delta = 0.99),
                 refresh = 0) |>
  add_criterion(criterion = "loo", save_psis = TRUE)

#' ## Latent spline model without interactions
#| results: hide
#| cache: true
fit_ord_s <- brm(friendship_importance ~ s(age_scaled) + gender + partner,
                 family = cumulative(),
                 prior = c(prior(R2D2(mean_R2 = 1/3, prec_R2 = 3, cons_D2 = 1/3, main = TRUE), class = b),
                           prior(R2D2(main = FALSE), class = sds)),
                 data = friends,
                 control = list(adapt_delta = 0.95),
                 refresh = 0) |>
  add_criterion(criterion = "loo", save_psis = TRUE)

#' ## Latent linear model with interactions
#| results: hide
#| cache: true
fit_ord_i <- brm(friendship_importance ~ age_scaled + age_scaled:gender + age_scaled:partner +
                   gender*partner,
                 family = cumulative(),
                 prior = prior(R2D2(mean_R2 = 1/3, prec_R2 = 3, cons_D2 = 1/3), class = b),
                 data = friends,
                 control = list(adapt_delta = 0.99),
                 refresh = 0) |>
  add_criterion(criterion = "loo", save_psis = TRUE)

#' ## Latent spline model with interactions
#| results: hide
#| cache: true
fit_ord_is <- brm(friendship_importance ~ s(age_scaled) + gender*partner +
                    s(age_scaled, by=gender) + s(age_scaled, by=partner),
                  family = cumulative(),
                  prior = c(prior(R2D2(mean_R2 = 1/3, prec_R2 = 3, cons_D2 = 1/3, main = TRUE), class = b),
                            prior(R2D2(main = FALSE), class = sds)),
                  data = friends,
                  control = list(adapt_delta = 0.95),
                  refresh = 0) |>
  add_criterion(criterion = "loo", save_psis = TRUE)

#' ## Comparison of predictive performance
#'
#' Again we have already computed LOO-CV results and compare
#' predictive performances.
loo_compare(fit_is, fit_ord, fit_ord_i, fit_ord_s, fit_ord_is,
            model_names = c("Normal spline with interactions",
                            "Ordinal linear",
                            "Ordinal linear with interactions",
                            "Ordinal spline",
                            "Ordinal spline with interactions"))

#' Ordinal models beat the normal model clearly (see [Nabiximols case
#' study](https://users.aalto.fi/~ave/casestudies/Nabiximols/nabiximols.html)
#' for explanation why we can compare continuous and discrete model in
#' this case). There is no practical difference in including splines
#' or interactions. The data are noisy and there are not that many
#' observations to be able to infer small non-linearities or
#' interactions.
#'
#' ## Predictive checking
#'
#' There is no practical difference in histograms.
pp_check(fit_ord_is, type = "hist", ndraws=1)

#' Bars plot looks great, but it is actually useless, as cumulative
#' ordinal model has category specific intercepts and bar plot would
#' look perfect even without any predictors [@Sailynoja-Johnson-Martin-etal:2025].
pp_check(fit_ord_is, type = "bars")

#' For ordinal models it is useful to examine calibration of
#' cumulative probabilities [@Sailynoja-Johnson-Martin-etal:2025]. If
#' the red calibration line stays inside the blue envelope, the
#' predictive probabilities are well calibrated.
#'
calibration_plot <- \(fit, i) {
  rd=reliabilitydiag(EMOS = E_loo((matrix(as.integer(posterior_predict(fit)),nrow=4000)<=i)+0,
                                  loo(fit)$psis_object)$value,
                     y = as.numeric(friends$friendship_importance<=i))
  autoplot(rd)+
    labs(x=paste0("Predicted probability of outcome <= ", i),
         y="Conditional event probabilities")+
    bayesplot::theme_default(base_family = "sans", base_size=16)
}
calibration_plot(fit_ord_is, 1)
calibration_plot(fit_ord_is, 2)
calibration_plot(fit_ord_is, 3)
calibration_plot(fit_ord_is, 4)

#' # Interpreting models
#'
#' We examine association between predictors and outcome. Usually we
#' would only examine models which have passed the model checking. For
#' demonstrations, we examine whether there is any difference between
#' using normal or ordinal model.
#'
#' We did several normal and ordinals models, and one option would be
#' to illustrate the results with all these models in multiverse style
#' [as Rohrer and Arel-Bundock
#' did](https://j-rohrer.github.io/marginal-psych/examples/relationship.html).
#' As Bayesians we can also average over the models to provide just
#' one combined model which takes into account the uncertainty in the
#' model choice. As the model with splines and interactions include
#' other models as nested inside, we can use the most complex
#' model. This is makes sense, especially when the most complex model
#' does not have much worse predictive performance. The most
#' complex model takes automatically into account the uncertainty in
#' different model choices.
#'
#' We use `marginaleffects` package as in the Rohrer and
#' Arel-Bundock's case study, but wrap it to produce data frames
#' useful for making our own plots overlaying the normal and ordinal
#' model results.
#' 
my_avg_comp <- \(fit, variables, modelname) {
  if (stringr::str_detect(modelname,"ordinal")) {
    comp <- avg_comparisons(fit, variables=variables, hypothesis = ~ I(sum(x * 1:5)))
  } else {
    comp <- avg_comparisons(fit, variables=variables)
  }
  comp <- comp |>
    get_draws() |>
    select("draw")
  comp$model <- modelname
  comp
}
plot_theme <- theme(axis.text.y=element_blank(),
                    axis.ticks.y=element_blank(),
                    axis.line.y=element_blank(),
                    axis.title.y=element_blank(),
                    plot.title = element_text(size=16),
                    legend.position="none")
clr <- colour("bright",names=FALSE)(7)

rbind(my_avg_comp(fit_is, "partner", "normal"),
      my_avg_comp(fit_ord_is, "partner", "ordinal")) |>
  ggplot(aes(x=draw, color=model)) +
  stat_slab(expand=TRUE, trim=FALSE, alpha=.6, fill=NA) +
  scale_color_bright() +
  plot_theme + 
  labs(x="Average difference in friendship importance",
       title="Partner no vs. yes") +
  annotate(geom="text", x=.03, y=0.7, label="Normal", hjust=0, color=clr[1], size=6) + 
  annotate(geom="text", x=-.14, y=0.7, label="Ordinal", hjust=1, color=clr[2], size=6)

rbind(my_avg_comp(fit_is, "gender", "normal"),
      my_avg_comp(fit_ord_is, "gender", "ordinal")) |>
  ggplot(aes(x=draw, color=model)) +
  stat_slab(expand=TRUE, trim=FALSE, alpha=.6, fill=NA) +
  scale_color_bright() +
  plot_theme + 
  labs(x="Average difference in friendship importance",
       title="Gender female vs. male") +
  annotate(geom="text", x=.02, y=0.7, label="Normal", hjust=0, color=clr[1], size=6) + 
  annotate(geom="text", x=-.14, y=0.7, label="Ordinal", hjust=1, color=clr[2], size=6)

rbind(my_avg_comp(fit_is, list("age_scaled"=5/60), "normal"),
      my_avg_comp(fit_ord_is, list("age_scaled"=5/60), "ordinal")) |>
  ggplot(aes(x=draw, color=model)) +
  stat_slab(expand=TRUE, trim=FALSE, alpha=.6, fill=NA) +
  scale_color_bright() +
  plot_theme + 
  labs(x="Average difference in friendship importance",
       title="Age +5 years") +
  annotate(geom="text", x=-.09, y=0.7, label="Normal", hjust=1, color=clr[1], size=6) + 
  annotate(geom="text", x=-.055, y=0.7, label="Ordinal", hjust=0, color=clr[2], size=6)

#' It seems partner status and gender do not have association to the
#' friendship importance, while age does. For age, posterior of the
#' ordinal model is sharper, but the overall conclusion would be the
#' same. In this case, the normal model posterior happens to be
#' similar to posterior of a more realistic model, as 1) the
#' conditional uncertainty is not very skewed, 2) has not very
#' non-normal like tails, and 3) the target is mean.  [Nabiximols case
#' study](https://users.aalto.fi/~ave/casestudies/Nabiximols/nabiximols.html)
#' demonstrates a case where using posterior of normal is quite
#' different from the posterior of a more realistic model. With this
#' data, there would be more difference in results if the target would
#' be for example the probability of answers 5.
#'
#' As the ordinal model with splines and interactions includes the
#' simpler ordinal models nested, it makes sense to to use only that
#' for analysing the age association, but for demonstration we compare
#' the age association posteriors from the four different ordinal
#' models we did built.
rbind(my_avg_comp(fit_ord, list("age_scaled"=5/60), "ordinal"),
      my_avg_comp(fit_ord_i, list("age_scaled"=5/60), "ordinal interactions"),
      my_avg_comp(fit_ord_s, list("age_scaled"=5/60), "ordinal splines"),
      my_avg_comp(fit_ord_is, list("age_scaled"=5/60), "ordinal interactions + splines")) |>
  ggplot(aes(x=draw, color=model)) +
  stat_slab(expand=TRUE, trim=FALSE, alpha=.6, fill=NA) +
  scale_color_bright() +
  plot_theme + 
  labs(x="Average difference in friendship importance",
       title="Age +5 years") +
  annotate(geom="text", x=-.095, y=0.85, label="Ordinal linear", hjust=1, color=clr[1], size=6) + 
  annotate(geom="text", x=-.143, y=0.75, label="Ordinal linear\n w. interactions", hjust=0, vjust=1, color=clr[2], size=6) +
  annotate(geom="text", x=-.058, y=0.85, label="Ordinal spline", hjust=0, color=clr[3], size=6) +
  annotate(geom="text", x=-.053, y=0.75, label="Ordinal spline\n w. interactions", hjust=0, vjust=1, color=clr[4], size=6)

#' Including interactions doesn't make much difference, but switching
#' from linear to spline makes clear difference although the main
#' conclusion would be the same. Although linear and spline models had
#' very similar predictive performance, I would trust the spline
#' models more as based on cross-validation the models re not
#' overfitted and they do take into account the uncertainty about the
#' functional shape of the age association.
#' 
#' 
#' <br />
#' 
#' # References {.unnumbered}
#' 
#' <div id="refs"></div>
#' 
#' # Licenses {.unnumbered}
#' 
#' * Code &copy; 2017-2025, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2017-2025, Aki Vehtari, licensed under CC-BY-NC 4.0.
#' 
#' # Original Computing Environment {.unnumbered}
#' 
sessionInfo()

#' 
#' <br />
#' 
