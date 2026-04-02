#' ---
#' title: "Modeling disc golf putting"
#' author: "Aki Vehtari"
#' date: 2026-03-12
#' date-modified: today
#' date-format: iso
#' format:
#'   html:
#'     number-sections: true
#'     code-copy: true
#'     code-download: true
#'     code-tools: true
#'     encoding: UTF-8
#'     theme: readable
#'     css: _styles.css
#'     toc: true
#'     toc-depth: 2
#'     toc-location: right
#'     smooth-scroll: true
#'     embed-resources: true
#'     anchor-sections: true
#'     html-math-method: katex
#'#' bibliography: ../casestudies.bib
#' ---
#'
#+ setup, include = FALSE
knitr::opts_chunk$set(
  cache = FALSE,
  message = FALSE,
  error = FALSE,
  warning = FALSE,
  comment = NA,
  out.width = '95%'
)
#| code-fold: true
library(rprojroot)
root <- has_file(".Workflow-Examples-root")$make_fix_file()
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(tibble)
library(cmdstanr)
# CmdStanR output directory makes Quarto cache to work
dir.create(root("disc_putting", "stan_output"))
options(cmdstanr_output_dir = root("disc_putting", "stan_output"))
options(mc.cores = 4)
library(posterior)
library(loo)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))
library(patchwork)
library(ggdist)
library(reliabilitydiag)
library(tinytable)
options(tinytable_format_num_fmt = "significant_cell",
        tinytable_format_digits = 2,
        tinytable_tt_digits = 2)
print_stan_file_fold <- function(file, summary = "Stan model code") {
  code <- readLines(file)
  if (isTRUE(getOption("knitr.in.progress")) &
        identical(knitr::opts_current$get("results"), "asis")) {
    cat("<details><summary>", summary, "</summary>\n\n", sep = "")
    cat("```stan\n")
    cat(code, sep = "\n")
    cat("\n```\n")
    cat("\n</details>\n")
  } else {
    writeLines(code)
  }
}

#' # Introduction
#'
#' We use a simple geometrical model and Bayesian statistical analysis
#' to make predictions of disc golf putting success from different
#' distances by professional disc golfers. [Player, event, and course
#' data ©2026 PDGA](http://www.pdga.com/). Based on the geometrical
#' model and Bayesian inference, the putting angle accuracies of top
#' MPO and FPO players are about 1° and 1.4°, respectively (This angle
#' accuracy is not about nose angle, but describes the imperfect
#' execution when the thrower aims to release the disc to initially
#' fly along a desired line. The effect of imperfect direction
#' increases with distance. See more below).
#'
#' The model used here was inspired by [Andrew Gelman's model for ball
#' golf
#' putting](https://mc-stan.org/learn-stan/case-studies/golf.html).
#'
#' ## Previous model for ball golf putting
#'
#' The first part in Gelman's model is uncertainty in angle. From
#' Gelman's case study:
#'
#' > The graph below shows a simplified sketch of a golf shot. The dotted line represents the angle within which the ball of radius $r$ must be hit so that it falls within the hole of radius $R$...  The graph, which is not to scale, is intended to illustrate the geometry of the ball needing to go into the hole.
#' ![](golf3.png)
#'
#' > The next step is to model human error. We assume that the golfer is attempting to hit the ball completely straight but that many small factors interfere with this goal, so that the actual angle follows a normal distribution centered at 0 with some standard deviation $\sigma$.
#' ![](golf4.png){width=60%}
#'
#' > The probability the ball goes in the hole is then the probability that the angle is less than the threshold... The only unknown parameter in this model is $\sigma$, the standard deviation of the distribution of shot angles. 
#'
#' The second part of Gelman's model takes into account how hard the ball is hit:
#'
#' > To get the ball in the hole, the angle isn’t the only thing you need to control; you also need to hit the ball just hard enough.
#' > Mark Broadie added this to our model by introducing another parameter corresponding to the golfer’s control over distance. Supposing $u$ is the distance that golfer’s shot would travel if there were no hole, Broadie assumes that the putt will go in if (a) the angle allows the ball to go over the hole, and (b) $u$ is in the range $[x,x+3]$. That is the ball must be hit hard enough to reach the hole but not go too far. Factor (a) is what we have considered earlier; we must now add factor (b).
#' > The following sketch, which is not to scale, illustrates the need for the distance as well as the angle of the shot to be in some range, in this case the gray zone which represents the trajectories for which the ball would reach the hole and stay in it.
#' ![](golf7.png)
#' Additional parameter $\sigma_{\mathrm{distance}}$ represents the uncertainty in the shot’s relative distance. 
#' 
#' # New model for disc golf putting
#'
#' - From distance less than 1m (about 3 feet), the disc can be put
#' into basket. We removed throws with recorded distance <=3 feet and
#' don't model these.
#' 
#' - At short distances (from 1m to 3.3m, 3 feet to 10 feet), almost
#' all putts go in, but some fail, maybe due to the chain assembly
#' spitting the disc out or other rare circumstances. To account for
#' this, a base uncertainty parameter is included to the model.
#'
#' - In disc golf there is angle uncertainty in both horizontal and
#' vertical direction, and the disc needs to stay within certain width
#' and height to hit the basket. It is unlikely that we could learn
#' horizontal and vertical uncertainty separately, and we use one
#' parameter for both, but the probability that disc hits the basket
#' takes into account one additional dimension in the geometrical
#' model. Second parameter models angle uncertainty.
#'
#' - The chain assembly height is about 50cm and width about
#' 55cm. Taking into account the disc width which is about 21cm, the
#' target area is about 50cm times 30cm, that is, about 0.15m$^2$. We
#' simplify by assuming a circular target area with the same area, that
#' is, with a diameter of 50cm. The exact target area doesn't matter for the
#' predicted probabilities as the angle uncertainty will scale
#' correspondingly.
#'
#' - There is also uncertainty in distance control. Too slow throw
#' misses the basket, but too hard throw hitting the chain assembly
#' sufficiently middle is likely to stay in the basket. Thus the model
#' is slightly different than in case of angles. As we don't have
#' information whether the miss was due to angle or distance control,
#' it is unlikely we can learn much of the distance control uncertainty
#' separately and the same uncertainty scale parameter is used for
#' distance control uncertainty.
#'
#' - At longer distances, the disc throwing mechanism changes more
#' than the ball golf putting mechanism, and we may expect the angle
#' and distance control uncertainties to increase with distance. An
#' additional uncertainty slope parameter and slope threshold
#' parameter describe how angle and distance control uncertainty
#' increase with distance. This adds a third and a fourth parameter to
#' the model.
#'
#' - These four parameters of the model are learned using Bayesian
#' posterior inference, and the learned geometrical model can predict
#' the disc golf putting probabilities from different distances quite
#' well.
#'
#' - To allow modeling the putting for each player, a hierarchical
#' model is used. The base uncertainty and slope threshold parameters
#' are common for all players. Each player has their own angle and
#' distance control uncertainty parameter and uncertainty slope
#' parameter, with a shared population prior distribution. This way
#' information between players is shared, but if there are differences
#' between players, they can be learned.
#'
#' - For part of throws distance has been recorded only to be in
#' certain interval, and these are included in the model as interval
#' observations.
#'
#' # Disc golf putting data
#'
#' The disc golf data are provided by
#' [Professional Disc Golf Association (PDGA)](http://www.pdga.com/).
#' [Player, event, and course data ©2026 PDGA](http://www.pdga.com/).
#' 
#' We started by collecting all throws from
#' 
#' - [DGPT Playoffs - The 10th Disc Golf Pro Tour Championship presented by Barbasol](https://www.pdga.com/tour/event/88302)
#'
#' and then added for 33 MPO and 20 FPO players, who had been in the Championship tournament, all the throws from two playoff tournaments
#' 
#' - [DGPT Playoffs - MVP Open x OTB](https://www.pdga.com/tour/event/88301)
#' - [DGPT Playoffs - Green Mountain Championship](https://www.pdga.com/tour/event/88299)
#'
#' We could collect more data, but this was enough for testing different models.
#'
#' The collected data are shared with permission from [PDGA](http://www.pdga.com/) and available in files [x3_MPO_all_throws_raw.Rdata](./x3_MPO_all_throws_raw.Rdata) and [x3_FPO_all_throws_raw.Rdata](./x3_FPO_all_throws_raw.Rdata). These files contain data frames (tibbles) tha look like following
#| code-fold: true
load("x3_MPO_all_throws_raw.Rdata")
load("x3_FPO_all_throws_raw.Rdata")
glimpse(throws_mpo)

#' We consider all throws inside Circle 1 (distance < 10m) and Circle
#' 2 (10m < distance < 20m) to be putts, although it is likely that
#' some throws near 20m are not aimed at the basket, but safely at the
#' ground under the basket.
#'
#' Here are basic statistics for the throws:
#' 
#| code-fold: true
throws_mpo <- throws_mpo |>
  group_by(player, round, hole) |>
  mutate(
    throwin = case_when(
      stringr::str_detect(lead(location), "Eagle|Birdie|Par|Bogey|Ace") ~ 1,
      .default = 0
    )
  ) |>
  ungroup()
throws <- throws_mpo
# All throws
n_throws <- throws |> nrow()
# Throw-ins outside of Circle 2
n_throwins_far <- throws |> filter(throwin == 1 & distance_ft > 65) |> nrow()
# Throws and throw-ins from Circle 1 or Circle 2
n_throws_12 <- throws |> filter(str_detect(location, "Circle")) |> nrow()
n_throwins_12 <- throws |> filter(throwin == 1 & str_detect(location, "Circle")) |> nrow()
mpo_stats <- tribble(
  ~label, ~result,
  "Total number of throws", n_throws,
  "Number of throw-ins >20m", n_throwins_far,
  "Number of throws <20m", n_throws_12,
  "Number of throw-ins <20m", n_throwins_12,
  )
throws_fpo <- throws_fpo |>
  group_by(player, round, hole) |>
  mutate(
    throwin = case_when(
      stringr::str_detect(lead(location), "Eagle|Birdie|Par|Bogey|Ace") ~ 1,
      .default = 0
    )
  ) |>
  ungroup()
throws <- throws_fpo
# All throws
n_throws <- throws |> nrow()
# Throw-ins outside of Circle 2
n_throwins_far <- throws |> filter(throwin == 1 & distance_ft > 65) |> nrow()
# Throws and throw-ins from Circle 1 or Circle 2
n_throws_12 <- throws |> filter(str_detect(location, "Circle")) |> nrow()
n_throwins_12 <- throws |> filter(throwin == 1 & str_detect(location, "Circle")) |> nrow()
fpo_stats <- tribble(
  ~label, ~result,
  "Total number of throws", n_throws,
  "Number of throw-ins >20m", n_throwins_far,
  "Number of throws <20m", n_throws_12,
  "Number of throw-ins <20m", n_throwins_12,
  )
cbind(mpo_stats, fpo_stats[,2]) |>
  tt(colnames = FALSE) |>
  group_tt(j = list("MPO" = 2, "FPO" = 3))

#' ## Missing distance data
#' 
#' Looking at the recorded distances, we see there are more throws at
#' distances (in feet) 6, 16, 26, 39, 49, 59 than at nearby
#' distances. Disc golf rules use meters, and we would prefer meters,
#' but since the distances recorded by [PDGA](http://www.pdga.com/) are in
#' integer-valued feet, we use a mix of meters and feet in this case
#' study.
#| code-fold: true
p1a <- throws_mpo |>
  filter(distance_ft <= 65) |>
  ggplot(aes(x=distance_ft)) +
  geom_histogram(breaks=(1:65)-0.5) +
  scale_x_continuous(breaks = seq(2,64,by=2)) +
  labs(x= "Distance (ft)", title = "MPO: All reported throw distances <= 65 ft") 
p2a <- throws_fpo |>
  filter(distance_ft <= 65) |>
  ggplot(aes(x=distance_ft)) +
  geom_histogram(breaks=(1:65)-0.5) +
  scale_x_continuous(breaks = seq(2,64,by=2)) +
  labs(x= "Distance (ft)", title = "FPO: All reported throw distances <= 65 ft") 
p1a/p2a

#' Checking the data in more detail, we see that all throw-ins have
#' distance recorded, but many non-throw-ins do not have distance
#' recorded.
#| code-fold: true
p1 <- throws_mpo |>
  filter(throwin == 0 & str_detect(location, "Circle")) |>
  group_by(player) |>
  summarise(ft_prop = mean(!is.na(distance_ft))) |>
  ggplot(aes(ft_prop)) +
  geom_histogram(breaks = seq(-.05,1.05,by=0.1), fill = "white", color = "steelblue") +
  labs(x = "Proportion",
       y = "Count",
       title = "MPO: Proportion of non-throw-ins (<20m) without distance per player") +
  scale_y_continuous(breaks = seq(0,8,by=2))
p2 <- throws_fpo |>
  filter(throwin == 0 & str_detect(location, "Circle")) |>
  group_by(player) |>
  summarise(ft_prop = mean(!is.na(distance_ft))) |>
  ggplot(aes(ft_prop)) +
  geom_histogram(breaks = seq(-.05,1.05,by=0.1), fill = "white", color = "steelblue") +
  labs(x = "Proportion",
       y = "Count",
       title = "FPO: Proportion of non-throw-ins (<20m) without distance per player") +
  scale_y_continuous(breaks = seq(0,8,by=2))
p1/p2

#' All throws within 20m have their location recorded as Circle 1 or Circle
#' 2. Some throws, whether they went in or not, have distance
#' recorded with one-foot accuracy. It seems that for many throws,
#' Circle 1 or Circle 2 would be recorded if the throw does not go in,
#' and if it goes in, a rough estimate is added, explaining the excess of
#' recorded distances at 6, 16, 26, 39, 49, and 59.
#'
#' We can't simply drop throws without distance in feet recorded.  As
#' all throw-ins have some distance in feet recorded, and only
#' non-throw-ins have missing distances, dropping throws with missing
#' distances would bias the results. Also as none of the players have
#' all distances recorded, we can't choose to include only players
#' with all distances recorded.
#'
#' The following figure shows raw probabilities of MPO putting success
#' at each recorded distance from 1 foot to 65 feet. All MPO throws
#' with a given recorded distance have been pooled together. The dot
#' shows the raw probability, that is, the number of throw-ins divided by
#' the number of throws from that distance. The lines show a 95% uncertainty
#' interval based on a simple independent binomial model assumption.
#' The raw probabilities at distances 6, 16, 26, 39, 49, and 59 feet
#' have been circled. We see that the raw probabilities at these
#' distances are higher than at nearby distances. This is likely due
#' to a coarse distance estimate being recorded for some throws only if
#' they go in the basket.
#| code-fold: true
putts_by_feet <- throws_mpo |>
  group_by(distance_ft) |>
  summarise(
    throws   = n(),
    success  = sum(throwin),
    .groups = "drop"
  ) |>
  filter(distance_ft <= 65)
# Raw probabilities by feet for putts with distance recorded
putts_by_feet |>
  ggplot(aes(x=distance_ft, y=success/throws, ymin=qbeta(.025, success, throws-success), ymax=qbeta(.975, success, throws-success))) +
  coord_cartesian(expand = FALSE) +
  scale_x_continuous(breaks=c(0,10,20,30,40,50,60)) +
  geom_pointinterval(alpha = 0.7, color="steelblue") +
  geom_vline(xintercept = c(3.3/0.3048, 10/0.3048, 20/0.3048), linetype = "dashed") +
  theme(legend.position = "none",
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5)) +
  labs(x="Distance (ft)", y="Probability of success", title="Putts with distance recorded") +
  annotate(geom = "text", x = 11.5, y = 0.02, label = "C1X (3.3m-10m)", hjust = 0) + 
  annotate(geom = "text", x = 33.5, y = 0.02, label = "C2 (10m-20m)", hjust = 0) +
  geom_point(data = putts_by_feet |> filter(distance_ft %in% c(6,16,26,39,49,59)), color="red", shape=1, size=5)

#' ## Unknown distances
#' 
#' To include also the non-throw-ins without distance recorded, but
#' location recorded as Circle 1 or Circle 2, the corresponding
#' distances are modelled as unknown, but constraint to be in
#' intervals [10, 33] (feet; Circle 1 starts from 0ft, but probability
#' of non-throw under 10ft is very small) and [33, 65] (feet). In
#' addition distances 6, 16, 26, 39, 49, and 59 are considered to be
#' unknown in intervals [4, 10], [10, 20], [20, 33], [33, 45], [45,
#' 55], [55, 65]. Each interval distance adds one parameter to the
#' model. A weakly informative prior distribution for unknown
#' distances is proportional to squared distance, based on the fact
#' that area at certain distance grows as squared distance.
#' 
#| code-fold: true
putts_mis <- throws_mpo |>
  group_by(player, round, hole) |>
  mutate(
    lead_dist = lead(distance_ft),
  ) |>
  ungroup() |>
  rowwise() |>
  mutate(
    distance_lower = case_when(
      distance_ft == 6 ~ 4,
      distance_ft == 16 ~ 10,
      distance_ft == 26 ~ 20,
      distance_ft == 39 ~ 33,
      distance_ft == 49 ~ 45,
      distance_ft == 59 ~ 55,
      (!is.na(location) && location == "Circle 1" && 
         is.na(distance_ft) && !is.na(lead_dist)) ~ 4,
      (!is.na(location) && location == "Circle 2" && is.na(distance_ft)) ~ 33,
      .default = distance_ft),
    distance_upper = case_when(
      distance_ft == 6 ~ 10,
      distance_ft == 16 ~ 20,
      distance_ft == 26 ~ 33,
      distance_ft == 39 ~ 45,
      distance_ft == 49 ~ 55,
      distance_ft == 59 ~ 65,
      (!is.na(location) && location == "Circle 1" && 
            is.na(distance_ft) && !is.na(lead_dist)) ~ 33,
      (!is.na(location) && location == "Circle 2" && is.na(distance_ft)) ~ 65,
      .default = distance_ft),
    missing = case_when(
      is.na(distance_ft) | (distance_ft != distance_lower) ~ 1,
      .default = 0)
    ) |>
  ungroup() |>
  select(-lead_dist) |>
  group_by(player, round, hole) |>
  mutate(
    throwin = case_when(
      stringr::str_detect(location, "Circle") &
      stringr::str_detect(lead(location), "Eagle|Birdie|Par|Bogey") ~ 1,
      .default = 0
    )
  ) |>
  ungroup() |>
  filter_out(is.na(distance_upper) | distance_upper <= 3 | distance_upper > 65)
putts_by_feet_players_obs <- putts_mis |> filter(missing==0) |>
  group_by(distance_ft, distance_lower, distance_upper, missing, player) |>
  summarise(
    throws   = n(),
    throwins  = sum(throwin),
    .groups = "drop"
  )
putts_by_feet_players <- rbind(putts_by_feet_players_obs,
                               putts_mis |>
                                 filter(missing==1) |>
                                 mutate(throws = 1,
                                        throwins = throwin) |>
                                 select(distance_ft, distance_lower, distance_upper, missing, player, throws, throwins))

#' # MPO predicted putting probabilities
#'
#' ## Pooled geometrical model with two parameters
#' 
#' To illustrate the power of geometrical model, we start with a
#' simpler model which has only two parameters: baseline probability
#' of success and one scale parameter quantifying both angle and
#' distance control uncertainty. We pool the throws from all the players.
#| echo: false
#| output: asis
print_stan_file_fold("disc_putting_1.stan")

#' The following figure shows the proportion of successful putts by
#' distance (where we have integrated out the missing distances) and
#' geometrical model 1 based putting probability by distances based on
#' data for 33 MPO players.
#| label: fit1
#| cache: true
#| code-fold: true
R <- 50 / 2 / 100 / 0.3047 # target radius 50cm/2 to feet
stan_data_m <- with(putts_by_feet_players,
                    list(
  N_players = n_distinct(player),
  player = as.numeric(factor(player)),
  N_obs = sum(missing==0),
  N_mis = sum(missing==1),
  x_obs = distance_ft[missing==0],
  x_lower = distance_lower[missing==1],
  x_upper = distance_upper[missing==1],
  throws = throws,
  throwins = throwins,
  R = R,
  N_pred = 62,
  x_pred = 4:65
))
mod1 <- cmdstan_model("disc_putting_1.stan")
fit1 <- mod1$sample(data = stan_data_m, init = 0.1,
                    refresh = 0, show_messages = FALSE, show_exceptions = FALSE)
draws1 <- fit1$draws(format = "df")
drvs <- as_draws_rvars(draws1)
# average over sampled distances to get number of throws and successes by feet
drvs$xr <- round(drvs$x)
putts_by_feet_players$p = mean(drvs$p)
putts_by_feet <-
  lapply(seq(1,4000,length.out=400), \(i) {
    qr <- subset_draws(drvs$xr, draw = i) |> as_draws_matrix() |> as.numeric();
    putts_by_feet_players$distance_imp = qr;
    putts_by_feet <- putts_by_feet_players |>
      group_by(distance_imp) |>
      summarise(
        throws   = sum(throws),
        success  = sum(throwins),
        p = mean(p),
        .groups = "drop"
      )
  }) |>
  bind_rows() |>
  group_by(distance_imp) |>
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
# plot proportion of success and independent binomial uncertainties by feet plus the model predictions
putts_by_feet |>
  ggplot(aes(x=distance_imp, y=success/throws, ymin=qbeta(.025, success, throws-success), ymax=qbeta(.975, success, throws-success))) +
  coord_cartesian(expand = FALSE) +
  scale_x_continuous(limits=c(0,65.7), breaks=c(0,10,20,30,40,50,60)) +
  scale_y_continuous(limits=c(0,1)) +
  geom_pointinterval(alpha = 0.7, color="steelblue") +
  geom_ribbon(data=data.frame(x=4:65), inherit.aes=FALSE, aes(x=x, y=mean(drvs$p_pred), ymin=quantile2(drvs$p_pred, 0.05), ymax=quantile2(drvs$p_pred, 0.95)), alpha=0.2) +
  geom_vline(xintercept = c(3.3/0.3048, 10/0.3048, 20/0.3048), linetype = "dashed") +
  theme(legend.position = "none",
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5)) +
  labs(x="Distance (ft)", y="Probability of success", title = "Model 1") +
  annotate(geom = "text", x = 11.5, y = 0.02, label = "C1X (3.3m-10m)", hjust = 0) + 
  annotate(geom = "text", x = 33.5, y = 0.02, label = "C2 (10m-20m)", hjust = 0)

#' The predicted probabilities are already quite close to the observed
#' success proportions even with just one parameter controlling the
#' geometrical model. The observed success proportions include also
#' now the throws with missing distance (but known circle) and the
#' distances which were clumped at 6, 16, 26, 39, 49, and 59 feet have
#' also been included as interval observations. Thus, the success
#' proportions are now smoother and with less uncertainty.
#'
#' The geometrical model base predicted probability starts to drop
#' slightly earlier than the observed data and does not go to as low
#' probabilities in distances over 15m / 45ft. These discrepancies
#' could be caused, for example, by throwing technique changing in
#' longer distances increasing the angle and distance control
#' uncertainty, the natural flight path obstructed by trees or bushes,
#' or decisions to avoid risk of throwing out-of-bounds or far
#' downhill and throwing under the basket. It is likely that in ideal
#' throwing conditions, the probability of success would be higher at
#' longer distances.
#'
#' ## Pooled geometrical model with four parameters
#' 
#' The next model includes two additional parameters: 1) threshold
#' distance after which 2) an additional term, depending on distance
#' from the threshold distance, is added to the scale parameter
#' describing the angle and distance control uncertainty.
#| echo: false
#| output: asis
print_stan_file_fold("disc_putting_2.stan")
#'
#' The following figure shows the proportion of successful putts by
#' distance (where we have integrated out the missing distances) and
#' geometrical model 2 based putting probability by distances based on
#' data for 33 MPO players.
#| label: fit2
#| cache: true
#| code-fold: true
mod2 <- cmdstan_model("disc_putting_2.stan")
fit2 <- mod2$sample(data = stan_data_m, init = 0.1,
                    refresh = 0, show_messages = FALSE, show_exceptions = FALSE)
draws2 <- fit2$draws(format = "df")
drvs <- as_draws_rvars(draws2)
# average over sampled distances to get number of throws and successes by feet
drvs$xr <- round(drvs$x)
putts_by_feet_players$p = mean(drvs$p)
putts_by_feet <-
  lapply(seq(1,4000,length.out=400), \(i) {
    qr <- subset_draws(drvs$xr, draw = i) |> as_draws_matrix() |> as.numeric();
    putts_by_feet_players$distance_imp = qr;
    putts_by_feet <- putts_by_feet_players |>
      group_by(distance_imp) |>
      summarise(
        throws   = sum(throws),
        success  = sum(throwins),
        p = mean(p),
        .groups = "drop"
      )
  }) |>
  bind_rows() |>
  group_by(distance_imp) |>
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
# plot proportion of success and independent binomial uncertainties by feet plus the model predictions
putts_by_feet |>
  ggplot(aes(x=distance_imp, y=success/throws, ymin=qbeta(.025, success, throws-success), ymax=qbeta(.975, success, throws-success))) +
  coord_cartesian(expand = FALSE) +
  scale_x_continuous(limits=c(0,65.7), breaks=c(0,10,20,30,40,50,60)) +
  scale_y_continuous(limits=c(0,1)) +
  geom_pointinterval(alpha = 0.7, color="steelblue") +
  geom_ribbon(data=data.frame(x=4:65), inherit.aes=FALSE, aes(x=x, y=mean(drvs$p_pred), ymin=quantile2(drvs$p_pred, 0.05), ymax=quantile2(drvs$p_pred, 0.95)), alpha=0.2) +
  geom_vline(xintercept = c(3.3/0.3048, 10/0.3048, 20/0.3048), linetype = "dashed") +
  theme(legend.position = "none",
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5)) +
  labs(x="Distance (ft)", y="Probability of success", title = "Model 2") +
  annotate(geom = "text", x = 11.5, y = 0.02, label = "C1X (3.3m-10m)", hjust = 0) + 
  annotate(geom = "text", x = 33.5, y = 0.02, label = "C2 (10m-20m)", hjust = 0)

#' Two additional parameters clearly improve the model fit. The
#' distance threshold after which the uncertainty in angle and
#' distance control starts to increase is about 11m / 36ft. The
#' baseline uncertainty in angle is about 1°.
#'
#' ## Hierarchical geometrical model
#' 
#' The next model adds player specific parameters describing 1) the
#' base probability, 2) angle and distance control uncertainty, and 3)
#' how angle and distance control changes after the distance
#' threshold. The data were not informative enough to be able to infer
#' player specific distance thresholds. The player specific parameters
#' have joint hierarchical prior, so that the model learns from the
#' data how much variation there is between players.
#| echo: false
#| output: asis
print_stan_file_fold("disc_putting_3.stan")
#'
#' The following figure shows the proportion of successful putts by
#' distance (where we have integrated out the missing distances) and
#' geometrical model 3 based putting probability by distances based on
#' data for 33 MPO players.
#| label: fit3
#| cache: true
#| code-fold: true
mod3 <- cmdstan_model("disc_putting_3.stan")
fit3 <- mod3$sample(data = stan_data_m, init = 0.1,
                    refresh = 0, show_messages = FALSE, show_exceptions = FALSE)
draws3 <- fit3$draws(format = "df")
drvs <- as_draws_rvars(draws3)
# average over sampled distances to get number of throws and successes by feet
drvs$xr <- round(drvs$x)
putts_by_feet_players$p = mean(drvs$p)
putts_by_feet <-
  lapply(seq(1,4000,length.out=400), \(i) {
    qr <- subset_draws(drvs$xr, draw = i) |> as_draws_matrix() |> as.numeric();
    putts_by_feet_players$distance_imp = qr;
    putts_by_feet <- putts_by_feet_players |>
      group_by(distance_imp) |>
      summarise(
        throws   = sum(throws),
        success  = sum(throwins),
        p = mean(p),
        .groups = "drop"
      )
  }) |>
  bind_rows() |>
  group_by(distance_imp) |>
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
# plot proportion of success and independent binomial uncertainties by feet plus the model predictions
p_pred <- as.matrix(mean(drvs$p_pred))
p_pred <- tibble(
  player   = rep(unique(factor(putts_by_feet_players$player)), each = 62),
  distance = rep(4:65, times = 33),
  value    = as.vector(t(p_pred))
)
putts_by_feet |>
  ggplot(aes(x=distance_imp, y=success/throws, ymin=qbeta(.025, success, throws-success), ymax=qbeta(.975, success, throws-success))) +
  coord_cartesian(expand = FALSE) +
  scale_x_continuous(limits=c(0,65.7), breaks=c(0,10,20,30,40,50,60)) +
  scale_y_continuous(limits=c(0,1)) +
  geom_pointinterval(alpha = 0.7, color="steelblue") +
  geom_line(data=p_pred,inherit.aes=FALSE,aes(x=distance, y=value, color=player, group=player), alpha=0.5) + 
  geom_vline(xintercept = c(3.3/0.3048, 10/0.3048, 20/0.3048), linetype = "dashed") +
  theme(legend.position = "none",
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5)) +
  labs(x="Distance (ft)", y="Probability of success", title = "Model 3") +
  annotate(geom = "text", x = 11.5, y = 0.02, label = "C1X (3.3m-10m)", hjust = 0) + 
  annotate(geom = "text", x = 33.5, y = 0.02, label = "C2 (10m-20m)", hjust = 0)
# pre-compute results used later in calibration checking
putts_by_feet_pl <-
  lapply(seq(1,4000,length.out=400), \(i) {
    qr <- subset_draws(drvs$xr, draw = i) |> as_draws_matrix() |> as.numeric();
    putts_by_feet_players$distance_imp = qr;
    putts_by_feet <- putts_by_feet_players |>
      group_by(distance_imp, player) |>
      summarise(
        throws   = sum(throws),
        success  = sum(throwins),
        p = mean(p),
        .groups = "drop"
      )
  }) |>
  bind_rows() |>
  group_by(distance_imp, player) |>
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
individual_throws_mpo <- putts_by_feet_pl |>
  mutate(throws=round(throws), throwins=round(success)) |>
  select(throws, throwins, p) |>
  mutate(row_id = row_number()) |>
  uncount(throws, .remove = FALSE) |>
  group_by(row_id) |>
  mutate(
    throw_num = row_number(),
    success_individual = as.integer(throw_num <= throwins)
  ) |>
  ungroup() |>
  select(-row_id, -throw_num, -throws, -throwins) |>
  rename(success = success_individual, prob = p)

#' As we have a hierarchical model, we can also look at putting
#' probabilities and associated posterior uncertainty for different
#' players. The following plot shows the putting probabilities for 33
#' MPO players from a distance of 10m (edge of Circle 1).
#| code-fold: true
tibble(player=levels(factor(putts_by_feet_players$player)), p33=drvs$p_pred[,33]) |>
  ggplot(aes(xdist = p33, y = reorder(player, p33, median))) +
  scale_x_continuous(limits=c(0.32, 0.78), breaks=c(0.4,0.5,0.6,0.7)) +
  stat_slabinterval(fill = "steelblue", alpha = 0.7) +
  labs(x = "Probability of successful putt from 10m / 33ft", y="", title = "Predicted putting probabilities (MPO top 33)") +
  theme(panel.grid.major.x = element_line(color = "grey90", linewidth = 0.5))

#' To better compare players, we look at the difference to
#' [Gannon Buhr](https://www.pdga.com/player/75412). 
#| code-fold: true
tibble(player=levels(factor(putts_by_feet_players$player)), p33=drvs$p_pred[,33]-drvs$p_pred[18,33]) |>
  ggplot(aes(xdist = p33, y = reorder(player, p33, median))) +
  scale_x_continuous(limits=c(-0.38, 0.16)) +
  stat_slabinterval(fill = "steelblue", alpha = 0.7) +
  labs(x = "Difference to Buhr in probability of successful putt from 10m / 33ft", y="", title = "Predicted putting probability differences (MPO top 33)") +
  theme(panel.grid.major.x = element_line(color = "grey90", linewidth = 0.5))

#' For 20 out of 32 players, the posterior probability is higher
#' than 95% that their 10m putting success probability is worse than
#' [Gannon Buhr](https://www.pdga.com/player/75412)'s.
#' Based on these three tournaments, there is still
#' significant uncertainty who are the best putters.
#' 
#' # FPO predicted putting probabilities
#' 
#| code-fold: true
putts_mis <- throws_fpo |>
  group_by(player, round, hole) |>
  mutate(
    lead_dist = lead(distance_ft),
  ) |>
  ungroup() |>
  rowwise() |>
  mutate(
    distance_lower = case_when(
      distance_ft == 6 ~ 4,
      distance_ft == 16 ~ 10,
      distance_ft == 26 ~ 20,
      distance_ft == 39 ~ 33,
      distance_ft == 49 ~ 45,
      distance_ft == 59 ~ 55,
      (!is.na(location) && location == "Circle 1" && 
         is.na(distance_ft) && !is.na(lead_dist)) ~ 4,
      (!is.na(location) && location == "Circle 2" && is.na(distance_ft)) ~ 33,
      .default = distance_ft),
    distance_upper = case_when(
      distance_ft == 6 ~ 10,
      distance_ft == 16 ~ 20,
      distance_ft == 26 ~ 33,
      distance_ft == 39 ~ 45,
      distance_ft == 49 ~ 55,
      distance_ft == 59 ~ 65,
      (!is.na(location) && location == "Circle 1" && 
            is.na(distance_ft) && !is.na(lead_dist)) ~ 33,
      (!is.na(location) && location == "Circle 2" && is.na(distance_ft)) ~ 65,
      .default = distance_ft),
    missing = case_when(
      is.na(distance_ft) | (distance_ft != distance_lower) ~ 1,
      .default = 0)
    ) |>
  ungroup() |>
  select(-lead_dist) |>
  group_by(player, round, hole) |>
  mutate(
    throwin = case_when(
      stringr::str_detect(location, "Circle") &
      stringr::str_detect(lead(location), "Eagle|Birdie|Par|Bogey") ~ 1,
      .default = 0
    )
  ) |>
  ungroup() |>
  filter_out(is.na(distance_upper) | distance_upper <= 3 | distance_upper > 65)
putts_by_feet_players_obs <- putts_mis |> filter(missing==0) |>
  group_by(distance_ft, distance_lower, distance_upper, missing, player) |>
  summarise(
    throws   = n(),
    throwins  = sum(throwin),
    .groups = "drop"
  )
putts_by_feet_players <- rbind(putts_by_feet_players_obs,
                               putts_mis |>
                                 filter(missing==1) |>
                                 mutate(throws = 1,
                                        throwins = throwin) |>
                                 select(distance_ft, distance_lower, distance_upper, missing, player, throws, throwins))

#' ## Pooled geometrical model with two parameters
#' 
#' The following figure shows the proportion of successful putts by
#' distance (where we have integrated out the missing distances) and
#' geometrical model 1 based putting probability by distances based on
#' data for 20 FPO players.
#| label: fitf1
#| cache: true
#| code-fold: true
R <- 50 / 2 / 100 / 0.3047 # target radius 50cm/2 to feet
stan_data_f <- with(putts_by_feet_players,
                    list(
  N_players = n_distinct(player),
  player = as.numeric(factor(player)),
  N_obs = sum(missing==0),
  N_mis = sum(missing==1),
  x_obs = distance_ft[missing==0],
  x_lower = distance_lower[missing==1],
  x_upper = distance_upper[missing==1],
  throws = throws,
  throwins = throwins,
  R = R,
  N_pred = 62,
  x_pred = 4:65
))
fitf1 <- mod1$sample(data = stan_data_f, init = 0.1,
                     refresh = 0, show_messages = FALSE, show_exceptions = FALSE)
drawsf1 <- fitf1$draws(format = "df")
drvs <- as_draws_rvars(drawsf1)
# average over sampled distances to get number of throws and successes by feet
drvs$xr <- round(drvs$x)
putts_by_feet_players$p = mean(drvs$p)
putts_by_feet <-
  lapply(seq(1,4000,length.out=400), \(i) {
    qr <- subset_draws(drvs$xr, draw = i) |> as_draws_matrix() |> as.numeric();
    putts_by_feet_players$distance_imp = qr;
    putts_by_feet <- putts_by_feet_players |>
      group_by(distance_imp) |>
      summarise(
        throws   = sum(throws),
        success  = sum(throwins),
        p = mean(p),
        .groups = "drop"
      )
  }) |>
  bind_rows() |>
  group_by(distance_imp) |>
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
# plot proportion of success and independent binomial uncertainties by feet plus the model predictions
putts_by_feet |>
  ggplot(aes(x=distance_imp, y=success/throws, ymin=qbeta(.025, success, throws-success), ymax=qbeta(.975, success, throws-success))) +
  coord_cartesian(expand = FALSE) +
  scale_x_continuous(limits=c(0,65.7), breaks=c(0,10,20,30,40,50,60)) +
  scale_y_continuous(limits=c(0,1)) +
  geom_pointinterval(alpha = 0.7, color="steelblue") +
  geom_ribbon(data=data.frame(x=4:65), inherit.aes=FALSE, aes(x=x, y=mean(drvs$p_pred), ymin=quantile2(drvs$p_pred, 0.05), ymax=quantile2(drvs$p_pred, 0.95)), alpha=0.2) +
  geom_vline(xintercept = c(3.3/0.3048, 10/0.3048, 20/0.3048), linetype = "dashed") +
  theme(legend.position = "none",
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5)) +
  labs(x="Distance (ft)", y="Probability of success", title = "Model 1") +
  annotate(geom = "text", x = 11.5, y = 0.02, label = "C1X (3.3m-10m)", hjust = 0) + 
  annotate(geom = "text", x = 33.5, y = 0.02, label = "C2 (10m-20m)", hjust = 0)

#' The predicted probabilities are already quite close to the observed
#' success proportions even with just one parameter controlling the
#' geometrical model.
#'
#' ## Pooled geometrical model with four parameters
#' 
#' The following figure shows the proportion of successful putts by
#' distance (where we have integrated out the missing distances) and
#' geometrical model 2 based putting probability by distances based on
#' data for 20 FPO players.
#| label: fitf2
#| cache: true
#| code-fold: true
fitf2 <- mod2$sample(data = stan_data_f, init = 0.1,
                     refresh = 0, show_messages = FALSE, show_exceptions = FALSE)
drawsf2 <- fitf2$draws(format = "df")
drvs <- as_draws_rvars(drawsf2)
# average over sampled distances to get number of throws and successes by feet
drvs$xr <- round(drvs$x)
putts_by_feet_players$p = mean(drvs$p)
putts_by_feet <-
  lapply(seq(1,4000,length.out=400), \(i) {
    qr <- subset_draws(drvs$xr, draw = i) |> as_draws_matrix() |> as.numeric();
    putts_by_feet_players$distance_imp = qr;
    putts_by_feet <- putts_by_feet_players |>
      group_by(distance_imp) |>
      summarise(
        throws   = sum(throws),
        success  = sum(throwins),
        p = mean(p),
        .groups = "drop"
      )
  }) |>
  bind_rows() |>
  group_by(distance_imp) |>
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
# plot proportion of success and independent binomial uncertainties by feet plus the model predictions
putts_by_feet |>
  ggplot(aes(x=distance_imp, y=success/throws, ymin=qbeta(.025, success, throws-success), ymax=qbeta(.975, success, throws-success))) +
  coord_cartesian(expand = FALSE) +
  scale_x_continuous(limits=c(0,65.7), breaks=c(0,10,20,30,40,50,60)) +
  scale_y_continuous(limits=c(0,1)) +
  geom_pointinterval(alpha = 0.7, color="steelblue") +
  geom_ribbon(data=data.frame(x=4:65), inherit.aes=FALSE, aes(x=x, y=mean(drvs$p_pred), ymin=quantile2(drvs$p_pred, 0.05), ymax=quantile2(drvs$p_pred, 0.95)), alpha=0.2) +
  geom_vline(xintercept = c(3.3/0.3048, 10/0.3048, 20/0.3048), linetype = "dashed") +
  theme(legend.position = "none",
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5)) +
  labs(x="Distance (ft)", y="Probability of success", title = "Model 2") +
  annotate(geom = "text", x = 11.5, y = 0.02, label = "C1X (3.3m-10m)", hjust = 0) + 
  annotate(geom = "text", x = 33.5, y = 0.02, label = "C2 (10m-20m)", hjust = 0)

#' Two additional parameters clearly improve the model fit. The
#' distance threshold after which the uncertainty in angle and
#' distance control starts to increase is about 8m / 25ft (compare to
#' 11m / 36ft for MPO). The baseline uncertainty in angle is about
#' 1.4° (compare to 1° for MPO).
#'
#' ## Hierarchical geometrical model
#'
#' The following figure shows the proportion of successful putts by
#' distance (where we have integrated out the missing distances) and
#' geometrical model 3 based putting probability by distances based on
#' data for 20 FPO players.
#| label: fitf3
#| cache: true
#| code-fold: true
fitf3 <- mod3$sample(data = stan_data_f, init = 0.1,
                     refresh = 0, show_messages = FALSE, show_exceptions = FALSE)
drawsf3 <- fitf3$draws(format = "df")
drvs <- as_draws_rvars(drawsf3)
# average over sampled distances to get number of throws and successes by feet
drvs$xr <- round(drvs$x)
putts_by_feet_players$p = mean(drvs$p)
putts_by_feet <-
  lapply(seq(1,4000,length.out=400), \(i) {
    qr <- subset_draws(drvs$xr, draw = i) |> as_draws_matrix() |> as.numeric();
    putts_by_feet_players$distance_imp = qr;
    putts_by_feet <- putts_by_feet_players |>
      group_by(distance_imp) |>
      summarise(
        throws   = sum(throws),
        success  = sum(throwins),
        p = mean(p),
        .groups = "drop"
      )
  }) |>
  bind_rows() |>
  group_by(distance_imp) |>
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
# plot proportion of success and independent binomial uncertainties by feet plus the model predictions
p_pred <- as.matrix(mean(drvs$p_pred))
p_pred <- tibble(
  player   = rep(unique(factor(putts_by_feet_players$player)), each = 62),
  distance = rep(4:65, times = 20),
  value    = as.vector(t(p_pred))
)
putts_by_feet |>
  ggplot(aes(x=distance_imp, y=success/throws, ymin=qbeta(.025, success, throws-success), ymax=qbeta(.975, success, throws-success))) +
  coord_cartesian(expand = FALSE) +
  scale_x_continuous(limits=c(0,65.7), breaks=c(0,10,20,30,40,50,60)) +
  scale_y_continuous(limits=c(0,1)) +
  geom_pointinterval(alpha = 0.7, color="steelblue") +
  geom_line(data=p_pred,inherit.aes=FALSE,aes(x=distance, y=value, color=player, group=player), alpha=0.5) + 
  geom_vline(xintercept = c(3.3/0.3048, 10/0.3048, 20/0.3048), linetype = "dashed") +
  theme(legend.position = "none",
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5)) +
  labs(x="Distance (ft)", y="Probability of success", title = "Model 3") +
  annotate(geom = "text", x = 11.5, y = 0.02, label = "C1X (3.3m-10m)", hjust = 0) + 
  annotate(geom = "text", x = 33.5, y = 0.02, label = "C2 (10m-20m)", hjust = 0)
# pre-compute results used later in calibration checking
putts_by_feet_pl <-
  lapply(seq(1,4000,length.out=400), \(i) {
    qr <- subset_draws(drvs$xr, draw = i) |> as_draws_matrix() |> as.numeric();
    putts_by_feet_players$distance_imp = qr;
    putts_by_feet <- putts_by_feet_players |>
      group_by(distance_imp, player) |>
      summarise(
        throws   = sum(throws),
        success  = sum(throwins),
        p = mean(p),
        .groups = "drop"
      )
  }) |>
  bind_rows() |>
  group_by(distance_imp, player) |>
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
individual_throws_fpo <- putts_by_feet_pl |>
  mutate(throws=round(throws), throwins=round(success)) |>
  select(throws, throwins, p) |>
  mutate(row_id = row_number()) |>
  uncount(throws, .remove = FALSE) |>
  group_by(row_id) |>
  mutate(
    throw_num = row_number(),
    success_individual = as.integer(throw_num <= throwins)
  ) |>
  ungroup() |>
  select(-row_id, -throw_num, -throws, -throwins) |>
  rename(success = success_individual, prob = p)

#' As we have a hierarchical model, we can also look at putting
#' probabilities and associated posterior uncertainty for different
#' players. The following plot shows the putting probabilities for 20
#' FPO players from a distance of 10m (edge of Circle 1).
#| code-fold: true
tibble(player=levels(factor(putts_by_feet_players$player)), p33=drvs$p_pred[,33]) |>
  ggplot(aes(xdist = p33, y = reorder(player, p33, median))) +
  scale_x_continuous(limits=c(0, 0.55), breaks=c(0,0.1,0.2,0.3,0.4,0.5)) +
  stat_slabinterval(fill = "steelblue", alpha = 0.7) +
  labs(x = "Probability of successful putt from 10m / 33ft", y="", title = "Predicted putting probabilities (FPO top 20)") +
  theme(panel.grid.major.x = element_line(color = "grey90", linewidth = 0.5))

#' To better compare players, we look at the difference to
#' [Cadence Burge](https://www.pdga.com/player/79233).
#| code-fold: true
tibble(player=levels(factor(putts_by_feet_players$player)), p33=drvs$p_pred[,33]-drvs$p_pred[2,33]) |>
  ggplot(aes(xdist = p33, y = reorder(player, p33, median))) +
  scale_x_continuous(limits=c(-0.48, 0.2)) +
  stat_slabinterval(fill = "steelblue", alpha = 0.7) +
  labs(x = "Difference to Burge in probability of successful putt from 10m / 33ft", y="", title = "Predicted putting probability differences (FPO top 20)") +
  theme(panel.grid.major.x = element_line(color = "grey90", linewidth = 0.5))

#' # Predictive model comparison and checking
#'
#' Here are some additional details on predictive model comparison and
#' checking used during the model building workflow.
#'
#' ## Predictive performance comparison
#' 
#' In addition of the three models shown above, we did try other model
#' candidates and compared the predictive performance using
#' leave-one-out cross-validation (LOO-CV) with log score
#' (elpd_loo). We use fast Pareto smoothed importance sampling (PSIS)
#' to compute LOO-CV. The observations with interval distance,
#' removing that observation can change the posterior that much that
#' PSIS is not reliable. To fix this we compute necessary log
#' likelihood terms by integrating out the unknown distance. This
#' approach is known also as integrated PSIS-LOO.
#'
#' **Compute LOO-CV for MPO model 1**
#| echo: false
#| output: asis
print_stan_file_fold("gq_ll_1.stan")
#| label: gq1
#| cache: true
#| results: hide
#| code-fold: true
modll1 <- cmdstan_model("gq_ll_1.stan")
qg1 <- modll1$generate_quantities(fit1, stan_data_m)
li1 <- loo(qg1$draws("log_liki", format="matrix"))
#' **Compute LOO-CV for MPO model 2**
#| echo: false
#| output: asis
print_stan_file_fold("gq_ll_2.stan")
#| label: gq2
#| cache: true
#| results: hide
#| code-fold: true
modll2 <- cmdstan_model("gq_ll_2.stan")
qg2 <- modll2$generate_quantities(fit2, stan_data_m)
li2 <- loo(qg2$draws("log_liki", format="matrix"))
#' **Compute LOO-CV for MPO model 3**
#| echo: false
#| output: asis
print_stan_file_fold("gq_ll_3.stan")
#| label: gq3
#| cache: true
#| results: hide
#| code-fold: true
modll3 <- cmdstan_model("gq_ll_3.stan")
qg3 <- modll3$generate_quantities(fit3, stan_data_m)
li3 <- loo(qg3$draws("log_liki", format="matrix"))

#' Model comparison with PSIS-LOO log score. Differences (elpd_diff)
#' bigger than twice the standard error (se_diff) are considered to be
#' significant.
#| code-fold: true
loo_compare(list(`Model 1` = li1, `Model 2` = li2)) |>
  as.data.frame() |> 
  rownames_to_column("model") |>
  select(model, elpd_diff, se_diff) |>
  tt()
loo_compare(list(`Model 2` = li2, `Model 3` = li3)) |>
  as.data.frame() |> 
  rownames_to_column("model") |>
  select(model, elpd_diff, se_diff) |>
  tt()

#' Model 2 is clearly better than model 1, and model 3 is clearly
#' better than model 2. Other tested models were not better than model
#' 3. The estimated putting probabilities for players at 10m were not
#' sensitive to different model choices for models that had similar
#' predictive performance as model 3.
#'
#' **Compute LOO-CV for FPO model 1**
#| label: gqf1
#| cache: true
#| results: hide
#| code-fold: true
qgf1 <- modll1$generate_quantities(fitf1, stan_data_f)
lif1 <- loo(qgf1$draws("log_liki", format="matrix"))
#' **Compute LOO-CV for FPO model 2**
#| label: gqf2
#| cache: true
#| results: hide
#| code-fold: true
qgf2 <- modll2$generate_quantities(fitf2, stan_data_f)
lif2 <- loo(qgf2$draws("log_liki", format="matrix"))
#' **Compute LOO-CV for FPO model 3**
#| label: gqf3
#| cache: true
#| results: hide
#| code-fold: true
qgf3 <- modll3$generate_quantities(fitf3, stan_data_f)
lif3 <- loo(qgf3$draws("log_liki", format="matrix"))

#' Model comparison with PSIS-LOO log score. Differences (elpd_diff)
#' bigger than twice the standard error (se_diff) are considered to be
#' significant.
#| code-fold: true
loo_compare(list(`Model 1` = lif1, `Model 2` = lif2)) |>
  as.data.frame() |> 
  rownames_to_column("model") |>
  select(model, elpd_diff, se_diff) |>
  tt()
loo_compare(list(`Model 2` = lif2, `Model 3` = lif3)) |>
  as.data.frame() |> 
  rownames_to_column("model") |>
  select(model, elpd_diff, se_diff) |>
  tt()

#' Model 2 is clearly better than model 1, and model 3 is clearly
#' better than model 2. Other tested models were not better than model
#' 3. The estimated putting probabilities for players at 10m were not
#' sensitive to different model choices for models that had similar
#' predictive performance as model 3.
#' 
#' ## Calibration check
#'
#' We can examine how well calibrated the predictions are with a
#' calibration plot, which has the predicted probabilities on the
#' x-axis and a monotonic curve fitted on the actual events. If the
#' red line stays inside the blue area, the model is well calibrated.
#' The following plot shows the calibration plot for MPO model 3.
#| code-fold: true
with(individual_throws_mpo,
     reliabilitydiag(EMOS = prob, y = success)) |>
  autoplot() +
  labs(x = "Predicted probability",
       y="Conditional event probability",
       title="Calibration plot for predicted putting probabilities (MPO top 33)")

#' We see that the model predicted probabilities are quite well
#' calibrated except that low probabilities are estimated to be a bit
#' higher than observed conditional event probabilities. The model
#' could be still improved, but it would be better to first collect
#' more data.
#' 
#' The correspondinf calibration plot for FPO model 3.
#| code-fold: true
with(individual_throws_fpo,
     reliabilitydiag(EMOS = prob, y = success)) |>
  autoplot() +
  labs(x = "Predicted probability",
       y="Conditional event probability",
       title="Calibration plot for predicted putting probabilities (FPO top 20)")

#' We see that the model predicted probabilities are quite well
#' calibrated except that very low probabilities are estimated to be a
#' bit higher than observed conditional event probabilities. The model
#' could be still improved, but it would be better to first collect
#' more data.
#' 
#' # Packages {.unnumbered}
#'
#' Packages used in this project:
#' 
#'  - R (version 4.5.2; R Core Team, 2025)
#'  - bayesplot (version 1.15.0.9000; Gabry J, Mahr T, 2025)
#'  - chromote (version 0.5.1; Aden-Buie G et al., 2025)
#'  - cmdstanr (version 0.9.0.9000; Gabry J et al., 2025)
#'  - dplyr (version 1.1.4; Wickham H et al., 2023)
#'  - ggdist (version 3.3.3; Kay M, 2024)
#'  - ggplot2 (version 4.0.1; Wickham H, 2016)
#'  - loo (version 2.9.0; Vehtari A et al., 2025)
#'  - patchwork (version 1.3.1; Pedersen T, 2025)
#'  - posterior (version 1.6.1.9000; Bürkner P et al., 2025)
#'  - purrr (version 1.2.1; Wickham H, Henry L, 2026)
#'  - reliabilitydiag (version 0.2.1; Dimitriadis T et al., 2021)
#'  - report (version 0.6.3; Makowski D et al., 2023)
#'  - rprojroot (version 2.1.1; Müller K, 2025)
#'  - rvest (version 1.0.4; Wickham H, 2024)
#'  - stringr (version 1.6.0; Wickham H, 2025)
#'  - tibble (version 3.3.1; Müller K, Wickham H, 2026)
#'  - tidyr (version 1.3.2; Wickham H et al., 2025)
#'  - tinytable (version 0.15.2; Arel-Bundock V, 2025)
#' 
#' # Licenses {.unnumbered}
#' 
#' * Code &copy; 2026, Aki Vehtari, licensed under BSD-3.
#' * Text &copy; 2026, Aki Vehtari, licensed under CC-BY-NC 4.0.
#' * [Player, event, and course data ©2026 PDGA](http://www.pdga.com/).
#' 
