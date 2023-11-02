
## Load relevant packages
################################################################
library(rstan)
library(R.matlab)
library(loo)
library(hBayesDM)
#library(bayesplot)
#library(bayestestR)
#library(see)
library(bridgesampling)
#####


## Set number of cores to run Stan models on and other default options
################################################################
# options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
set.seed(1331)
#####


## Load data and create input data list for Stan models
################################################################
setwd("~/Downloads/Dopamine-ET-Manuscript/Data/")
data <- readMat('ET_stan_data.mat')

idx    <- c(1:11)
Trials <- c(data$num.trials[idx])
dat    <- list(N       = 11,
               T_max   = 150,
               T_subjs = Trials,
               choice  = data$choice[idx,],
               option1 = data$option1[idx,],
               option2 = data$option2[idx,],
               outcome = data$outcome[idx,],
               reward  = data$reward[idx,],
               punish  = data$punish[idx,])
#####


## Form stan model objects for desired models (to be sampled from)
################################################################
model_arg_TDRL <- stan_model("~/Downloads/Dopamine-ET-Manuscript/Code/TDRL.stan")
model_arg_VPRL <- stan_model("~/Downloads/Dopamine-ET-Manuscript/Code/VPRL.stan")
#####


## Run model sampling for TDRL and VPRL models
################################################################
fit_TDRL <- sampling(model_arg_TDRL, 
                     data    = dat, 
                     iter    = 12000, 
                     warmup  = 2000, 
                     cores   = 3, 
                     chains  = 3,
                     seed    = 1331,
                     control = list(adapt_delta=0.95))

fit_VPRL <- sampling(model_arg_VPRL, 
                     data    = dat, 
                     iter    = 12000, 
                     warmup  = 2000, 
                     cores   = 3, 
                     chains  = 3,
                     seed    = 1331,
                     control = list(adapt_delta=0.95))
#####


## Calculate predictive density of models for comparison with ELPD
################################################################
loglik_TDRL <- extract_log_lik(fit_TDRL, merge_chains = FALSE)
r_eff_TDRL  <- relative_eff(exp(loglik_TDRL), chain_id = rep(1:3, each = nrow(loglik_TDRL) / 3), cores = 3)
loo_TDRL    <- loo(loglik_TDRL, r_eff = r_eff_TDRL, save_psis = TRUE, cores = 3)
print(loo_TDRL)

loglik_VPRL <- extract_log_lik(fit_VPRL, merge_chains = FALSE)
r_eff_VPRL  <- relative_eff(exp(loglik_VPRL), chain_id = rep(1:3, each = nrow(loglik_VPRL) / 3), cores = 3)
loo_VPRL    <- loo(loglik_VPRL, r_eff = r_eff_VPRL, save_psis = TRUE, cores = 3)
print(loo_VPRL)

loo_compare(loo_TDRL,loo_VPRL)
#####


## Calculate marginal likelihood of models for comparison with Bayes Factors
################################################################
bridge_TDRL <- bridge_sampler(fit_TDRL, data = dat, repetitions = 10)
print(bridge_TDRL)

bridge_VPRL <- bridge_sampler(fit_VPRL, data = dat, repetitions = 10)
print(bridge_VPRL)

bayes_factor <- bf(bridge_TDRL,bridge_VPRL,log=TRUE)
print(bayes_factor)
#####


## Extract model parameters
################################################################
pars_TDRL <- extract(fit_TDRL)
pars_VPRL <- extract(fit_VPRL)
#####


## Save Stan model fit and write .mat file of posterior distribution samples for model parameters
################################################################
writeMat('TDRL_params.mat', learnrate_TDRL    = pars_TDRL$learnrate,
                            tau_TDRL          = pars_TDRL$tau,
                            discount_TDRL     = pars_TDRL$discount,
                            mu_learnrate_TDRL = pars_TDRL$mu_learnrate,
                            mu_tau_TDRL       = pars_TDRL$mu_tau,
                            mu_discount_TDRL  = pars_TDRL$mu_discount,
                            ppc_TDRL          = pars_TDRL$y_pred)

writeMat('VPRL_params.mat',learnrate_rew_VPRL    = pars_VPRL$learnrate_rew, 
                           learnrate_pun_VPRL    = pars_VPRL$learnrate_pun, 
                           tau_VPRL              = pars_VPRL$tau, 
                           discount_rew_VPRL     = pars_VPRL$discount_rew, 
                           discount_pun_VPRL     = pars_VPRL$discount_pun,
                           mu_learnrate_rew_VPRL = pars_VPRL$mu_learnrate_rew,
                           mu_learnrate_pun_VPRL = pars_VPRL$mu_learnrate_pun,
                           mu_tau_VPRL           = pars_VPRL$mu_tau,
                           mu_discount_rew_VPRL  = pars_VPRL$mu_discount_rew,
                           mu_discount_pun_VPRL  = pars_VPRL$mu_discount_pun,
                           ppc_VPRL              = pars_VPRL$y_pred)
#####

