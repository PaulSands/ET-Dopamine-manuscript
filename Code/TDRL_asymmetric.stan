data {
  int<lower=1> N;
  int<lower=1> T_max;
  int<lower=1, upper=T_max> T_subjs[N];
  int<lower=-1, upper=8> option1[N, T_max];
  int<lower=-1, upper=8> option2[N, T_max];
  int<lower=-1, upper=2> choice[N, T_max];
  real outcome[N, T_max];
}

transformed data {
  row_vector[3] initV;
  initV = rep_row_vector(0.0, 3);
}

parameters {
  // Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[4] mu_pr;
  vector<lower=0>[4] sigma_pr;

  // Subject-level raw parameters 
  vector[N] learnrate_posPE_pr;
  vector[N] learnrate_negPE_pr;
  vector[N] discount_pr;
  vector[N] tau_pr;
}

transformed parameters {
  // subject-level parameters
  vector<lower=0, upper=1>[N] learnrate_posPE;
  vector<lower=0, upper=1>[N] learnrate_negPE;
  vector<lower=0, upper=1>[N] discount;
  vector<lower=0, upper=20>[N] tau;
  
  for (i in 1:N) {
    learnrate_posPE[i] = Phi_approx(mu_pr[1] + sigma_pr[1] * learnrate_posPE_pr[i]);
    learnrate_negPE[i] = Phi_approx(mu_pr[2] + sigma_pr[2] * learnrate_negPE_pr[i]);
    discount[i]        = Phi_approx(mu_pr[3] + sigma_pr[3] * discount_pr[i]);
    tau[i]             = Phi_approx(mu_pr[4] + sigma_pr[4] * tau_pr[i])*20;
  }
}

model {
  // Hyperppriors defining mean and standard deviation of {learnrate, tau, gamma} parameters
  mu_pr ~ normal(0,1);
  sigma_pr ~ normal(0,1);

  // individual parameters
  learnrate_posPE_pr ~ normal(0,1);
  learnrate_negPE_pr ~ normal(0,1);
  discount_pr        ~ normal(0,1);
  tau_pr             ~ normal(0,1);

  // subject loop and trial loop
  for (i in 1:N) {
    matrix[6,3] ev;
    int decision;
    real PE;
    real learnrate;
    vector[2] option_values = [ 0, 0 ]'; 			

    for (idx in 1:6) { ev[idx] = initV; }

    for (t in 1:T_subjs[i]) {

      decision = (choice[i, t] > 1) ? option2[i, t] : option1[i, t];

      option_values = [ ev[option1[i, t], 1] , ev[option2[i, t], 1] ]';
   
      // compute action probabilities
      target += categorical_lpmf(choice[i, t] | softmax( option_values*tau[i] ));

      for (e in 1:3) {
	      if (e < 3) {
	        PE = discount[i] * ev[decision, e+1] - ev[decision, e];
          learnrate = (PE >= 0) ? learnrate_posPE[i] : learnrate_negPE[i];
          ev[decision, e] += learnrate * PE;
	      } else {
	        PE = outcome[i, t] - ev[decision, e];
          learnrate = (PE >= 0) ? learnrate_posPE[i] : learnrate_negPE[i];
          ev[decision, e] += learnrate * PE;
	      }
      }
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_learnrate_posPE;
  real<lower=0, upper=1> mu_learnrate_negPE;
  real<lower=0, upper=1> mu_discount;
  real<lower=0, upper=20> mu_tau;

  // For log-likelihood values and posterior predictive check
  real log_lik[N];
  real y_pred[N, T_max];

  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
   for (t in 1:T_max) {
     y_pred[i,t] = -1;
   }
  }
    
  mu_learnrate_posPE   = Phi_approx(mu_pr[1]);
  mu_learnrate_negPE   = Phi_approx(mu_pr[2]);
  mu_discount          = Phi_approx(mu_pr[3]);
  mu_tau               = Phi_approx(mu_pr[4])*20;

  { 

  for (i in 1:N) {

    matrix[6,3] ev;
    int decision;
    real PE;
    real learnrate;
    vector[2] option_values = [ 0, 0 ]'; 	

    log_lik[i] = 0;

    for (idx in 1:6) { ev[idx] = initV; }

    for (t in 1:T_subjs[i]) {

      decision = (choice[i, t] > 1) ? option2[i, t] : option1[i, t];

      option_values = [ ev[option1[i, t], 1] , ev[option2[i, t], 1] ]';

      // compute log likelihood of current trial
      log_lik[i] += categorical_lpmf(choice[i, t] | softmax( option_values*tau[i] ));

      // generate posterior prediction for current trial
      y_pred[i,t] = categorical_rng( softmax( option_values*tau[i] ));

      for (e in 1:3) {

        if (e < 3) {
	        PE = discount[i] * ev[decision, e+1] - ev[decision, e];
          learnrate = (PE >= 0) ? learnrate_posPE[i] : learnrate_negPE[i];
	        ev[decision, e] += learnrate * PE;
	      } else {
	        PE = outcome[i, t] - ev[decision, e];
          learnrate = (PE >= 0) ? learnrate_posPE[i] : learnrate_negPE[i];
          ev[decision, e] += learnrate * PE;
	      }
        } // episode loop
      } // trial loop
    } // individual loop
  } // local section
}


