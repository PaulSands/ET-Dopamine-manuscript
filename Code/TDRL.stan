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
  vector[3] mu_pr;
  vector<lower=0>[3] sigma_pr;

  // Subject-level raw parameters 
  vector[N] learnrate_pr;
  vector[N] discount_pr;
  vector[N] tau_pr;
}

transformed parameters {
  // subject-level parameters
  vector<lower=0, upper=1>[N] learnrate;
  vector<lower=0, upper=1>[N] discount;
  vector<lower=0, upper=20>[N] tau;
  
  for (i in 1:N) {
    learnrate[i] = Phi_approx(mu_pr[1] + sigma_pr[1] * learnrate_pr[i]);
    discount[i]  = Phi_approx(mu_pr[2] + sigma_pr[2] * discount_pr[i]);
    tau[i]       = Phi_approx(mu_pr[3] + sigma_pr[3] * tau_pr[i])*20;
  }
}

model {
  // Hyperppriors defining mean and standard deviation of {learnrate, tau, gamma} parameters
  mu_pr ~ normal(0,1);
  sigma_pr ~ normal(0,1);

  // individual parameters
  learnrate_pr ~ normal(0,1);
  discount_pr  ~ normal(0,1);
  tau_pr       ~ normal(0,1);

  // subject loop and trial loop
  for (i in 1:N) {
    matrix[6,3] ev;
    int decision;
    real PE;
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
          ev[decision, e] += learnrate[i] * PE;
	      } else {
	        PE = outcome[i, t] - ev[decision, e];
          ev[decision, e] += learnrate[i] * PE;
	      }
      }
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_learnrate;
  real<lower=0, upper=1> mu_discount;
  real<lower=0, upper=20> mu_tau;

  // For log-likelihood values and posterior predictive check
  real log_lik[N];
  real y_pred[N,T_max];

  for (i in 1:N) {
    for (t in 1:T_max) {
      y_pred[i,t] = -1;
    }
  }

  mu_learnrate   = Phi_approx(mu_pr[1]);
  mu_discount    = Phi_approx(mu_pr[2]);
  mu_tau         = Phi_approx(mu_pr[3])*20;    

  { for (i in 1:N) {

    matrix[6,3] ev;
    int decision;
    real PE;
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
          ev[decision, e] += learnrate[i] * PE;
	      } else {
	        PE = outcome[i, t] - ev[decision, e];
          ev[decision, e] += learnrate[i] * PE;
	      }
      } // episode loop
    } // trial loop
  } // individual loop
  } // local section
}



