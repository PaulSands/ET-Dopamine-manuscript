data {
  int<lower=1> N;
  int<lower=1> T_max;
  int<lower=1, upper=T_max> T_subjs[N];
  int<lower=-1, upper=8> option1[N, T_max];
  int<lower=-1, upper=8> option2[N, T_max];
  int<lower=-1, upper=2> choice[N, T_max];
  real reward[N, T_max];
  real punish[N, T_max];
}
transformed data {
  row_vector[3] initV;
  initV = rep_row_vector(0.0, 3);
}
parameters {
  // Hyper(group)-parameters
  vector[5] mu_pr;
  vector<lower=0>[5] sigma_pr;

  // Subject-level raw parameters
  vector[N] learnrate_rew_pr;
  vector[N] learnrate_pun_pr;
  vector[N] discount_rew_pr;
  vector[N] discount_pun_pr;
  vector[N] tau_pr;
}
transformed parameters {
                        
  // subject-level parameters
  vector<lower=0, upper=1>[N] learnrate_rew;
  vector<lower=0, upper=1>[N] learnrate_pun;
  vector<lower=0, upper=1>[N] discount_rew;
  vector<lower=0, upper=1>[N] discount_pun;
  vector<lower=0, upper=20>[N] tau;

  for (i in 1:N) {
    learnrate_rew[i] = Phi_approx(mu_pr[1] + sigma_pr[1] * learnrate_rew_pr[i]);
    learnrate_pun[i] = Phi_approx(mu_pr[2] + sigma_pr[2] * learnrate_pun_pr[i]);
    discount_rew[i]  = Phi_approx(mu_pr[3] + sigma_pr[3] * discount_rew_pr[i]);
    discount_pun[i]  = Phi_approx(mu_pr[4] + sigma_pr[4] * discount_pun_pr[i]);
    tau[i] 	         = Phi_approx(mu_pr[5] + sigma_pr[5] * tau_pr[i])*20;
  }
}
model {
  // Hyperparameters
  mu_pr     ~ normal(0,1);
  sigma_pr  ~ normal(0,1);
        
  // individual parameters
  learnrate_rew_pr  ~ normal(0,1);
  learnrate_pun_pr  ~ normal(0,1);
  discount_rew_pr   ~ normal(0,1);
  discount_pun_pr   ~ normal(0,1);
  tau_pr            ~ normal(0,1);


  // subject loop and trial loop
  for (i in 1:N) {
    matrix[6,3] ev_pos; 		
    matrix[6,3] ev_neg; 		
    matrix[6,3] ev_sum;
    int decision; 		
    real PE_pos;
    real PE_neg;
    vector[2] option_values = [ 0, 0 ]';

    for (idx in 1:6) {
      ev_pos[idx] = initV;
      ev_neg[idx] = initV;
      ev_sum[idx] = initV;
    }
          
    for (t in 1:T_subjs[i]) {
            
      decision = (choice[i, t] > 1) ? option2[i, t] : option1[i, t];

      option_values = [ ev_sum[option1[i, t], 1] , ev_sum[option2[i, t], 1] ]';
            
      // compute action probabilities
      target += categorical_lpmf(choice[i,t] | softmax( option_values*tau[i] ));

      for (e in 1:3) {

        if (e < 3) {

           // prediction error and value updating
           PE_pos = discount_rew[i] * ev_pos[decision, e+1] - ev_pos[decision, e];
           ev_pos[decision, e] += learnrate_rew[i] * PE_pos;

           PE_neg = discount_pun[i] * ev_neg[decision, e+1] - ev_neg[decision, e];
           ev_neg[decision, e] += learnrate_pun[i] * PE_neg;

           ev_sum[decision, e] = ev_pos[decision, e] - ev_neg[decision, e];

        } else {

           PE_pos = reward[i, t] - ev_pos[decision, e];
           ev_pos[decision, e] += learnrate_rew[i] * PE_pos;

           PE_neg = punish[i, t] - ev_neg[decision, e];
           ev_neg[decision, e] += learnrate_pun[i] * PE_neg;

           ev_sum[decision, e] = ev_pos[decision, e] - ev_neg[decision, e];	   
        }

      } // episode loop
    } // trial loop
  } // individual loop
}
generated quantities {
                    
  // For group level parameters
  real<lower=0, upper=1> mu_learnrate_rew;
  real<lower=0, upper=1> mu_learnrate_pun;
  real<lower=0, upper=1> mu_discount_rew;
  real<lower=0, upper=1> mu_discount_pun;
  real<lower=0, upper=20> mu_tau;
            

  // For log-likelihood values and posterior predictive check
  real log_lik[N];
  real y_pred[N,T_max];

  for (i in 1:N) {
    for (t in 1:T_max) {
      y_pred[i,t] = -1;
    }
  }
           
  mu_learnrate_rew = Phi_approx(mu_pr[1]);
  mu_learnrate_pun = Phi_approx(mu_pr[2]);
  mu_discount_rew  = Phi_approx(mu_pr[3]);
  mu_discount_pun  = Phi_approx(mu_pr[4]);
  mu_tau 	         = Phi_approx(mu_pr[5])*20;

  { for (i in 1:N) {
     
    matrix[6,3] ev_pos; 		
    matrix[6,3] ev_neg; 		
    matrix[6,3] ev_sum; 
    int decision; 				
    real PE_pos;
    real PE_neg;
    vector[2] option_values = [ 0, 0 ]';

    log_lik[i] = 0;

    for (idx in 1:6) {
      ev_pos[idx] = initV;
      ev_neg[idx] = initV;
      ev_sum[idx] = initV;
    }

    for (t in 1:T_subjs[i]) {

      decision = (choice[i, t] > 1) ? option2[i, t] : option1[i, t];
            
      option_values = [ ev_sum[option1[i, t], 1] , ev_sum[option2[i, t], 1] ]';
            
      // compute action probabilities
      log_lik[i] += categorical_lpmf(choice[i, t] | softmax( option_values*tau[i] ));

      // compute posterior predicted choice
      y_pred[i,t] = categorical_rng( softmax( option_values*tau[i] ));

      for (e in 1:3) {

        if (e < 3) {

           // prediction error and value updating
           PE_pos = discount_rew[i] * ev_pos[decision, e+1] - ev_pos[decision, e];
           ev_pos[decision, e] += learnrate_rew[i] * PE_pos;

           PE_neg = discount_pun[i] * ev_neg[decision, e+1] - ev_neg[decision, e];
           ev_neg[decision, e] += learnrate_pun[i] * PE_neg;

           ev_sum[decision, e] = ev_pos[decision, e] - ev_neg[decision, e];

        } else {

           PE_pos = reward[i, t] - ev_pos[decision, e];
           ev_pos[decision, e] += learnrate_rew[i] * PE_pos;

           PE_neg = punish[i, t] - ev_neg[decision, e];
           ev_neg[decision, e] += learnrate_pun[i] * PE_neg;

           ev_sum[decision, e] = ev_pos[decision, e] - ev_neg[decision, e];	   
        }
      } // episode loop
    }  // trial loop
    }  // individual loop
  }  // local section
}
