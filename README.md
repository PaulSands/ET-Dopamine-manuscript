# ET-Dopamine-manuscript
Data and code accompanying Science Advances manuscript "Sub-second fluctuations in extracellular dopamine encode reward and punishment prediction errors in humans"

Please note that users must alter the file paths in the scripts to match their local storage of this repository.

This code is released with a permissive open-source license, and the code in this repository may be used and adapted only in compliance with the terms of the license. We appreciate attribution in the instance the code of this repository is used in future research.

The 'Data' folder contains (i) the raw task data for the eleven Essential Tremor (ET) patients (ET_task_data.mat) and the forty-two Control cohort participants (Control_task_data.mat), (ii) the task data for both the ET and Control cohorts compiled in a specific manner for input to the R script/routine for fitting reinforcement learning models via rStan (ET_stan_data.mat, Control_stan_data.mat), and (iii) the voltammetry data and timing variables for ET patients 5, 6, and 7 (the three patients included in the main analyses of the manuscript).

The 'Code' folder contains (i) the MATLAB script 'Voltammetry_analysis.m', which contains all code for analyzing dopamine time series data for replicating Main Text Figures 2 and 3, and (ii) the R script 'RL_stan_models.R' and four Stan models for fitting the reinforcement learning models to the ET patient choice behavior.

Any questions, comments, or concerns with this repository can be sent to psands@vt.edu or kkishida@wakehealth.edu.
