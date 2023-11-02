clearvars -except ET_data  

os = 'Windows';
% os = 'Mac';

ET_subjID    = 5:7;
ET_volt_data = struct([]);

switch os
    case 'Mac'
        if exist('ET_data','var') == 1
        else
            load('~/Downloads/ET-Dopamine-manuscript/Data/ET_task_data.mat')
        end

        for i = 1:numel(ET_subjID)

            % Load timing data for events of interest within task
            load(['~/Downloads/ET-Dopamine-manuscript/Data/ET' num2str(ET_subjID(i)) '_timeCorrectedBehaviorVariables.mat'], ...
                'adjChoiceOnset','adjDecisionOnset','adjOutcomeOnset')
            ET_volt_data{i,1}.Choice_onset  = round(adjChoiceOnset,1);
            ET_volt_data{i,1}.Action_onset  = round(adjDecisionOnset,1);
            ET_volt_data{i,1}.Outcome_onset = round(adjOutcomeOnset,1);
            
            % Load voltammetry data
            load(['~/Downloads/ET-Dopamine-manuscript/Data/ET' num2str(ET_subjID(i)) '_TimeSeriesVar.mat'],'dopamine','ch1DataPointTimes')
            ET_volt_data{i,1}.analytes{1,1} = dopamine;
            ET_volt_data{i,1}.Timing        = round(ch1DataPointTimes,1);
        end

    case 'Windows'
        if exist('ET_data','var') == 1
        else
            load('C:\Users\psands\Downloads\ET-Dopamine-manuscript\Data\ET_task_data.mat')
        end

        for i = 1:numel(ET_subjID)

            % Load timing data for events of interest within task
            load(['C:\Users\psands\Downloads\ET-Dopamine-manuscript\Data\ET' num2str(ET_subjID(i)) '_timeCorrectedBehaviorVariables.mat'], ...
                'adjChoiceOnset','adjDecisionOnset','adjOutcomeOnset')
            ET_volt_data{i,1}.Choice_onset  = round(adjChoiceOnset,1);
            ET_volt_data{i,1}.Action_onset  = round(adjDecisionOnset,1);
            ET_volt_data{i,1}.Outcome_onset = round(adjOutcomeOnset,1);
            
            % Load voltammetry data
            load(['C:\Users\psands\Downloads\ET-Dopamine-manuscript\Data\ET' num2str(ET_subjID(i)) '_TimeSeriesVar.mat'],'dopamine','ch1DataPointTimes')
            ET_volt_data{i,1}.analytes{1,1} = dopamine;
            ET_volt_data{i,1}.Timing        = round(ch1DataPointTimes,1);
        end
end

clc
disp('Done loading data')


%% Generate RL model-based variables (i.e., prediction errors) from fitted participant model parameters and choice data
models = {'TDRL','VPRL'};

for i = 1:numel(ET_subjID)
    choice         = ET_data{i,1}.Choices;
    decision       = ET_data{i,1}.Decision;
    reward_outcome = ET_data{i,1}.Outcome;
    for m = 1:numel(models)
        model_tmp = char(models(m));
        ET_data = get_RLmodelbased_variables(ET_subjID(i),model_tmp,ET_data);
    end
end
        

%% Voltammetry data pre-processing and sorting

% Initialize variables for pre-processing DA time series data
sampling_rate = 0.1;                            % Voltammetry data is 10Hz
time_window   = [0 0.7];                        % Define peri-event time window (default: 0-700msec, 0-time-point = event onset)
time_samples  = time_window./sampling_rate;     % Number of data points to include in analysis
gap_bt_trials = 10;                             % Define how much of a time gap to have between trials when cutting out individual trial data (default = 1 second)
smooth_window = 3;                              % Define how many samples to include in smoothing of time series (default = 300 ms)

% Initialize variables for storing parsed DA time series data
dat_tot             = 0;
DA_fulltrial        = struct([]);
trial_type          = nan(numel(ET_subjID),150);
TD_RPE_choice_full  = nan(numel(ET_subjID),150);
TD_RPE_action_full  = nan(numel(ET_subjID),150);
TD_RPE_outcome_full = nan(numel(ET_subjID),150);
VP_RPE_choice_full  = nan(numel(ET_subjID),150);
VP_RPE_action_full  = nan(numel(ET_subjID),150);
VP_RPE_outcome_full = nan(numel(ET_subjID),150);
VP_PPE_choice_full  = nan(numel(ET_subjID),150);
VP_PPE_action_full  = nan(numel(ET_subjID),150);
VP_PPE_outcome_full = nan(numel(ET_subjID),150);


for i = 1:numel(ET_subjID)
    
    subj_idx = ET_subjID(i);    
    nTrials  = numel(ET_data{subj_idx,1}.Decision);
        
    for t = 1:nTrials            
        choice_idx  = find(ET_volt_data{i,1}.Timing == ET_volt_data{i,1}.Choice_onset(t,1));
        outcome_idx = find(ET_volt_data{i,1}.Timing == ET_volt_data{i,1}.Outcome_onset(t,1));
        if t < nTrials
            choice_next_idx = find(ET_volt_data{i,1}.Timing == ET_volt_data{i,1}.Choice_onset(t+1,1));
            if isempty(choice_next_idx) == 1
                ET_volt_data{i,1}.analytes{1,2}.Full_trial{t,1} = nan(1,(sum(samples)+1));
            else
                ET_volt_data{i,1}.analytes{1,2}.Full_trial{t,1} = ET_volt_data{i,1}.analytes{1,1}((choice_idx-gap_bt_trials):(choice_next_idx-1));
            end
        else
            ET_volt_data{i,1}.analytes{1,2}.Full_trial{t,1} = ET_volt_data{i,1}.analytes{1,1}((choice_idx-gap_bt_trials):(outcome_idx+30));
        end

        % Z-score then smooth trial-level data
        ET_volt_data{i,1}.analytes{1,3}.Full_trial{t,:} = zscore(ET_volt_data{i,1}.analytes{1,2}.Full_trial{t,:});
        ET_volt_data{i,1}.analytes{1,4}.Full_trial{t,1} = movmean(ET_volt_data{i,1}.analytes{1,3}.Full_trial{t,1},[smooth_window-1 0]);
    end                
    
    % Compile pre-processed trial-level DA time series data into 'DA_fulltrial' data structure
    for t = (dat_tot+1):(dat_tot+nTrials)
        DA_fulltrial{t,1} = ET_volt_data{i,1}.analytes{1,4}.Full_trial{t-dat_tot,1};                                                                        % trial DA time series data
        DA_fulltrial{t,2} = (gap_bt_trials + 1) + round((ET_volt_data{i,1}.Action_onset(t-dat_tot,1) - ET_volt_data{i,1}.Choice_onset(t-dat_tot,1))*10);    % trial response time (in seconds)
        DA_fulltrial{t,3} = DA_fulltrial{t,2}+30;                                                                                                           % trial outcome reveal time (3 seconds post-action)
    end
    
    % Compile prediction error values for TDRL and VPRL models for all ET patients
    TD_RPE_choice_full(i,1:nTrials)  = ET_data{subj_idx,1}.Models.TDRL.Choice.RPE;
    TD_RPE_action_full(i,1:nTrials)  = ET_data{subj_idx,1}.Models.TDRL.Action.RPE;
    TD_RPE_outcome_full(i,1:nTrials) = ET_data{subj_idx,1}.Models.TDRL.Outcome.RPE;
    VP_RPE_choice_full(i,1:nTrials)  = ET_data{subj_idx,1}.Models.VPRL.Choice.RPE;
    VP_RPE_action_full(i,1:nTrials)  = ET_data{subj_idx,1}.Models.VPRL.Action.RPE;
    VP_RPE_outcome_full(i,1:nTrials) = ET_data{subj_idx,1}.Models.VPRL.Outcome.RPE;
    VP_PPE_choice_full(i,1:nTrials)  = ET_data{subj_idx,1}.Models.VPRL.Choice.PPE;
    VP_PPE_action_full(i,1:nTrials)  = ET_data{subj_idx,1}.Models.VPRL.Action.PPE;
    VP_PPE_outcome_full(i,1:nTrials) = ET_data{subj_idx,1}.Models.VPRL.Outcome.PPE;
    
    dat_tot = dat_tot + nTrials;    
end

TD_RPE_choice  = rmmissing(reshape(TD_RPE_choice_full',size(TD_RPE_choice_full,1)*size(TD_RPE_choice_full,2),1));
TD_RPE_action  = rmmissing(reshape(TD_RPE_action_full',size(TD_RPE_action_full,1)*size(TD_RPE_action_full,2),1));
TD_RPE_outcome = rmmissing(reshape(TD_RPE_outcome_full',size(TD_RPE_outcome_full,1)*size(TD_RPE_outcome_full,2),1));
VP_RPE_choice  = rmmissing(reshape(VP_RPE_choice_full',size(VP_RPE_choice_full,1)*size(VP_RPE_choice_full,2),1));
VP_RPE_action  = rmmissing(reshape(VP_RPE_action_full',size(VP_RPE_action_full,1)*size(VP_RPE_action_full,2),1));
VP_RPE_outcome = rmmissing(reshape(VP_RPE_outcome_full',size(VP_RPE_outcome_full,1)*size(VP_RPE_outcome_full,2),1));
VP_PPE_choice  = rmmissing(reshape(VP_PPE_choice_full',size(VP_PPE_choice_full,1)*size(VP_PPE_choice_full,2),1));
VP_PPE_action  = rmmissing(reshape(VP_PPE_action_full',size(VP_PPE_action_full,1)*size(VP_PPE_action_full,2),1));
VP_PPE_outcome = rmmissing(reshape(VP_PPE_outcome_full',size(VP_PPE_outcome_full,1)*size(VP_PPE_outcome_full,2),1));


%% Segment full-trial DA time series into peri-event windows, baseline correct, and sort by RL model & prediction error type
choice_ts  = nan(size(TD_RPE_choice,1),1+round(sum(time_samples)));
action_ts  = nan(size(TD_RPE_action,1),1+round(sum(time_samples)));
outcome_ts = nan(size(TD_RPE_outcome,1),1+round(sum(time_samples)));
tb         = 3;   % amount of time to use for baseline correction (default = 300 ms preceeding event)

for i = 1:size(TD_RPE_choice,1)
    % Baseline correct to mean of baseline window 300 msec prior to event onset
    choice_ts(i,:)  = (DA_fulltrial{i,1}((gap_bt_trials+1)-time_samples(1):(gap_bt_trials+1)+time_samples(2)) - mean(DA_fulltrial{i,1}((gap_bt_trials+1-tb):(gap_bt_trials+1))))./std(DA_fulltrial{i,1}((gap_bt_trials+1-tb):(gap_bt_trials+1)));
    action_ts(i,:)  = (DA_fulltrial{i,1}(DA_fulltrial{i,2}-time_samples(1):DA_fulltrial{i,2}+time_samples(2)) - mean(DA_fulltrial{i,1}((DA_fulltrial{i,2}-tb):DA_fulltrial{i,2})))./std(DA_fulltrial{i,1}((DA_fulltrial{i,2}-tb):DA_fulltrial{i,2}));
    outcome_ts(i,:) = (DA_fulltrial{i,1}(DA_fulltrial{i,3}-time_samples(1):DA_fulltrial{i,3}+time_samples(2)) - mean(DA_fulltrial{i,1}((DA_fulltrial{i,3}-tb):DA_fulltrial{i,3})))./std(DA_fulltrial{i,1}((DA_fulltrial{i,3}-tb):DA_fulltrial{i,3}));
end
all_states_ts = [choice_ts ; action_ts ; outcome_ts];

% Define reward-associated trials and punishment-associated trials
idx_trial_type = nan(size(DA_fulltrial,1),1);
dat_tot = 0;
for i = 1:numel(ET_subjID)
    
    dat_add = numel(ET_data{ET_subjID(i),1}.Decision);
    dat_range = dat_tot+1:dat_tot+dat_add;
    
    for j = 1:numel(dat_range)
        if ET_data{ET_subjID(i),1}.Decision(j) < 4
            if ET_data{ET_subjID(i),1}.Outcome(j) > 0
                idx_trial_type(dat_range(j),1) = 1;
            else
                idx_trial_type(dat_range(j),1) = 2;
            end
        else
            if ET_data{ET_subjID(i),1}.Outcome(j) < 0
                idx_trial_type(dat_range(j),1) = 3;
            else
                idx_trial_type(dat_range(j),1) = 4;
            end
        end
    end
    dat_tot = dat_tot + dat_add;
end

% Define threshold for defining positive and negative prediction errors and sort DA responses by prediction error type
pos_threshold = 0;
neg_threshold = 0;

all_trial_TDRL_posRPE    = [choice_ts(TD_RPE_choice>pos_threshold,:) ; ...
                            action_ts(TD_RPE_action>pos_threshold,:) ; ...
                            outcome_ts(TD_RPE_outcome>pos_threshold,:)];
all_trial_TDRL_negRPE    = [choice_ts(TD_RPE_choice<neg_threshold,:) ; ...
                            action_ts(TD_RPE_action<neg_threshold,:) ; ...
                            outcome_ts(TD_RPE_outcome<neg_threshold,:)];
reward_trial_TDRL_posRPE = [choice_ts(idx_trial_type<3&TD_RPE_choice>pos_threshold,:) ; ...
                            action_ts(idx_trial_type<3&TD_RPE_action>pos_threshold,:) ; ...
                            outcome_ts(idx_trial_type<3&TD_RPE_outcome>pos_threshold,:)];
reward_trial_TDRL_negRPE = [choice_ts(idx_trial_type<3&TD_RPE_choice<neg_threshold,:) ; ...
                            action_ts(idx_trial_type<3&TD_RPE_action<neg_threshold,:) ; ...
                            outcome_ts(idx_trial_type<3&TD_RPE_outcome<neg_threshold,:)];
punish_trial_TDRL_posRPE = [choice_ts(idx_trial_type>2&TD_RPE_choice>pos_threshold,:) ; ...
                            action_ts(idx_trial_type>2&TD_RPE_action>pos_threshold,:) ; ...
                            outcome_ts(idx_trial_type>2&TD_RPE_outcome>pos_threshold,:)];
punish_trial_TDRL_negRPE = [choice_ts(idx_trial_type>2&TD_RPE_choice<neg_threshold,:) ; ...
                            action_ts(idx_trial_type>2&TD_RPE_action<neg_threshold,:) ; ...
                            outcome_ts(idx_trial_type>2&TD_RPE_outcome<neg_threshold,:)];
                                   
                                   
all_trial_VPRL_posRPE = [choice_ts(VP_RPE_choice>pos_threshold,:) ; ...
                         action_ts(VP_RPE_action>pos_threshold,:) ; ...
                         outcome_ts(VP_RPE_outcome>pos_threshold,:)];
all_trial_VPRL_negRPE = [choice_ts(VP_RPE_choice<neg_threshold,:) ; ...
                         action_ts(VP_RPE_action<neg_threshold,:) ; ...
                         outcome_ts(VP_RPE_outcome<neg_threshold,:)];
all_trial_VPRL_posPPE = [choice_ts(VP_PPE_choice>pos_threshold,:) ; ...
                         action_ts(VP_PPE_action>pos_threshold,:) ; ...
                         outcome_ts(VP_PPE_outcome>pos_threshold,:)];
all_trial_VPRL_negPPE = [choice_ts(VP_PPE_choice<neg_threshold,:) ; ...
                         action_ts(VP_PPE_action<neg_threshold,:) ; ...
                         outcome_ts(VP_PPE_outcome<neg_threshold,:)];


%% Plot Main Text Figure 2

ylimit = [-1 1];
xlimit = [1 size(all_states_ts,2)];

figure
set(gcf,'color','w');

subplot(4,4,[1 2 5 6])
hold on
options.color = 'Dark Grey';
plot_areaerrorbar_2color(all_trial_TDRL_posRPE,options)
options.color = 'Light Grey';
plot_areaerrorbar_2color(all_trial_TDRL_negRPE,options)
line([0 round(sum(time_samples))+1],[0 0],'Color','k','LineWidth',2,'LineStyle','--','HandleVisibility','off')
ylim(ylimit)
ylabel('[Dopamine] (z-scored)')
xlim(xlimit)
xticks(1:xlimit(2))
xticklabels(round((-1*time_samples(1):time_samples(2)+1)*100))
legend({[' TD-RPE > 0 (n=' num2str(size(all_trial_TDRL_posRPE,1)) ')'] , ...
        [' TD-RPE < 0 (n=' num2str(size(all_trial_TDRL_negRPE,1)) ')']},'box','off')
set(gca,'FontSize',20)
xlabel('Time (msec)')


subplot(4,4,[9 13])
hold on
options.color = 'Dark Green';
plot_areaerrorbar_2color(reward_trial_TDRL_posRPE,options)
options.color = 'Light Green';
plot_areaerrorbar_2color(reward_trial_TDRL_negRPE,options)
line([0 round(sum(time_samples))+1],[0 0],'Color','k','LineWidth',2,'LineStyle','--','HandleVisibility','off')
ylim(ylimit)
ylabel('[Dopamine] (z-scored)')
xlim(xlimit)
xticks([1 xlimit(2)])
xticklabels([0 round((time_samples(2))*100)])
legend({[' TD-RPE > 0 (n=' num2str(size(reward_trial_TDRL_posRPE,1)) ')'] , ...
        [' TD-RPE < 0 (n=' num2str(size(reward_trial_TDRL_negRPE,1)) ')']},'box','off','FontSize',14)
set(gca,'FontSize',20)
xlabel('Time (msec)')


subplot(4,4,[10 14])
hold on
options.color = 'Dark Pink';
plot_areaerrorbar_2color(punish_trial_TDRL_posRPE,options)
options.color = 'Light Pink';
plot_areaerrorbar_2color(punish_trial_TDRL_negRPE,options)
line([0 round(sum(time_samples))+1],[0 0],'Color','k','LineWidth',2,'LineStyle','--','HandleVisibility','off')
ylim(ylimit)
xlim(xlimit)
xticks(1:xlimit(2))
xticklabels([0 round((time_samples(2))*100)])
legend({[' TD-RPE > 0 (n=' num2str(size(punish_trial_TDRL_posRPE,1)) ')'] , ...
        [' TD-RPE < 0 (n=' num2str(size(punish_trial_TDRL_negRPE,1)) ')']},'box','off','FontSize',14)
set(gca,'FontSize',20)
xlabel('Time (msec)')


subplot(4,4,[3 4 7 8])
hold on
options.color = 'Dark Green';
plot_areaerrorbar_2color(all_trial_VPRL_posRPE,options)
options.color = 'Light Green';
plot_areaerrorbar_2color(all_trial_VPRL_negRPE,options)
line([0 round(sum(time_samples))+1],[0 0],'Color','k','LineWidth',2,'LineStyle','--','HandleVisibility','off')
ylim(ylimit)
ylabel('[Dopamine] (z-scored)')
xlim(xlimit)
xticks(1:xlimit(2))
xticklabels(round((-1*time_samples(1):time_samples(2)+1)*100))
xlabel('Time (msec)')
set(gca,'FontSize',20)
legend({[' VP-RPE > 0 (n=' num2str(size(all_trial_VPRL_posRPE,1)) ')'] , ...
        [' VP-RPE < 0 (n=' num2str(size(all_trial_VPRL_negRPE,1)) ')']},'box','off')
    
    
subplot(4,4,[11 12 15 16])
hold on
options.color = 'Dark Pink';
plot_areaerrorbar_2color(all_trial_VPRL_posPPE,options)
options.color = 'Light Pink';
plot_areaerrorbar_2color(all_trial_VPRL_negPPE,options)
line([0 round(sum(time_samples))+1],[0 0],'Color','k','LineWidth',2,'LineStyle','--','HandleVisibility','off')
ylim(ylimit)
ylabel('[Dopamine] (z-scored)')
xlim(xlimit)
xticks(1:xlimit(2))
xticklabels(round((-1*time_samples(1):time_samples(2)+1)*100))
xlabel('Time (msec)')
set(gca,'FontSize',20)
legend({[' VP-PPE > 0 (n=' num2str(size(all_trial_VPRL_posPPE,1)) ')'] , ...
        [' VP-PPE < 0 (n=' num2str(size(all_trial_VPRL_negPPE,1)) ')']},'box','off')



%% Logistic classifier analysis

% Label positive and negative TDRL prediction error time series as {1,0}, respectively
reward_TD = [[reward_trial_TDRL_posRPE(:,round(time_samples(1)+1):end) , 1*ones(size(reward_trial_TDRL_posRPE,1),1)] ; ...
             [reward_trial_TDRL_negRPE(:,round(time_samples(1)+1):end) , 0*ones(size(reward_trial_TDRL_negRPE,1),1)]];
punish_TD = [[punish_trial_TDRL_posRPE(:,round(time_samples(1)+1):end) , 1*ones(size(punish_trial_TDRL_posRPE,1),1)] ; ...
             [punish_trial_TDRL_negRPE(:,round(time_samples(1)+1):end) , 0*ones(size(punish_trial_TDRL_negRPE,1),1)]];

% Define time windows for decoding -- 0-300msec for positive prediction errors ; 400-700msec for negative prediction errors
TDreward_window = 1:4;
TD_dat1_resp = reward_TD(:,TDreward_window);   % DA response associated with TDRL reward prediction errors
TD_dat1_lab  = reward_TD(:,end);               % labels associated with TDRL reward prediction errors

TDpunish_window = 5:8;
TD_dat2_resp = punish_TD(:,TDpunish_window);   % DA response associated with TDRL punishment prediction errors
TD_dat2_lab  = punish_TD(:,end);               % labels associated with TDRL punishment prediction errors

% Run models
Log_TD_reward                            = fitglm(TD_dat1_resp,TD_dat1_lab,'Distribution','binomial','Link','logit');
score_logTD1                             = Log_TD_reward.Fitted.Probability;
[XlogTD1,YlogTD1,TlogTD1,AUClogTD1_true] = perfcurve(TD_dat1_lab,score_logTD1,1);

Log_TD_punish                            = fitglm(TD_dat2_resp,TD_dat2_lab,'Distribution','binomial','Link','logit');
score_logTD2                             = Log_TD_punish.Fitted.Probability;
[XlogTD2,YlogTD2,TlogTD2,AUClogTD2_true] = perfcurve(TD_dat2_lab,score_logTD2,1);



% Label positive and negative VPRL prediction error time series as {1,0}, respectively
reward_VP = [[all_trial_VPRL_posRPE(:,round(time_samples(1)+1):end) , 1*ones(size(all_trial_VPRL_posRPE,1),1)] ; ...
             [all_trial_VPRL_negRPE(:,round(time_samples(1)+1):end) , 0*ones(size(all_trial_VPRL_negRPE,1),1)]];
punish_VP = [[all_trial_VPRL_posPPE(:,round(time_samples(1)+1):end) , 1*ones(size(all_trial_VPRL_posPPE,1),1)] ; ...
             [all_trial_VPRL_negPPE(:,round(time_samples(1)+1):end) , 0*ones(size(all_trial_VPRL_negPPE,1),1)]];

% Define time windows for decoding -- 0-300msec for positive prediction errors ; 400-700msec for negative prediction errors
VPreward_window = 1:4;
VP_dat1_resp    = reward_VP(:,VPreward_window);   % DA response associated with VPRL reward prediction errors
VP_dat1_lab     = reward_VP(:,end);               % labels associated with VPRL reward prediction errors

VPpunish_window = 5:8;
VP_dat2_resp    = punish_VP(:,VPpunish_window);   % DA response associated with VPRL punishment prediction errors
VP_dat2_lab     = punish_VP(:,end);               % labels associated with VPRL punishment prediction errors

% Run models
Log_VP_reward                            = fitglm(VP_dat1_resp,VP_dat1_lab,'Distribution','binomial','Link','logit');
score_logVP1                             = Log_VP_reward.Fitted.Probability;
[XlogVP1,YlogVP1,TlogVP1,AUClogVP1_true] = perfcurve(VP_dat1_lab,score_logVP1,1);

Log_VP_punish                            = fitglm(VP_dat2_resp,VP_dat2_lab,'Distribution','binomial','Link','logit');
score_logVP2                             = Log_VP_punish.Fitted.Probability;
[XlogVP2,YlogVP2,TlogVP2,AUClogVP2_true] = perfcurve(VP_dat2_lab,score_logVP2,1);


%% Permutation testing of logistic classifier performance
% Initialize variables and matrices to store outputs
perm_iter = 10000;    % number of permutations (default = 10000)    
AUClogTD1 = zeros(1,perm_iter);     % for storing permuted TDRL reward prediction error classifier results
AUClogTD2 = zeros(1,perm_iter);     % for storing permuted TDRL punishment prediction error classifier results
AUClogVP1 = zeros(1,perm_iter);     % for storing permuted VPRL reward prediction error classifier results
AUClogVP2 = zeros(1,perm_iter);     % for storing permuted VPRL punishment prediction error classifier results

% parallelize permutations
parfor i = 1:perm_iter
    
    idx = randperm(numel(TD_dat1_lab));
    Log_TD_reward        = fitglm(TD_dat1_resp,TD_dat1_lab(idx),'Distribution','binomial','Link','logit');
    score_logTD1         = Log_TD_reward.Fitted.Probability;
    [~,~,~,AUClogTD1(i)] = perfcurve(TD_dat1_lab(idx),score_logTD1,1);
    
    idx = randperm(numel(TD_dat2_lab));
    Log_TD_punish        = fitglm(TD_dat2_resp,TD_dat2_lab(idx),'Distribution','binomial','Link','logit');
    score_logTD2         = Log_TD_punish.Fitted.Probability;
    [~,~,~,AUClogTD2(i)] = perfcurve(TD_dat2_lab(idx),score_logTD2,1);
    
    idx = randperm(numel(VP_dat1_lab));
    Log_VP_reward = fitglm(VP_dat1_resp,VP_dat1_lab(idx),'Distribution','binomial','Link','logit');
    score_logVP1 = Log_VP_reward.Fitted.Probability;
    [~,~,~,AUClogVP1(i)] = perfcurve(VP_dat1_lab(idx),score_logVP1,1);
    
    idx = randperm(numel(VP_dat2_lab));
    Log_VP_punish = fitglm(VP_dat2_resp,VP_dat2_lab(idx),'Distribution','binomial','Link','logit');
    score_logVP2 = Log_VP_punish.Fitted.Probability;
    [~,~,~,AUClogVP2(i)] = perfcurve(VP_dat2_lab(idx),score_logVP2,1);

end

% Find p-value of permutation tests for TDRL and VPRL reward prediction errors
VP1_perm_p = sum(AUClogVP1 > AUClogVP1_true)/perm_iter;
TD1_perm_p = sum(AUClogTD1 > AUClogTD1_true)/perm_iter;

% Find p-value of permutation tests for TDRL and VPRL punishment prediction errors
VP2_perm_p = sum(AUClogVP2 > AUClogVP2_true)/perm_iter;
TD2_perm_p = sum(AUClogTD2 > AUClogTD2_true)/perm_iter;

% Compute p-value of difference in classifier performances
Diff1_perm_p = sum((AUClogVP1-AUClogTD1)>(AUClogVP1_true-AUClogTD1_true))/perm_iter;
Diff2_perm_p = sum((AUClogVP2-AUClogTD2)>(AUClogVP2_true-AUClogTD2_true))/perm_iter;


%% Plot Main Text Figure 2

figure
set(gcf,'color','w')

subplot(2,2,1)
plot(XlogTD1,YlogTD1,'LineWidth',2,'Color','r')
hold on
plot(XlogVP1,YlogVP1,'LineWidth',2,'Color','b')
hold on
line(0:0.01:1,0:0.01:1,'LineStyle','--','Color','k','HandleVisibility','off')
set(gca,'FontSize',18)
legend({['TDRL auROC = ' num2str(AUClogTD1_true,2)] , ['VPRL auROC = ' num2str(AUClogVP1_true,2)]},'Location','Northwest','box','off')
title(' Logistic Reg: RPE>0 or RPE<0')
xlabel('False positive rate')
ylabel('True positive rate')

subplot(2,2,2)
plot(XlogTD2,YlogTD2,'LineWidth',2,'Color','r')
hold on
plot(XlogVP2,YlogVP2,'LineWidth',2,'Color','b')
hold on
line(0:0.01:1,0:0.01:1,'LineStyle','--','Color','k','HandleVisibility','off')
set(gca,'FontSize',18)
legend({['TDRL auROC = ' num2str(AUClogTD2_true,2)] , ['VPRL auROC = ' num2str(AUClogVP2_true,2)]},'Location','Northwest','box','off')
title(' Logistic Reg: PPE>0 or PPE<0')
xlabel('False positive rate')
ylabel('True positive rate')

subplot(2,2,3)
histogram(AUClogVP1-AUClogTD1,100,'FaceColor',[0.5 0.5 0.5])
hold on
line([AUClogVP1_true-AUClogTD1_true AUClogVP1_true-AUClogTD1_true],[0 200],'Color','k','LineWidth',4)
set(gca,'FontSize',20)
xlabel('Difference in auROC')
ylabel('PDF')
box off
title('Difference in classifier performance (RPE)')

subplot(2,2,4)
histogram(AUClogVP2-AUClogTD2,100,'FaceColor',[0.5 0.5 0.5])
hold on
line([AUClogVP2_true-AUClogTD2_true AUClogVP2_true-AUClogTD2_true],[0 200],'Color','k','LineWidth',4)
set(gca,'FontSize',20)
xlabel('Difference in auROC')
ylabel('PDF')
box off
title('Difference in classifier performance (PPE)')



%% Functions
function ET_data = get_RLmodelbased_variables(idx,model,data)

i       = idx;
ET_data = data;

nOptions = 6;
nEpisode = 4;

switch model
    case 'TDRL'
        alpha           = ET_data{i,1}.Models.TDRL.pars(1,1);
        gamma           = ET_data{i,1}.Models.TDRL.pars(1,3);
        nTrials         = numel(ET_data{i,1}.Decision);
        outcome_episode = [zeros(2,nTrials) ; ET_data{i,1}.Outcome ; zeros(1,nTrials)];
        decision        = ET_data{i,1}.Decision;

        Q_episode     = zeros(nOptions,nEpisode);
        Delta_episode = zeros(nOptions,nEpisode);

        for t = 1:nTrials
            for e = 1:nEpisode-1
                Delta_episode(decision(1,t),e) = outcome_episode(e,t) + gamma*Q_episode(decision(1,t),e+1) - Q_episode(decision(1,t),e);
                Q_episode(decision(1,t),e)     = Q_episode(decision(1,t),e) + alpha*Delta_episode(decision(1,t),e);
            end
            ET_data{i,1}.Models.TDRL.Choice.RPE(1,t)  = Delta_episode(decision(1,t),1);
            ET_data{i,1}.Models.TDRL.Action.RPE(1,t)  = Delta_episode(decision(1,t),2);
            ET_data{i,1}.Models.TDRL.Outcome.RPE(1,t) = Delta_episode(decision(1,t),3);
        end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    case 'VPRL'
        alpha_rew       = ET_data{i,1}.Models.VPRL.pars(1,1);
        alpha_pun       = ET_data{i,1}.Models.VPRL.pars(1,2);
        gamma_rew       = ET_data{i,1}.Models.VPRL.pars(1,4);
        gamma_pun       = ET_data{i,1}.Models.VPRL.pars(1,5);
        nTrials         = numel(ET_data{i,1}.Decision);
        outcome_episode = [zeros(2,nTrials) ; ET_data{i,1}.Outcome ; zeros(1,nTrials)];
        decision        = ET_data{i,1}.Decision;

        Q_episode         = zeros(nOptions,nEpisode);
        Q_episode_rew     = zeros(nOptions,nEpisode);
        Q_episode_pun     = zeros(nOptions,nEpisode);
        Delta_episode_rew = zeros(nOptions,nEpisode);
        Delta_episode_pun = zeros(nOptions,nEpisode);

        for t = 1:nTrials
            for e = 1:nEpisode-1
                if outcome_episode(e,t) > 0
                    Delta_episode_rew(decision(1,t),e) = outcome_episode(e,t) + gamma_rew*Q_episode_rew(decision(1,t),e+1) - Q_episode_rew(decision(1,t),e);
                    Delta_episode_pun(decision(1,t),e) = 0 + gamma_pun*Q_episode_pun(decision(1,t),e+1) - Q_episode_pun(decision(1,t),e);
                elseif outcome_episode(e,t) == 0
                    Delta_episode_rew(decision(1,t),e) = 0 + gamma_rew*Q_episode_rew(decision(1,t),e+1) - Q_episode_rew(decision(1,t),e);
                    Delta_episode_pun(decision(1,t),e) = 0 + gamma_pun*Q_episode_pun(decision(1,t),e+1) - Q_episode_pun(decision(1,t),e);
                elseif outcome_episode(e,t) < 0
                    Delta_episode_rew(decision(1,t),e) = 0 + gamma_rew*Q_episode_rew(decision(1,t),e+1) - Q_episode_rew(decision(1,t),e);
                    Delta_episode_pun(decision(1,t),e) = abs(outcome_episode(e,t)) + gamma_pun*Q_episode_pun(decision(1,t),e+1) - Q_episode_pun(decision(1,t),e);
                end
                Q_episode_rew(decision(1,t),e) = Q_episode_rew(decision(1,t),e) + alpha_rew*Delta_episode_rew(decision(1,t),e);
                Q_episode_pun(decision(1,t),e) = Q_episode_pun(decision(1,t),e) + alpha_pun*Delta_episode_pun(decision(1,t),e);
                Q_episode(decision(1,t),e)     = Q_episode_rew(decision(1,t),e) - Q_episode_pun(decision(1,t),e);

                ET_data{i,1}.Models.VPRL.Choice.RPE(1,t)  = Delta_episode_rew(decision(1,t),1);
                ET_data{i,1}.Models.VPRL.Choice.PPE(1,t)  = Delta_episode_pun(decision(1,t),1);
                ET_data{i,1}.Models.VPRL.Action.RPE(1,t)  = Delta_episode_rew(decision(1,t),2);
                ET_data{i,1}.Models.VPRL.Action.PPE(1,t)  = Delta_episode_pun(decision(1,t),2);
                ET_data{i,1}.Models.VPRL.Outcome.RPE(1,t) = Delta_episode_rew(decision(1,t),3);
                ET_data{i,1}.Models.VPRL.Outcome.PPE(1,t) = Delta_episode_pun(decision(1,t),3);

            end
        end
end
end



% ----------------------------------------------------------------------- %
% Function plot_areaerrorbar plots the mean and standard deviation of a   %
% set of data filling the space between the positive and negative mean    %
% error using a semi-transparent background, completely customizable.     %
%                                                                         %
%   Input parameters:                                                     %
%       - data:     Data matrix, with rows corresponding to observations  %
%                   and columns to samples.                               %
%       - options:  (Optional) Struct that contains the customized params.%
%           * options.handle:       Figure handle to plot the result.     %
%           * options.color_area:   RGB color of the filled area.         %
%           * options.color_line:   RGB color of the mean line.           %
%           * options.alpha:        Alpha value for transparency.         %
%           * options.line_width:   Mean line width.                      %
%           * options.x_axis:       X time vector.                        %
%           * options.error:        Type of error to plot (+/-).          %
%                   if 'std',       one standard deviation;               %
%                   if 'sem',       standard error mean;                  %
%                   if 'var',       one variance;                         %
%                   if 'c95',       95% confidence interval.              %
% ----------------------------------------------------------------------- %
%   Example of use:                                                       %
%       data = repmat(sin(1:0.01:2*pi),100,1);                            %
%       data = data + randn(size(data));                                  %
%       plot_areaerrorbar(data);                                          %
% ----------------------------------------------------------------------- %
%   Author:  Victor Martinez-Cagigal                                      %
%   Date:    30/04/2018                                                   %
%   E-mail:  vicmarcag (at) gmail (dot) com                               %
% ----------------------------------------------------------------------- %

% Original function modified by Paul Sands in 2022 (psands@vt.edu)
% to allow for different colors for plotting

function plot_areaerrorbar_2color(data, options)
    % Default options
    if(nargin<=2)
        switch options.color
            case 'Light Grey'
                options.color_area = [.86 .86 .86];
                options.color_line = [.86 .86 .86];
                options.color_edge = [.49 .49 .49];
            case 'Dark Grey'
                options.color_area = [.49 .49 .49];
                options.color_line = [.49 .49 .49];
                options.color_edge = [.49 .49 .49];
            case 'Light Orange'
                options.color_area = [.95 .87 .73];
                options.color_line = [.95 .87 .73];
                options.color_edge = [.58 .39 .39];
            case 'Dark Orange'
                options.color_area = [.58 .39 .39];
                options.color_line = [.58 .39 .39];
                options.color_edge = [.58 .39 .39];
            case 'Light Blue'
                options.color_area = [.73 .83 .96];
                options.color_line = [.73 .83 .96];
                options.color_edge = [.39 .47 .64];
            case 'Dark Blue'
                options.color_area = [.39 .47 .64];
                options.color_line = [.39 .47 .64];
                options.color_edge = [.39 .47 .64];
            case 'Light Pink'
                options.color_area = [.93 .84 .84];
                options.color_line = [.93 .84 .84];
                options.color_edge = [.51 .38 .48];
            case 'Dark Pink'
                options.color_area = [.51 .38 .48];
                options.color_line = [.51 .38 .48];
                options.color_edge = [.51 .38 .48];
            case 'Light Green'
                options.color_area = [.76 .87 .78];
                options.color_line = [.76 .87 .78];
                options.color_edge = [.23 .44 .34];
            case 'Dark Green'
                options.color_area = [.23 .44 .34];
                options.color_line = [.23 .44 .34];
                options.color_edge = [.23 .44 .34];
        end
        options.face_alpha = 0.3;
        options.edge_alpha = 0.3;
        options.line_width = 4;
        options.edge_width = 2;
        options.error      = 'sem';
    end
    if(isfield(options,'x_axis')==0), options.x_axis = 1:size(data,2); end
    options.x_axis = options.x_axis(:);
    
    % Computing the mean and standard deviation of the data matrix
    data_mean = mean(data,1,'omitnan');
    data_std  = std(data,0,1,'omitnan');
    
    % Type of error plot
    switch(options.error)
        case 'std', error = data_std;
        case 'sem', error = (data_std./sqrt(size(data,1)));
        case 'var', error = (data_std.^2);
        case 'c95', error = (data_std./sqrt(size(data,1))).*1.96;
    end
    
    % Plotting the result
    x_vector = [options.x_axis', fliplr(options.x_axis')];
    patch = fill(x_vector, [data_mean+error,fliplr(data_mean-error)], options.color_area);
    set(patch, 'edgecolor', options.color_edge);
    set(patch, 'HandleVisibility','off');
    set(patch, 'FaceAlpha', options.face_alpha);
    hold on;
    plot(options.x_axis, data_mean, 'color', options.color_line, ...
        'LineWidth', options.line_width);
%     hold off;
    
end
