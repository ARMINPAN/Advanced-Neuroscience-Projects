%% HW7 - Advance Neuroscience - Evidence Accumulation - LIP and MT area
% Armin Panjehpour - 98101288

%% Part.1 - Drift Diffusion

%% Part.1.1 - a simple model of drift diffusion 
clc; close all; clear;

% a example
bias = 0;
sigma = 1;
dt = 0.01;
time_interval = 0:dt:30-dt;

[~, choice_taken] = simple_model(bias, sigma, dt, time_interval)

%% Part.1.2.1 - distribution of final evidence values over 10000 trials
clc; close all; clear;

bias = 1;
sigma = 1;
dt = 0.1;
time_interval = 0:dt:1-dt;
number_of_trials = 10^4;

choices = zeros(1,number_of_trials);
evidence_vals = cell(1,number_of_trials);
final_evidence_val = zeros(1,number_of_trials);

for i = 1:number_of_trials
    [evidence_vals{i}, choices(i)] = simple_model(bias, sigma, dt, time_interval);
    final_evidence_val(i) = sum(evidence_vals{i});
end


figure;
nbins = 20;
histfit(final_evidence_val,nbins)
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(final_evidence_val))
title("Distribution of Final Evidence Accumulated over " + number_of_trials + ...
    " Trials",'interpreter','latex');
xlabel('Final Evidence Value','interpreter','latex');
grid on; grid minor;

%% Part.1.2.2 - bias value effect on evidence accumulation
clc; close all; clear;

bias = [-1 0 0.1 1 10];
sigma = 1;
dt = 0.1;
time_interval = 0:dt:10-dt;

choices = zeros(1,length(bias));
evidence_vals = cell(1,length(bias));
final_evidence_val = zeros(1,length(bias));

for i = 1:length(bias)
    [evidence_vals{i}, choices(i)] = simple_model(bias(i), sigma, dt, time_interval);
    final_evidence_val(i) = sum(evidence_vals{i});
end


figure;
for i = 1:length(bias)
    plot(time_interval,evidence_vals{i},'LineWidth',1.5);
    hold on;
end


xlabel('Time(s)','interpreter','latex')
ylabel('Evidence Value','interpreter','latex')
title('Evidence Value Vs Time for Different Bias Values','interpreter','latex')
legend('bias = -1', 'bias = 0', 'bias = 0.1', 'bias = 1', 'bias = 10', 'location', 'best')
grid on; grid minor;

%% Part.1.3 - time length of task effect on choosing error
clc; clear; close all;

bias = 0.1;
sigma = 1;
dt = 0.1;
time_lenghts = linspace(0.5,10,100);
iterations_num = 10000;

preError = zeros(iterations_num,length(time_lenghts));
choices = zeros(iterations_num,length(length(time_lenghts)));
error = zeros(1,length(length(time_lenghts)));

for i = 1:length(time_lenghts)
    for j = 1:iterations_num
        [i j]
        time_interval = 0:dt:time_lenghts(i);
        [~, choices(j,i)] = simple_model(bias, sigma, dt, time_interval);
        preError(j,i) = choices(j,i)*bias > 0;
    end
    error(i) = length(find(preError(:,i) == 0))/iterations_num*100;
end

figure;
plot(time_lenghts,error,'LineWidth',1.5);
xlabel('Trial Time Length','interpreter','latex')
ylabel('Error','interpreter','latex')
title('Error Vs Trial Time Length','interpreter','latex')
grid on; grid minor;


%% Part.1.4 - distribution of evidence over time in 10000 trials

clc; close all; clear;

bias = 0.1;
sigma = 1;
dt = 0.1;
time_interval = 0:dt:10-dt;
number_of_trials = 10000;

choices = zeros(1,number_of_trials);
evidence_vals = zeros(length(time_interval),number_of_trials);
final_evidence_val = zeros(1,number_of_trials);


for i = 1:number_of_trials
    [evidence_vals(:,i), choices(i)] = simple_model(bias, sigma, dt, time_interval);
end

evidence_val_meanTrial = mean(evidence_vals,2);
evidence_val_stdTrial = std(evidence_vals,0,2);


figure('units','normalized','outerposition',[0 0 1 1])
nbins = 20;
subplot(1,3,1)
plot(time_interval,evidence_vals)
title("Evidence Value(t) For " + number_of_trials + ...
    " Trials",'interpreter','latex');
xlabel('Time(s)','interpreter','latex')
ylabel('Evidence Value','interpreter','latex');
grid on; grid minor;

subplot(1,3,2)
plot(time_interval,evidence_val_meanTrial,'LineWidth',1.5)
title("Mean Evidence Value(t) Over " + number_of_trials + ...
    " Trials",'interpreter','latex');
xlabel('Time(s)','interpreter','latex')
ylabel('Evidence Value','interpreter','latex');
grid on; grid minor;


subplot(1,3,3)
plot(time_interval,evidence_val_stdTrial,'LineWidth',1.5)
title("std Evidence Value(t) Over " + number_of_trials + ...
    " Trials",'interpreter','latex');
xlabel('Time(s)','interpreter','latex')
ylabel('Evidence Value','interpreter','latex');
grid on; grid minor;


%% Part.1.5 - theoric implementation using the distribution of X(==Evidence)
% proved in report: X follow a normal distribution with mean of Bt and var
% of sigma*t

clc; close all; clear;

bias = 0.1;
sigma = 1;
dt = 0.1;
time_interval = 0:dt:10-dt;
number_of_trials = 10000;
start_point = 0.6;

choices = zeros(1,number_of_trials);
evidence_vals = zeros(length(time_interval),number_of_trials);
final_evidence_val = zeros(1,number_of_trials);



for i = 1:number_of_trials
    [~, ~, choices(i)] = simple_model2(bias, sigma, start_point, ...
        dt, time_interval);
end

figure;
nbins = 2;
histogram(choices,nbins);
title("Number of Choice Below and above Starting point Over " + number_of_trials + ...
    " Trials",'interpreter','latex');
xlabel('Choice','interpreter','latex');
grid on; grid minor; 


%% Part.1.6 - Evidence Accumulation with threshold
clc; close all; clear;

bias = 0;
sigma = 1;
dt = 0.1;
start_point = 0;
pos_thresh = 10;
neg_thresh = -10;

[X, reaction_time, choice] = two_choice_trial(pos_thresh, neg_thresh, dt, ...
    sigma, start_point, bias);

figure;
plot(0:dt:reaction_time-dt,X,'LineWidth',1.5);
title("Evidence Vs Time, Reaction Time = " + reaction_time + "s, Choice = " + choice ...
    ,'interpreter','latex');
xlabel('time(s)','interpreter','latex')
ylabel('Evidence Value(t)','interpreter','latex')
grid on; grid minor;
axis square
hold on;
yline(pos_thresh,'LineWidth',1.5,'Color','r');
hold on;
yline(neg_thresh,'LineWidth',1.5,'Color','g');
ylim([neg_thresh-1 pos_thresh+1]);
legend('Drift Diffusion','choice = 1','choice = -1','Location','best')

%% Part.1.7 - reaction time on true and false choices
clc; close all; clear;

bias = 0.1;
sigma = 1;
dt = 0.1;
start_point = 0;
pos_thresh = 10;
neg_thresh = -10;
trial_num = 10000;
ture_choice = zeros(1,trial_num);
reaction_time = zeros(1,trial_num);


for i = 1:trial_num
    [~, reaction_time(i), choice] = two_choice_trial(pos_thresh, neg_thresh, dt, ...
        sigma, start_point, bias);
    true_choice(i) = choice*bias > 0;
end


reaction_time_trueChoice = reaction_time(find(true_choice == 1));
reaction_time_falseChoice = reaction_time(find(true_choice == 0));

figure;
subplot(1,2,1);
nbins = 20;
histfit(reaction_time_trueChoice,nbins,'gamma');
title('Distribution of Reaction Time for Trials with True Choice', 'interpreter', 'latex');
xlabel('Reaction Time Value', 'interpreter', 'latex');
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(reaction_time_trueChoice))

subplot(1,2,2);
nbins = 20;
histfit(reaction_time_falseChoice,nbins,'gamma');
title('Distribution of Reaction Time for Trials with False Choice', 'interpreter', 'latex');
xlabel('Reaction Time Value', 'interpreter', 'latex');
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(reaction_time_falseChoice))



%%  Part.1.8 - race diffusion model for two choices
% an extension of drift diffusion
clear; clc; close all;

% choices biases
bias = [0.1 -0.1];

% choices sigmas
sigma = [1 1];

% choices dt 
dt = [0.1 0.1];

% choices thresh - first positive, second negative
thresh = [10 -10];

start_point = 0;

choice = race_trial(start_point, thresh, dt, sigma, bias)


%% Part.1.9 - race diffusion model for two choices
% an extension of drift diffusion
clear; clc; close all;

% choices biases
bias = [0.1 -0.1];

% choices sigmas
sigma = [1 1];

% choices dt - should be equal here
dt = [0.1 0.1];

% time interval 
time_interval = 0:dt(1):20;

% choices thresh
thresh = [10 -10];

start_point = 0;

choice = extended_race_trial(start_point, thresh, dt, sigma, bias, time_interval)

%% Part.2 - Model for interaction of MT and LIP 

%% Part.2.1 -  Model for interaction of MT and LIP 
clc; close all; clear;

MT_p_values = [0.1 0.05];
LIP_weights = [0.1 -0.12];
LIP_threshold = 200;


[LIP_thresh_evidence, p_LIP, t, dt, LIP_events, MT_E_events, MT_I_events] = ...
    lip_activity(MT_p_values,LIP_weights,LIP_threshold);

figure;
hold on;

% raster plot - for mt_e, mt_i, lip neurons
mt_e = 1;
scatter(MT_E_events, mt_e*ones(1,length(MT_E_events)),'.');

mt_i = 2;
scatter(MT_I_events, mt_i*ones(1,length(MT_I_events)),'.');

lip = 3;
scatter(LIP_events, lip*ones(1,length(LIP_events)),'.');

% evidence accumulated
plot(0:dt:t,p_LIP,'LineWidth',1.5)
yline(LIP_thresh_evidence,'Color','r','LineWidth',1.5)
s = area([LIP_events(1), LIP_events(1), t, t],[0, max(p_LIP), max(p_LIP), 0],...
    'EdgeColor','none','FaceColor',[0.6350 0.0780 0.1840]);
alpha(s,.2)


xlim([0 t]);
ylim([0 max(p_LIP)]);
legend('MT Excitatory','MT Inhibitory','LIP','Evidence Accumulated(t)','Location','best')
grid on; grid minor;
xlabel('Time(s)','interpreter','latex');
title('Raster Plot for 2 MT & 1 LIP neuron','interpreter','latex');


%% Part.2.2 -  directionally oriented stimulus  - LIP MT Neurons interaction
clc; close all; clear;

n = 2000;
MT_p_values = [];
x = 0.1:0.1:180;
MT1_p_values = normpdf(x,100,10);
MT2_p_values = normpdf(x,140,10);

% tuning curves
figure;
plot(x,MT1_p_values,'LineWidth',1.5)

hold on;
plot(x,MT2_p_values,'LineWidth',1.5)


grid on; grid minor;
legend('MT 1 Neuron','MT 2 Neuron','Location','best')
title('Tuning Curves of MT Neurons','interpreter','latex')
xlabel('Stimuli Orientation','interpreter','latex');
ylabel('Orientation Preference','interpreter','latex');


% MT1 to LIP1/ MT1 to LIP2/ MT2 to LIP1/ MT2 to LIP2
MT_to_LIP_weights = [-1 1; 1 -1];
MT_p_values = [MT1_p_values; MT2_p_values];

% first LIP activity
[dt, LIP_thresh_evidence1, LIP_thresh_evidence2,...
    p_LIP1, p_LIP2, LIP1_events, LIP2_events, MT_1_events, MT_2_events]...
    = lip_activity_2(MT_p_values,MT_to_LIP_weights);





figure;
hold on;

% raster plot - for mt_e, mt_i, lip neurons
mt_1 = 1;
scatter(MT_1_events, mt_1*ones(1,length(MT_1_events)),'.');

mt_2 = 2;
scatter(MT_2_events, mt_2*ones(1,length(MT_2_events)),'.');

lip_1 = 3;
scatter(LIP1_events, lip_1*ones(1,length(LIP1_events)),'.');

lip_2 = 4;
scatter(LIP2_events, lip_2*ones(1,length(LIP2_events)),'.');



% evidence accumulated
t = 0:dt:(length(MT_p_values)-1)*dt;
s1 = plot(t,p_LIP1,'LineWidth',1.5,'Color',[1, 0, 0, 0.3])
alpha(s1,0.2)
s2 = plot(t,p_LIP2,'LineWidth',1.5,'Color',[0, 0, 1, 0.3])
alpha(s2,1)
yline(LIP_thresh_evidence1,'Color','r','LineWidth',1.5)
yline(LIP_thresh_evidence2,'Color','r','LineWidth',1.5)



xlim([0 t(end)]);
legend('MT 1','MT 2','LIP 1','LIP 2','Evidence Accumulated LIP 1(t)'...
    ,'Evidence Accumulated LIP 2(t)','Location','best')
grid on; grid minor;
xlabel('Time(s)','interpreter','latex');
title('Raster Plot for 2 MT & 2 LIP neuron','interpreter','latex');



