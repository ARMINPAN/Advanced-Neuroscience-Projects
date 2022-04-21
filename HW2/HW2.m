%% Advanced Neuroscience - Armin Panjehpour
%%% Studying the Population Response Structure 

%% part0 - importing the data 
%%%%%%%%%%%%%%%%%%%%%%%% run this part before the previous parts

% data recorded from 481 neurons in area 7a in parietal cortex
% using an electrode array in 192 3.2s trials
clc; clear; close all;
load('UnitsData.mat')
%% part1 - PSTH
clc; close all;

dt = 0.001;  
windowLength = 40; % in samples

% we know each trial is 3.2 seconds , [-1.2 2]
duration = -1.2:dt:2-dt;
PSTHs = zeros(length(Unit),length(Unit(1).Trls),length(duration)/windowLength);
time = zeros(1,length(duration)/windowLength); % x axis of the plots
% go over all neurons and calcute psth for all their trials
for i=1:length(Unit)
    % target neuron
    targetNeuronAllTrials = Unit(i).Trls;
    
    for j=1:length(targetNeuronAllTrials)
        spikeTimes = cell2mat(targetNeuronAllTrials(j));
        for k=1:length(duration)/windowLength
            PSTHs(i,j,k) = length(find((k-1)*windowLength*dt <= (spikeTimes+1.2) & ...
                (k)*windowLength*dt > (spikeTimes+1.2)))/(windowLength*dt);
            time(k) = (k-1)*windowLength*dt - 1.2;
        end
    end
end

AveragedConditionPSTHs = zeros(length(Unit),length(Unit(1).Cnd),length(duration)/windowLength);

% averaging over conditions 
for i=1:length(Unit)
    for j=1:length(Unit(1).Cnd)
        AveragedConditionPSTHs(i,j,:) = mean(PSTHs(i,Unit(i).Cnd(j).TrialIdx,:),2);
    end
end

% average over neurons
AllAveragedPSTHs = reshape(mean(AveragedConditionPSTHs,1),[size(AveragedConditionPSTHs,2) ...
    size(AveragedConditionPSTHs,3)]);

% select the neuron you want to plot the PSTHs for in 6 conditions
targetNeuron = 199;
figure; 
for i=1:length(Unit(1).Cnd)
    subplot(2,3,i);
    bar(time/dt,reshape(AveragedConditionPSTHs(targetNeuron,i,:),[1 size(AveragedConditionPSTHs,3)]),'FaceColor','#A2142F','EdgeColor','#A2142F');
    % CueOnset
    xhp = xline(0/dt,'--r',{'Cue-ONSET'},'Color','black','LineWidth',2); 
    xhp.FontSize = 8;
    % DelayOnset
    xhp = xline(0.3/dt,'--r',{'Delay-ONSET'},'Color','#0072BD','LineWidth',2); 
    xhp.FontSize = 8;
    % targetOnset
    xhp = xline(0.9/dt,'--r',{'Target-ONSET'},'Color','#77AC30','LineWidth',2); 
    xhp.FontSize = 8;
    grid on; grid minor;
    title("window length = " + windowLength+" | Condition = " + i + " | NeuronNumber = " + targetNeuron,'interpreter','latex');
    xlabel('Time(ms)','interpreter','latex');
    ylabel('Fire Rate(Hz)','interpreter','latex');
   % ylim([0 40]); % change it for better visualisation
end

% % plot mean over all trials for a selected neuron
figure;
bar(time/dt,reshape(mean(PSTHs(targetNeuron,:,:),2),[1 size(PSTHs,3)]),'FaceColor','#A2142F','EdgeColor','#A2142F');
% CueOnset
xhp = xline(0/dt,'--r',{'Cue-ONSET'},'Color','black','LineWidth',2); 
xhp.FontSize = 8;
% DelayOnset
xhp = xline(0.3/dt,'--r',{'Delay-ONSET'},'Color','#0072BD','LineWidth',2); 
xhp.FontSize = 8;
% targetOnset
xhp = xline(0.9/dt,'--r',{'Target-ONSET'},'Color','#77AC30','LineWidth',2); 
xhp.FontSize = 8;
grid on; grid minor;
title("window length = " + windowLength+ " | NeuronNumber = " + targetNeuron + " | Mean over All Trials",'interpreter','latex');
xlabel('Time(ms)','interpreter','latex');
ylabel('Fire Rate(Hz)','interpreter','latex');


% % plot mean over all neurons for each con
figure; 
for i=1:length(Unit(1).Cnd)
    subplot(2,3,i);
    bar(time/dt,reshape(AllAveragedPSTHs(i,:),[1 size(AllAveragedPSTHs,2)]),'FaceColor','#A2142F','EdgeColor','#A2142F');
  % CueOnset
    xhp = xline(0/dt,'--r',{'Cue-ONSET'},'Color','black','LineWidth',2); 
    xhp.FontSize = 8;
    % DelayOnset
    xhp = xline(0.3/dt,'--r',{'Delay-ONSET'},'Color','#0072BD','LineWidth',2); 
    xhp.FontSize = 8;
    % targetOnset
    xhp = xline(0.9/dt,'--r',{'Target-ONSET'},'Color','#77AC30','LineWidth',2); 
    xhp.FontSize = 8;
    grid on; grid minor;
    title("window length = " + windowLength+" | Condition = " + i + " | meanOfAll",'interpreter','latex');
    xlabel('Time(ms)','interpreter','latex');
    ylabel('Fire Rate(Hz)','interpreter','latex');
end

%% part2 - GLM - fit neural respones over reward and cue location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% run part.1 before this part.
clc; close all;

meanNerualResponse = [];
locationCues = [-1 1];
Rewards = [3 6 9];

conNum = zeros(481,6);
% fireConOrder = zeros(size(firingRate));
gg = [];

PSTHconOrder = zeros(size(PSTHs));
for i=1:length(Unit)
    k = 0;
    for j=1:length(Unit(1).Cnd)
       PSTHconOrder(i,k+1:(k+length(Unit(i).Cnd(j).TrialIdx)),:) = (PSTHs(i,Unit(i).Cnd(j).TrialIdx,:));
       k = k + length(Unit(i).Cnd(j).TrialIdx);
       conNum(i,j) = length(Unit(i).Cnd(j).TrialIdx);
    end
end

targetOnsetStart = 1.7/(windowLength*dt);
targetOnsetEnd = 1/(windowLength*dt);
meanNerualResponse = mean(PSTHconOrder(:,:,(end-(targetOnsetStart-1)):(end-(targetOnsetEnd))),3);

XX = [1 -1 3; 1 1 3; 1 -1 6; 1 1 6; 1 -1 9; 1 1 9];
for kk=1:481
    X = [];

    for i=1:6
        for j=1:conNum(kk,i)
            X = [X; XX(i,:)];
        end
    end
    targetNeuron = kk;
    y = meanNerualResponse(targetNeuron,:).';
    [b,bint,r,rint,stats] = regress(y,X);    % Removes NaN data
    gg = [gg stats(3)]; % p values of all neurons
end

% select neurons with a specific p value
pval = 0.001;
savee = find(gg<pval);
gg = [];

% plot regressed data 
for i=1:length(savee)
    targetNeuron = savee(i);
    y = meanNerualResponse(targetNeuron,:).';
    [b,bint,r,rint,stats] = regress(y,X);    % Removes NaN data
    gg = [gg stats(3)];


    % plot the data
    figure;
    scatter3(X(:,2),X(:,3),y,'filled')
    grid on; grid minor;
    hold on
    x1fit = min(X(:,2)):0.01:max(X(:,2));
    x2fit = min(X(:,3)):0.1:max(X(:,3));
    [X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
    YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
    mesh(X1FIT,X2FIT,YFIT)
    xlabel('Cue Loc','interpreter','latex')
    ylabel('Reward Val','interpreter','latex')
    zlabel('firingRate','interpreter','latex')
    title("Neuron Number = " + savee(i),'interpreter','latex');
    view(50,10)
    hold off
end

%% part3 - dimension reduction 
% cal firing rate in time for each neuron
% each window is 10*0.001 = 0.01s long
clc; 
figure;
firingR = (AveragedConditionPSTHs);

colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE"];
for i=1:6
    selectedFirings = reshape(firingR(:,i,:),...
        [size(firingR,1) size(firingR,3)]);
    coeff = pca(selectedFirings.');
    newdata = selectedFirings.'*coeff(:,1:3);
    plot3(smooth(newdata(:,1)),smooth(newdata(:,2)),smooth(newdata(:,3)),'Color',colors(i),...
        'LineWidth',2);
    hold on;
    startOfPlot = scatter3((newdata(1,1)),(newdata(1,2)),(newdata(1,3)),'filled','MarkerFaceColor',colors(i));
    startOfPlot.HandleVisibility = 'off';
    hold on;
    endOfPlot = scatter3((newdata(end,1)),(newdata(end,2)),(newdata(end,3)),'filled','MarkerFaceColor',colors(i),'Marker','s');
    endOfPlot.HandleVisibility = 'off';
    hold on;
    grid on; grid minor;
    xlabel('PC1','interpreter','latex');
    ylabel('PC2','interpreter','latex');
    zlabel('PC3','interpreter','latex');
    title('Not Shuffled Data','interpreter','latex');
end
legend('con1','con2','con3','con4','con5','con6')

%% part4 - shuffeling

conditionNumbers = size(AveragedConditionPSTHs,2); % has to be six
shufflingNumbers = 100;
surrData = CFR(permute(AveragedConditionPSTHs,[3 1 2]),conditionNumbers,shufflingNumbers);
% surrData = load('surrData.mat'); % load shuffled data from zip 
% surrData = surrData.surrData;
colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE"];

figure;
for i=1:6
    selectedFirings = reshape(surrData(:,:,i),[size(surrData,1) size(surrData,2)]);
    coeff = pca(selectedFirings);
    newdata = (selectedFirings*coeff(:,1:3));
    newdata = newdata;
    plot3(smooth(newdata(:,1)),smooth(newdata(:,2)),smooth(newdata(:,3)),'Color',colors(i),...
        'LineWidth',2);
    hold on;
    startOfPlot = scatter3((newdata(1,1)),(newdata(1,2)),(newdata(1,3)),'filled','MarkerFaceColor',colors(i));
    startOfPlot.HandleVisibility = 'off';
    hold on;
    endOfPlot = scatter3((newdata(end,1)),(newdata(end,2)),(newdata(end,3)),'filled','MarkerFaceColor',colors(i),'Marker','s');
    endOfPlot.HandleVisibility = 'off';
    hold on;
    grid on; grid minor;
    xlabel('PC1','interpreter','latex');
    ylabel('PC2','interpreter','latex');
    zlabel('PC3','interpreter','latex');
    title('Shuffled Data','interpreter','latex');
end

legend('con1','con2','con3','con4','con5','con6')
