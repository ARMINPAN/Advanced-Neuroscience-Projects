%% Advanced Neuroscience - Armin Panjehpour
%%% Neural population noise/ encoding and decoding
%%%%%%%%%%%% sections should be run in order for not getting errors
% but the part01 which shouldn`t be run

%% part01 - pre-processing on the raw data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% don't run this part!!!!
clc; clear; close all;

% pre-process function of the paper
script_mean_firing_rates_gratings

%% part02 - load the pre-processed datas
% run this part!
clc; close all;

% monkey1
monkey1DataPreProcessed = load('PreProcessedData\S_monkey1.mat');
% monkey2
monkey2DataPreProcessed = load('PreProcessedData\S_monkey2.mat');
% monkey3
monkey3DataPreProcessed = load('PreProcessedData\S_monkey3.mat');

% load removed datas
removedNeurons = load('PreProcessedData\removedNeurons.mat');


%% part1 - Tuning Curve for monkey 1
clc; close all;

% initialization
dt = 1/length(monkey1DataPreProcessed.S(1).trial(1).spikes);
gratingDegrees = 0:30:330;

% monkey1
% find number of spikes averaged over all trials for each neuron
neuronNumbers = 83;
numberOfSpikesMeanOverTrialsMonkey1 = zeros(neuronNumbers,12);
for i=1:size(numberOfSpikesMeanOverTrialsMonkey1,2)
    numberOfSpikesMeanOverTrialsMonkey1(:,i) = sum(monkey1DataPreProcessed.S(i).mean_FRs,2);
end

[targetNeuron, ~] = find(ismember(numberOfSpikesMeanOverTrialsMonkey1,...
    max(numberOfSpikesMeanOverTrialsMonkey1(:))));
figure;
subplot(2,3,1)
plot(gratingDegrees,numberOfSpikesMeanOverTrialsMonkey1(targetNeuron,:),'LineWidth',2);
grid on; grid minor;
xlabel('Grating Degree','interpreter','latex');
ylabel('firingRate(Hz)','interpreter','latex');
title("Tuning Curve for Monkey 1 || Neuron Num = " + targetNeuron,'interpreter','latex');
xlim([0 330]);


targetNeuron = randperm(neuronNumbers,5);
for i=2:6
    subplot(2,3,i);
    plot(gratingDegrees,numberOfSpikesMeanOverTrialsMonkey1(targetNeuron(i-1),:),'LineWidth',2);
    grid on; grid minor;
    xlabel('Grating Degree','interpreter','latex');
    ylabel('firingRate(Hz)','interpreter','latex');
    title("Tuning Curve for Monkey 1 || Neuron Num = " + targetNeuron(i-1),'interpreter','latex');
    xlim([0 330]);
end
    
%% part1 - Tuning Curve for monkey 2
clc; close all;

% initialization
dt = 1/length(monkey2DataPreProcessed.S(1).trial(1).spikes);
gratingDegrees = 0:30:330;

% monkey1
% find number of spikes averaged over all trials for each neuron
neuronNumbers = 59;
numberOfSpikesMeanOverTrialsMonkey2 = zeros(neuronNumbers,12);
for i=1:size(numberOfSpikesMeanOverTrialsMonkey2,2)
    numberOfSpikesMeanOverTrialsMonkey2(:,i) = sum(monkey2DataPreProcessed.S(i).mean_FRs,2);
end

[targetNeuron, ~] = find(ismember(numberOfSpikesMeanOverTrialsMonkey2,...
    max(numberOfSpikesMeanOverTrialsMonkey2(:))));

figure;
subplot(2,3,1);
plot(gratingDegrees,numberOfSpikesMeanOverTrialsMonkey2(targetNeuron,:),'LineWidth',2);
grid on; grid minor;
xlabel('Grating Degree','interpreter','latex');
ylabel('firingRate(Hz)','interpreter','latex');
title("Tuning Curve for Monkey 2 || Neuron Num = " + targetNeuron,'interpreter','latex');
xlim([0 330]);

targetNeuron = randperm(neuronNumbers,5);
for i=2:6
    subplot(2,3,i);
    plot(gratingDegrees,numberOfSpikesMeanOverTrialsMonkey2(targetNeuron(i-1),:),'LineWidth',2);
    grid on; grid minor;
    xlabel('Grating Degree','interpreter','latex');
    ylabel('firingRate(Hz)','interpreter','latex');
    title("Tuning Curve for Monkey 2 || Neuron Num = " + targetNeuron(i-1),'interpreter','latex');
    xlim([0 330]);
end
    
    


%% part1 - Tuning Curve for monkey 3
clc; 

% initialization
dt = 1/length(monkey3DataPreProcessed.S(1).trial(1).spikes);
gratingDegrees = 0:30:330;

% monkey1
% find number of spikes averaged over all trials for each neuron
neuronNumbers = 105;
numberOfSpikesMeanOverTrialsMonkey3 = zeros(neuronNumbers,12);
for i=1:size(numberOfSpikesMeanOverTrialsMonkey3,2)
    numberOfSpikesMeanOverTrialsMonkey3(:,i) = sum(monkey3DataPreProcessed.S(i).mean_FRs,2);
end

[targetNeuron, ~] = find(ismember(numberOfSpikesMeanOverTrialsMonkey3,...
    max(numberOfSpikesMeanOverTrialsMonkey3(:))));

figure;
subplot(2,3,1)
plot(gratingDegrees,numberOfSpikesMeanOverTrialsMonkey3(targetNeuron,:),'LineWidth',2);
grid on; grid minor;
xlabel('Grating Degree','interpreter','latex');
ylabel('firingRate(Hz)','interpreter','latex');
title("Tuning Curve for Monkey 3 || Neuron Num = " + targetNeuron,'interpreter','latex');
xlim([0 330]);

targetNeuron = randperm(neuronNumbers,5);
for i=2:6
    subplot(2,3,i);
    plot(gratingDegrees,numberOfSpikesMeanOverTrialsMonkey3(targetNeuron(i-1),:),'LineWidth',2);
    grid on; grid minor;
    xlabel('Grating Degree','interpreter','latex');
    ylabel('firingRate(Hz)','interpreter','latex');
    title("Tuning Curve for Monkey 3 || Neuron Num = " + targetNeuron(i-1),'interpreter','latex');
    xlim([0 330]);
end
    

%% part2 - 10*10 Mesh - Each Unit Prefered Orientation - Monkey 1
%%%%%%% run part1\monkey1 for before this part
clc; close all;
monkey1Data = load('RawData\data_monkey1_gratings.mat');

preferedOriMonkey1 = zeros(83,1);
alam = zeros(83,1);
for i = 1:length(preferedOriMonkey1)
    preferedOriMonkey1(i) = gratingDegrees(numberOfSpikesMeanOverTrialsMonkey1(i,:) == ...
        max(numberOfSpikesMeanOverTrialsMonkey1(i,:)));
end

% 0 to 150
% preferedOriMonkey1(preferedOriMonkey1>=180) = preferedOriMonkey1(preferedOriMonkey1>=180) - 180;

% remove the neurons which are removed in preprocessing from MAP and
% Channels locs and SNR vector

% Channels
monkey1Data.data.CHANNELS(removedNeurons.removedNeurons{1, 1},:) = [];
% SNR
monkey1Data.data.SNR(removedNeurons.removedNeurons{1, 1},:) = [];

% MAPS
for i = 1:10
    for j = 1:10
        if(~(ismember(monkey1Data.data.MAP(i,j),monkey1Data.data.CHANNELS(:,1))))
            monkey1Data.data.MAP(i,j) = nan;
        end
    end
end

% NaN = -30;
Y = 1:10;
X = 1:10;
C = zeros(10,10)*nan;

for i = 1:10
    for j = 1:10
        if(~isnan(monkey1Data.data.MAP(i,j)))
%             [i j find(monkey1Data.data.CHANNELS(:,1) == ...
%                 monkey1Data.data.MAP(i,j),1)]
            findeds = (find(monkey1Data.data.CHANNELS(:,1) == ...
                            monkey1Data.data.MAP(i,j)));
            findeds = find(monkey1Data.data.SNR == (max(monkey1Data.data.SNR(findeds))));
            C(i,j) = preferedOriMonkey1(findeds(1));
        end
    end
end

figure;
h = imagesc(C);
title('Prefered Orientation of Units for Monkey 1','interpreter','latex');
set(h, 'AlphaData', ~isnan(C))
colormap jet	

%% part2 - 10*10 Mesh - Each Unit Prefered Orientation - Monkey 2
%%%%%%% run part1\monkey2 for before this part
clc; close all;
monkey2Data = load('RawData\data_monkey2_gratings.mat');

preferedOriMonkey2 = zeros(59,1);
for i = 1:length(preferedOriMonkey2)
    preferedOriMonkey2(i) = gratingDegrees(numberOfSpikesMeanOverTrialsMonkey2(i,:) == ...
        max(numberOfSpikesMeanOverTrialsMonkey2(i,:)));
end

% 0 to 150
% preferedOriMonkey2(preferedOriMonkey2>=180) = preferedOriMonkey2(preferedOriMonkey2>=180) - 180;

% remove the neurons which are removed in preprocessing from MAP and
% Channels locs and SNR vector

% Channels
monkey2Data.data.CHANNELS(removedNeurons.removedNeurons{1, 2},:) = [];
% SNR
monkey2Data.data.SNR(removedNeurons.removedNeurons{1, 2},:) = [];

% MAPS
for i = 1:10
    for j = 1:10
        if(~(ismember(monkey2Data.data.MAP(i,j),monkey2Data.data.CHANNELS(:,1))))
            monkey2Data.data.MAP(i,j) = nan;
        end
    end
end

% NaN = -30;
C = zeros(10,10)*nan;

for i = 1:10
    for j = 1:10
        if(~isnan(monkey2Data.data.MAP(i,j)))
%             [i j find(monkey2Data.data.CHANNELS(:,1) == ...
%                 monkey2Data.data.MAP(i,j),1)]
            findeds = (find(monkey2Data.data.CHANNELS(:,1) == ...
                            monkey2Data.data.MAP(i,j)));
            findeds = find(monkey2Data.data.SNR == (max(monkey2Data.data.SNR(findeds))));
            C(i,j) = preferedOriMonkey2(findeds(1));
        end
    end
end

figure;
h = imagesc(C);
title('Prefered Orientation of Units for Monkey 2','interpreter','latex');
set(h, 'AlphaData', ~isnan(C))
colormap jet	

%% part2 - 10*10 Mesh - Each Unit Prefered Orientation - Monkey 3
%%%%%%% run part1\monkey3 for before this part
clc; close all;
monkey3Data = load('RawData\data_monkey3_gratings.mat');

preferedOriMonkey3 = zeros(105,1);
for i = 1:length(preferedOriMonkey3)
    preferedOriMonkey3(i) = gratingDegrees(numberOfSpikesMeanOverTrialsMonkey3(i,:) == ...
        max(numberOfSpikesMeanOverTrialsMonkey3(i,:)));
end

% 0 to 150
% preferedOriMonkey3(preferedOriMonkey3>=180) = preferedOriMonkey3(preferedOriMonkey3>=180) - 180;

% remove the neurons which are removed in preprocessing from MAP and
% Channels locs and SNR vector

% Channels
monkey3Data.data.CHANNELS(removedNeurons.removedNeurons{1, 3},:) = [];
% SNR
monkey3Data.data.SNR(removedNeurons.removedNeurons{1, 3},:) = [];

% MAPS
for i = 1:10
    for j = 1:10
        if(~(ismember(monkey3Data.data.MAP(i,j),monkey3Data.data.CHANNELS(:,1))))
            monkey3Data.data.MAP(i,j) = nan;
        end
    end
end

% NaN = -30;
Y = 0:10;
X = 0:10;
C = zeros(10,10)*nan;

for i = 1:10
    for j = 1:10
        if(~isnan(monkey3Data.data.MAP(i,j)))
%             [i j find(monkey3Data.data.CHANNELS(:,1) == ...
%                 monkey3Data.data.MAP(i,j),1)]
            findeds = (find(monkey3Data.data.CHANNELS(:,1) == ...
                            monkey3Data.data.MAP(i,j)));
            findeds = find(monkey3Data.data.SNR == (max(monkey3Data.data.SNR(findeds))));
            C(i,j) = preferedOriMonkey3(findeds(1));
        end
    end
end

figure;
h = imagesc(C);
title('Prefered Orientation of Units for Monkey 3','interpreter','latex');
set(h, 'AlphaData', ~isnan(C))
colormap jet	

%% part31 - noise correlation(spike count corr vs elecs distance) - monkey1
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey1DataPreProcessed.S(1).mean_FRs);
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey1DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey1 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey1 = corrcoef(numberOfSpikesMeanOverTrialsMonkey1.');
%%%% let`s create 4 groups - rsignal < -0.5/ rsignal > 0.5 
%%%% 0 < rsignal < 0.5 / -0.5 < rsignal < 0
CorrMatrixSignalMonkey1 = tril(CorrMatrixSignalMonkey1,-1);
[gp1y, gp1x] = find(CorrMatrixSignalMonkey1 < -0.5);
[gp2y, gp2x] = find(CorrMatrixSignalMonkey1 >= -0.5 & CorrMatrixSignalMonkey1 < 0);
[gp3y, gp3x] = find(CorrMatrixSignalMonkey1 > 0 & CorrMatrixSignalMonkey1 <= 0.5);
[gp4y, gp4x] = find(CorrMatrixSignalMonkey1 > 0.5);


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey1Data.data.MAP,monkey1Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey1Data.data.MAP,monkey1Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

% groups
%%%%% group1 
Xgp1 = zeros(1,length(gp1x));
Ygp1 = zeros(1,length(gp1x));

for i = 1:length(gp1x)
        Xgp1(i) = NeuronDistanceMatrix(gp1y(i),gp1x(i));
        Ygp1(i) = CorrMatrixSpikeCountMonkey1(gp1y(i),gp1x(i));
end

%%%%% group2
Xgp2 = zeros(1,length(gp2x));
Ygp2 = zeros(1,length(gp2x));

for i = 1:length(gp2x)
        Xgp2(i) = NeuronDistanceMatrix(gp2y(i),gp2x(i));
        Ygp2(i) = CorrMatrixSpikeCountMonkey1(gp2y(i),gp2x(i));
end

%%%%% group3
Xgp3 = zeros(1,length(gp3x));
Ygp3 = zeros(1,length(gp3x));

for i = 1:length(gp3x)
        Xgp3(i) = NeuronDistanceMatrix(gp3y(i),gp3x(i));
        Ygp3(i) = CorrMatrixSpikeCountMonkey1(gp3y(i),gp3x(i));
end

%%%%% group4
Xgp4 = zeros(1,length(gp4x));
Ygp4 = zeros(1,length(gp4x));

for i = 1:length(gp4x)
        Xgp4(i) = NeuronDistanceMatrix(gp4y(i),gp4x(i));
        Ygp4(i) = CorrMatrixSpikeCountMonkey1(gp4y(i),gp4x(i));
end


% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = 0.00025:0.0005:0.00425;
binL = 0.00025;
Ymeangp1 = zeros(1,length(Xunique));
Yvargp1 = zeros(1,length(Xunique));
Ymeangp2 = zeros(1,length(Xunique));
Yvargp2 = zeros(1,length(Xunique));
Ymeangp3 = zeros(1,length(Xunique));
Yvargp3 = zeros(1,length(Xunique));
Ymeangp4 = zeros(1,length(Xunique));
Yvargp4 = zeros(1,length(Xunique));

for i = 1:length(Xunique)
    Ymeangp1(i) = mean(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
    Yvargp1(i) = var(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp2(i) = mean(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
    Yvargp2(i) = var(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp3(i) = mean(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
    Yvargp3(i) = var(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp4(i) = mean(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
    Yvargp4(i) = var(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
end



figure;
errorbar(Xunique*1000,(Ymeangp1),Yvargp1);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp2),Yvargp2);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp3),Yvargp3);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp4),Yvargp4);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');
title('Monkey1','interpreter','latex');
legend('rsignal < -0.5','rsignal -0.5 to 0','rsignal 0 to 0.5','rsignal > 0.5');


%% part32 - noise correlation(spike count corr vs elecs distance) - monkey2
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey2DataPreProcessed.S(1).mean_FRs);
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey2DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey2 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey2 = corrcoef(numberOfSpikesMeanOverTrialsMonkey2.');
%%%% let`s create 4 groups - rsignal < -0.5/ rsignal > 0.5 
%%%% 0 < rsignal < 0.5 / -0.5 < rsignal < 0
CorrMatrixSignalMonkey2 = tril(CorrMatrixSignalMonkey2,-1);
[gp1y, gp1x] = find(CorrMatrixSignalMonkey2 < -0.5);
[gp2y, gp2x] = find(CorrMatrixSignalMonkey2 >= -0.5 & CorrMatrixSignalMonkey2 < 0);
[gp3y, gp3x] = find(CorrMatrixSignalMonkey2 > 0 & CorrMatrixSignalMonkey2 <= 0.5);
[gp4y, gp4x] = find(CorrMatrixSignalMonkey2 > 0.5);


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey2Data.data.MAP,monkey2Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey2Data.data.MAP,monkey2Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

% groups
%%%%% group1 
Xgp1 = zeros(1,length(gp1x));
Ygp1 = zeros(1,length(gp1x));

for i = 1:length(gp1x)
        Xgp1(i) = NeuronDistanceMatrix(gp1y(i),gp1x(i));
        Ygp1(i) = CorrMatrixSpikeCountMonkey2(gp1y(i),gp1x(i));
end

%%%%% group2
Xgp2 = zeros(1,length(gp2x));
Ygp2 = zeros(1,length(gp2x));

for i = 1:length(gp2x)
        Xgp2(i) = NeuronDistanceMatrix(gp2y(i),gp2x(i));
        Ygp2(i) = CorrMatrixSpikeCountMonkey2(gp2y(i),gp2x(i));
end

%%%%% group3
Xgp3 = zeros(1,length(gp3x));
Ygp3 = zeros(1,length(gp3x));

for i = 1:length(gp3x)
        Xgp3(i) = NeuronDistanceMatrix(gp3y(i),gp3x(i));
        Ygp3(i) = CorrMatrixSpikeCountMonkey2(gp3y(i),gp3x(i));
end

%%%%% group4
Xgp4 = zeros(1,length(gp4x));
Ygp4 = zeros(1,length(gp4x));

for i = 1:length(gp4x)
        Xgp4(i) = NeuronDistanceMatrix(gp4y(i),gp4x(i));
        Ygp4(i) = CorrMatrixSpikeCountMonkey2(gp4y(i),gp4x(i));
end


% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = 0.00025:0.0005:0.00425;
binL = 0.00025;
Ymeangp1 = zeros(1,length(Xunique));
Yvargp1 = zeros(1,length(Xunique));
Ymeangp2 = zeros(1,length(Xunique));
Yvargp2 = zeros(1,length(Xunique));
Ymeangp3 = zeros(1,length(Xunique));
Yvargp3 = zeros(1,length(Xunique));
Ymeangp4 = zeros(1,length(Xunique));
Yvargp4 = zeros(1,length(Xunique));

for i = 1:length(Xunique)
    Ymeangp1(i) = mean(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
    Yvargp1(i) = var(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp2(i) = mean(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
    Yvargp2(i) = var(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp3(i) = mean(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
    Yvargp3(i) = var(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp4(i) = mean(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
    Yvargp4(i) = var(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
end



figure;
errorbar(Xunique*1000,(Ymeangp1),Yvargp1);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp2),Yvargp2);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp3),Yvargp3);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp4),Yvargp4);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');
title('Monkey2','interpreter','latex');
legend('rsignal < -0.5','rsignal -0.5 to 0','rsignal 0 to 0.5','rsignal > 0.5');

%% part33 - noise correlation(spike count corr vs elecs distance) - monkey3
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey3DataPreProcessed.S(1).mean_FRs);
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey3DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey3 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey3 = corrcoef(numberOfSpikesMeanOverTrialsMonkey3.');
%%%% let`s create 4 groups - rsignal < -0.5/ rsignal > 0.5 
%%%% 0 < rsignal < 0.5 / -0.5 < rsignal < 0
CorrMatrixSignalMonkey3 = tril(CorrMatrixSignalMonkey3,-1);
[gp1y, gp1x] = find(CorrMatrixSignalMonkey3 < -0.5);
[gp2y, gp2x] = find(CorrMatrixSignalMonkey3 >= -0.5 & CorrMatrixSignalMonkey3 < 0);
[gp3y, gp3x] = find(CorrMatrixSignalMonkey3 > 0 & CorrMatrixSignalMonkey3 <= 0.5);
[gp4y, gp4x] = find(CorrMatrixSignalMonkey3 > 0.5);


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey3Data.data.MAP,monkey3Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey3Data.data.MAP,monkey3Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

% groups
%%%%% group1 
Xgp1 = zeros(1,length(gp1x));
Ygp1 = zeros(1,length(gp1x));

for i = 1:length(gp1x)
        Xgp1(i) = NeuronDistanceMatrix(gp1y(i),gp1x(i));
        Ygp1(i) = CorrMatrixSpikeCountMonkey3(gp1y(i),gp1x(i));
end

%%%%% group2
Xgp2 = zeros(1,length(gp2x));
Ygp2 = zeros(1,length(gp2x));

for i = 1:length(gp2x)
        Xgp2(i) = NeuronDistanceMatrix(gp2y(i),gp2x(i));
        Ygp2(i) = CorrMatrixSpikeCountMonkey3(gp2y(i),gp2x(i));
end

%%%%% group3
Xgp3 = zeros(1,length(gp3x));
Ygp3 = zeros(1,length(gp3x));

for i = 1:length(gp3x)
        Xgp3(i) = NeuronDistanceMatrix(gp3y(i),gp3x(i));
        Ygp3(i) = CorrMatrixSpikeCountMonkey3(gp3y(i),gp3x(i));
end

%%%%% group4
Xgp4 = zeros(1,length(gp4x));
Ygp4 = zeros(1,length(gp4x));

for i = 1:length(gp4x)
        Xgp4(i) = NeuronDistanceMatrix(gp4y(i),gp4x(i));
        Ygp4(i) = CorrMatrixSpikeCountMonkey3(gp4y(i),gp4x(i));
end


% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = 0.00025:0.0005:0.00425;
binL = 0.00025;
Ymeangp1 = zeros(1,length(Xunique));
Yvargp1 = zeros(1,length(Xunique));
Ymeangp2 = zeros(1,length(Xunique));
Yvargp2 = zeros(1,length(Xunique));
Ymeangp3 = zeros(1,length(Xunique));
Yvargp3 = zeros(1,length(Xunique));
Ymeangp4 = zeros(1,length(Xunique));
Yvargp4 = zeros(1,length(Xunique));

for i = 1:length(Xunique)
    Ymeangp1(i) = mean(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
    Yvargp1(i) = var(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp2(i) = mean(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
    Yvargp2(i) = var(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp3(i) = mean(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
    Yvargp3(i) = var(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp4(i) = mean(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
    Yvargp4(i) = var(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
end



figure;
errorbar(Xunique*1000,(Ymeangp1),Yvargp1);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp2),Yvargp2);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp3),Yvargp3);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique*1000,(Ymeangp4),Yvargp4);
grid on; grid minor;
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');
title('Monkey3','interpreter','latex');
legend('rsignal < -0.5','rsignal -0.5 to 0','rsignal 0 to 0.5','rsignal > 0.5');

%% part31 - noise correlation(spike count corr vs rsignal) - monkey1
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey1DataPreProcessed.S(1).mean_FRs)
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey1DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey1 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey1 = corrcoef(numberOfSpikesMeanOverTrialsMonkey1.');


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey1Data.data.MAP,monkey1Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey1Data.data.MAP,monkey1Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

%%%% let`s create 4 groups - dis 0 to 1mm - 1 to 2mm - 2 to 3mm - 3 to 10mm
NeuronDistanceMatrix = tril(NeuronDistanceMatrix,-1);
[gp1y, gp1x] = find(NeuronDistanceMatrix > 0 & NeuronDistanceMatrix <= 0.001);
[gp2y, gp2x] = find(NeuronDistanceMatrix > 0.001 & NeuronDistanceMatrix <= 0.002);
[gp3y, gp3x] = find(NeuronDistanceMatrix > 0.002 & NeuronDistanceMatrix <= 0.003);
[gp4y, gp4x] = find(NeuronDistanceMatrix > 0.003 & NeuronDistanceMatrix <= 0.010);

% groups
%%%%% group1 
Xgp1 = zeros(1,length(gp1x));
Ygp1 = zeros(1,length(gp1x));

for i = 1:length(gp1x)
        Xgp1(i) = CorrMatrixSignalMonkey1(gp1y(i),gp1x(i));
        Ygp1(i) = CorrMatrixSpikeCountMonkey1(gp1y(i),gp1x(i));
end

%%%%% group2
Xgp2 = zeros(1,length(gp2x));
Ygp2 = zeros(1,length(gp2x));

for i = 1:length(gp2x)
        Xgp2(i) = CorrMatrixSignalMonkey1(gp2y(i),gp2x(i));
        Ygp2(i) = CorrMatrixSpikeCountMonkey1(gp2y(i),gp2x(i));
end

%%%%% group3
Xgp3 = zeros(1,length(gp3x));
Ygp3 = zeros(1,length(gp3x));

for i = 1:length(gp3x)
        Xgp3(i) = CorrMatrixSignalMonkey1(gp3y(i),gp3x(i));
        Ygp3(i) = CorrMatrixSpikeCountMonkey1(gp3y(i),gp3x(i));
end

%%%%% group4
Xgp4 = zeros(1,length(gp4x));
Ygp4 = zeros(1,length(gp4x));

for i = 1:length(gp4x)
        Xgp4(i) = CorrMatrixSignalMonkey1(gp4y(i),gp4x(i));
        Ygp4(i) = CorrMatrixSpikeCountMonkey1(gp4y(i),gp4x(i));
end


% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = -0.75:0.25:0.75;
binL = 0.125;
Ymeangp1 = zeros(1,length(Xunique));
Yvargp1 = zeros(1,length(Xunique));
Ymeangp2 = zeros(1,length(Xunique));
Yvargp2 = zeros(1,length(Xunique));
Ymeangp3 = zeros(1,length(Xunique));
Yvargp3 = zeros(1,length(Xunique));
Ymeangp4 = zeros(1,length(Xunique));
Yvargp4 = zeros(1,length(Xunique));


for i = 1:length(Xunique)
    Ymeangp1(i) = mean(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
    Yvargp1(i) = var(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp2(i) = mean(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
    Yvargp2(i) = var(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp3(i) = mean(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
    Yvargp3(i) = var(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp4(i) = mean(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
    Yvargp4(i) = var(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
end




figure;
errorbar(Xunique,(Ymeangp1),Yvargp1);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp2),Yvargp2);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp3),Yvargp3);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp4),Yvargp4);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');
title('Monkey1','interpreter','latex');

legend('Distance 0 - 1mm','Distance 1mm - 2mm','Distance 2mm - 3mm',...
    'Distance 3mm - 10mm','NumColumns',2);

%% part32 - noise correlation(spike count corr vs rsignal) - monkey2
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey2DataPreProcessed.S(1).mean_FRs);
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey2DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey2 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey2 = corrcoef(numberOfSpikesMeanOverTrialsMonkey2.');
CorrMatrixSignalMonkey2 = tril(CorrMatrixSignalMonkey2,-1);


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey2Data.data.MAP,monkey2Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey2Data.data.MAP,monkey2Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

%%%% let`s create 4 groups - dis 0 to 1mm - 1 to 2mm - 2 to 3mm - 3 to 10mm
NeuronDistanceMatrix = tril(NeuronDistanceMatrix,-1);
[gp1y, gp1x] = find(NeuronDistanceMatrix > 0 & NeuronDistanceMatrix <= 0.001);
[gp2y, gp2x] = find(NeuronDistanceMatrix > 0.001 & NeuronDistanceMatrix <= 0.002);
[gp3y, gp3x] = find(NeuronDistanceMatrix > 0.002 & NeuronDistanceMatrix <= 0.003);
[gp4y, gp4x] = find(NeuronDistanceMatrix > 0.003 & NeuronDistanceMatrix <= 0.010);

% groups
%%%%% group1 
Xgp1 = zeros(1,length(gp1x));
Ygp1 = zeros(1,length(gp1x));

for i = 1:length(gp1x)
        Xgp1(i) = CorrMatrixSignalMonkey2(gp1y(i),gp1x(i));
        Ygp1(i) = CorrMatrixSpikeCountMonkey2(gp1y(i),gp1x(i));
end

%%%%% group2
Xgp2 = zeros(1,length(gp2x));
Ygp2 = zeros(1,length(gp2x));

for i = 1:length(gp2x)
        Xgp2(i) = CorrMatrixSignalMonkey2(gp2y(i),gp2x(i));
        Ygp2(i) = CorrMatrixSpikeCountMonkey2(gp2y(i),gp2x(i));
end

%%%%% group3
Xgp3 = zeros(1,length(gp3x));
Ygp3 = zeros(1,length(gp3x));

for i = 1:length(gp3x)
        Xgp3(i) = CorrMatrixSignalMonkey2(gp3y(i),gp3x(i));
        Ygp3(i) = CorrMatrixSpikeCountMonkey2(gp3y(i),gp3x(i));
end

%%%%% group4
Xgp4 = zeros(1,length(gp4x));
Ygp4 = zeros(1,length(gp4x));

for i = 1:length(gp4x)
        Xgp4(i) = CorrMatrixSignalMonkey2(gp4y(i),gp4x(i));
        Ygp4(i) = CorrMatrixSpikeCountMonkey2(gp4y(i),gp4x(i));
end


% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = -0.75:0.25:0.75;
binL = 0.125;
Ymeangp1 = zeros(1,length(Xunique));
Yvargp1 = zeros(1,length(Xunique));
Ymeangp2 = zeros(1,length(Xunique));
Yvargp2 = zeros(1,length(Xunique));
Ymeangp3 = zeros(1,length(Xunique));
Yvargp3 = zeros(1,length(Xunique));
Ymeangp4 = zeros(1,length(Xunique));
Yvargp4 = zeros(1,length(Xunique));


for i = 1:length(Xunique)
    Ymeangp1(i) = mean(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
    Yvargp1(i) = var(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp2(i) = mean(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
    Yvargp2(i) = var(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp3(i) = mean(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
    Yvargp3(i) = var(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp4(i) = mean(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
    Yvargp4(i) = var(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
end




figure;
errorbar(Xunique,(Ymeangp1),Yvargp1);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp2),Yvargp2);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp3),Yvargp3);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp4),Yvargp4);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');
title('Monkey2','interpreter','latex');

legend('Distance 0 - 1mm','Distance 1mm - 2mm','Distance 2mm - 3mm',...
    'Distance 3mm - 10mm');


%% part33 - noise correlation(spike count corr vs rsignal) - monkey3
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey3DataPreProcessed.S(1).mean_FRs);
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey3DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey3 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey3 = corrcoef(numberOfSpikesMeanOverTrialsMonkey3.');
CorrMatrixSignalMonkey3 = tril(CorrMatrixSignalMonkey3,-1);


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey3Data.data.MAP,monkey3Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey3Data.data.MAP,monkey3Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

%%%% let`s create 4 groups - dis 0 to 1mm - 1 to 2mm - 2 to 3mm - 3 to 10mm
NeuronDistanceMatrix = tril(NeuronDistanceMatrix,-1);
[gp1y, gp1x] = find(NeuronDistanceMatrix > 0 & NeuronDistanceMatrix <= 0.001);
[gp2y, gp2x] = find(NeuronDistanceMatrix > 0.001 & NeuronDistanceMatrix <= 0.002);
[gp3y, gp3x] = find(NeuronDistanceMatrix > 0.002 & NeuronDistanceMatrix <= 0.003);
[gp4y, gp4x] = find(NeuronDistanceMatrix > 0.003 & NeuronDistanceMatrix <= 0.010);

% groups
%%%%% group1 
Xgp1 = zeros(1,length(gp1x));
Ygp1 = zeros(1,length(gp1x));

for i = 1:length(gp1x)
        Xgp1(i) = CorrMatrixSignalMonkey3(gp1y(i),gp1x(i));
        Ygp1(i) = CorrMatrixSpikeCountMonkey3(gp1y(i),gp1x(i));
end

%%%%% group2
Xgp2 = zeros(1,length(gp2x));
Ygp2 = zeros(1,length(gp2x));

for i = 1:length(gp2x)
        Xgp2(i) = CorrMatrixSignalMonkey3(gp2y(i),gp2x(i));
        Ygp2(i) = CorrMatrixSpikeCountMonkey3(gp2y(i),gp2x(i));
end

%%%%% group3
Xgp3 = zeros(1,length(gp3x));
Ygp3 = zeros(1,length(gp3x));

for i = 1:length(gp3x)
        Xgp3(i) = CorrMatrixSignalMonkey3(gp3y(i),gp3x(i));
        Ygp3(i) = CorrMatrixSpikeCountMonkey3(gp3y(i),gp3x(i));
end

%%%%% group4
Xgp4 = zeros(1,length(gp4x));
Ygp4 = zeros(1,length(gp4x));

for i = 1:length(gp4x)
        Xgp4(i) = CorrMatrixSignalMonkey3(gp4y(i),gp4x(i));
        Ygp4(i) = CorrMatrixSpikeCountMonkey3(gp4y(i),gp4x(i));
end


% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = -0.75:0.25:0.75;
binL = 0.125;
Ymeangp1 = zeros(1,length(Xunique));
Yvargp1 = zeros(1,length(Xunique));
Ymeangp2 = zeros(1,length(Xunique));
Yvargp2 = zeros(1,length(Xunique));
Ymeangp3 = zeros(1,length(Xunique));
Yvargp3 = zeros(1,length(Xunique));
Ymeangp4 = zeros(1,length(Xunique));
Yvargp4 = zeros(1,length(Xunique));


for i = 1:length(Xunique)
    Ymeangp1(i) = mean(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
    Yvargp1(i) = var(Ygp1(find(Xgp1 < Xunique(i)+binL & Xgp1 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp2(i) = mean(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
    Yvargp2(i) = var(Ygp2(find(Xgp2 < Xunique(i)+binL & Xgp2 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp3(i) = mean(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
    Yvargp3(i) = var(Ygp3(find(Xgp3 < Xunique(i)+binL & Xgp3 >= Xunique(i)-binL)));
end

for i = 1:length(Xunique)
    Ymeangp4(i) = mean(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
    Yvargp4(i) = var(Ygp4(find(Xgp4 < Xunique(i)+binL & Xgp4 >= Xunique(i)-binL)));
end




figure;
errorbar(Xunique,(Ymeangp1),Yvargp1);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp2),Yvargp2);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp3),Yvargp3);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');

hold on;
errorbar(Xunique,(Ymeangp4),Yvargp4);
grid on; grid minor;
xlabel('rsignal','interpreter','latex');
ylabel('Spike Count Correlation','interpreter','latex');
title('Monkey3','interpreter','latex');

legend('Distance 0 - 1mm','Distance 1mm - 2mm','Distance 2mm - 3mm',...
    'Distance 3mm - 10mm','NumColumns',2);

%% part31 - noise correlation(spike count corr vs elecs distance & rsignal) - monkey1
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey1DataPreProcessed.S(1).mean_FRs);
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey1DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey1 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey1 = corrcoef(numberOfSpikesMeanOverTrialsMonkey1.');
CorrMatrixSignalMonkey1 = tril(CorrMatrixSignalMonkey1,-1);


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey1Data.data.MAP,monkey1Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey1Data.data.MAP,monkey1Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

X = [];
for i = 1:length(NeuronDistanceMatrix)
    X = [X; NeuronDistanceMatrix(i+1:end,i)];
end

Y = [];
for i = 1:length(CorrMatrixSignalMonkey1)
    Y = [Y; CorrMatrixSignalMonkey1(i+1:end,i)];
end

Z = [];
for i = 1:length(CorrMatrixSpikeCountMonkey1)
    Z = [Z; CorrMatrixSpikeCountMonkey1(i+1:end,i)];
end

% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = 0.00025:0.0005:0.00425;
binLX = 0.00025;

Yunique = -1:0.25:1;
binLY = 0.125;

ZZ = zeros(length(Xunique),length(Yunique));

for i = 1:length(Xunique)
    for j = 1:length(Yunique)
        ZZ(i,j) = mean(Z(find(X < Xunique(i) + binLX & X >= Xunique(i) - binLX & ...
            Y < Yunique(j) + binLY & Y >= Yunique(j) - binLY)));
    end
end


figure;
pcolor(Xunique*1000,(Yunique),ZZ.');
colormap jet
shading interp
title('Spike Count Correlation || Monkey1','interpreter','latex');
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('rsignal','interpreter','latex');

%% part32 - noise correlation(spike count corr vs elecs distance & rsignal) - monkey2
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey2DataPreProcessed.S(1).mean_FRs);
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey2DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey2 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey2 = corrcoef(numberOfSpikesMeanOverTrialsMonkey2.');
CorrMatrixSignalMonkey2 = tril(CorrMatrixSignalMonkey2,-1);


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey2Data.data.MAP,monkey2Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey2Data.data.MAP,monkey2Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

X = [];
for i = 1:length(NeuronDistanceMatrix)
    X = [X; NeuronDistanceMatrix(i+1:end,i)];
end

Y = [];
for i = 1:length(CorrMatrixSignalMonkey2)
    Y = [Y; CorrMatrixSignalMonkey2(i+1:end,i)];
end

Z = [];
for i = 1:length(CorrMatrixSpikeCountMonkey2)
    Z = [Z; CorrMatrixSpikeCountMonkey2(i+1:end,i)];
end

% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = 0.00025:0.0005:0.00425;
binLX = 0.00025;

Yunique = -1:0.25:1;
binLY = 0.125;

ZZ = zeros(length(Xunique),length(Yunique));

for i = 1:length(Xunique)
    for j = 1:length(Yunique)
        ZZ(i,j) = mean(Z(find(X < Xunique(i) + binLX & X >= Xunique(i) - binLX & ...
            Y < Yunique(j) + binLY & Y >= Yunique(j) - binLY)));
    end
end


figure;
pcolor(Xunique*1000,(Yunique),ZZ.');
colormap jet
shading interp
title('Spike Count Correlation || Monkey2','interpreter','latex');
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('rsignal','interpreter','latex');

%% part33 - noise correlation(spike count corr vs elecs distance & rsignal) - monkey3
clc; close all;


% spike count correlation 

TrialsCounts = 200;
numberOfNeurons = length(monkey3DataPreProcessed.S(1).mean_FRs);
spikeCount = zeros(numberOfNeurons,TrialsCounts*12);
for j = 1:12
    for i = 1:TrialsCounts
        spikeCount(:,((j-1)*TrialsCounts+i)) = sum(monkey3DataPreProcessed.S(j).trial(i).spikes,2);
    end
    spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))) = ...
        zscore(spikeCount(:,(((j-1)*TrialsCounts+1):((j)*TrialsCounts))).').';
end

CorrMatrixSpikeCountMonkey3 = corrcoef(spikeCount.');


% Signal Correlation
CorrMatrixSignalMonkey3 = corrcoef(numberOfSpikesMeanOverTrialsMonkey3.');
CorrMatrixSignalMonkey3 = tril(CorrMatrixSignalMonkey3,-1);


% Neuron Distance
NeuronDistanceMatrix = zeros(numberOfNeurons,numberOfNeurons);
dis = 400*10^-6; % 400um
for i = 1:numberOfNeurons
    for j = 1:numberOfNeurons
        % electrode1
        [pos1y, pos1x] = find(ismember(monkey3Data.data.MAP,monkey3Data.data.CHANNELS(i,1)));
        % electrode2
        [pos2y, pos2x] = find(ismember(monkey3Data.data.MAP,monkey3Data.data.CHANNELS(j,1)));
        % find the distance between two electrodes
        NeuronDistanceMatrix(i,j) = sqrt(((pos2y-pos1y)*dis)^2 + ((pos2x-pos1x)*dis)^2);
    end
end

X = [];
for i = 1:length(NeuronDistanceMatrix)
    X = [X; NeuronDistanceMatrix(i+1:end,i)];
end

Y = [];
for i = 1:length(CorrMatrixSignalMonkey3)
    Y = [Y; CorrMatrixSignalMonkey3(i+1:end,i)];
end

Z = [];
for i = 1:length(CorrMatrixSpikeCountMonkey3)
    Z = [Z; CorrMatrixSpikeCountMonkey3(i+1:end,i)];
end

% plot spike count correlation vs Electrode distances (electrode = units)

% sorting
Xunique = 0.00025:0.0005:0.00425;
binLX = 0.00025;

Yunique = -1:0.25:1;
binLY = 0.125;

ZZ = zeros(length(Xunique),length(Yunique));

for i = 1:length(Xunique)
    for j = 1:length(Yunique)
        ZZ(i,j) = mean(Z(find(X < Xunique(i) + binLX & X >= Xunique(i) - binLX & ...
            Y < Yunique(j) + binLY & Y >= Yunique(j) - binLY)));
    end
end


figure;
pcolor(Xunique*1000,(Yunique),ZZ.');
colormap jet
shading interp
title('Spike Count Correlation || Monkey3','interpreter','latex');
xlabel('Distance Between Electrodes(mm)','interpreter','latex');
ylabel('rsignal','interpreter','latex');
