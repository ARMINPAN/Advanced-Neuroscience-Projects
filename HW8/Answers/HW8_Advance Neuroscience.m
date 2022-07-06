%% HW8 - Learning to predict where people look - Visual Search
% Armin Panjehpour - 98101288

%% Database Code
% add the functions of Eye tracking database to the path before running
clc; close all; clear;

showEyeData('../DATA/hp', '../ALLSTIMULI')
%% Saliancy Map output with just one feature
clear; close all; clc;

imName = 'i10feb04_static_cars_highland_img_0847.jpeg';
selectedStimuli = imread(imName);
[w, h, c] = size(selectedStimuli);
dims = [200, 200];


% load features
for i = 1:8
    FEATURES = [];
    if i == 1
        FEATURES(:, 1:13) = findSubbandFeatures(selectedStimuli, dims);
    elseif i == 2
        FEATURES(:, 14:16) = findIttiFeatures(selectedStimuli, dims);
    elseif i == 3
        FEATURES(:, 17:27) = findColorFeatures(selectedStimuli, dims);
    elseif i == 4
        FEATURES(:, 28) = findTorralbaSaliency(selectedStimuli, dims);
    elseif i == 5
        FEATURES(:, 29) = findHorizonFeatures(selectedStimuli, dims);
    elseif i == 6
        FEATURES(:, 30:31) = findObjectFeatures(selectedStimuli, dims);
    elseif i == 7
        FEATURES(:, 32) = findDistToCenterFeatures(selectedStimuli, dims);
    else
        FEATURES(:, 1:13) = findSubbandFeatures(selectedStimuli, dims);
        FEATURES(:, 14:16) = findIttiFeatures(selectedStimuli, dims);
        FEATURES(:, 17:27) = findColorFeatures(selectedStimuli, dims);
        FEATURES(:, 28) = findTorralbaSaliency(selectedStimuli, dims);
        FEATURES(:, 29) = findHorizonFeatures(selectedStimuli, dims);
        FEATURES(:, 30:31) = findObjectFeatures(selectedStimuli, dims);
        FEATURES(:, 32) = findDistToCenterFeatures(selectedStimuli, dims);
    end
    % for exporting the maps, uncomment the save function in saliency
    saliencyMap = saliency(selectedStimuli,FEATURES,i);
end

%% Saliancy Map output with just one feature commented
clear; close all; clc;

imName = 'i10feb04_static_cars_highland_img_0847.jpeg';
selectedStimuli = imread(imName);
[w, h, c] = size(selectedStimuli);
dims = [200, 200];

% load features
for i = 1:8
    FEATURES(:, 1:13) = findSubbandFeatures(selectedStimuli, dims);
    FEATURES(:, 14:16) = findIttiFeatures(selectedStimuli, dims);
    FEATURES(:, 17:27) = findColorFeatures(selectedStimuli, dims);
    FEATURES(:, 28) = findTorralbaSaliency(selectedStimuli, dims);
    FEATURES(:, 29) = findHorizonFeatures(selectedStimuli, dims);
    FEATURES(:, 30:31) = findObjectFeatures(selectedStimuli, dims);
    FEATURES(:, 32) = findDistToCenterFeatures(selectedStimuli, dims);
    if i == 1
        FEATURES(:, 1:13) = [];
    elseif i == 2
        FEATURES(:, 14:16) = [];
    elseif i == 3
        FEATURES(:, 17:27) = [];
    elseif i == 4
        FEATURES(:, 28) = [];
    elseif i == 5
        FEATURES(:, 29) = [];
    elseif i == 6
        FEATURES(:, 30:31) = [];
    elseif i == 7
        FEATURES(:, 32) = [];
    end
    % for exporting the maps, uncomment the save function in saliency
    saliencyMap = saliency(selectedStimuli,FEATURES,i);
end

%% All Saliancy Maps and then ROC
clc; close all; clear;

files = dir(fullfile('ALLSTIMULI/','*.jpeg'));
[filenames{1:size(files,1)}] = deal(files.name);

ROCs = zeros(15,length(filenames),8,2);


for j = 1:length(filenames)
    name = filenames{j};
    name = name(1:end-5);
    fixationPoints = dir(fullfile(convertStringsToChars("DATA/**/"+(name + "*.mat"))));

    imName = filenames{j};
    selectedStimuli = imread(imName);
    [w, h, c] = size(selectedStimuli);
    dims = [200, 200];

    % load features
    FEATURES(:, 1:13) = findSubbandFeatures(selectedStimuli, dims);
    FEATURES(:, 14:16) = findIttiFeatures(selectedStimuli, dims);
    FEATURES(:, 17:27) = findColorFeatures(selectedStimuli, dims);
    FEATURES(:, 28) = findTorralbaSaliency(selectedStimuli, dims);
    FEATURES(:, 29) = findHorizonFeatures(selectedStimuli, dims);
    FEATURES(:, 30:31) = findObjectFeatures(selectedStimuli, dims);
    FEATURES(:, 32) = findDistToCenterFeatures(selectedStimuli, dims);
    FEATURES_saved = FEATURES;
    for i = 1:8
        FEATURES = FEATURES_saved;
        if i == 1
            FEATURES(:, 1:13) = [];
        elseif i == 2
            FEATURES(:, 14:16) = [];
        elseif i == 3
            FEATURES(:, 17:27) = [];
        elseif i == 4
            FEATURES(:, 28) = [];
        elseif i == 5
            FEATURES(:, 29) = [];
        elseif i == 6
            FEATURES(:, 30:31) = [];
        elseif i == 7
            FEATURES(:, 32) = [];
        else 
            FEATURES = FEATURES_saved;
        end
        saliencyMap = saliency(selectedStimuli,FEATURES,i);
        for k = 1:15 
            [j i k]
            FP = load(fullfile(fixationPoints(k).folder,fixationPoints(k).name));
            fielVal = fieldnames(FP);
            FP = getfield(FP,fielVal{1});
            X = FP.DATA.eyeData(:,1);
            Y = FP.DATA.eyeData(:,2);
            X(isnan(X)) = [];
            Y(isnan(Y)) = [];
            origimgsize = size(saliencyMap);
            X1 = X(1:floor(end/2)); X2 = X(floor(end/2)+1:end);
            Y1 = Y(1:floor(end/2)); Y2 = Y(floor(end/2)+1:end);
            ROCs(k,j,i,1) = rocScoreSaliencyVsFixations(saliencyMap,X1,Y1,origimgsize);
            ROCs(k,j,i,2) = rocScoreSaliencyVsFixations(saliencyMap,X2,Y2,origimgsize);
        end
    end
    
end

%% histogram of ROCs over subjects and images
clc; close all;
ROCss = ROCs(:,:,:,:);
figure
histogram(ROCss(:,:,7,1),20)
hold on;
histogram(ROCss(:,:,8,1),20)

title('Comparing ROCs - First 1.5s','interpreter','latex');
xlabel('ROC Values','interpreter','latex');
ylabel('Count','interpreter','latex');
legend('Just Feature 7','ALL Features Used','Location','best')

%% mean ROC over subjects and images for bottom up and top down

figure;
meanRocs = squeeze(mean(ROCs,[1 2]));
errorROCs = squeeze(std(ROCs,0,[1 2]))/sqrt(15*1003);

errorbar(1:8,meanRocs(:,1),errorROCs(:,1))
hold on;
errorbar(1:8,meanRocs(:,2),errorROCs(:,2))
grid on; grid minor;
legend('bottom-up','top-down','location','best')
title('Mean ROC Over All Images and Subjects','interpreter','latex')
ylabel('ROC Value','interpreter','latex')

names = {'SubbandFeatures'; 'IttiFeatures'; 'ColorFeatures';...
    'TorralbaSaliency'; 'HorizonFeatures'; 'ObjectFeatures'; 'DistToCenterFeatures'; ...
    'AllFeatures'};

set(gca,'xtick',[1:8],'xticklabel',names)
xtickangle(45)
