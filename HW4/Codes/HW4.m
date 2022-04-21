%% Advanced Neuroscience - Armin Panjehpour
%%% Traveling Wave

%% part 0&1 - preprocessings
%%%%%%%%%%%%%%%%%%%%%%%%%%

%% part.0.1 - load the data
clc; clear; close all;
ArrayData = load('ArrayData.mat');
CleanTrials = load('CleanTrials.mat');

%% part.0.2 - keep clean trials
clc; close all;

LFPsignals = zeros(length(ArrayData.chan),size(ArrayData.chan(1).lfp,1),...
    length(CleanTrials.Intersect_Clean_Trials));

for i = 1:size(LFPsignals,1)
    LFPsignals(i,:,:) = ArrayData.chan(i).lfp(:,CleanTrials.Intersect_Clean_Trials);
end

%% part.1.a - removing pink and finding dominant frequencies
clc; close all;


Fs = 1/(ArrayData.Time(2)-ArrayData.Time(1));
DominantFreqs = zeros(1,size(LFPsignals,1));
powAllelecs = zeros(size(LFPsignals,1),size(LFPsignals,3),320);
powAllelecsPinkFree = zeros(size(LFPsignals,1),size(LFPsignals,3),320);
pinkNoiseAll = zeros(size(LFPsignals,1),size(LFPsignals,3),320);


for i = 1:size(LFPsignals,1)
    for j = 1:size(LFPsignals,3)
        targetTrial = j;
        targetElectrode = i;
        targetSig = (LFPsignals(targetElectrode,:,j));

        % signal fft
        [fSignal, powSignal] = calFFT(targetSig,Fs);
        powAllelecs(i,j,:) = powSignal;

        % pink noise 
        a = -1;
        b = mean(log10(powSignal) - a*log10((fSignal)));
        pinkNoiseAll(i,j,:) = a*log10(fSignal) + b;
        
        % remove pink noise
        dePinkedSignal = (log10(powSignal)-...
            reshape(pinkNoiseAll(i,j,:),[1 320]));
        
        pinkNoiseAll(i,j,:) = 10.^pinkNoiseAll(i,j,:);
        powAllelecsPinkFree(i,j,:) = 10.^dePinkedSignal;
    end
    
    %%%% plot raw signal and pink noise - plot 48 figures! be careful.
%     plotRawSigPinkNoise(log10(fSignal),...
%     reshape(log10(mean(pinkNoiseAll(i,:,:),2)),[1 320]),...
%     reshape(log10(mean(powAllelecs(i,:,:),2)),[1 320]),...
%     reshape(log10(mean(powAllelecsPinkFree(i,:,:),2)),[1 320]),...
%     targetElectrode);

    % find the dominant frequency
    finded = (find(mean(powAllelecsPinkFree(i,:,:),2) == ...
        max(mean(powAllelecsPinkFree(i,:,:),2))));
    DominantFreqs(i) = fSignal(finded);
    
end

% fft average over all channels and trials
figure;
plot(fSignal,reshape((pow2db(mean(powAllelecs,[1 2]))),[1 320]),...
    'LineWidth',2,'Color','r');
hold on;
plot(fSignal,reshape((pow2db(mean(powAllelecsPinkFree,[1 2]))),[1 320]),...
    'LineWidth',2,'Color','b');
title('fft average over all trials of all channels','interpreter','latex');
xlabel('Frequency(Hz)','interpreter','latex');
ylabel('Power(dB)','interpreter','latex');
legend('Raw Data','DePinkNoised Data');
grid on; grid minor;

% domaninant frequenceis on a matrix formation
DominantFreqsMat = nan*zeros(5,10);
for i = 1:length(ArrayData.chan)
    [yy, xx] = find(ismember(ArrayData.ChannelPosition,i));
    DominantFreqsMat(yy,xx) = DominantFreqs(i);
end

xElectrodes = 1:10;
yElectrodes = 1:5;

figure;
h = heatmap(xElectrodes,yElectrodes,DominantFreqsMat);
h.Title = 'Dominant Frequencies of Electrodes (Hz)';
colormap winter

%% part.1.b - clustering
% as the dominant frequencies are almost the same, we get all
% electrodes in 1 cluster in the range of [12 13]


%% part.1.c - time frequency analysis with stft
clc; close all;


Fs = 1/(ArrayData.Time(2)-ArrayData.Time(1));
powAll = zeros(size(LFPsignals,1),size(LFPsignals,3),64,22);
powAllDepinkedNoise = zeros(size(LFPsignals,1),size(LFPsignals,3),64,22);


for i = 1:size(LFPsignals,1)
    for j = 1:size(LFPsignals,3)
        targetElectrode = i;
        targetSig = (LFPsignals(targetElectrode,:,j));

        % signal fft
        [s, f, t] = stft(targetSig,Fs,'Window',kaiser(100,4),...
            'OverlapLength',75,'FFTLength',128);
        f = f(end/2+1:end);
        s = s(end/2+1:end,:);
          
        powSignal = (abs(s).^2);
        powAll(i,j,:,:) = pow2db(powSignal);

     
        % pink noise 
        a = -1*ones(1,size(powSignal,2));
        aMat = (repmat(a(1)*log10(f),...
            1,size(powSignal,2)));
        aMatt = repmat(a,size(powSignal,1),1);
        b = mean(log10(powSignal) - aMat,1);
        pinkNoise = aMatt.*log10(f) + b;

        % remove pink noise
        dePinkedSignal = (log10(powSignal)-...
            pinkNoise);
        powAllDepinkedNoise(i,j,:,:) = pow2db(10.^dePinkedSignal);
    end 
end

powerr = reshape(mean(powAll,[1 2]),[64,22]);
powerrDePinkNoised = reshape(mean(powAllDepinkedNoise,[1 2]),[64,22]);


figure;
subplot(1,2,1);
pcolor(t,f,powerr);
colormap jet	
h = colorbar;
h.Label.String = 'Power(dB)';
shading interp
title('Raw Data - Average Over All Trials of All Channels','interpreter','latex');
xlabel('Time(s)','interpreter','latex');
ylabel('Frequency(Hz)','interpreter','latex');
tts = xline(1.2,'--r',{'ON-SET'},'Color','black','LineWidth',1.5);


subplot(1,2,2);
pcolor(t,f,powerrDePinkNoised);
shading interp
h = colorbar;
h.Label.String = 'Power(dB)';
title(' DepinkedNoise - Average Over All Trials of All Channels','interpreter','latex');
xlabel('Time(s)','interpreter','latex');
ylabel('Frequency(Hz)','interpreter','latex');
hold on;
tts = xline(1.2,'--r',{'ON-SET'},'Color','black','LineWidth',1.5);


%% part 1.4
% theory - in report pdf

%% part 2 - Phase Propagation(Traveling Wave)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% part 2.a - bandpass around dominant frequency
clc; close all;

% as we founded on part 1.a, the dominant freq of all electrodes where
% 12.48 Hz but one which was 12.79 Hz
% so we bandpass the data around [11 14]

% design a second order butterworth
fc = 12.5;
w_filter = 1.5;
Fs = 1/(ArrayData.Time(2)-ArrayData.Time(1));
order = 2;
[b,a] = butter(order, [fc-w_filter, fc+w_filter]/Fs*2,'bandpass');

% as you can see, the phase of the designed filter is linear in our
% frequency range of interest ([12 13])
figure;
freqz(b,a,1000,Fs);
grid on; grid minor;



% filter the data
filteredData = filter(b,a,permute(LFPsignals,[2 1 3]));

% to make sure of our filtering, we plot the mean filtered data stft
powAll = zeros(size(LFPsignals,1),size(LFPsignals,3),64,22);
powAllDepinkedNoise = zeros(size(LFPsignals,1),size(LFPsignals,3),64,22);


for i = 1:size(LFPsignals,1)
    for j = 1:size(LFPsignals,3)
        targetElectrode = i;
        targetSig = (LFPsignals(targetElectrode,:,j));

        % signal fft of raw data 
        [s, f, t] = stft(targetSig,Fs,'Window',kaiser(100,4),...
            'OverlapLength',75,'FFTLength',128);
        
        f = f(end/2+1:end);
        s = s(end/2+1:end,:);
        powSignal = (abs(s).^2);
        powAll(i,j,:,:) = pow2db(powSignal);

        % signal fft of filtered data 
        targetSig = (filteredData(:,targetElectrode,j));
        [s, f, t] = stft(targetSig,Fs,'Window',kaiser(100,4),...
            'OverlapLength',75,'FFTLength',128);
        
        f = f(end/2+1:end);
        s = s(end/2+1:end,:);
        powSignal = (abs(s).^2);
        powAllDepinkedNoise(i,j,:,:) = pow2db(powSignal);
    end 
end

powerr = reshape(mean(powAll,[1 2]),[64,22]);
powerrFiltered = reshape(mean(powAllDepinkedNoise,[1 2]),[64,22]);

figure;
subplot(1,2,1);
pcolor(t,f,powerr)
shading interp
h = colorbar;
h.Label.String = 'Power(dB)';
title('Raw Data - Averaged Over All Trials of All Channels',...
    'interpreter','latex')
xlabel('Time(s)','interpreter','latex');
ylabel('Frequency(Hz)','interpreter','latex');
colormap jet	

subplot(1,2,2);
pcolor(t,f,powerrFiltered)
shading interp
h = colorbar;
h.Label.String = 'Power(dB)';
title('Filtered Data - Averaged Over All Trials of All Channels',...
    'interpreter','latex')
xlabel('Time(s)','interpreter','latex');
ylabel('Frequency(Hz)','interpreter','latex');

%% part 2.b - instantaneous phase of electrodes 
clc; close all;

phi = nan*zeros(size(ArrayData.ChannelPosition,1),...
    size(ArrayData.ChannelPosition,2),...
    size(filteredData,3),size(filteredData,1));

for i = 1:size(filteredData,2)
    for j = 1:size(filteredData,3)
        yy = mod(i,5)+1;
        xx = floor(i/5)+1;
        phi(yy,xx,j,:) = angle(hilbert(filteredData(:,i,j)));
    end
end

%% part 2.c - cos(phi) and phi demo
clc; close all;

Fs = 1/(ArrayData.Time(2)-ArrayData.Time(1));
x = 1:10;
y = 1:5;

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);


frames = [];

for i = 1:size(phi,4)
    imagesc(x,y,cos(phiSelected(:,:,i)));
    colormap hot
    colorbar
    caxis([-1 1])
    formatSpec = '%0.4f';
    title("Wave - TrialNum = " + selectedTrial +...
        " - time(s) = " + compose(formatSpec,((i-1.2*Fs)/Fs)), 'interpreter', 'latex')
    frames = [frames getframe(gcf)];
    drawnow
end

writerObj = VideoWriter("Wave");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)

 
%% part 2.d.1 - Gradiant Directionality and Contour
clc; close all

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);

x = 1:10;
y = 1:5;


% now we have to calculate the gradiant vectors
[px,py] = gradient(unwrap(phiSelected));

% plot gradiants
figure
frames = [];

for i = 1:size(phi,4)
    colormap(hot)
    contour(x,y,(phiSelected(:,:,i)))
    hold on
    quiver(x,y,px(:,:,i),py(:,:,i),'color',[0 0 0])
    set(gca,'YDir','reverse')
    formatSpec = '%0.4f';
    title("Gradiant Directions and Contour of phi - TrialNum = " + selectedTrial +...
        " - time(s) = " + compose(formatSpec,((i-1.2*Fs)/Fs)),'interpreter', 'latex')
    hold off
    frames = [frames getframe(gcf)];
    drawnow
end

writerObj = VideoWriter("GradDir");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)



%% part 2.d.2 - Gradiant Directionality Histogram
clc; close all

frames = [];

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);

% now we have to calculate the gradiant vectors
[px,py] = gradient(unwrap(phiSelected));

for i = 1:size(phi,4)
    theta = atan2(py(:,:,i),px(:,:,i));
    nbin = 20;
    polarhistogram(theta,nbin,'FaceColor','red');
    formatSpec = '%0.4f';
    title("Gradiant Directions Histogram - TrialNum = " + selectedTrial + ...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    frames = [frames getframe(gcf)];
    drawnow
end

writerObj = VideoWriter("GradDirHist");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)


%% part 2.d.3 - Averaged Gradiant Directionality 
clc; close all

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);

% now we have to calculate the gradiant vectors
[px,py] = gradient(unwrap(phiSelected));


frames = [];
averageDir = zeros(1,size(phi,4));

for i = 1:size(phi,4)
    theta = atan2(py(:,:,i),px(:,:,i));
    theta = circularMean(theta(~isnan(theta)));
    averageDir(i) = theta;
    polarhistogram(averageDir(i)...
        ,200,'FaceColor','blue',...
        'EdgeColor','blue');
    formatSpec = '%0.4f';
    title("Averaged Gradiant Direction - TrialNum = " + selectedTrial + ...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    frames = [frames getframe(gcf)];
    
    drawnow
end

writerObj = VideoWriter("AvgGradDir");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)


%% part 2.d.4 - Phase Gradient Directionality(PGD)
clc; close all

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);

% now we have to calculate the gradiant vectors
[px,py] = gradient(unwrap(phiSelected));

pgd = zeros(1,size(phi,4));

frames = [];
saveOnset = 0;

for i = 1:size(phi,4)
    pgd(i) = calPGD(py(:,:,i),px(:,:,i));
    bar(pgd(i),'FaceColor',[0 0.4470 0.7410]);
    grid on; grid minor;
    ylim([0 1])
    formatSpec = '%0.4f';
    title("PGD - TrialNum = " + selectedTrial + ...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    hold on;
    if(i/Fs < 1.2)
        yline(mean(pgd(1:i)),'Color','red','LineWidth',2);
        saveOnset = i;
    else
        yline(mean(pgd(1:saveOnset)),'Color','red','LineWidth',2);
        yline(mean(pgd(saveOnset+1:i)),'Color','black','LineWidth',2);
    end
    hold off
    legend('PGD(t)','mean(PGD(t) || BeforeOnset)','mean(PGD(t) || AfterOnset)')
    drawnow
    frames = [frames getframe(gcf)];
end


writerObj = VideoWriter("PGD");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)


%histogram for one trial
figure; 
histogram(pgd,100)
xlabel('PGD Value','interpreter','latex')
ylabel('PGD Count','interpreter','latex')
title("PGD histogram - Trial = " + selectedTrial,'interpreter','latex');
grid on; grid minor;


%% part 2.d.5 - Speed Magnitude of wave
clc; close all;

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);

% now we have to calculate the gradiant vectors
[px,py] = gradient(unwrap(phiSelected));


speed = zeros(1,size(phi,4)-1);

frames = [];
deriv = diff(phiSelected,1,3);
saveOnset = 0;

for i = 1:size(phi,4)-1
    speed(i) = calSpeed(py(:,:,i),px(:,:,i),deriv(:,:,i));
    bar(speed(i),'FaceColor',[0 0.4470 0.7410]);
    grid on; grid minor;
    ylim([0 100])
    formatSpec = '%0.4f';
    title("Speed Magnitude(cm/s) - TrialNum = " + selectedTrial + ...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    hold on;
    if(i/Fs < 1.2)
        yline(mean(speed(1:i)),'Color','red','LineWidth',2);
        saveOnset = i;
    else
        yline(mean(speed(saveOnset)),'Color','red','LineWidth',2);
        yline(mean(speed(saveOnset+1:i)),'Color','black','LineWidth',2);
    end
    hold off
    legend('Speed(t)','mean(Speed(t) || BeforeOnset)','mean(Speed(t) || AfterOnset)')
    
    drawnow

    frames = [frames getframe(gcf)];    
end


writerObj = VideoWriter("WaveSpeedMagnitude");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)


%histogram for one trial
figure; 
histogram(speed,100)
xlabel('Speed Value(cm/s)','interpreter','latex')
ylabel('Speed Count','interpreter','latex')
title("Speed histogram - Trial = " + selectedTrial,'interpreter','latex');
grid on; grid minor;


%% part 2.e - plot all 6 previous plots in a figure;
%%% cos(phi) - wave - gradiant direction & contour - gradiant direction
%%% histogram - average gradient direction - PGD - speed magnitude of wave
clc; close all;


formatSpec = '%0.4f';
% full screen figure
figure('units','normalized','outerposition',[0 0 1 1])

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);

x = 1:10;
y = 1:5;

% now we have to calculate the gradiant vectors
[px,py] = gradient(unwrap(phiSelected));

frames = [];
for i = 1:size(phi,4)-1
    %%% Wave
    subplot(2,3,1)
    imagesc(x,y,cos(phiSelected(:,:,i)));
    colormap hot
    colorbar
    caxis([-1 1])
    title("Wave - TrialNum = " + selectedTrial +...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    
    
    %%% Gradiant Directions & Contour of phi
    subplot(2,3,4)
    colormap hot
    contour(x,y,(phiSelected(:,:,i)))
    hold on
    quiver(x,y,px(:,:,i),py(:,:,i),'color',[0 0 0])
    set(gca,'YDir','reverse')
    title("Gradiant Directions and Contour of phi - TrialNum = " + selectedTrial +...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)),'interpreter', 'latex')
    hold off
    
    
    %%% Gradiant Directions Histogram
    subplot(2,3,2)
    theta = atan2(py(:,:,i),px(:,:,i));
    nbin = 20;
    polarhistogram(theta,nbin,'FaceColor','red');
    title("Gradiant Directions Histogram - TrialNum = " + selectedTrial + ...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    
    
    %%% Averaged Gradiant Direction
    subplot(2,3,5)
    polarhistogram(averageDir(i)...
        ,200,'FaceColor','blue',...
        'EdgeColor','blue');
    title("Averaged Gradiant Direction - TrialNum = " + selectedTrial + ...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    
    
    %%% PGD
    subplot(2,3,3)
    bar(pgd(i),'FaceColor',[0 0.4470 0.7410]);
    grid on; grid minor;
    ylim([0 1])
    title("PGD - TrialNum = " + selectedTrial + ...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    hold on;
    if(i/Fs < 1.2)
        yline(mean(pgd(1:i)),'Color','red','LineWidth',2);
        saveOnset = i;
    else
        yline(mean(pgd(saveOnset)),'Color','red','LineWidth',2);
        yline(mean(pgd(saveOnset+1:i)),'Color','black','LineWidth',2);
    end
    hold off
    legend('PGD(t)','mean(PGD(t) || BeforeOnset)','mean(PGD(t) || AfterOnset)')
    
    
    %%% Speed Magnitude
    subplot(2,3,6)
    bar(speed(i),'FaceColor',[0 0.4470 0.7410]);
    grid on; grid minor;
    ylim([0 100])
    title("Speed Magnitude(cm/s) - TrialNum = " + selectedTrial + ...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    hold on;
    if(i/Fs < 1.2)
        yline(mean(speed(1:i)),'Color','red','LineWidth',2);
        saveOnset = i;
    else
        yline(mean(speed(saveOnset)),'Color','red','LineWidth',2);
        yline(mean(speed(saveOnset+1:i)),'Color','black','LineWidth',2);
    end
    hold off
    legend('Speed(t)','mean(Speed(t) || BeforeOnset)','mean(Speed(t) || AfterOnset)')
    
    frames = [frames getframe(gcf)];    
    drawnow
end


writerObj = VideoWriter("AllInOne");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)

%% part 2.f - distribution of the wave direction
clc; close all;

%%% if we see the demo of the wave, it seems the wave, propagates from
%%% right or top right to the left or bottom left. it seems that the
%%% prefered direction of the wave is something between [0(180) 30(210)].
%%% to validate this, we design a test to prove its significance

% we take the times which PGD is more than 0.5 and check their propagation
% direction for all trials 


%%%%%% select all trials
phiSelected = phi;

px = zeros(size(phi));
py = zeros(size(phi));
pgdALL = zeros(size(phi));
GradDirectionsAllTrials = zeros(size(phi));

selectedWavesTrials = [];
selectedWavesTimes = [];

for j = 1:size(phi,3) % trials
    
    % now we have to calculate the gradiant vectors for this trial
    [px(:,:,j,:),py(:,:,j,:)] = gradient(unwrap(phiSelected(:,:,j,:)));
    
    % calculate PGD and Wave Direction for this trial over time
    
    for i = 1:size(phi,4) % time
        [j i]
        GradDirectionsAllTrials(:,:,j,i) = atan2(py(:,:,j,i),px(:,:,j,i));
        pgdALL(j,i) = calPGD(py(:,:,j,i),px(:,:,j,i));
        
        % waves with PGD more than 0.5
        if(pgdALL(j,i) >= 0.5)
            selectedWavesTrials = [selectedWavesTrials j];
            selectedWavesTimes = [selectedWavesTimes i];
        end
    end
end


direcs = zeros(size(phi,1),size(phi,2),length(selectedWavesTrials));
for i = 1:length(selectedWavesTrials)
    direcs(:,:,i) = (GradDirectionsAllTrials(:,:,selectedWavesTrials((i))...
        ,selectedWavesTimes((i))));
end


directionsFinal = [];
for i = 1:size(direcs,3)
    directionsFinal = [directionsFinal rad2deg(circularMean(direcs(:,:,i)))];
end
% full screen figure
figure('units','normalized','outerposition',[0 0 1 1])
polarhistogram(deg2rad(directionsFinal),50)
grid on; grid minor; 
title("Averaged Propagation Direction over all trials, electrodes and all times with PGD bigger than 0.5",...
    'interpreter','latex');

%% part 2.g - distribution of speed when PGD >= 0.5
clc; close all;


%%%%%% select all trials
phiSelected = phi;

px = zeros(size(phi));
py = zeros(size(phi));
pgdALL = zeros(size(phi,3),size(phi,4));
speedAllTrials = zeros(size(phi,3),size(phi,4));

selectedWavesTrials = [];
selectedWavesTimes = [];

for j = 1:size(phi,3) % trials
    
    % now we have to calculate the gradiant vectors for this trial
    [px(:,:,j,:),py(:,:,j,:)] = gradient(unwrap(phiSelected(:,:,j,:)));
    deriv = diff(phiSelected(:,:,j,:),1,4);
    deriv = reshape(deriv,[size(deriv,1) size(deriv,2) size(deriv,4)]);
        
    % calculate PGD and Wave Direction for this trial over time
    
    for i = 1:size(phi,4)-1 % time
        [j i]
        speedAllTrials(j,i) = calSpeed(py(:,:,j,i),px(:,:,j,i),deriv(:,:,i));
        pgdALL(j,i) = calPGD(py(:,:,j,i),px(:,:,j,i));
        
        % waves with PGD more than 0.5
        if(pgdALL(j,i) >= 0.5)
            selectedWavesTrials = [selectedWavesTrials j];
            selectedWavesTimes = [selectedWavesTimes i];
        end
    end
end


speedsAll = zeros(1,length(selectedWavesTrials));
for i = 1:length(selectedWavesTrials)
    speedsAll(i) = (speedAllTrials(selectedWavesTrials((i))...
        ,selectedWavesTimes((i))));
end


% full screen figure
figure('units','normalized','outerposition',[0 0 1 1])
histogram(speedsAll,900);
xlim([0 200])
grid on; grid minor; 
xlabel('Speed (cm/s)','interpreter','latex')
ylabel('Speed Count','interpreter','latex')
title("Averaged Speed Direction over all trials, electrodes and all times with PGD bigger than 0.5",...
    'interpreter','latex');

%% %% part 2.h - distribution of PGDs over all trials and times
clc; close all;

histogram(pgdALL(:),200);
xlabel('PGD Value','interpreter','latex')
ylabel('PGD Count','interpreter','latex')
title("PGD histogram",'interpreter','latex');
grid on; grid minor;

%% part 2.I - arbitary - calculating the propagation consistency by fitting a plane
clc; close all;

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);


% fit a plane to our phase in each time sample

formatSpec = '%0.4f';

X = [];
for i = 1:10
    for j = 1:5
        X = [X; [1 j i]];
    end
end

frames = [];


for i = 1:size(phi,4)
    
    % fit the plane
    Y = cosd(rad2deg(reshape(phiSelected(:,:,i),[],size(phi,1)*size(phi,2)))+180);
    [b,bint,r,rint,stats]  = regress(Y',X);
    
    
    x1fit = min(X(:,2)):1:max(X(:,2));
    x2fit = min(X(:,3)):1:max(X(:,3));
    [X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
    YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
    scatter3(X(:,2),X(:,3),(r+reshape(YFIT,[1 size(phi,1)*size(phi,2)]).'),...
    'filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor','black');
    zlim([-1 1])
    
    hold on;
    s = mesh(X1FIT,X2FIT,YFIT,'EdgeColor','black','FaceAlpha','0.9');
    s.FaceColor = 'flat';
    colorbar
    caxis([-1 1])
    hold off
    title("cos (Relative Phase) - TrialNum = " + selectedTrial +...
        " - time(s) = " + compose(formatSpec,(((i)-1.2*Fs)/Fs)), 'interpreter', 'latex')
    view(65,20)
    
    drawnow
    
    frames = [frames getframe(gcf)];    
end

writerObj = VideoWriter("FittedPlane");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)


%% part 2.J - arbitary - calculating the alternate form of PGD for the fitted plane

clc; close all;

% select a trial
selectedTrial = 259;
phiSelected = reshape(phi(:,:,selectedTrial,:),[size(phi,1) size(phi,2)...
    size(phi,4)]);


% fit a plane to our phase in each time sample

X = [];
for i = 1:10
    for j = 1:5
        X = [X; [1 j i]];
    end
end

frames = [];
FittingMeasure = zeros(1,size(phi,4));
alterPGD = zeros(1,size(phi,4));


% full screen figure
figure('units','normalized','outerposition',[0 0 1 1])
saveOnset = 0;
for i = 1:size(phi,4)
    

    % fit the plane
    Y = cosd(rad2deg(reshape(phiSelected(:,:,i),[],size(phi,1)*size(phi,2)))+180);
    [b,bint,r,rint,stats]  = regress(Y',X);
    

    
    x1fit = min(X(:,2)):1:max(X(:,2));
    x2fit = min(X(:,3)):1:max(X(:,3));
    [X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
    YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
    subplot(1,3,1);
    scatter3(X(:,2),X(:,3),(r+reshape(YFIT,[1 size(phi,1)*size(phi,2)]).'),...
        'filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor','black');
    zlim([-1 1])
    
    hold on;
    s = mesh(X1FIT,X2FIT,YFIT,'EdgeColor','black','FaceAlpha','0.9');
    s.FaceColor = 'flat';
    colorbar
    caxis([-1 1])
    title("cos(Relative Phase) - TrialNum = " + selectedTrial +...
        " - time(s) = " + (i-1.2*Fs)/Fs, 'interpreter', 'latex')
    view(65,20)
    
    hold off

    %%% plot alternate pgd
    subplot(1,3,2);
    Y = (rad2deg(reshape(phiSelected(:,:,i),[],size(phi,1)*size(phi,2)))+180);
    [b,bint,r,rint,stats]  = regress(Y',X);
    x1fit = min(X(:,2)):1:max(X(:,2));
    x2fit = min(X(:,3)):1:max(X(:,3));
    [X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
    YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
    
    % measure of fitting
    FittingMeasure(i) = FitPlaneMeasure((rad2deg(phiSelected(:,:,i).').'+180),(YFIT.'));
    bar(FittingMeasure(i),'FaceColor',[0 0.4470 0.7410]);
    ylim([0 1]);
    grid on; grid minor;
    title("FittingMeasure - TrialNum = " + selectedTrial +...
        " - time(s) = " + (i-1.2*Fs)/Fs, 'interpreter', 'latex')
    
        hold on;
    if(i/Fs < 1.2)
        yline(mean(FittingMeasure(1:i)),'Color','red','LineWidth',2);
        saveOnset = i;
    else
        yline(mean(FittingMeasure(1:saveOnset)),'Color','red','LineWidth',2);
        yline(mean(FittingMeasure(saveOnset+1:i)),'Color','black','LineWidth',2);
    end
    legend('FittingMeasure(t)','mean(FittingMeasure(t) || BeforeOnset)','mean(FittingMeasure(t) || AfterOnset)')
    hold off
    
    % PGD
    subplot(1,3,3);
    alterPGD(i) = AlterPGD((rad2deg(phiSelected(:,:,i).').'+180),(YFIT.'));
    bar(alterPGD(i),'FaceColor',[0 0.4470 0.7410]);
    ylim([-0.1 1]);
    grid on; grid minor;
    title("AlternatePGD - TrialNum = " + selectedTrial +...
        " - time(s) = " + (i-1.2*Fs)/Fs, 'interpreter', 'latex')
    
        hold on;
    if(i/Fs < 1.2)
        yline(mean(alterPGD(1:i)),'Color','red','LineWidth',2);
        saveOnset = i;
    else
        yline(mean(alterPGD(1:saveOnset)),'Color','red','LineWidth',2);
        yline(mean(alterPGD(saveOnset+1:i)),'Color','black','LineWidth',2);
    end
    legend('AlternatePGD(t)','mean(AlternatePGD(t) || BeforeOnset)','mean(AlternatePGD(t) || AfterOnset)')
    hold off;
    
    
    drawnow
    
    frames = [frames getframe(gcf)];    
end

writerObj = VideoWriter("FittedPlaneAndPGD");
writerObj.FrameRate = 20;
writerObj.Quality = 100;

% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(frames)
	% convert the image to a frame
    frame = frames(i) ;
    writeVideo(writerObj,frame);
end
% close the writer o bject
close(writerObj)



%% part 2.11 - Alternate vs Original PGD
clc; close all;

% select a trial
phiSelected = phi;


% fit a plane to our phase in each time sample

X = [];
for i = 1:10
    for j = 1:5
        X = [X; [1 j i]];
    end
end


alterPGD = zeros(size(phi,3),size(phi,4));

for j = 1:size(phi,3)
    for i = 1:size(phi,4)
        [j i]
        % fit the plane
        Y = cosd(rad2deg(reshape(phiSelected(:,:,j,i),[],size(phi,1)*size(phi,2)))+180);
        [b,bint,r,rint,stats]  = regress(Y',X);


        x1fit = min(X(:,2)):1:max(X(:,2));
        x2fit = min(X(:,3)):1:max(X(:,3));
        [X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
        YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;

        alterPGD(j,i) = AlterPGD((rad2deg(phiSelected(:,:,j,i).').'+180),(YFIT.'));
    end
end


figure
histogram(alterPGD(:),200);
xlabel('Alternate PGD Value','interpreter','latex')
ylabel('Alternate PGD Count','interpreter','latex')
title("Alternate PGD histogram",'interpreter','latex');
grid on; grid minor;
