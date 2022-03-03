%% Advanced Neuroscience - Armin Panjehpour
%%% Neural Coding - Softky & Koch 1993 - IF & LIF model

%% Part.1 - Integrate & Fire Neuron

%% Part.1.a - Poisson Renewal Point Process - Spike Train
clc; clear; close all;
r = 100; % rate
dt = 0.00001; % s - sample rate

t = 0;
countSpikes = 0;
ISI = []; % a vector to keep ISI values
timeSpikes = []; % a vector to keep spike times

% a total duration of 1s
while t < 1
    ISI = [ISI exprnd(1/r)];
    t = t + ISI(countSpikes+1);
    timeSpikes = [timeSpikes t];
    countSpikes = countSpikes + 1;
end

% Spike train plot
figure;
stem(timeSpikes,ones(1,countSpikes),'filled');
xlabel('time(s)','interpreter','latex');
title('spike train','interpreter','latex');
grid on; grid minor;
xlim([0 timeSpikes(end)]);
ylim([0 2]);
%% Part.1.b - spike count probability
clc; clear; close all;
r = 100; % rate
dt = 0.00001; % s - sample rate

t = 0;
countSpikes = [0];
ISI = []; % a vector to keep ISI values

% calculate for 1000 trials
for i = 1:1000
    % a total duration of 1s
    while t < 1
        ISI = [ISI exprnd(1/r)];
        t = t + ISI(countSpikes(i)+1);
        countSpikes(i) = countSpikes(i) + 1;
    end
    countSpikes = [countSpikes 0];
    t = 0;
    ISI = [];
end
countSpikes(end) = []; % 

% fit
figure;
subplot(1,2,1);
histogram(countSpikes,100,'Normalization','probability');
title('spike count probability','interpreter','latex');
xlabel('spike count','interpreter','latex');
ylabel('probability','interpreter','latex');
grid on; grid minor; 
hold on;

% theorical poisson - mean =  
poiss = poisspdf(1:2*r,r);
plot(poiss,'r','LineWidth',2)


subplot(1,2,2);
hist(countSpikes,10);
h = hist(countSpikes,10);
grid on; grid minor;
title('spike count rate','interpreter','latex');
xlabel('spike count','interpreter','latex');
ylabel('rate','interpreter','latex');
hold on;

% fit
plot((max(h)/max(poiss))*poiss,'r','LineWidth',2)

%% Part.1.c - ISI histogram - part.1.a should be ran before this part
figure;
hist(ISI,7);
h = hist(ISI,7);
title('ISI histogram','interpreter','latex');
grid on; grid minor;

hold on;

% theorical poisson - mean =  
expfun = r*exp(-r*(0:dt:0.1));
expfun = (max(h)/max(expfun)).*expfun;
plot((0:dt:0.1),expfun,'r','LineWidth',2)
xlabel('ISI(s)','interpreter','latex');
ylabel('Count','interpreter','latex');

%% Part.1.c_d(a) - integration over inputs
clc; clear; close all;
r = 100; % rate
dt = 0.00001; % s - sample rate

t = 0;
countSpikes = 0;
ISI = []; % a vector to keep ISI values
timeSpikes = []; % a vector to keep spike times

% a total duration of 1s
while t < 1
    ISI = [ISI exprnd(1/r)];
    t = t + ISI(countSpikes+1);
    timeSpikes = [timeSpikes t];
    countSpikes = countSpikes + 1;
end

figure;
subplot(2,1,1);
stem(timeSpikes,ones(1,countSpikes));
xlabel('time(s)','interpreter','latex');
title('Poisson spike train','interpreter','latex');
grid on; grid minor;
xlim([0 timeSpikes(end)]);
ylim([0 2]);

% just keep each k`th spike - using down sample 
k = 5;
timeSpikes = downsample(timeSpikes,k);
countSpikes = length(timeSpikes);
ISI = diff(timeSpikes);

subplot(2,1,2);
stem(timeSpikes,ones(1,countSpikes));
xlabel('time(s)','interpreter','latex');
title('Integrated over inputs spike train','interpreter','latex');
grid on; grid minor;
xlim([0 timeSpikes(end)]);
ylim([0 2]);

%% Part.1.c_d(b) - spike count probability

clc; clear; close all;
r = 100; % rate
dt = 0.00001; % s - sample rate

% simulation for 4 ks
k = [1 2 4 6];
% calculate for 1000 periods
figure;
for j = 1:4
    t = 0;
    countSpikes = [0];
    ISI = []; % a vector to keep ISI values
    timeSpikes = [];
    for i = 1:10000
        [j i]
        % a total duration of 5s
        while t < 1
            ISI = [ISI exprnd(1/r)];
            t = t + ISI(countSpikes(i)+1);
            timeSpikes = [timeSpikes t];
            countSpikes(i) = countSpikes(i) + 1;
        end

        % just keep each k`th spike - using down sample 

        timeSpikes = downsample(timeSpikes,k(j));
        ISI = diff(timeSpikes);
        countSpikes(i) = length(timeSpikes);
        countSpikes = [countSpikes 0];
        t = 0;
        ISI = [];
        timeSpikes = [];
    end
    countSpikes(end) = []; % 
    
    subplot(2,2,j)
    % fit
    histogram(countSpikes,100,'Normalization','probability');
    title("Spike Count Probability for k = " + k(j),'interpreter','latex');
    xlabel('Spike Count','interpreter','latex');
    ylabel('Probability','interpreter','latex');
    grid on; grid minor; 
    hold on;

    % theorical poisson - mean =  
    poiss = poisspdf(1:2*mean(countSpikes),mean(countSpikes));
    plot(poiss,'r','LineWidth',2)
    xlim([1 2*mean(countSpikes)])
end


%% Part.1.c_d(c) - ISI histogram - part.1.c_d(a) should be ran before this part
figure;
hist(ISI,15);
h = hist(ISI,15);
title('ISI histogram','interpreter','latex');
grid on; grid minor;

hold on;

% theorical poisson - mean =  
expfun = r*exp(-r*(0:dt:0.1));
expfun = (max(h)/max(expfun)).*expfun;
plot((0:dt:0.1),expfun,'r','LineWidth',2)
xlabel('ISI(s)','interpreter','latex');
ylabel('Count','interpreter','latex');
%% Part.1.d
clc; clear; close all;

% poisson
r = 100; % rate
dt = 0.00001; % s - sample rate


CVpoisson = zeros(1,100);
CVRenewal = zeros(1,100);

% simulation for 4 ks
k = [1 4 7 20];
c ={'black','b'};
% calculate CV for 100 simulations
for j=1:4
    for i=1:100
        % poisson process
        tPoisson = 0;
        countSpikesPoisson = 0;
        ISIPoisson = []; % a vector to keep ISI values
        timeSpikesPoisson = []; % a vector to keep spike times

        % a total duration of 1s
        while tPoisson < 1
            ISIPoisson = [ISIPoisson exprnd(1/r)];
            tPoisson = tPoisson + ISIPoisson(countSpikesPoisson+1);
            timeSpikesPoisson = [timeSpikesPoisson tPoisson];
            countSpikesPoisson = countSpikesPoisson + 1;
        end

        % Coefficients of Variation
        CVpoisson(i) = std(ISIPoisson)/mean(ISIPoisson);

        % just keep each k`th spike - using down sample 
        timeSpikesRenewal = downsample(timeSpikesPoisson,k(j));
        ISIRenewal = diff(timeSpikesRenewal);
        countSpikesRenewal = length(timeSpikesRenewal);

        % Coefficients of Variation
        CVRenewal(i) = std(ISIRenewal)/mean(ISIRenewal);
    end
    subplot(2,2,j);
    stem(CVpoisson)
    hold on;
    stem(CVRenewal)
    % averaged values over 100 simulations
    yline(mean(CVpoisson),'color',c{1},'LineWidth',1.5);
    yline(mean(CVRenewal),'color',c{2},'LineWidth',1.5);
    grid on; grid minor;
    title("Coefficient of Variation for k = "+k(j),'interpreter','latex');
    xlabel('Number of Simulation','interpreter','latex');
    ylabel('CV','interpreter','latex');
    legend('poisson','integrated','average poisson','averaged integrated')
end


%% Part.1.g
% at first we will create a spike train with a refractory period of 1ms

clc; clear; close all;
deltats = 0.0001:0.0001:30*10^(-3);
dt = 0.00001; % s - sample rate

t0 = 0.001:0.002:0.011; % refractory periods
k = [1 4 7 20]; % simulation for 4 ks

CVRenewal = zeros(length(deltats),length(t0),length(k));
deltaT = zeros(length(deltats),6,4);
jjj = 1;


figure;
for i=1:length(t0)
    for j=1:length(k)
        jjj = 1;
        while jjj <= length(deltats)
            [i j jjj]
            jj = 1;
            % run 100 simulations and averaged for better responses
            while jj <= 100
                % spike trains
                countSpikes = 0;
                t = 0;
                ISI = []; % a vector to keep ISI values
                timeSpikes = []; % a vector to keep spike times
                while t < 3
                    y = exprnd(deltats(jjj)) + t0(i);
                    ISI = [ISI y];
                    t = t + ISI(countSpikes+1);
                    timeSpikes = [timeSpikes t];
                    countSpikes = countSpikes + 1;
                end
                % integration over inputs
                % just keep each k`th spike - using down sample 
                timeSpikesRenewal = downsample(timeSpikes,k(j));
                ISIRenewal = diff(timeSpikesRenewal);
                countSpikesRenewal = length(timeSpikesRenewal);
                % Coefficients of Variation
                CVRenewal(jjj,i,j) = CVRenewal(jjj,i,j) + std(ISIRenewal)/mean(ISIRenewal);
                deltaT(jjj,i,j) = deltaT(jjj,i,j) + mean(ISI);
                if(jj == 100)
                    break
                end
                jj = jj + 1;
            end
            % average on trials
        CVRenewal(jjj,i,j) = CVRenewal(jjj,i,j) / jj;
        deltaT(jjj,i,j) = deltaT(jjj,i,j) / jj;
        jjj = jjj + 1;  
        end
    end
    subplot(2,3,i);
    stem(timeSpikes,ones(1,countSpikes));
    xlabel('time(s)','interpreter','latex');
    title("Poisson spike train for t0 = " + t0(i),'interpreter','latex');
    grid on; grid minor;
    xlim([0 timeSpikes(end)]);
    ylim([0 2]);
end

% CVs
figure;
c ={'g','r','b','k'};
for i=1:length(t0)
    subplot(2,3,i);
    title("for t0 = " + t0(i),'interpreter','latex');
    xlabel('mean(ISI(ms))','interpreter','latex');
    ylabel('CV','interpreter','latex');
    for j=1:length(k)
       hold on;
       yline(CVRenewal(end,i,j),'color',c{j},'LineWidth',1.5);
       scatter((deltaT(:,i,j)*1000),CVRenewal(:,i,j),8,'fill')
       legend("k = " + k(1),"k = " + k(1),"k = " + k(2),"k = " + k(2),...
           "k = " + k(3),"k = " + k(3),"k = " + k(4),"k = " + k(4));
       ylim([0 1.2]);
       grid on; grid minor; 
    end
end



%% Part.2 - Leaky Integrate & Fire Neuron
 
%% Part.2.a - LIF model with constent input current
clc; close all; clear;
dt = 0.01; % Simulation time step
Duration = 100; % Simulation length - ms
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms

vr = 0; % in mv, resting potential
tau_m = 5; %Arbitrary
v = vr * ones(1,T); % Vector of output voltage
vth = 0.015; % Threshold Voltage - in V
RI = 0.02; % Leaky part - in V
dv = 0;
flag = 0;
i = 1;

while i <= (T-1)
    % Spike
    if (v(i) < vth)
        dv = (-v(i) + RI) / tau_m;
        v(i+1) = v(i) + dv*dt;
        flag = 0;
    % Rest
    elseif (v(i) >= vth) 
        if(flag == 1)
            v(i) = 0;
        else
            dv = (-v(i) + 0.04)^(0.5);
            v(i+1) = v(i) + dv*dt;
            if(v(i) >= 0.025)
                flag = 1;
            end
        end
    end
    i = i+1;
end

figure;
plot(t,v*1000);
grid on; grid minor;
title('Voltage vs Time','interpreter','latex');
xlabel('Time(ms)','interpreter','latex');
ylabel('Voltage(mV)','interpreter','latex');

%% Part.2.b - theorical - in the report 

%% Part.2.c.1 - LIF model with time-varying input
clc; clear; 

dt = 0.01; % Simulation time step
Duration = 100; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms


vr = 0; % in mv, resting potential
tau_m = 20; %Arbitrary
tau_peak = 1;
Kernel = t .* exp(-t/tau_peak); 
v = vr * ones(1,T); % Vector of output voltage
R = 1; 
I = ones(1,T);
dv = 0;
flag = 0;
vth = 0.025; % threshold voltage

% exp distribution
n = 3;
r = 10^n/5+50; % rate

tt = 0;
countSpikes = 0;
ISI = []; % a vector to keep ISI values
train = zeros(1,T);

while tt < t(end)/1000
    ISI = [ISI exprnd(1/r)];
    tt = tt + round(ISI(countSpikes+1),n+1);
    if tt > t(end)/1000
        break
    end
    train(int32(tt*1000/dt)) = 1;
    countSpikes = countSpikes + 1;
end

I = conv(Kernel,train);


% Euler method for v(t)
for i = 1:(T-1)
    % Spike
    if (v(i) < vth)
        dv = (-v(i) + R*I(i)) / tau_m;
        v(i+1) = v(i) + dv*dt;
        flag = 0;
    % Rest
    elseif (v(i) >= vth) 
        if(flag == 1)
            v(i) = 0;
        else
            dv = (-v(i) + 0.04)^(0.5);
            v(i+1) = v(i) + dv*dt;
            if(v(i) >= 0.035)
                flag = 1;
            end
        end
    end
end

figure;
subplot(3,1,3);
plot(t,v);
xlabel('Time(ms)','interpreter','latex');
ylabel('Voltage(mV)','interpreter','latex');
grid on; grid minor;
title('Voltage vs Time','interpreter','latex');
subplot(3,1,2);
plot(t,I(1:t(end)/dt)); 
xlabel('Time(ms)','interpreter','latex');
ylabel('Current(mA)','interpreter','latex');
grid on; grid minor;
title('Current vs Time','interpreter','latex');
yline(0.2);
%legend('Current','estimta
subplot(3,1,1);
stem(t,train); 
xlabel('Time(ms)','interpreter','latex');
ylabel('Spike Train','interpreter','latex');
grid on; grid minor;
title('SpikeTrain vs Time','interpreter','latex');
ylim([0 1.2]);

%% Part.2.c.2 - CV contour over Nth and tau_m
clc; clear; close all;

dt = 0.01; % Simulation time step
Duration = 1000; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms

vth = 0.025; % threshold voltage
vr = 0; % in mv, resting potential
tau_m = (0.1:0.1:10); %Arbitrary
k = 1:100;
CV = zeros(length(k),length(tau_m));
tau_peak = 1;

Kernel = t .* exp(-t/tau_peak); 

for j = 1:size(CV,1) % Nth
    for jj = 1:size(CV,2) % tau
        v = vr * ones(1,T); % Vector of output voltage
        R = 1;
        I = ones(1,T);
        dv = 0;
        flag = 0;

        % exp distribution
        n = 4;
        r = 200; % rate

        tt = 0;
        countSpikes = 0;
        ISI = []; % a vector to keep ISI values
        timeSpikes = [];
        train = zeros(1,T);
        refT = 0.001;

        while tt < t(end)/1000
            ISI = [ISI exprnd(1/r)];
            tt = tt + round(ISI(countSpikes+1),5);
            if ISI(end) < refT
                ISI(end) = [];
            else
                if tt > t(end)/1000
                    break
                end
                timeSpikes = [timeSpikes tt];
                train(int32(tt*1000/dt)) = 1;
                countSpikes = countSpikes + 1;
            end
        end
        ksdown = (downsample(find(train == 1),k(j)));
        train_downS = zeros(1,T);
        train_downS(ksdown) = 1;
        timeSpikes = downsample(timeSpikes,k(j));
        I = conv(Kernel,train_downS);
        i = 1;
        % Euler method for v(t)
        while i <= (T-1)
            % Spike
            if (v(i) < vth)
                dv = (-v(i) + R*I(i)) / tau_m(jj);
                v(i+1) = v(i) + dv*dt;
                flag = 0;
            % Rest
            elseif (v(i) >= vth) 
                if(flag == 1)
                    v(i) = 0;
                    % ref period of 1ms
                    if (i+1/dt+1 < length(t))
                        v(i:i+1/dt) = 0;
                        i = i+1/dt;
                    end
                else
                    dv = (-v(i) + 0.04)^(0.5);
                    v(i+1) = v(i) + dv*dt;
                    if(v(i) >= 0.035)
                        flag = 1;
                    end
                end
            end
            i = i+1;
        end
        % cal CVs
        spkes = find(v > vth);
        spkes = (find(v(spkes-1) <= vth));
        CV(j,jj) = std(diff(spkes))/mean(diff(spkes));
    end
end

figure;
subplot(4,1,4);
plot(t,v);
xlabel('Time(ms)','interpreter','latex');
ylabel('Voltage(mV)','interpreter','latex');
grid on; grid minor;
title('Voltage vs Time','interpreter','latex');
subplot(4,1,2);
stem(t,train_downS);
xlabel('Time(ms)','interpreter','latex');
ylabel('Spike Train','interpreter','latex');
grid on; grid minor;
title('Down Sampled Spike Train vs Time','interpreter','latex');
subplot(4,1,3);
plot(t,I(1:t(end)/dt)); 
xlabel('Time(ms)','interpreter','latex');
ylabel('Current(mA)','interpreter','latex');
grid on; grid minor;
title('Current vs Time','interpreter','latex');
yline(0.2);
subplot(4,1,1);
stem(t,train); 
xlabel('Time(ms)','interpreter','latex');
ylabel('Spike Train','interpreter','latex');
grid on; grid minor;
title('SpikeTrain vs Time','interpreter','latex');
ylim([0 1.2]);


figure;
contourf(tau_m,k,CV);
set(gca,'xscale','log');
set(gca,'yscale','log');
ax = gca;
xlabel('tau(ms)','interpreter','latex');
ylabel('Nth','interpreter','latex');
title('Contour','interpreter','latex');

%% Part.2.c.3 - CV contour over EPSC properties
clc; clear; close all;

dt = 0.01; % Simulation time step
Duration = 100; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
k = 1;
vth = 0.025; % threshold voltage
vr = 0; % in mv, resting potential
tau_m = 20; %Arbitrary
tau_peak = 0.1:0.1:10;
mag = 0.1:0.1:10;
CV = zeros(length(mag),length(tau_peak));

for j = 1:size(CV,1) % magnitude of the EPSCs
    for jj = 1:size(CV,2) % tau_peak of the EPSCs
        [j jj]
        Kernel = mag(j)*t .* exp(-t/tau_peak(jj)); 
        v = vr * ones(1,T); % Vector of output voltage
        R = 1;
        I = ones(1,T);
        dv = 0;
        flag = 0;

        % exp distribution
        n = 4;
        r = 200; % rate

        tt = 0;
        countSpikes = 0;
        ISI = []; % a vector to keep ISI values
        timeSpikes = [];
        train = zeros(1,T);
        refT = 0.001;

        while tt < t(end)/1000
            ISI = [ISI exprnd(1/r)+t0];
            tt = tt + round(ISI(countSpikes+1),5);
            if ISI(end) < refT
                ISI(end) = [];
            else
            if tt > t(end)/1000
                break
            end
            timeSpikes = [timeSpikes tt];
            train(int32(tt*1000/dt)) = 1;
            countSpikes = countSpikes + 1;
            end
        end
        ksdown = (downsample(find(train == 1),k));
        train_downS = zeros(1,T);
        train_downS(ksdown) = 1;
        timeSpikes = downsample(timeSpikes,k);
        I = conv(Kernel,train_downS);
        i = 1;
        % Euler method for v(t)
        while i <= (T-1)
            % Spike
            if (v(i) < vth)
                dv = (-v(i) + R*I(i)) / tau_m;
                v(i+1) = v(i) + dv*dt;
                flag = 0;
            % Rest
            elseif (v(i) >= vth) 
                if(flag == 1)
                    v(i) = 0;
                    % ref period of 1ms
                    if (i+1/dt+1 < length(t))
                        v(i:i+1/dt) = 0;
                        i = i+1/dt;
                    end
                else
                    dv = (-v(i) + 0.04)^(0.5);
                    v(i+1) = v(i) + dv*dt;
                    if(v(i) >= 0.035)
                        flag = 1;
                    end
                end
            end
            i = i+1;
        end
        % cal CVs
        spkes = find(v > vth);
        spkes = (find(v(spkes-1) <= vth));
        CV(j,jj) = std(diff(spkes))/mean(diff(spkes));
    end
end

figure;
contourf(mag,tau_peak,CV);
set(gca,'xscale','log');
set(gca,'yscale','log');
ax = gca;
xlabel('magnitude','interpreter','latex');
ylabel('tau_peak','interpreter','latex');
title('Contour','interpreter','latex');

%% Part.2.d - LIF model with time-varing input
clc; clear; close all;

dt = 0.01; % Simulation time step
Duration = 100; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
t0 = 0.001; % refractory period

vr = 0; % in mv, resting potential
tau_m = 10; % Arbitrary
tau_peak = 1;
Kernel = t .* exp(-t/tau_peak); 
v = vr * ones(1,T); % Vector of output voltage
R = 1;
dv = 0;
flag = 0;
vth = 0.025;

vector = zeros(450,T); %for inhibitory and excitatory
I_total = zeros(1,2*T-1); % for inhibitory and excitatory
coutnerInhibitory = 0;
coutnerTot = 0;
r = 100;
percentage = 0.33
k = 5;
train_downS = zeros(450,T);
    
for j=1:450% 450 currents for inhibitory and excitatory
    ISI = [];
    timeSpikes = [];
    tt = 0;
    countSpikes = 0;
    while tt < t(end)/1000
        ISI = [ISI exprnd(1/r)+t0];
        tt = tt + round(ISI(countSpikes+1),5);
        if tt > t(end)/1000
            break
        end
        timeSpikes = [timeSpikes tt];
        vector(j,int32(tt*1000/dt)) = 1;
        countSpikes = countSpikes + 1;
    end
    ksdown = (downsample(find(vector(j,:) == 1),k));
    train_downS(j,ksdown) = 1;    
    I(j,:) = conv(Kernel,train_downS(j,:)); % for inhibitory and excitatory
    coutnerTot = coutnerTot + 1; % for inhibitory and excitatory
    flag = binornd(1,percentage); % for inhibitory and excitatory
    if(flag == 0) 
        I_total = I_total + I(j,:); % for inhibitory and excitatory
    else
        I_total = I_total - I(j,:); % for inhibitory and excitatory
        coutnerInhibitory = coutnerInhibitory + 1; % for inhibitory and excitatory
    end 
end



% Euler method for v(t)
for i = 1:(T-1)
    % Spike
    if (v(i) < vth)
        dv = (-v(i) + R*I_total(i)) / tau_m;
        v(i+1) = v(i) + dv*dt;
        flagg = 0;
    % Rest
    elseif (v(i) >= vth) 
        if(flagg == 1)
        v(i) = 0;
            % ref period of 1ms
            if (i+1/dt+1 < length(t))
                v(i:i+1/dt) = 0;
                i = i+1/dt;
            end
        else
            dv = (-v(i) + 0.04)^(0.5);
            v(i+1) = v(i) + dv*dt;
            if(v(i) >= 0.035)
                flagg = 1;
            end
        end
    end
end

% cal CVs
spkes = find(v > vth);
spkes = (find(v(spkes-1) <= vth));
if length(spkes) ~= 0
    CV = std(diff(spkes))/mean(diff(spkes))
end

subplot(2,1,2);
plot(t,v);
xlabel('Time(ms)','interpreter','latex');
ylabel('Voltage(mV)','interpreter','latex');
title('Voltage vs Time','interpreter','latex');
grid on; grid minor;
subplot(2,1,1);
plot(t,I_total(1:t(end)/dt));  %for inhibitory and excitatory
xlabel('Time(ms)','interpreter','latex');
ylabel('Current(A)','interpreter','latex');
title('Current vs Time','interpreter','latex');
grid on; grid minor;

%% Part.2.e - Coincidence detection
clc; clear; close all;
duration = 1000;

% exp distribution
r = 200; % rate
t0 = 0.001; % refractory period 
W = 10:100;
N = 0:1:5*(max(W)/1000)*r;
CV = zeros(100,length(W),length(N));

for ii=1:100
    dt = 0.01; % Simulation time step
  
    tt = 0; % on going time
    countSpikes = 0;
    ISI = []; % a vector to keep ISI values

    % paramters
    timeSpikesFinal = [];
    r = 100; % rate
    t = 0;
    countSpikes = 0;
    ISI = []; % a vector to keep ISI values
    timeSpikes = []; % a vector to keep spike times

    % a total duration of 
    while t < duration/1000
        ISI = [ISI exprnd(1/r) + t0];
        t = t + (ISI(countSpikes+1));
        timeSpikes = [timeSpikes t];
        countSpikes = countSpikes + 1;
    end
    timeSpikes(end) = [];


    for j=1:length(W)
        for i=1:length(N)
            tt = 0;
            timeSpikesFinal = [];
            while tt < duration
                Nfind = (find(timeSpikes*1000 >= tt));
                Nfind = (find(timeSpikes(Nfind)*1000 < tt+W(j)));
                tt = tt+W(j);
                if(length(Nfind) == N(i))
                    timeSpikesFinal = [timeSpikesFinal tt/1000];
                end
            end
            if ~isempty(timeSpikesFinal ~= 0) && length(timeSpikesFinal) > 1
                CV(ii,j,i) = std(diff(timeSpikesFinal))/mean(diff(timeSpikesFinal));
            end
        end
    end
end
CV = reshape(mean(CV,1),[length(W),length(N)]);
figure;
[X,Y] = meshgrid(W,N);
surface(X.',(Y.')./(length(N)-1),CV);
xlabel('Window Length(ms)','interpreter','latex');
ylabel('N/M','interpreter','latex');
title("total simulation time in seconds = " + duration/1000,'interpreter','latex');
figure;
selected = [];
for i=1:4
subplot(1,2,1);
plot(W,(CV(:,i)),'LineWidth',2)
xlabel('W','interpreter','latex');
ylabel('Mean CV over 100 trials','interpreter','latex');
title('CV vs W','interpreter','latex');
grid on; grid minor;
hold on;
    selected = [selected i];
end
legend("N = " + selected(1),"N = " + selected(2),"N = " + selected(3),"N = " + selected(4));

selected = [];
for i=10:10:40
    subplot(1,2,2);
    plot(N/length(timeSpikes),(CV(i,:)),'LineWidth',2)
    xlim([0 0.2])
    xlabel('N/M','interpreter','latex');
    ylabel('Mean CV over 100 trials','interpreter','latex');
    title('CV vs N/M','interpreter','latex');
    grid on; grid minor;
    hold on;
    selected = [selected i];
end
legend("W = " + selected(1),"W = " + selected(2),"W = " + selected(3),"W = " + selected(4));

%% Part.2.f - Coincidence detection - including inhibitories
clc; clear; close all;
duration = 1000;

% exp distribution
rExcitatory = 300; % rate of excitatories
rInhibitory = 150; %rate of inhibitories
t0 = 0.001; % refractory period 
W = 1:100;
N = 0:1:5*(max(W)/1000)*max(rExcitatory,rInhibitory);
CV = zeros(100,length(W),length(N));
diffTimes = zeros(100,length(W),length(N));
flag = 1;
for ii=1:100  
    tt = 0; % on going time

    % paramters excitatory
    countSpikesExcitatory = 0;
    ISIExcitatory = []; % a vector to keep ISI values
    timeSpikesFinal = [];
    t = 0;
    countSpikesExcitatory = 0;
    ISIExcitatory = []; % a vector to keep ISI values
    timeSpikesExcitatory = []; % a vector to keep spike times

    % a total duration of 100s
    while t < duration/1000
        ISIExcitatory = [ISIExcitatory exprnd(1/rExcitatory) + t0];
        t = t + (ISIExcitatory(countSpikesExcitatory+1));
        timeSpikesExcitatory = [timeSpikesExcitatory t];
        countSpikesExcitatory = countSpikesExcitatory + 1;
    end
    timeSpikesExcitatory(end) = [];


    % paramters inhibitory
    countSpikesInhibitory = 0;
    ISIInhibitory = []; % a vector to keep ISI values
    t = 0;
    countSpikesInhibitory = 0;
    ISIInhibitory = []; % a vector to keep ISI values
    timeSpikesInhibitory = []; % a vector to keep spike times

    % a total duration of 100s
    while t < duration/1000
        ISIInhibitory = [ISIInhibitory exprnd(1/rInhibitory) + t0];
        t = t + (ISIInhibitory(countSpikesInhibitory+1));
        timeSpikesInhibitory = [timeSpikesInhibitory t];
        countSpikesInhibitory = countSpikesInhibitory + 1;
    end
    timeSpikesInhibitory(end) = [];   
    
    
    
    for j=1:length(W)
        for i=1:length(N)
            tt = 0;
            timeSpikesFinal = [];
            while tt < duration
                NfindExcitatory = (find(timeSpikesExcitatory*1000 >= tt));
                NfindExcitatory = (find(timeSpikesExcitatory(NfindExcitatory)*1000 < tt+W(j)));
                NfindInhibitory = (find(timeSpikesInhibitory*1000 >= tt));
                NfindInhibitory = (find(timeSpikesInhibitory(NfindInhibitory)*1000 < tt+W(j)));
                tt = tt+W(j);
                if((length(NfindExcitatory)-length(NfindInhibitory)) >= N(i))
                    timeSpikesFinal = [timeSpikesFinal tt/1000];
                end
            end
            if ~isempty(timeSpikesFinal ~= 0) && length(timeSpikesFinal) > 1
                CV(ii,j,i) = std(diff(timeSpikesFinal))/mean(diff(timeSpikesFinal));
            end
        end
    end
end
CV = reshape(mean(CV,1),[length(W),length(N)]);
figure;
[X,Y] = meshgrid(W,N);
surface(X.',(Y.')./(length(N)-1),CV);
xlabel('Window Length(ms)','interpreter','latex');
ylabel('N/M','interpreter','latex');
title("total simulation time in seconds = " + duration/1000,'interpreter','latex');

figure;
selected = [];
for i=1:4
subplot(1,2,1);
plot(W,(CV(:,i)),'LineWidth',2)
xlabel('W','interpreter','latex');
ylabel('CV','interpreter','latex');
title('CV over 100 trials','interpreter','latex');
grid on; grid minor;
hold on;
selected = [selected i];
end
legend("N = " + selected(1),"N = " + selected(2),"N = " + selected(3),"N = " + selected(4));

selected = [];
for i=10:10:40
    subplot(1,2,2);
    plot(N,(CV(i,:)),'LineWidth',2)
    xlim([0 20]);
    xlabel('Nnet','interpreter','latex');
    ylabel('CV','interpreter','latex');
    title('CV over 100 trials','interpreter','latex');
    grid on; grid minor;
    hold on;
    selected = [selected i];
end
legend("W = " + selected(1),"W = " + selected(2),"W = " + selected(3),"W = " + selected(4));
