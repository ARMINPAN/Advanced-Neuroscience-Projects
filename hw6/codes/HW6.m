%% HW6 - Reinforcement Learning
%%% Armin Panjehpour - 98101288

%% Part.0 - initialization of the map
clc; clear; close all;

target_loc = [7, 7];
cat_loc = [11, 3];


cat = imread('cat.png');
mouse = imread('mouse.png');
mouseCheese = imread('mouseCheese.png');
cheese = imread('cheese.jpg');
catHitting = imread('catHitting.jpg');

% initial map
figure('units','normalized','outerposition',[0 0 1 1])

% initial mouse loc
mouse_loc = randi(14,[1 2]);

updateMap(cat_loc, target_loc, mouse_loc, cat, mouse, cheese)

title('Initial Map of the Game', 'interpreter', 'latex')

%% Part.0 - state values and action probabilites initialization
clc; close all;

% values of states
value_map = zeros(15,15);


% dircetion indicies
%%%      1: left,     2: up,     3: right,     4: down
probabilities_map = zeros(15,15,4);

% border probabilites
probabilities_map(2:14,1,2:4) = 1/3; % left border
probabilities_map(1,2:14,[1, 3, 4]) = 1/3; % up border
probabilities_map(2:14,15,[1, 2, 4]) = 1/3; % right border
probabilities_map(15,2:14,1:3) = 1/3; % bottom border

% vertices
probabilities_map(1,1,[3, 4]) = 1/2; % up left vertex
probabilities_map(15,1,2:3) = 1/2; % bottom left vertex
probabilities_map(1,15,[1, 4]) = 1/2; % up right vertex
probabilities_map(15,15,1:2) = 1/2; % bottom right vertex

% non border, non vertices probabilites
probabilities_map(2:14,2:14,:) = 1/4;


% policy probabilites 
policy = probabilities_map;

% reward map
reward_map = value_map;

% intial gradient contour
[px, py] = gradient(value_map);

% initial plots
% map
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,3,[1, 2, 4, 5]);
updateMap(cat_loc, target_loc, mouse_loc, cat, mouse, cheese)
title('Initial Map of the Game', 'interpreter', 'latex')

% state values
subplot(2,3,3);
x = 0:14; y = 0:14;
imagesc(x,y,value_map);
colorbar
colormap jet
title('Initial State Values','interpreter','latex')


% gradient contours
subplot(2,3,6);
quiver(x,y,px,py)
set(gca,'YDir','reverse')
title('Initial Gradient Contour','interpreter','latex')
%% Part.1 - plot path before and after training
clc; close all;

forgetting_factor = 1;
LR = 1;
discount_term = 1;
trial_num = 400;
counts = zeros(1,trial_num);

% values of states
value_map = zeros(15,15);
policy = probabilities_map;
frames = [];

% path
trial1_path = [];
trialFinal_path = [];


for i = 1:trial_num
    i
    % initial mouse loc
    mouse_loc = randi(14,[1 2]);
    currentState = mouse_loc;
    value_map = forgetting_factor*value_map;
    count = 0;
    while true
        if(i == 1)
            trial1_path = [trial1_path; currentState];
        elseif(i == trial_num)
            trialFinal_path = [trialFinal_path; currentState];
        end
        
        % reached target
        if(currentState(1) == target_loc(1) && currentState(2) == target_loc(2))
            value_map(currentState(1)+1,currentState(2)+1) = 1;
            break;
        % reached cat  
        elseif(currentState(1) == cat_loc(1) && currentState(2) == cat_loc(2))
            value_map(currentState(1)+1,currentState(2)+1) = -1;
            break;
        end
        
        % update
        reward_currentState = reward_map(currentState(1)+1,currentState(2)+1);
        value_currentState = value_map(currentState(1)+1,currentState(2)+1);
        targetSigma = calSigma(currentState+1, value_map, probabilities_map, policy);
        delta = reward_currentState + (discount_term)*targetSigma - value_currentState;
        
        % update state values and probabilites
        value_map(currentState(1)+1,currentState(2)+1) = ...
            value_currentState + LR*delta;
        policy(currentState(1)+1,currentState(2)+1,:) = ...
            policyUpdate(currentState+1, value_map, policy);
        prevState = currentState;
        currentState = nextStateCal(currentState, policy);
        count = count + 1;
    end
    counts(i) = count;
end


% plot paths
figure;
% trial 1 path
subplot(1,2,1);
set(gca,'YDir','reverse')
xlim([0 15]);
ylim([0 15]);
title("path of first trial, step numbers: " + length(trial1_path),'interpreter','latex')
grid on; grid minor;
for i = 1:length(trial1_path)-1
    hold on;
    h = line([trial1_path(i,2), trial1_path(i+1,2)], [trial1_path(i,1), trial1_path(i+1,1)]);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on;
    if(i == 1)
        scatter(trial1_path(i,2),trial1_path(i,1),30,'filled')
    elseif(i == length(trial1_path)-1)
        scatter(trial1_path(i+1,2),trial1_path(i+1,1),30,'filled')
    end
end
legend('Starting Point','Ending Point')

% trial final path
subplot(1,2,2);
set(gca,'YDir','reverse')
grid on; grid minor;
xlim([0 15]);
ylim([0 15]);
title("path of final trial, step numbers: " + length(trialFinal_path),'interpreter','latex')
for i = 1:length(trialFinal_path)-1
    hold on;
    h = line([trialFinal_path(i,2), trialFinal_path(i+1,2)],...
        [trialFinal_path(i,1), trialFinal_path(i+1,1)]);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on;
    if(i == 1)
        scatter(trialFinal_path(i,2),trialFinal_path(i,1),30,'filled')
    elseif(i == length(trialFinal_path)-1)
        scatter(trialFinal_path(i+1,2),trialFinal_path(i+1,1),30,'filled')
    end
end
legend('Starting Point','Ending Point')



%% Part.1.1&2 - model based learning with live figures
clc; close all;
% uncomment the target codes if you want live plots - let it be if you want
% just the final values

forgetting_factor = 1;
LR = 1;
discount_term = 1;
trial_num = 400;
counts = zeros(1,trial_num);

% values of states
value_map = zeros(15,15);
policy = probabilities_map;
frames = [];


% comment/uncomment if using/not_using plots
 figure('units','normalized','outerposition',[0 0 1 1])

for i = 1:trial_num
    i
    % initial mouse loc
    mouse_loc = randi(14,[1 2]);
    currentState = mouse_loc;
    value_map = forgetting_factor*value_map;
    count = 0;
    while true
        % update plot - comment/uncomment if using/not_using plots
        % plot the current state - each 20 steps
        if(mod(count,20) == 0)
            subplot(2,3,[1, 2, 4, 5]);
            cla
            if(currentState(1) == target_loc(1) && currentState(2) == target_loc(2))
                updateMap(cat_loc, target_loc, currentState, cat, mouseCheese, cheese)
            else
                updateMap(cat_loc, target_loc, currentState, cat, mouse, cheese)
            end
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
            frames = [frames getframe(gcf)];
            drawnow
        end
        
        % reached target
        if(currentState(1) == target_loc(1) && currentState(2) == target_loc(2))
            value_map(currentState(1)+1,currentState(2)+1) = 1;
            % update plot - comment/uncomment if using/not_using plots
            subplot(2,3,[1, 2, 4, 5]);
            cla
            updateMap(cat_loc, target_loc, currentState, cat, mouseCheese, cheese)    
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
            frames = [frames getframe(gcf)];
            pause(0.4)
            break;
        % reached cat  
        elseif(currentState(1) == cat_loc(1) && currentState(2) == cat_loc(2))
            value_map(currentState(1)+1,currentState(2)+1) = -1;
            subplot(2,3,[1, 2, 4, 5]);
            cla
            updateMap(cat_loc, target_loc, currentState, cat, catHitting, cheese)
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
            frames = [frames getframe(gcf)];
            pause(0.4)
            break;
        end
        
        
        % update
        reward_currentState = reward_map(currentState(1)+1,currentState(2)+1);
        value_currentState = value_map(currentState(1)+1,currentState(2)+1);
        targetSigma = calSigma(currentState+1, value_map, probabilities_map, policy);
        delta = reward_currentState + (discount_term)*targetSigma - value_currentState;
        
        % update state values and probabilites
        value_map(currentState(1)+1,currentState(2)+1) = ...
            value_currentState + LR*delta;
        policy(currentState(1)+1,currentState(2)+1,:) = ...
            policyUpdate(currentState+1, value_map, policy);
        currentState = nextStateCal(currentState, policy);
        count = count + 1;
    end
    counts(i) = count;
end


% uncomment if you want output demo 
% writerObj = VideoWriter("mouseOneNextMoveFutureSee");
% writerObj.FrameRate = 20;
% writerObj.Quality = 100;
% 
% % open the video writer
% open(writerObj);
% % write the frames to the video
% for i=1:length(frames)
% 	% convert the image to a frame
%     frame = frames(i) ;
%     writeVideo(writerObj,frame);
% end
% % close the writer o bject
% close(writerObj)


%% final plots of part.1&2

figure
% state values
subplot(1,2,1);
x = 0:14; y = 0:14;
imagesc(x,y,value_map);
axis square
colorbar
colormap jet
caxis([-1 1])
title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
    ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
% gradient contours
subplot(1,2,2);
[px, py] = gradient(value_map);
quiver(x,y,px,py)
axis square

set(gca,'YDir','reverse')
title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
    ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
xlim([0 15]); ylim([0 15])

%% Part.3 - learning rate and discount rate effect on convergence
clc; close all;

% run each trial 100 times, and average on the count, take the mean of the
% 10 last trials as the number of iterations needed for that specific LR
% and discount rate

LR = 0.2:0.2:1;
discount_rate = 0.2:0.2:1;
trial_num = 400;
iterations = 100;
counts = zeros(length(LR),length(discount_rate),trial_num,iterations);
count = 0;


for i = 1:length(LR)
    for j = 1:length(discount_rate)
        % reset
        value_map = zeros(15,15);
        policy = probabilities_map;
        for k = 1:trial_num
            % initial mouse loc
            for kk = 1:iterations
                mouse_loc = randi(14,[1 2]);
                currentState = mouse_loc;
                [i j k kk]
                count = 0;
                while true
                    % reached target
                    if(currentState(1) == target_loc(1) && currentState(2) == target_loc(2))
                        if(kk == iterations)
                            value_map(currentState(1)+1,currentState(2)+1) = 1;
                        end
                        break;
                    % reached cat  
                    elseif(currentState(1) == cat_loc(1) && currentState(2) == cat_loc(2))
                        if(kk == iterations)
                            value_map(currentState(1)+1,currentState(2)+1) = -1;
                        end
                        break;
                    end

                    if(kk == iterations)
                        % update
                        reward_currentState = reward_map(currentState(1)+1,currentState(2)+1);
                        value_currentState = value_map(currentState(1)+1,currentState(2)+1);
                        targetSigma = calSigma(currentState+1, value_map, probabilities_map, policy);
                        delta = reward_currentState + (discount_rate(j))*targetSigma - value_currentState;

                        % update state values and probabilites
                        value_map(currentState(1)+1,currentState(2)+1) = ...
                            value_currentState + LR(i)*delta;
                        policy(currentState(1)+1,currentState(2)+1,:) = ...
                            policyUpdate(currentState+1, value_map, policy);
                    end
                    currentState = nextStateCal(currentState, policy);
                    count = count + 1;
                end
                counts(i,j,k,kk) = count;
            end
        end
    end
end
 
meanCountsIteration = (mean(counts,4));
meanCountsTotal = mean(meanCountsIteration(:,:,end-50:end),3);

figure;
imagesc(discount_rate,LR,log10(meanCountsTotal));
set(gca,'YDir','normal')
axis square
colorbar 
colormap hot
title('Log of Reaching Steps vs LR and Lambda','interpreter','latex')
ylabel('LR','interpreter','latex')
xlabel('Discount_rate','interpreter','latex')

figure
subplot(1,2,1)
imagesc(discount_rate,LR(end),log10(meanCountsTotal(end,:)));
set(gca,'YDir','normal')
axis square
colorbar 
colormap hot
title('Log of Reaching Steps vs LR and Lambda','interpreter','latex')
ylabel('LR','interpreter','latex')
xlabel('Discount_rate','interpreter','latex')

subplot(1,2,2)
imagesc(discount_rate(end),LR,log10(meanCountsTotal(:,end)));
set(gca,'YDir','normal')
axis square
colorbar 
colormap hot
title('Log of Reaching Steps vs LR and Lambda','interpreter','latex')
ylabel('LR','interpreter','latex')
xlabel('Discount_rate','interpreter','latex')

%% part.arbitary - count vs trial - for part.1 
clc; close all;

mean_value = zeros(1,length(counts));

for i = 1:length(counts)
    mean_value(i) = mean(counts(1:i));
end

figure;
stem(1:trial_num,counts)
hold on;
plot(1:trial_num,mean_value,'LineWidth', 2);
title("count vs trial" + ", LR: " + LR +...
    ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex');
xlabel('trial num','interpreter','latex');
ylabel('step count','interpreter','latex');

legend('counts','mean counts(t)')



%% Part.4 - two targets 
clc; close all;

forgetting_factor = 1;
LR = 1;
discount_term = 1;
trial_num = 400;
counts = zeros(1,trial_num);
target_loc2 = [3, 11];

% values of states
value_map = zeros(15,15);
policy = probabilities_map;

figure('units','normalized','outerposition',[0 0 1 1])
frames = [];
for i = 1:trial_num
    i
    % initial mouse loc
    mouse_loc = randi(14,[1 2]);
    currentState = mouse_loc;
    value_map = forgetting_factor*value_map;
    count = 0;
    while true
        % plot the current state - each 5 steps
        if(mod(count,20) == 0)
            subplot(2,3,[1, 2, 4, 5]);
            cla
            if((currentState(1) == target_loc(1) && currentState(2) == target_loc(2)) || ...
                    (currentState(1) == target_loc2(1) && currentState(2) == target_loc2(2)))
                updateMapTwoTarget(cat_loc, target_loc, currentState, cat, mouseCheese, cheese, ...
                    target_loc2)
            else
                updateMapTwoTarget(cat_loc, target_loc, currentState, cat, mouse, cheese, ...
                    target_loc2)
            end
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
            frames = [frames getframe(gcf)];
            drawnow
        end
        
        % update
        reward_currentState = reward_map(currentState(1)+1,currentState(2)+1);
        value_currentState = value_map(currentState(1)+1,currentState(2)+1);
        targetSigma = calSigma(currentState+1, value_map, probabilities_map, policy);
        delta = reward_currentState + (discount_term)*targetSigma - value_currentState;
        % reached target
        if((currentState(1) == target_loc(1) && currentState(2) == target_loc(2)) || ...
                    (currentState(1) == target_loc2(1) && currentState(2) == target_loc2(2)))
            if((currentState(1) == target_loc(1) && currentState(2) == target_loc(2)))
                value_map(target_loc(1)+1,target_loc(2)+1) = 0.4;
            elseif(currentState(1) == target_loc2(1) && currentState(2) == target_loc2(2))
                value_map(target_loc2(1)+1,target_loc2(2)+1) = 1;
            end
            subplot(2,3,[1, 2, 4, 5]);
            cla
            if((currentState(1) == target_loc(1) && currentState(2) == target_loc(2)) || ...
                    (currentState(1) == target_loc2(1) && currentState(2) == target_loc2(2)))
                updateMapTwoTarget(cat_loc, target_loc, currentState, cat, mouseCheese, cheese, ...
                    target_loc2)
            elseif(currentState(1) == cat_loc(1) && currentState(2) == cat_loc(2))
                updateMapTwoTarget(cat_loc, target_loc, currentState, cat, catHitting, cheese, ...
                    target_loc2)
            else
                updateMapTwoTarget(cat_loc, target_loc, currentState, cat, mouse, cheese, ...
                    target_loc2)
            end
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
            frames = [frames getframe(gcf)];
           pause(0.4)
            break;
        % reached cat  
        elseif(currentState(1) == cat_loc(1) && currentState(2) == cat_loc(2))
            value_map(currentState(1)+1,currentState(2)+1) = -1;
            subplot(2,3,[1, 2, 4, 5]);
            cla
            if(((currentState(1) == target_loc(1) && currentState(2) == target_loc(2)) || ...
                    (currentState(1) == target_loc2(1) && currentState(2) == target_loc2(2))))
                updateMapTwoTarget(cat_loc, target_loc, currentState, cat, mouseCheese, cheese, ...
                    target_loc2)
            elseif(currentState(1) == cat_loc(1) && currentState(2) == cat_loc(2))
                updateMapTwoTarget(cat_loc, target_loc, currentState, cat, catHitting, cheese, ...
                    target_loc2)
            else
                updateMapTwoTarget(cat_loc, target_loc, currentState, cat, mouse, cheese, ...
                    target_loc2)
            end
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            % gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
            ", Forgetting Factor: " + forgetting_factor, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
           frames = [frames getframe(gcf)];
            break;
        end
        
        % update state values and probabilites
        value_map(currentState(1)+1,currentState(2)+1) = ...
            value_currentState + LR*delta;
        policy(currentState(1)+1,currentState(2)+1,:) = ...
            policyUpdate(currentState+1, value_map, policy);
        currentState = nextStateCal(currentState, policy);
        count = count + 1;
    end
    counts(i) = count;
end


writerObj = VideoWriter("mouseOneNextMoveModelBasedTwoTarget");
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

%% Part.5 - TD Lambda Learning Implementation

clc; close all;
% uncomment the target codes if you want live plots - let it be if you want
% just the final values

forgetting_factor = 1;
discount_term = 0.3;
trial_num = 100;
counts = zeros(1,trial_num);
LR = 1;
target_loc = [7, 7];
cat_loc = [11, 3];

% values of states
value_map = zeros(15,15);
policy = probabilities_map;
frames = [];
trial_path = [];

% comment/uncomment if using/not_using plots
figure('units','normalized','outerposition',[0 0 1 1])

for i = 1:trial_num
    % initial mouse loc
    mouse_loc = randi(14,[1 2]);
    currentState = mouse_loc;
    value_map = forgetting_factor*value_map;
    count = 0;
    trial_path = [mouse_loc];
    while true
        % update plot - comment/uncomment if using/not_using plots
        % plot the current state - each 20 steps
        if(mod(count,20) == 0)
            subplot(2,3,[1, 2, 4, 5]);
            cla
            if(currentState(1) == target_loc(1) && currentState(2) == target_loc(2))
                updateMap(cat_loc, target_loc, currentState, cat, mouseCheese, cheese)
            else
                updateMap(cat_loc, target_loc, currentState, cat, mouse, cheese)
            end
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
            frames = [frames getframe(gcf)];
            drawnow
        end
        value_prevState = value_map(currentState(1)+1,currentState(2)+1);   
        % reached target
        if(currentState(1) == target_loc(1) && currentState(2) == target_loc(2))
            value_map(currentState(1)+1,currentState(2)+1) = 1;
            value_currentState = value_map(currentState(1)+1,currentState(2)+1);
            delta = value_currentState - value_prevState;

            % update
            

            for j = size(trial_path,1)-1:-1:1
                % update state values and probabilites
                value_currentStateInfor = value_map(trial_path(j,1)+1,trial_path(j,2)+1);
                value_map(trial_path(j,1)+1,trial_path(j,2)+1) = ...
                    value_currentStateInfor + (discount_term^(size(trial_path,1)-j))*delta;
                policy(trial_path(j,1)+1,trial_path(j,2)+1,:) = ...
                    policyUpdate(trial_path(j,:)+1, value_map, policy);
            end
            % update plot - comment/uncomment if using/not_using plots
            subplot(2,3,[1, 2, 4, 5]);
            cla
            updateMap(cat_loc, target_loc, currentState, cat, mouseCheese, cheese)
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            % gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
            frames = [frames getframe(gcf)];
            pause(0.4)
            break;
        % reached cat  
        elseif(currentState(1) == cat_loc(1) && currentState(2) == cat_loc(2))
            value_map(currentState(1)+1,currentState(2)+1) = -1;
            value_currentState = value_map(currentState(1)+1,currentState(2)+1);
            delta = value_currentState - value_prevState;

            % update
            

            for j = size(trial_path,1)-1:-1:1
                % update state values and probabilites
                value_currentStateInfor = value_map(trial_path(j,1)+1,trial_path(j,2)+1);
                value_map(trial_path(j,1)+1,trial_path(j,2)+1) = ...
                    value_currentStateInfor + (discount_term^(size(trial_path,1)-j))*delta;
                policy(trial_path(j,1)+1,trial_path(j,2)+1,:) = ...
                    policyUpdate(trial_path(j,:)+1, value_map, policy);
            end
            subplot(2,3,[1, 2, 4, 5]);
            cla
            updateMap(cat_loc, target_loc, currentState, cat, catHitting, cheese)
            title("Map of the Game, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            % state values
            subplot(2,3,6);
            x = 0:14; y = 0:14;
            imagesc(x,y,value_map);
            colorbar
            colormap jet
            caxis([-1 1])
            title("State Values, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            % gradient contours
            subplot(2,3,3);
            [px, py] = gradient(value_map);
            quiver(x,y,px,py)
            set(gca,'YDir','reverse')
            title("Gradient Contour, Trial: " + i + ", Step: " + count + ", LR: " + LR +...
                ", Discount Term: " + discount_term, 'interpreter', 'latex')
            xlim([0 15]); ylim([0 15])
            frames = [frames getframe(gcf)];
            pause(0.4)
            break;
        end
        
                    
        
        currentState = nextStateCal(currentState, policy);
        value_currentState = value_map(currentState(1)+1,currentState(2)+1);
        delta = value_currentState - value_prevState;
        
        % update
        trial_path = [trial_path; currentState];
        
        for j = size(trial_path,1)-1:-1:1
            % update state values and probabilites
            value_currentStateInfor = value_map(trial_path(j,1)+1,trial_path(j,2)+1);
            value_map(trial_path(j,1)+1,trial_path(j,2)+1) = ...
                value_currentStateInfor + (discount_term^(size(trial_path,1)-j))*delta;
            policy(trial_path(j,1)+1,trial_path(j,2)+1,:) = ...
                policyUpdate(trial_path(j,:)+1, value_map, policy);
        end
        
        

        count = count + 1;
    end
    counts(i) = count;
end

writerObj = VideoWriter("lambdaLearning");
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
