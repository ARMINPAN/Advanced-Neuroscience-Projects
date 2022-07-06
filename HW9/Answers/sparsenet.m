% sparsenet.m - simulates the sparse coding algorithm
% 
% Before running you must first define A and load IMAGES.
% See the README file for further instructions.

clc; clear; close all;

A = rand(576,64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));

%%%  natural images
% IMAGES = load('IMAGES.mat');
% IMAGES = IMAGES.IMAGES;
% IMAGES = IMAGES(:,:,randperm(size(IMAGES,3),10)); % randomly select 10 images

%%% yale images
% IMAGES = load('All_YALE_dewhited');
% IMAGES = IMAGES.IMAGES;
% IMAGES = IMAGES(:,:,randperm(size(IMAGES,3),10)); % randomly select 10 images

%%% MNIST images
% IMAGES = load('All_MNIST_dewhited');
% IMAGES = IMAGES.IMAGES;
% IMAGES = IMAGES(:,:,randperm(size(IMAGES,3),10)); % randomly select 10 images

%%% Caltech101 images
% IMAGES = load('All_Caltech_dewhited');
% IMAGES = IMAGES.IMAGES;
% IMAGES = IMAGES(:,:,randperm(size(IMAGES,3),10)); % randomly select 10 images

%%% BIRD Video Frames
[rgb_vid, video_patches] = import_process_video();

% cal sparse functions for the first patch
first_video_patch = video_patches(:,:,1:10);
IMAGES = first_video_patch;

% imshow selected images
figure;
for i = 1:size(IMAGES,3)
    subplot(2,5,i);
    imshow(IMAGES(:,:,i));
    title("IMAGE " + i,'interpreter','latex');
end

num_trials = 10000;
batch_size = 100;

num_images=size(IMAGES,3);
image_size=size(IMAGES,1);
BUFF = 4;

[L M] = size(A);
sz = sqrt(L);

eta = 1.0;
noise_var = 0.01;
beta = 2.2;
sigma =0.316;
tol =.01;

VAR_GOAL = 0.1;
S_var = VAR_GOAL*ones(M,1);
var_eta = .001;
alpha = .02;
gain = sqrt(sum(A.*A));

X = zeros(L,batch_size);

display_every = 10;

h = display_network(A,S_var);

% main loop
for t = 1:num_trials

    % choose an image for this batch
    i = ceil(num_images*rand);
    this_image = IMAGES(:,:,i);
    if(length(find(isnan(this_image) == 1)) == 0)
        % extract subimages at random from this image to make data vector X
        for i = 1:batch_size
            r = BUFF+ceil((image_size-sz-2*BUFF)*rand);
            c = BUFF+ceil((image_size-sz-2*BUFF)*rand);
            X(:,i) = reshape(this_image(r:r+sz-1,c:c+sz-1),L,1);
        end

        % calculate coefficients for these data via conjugate gradient routine
        S = cgf_fitS(A,X,noise_var,beta,sigma,tol);

        % calculate residual error
        E = X-A*S;

        % update bases
        dA = zeros(L,M);

        for i = 1:batch_size
            dA = dA + E(:,i)*S(:,i)';
        end

        dA = dA/batch_size;
        A = A + eta*dA;

        % normalize bases to match desired output variance
        for i=1:batch_size
            S_var = (1-var_eta)*S_var + var_eta*S(:,i).*S(:,i);
        end
        gain = gain .* ((S_var/VAR_GOAL).^alpha);
        normA = sqrt(sum(A.*A));
        for i = 1:M
            A(:,i) = gain(i)*A(:,i)/normA(i);
        end

        % display
        if (mod(t,display_every) == 0)
            display_network(A,S_var,h);
        end
    end
end

%% for bird video part
% cal coefficients for other patches using bases functions found from the
% first patch
clc; close all;


S_BIRD = zeros(size(video_patches,3)-10+1,size(S,1),size(S,2));
S_BIRD(1,:,:) = S;

beta = 2.2;
sigma = 0.316;
tol = .01;

for kk = 11:size(video_patches,3)
    % select kk'th patch
    selected_video_patch = video_patches(:,:,kk);
    IMAGES = selected_video_patch;

    num_images = size(IMAGES,3);
    image_size = size(IMAGES,1);

    this_image = IMAGES;
    X = zeros(size(A,1),(size(this_image,1)/sqrt(size(A,1)))^2);
    for i = 1:size(X,2)
        X(:,i) = reshape(this_image((int32((i*sqrt(size(A,1)))/image_size)+1):...
            (int32((i*sqrt(size(A,1)))/image_size)+24),...
            mod(i*sqrt(size(A,1)),image_size)+1:...
            mod(i*sqrt(size(A,1)),image_size)+24),...
            [sqrt(size(A,1))^2 1]);
    end

    % calculate coefficients for these data via conjugate gradient routine
    S = cgf_fitS(A,X,noise_var,beta,sigma,tol);
    
    S_BIRD(kk-9,:,:) = S;
end

frames = [];

figure('Units','normalized','Position',[0 0 1 1])
for i = 1:size(S_BIRD,1)
    subplot(1,2,1);
    axis square
    pcolor(1:144,1:64,squeeze(S_BIRD(i,:,:)));
    xlabel('window num','interpreter','latex');
    ylabel('basis function num','interpreter','latex');
    colorbar
    caxis([min(S_BIRD(:)) max(S_BIRD(:))])
    subplot(1,2,2);
    imshow((rgb_vid(:,:,:,i+8)));
    frames = [frames getframe(gcf)];
    drawnow
end

writerObj = VideoWriter("heatmap");
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

frames = [];

figure('Units','normalized','Position',[0 0 1 1])
selected_window = 60;
for i = 1:size(S_BIRD,1)
    stem(1:64,S_BIRD(i,:,selected_window))
    hold on;
    plot(1:64,S_BIRD(i,:,selected_window),'LineWidth',1.5,'Color','r')
    ylim([-2 2])
    xlim([0 64])
    hold off
    xlabel('basis function num','interpreter','latex');
    ylabel('basis coeff val','interpreter','latex');
    grid on; grid minor;
    frames = [frames getframe(gcf)];
    drawnow
end

writerObj = VideoWriter("coeff_vs_basisNum");
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