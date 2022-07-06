function [rgb_vid, patches] = import_process_video()
    % load the video
    v = VideoReader('Data\BIRD.avi');
    v_frames_num = v.Duration*v.FrameRate;
    
    % export the video frames
    channels_num = 1; % will be converted to grayscale frames
    frame = zeros(v.Height,v.Width,channels_num,v_frames_num);
    frame_rgb = zeros(v.Height,v.Width,3,v_frames_num);
    
    im_size = size(frame,1);
    
    k = 1;
    while hasFrame(v)
        frame_rgb(:,:,:,k) = readFrame(v);
        frame(:,:,:,k) = im2gray(uint8(frame_rgb(:,:,:,k))); 
        k = k+1;
    end
    
    frame = squeeze(frame);
    
    % resize images
    frame = frame(1:im_size,1:im_size,:);
    rgb_vid = frame_rgb(1:im_size,1:im_size,:,:);
    rgb_vid = uint8(rgb_vid);
    
    % remove white noise from frames
    make_your_own_images(frame,'Bird_Frames_dewhited');
    frame = load('Bird_Frames_dewhited.mat').IMAGES;
    

    v_frames_num = size(frame,3);
    patch_size = 10;
    patches = frame;
    
end