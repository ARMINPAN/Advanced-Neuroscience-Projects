
% If you want to use some other images, there are a number of
% preprocessing steps you need to consider beforehand.  First, you
% should make sure all images have approximately the same overall
% contrast.  One way of doing this is to normalize each image so that
% the variance of the pixels is the same (i.e., 1).  Then you will need
% to prewhiten the images.  For a full explanation of whitening see
% 
%   Olshausen BA, Field DJ (1997)  Sparse Coding with an Overcomplete
%   Basis Set: A Strategy Employed by V1?  Vision Research, 37: 3311-3325. 
% 
% Basically, to whiten an image of size NxN, you multiply by the filter
% f*exp(-(f/f_0)^4) in the frequency domain, where f_0=0.4*N (f_0 is the
% cutoff frequency of a lowpass filter that is combined with the
% whitening filter).  Once you have preprocessed a number of images this
% way, all the same size, then you should combine them into one big N^2
% x M array, where M is the number of images.  Then rescale this array
% so that the average image variance is 0.1 (the parameters in sparsenet
% are set assuming that the data variance is 0.1).  Name this array
% IMAGES, save it to a file for future use, and you should be off and
% running.  The following Matlab commands should do this:


function make_your_own_images(input_images,name)
    image_size = size(input_images,1);
    num_images = size(input_images,3);
    N = image_size;
    M = num_images;

    [fx fy] = meshgrid(-N/2:N/2-1,-N/2:N/2-1);
    rho = sqrt(fx.*fx+fy.*fy);
    f_0 = 0.4*N;
    filt = rho.*exp(-(rho/f_0).^4);

    for i = 1:M
        image = input_images(:,:,i);  % you will need to provide get_image
        If = fft2(image);
        imagew = real(ifft2(If.*fftshift(filt)));
        IMAGES(:,i) = reshape(imagew,N^2,1);
    end
    
    IMAGES = sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));
    IMAGES = reshape(IMAGES,[image_size image_size num_images]);
    IMAGES = 0.1./var(IMAGES,0,[1 2]).*IMAGES; % set var to var goal = 0.1
    save(name,'IMAGES')
end
