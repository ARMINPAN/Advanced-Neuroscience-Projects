% a part of code for loading all images of MNIST
clc; close all; clear;

files = dir(fullfile('MNIST/MNIST - training/*.jpg')); 
[filenames{1:size(files,1)}] = deal(files.name);

im_size = 28;
im_num = 60000;
all_images = zeros(im_size,im_size,im_num);

% load all training images
for i = 1:length(filenames)
    i
    all_images(:,:,i) = imread(filenames{i});
end

% shuffle the images
shuffled_nums = randperm(size(all_images,3),size(all_images,3));
all_images = all_images(:,:,shuffled_nums);

% rmv white noise from images and save the final result
all_images = zscore(all_images,0,[1 2]);
make_your_own_images(all_images,'All_MNIST_dewhited');

