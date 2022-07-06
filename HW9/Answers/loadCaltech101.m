% a part of code for loading all images of all subjects in The Extended Yale Face Database B
clc; close all; clear;

files = dir(fullfile('Caltech101/Caltech101/')); 
[filenames{1:size(files,1)}] = deal(files.name);

image_num = 9146;
image_size = 200;
SampleIm = imread('Caltech101/Caltech101/000/1.jpg');

AllImagesCal_resized = [];



k = 1;
for i = 3:length(files)
    name = filenames{i};
    foundedImages =...
        dir(fullfile(convertStringsToChars("Caltech101/Caltech101/"+(name + "/*.jpg"))));
    for j = 1:length(foundedImages)
        this_image = im2gray(imread("Caltech101/Caltech101/"+name + "/" + ...
            foundedImages(j).name));
        % resize the images
        AllImagesCal_resized(:,:,k) = imresize(this_image,[image_size image_size]);
        k = k+1
    end
end

% shuffle the images
shuffled_nums = randperm(size(AllImagesCal_resized,3),size(AllImagesCal_resized,3));
AllImagesCal_resized = AllImagesCal_resized(:,:,shuffled_nums);

% rmv white noise from images and save the final result
AllImagesCal_resized = zscore(AllImagesCal_resized,0,[1 2]);
% since the dataset is so big, we just pass half of the images for saving
% in the final .mat file (no worries because of shuffling done)
make_your_own_images(AllImagesCal_resized(:,:,1:int32(end/2)),'All_Caltech_dewhited.mat');
