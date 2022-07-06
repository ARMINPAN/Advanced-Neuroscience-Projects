% a part of code for loading all images of all subjects in The Extended Yale Face Database B
clc; close all; clear;

files = dir(fullfile('CroppedYale/')); 
[filenames{1:size(files,1)}] = deal(files.name);

subjects_num = 28;
image_num_eachSub = 65; % first one is ambient file which will be ignored

SampleIm = imread('yaleB02_P00A-005E-10.pgm');

AllImagesYale_resized = zeros(size(SampleIm,2),size(SampleIm,2),...
    subjects_num*(image_num_eachSub-1));

num_images = size(AllImagesYale_resized,3);
image_size = size(AllImagesYale_resized,2);

for i = 3:length(files)
    name = filenames{i};
    foundedImages =...
        dir(fullfile(convertStringsToChars("CroppedYale/"+(name + "/*.pgm"))));
    for j = 1:length(foundedImages)-1
        this_image = imread(foundedImages(j).name);
        % resize the images
        AllImagesYale_resized(:,:,(i-3)*image_num_eachSub+j) = ...
            this_image(1:image_size,1:image_size);
    end
end

% shuffle the images
shuffled_nums = randperm(size(AllImagesYale_resized,3),size(AllImagesYale_resized,3));
AllImagesYale_resized = AllImagesYale_resized(:,:,shuffled_nums);

% rmv white noise from images and save the final result
AllImagesYale_resized = zscore(AllImagesYale_resized,0,[1 2]);
make_your_own_images(AllImagesYale_resized,'All_YALE_dewhited.mat');
