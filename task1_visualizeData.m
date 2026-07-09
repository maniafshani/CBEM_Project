% task1_visualizeData.m
% Task 1: Take a first look at the MNIST data.
%   (a) Plot one 28x28 image as a 2d intensity plot.
%   (b) Show multiple examples of each digit 0-9 from the large
%       training data set.
clear; close all; clc;

%% (a) One single digit as 2d intensity plot
data = csvread('mnist_train_100.csv');

label  = data(1, 1);
pixels = data(1, 2:end);
% The 784 values are stored row by row. MATLAB's reshape fills
% column-wise, so we reshape to 28x28 and transpose.
img = reshape(pixels, 28, 28)';

figure('Name', 'Task 1a: single digit');
imagesc(img);
colormap(gray);          % intensity plot in gray scale
colorbar;
axis image;
title(sprintf('First training image, label = %d', label));
print('task1_single_digit.png', '-dpng');

%% (b) Multiple examples of each digit from the large training set
nExamples = 8;                       % examples per digit
data = csvread('mnist_train.csv');
labels = data(:, 1);

figure('Name', 'Task 1b: examples of each digit', 'Position', [50 50 900 1100]);
for digit = 0:9
    idx = find(labels == digit, nExamples);   % first nExamples of this digit
    for j = 1:nExamples
        img = reshape(data(idx(j), 2:end), 28, 28)';
        subplot(10, nExamples, digit * nExamples + j);
        imagesc(img);
        colormap(gray);
        axis image off;
        if j == 1
            ylabel(num2str(digit));
            axis on;
            set(gca, 'XTick', [], 'YTick', []);
        end
    end
end
print('task1_digit_examples.png', '-dpng');

fprintf('Task 1 done. Figures saved as task1_single_digit.png and task1_digit_examples.png\n');
