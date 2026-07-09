% task6_ownDigits_optional.m
% Task 6 (optional): Classify your own handwritten digits.
%
% How to prepare the images:
%  - write digits with a dark pen on white paper (or in a paint program),
%  - crop each digit to a square, scale it to 28x28 pixels,
%  - save each digit as a png file named e.g. own_0.png, own_1.png, ...
%    in this folder.
%
% MNIST digits are white on a black background, therefore the intensity
% values of a typical scan/photo (dark digit on bright background) have
% to be inverted.
clear; close all; clc;

load('trainedNetwork.mat');
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, lr);
nn.w_ih = w_ih;
nn.w_ho = w_ho;

files = dir('own_*.png');
if isempty(files)
    fprintf(['No files own_*.png found.\n' ...
             'Create 28x28 png images of your own digits first (see comments).\n']);
    return;
end

figure('Name', 'Task 6: own handwriting', 'Position', [50 50 900 300]);
for k = 1:numel(files)
    img = imread(files(k).name);
    if size(img, 3) == 3
        img = rgb2gray(img);         % convert color image to gray scale
    end
    img = double(imresize(img, [28 28]));

    img = 255 - img;                 % invert: MNIST is white on black
    inputs = img' ;                  % transpose to match MNIST row order
    inputs = inputs(:) / 255.0 * 0.99 + 0.01;

    outputs = nn.query(inputs);
    [conf, idx] = max(outputs);
    prediction = idx - 1;

    subplot(1, numel(files), k);
    imagesc(img); colormap(gray); axis image off;
    title(sprintf('%s\npred: %d (%.2f)', files(k).name, prediction, conf), ...
          'Interpreter', 'none');
    fprintf('%s -> predicted digit: %d (output signal %.3f)\n', ...
            files(k).name, prediction, conf);
end
print('task6_own_digits.png', '-dpng');
