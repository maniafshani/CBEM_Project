clear;
clear classes;
clc;
close all;

% Create a new neural network object.
%
% 784 input nodes:
% one for each pixel in a 28x28 image
%
% 100 hidden nodes
%
% 10 output nodes:
% one for each digit from 0 to 9
%
% Learning rate:
% 0.1
load("trainedNetwork.mat", "net");

% Load the CSV file.
%
% Replace the filename below with the exact name
% of your CSV file.
data = readmatrix("mnist_test_10.csv");

% Select the first row from the CSV file.
rowNumber = 6;

% The first value in the row is the correct digit label.
label = data(rowNumber, 1);

% The remaining 784 values are the image pixels.
pixelValues = data(rowNumber, 2:end);

% Convert the 1x784 row vector into a 784x1 column vector.
inputs = pixelValues';

% Scale the pixel values from 0-255 into 0.01-1.00.
inputs = inputs / 255.0 * 0.99 + 0.01;

% Pass the image through the network.
outputs = query(net, inputs);

% Find the output node with the highest value.
[~, predictedIndex] = max(outputs);

% MATLAB indices are 1-10, but digits are 0-9.
predictedDigit = predictedIndex - 1;

fprintf("Actual digit: %d\n", label);
fprintf("Predicted digit: %d\n", predictedDigit);

% Display the image stored in the CSV row.
imageMatrix = reshape(pixelValues, 28, 28)';

figure;
imagesc(imageMatrix);
axis image;
axis off;
colormap gray;

title(sprintf( ...
    "Actual: %d | Predicted: %d", ...
    label, predictedDigit));
