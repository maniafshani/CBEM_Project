clear;
clear classes;
clc;
close all;

%% Load the training dataset
trainingData = readmatrix("mnist_train.csv");

%% Create the network
net = neuralNetwork(784, 100, 10, 0.1);

%% Save the original weights before training
weightsIHBefore = net.w_ih;
weightsHOBefore = net.w_ho;

%% Choose the number of training epochs
%
% One epoch means that every training image is used once.
epochs = 3;

numberOfSamples = size(trainingData, 1);

%% Train the network
for epoch = 1:epochs

    fprintf("Starting epoch %d of %d\n", epoch, epochs);

    % Randomize the order of the training samples.
    sampleOrder = randperm(numberOfSamples);

    for position = 1:numberOfSamples

        % Find the row that will be trained on now.
        rowNumber = sampleOrder(position);

        % First column is the correct digit label.
        label = trainingData(rowNumber, 1);

        % Remaining 784 columns are pixel values.
        pixelValues = trainingData(rowNumber, 2:end);

        % Convert pixels into a 784x1 column vector.
        inputs = pixelValues';

        % Scale from 0-255 into approximately 0.01-1.00.
        inputs = inputs / 255.0 * 0.99 + 0.01;

        % Create the target output vector.
        %
        % Start with 0.01 for every digit.
        targets = ones(10, 1) * 0.01;

        % Set the correct digit position to 0.99.
        %
        % label is 0-9, but MATLAB indices are 1-10.
        targets(label + 1) = 0.99;

        % This is the line that actually trains the network.
        %
        % It performs forward propagation, error calculation,
        % backpropagation, and weight updates.
        net = net.train(inputs, targets);
    end
end

disp("Training complete.");

%% Save the trained weights
weightsIHAfter = net.w_ih;
weightsHOAfter = net.w_ho;

%% Save the trained network
save("trainedNetwork.mat", "net");
