% task3_trainNetwork.m
% Task 3: Train a 3-layer network with the large training data set
% (mnist_train.csv, 60000 samples) using the neuralNetwork class.
% The trained network is saved to 'trainedNetwork.mat' so it can be
% evaluated in task 4 without retraining.
clear; close all; clc;

%% Network design (see project sheet)
input_nodes   = 784;    % 28x28 pixels
hidden_nodes  = 100;
output_nodes  = 10;     % one node per digit 0-9
learning_rate = 0.1;

rng(42);   % use rand('seed',42) in Octave if rng is not available
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

%% Load and prepare the training data
fprintf('Loading training data ...\n');
[inputs, labels] = loadMNIST('mnist_train.csv');
nSamples = size(inputs, 1);

% Build the target vectors: 0.99 for the node of the correct digit,
% 0.01 for all the other nodes.
targets = zeros(nSamples, output_nodes) + 0.01;
for i = 1:nSamples
    % careful with indexing: digit 0 -> node 1, digit 9 -> node 10
    targets(i, labels(i) + 1) = 0.99;
end

%% Train the network (one epoch over the whole training set)
epochs = 1;
fprintf('Training on %d samples, %d epoch(s) ...\n', nSamples, epochs);
tic;
for e = 1:epochs
    for i = 1:nSamples
        nn = nn.train(inputs(i, :)', targets(i, :)');
        if mod(i, 10000) == 0
            fprintf('  epoch %d: %d / %d samples\n', e, i, nSamples);
        end
    end
end
fprintf('Training finished after %.1f s.\n', toc);

%% Save the trained network
% The weights and parameters are stored as plain arrays, which works in
% MATLAB as well as in Octave (Octave cannot save classdef objects).
w_ih = nn.w_ih;  w_ho = nn.w_ho;  lr = nn.lr;
save('-mat', 'trainedNetwork.mat', 'w_ih', 'w_ho', 'lr', ...
     'input_nodes', 'hidden_nodes', 'output_nodes');
fprintf('Trained network saved to trainedNetwork.mat\n');
