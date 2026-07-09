% task5_weightInitialization.m
% Task 5: Repeat the training and testing at least 3 times to observe
% the influence of different random initializations of the link weights.
% Additionally, initialize all link weights with the same constant value
% (w_jk = 0.5) and compare.
%
% Expected observation:
%  - different random seeds lead to slightly different, but always good
%    accuracies (the differences are typically well below 1 %);
%  - a constant initialization breaks the training completely: all
%    hidden nodes receive identical weights, therefore compute identical
%    outputs and identical weight updates ("symmetry problem"). The
%    network cannot distinguish its hidden nodes and the accuracy drops
%    dramatically.
clear; close all; clc;

input_nodes   = 784;
hidden_nodes  = 100;
output_nodes  = 10;
learning_rate = 0.1;
nRuns         = 3;

%% Load data once
fprintf('Loading data ...\n');
[trainInputs, trainLabels] = loadMNIST('mnist_train.csv');
[testInputs,  testLabels ] = loadMNIST('mnist_test.csv');
nTrain = size(trainInputs, 1);

targets = zeros(nTrain, output_nodes) + 0.01;
for i = 1:nTrain
    targets(i, trainLabels(i) + 1) = 0.99;
end

%% Several runs with different random initializations
accuracies = zeros(nRuns + 1, 1);
runNames = cell(nRuns + 1, 1);

for run = 1:nRuns
    rng(run);   % different seed -> different random initial weights
    nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

    fprintf('Run %d (random initialization, seed %d): training ...\n', run, run);
    for i = 1:nTrain
        nn = nn.train(trainInputs(i, :)', targets(i, :)');
    end

    accuracies(run) = testAccuracy(nn, testInputs, testLabels);
    runNames{run} = sprintf('random, seed %d', run);
    fprintf('   accuracy on test set: %.2f %%\n', accuracies(run));
end

%% One run with constant initialization w_jk = 0.5
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);
nn.w_ih(:) = 0.5;    % overwrite the random weights with a constant
nn.w_ho(:) = 0.5;

fprintf('Run %d (constant initialization w = 0.5): training ...\n', nRuns + 1);
for i = 1:nTrain
    nn = nn.train(trainInputs(i, :)', targets(i, :)');
end
accuracies(nRuns + 1) = testAccuracy(nn, testInputs, testLabels);
runNames{nRuns + 1} = 'constant, w = 0.5';
fprintf('   accuracy on test set: %.2f %%\n', accuracies(nRuns + 1));

%% Summary and visualization
fprintf('\nSummary:\n');
for run = 1:nRuns + 1
    fprintf('  %-18s : %.2f %%\n', runNames{run}, accuracies(run));
end

figure('Name', 'Task 5: influence of weight initialization', 'Position', [50 50 800 500]);
bar(accuracies);
grid on;
set(gca, 'XTickLabel', runNames);
ylabel('Accuracy on test set in %');
title('Influence of the weight initialization');
ylim([0 100]);
for run = 1:nRuns + 1
    text(run, accuracies(run) + 2, sprintf('%.1f %%', accuracies(run)), ...
         'HorizontalAlignment', 'center');
end
print('task5_initialization.png', '-dpng');

fprintf('\nTask 5 done. Figure saved as task5_initialization.png\n');
