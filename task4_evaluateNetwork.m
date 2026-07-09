% task4_evaluateNetwork.m
% Task 4: Evaluate the performance of the trained network on the large
% test data set (mnist_test.csv, 10000 samples).
%   - total number / fraction of correct predictions
%   - percentage of correct predictions separately for each digit
%   - visualization of the evaluation
clear; close all; clc;

%% Load the trained network from task 3
load('trainedNetwork.mat');
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, lr);
nn.w_ih = w_ih;
nn.w_ho = w_ho;

%% Load and prepare the test data
fprintf('Loading test data ...\n');
[inputs, labels] = loadMNIST('mnist_test.csv');
nSamples = size(inputs, 1);

%% Query the network for every test image
predictions = zeros(nSamples, 1);
for i = 1:nSamples
    outputs = nn.query(inputs(i, :)');
    [~, idx] = max(outputs);       % node with the strongest signal
    predictions(i) = idx - 1;      % node 1 -> digit 0, ..., node 10 -> digit 9
end

%% Overall performance
nCorrect = sum(predictions == labels);
fprintf('\nTotal: %d of %d test cases correct (%.2f %%)\n', ...
        nCorrect, nSamples, 100 * nCorrect / nSamples);

%% Per-digit performance
accPerDigit = zeros(10, 1);
wrongPerDigit = zeros(10, 1);
fprintf('\nDigit | correct | total | accuracy\n');
fprintf('---------------------------------\n');
for digit = 0:9
    idx = (labels == digit);
    accPerDigit(digit + 1)   = 100 * sum(predictions(idx) == digit) / sum(idx);
    wrongPerDigit(digit + 1) = sum(predictions(idx) ~= digit);
    fprintf('  %d   |  %4d   | %4d  |  %.2f %%\n', digit, ...
            sum(predictions(idx) == digit), sum(idx), accPerDigit(digit + 1));
end

[~, best]  = max(accPerDigit);
[~, worst] = min(accPerDigit);
[~, mostWrong] = max(wrongPerDigit);
fprintf('\nBest predicted digit:  %d (%.2f %%)\n', best - 1, accPerDigit(best));
fprintf('Worst predicted digit: %d (%.2f %%)\n', worst - 1, accPerDigit(worst));
fprintf('Most wrong predictions: digit %d (%d wrong cases)\n', ...
        mostWrong - 1, wrongPerDigit(mostWrong));

%% Visualization: bar chart of per-digit accuracy
figure('Name', 'Task 4: accuracy per digit', 'Position', [50 50 800 500]);
bar(0:9, accPerDigit);
grid on;
xlabel('Digit');
ylabel('Correct predictions in %');
title(sprintf('Accuracy per digit (overall: %.2f %%)', 100 * nCorrect / nSamples));
ylim([80 100]);
for d = 0:9
    text(d, accPerDigit(d + 1) + 0.4, sprintf('%.1f', accPerDigit(d + 1)), ...
         'HorizontalAlignment', 'center', 'FontSize', 8);
end
print('task4_accuracy_per_digit.png', '-dpng');

%% Visualization: confusion matrix (which digit is mistaken for which)
confMat = zeros(10, 10);
for i = 1:nSamples
    confMat(labels(i) + 1, predictions(i) + 1) = ...
        confMat(labels(i) + 1, predictions(i) + 1) + 1;
end

figure('Name', 'Task 4: confusion matrix', 'Position', [50 50 700 600]);
imagesc(0:9, 0:9, log10(confMat + 1));   % log scale to make errors visible
colorbar;
xlabel('Predicted digit');
ylabel('True digit');
title('Confusion matrix, log_{10}(count + 1)');
axis image;
set(gca, 'XTick', 0:9, 'YTick', 0:9);
print('task4_confusion_matrix.png', '-dpng');

fprintf('\nTask 4 done. Figures saved as task4_accuracy_per_digit.png and task4_confusion_matrix.png\n');
