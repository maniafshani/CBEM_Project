%% Task 1: Data Visualization
disp('Task 1: Loading Data and Generating Plots...');
train_data = readmatrix('mnist_train.csv');
test_data = readmatrix('mnist_test.csv');

% Plot a single 28x28 digit
figure(1);
single_img = reshape(train_data(1, 2:end), 28, 28)';
imagesc(single_img);
colormap(flipud(gray));
title(['Label: ', num2str(train_data(1, 1))]);
axis off;

% Plot multiple examples of digits 0-9
figure(2);
t = tiledlayout(10, 10, 'TileSpacing', 'none', 'Padding', 'none');
for digit = 0:9
    idx = find(train_data(:, 1) == digit);
    for i = 1:min(10, length(idx))
        nexttile;
        img = reshape(train_data(idx(i), 2:end), 28, 28)';
        imagesc(img);
        colormap(flipud(gray));
        axis off;
    end
    % Fill empty grid spaces if fewer than 10 examples exist
    for i = length(idx)+1:10
        nexttile;
        axis off;
    end
end
title(t, 'Multiple Examples of Digits 0-9 (mnist_train_100.csv)');

%% Task 3: Train Network
disp('Task 3: Training the 3-Layer Network...');
input_nodes = 784;
hidden_nodes = 100;
output_nodes = 10;
learning_rate = 0.1;

nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

% Run 10 epochs to extract basic patterns from the small 100-sample set
epochs = 1;
for e = 1:epochs
    for i = 1:size(train_data, 1)
        % Scale and shift inputs to avoid zero values
        inputs = (train_data(i, 2:end) / 255.0 * 0.99) + 0.01;
        
        % Create target output vector
        targets = repmat(0.01, 10, 1);
        targets(train_data(i, 1) + 1) = 0.99;
        
        % Train the network
        nn = nn.train(inputs, targets);
    end
end

%% Task 4: Evaluate Performance
disp('Task 4: Evaluating Network on Test Data...');
correct = 0;
total = size(test_data, 1);
digit_correct = zeros(1, 10);
digit_total = zeros(1, 10);

for i = 1:total
    correct_label = test_data(i, 1);
    inputs = (test_data(i, 2:end) / 255.0 * 0.99) + 0.01;
    outputs = nn.query(inputs);
    
    [~, max_idx] = max(outputs);
    predicted_label = max_idx - 1;
    
    digit_total(correct_label + 1) = digit_total(correct_label + 1) + 1;
    if predicted_label == correct_label
        correct = correct + 1;
        digit_correct(correct_label + 1) = digit_correct(correct_label + 1) + 1;
    end
end

total_accuracy = correct / total;
digit_accuracy = digit_correct ./ digit_total;

fprintf('\n=== Evaluation Results ===\n');
fprintf('Total Correct: %d / %d\n', correct, total);
fprintf('Total Accuracy: %.2f%%\n\n', total_accuracy * 100);

disp('Accuracy per digit:');
for d = 0:9
    fprintf('Digit %d: %.2f%%\n', d, digit_accuracy(d+1) * 100);
end

% Visualize evaluation results
figure(3);
bar(0:9, digit_accuracy * 100, 'FaceColor', '#4C72B0');
xlabel('Digit');
ylabel('Accuracy (%)');
title(sprintf('Prediction Accuracy per Digit (Total: %.1f%%)', total_accuracy * 100));
xticks(0:9);
ylim([0 100]);

%% Task 5: Different Weight Initializations
disp('\nTask 5: Running Initialization Trials...');

% Trial Random Weights 3 Times
for trial = 1:3
    nn_rand = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);
    for e = 1:epochs
        for i = 1:size(train_data, 1)
            inputs = (train_data(i, 2:end) / 255.0 * 0.99) + 0.01;
            targets = repmat(0.01, 10, 1);
            targets(train_data(i, 1) + 1) = 0.99;
            nn_rand = nn_rand.train(inputs, targets);
        end
    end
    
    c = 0;
    for i = 1:size(test_data, 1)
        inputs = (test_data(i, 2:end) / 255.0 * 0.99) + 0.01;
        outputs = nn_rand.query(inputs);
        [~, max_idx] = max(outputs);
        if (max_idx - 1) == test_data(i, 1)
            c = c + 1;
        end
    end
    fprintf('Random Trial %d Accuracy: %.2f%%\n', trial, (c/total)*100);
end

% Fixed Weights Initialization (0.5)
nn_fixed = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);
nn_fixed.w_ih = repmat(0.5, hidden_nodes, input_nodes);
nn_fixed.w_ho = repmat(0.5, output_nodes, hidden_nodes);

for e = 1:epochs
    for i = 1:size(train_data, 1)
        inputs = (train_data(i, 2:end) / 255.0 * 0.99) + 0.01;
        targets = repmat(0.01, 10, 1);
        targets(train_data(i, 1) + 1) = 0.99;
        nn_fixed = nn_fixed.train(inputs, targets);
    end
end

c = 0;
for i = 1:size(test_data, 1)
    inputs = (test_data(i, 2:end) / 255.0 * 0.99) + 0.01;
    outputs = nn_fixed.query(inputs);
    [~, max_idx] = max(outputs);
    if (max_idx - 1) == test_data(i, 1)
        c = c + 1;
    end
end
fprintf('Fixed Weights (0.5) Accuracy: %.2f%%\n', (c/total)*100);