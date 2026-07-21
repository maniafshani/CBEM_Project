function experiment = project04_main(useSmallData, dataDir)
%PROJECT04_MAIN Complete solution workflow for CBEM Project 4.
%
% Small development run:
%  experiment = project04_main(true);
%
% Required final run with the large MNIST datasets:
%   experiment = project04_main(false);
%
% Supply a custom data folder:
%   experiment = project04_main(false, 'C:\path\to\mnist\data');
%
% Required tasks covered:
%   1. Plot MNIST images as 28 x 28 intensity plots.
%   2. Use the neuralNetwork class based on the Moodle template.
%   3. Train the 784-100-10 network.
%   4. Evaluate overall and per-digit accuracy and visualize errors.
%   5. Repeat random initialization three times.
%   6. Compare with every weight initialized to 0.5.
%
% The optional personal-handwriting dataset is not included because it
% requires images supplied by the group.

    if nargin < 1
        useSmallData = false;
    end

    projectDir = fileparts(mfilename('fullpath'));

    if nargin < 2
        dataDir = fullfile(projectDir, 'data');
    end

    outputDir = fullfile(projectDir, 'results');
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    if useSmallData
        trainingFile = fullfile(dataDir, 'mnist_train_100.csv');
        testFile = fullfile(dataDir, 'mnist_test_10.csv');
        dataSetName = 'small development datasets';
    else
        trainingFile = fullfile(dataDir, 'mnist_train.csv');
        testFile = fullfile(dataDir, 'mnist_test.csv');
        dataSetName = 'large MNIST datasets';
    end

    fprintf('\n============================================================\n');
    fprintf('CBEM Project 4: A 3-layer neural network\n');
    fprintf('Using: %s\n', dataSetName);
    fprintf('============================================================\n');

    % Read and validate the CSV files.
    trainingData = readMNISTCsv(trainingFile);
    testData = readMNISTCsv(testFile);

    if ~useSmallData
        if size(trainingData, 1) ~= 60000
            warning('Expected 60,000 training rows, but found %d.', ...
                    size(trainingData, 1));
        end

        if size(testData, 1) ~= 10000
            error(['The large test dataset must contain 10,000 complete ' ...
                   'rows. Found %d.'], size(testData, 1));
        end
    end

    % Convert labels and pixels into the vectors used by the network.
    [trainingInputs, trainingLabels, trainingTargets] = ...
        prepareMNIST(trainingData, true);
    clear trainingData;

    [testInputs, testLabels] = prepareMNIST(testData, false);
    clear testData;

    % Task 1: visualize examples before training.
    plotMNISTExamples(trainingInputs, trainingLabels, outputDir);

    % Homework network dimensions.
    inputNodes = 784;
    hiddenNodes = 100;
    outputNodes = 10;

    % Kept from the completed project setup.
    learningRate = 0.1;
    epochs = 1;

    % Task 5: at least three different random initializations.
    seeds = [11, 22, 33];
    numberOfTrials = numel(seeds);
    randomTrials = cell(1, numberOfTrials);

    fprintf('\nNetwork architecture: %d -> %d -> %d\n', ...
            inputNodes, hiddenNodes, outputNodes);
    fprintf('Learning rate: %.2f\n', learningRate);
    fprintf('Epochs: %d\n', epochs);

    % =============================================================
    % THREE RANDOM-INITIALIZATION EXPERIMENTS
    % =============================================================
    %
    % The training examples remain in the same order for every trial.
    % This isolates the requested difference: the random starting weights.

    for trial = 1:numberOfTrials
        fprintf('\n--- Random initialization trial %d/%d, seed %d ---\n', ...
                trial, numberOfTrials, seeds(trial));

        rng(seeds(trial), 'twister');

        % The default constructor activates the PDF's rand(...) weights.
        net = neuralNetwork(inputNodes, hiddenNodes, outputNodes, ...
                            learningRate);

        startTime = tic;
        net = trainForEpochs(net, trainingInputs, trainingTargets, epochs);
        trainingSeconds = toc(startTime);

        metrics = evaluateNetwork(net, testInputs, testLabels);
        metrics.seed = seeds(trial);
        metrics.trainingSeconds = trainingSeconds;

        randomTrials{trial} = metrics;
        displayMetrics(metrics);
    end

      randomTrials = [randomTrials{:}];

    % =============================================================
    % CONSTANT 0.5 EXPERIMENT REQUIRED BY TASK 5
    % =============================================================

    fprintf('\n--- Constant initialization: every weight = 0.5 ---\n');

    constantNet = neuralNetwork(inputNodes, hiddenNodes, outputNodes, ...
                                learningRate, 0.5);

    startTime = tic;
    constantNet = trainForEpochs(constantNet, trainingInputs, ...
                                 trainingTargets, epochs);
    trainingSeconds = toc(startTime);

    constantMetrics = evaluateNetwork(constantNet, testInputs, testLabels);
    constantMetrics.seed = NaN;
    constantMetrics.trainingSeconds = trainingSeconds;

    displayMetrics(constantMetrics);

    % Required visualizations and result tables.
    plotEvaluation(randomTrials, constantMetrics, outputDir);
    writeResultTables(randomTrials, constantMetrics, seeds, outputDir);

    % Save everything in a MAT file for later use in the presentation.
    experiment.settings.inputNodes = inputNodes;
    experiment.settings.hiddenNodes = hiddenNodes;
    experiment.settings.outputNodes = outputNodes;
    experiment.settings.learningRate = learningRate;
    experiment.settings.epochs = epochs;
    experiment.settings.seeds = seeds;
    experiment.settings.trainingCases = size(trainingInputs, 2);
    experiment.settings.testCases = size(testInputs, 2);
    experiment.settings.activeInitialization = ...
        'MATLAB rand: homework random values in [0,1)';
    experiment.settings.constantComparison = 0.5;

    experiment.randomTrials = randomTrials;
    experiment.constant05 = constantMetrics;

    save(fullfile(outputDir, 'project04_results.mat'), 'experiment');

    fprintf('\n============================================================\n');
    fprintf('Finished. Figures and tables are in:\n%s\n', outputDir);
    fprintf('============================================================\n');
end

function net = trainForEpochs(net, inputs, targets, epochs)
%TRAINFOREPOCHS Train sample-by-sample.

    numberOfCases = size(inputs, 2);

    for epoch = 1:epochs
        fprintf('Epoch %d/%d\n', epoch, epochs);

        % The original CSV order is intentionally used for every trial.
        for sample = 1:numberOfCases
            net = net.train(inputs(:, sample), targets(:, sample));

            if mod(sample, 10000) == 0 || sample == numberOfCases
                fprintf('  trained %d/%d examples\n', ...
                        sample, numberOfCases);
            end
        end
    end
end

function displayMetrics(metrics)
%DISPLAYMETRICS Print a readable summary in the command window.

    fprintf('Correct predictions: %d/%d\n', ...
            metrics.correctCount, metrics.numberOfCases);
    fprintf('Overall accuracy: %.2f%%\n', metrics.overallAccuracy);
    fprintf('Best predicted digit: %d (%.2f%%)\n', ...
            metrics.bestDigit, metrics.bestDigitAccuracy);
    fprintf('Digit with most wrong predictions: %d (%d wrong)\n', ...
            metrics.mostWrongDigit, metrics.largestWrongCount);
end
