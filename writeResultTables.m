function writeResultTables(randomTrials, constantMetrics, seeds, outputDir)
%WRITERESULTTABLES Save evaluation values as CSV files for the presentation.

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    numberOfTrials = numel(randomTrials);

    experimentName = strings(numberOfTrials + 1, 1);
    seedColumn = NaN(numberOfTrials + 1, 1);
    correctCount = zeros(numberOfTrials + 1, 1);
    testCases = zeros(numberOfTrials + 1, 1);
    overallAccuracy = zeros(numberOfTrials + 1, 1);
    trainingSeconds = zeros(numberOfTrials + 1, 1);
    bestDigit = zeros(numberOfTrials + 1, 1);
    bestDigitAccuracy = zeros(numberOfTrials + 1, 1);
    mostWrongDigit = zeros(numberOfTrials + 1, 1);
    largestWrongCount = zeros(numberOfTrials + 1, 1);

    for trial = 1:numberOfTrials
        experimentName(trial) = sprintf('Random trial %d', trial);
        seedColumn(trial) = seeds(trial);
        correctCount(trial) = randomTrials(trial).correctCount;
        testCases(trial) = randomTrials(trial).numberOfCases;
        overallAccuracy(trial) = randomTrials(trial).overallAccuracy;
        trainingSeconds(trial) = randomTrials(trial).trainingSeconds;
        bestDigit(trial) = randomTrials(trial).bestDigit;
        bestDigitAccuracy(trial) = randomTrials(trial).bestDigitAccuracy;
        mostWrongDigit(trial) = randomTrials(trial).mostWrongDigit;
        largestWrongCount(trial) = randomTrials(trial).largestWrongCount;
    end

    finalRow = numberOfTrials + 1;

    experimentName(finalRow) = 'All weights = 0.5';
    correctCount(finalRow) = constantMetrics.correctCount;
    testCases(finalRow) = constantMetrics.numberOfCases;
    overallAccuracy(finalRow) = constantMetrics.overallAccuracy;
    trainingSeconds(finalRow) = constantMetrics.trainingSeconds;
    bestDigit(finalRow) = constantMetrics.bestDigit;
    bestDigitAccuracy(finalRow) = constantMetrics.bestDigitAccuracy;
    mostWrongDigit(finalRow) = constantMetrics.mostWrongDigit;
    largestWrongCount(finalRow) = constantMetrics.largestWrongCount;

    overallTable = table(experimentName, seedColumn, correctCount, ...
                         testCases, overallAccuracy, trainingSeconds, ...
                         bestDigit, bestDigitAccuracy, mostWrongDigit, ...
                         largestWrongCount);

    writetable(overallTable, ...
               fullfile(outputDir, 'overall_experiment_results.csv'));

    % Per-digit results for all random trials.
    digit = (0:9)';
    perDigitTable = table(digit);

    for trial = 1:numberOfTrials
        accuracyName = sprintf('Trial%dAccuracyPercent', trial);
        wrongName = sprintf('Trial%dWrongCount', trial);

        perDigitTable.(accuracyName) = ...
            randomTrials(trial).accuracyPerDigit;
        perDigitTable.(wrongName) = ...
            randomTrials(trial).wrongPerDigit;
    end

    perDigitTable.Constant05AccuracyPercent = ...
        constantMetrics.accuracyPerDigit;
    perDigitTable.Constant05WrongCount = ...
        constantMetrics.wrongPerDigit;

    writetable(perDigitTable, ...
               fullfile(outputDir, 'per_digit_results.csv'));

    % Save the best random trial's confusion matrix.
    randomAccuracies = arrayfun( ...
        @(trial) trial.overallAccuracy, randomTrials);

    [~, bestTrialIndex] = max(randomAccuracies);
    bestConfusion = randomTrials(bestTrialIndex).confusionMatrix;

    confusionTable = array2table(bestConfusion, ...
        'VariableNames', compose('Predicted_%d', 0:9), ...
        'RowNames', compose('True_%d', 0:9));

    writetable(confusionTable, ...
               fullfile(outputDir, 'best_trial_confusion_matrix.csv'), ...
               'WriteRowNames', true);
end
