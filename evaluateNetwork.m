function metrics = evaluateNetwork(net, testInputs, testLabels)
%EVALUATENETWORK Evaluate total and per-digit prediction performance.

    if size(testInputs, 2) ~= numel(testLabels)
        error('The number of test inputs and labels must be equal.');
    end

    % query accepts all test images as columns.
    finalOutputs = net.query(testInputs);

    % max returns indices 1...10, so subtract one to obtain digits 0...9.
    [~, predictedIndices] = max(finalOutputs, [], 1);
    predictedLabels = predictedIndices - 1;

    testLabels = testLabels(:)';
    predictedLabels = predictedLabels(:)';

    correctMask = predictedLabels == testLabels;
    numberOfCases = numel(testLabels);
    correctCount = sum(correctMask);

    % Rows = true labels, columns = predicted labels.
    confusionMatrix = accumarray( ...
        [testLabels(:) + 1, predictedLabels(:) + 1], ...
        1, [10, 10]);

    totalPerDigit = sum(confusionMatrix, 2);
    correctPerDigit = diag(confusionMatrix);
    wrongPerDigit = totalPerDigit - correctPerDigit;

    accuracyPerDigit = NaN(10, 1);
    existingDigits = totalPerDigit > 0;
    accuracyPerDigit(existingDigits) = ...
        100 * correctPerDigit(existingDigits) ...
        ./ totalPerDigit(existingDigits);

    availableAccuracies = accuracyPerDigit;
    availableAccuracies(~existingDigits) = -Inf;
    [bestDigitAccuracy, bestDigitIndex] = max(availableAccuracies);

    [largestWrongCount, mostWrongDigitIndex] = max(wrongPerDigit);

    metrics.numberOfCases = numberOfCases;
    metrics.correctCount = correctCount;
    metrics.overallAccuracy = 100 * correctCount / numberOfCases;

    metrics.testLabels = testLabels;
    metrics.predictedLabels = predictedLabels;
    metrics.correctMask = correctMask;

    metrics.confusionMatrix = confusionMatrix;
    metrics.totalPerDigit = totalPerDigit;
    metrics.correctPerDigit = correctPerDigit;
    metrics.wrongPerDigit = wrongPerDigit;
    metrics.accuracyPerDigit = accuracyPerDigit;

    metrics.bestDigit = bestDigitIndex - 1;
    metrics.bestDigitAccuracy = bestDigitAccuracy;
    metrics.mostWrongDigit = mostWrongDigitIndex - 1;
    metrics.largestWrongCount = largestWrongCount;
end
