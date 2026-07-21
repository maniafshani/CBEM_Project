function plotEvaluation(randomTrials, constantMetrics, outputDir)
%PLOTEVALUATION Create all performance graphs required by Tasks 4 and 5.

    if isempty(randomTrials)
        error('At least one random trial is required.');
    end

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    numberOfTrials = numel(randomTrials);
    randomAccuracies = zeros(1, numberOfTrials);

    for trial = 1:numberOfTrials
        randomAccuracies(trial) = randomTrials(trial).overallAccuracy;
    end

    % Select the best random trial for detailed visualizations.
    [~, bestTrialIndex] = max(randomAccuracies);
    bestTrial = randomTrials(bestTrialIndex);

    % =============================================================
    % FIGURE 3: ACCURACY OF THE THREE RANDOM INITIALIZATIONS
    % =============================================================

    figure('Name', 'Random initialization trials', 'Color', 'white');
    bar(1:numberOfTrials, randomAccuracies);
    ylim([0, 100]);
    xticks(1:numberOfTrials);
    xticklabels(compose('Trial %d', 1:numberOfTrials));
    ylabel('Correct predictions (%)');
    title('Influence of different random initializations');
    grid on;

    addBarLabels(randomAccuracies, '%.2f%%', 1:numberOfTrials);

    saveas(gcf, fullfile(outputDir, ...
                        '03_random_initialization_accuracy.png'));
    close(gcf);

    % =============================================================
    % FIGURE 4: MEAN ACCURACY FOR EACH DIGIT
    % =============================================================

    allPerDigit = zeros(10, numberOfTrials);

    for trial = 1:numberOfTrials
        allPerDigit(:, trial) = randomTrials(trial).accuracyPerDigit;
    end

    meanPerDigit = mean(allPerDigit, 2, 'omitnan');

    figure('Name', 'Per-digit accuracy', 'Color', 'white');
    bar(0:9, meanPerDigit);
    ylim([0, 100]);
    xticks(0:9);
    xlabel('True digit');
    ylabel('Correct predictions (%)');
    title('Mean accuracy per digit across random trials');
    grid on;

    addBarLabels(meanPerDigit, '%.1f%%', 0:9);

    saveas(gcf, fullfile(outputDir, '04_mean_accuracy_per_digit.png'));
    close(gcf);

    % =============================================================
    % FIGURE 5: NUMBER OF WRONG PREDICTIONS FOR EACH DIGIT
    % =============================================================

    figure('Name', 'Wrong predictions by digit', 'Color', 'white');
    bar(0:9, bestTrial.wrongPerDigit);
    xticks(0:9);
    xlabel('True digit');
    ylabel('Number of wrong predictions');
    title(sprintf(['Wrong predictions by true digit ' ...
                   '(best random trial: %.2f%%)'], ...
                  bestTrial.overallAccuracy));
    grid on;

    addBarLabels(bestTrial.wrongPerDigit, '%d', 0:9);

    saveas(gcf, fullfile(outputDir, ...
                        '05_wrong_predictions_by_digit.png'));
    close(gcf);

    % =============================================================
    % FIGURE 6: CONFUSION MATRIX
    % =============================================================

    figure('Name', 'Confusion matrix', 'Color', 'white');
    imagesc(bestTrial.confusionMatrix);
    axis image;
    colormap(parula);
    colorbar;

    xticks(1:10);
    yticks(1:10);
    xticklabels(0:9);
    yticklabels(0:9);

    xlabel('Predicted digit');
    ylabel('True digit');
    title(sprintf('Confusion matrix: best random trial (%.2f%%)', ...
                  bestTrial.overallAccuracy));

    maximumValue = max(bestTrial.confusionMatrix(:));

    for trueIndex = 1:10
        for predictedIndex = 1:10
            value = bestTrial.confusionMatrix(trueIndex, predictedIndex);

            if value > maximumValue / 2
                textColor = 'white';
            else
                textColor = 'black';
            end

            text(predictedIndex, trueIndex, num2str(value), ...
                 'HorizontalAlignment', 'center', ...
                 'Color', textColor, ...
                 'FontSize', 8);
        end
    end

    saveas(gcf, fullfile(outputDir, '06_confusion_matrix.png'));
    close(gcf);

    % =============================================================
    % FIGURE 7: RANDOM INITIALIZATION VS ALL WEIGHTS = 0.5
    % =============================================================

    meanRandomAccuracy = mean(randomAccuracies);
    comparison = [meanRandomAccuracy, constantMetrics.overallAccuracy];

    figure('Name', 'Initialization comparison', 'Color', 'white');
    bar(1:2, comparison);
    ylim([0, 100]);
    xticks(1:2);
    xticklabels({'Mean random', 'All weights = 0.5'});
    ylabel('Correct predictions (%)');
    title('Random initialization versus identical 0.5 weights');
    grid on;

    addBarLabels(comparison, '%.2f%%', 1:2);

    saveas(gcf, fullfile(outputDir, ...
                        '07_random_vs_constant_05.png'));
    close(gcf);
end

function addBarLabels(values, formatText, xPositions)
%ADDBARLABELS Add numeric values above bars without extra toolboxes.

    values = values(:)';
    xPositions = xPositions(:)';

    if numel(values) ~= numel(xPositions)
        error('values and xPositions must have the same length.');
    end
    verticalOffset = max(1, 0.02 * max([values, 1]));

    for index = 1:numel(values)
        if isnan(values(index))
            labelText = 'N/A';
            labelHeight = 0;
        else
            labelText = sprintf(formatText, values(index));
            labelHeight = values(index);
        end

        text(xPositions(index), labelHeight + verticalOffset, labelText, ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom');
    end
end
