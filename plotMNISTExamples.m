function plotMNISTExamples(inputs, labels, outputDir)
%PLOTMNISTEXAMPLES Create the image visualizations required by Task 1.

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    % =============================================================
    % FIGURE 1: ONE 28 x 28 INTENSITY PLOT
    % =============================================================

    firstImage = reshape(inputs(:, 1), 28, 28)';

    figure('Name', 'MNIST intensity plot', 'Color', 'white');
    imagesc(firstImage);
    axis image off;
    colormap(gray);
    colorbar;
    title(sprintf('MNIST example: true digit %d', labels(1)));

    saveas(gcf, fullfile(outputDir, '01_single_mnist_intensity.png'));
    close(gcf);

    % =============================================================
    % FIGURE 2: FIVE EXAMPLES OF EVERY DIGIT
    % =============================================================

    examplesPerDigit = 5;

    figure('Name', 'MNIST examples by digit', ...
           'Color', 'white', ...
           'Position', [100, 100, 1000, 1400]);

    plotIndex = 1;

    for digit = 0:9
        matching = find(labels == digit, examplesPerDigit, 'first');

        for example = 1:examplesPerDigit
            subplot(10, examplesPerDigit, plotIndex);

            if example <= numel(matching)
                imageVector = inputs(:, matching(example));
                imageMatrix = reshape(imageVector, 28, 28)';
                imagesc(imageMatrix);
                axis image off;
                colormap(gray);

                if example == 1
                    title(sprintf('Digit %d', digit));
                end
            else
                axis off;
                text(0.5, 0.5, 'No example', ...
                     'HorizontalAlignment', 'center');
            end

            plotIndex = plotIndex + 1;
        end
    end

    saveas(gcf, fullfile(outputDir, '02_examples_of_digits_0_to_9.png'));
    close(gcf);
end
