CBEM PROJECT 4 — MATLAB NEURAL NETWORK
=======================================

PROJECT STRUCTURE
-----------------
neuralNetwork.m
    The 784 -> 100 -> 10 neural-network class based on the Moodle template.

project04_main.m
    Runs the complete assignment:
    - data visualization
    - three random initializations
    - training and testing
    - total accuracy
    - per-digit accuracy
    - wrong prediction counts
    - confusion matrix
    - constant 0.5 initialization comparison
    - CSV result tables and PNG figures

readMNISTCsv.m
    Reads and validates each MNIST CSV file.

prepareMNIST.m
    Applies the exact homework scaling:
        x = (v / 255.0) * 0.99 + 0.01

    It creates targets with:
        0.99 at the correct output node
        0.01 at all other output nodes

evaluateNetwork.m
    Calculates total accuracy, per-digit accuracy, wrong counts, predictions,
    and the confusion matrix.

plotMNISTExamples.m
    Creates a 28 x 28 intensity plot and a grid of examples for digits 0-9.

plotEvaluation.m
    Creates all evaluation graphs.

writeResultTables.m
    Saves the numerical results as CSV tables.

run_project04.m
    Convenience script that launches the final experiment.


DATA FILES
----------
Place these Moodle files inside the data folder:

    mnist_train_100.csv
    mnist_test_10.csv
    mnist_train.csv
    mnist_test.csv


HOW TO RUN
----------
Development test using the small files:

    experiment = project04_main(true);

Final homework run using the large files:

    experiment = project04_main(false);

You can also open and run:

    run_project04.m


INITIALIZATION DECISIONS
------------------------
The active default follows the homework PDF literally:

    self.w_ih = rand(hiddennodes, inputnodes);
    self.w_ho = rand(outputnodes, hiddennodes);

Task 5 separately runs:

    self.w_ih = 0.5 * ones(hiddennodes, inputnodes);
    self.w_ho = 0.5 * ones(outputnodes, hiddennodes);

Xavier-uniform initialization is preserved as COMMENTED code directly under
the active homework initialization. It is clearly marked as our suggested
alternative and is not active.


IMPORTANT PRESENTATION NOTE
---------------------------
The previous presentation values near 95.5% were produced with Xavier-uniform
initialization. They must not be described as results of the revised active
rand(0,1) implementation.

Run project04_main(false) in MATLAB with the complete Moodle datasets, then use
the regenerated files in the results folder for the final presentation.


OUTPUT FILES
------------
The results folder will contain:

    01_single_mnist_intensity.png
    02_examples_of_digits_0_to_9.png
    03_random_initialization_accuracy.png
    04_mean_accuracy_per_digit.png
    05_wrong_predictions_by_digit.png
    06_confusion_matrix.png
    07_random_vs_constant_05.png

    overall_experiment_results.csv
    per_digit_results.csv
    best_trial_confusion_matrix.csv
    project04_results.mat


OPTIONAL TASK
-------------
The personal-handwriting dataset is optional and is not included because the
group must provide its own handwritten digit images.
