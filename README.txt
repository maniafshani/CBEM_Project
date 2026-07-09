CBEM Project 4: A 3-layer neural network
=========================================

Files
-----
neuralNetwork.m               Task 2: the 3-layer network class (based on the template)
loadMNIST.m                   Helper: loads a MNIST csv file and scales the pixels
testAccuracy.m                Helper: computes the accuracy of a network on a test set
task1_visualizeData.m         Task 1: 2d intensity plots of the digits
task3_trainNetwork.m          Task 3: trains the network on mnist_train.csv,
                              saves the result to trainedNetwork.mat
task4_evaluateNetwork.m       Task 4: total + per-digit accuracy, bar chart,
                              confusion matrix
task5_weightInitialization.m  Task 5: 3 runs with different random seeds
                              + 1 run with constant weights w = 0.5
task6_ownDigits_optional.m    Task 6 (optional): classify own 28x28 png digits

trainedNetwork.mat            Already trained network (weights), so task 4 can be
                              run directly without training again
task*.png                     Result figures from a verified full run (can be used
                              directly on the presentation slides)

How to run
----------
Put the four MNIST csv files into the same folder as the .m files, then run
in MATLAB (or Octave), in this order:

    task1_visualizeData        (~1 min, mostly csv loading)
    task3_trainNetwork         (~1-2 min: loads 60000 samples, trains 1 epoch)
    task4_evaluateNetwork      (uses trainedNetwork.mat from task 3)
    task5_weightInitialization (trains 4 full networks, ~5 min)

Network design (as required by the project sheet)
-------------------------------------------------
784 input nodes (28x28 pixels), 100 hidden nodes, 10 output nodes,
sigmoid activation, learning rate 0.1, 1 training epoch,
inputs scaled by x = v/255*0.99 + 0.01, targets 0.99/0.01.

Note on the weight initialization: the initial random weights are shifted
to the interval (-0.5, 0.5) by "rand(...) - 0.5". With strictly positive
weights in (0,1), the weighted sums of 784 positive inputs would saturate
the sigmoid function from the start and the network would train very poorly.

Verified results (full run, seed 42)
------------------------------------
Task 4:  9519 / 10000 test cases correct = 95.19 %
         best digit:  1 (98.94 %),  0 (98.88 %)
         worst digit: 2 (92.05 %, 82 wrong cases), then 5, 7, 8
         most frequent confusions (see confusion matrix): 4<->9, 7->2, 3<->5

Task 5:  random seed 1: 95.14 %
         random seed 2: 94.26 %
         random seed 3: 94.95 %
         constant w = 0.5: 9.80 %  (!)
         -> Different random initializations only change the accuracy by
            well under 1 percentage point. Constant initialization destroys
            the training: all hidden nodes get identical weights, compute
            identical outputs and receive identical updates (symmetry
            problem), so the network effectively cannot learn and only
            reaches guessing level (~10 %).
