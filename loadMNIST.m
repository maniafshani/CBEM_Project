function [inputs, labels] = loadMNIST(filename)
% loadMNIST: Load an MNIST csv file and prepare the data for the network.
%
% Each line of the file contains 785 values: the first value is the
% label (the digit 0-9), the remaining 784 values are the intensities
% (0-255) of the 28x28 pixels.
%
% Output:
%   inputs : (N x 784) matrix, intensities scaled to [0.01, 1.0]
%            x = v/255 * 0.99 + 0.01
%   labels : (N x 1) vector with the digits 0-9

    data   = csvread(filename);
    labels = data(:, 1);
    inputs = data(:, 2:end) / 255.0 * 0.99 + 0.01;
end
