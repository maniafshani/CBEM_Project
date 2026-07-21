function [inputs, labels, targets] = prepareMNIST(data, createTargets)
%PREPAREMNIST Scale pixels and optionally create target vectors.
%
% Homework pixel scaling:
%
%           v_k
%   x_k = ------- * 0.99 + 0.01
%          255.0
%
% This maps:
%   0   -> 0.01
%   255 -> 1.00
%
% Target vectors use:
%   0.99 for the correct digit
%   0.01 for every other digit
%
% The PDF prose contains "0.1" once, but both explicit examples use 0.01.
% This implementation follows those explicit target-vector examples.

    if nargin < 2
        createTargets = false;
    end

    labels = data(:, 1)';

    rawPixels = data(:, 2:end)';

    % Exact scaling formula from the assignment.
    inputs = (rawPixels / 255.0) * 0.99 + 0.01;

    if createTargets
        numberOfCases = size(data, 1);

        % Start every output node at the low target value.
        targets = 0.01 * ones(10, numberOfCases);

        % MATLAB indices start at 1:
        % digit 0 -> target row 1
        % digit 1 -> target row 2
        % ...
        % digit 9 -> target row 10
        correctIndices = sub2ind([10, numberOfCases], ...
                                 labels + 1, ...
                                 1:numberOfCases);

        targets(correctIndices) = 0.99;
    else
        targets = [];
    end
end
