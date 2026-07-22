function acc = testAccuracy(nn, inputs, labels)
% testAccuracy: Evaluate a trained network on a test set.
%   nn     : trained neuralNetwork object
%   inputs : (N x 784) prepared input signals
%   labels : (N x 1) true digits
% Returns the percentage of correctly predicted digits.

    n = size(inputs, 1);
    correct = 0;
    for i = 1:n
        [~, idx] = max(nn.query(inputs(i, :)'));
        correct = correct + ((idx - 1) == labels(i));
    end
    acc = 100 * correct / n;
end
