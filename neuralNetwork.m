classdef neuralNetwork
    % neuralNetwork: Class for a three layer neural network
    %
    % The network consists of an input layer, one hidden layer and an
    % output layer. Signals are propagated as
    %       z = f( W_ho * f( W_ih * x ) )
    % with the sigmoid function f as activation function.
    % Training is done by backpropagation of the output errors
    % (gradient descent on the squared error).
    %
    % Usage:
    %   nn = neuralNetwork(784, 100, 10, 0.1);   % create network
    %   nn = nn.train(inputs, targets);          % one training step
    %   z  = nn.query(inputs);                   % prediction

    properties
        inodes; % number of input nodes
        hnodes; % number of hidden nodes
        onodes; % number of output nodes
        lr;     % learning rate
        w_ih;   % link weights between input layer and hidden layer
        w_ho;   % link weights between hidden layer and output layer
    end

    methods
        function self = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
            % neuralNetwork: Construct an instance of this class
            self.inodes = inputnodes;
            self.hnodes = hiddennodes;
            self.onodes = outputnodes;
            self.lr     = learningrate;

            % Initialize the link weights with random values 0 < w < 1,
            % centered around zero by subtracting 0.5. Centering the
            % initial weights avoids saturating the sigmoid function
            % right from the start (with 784 strictly positive weights
            % the weighted sums would otherwise be far in the flat
            % region of the sigmoid and the network could hardly learn).
            self.w_ih = rand(hiddennodes, inputnodes)  - 0.5;  % (nh x nin)
            self.w_ho = rand(outputnodes, hiddennodes) - 0.5;  % (nout x nh)
        end

        % Query the neural network. The function takes a vector of inputs
        % and computes the outputs, i.e. a "prediction".
        function final_outputs = query(self, inputs)
            inputs = inputs(:);                            % column vector
            hidden_outputs = self.actfun(self.w_ih * inputs);        % s_h = f(W_ih * x)
            final_outputs  = self.actfun(self.w_ho * hidden_outputs); % z   = f(W_ho * s_h)
        end

        % Train the neural network. This function learns the link weights
        % from the training data (one input vector and its target vector).
        function self = train(self, inputs, targets)
            inputs  = inputs(:);
            targets = targets(:);

            % --- forward pass ---
            hidden_outputs = self.actfun(self.w_ih * inputs);
            final_outputs  = self.actfun(self.w_ho * hidden_outputs);

            % --- backpropagation of the errors ---
            output_errors = targets - final_outputs;      % e_out = t - z
            hidden_errors = self.w_ho' * output_errors;   % e_h = W_ho^T * e_out

            % --- update the link weights ---
            % dW = alpha * [e .* s .* (1 - s)] * s_previous^T
            self.w_ho = self.w_ho + self.lr * ...
                (output_errors .* final_outputs .* (1 - final_outputs)) * hidden_outputs';
            self.w_ih = self.w_ih + self.lr * ...
                (hidden_errors .* hidden_outputs .* (1 - hidden_outputs)) * inputs';
        end

    end

    methods(Static)
        function out = actfun(x)
            % Activation function: sigmoid (logistic function)
            out = 1 ./ (1 + exp(-x));
        end
    end

end
