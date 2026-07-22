classdef neuralNetwork
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
            self.inodes = inputnodes;
            self.hnodes = hiddennodes;
            self.onodes = outputnodes;
            self.lr = learningrate;
            
            % Initialization: rand() generates values between 0 and 1. 
            % Subtracting 0.5 centers them around 0 to prevent sigmoid saturation.
            self.w_ih = rand(self.hnodes, self.inodes) - 0.5;
            self.w_ho = rand(self.onodes, self.hnodes) - 0.5;
        end
            
        % Query the neural network. Computes the predictions from inputs.
        function final_outputs = query(self, inputs)
            % Ensure input is a column vector
            inputs = inputs(:);
            
            % Hidden layer calculations
            hidden_inputs = self.w_ih * inputs;
            hidden_outputs = neuralNetwork.actfun(hidden_inputs);
            
            % Output layer calculations
            final_inputs = self.w_ho * hidden_outputs;
            final_outputs = neuralNetwork.actfun(final_inputs);
        end

        % Train the neural network using backpropagation
        function self = train(self, inputs, targets)
            % Ensure vectors are column vectors
            inputs = inputs(:);
            targets = targets(:);
            
            % Forward pass (identical to query)
            hidden_inputs = self.w_ih * inputs;
            hidden_outputs = neuralNetwork.actfun(hidden_inputs);
            
            final_inputs = self.w_ho * hidden_outputs;
            final_outputs = neuralNetwork.actfun(final_inputs);
            
            % Calculate errors
            output_errors = targets - final_outputs;
            hidden_errors = self.w_ho' * output_errors;
            
            % Update link weights between hidden and output layer
            self.w_ho = self.w_ho + self.lr * (output_errors .* final_outputs .* (1.0 - final_outputs)) * hidden_outputs';
            
            % Update link weights between input and hidden layer
            self.w_ih = self.w_ih + self.lr * (hidden_errors .* hidden_outputs .* (1.0 - hidden_outputs)) * inputs';
        end
    end

    methods(Static)
        % Sigmoid activation function
        function out = actfun(x)
            out = 1 ./ (1 + exp(-x));
        end
    end
end