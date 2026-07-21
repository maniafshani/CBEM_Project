classdef neuralNetwork
    % neuralNetwork
    % Basic three-layer neural network for CBEM Project 4.
    %
    % Architecture used by the homework:
    %   784 input nodes -> 100 hidden nodes -> 10 output nodes
    %
    % The active implementation follows the homework PDF:
    %   - sigmoid activation
    %   - random initial weights from MATLAB rand (0 to 1)
    %   - no bias nodes, because they are not included in the assignment
    %
    % Task 5 also asks for a comparison in which every weight starts at 0.5.

    properties
        inodes; % number of input nodes
        hnodes; % number of hidden nodes
        onodes; % number of output nodes
        lr;     % learning rate

        w_ih;   % weights from input layer to hidden layer
        w_ho;   % weights from hidden layer to output layer
    end

    methods
        function self = neuralNetwork(inputnodes, hiddennodes, ...
                                      outputnodes, learningrate, ...
                                      initializationMode)
            % neuralNetwork Construct a new neural network.
            %
            % Homework random initialization:
            %   net = neuralNetwork(784, 100, 10, 0.1);
            %
            % Required Task 5 constant-weight comparison:
            %   net = neuralNetwork(784, 100, 10, 0.1, 0.5);

            if nargin < 4
                error(['The constructor needs inputnodes, hiddennodes, ' ...
                       'outputnodes, and learningrate.']);
            end

            if nargin < 5
                initializationMode = 'homework-random';
            end

            if inputnodes <= 0 || hiddennodes <= 0 || outputnodes <= 0
                error('Every layer must contain a positive number of nodes.');
            end

            if learningrate <= 0 || learningrate >= 1
                error('The learning rate must satisfy 0 < learningrate < 1.');
            end

            self.inodes = inputnodes;
            self.hnodes = hiddennodes;
            self.onodes = outputnodes;
            self.lr = learningrate;

            % Convert the requested initialization into a simple mode name.
            if isnumeric(initializationMode)
                if isscalar(initializationMode) && initializationMode == 0.5
                    mode = 'constant-0.5';
                else
                    error('The only supported numeric initialization is 0.5.');
                end
            else
                mode = lower(char(initializationMode));
            end

            switch mode
                case {'homework-random', 'random'}
                    % =====================================================
                    % ACTIVE HOMEWORK INITIALIZATION
                    %
                    % The PDF says that the link weights are initialized
                    % with random values:
                    %
                    %               0 < w_jk < 1
                    %
                    % MATLAB rand produces pseudorandom values in [0, 1).
                    % =====================================================
                    self.w_ih = rand(hiddennodes, inputnodes);
                    self.w_ho = rand(outputnodes, hiddennodes);

                    % =====================================================
                    % OUR SUGGESTED ALTERNATIVE: XAVIER UNIFORM
                    %
                    % These lines are intentionally COMMENTS. They show the
                    % professor our suggested improvement, but the submitted
                    % active implementation above still follows the PDF.
                    %
                    % limit_ih = sqrt(6 / (inputnodes + hiddennodes));
                    % self.w_ih = ...
                    %     (2 * rand(hiddennodes, inputnodes) - 1) * limit_ih;
                    %
                    % limit_ho = sqrt(6 / (hiddennodes + outputnodes));
                    % self.w_ho = ...
                    %     (2 * rand(outputnodes, hiddennodes) - 1) * limit_ho;
                    % =====================================================

                case {'constant-0.5', 'constant', 'same'}
                    % Required comparison from Task 5:
                    % every input-to-hidden and hidden-to-output weight has
                    % exactly the same initial value.
                    self.w_ih = 0.5 * ones(hiddennodes, inputnodes);
                    self.w_ho = 0.5 * ones(outputnodes, hiddennodes);

                otherwise
                    error(['Unknown initialization mode. Use the default ' ...
                           'homework-random mode or the numeric value 0.5.']);
            end
        end

        function final_outputs = query(self, inputs)
            % query Perform the forward pass.
            %
            % inputs can contain one image or several images:
            %   one image:       784 x 1
            %   several images:  784 x numberOfImages

            if size(inputs, 1) ~= self.inodes
                error('inputs must contain %d rows.', self.inodes);
            end

            % Input layer -> hidden layer:
            %
            % hidden_inputs = W_ih * x
            hidden_inputs = self.w_ih * inputs;

            % Apply the sigmoid separately to every hidden neuron:
            %
            % hidden_outputs = sigmoid(hidden_inputs)
            hidden_outputs = neuralNetwork.actfun(hidden_inputs);

            % Hidden layer -> output layer:
            %
            % final_inputs = W_ho * hidden_outputs
            final_inputs = self.w_ho * hidden_outputs;

            % Apply the sigmoid separately to every output neuron:
            %
            % z = sigmoid(final_inputs)
            final_outputs = neuralNetwork.actfun(final_inputs);
        end

        function self = train(self, inputs, targets)
            % train Learn from one training image and its target vector.
            %
            % The neuralNetwork class is a MATLAB value class. Therefore,
            % this method returns the updated object:
            %
            %       net = net.train(inputs, targets);

            if ~isequal(size(inputs), [self.inodes, 1])
                error('train expects one input column of size %d x 1.', ...
                      self.inodes);
            end

            if ~isequal(size(targets), [self.onodes, 1])
                error('train expects one target column of size %d x 1.', ...
                      self.onodes);
            end

            % =============================================================
            % 1. FORWARD PASS
            % =============================================================

            hidden_inputs = self.w_ih * inputs;
            hidden_outputs = neuralNetwork.actfun(hidden_inputs);

            final_inputs = self.w_ho * hidden_outputs;
            final_outputs = neuralNetwork.actfun(final_inputs);

            % =============================================================
            % 2. BACKPROPAGATE THE ERRORS
            % =============================================================

            % Output error from the assignment:
            %
            % e_out = t - z
            output_errors = targets - final_outputs;

            % Hidden error from the assignment:
            %
            % e_h = W_ho^T * e_out
            %
            % Calculate this before changing W_ho, so both updates refer to
            % the same forward pass and the same old weight matrices.
            hidden_errors = self.w_ho' * output_errors;

            % =============================================================
            % 3. UPDATE HIDDEN-TO-OUTPUT WEIGHTS
            % =============================================================

            % Local output gradient:
            %
            % e_out .* z .* (1 - z)
            output_gradient = output_errors ...
                            .* final_outputs ...
                            .* (1 - final_outputs);

            % Vectorized homework update:
            %
            % Delta W_ho =
            % alpha * [e_out .* z .* (1-z)] * hidden_outputs^T
            self.w_ho = self.w_ho + self.lr ...
                      * (output_gradient * hidden_outputs');

            % =============================================================
            % 4. UPDATE INPUT-TO-HIDDEN WEIGHTS
            % =============================================================

            % Local hidden gradient:
            %
            % e_h .* hidden_outputs .* (1 - hidden_outputs)
            hidden_gradient = hidden_errors ...
                            .* hidden_outputs ...
                            .* (1 - hidden_outputs);

            % Vectorized homework update:
            %
            % Delta W_ih =
            % alpha * [e_h .* h .* (1-h)] * inputs^T
            self.w_ih = self.w_ih + self.lr ...
                      * (hidden_gradient * inputs');
        end
    end

    methods (Static)
        function out = actfun(x)
            % actfun Sigmoid activation function.
            %
            %               1
            % sigmoid(x) = --------
            %              1 + e^-x
            out = 1 ./ (1 + exp(-x));
        end
    end
end
