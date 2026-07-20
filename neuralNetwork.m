classdef neuralNetwork
    % neuralNetwork
    %
    % A simple three-layer neural network class.
    %
    % The network contains:
    %   1. An input layer
    %   2. One hidden layer
    %   3. An output layer
    %
    % The network uses:
    %   - sigmoid activation functions
    %   - forward propagation for predictions
    %   - backpropagation for training
    %   - gradient descent for updating the weights
    %
    % Example creation:
    %
    %   net = neuralNetwork(784, 100, 10, 0.1);
    %
    % This creates a network with:
    %   784 input nodes
    %   100 hidden nodes
    %   10 output nodes
    %   learning rate 0.1

    properties
        % Number of neurons in the input layer.
        %
        % For MNIST:
        % 28 x 28 pixels = 784 input nodes.
        inodes;

        % Number of neurons in the hidden layer.
        %
        % The project suggests using 100 hidden nodes.
        hnodes;

        % Number of neurons in the output layer.
        %
        % For MNIST, there are 10 output nodes,
        % representing the digits 0 to 9.
        onodes;

        % Learning rate.
        %
        % This controls how large each weight update is.
        %
        % A larger value makes learning faster but may
        % make training unstable.
        %
        % A smaller value makes learning slower but may
        % make training more stable.
        lr;

        % Weight matrix between the input layer
        % and the hidden layer.
        %
        % Dimensions:
        %
        %   hidden nodes x input nodes
        %
        % For the standard MNIST network:
        %
        %   100 x 784
        %
        % Each row contains the weights going into
        % one hidden neuron.
        w_ih;

        % Weight matrix between the hidden layer
        % and the output layer.
        %
        % Dimensions:
        %
        %   output nodes x hidden nodes
        %
        % For the standard MNIST network:
        %
        %   10 x 100
        %
        % Each row contains the weights going into
        % one output neuron.
        w_ho;
    end

    methods
        function self = neuralNetwork( ...
                inputnodes, hiddennodes, outputnodes, learningrate)
            % neuralNetwork
            %
            % Constructor for the neuralNetwork class.
            %
            % The constructor is called when a new network
            % object is created.
            %
            % Example:
            %
            %   net = neuralNetwork(784, 100, 10, 0.1);
            %
            % Input arguments:
            %
            %   inputnodes
            %       Number of input neurons.
            %
            %   hiddennodes
            %       Number of hidden neurons.
            %
            %   outputnodes
            %       Number of output neurons.
            %
            %   learningrate
            %       Step size used during weight updates.

            % Store the number of input nodes.
            self.inodes = inputnodes;

            % Store the number of hidden nodes.
            self.hnodes = hiddennodes;

            % Store the number of output nodes.
            self.onodes = outputnodes;

            % Store the learning rate.
            self.lr = learningrate;

            % Initialize the weights between the input layer
            % and the hidden layer.
            %
            % randn generates normally distributed random numbers
            % centered around zero.
            %
            % The matrix dimensions are:
            %
            %   hnodes x inodes
            %
            % Dividing by sqrt(inodes) keeps the starting weights
            % relatively small.
            %
            % Small initial weights reduce the risk that the sigmoid
            % neurons immediately become saturated near 0 or 1.
            self.w_ih = randn(self.hnodes, self.inodes) ...
                / sqrt(self.inodes);

            % Initialize the weights between the hidden layer
            % and the output layer.
            %
            % The matrix dimensions are:
            %
            %   onodes x hnodes
            %
            % Dividing by sqrt(hnodes) again keeps the initial
            % weighted sums in a reasonable range.
            self.w_ho = randn(self.onodes, self.hnodes) ...
                / sqrt(self.hnodes);
        end

        function final_outputs = query(self, inputs)
            % query
            %
            % Performs forward propagation through the network.
            %
            % This method does not change the weights.
            % It only calculates the output of the network.
            %
            % Input:
            %
            %   inputs
            %       A vector containing one input sample.
            %
            %       For MNIST, this should contain 784 scaled
            %       pixel values.
            %
            % Output:
            %
            %   final_outputs
            %       A column vector containing the network outputs.
            %
            %       For MNIST, this has 10 values, one for each
            %       digit from 0 to 9.

            % Convert the input to a column vector.
            %
            % If inputs is already a column vector, this changes nothing.
            %
            % If inputs is a row vector, this converts it into the form:
            %
            %   inodes x 1
            inputs = inputs(:);

            % Check that the number of input values is correct.
            %
            % This helps detect mistakes such as accidentally
            % passing 785 values, including the label.
            if length(inputs) ~= self.inodes
                error( ...
                    "Expected %d input values, but received %d.", ...
                    self.inodes, length(inputs));
            end

            % ---------------------------------------------------------
            % FORWARD PROPAGATION: INPUT LAYER TO HIDDEN LAYER
            % ---------------------------------------------------------

            % Calculate the weighted input entering each hidden neuron.
            %
            % Matrix dimensions:
            %
            %   w_ih            = hnodes x inodes
            %   inputs          = inodes x 1
            %   hidden_inputs   = hnodes x 1
            %
            % Each hidden neuron receives a weighted sum of all inputs.
            hidden_inputs = self.w_ih * inputs;

            % Apply the sigmoid activation function.
            %
            % This transforms each weighted sum into a value
            % between 0 and 1.
            %
            % The resulting hidden_outputs vector contains the
            % signals passed from the hidden layer to the output layer.
            hidden_outputs = neuralNetwork.actfun(hidden_inputs);

            % ---------------------------------------------------------
            % FORWARD PROPAGATION: HIDDEN LAYER TO OUTPUT LAYER
            % ---------------------------------------------------------

            % Calculate the weighted input entering each output neuron.
            %
            % Matrix dimensions:
            %
            %   w_ho            = onodes x hnodes
            %   hidden_outputs  = hnodes x 1
            %   final_inputs    = onodes x 1
            final_inputs = self.w_ho * hidden_outputs;

            % Apply the sigmoid activation function to the final inputs.
            %
            % For MNIST, final_outputs contains 10 values.
            %
            % The output with the highest value is interpreted as
            % the predicted digit.
            final_outputs = neuralNetwork.actfun(final_inputs);
        end

        function self = train(self, inputs, targets)
            % train
            %
            % Trains the neural network using one input sample
            % and its corresponding target vector.
            %
            % The training procedure contains four main stages:
            %
            %   1. Forward propagation
            %   2. Error calculation
            %   3. Backpropagation
            %   4. Weight updates
            %
            % Because this class is a MATLAB value class, the changed
            % object must be returned.
            %
            % Therefore, call this method using:
            %
            %   net = net.train(inputs, targets);
            %
            % Input arguments:
            %
            %   inputs
            %       Input vector for one training sample.
            %
            %   targets
            %       Desired output vector for that sample.
            %
            %       For MNIST, the correct digit position should
            %       normally contain 0.99 and the others 0.01.

            % Convert inputs to a column vector.
            inputs = inputs(:);

            % Convert targets to a column vector.
            targets = targets(:);

            % Check that the input vector has the expected size.
            if length(inputs) ~= self.inodes
                error( ...
                    "Expected %d input values, but received %d.", ...
                    self.inodes, length(inputs));
            end

            % Check that the target vector has the expected size.
            if length(targets) ~= self.onodes
                error( ...
                    "Expected %d target values, but received %d.", ...
                    self.onodes, length(targets));
            end

            % =========================================================
            % STEP 1: FORWARD PROPAGATION
            % =========================================================
            %
            % First, calculate the network prediction using the
            % current weight values.

            % Calculate the weighted inputs entering the hidden layer.
            hidden_inputs = self.w_ih * inputs;

            % Apply sigmoid to obtain the hidden-layer outputs.
            hidden_outputs = neuralNetwork.actfun(hidden_inputs);

            % Calculate the weighted inputs entering the output layer.
            final_inputs = self.w_ho * hidden_outputs;

            % Apply sigmoid to obtain the final network outputs.
            final_outputs = neuralNetwork.actfun(final_inputs);

            % =========================================================
            % STEP 2: CALCULATE THE OUTPUT ERRORS
            % =========================================================

            % Compare the desired target values with the actual outputs.
            %
            % Error is defined as:
            %
            %   target - output
            %
            % A positive error means the output is too low.
            %
            % A negative error means the output is too high.
            output_errors = targets - final_outputs;

            % =========================================================
            % STEP 3: BACKPROPAGATE THE ERRORS
            % =========================================================

            % Estimate the errors associated with the hidden layer.
            %
            % The hidden neurons do not have direct target values.
            %
            % Therefore, output errors are propagated backward using
            % the transpose of the hidden-to-output weight matrix.
            %
            % Matrix dimensions:
            %
            %   w_ho'           = hnodes x onodes
            %   output_errors   = onodes x 1
            %   hidden_errors   = hnodes x 1
            %
            % The resulting hidden_errors vector estimates how much
            % each hidden neuron contributed to the final output errors.
            hidden_errors = self.w_ho' * output_errors;

            % =========================================================
            % STEP 4A: UPDATE HIDDEN-TO-OUTPUT WEIGHTS
            % =========================================================

            % Calculate the output-layer gradients.
            %
            % The derivative of sigmoid is:
            %
            %   sigmoid(x) * (1 - sigmoid(x))
            %
            % Since final_outputs already contains sigmoid(final_inputs),
            % the derivative can be calculated directly as:
            %
            %   final_outputs .* (1 - final_outputs)
            %
            % The output gradient combines:
            %
            %   - the output error
            %   - the sensitivity of the sigmoid function
            output_gradients = output_errors ...
                .* final_outputs ...
                .* (1 - final_outputs);

            % Calculate the correction for every hidden-to-output weight.
            %
            % Matrix dimensions:
            %
            %   output_gradients   = onodes x 1
            %   hidden_outputs'    = 1 x hnodes
            %   delta_w_ho         = onodes x hnodes
            %
            % This outer product creates one correction for every
            % connection from a hidden neuron to an output neuron.
            %
            % The learning rate controls the overall update size.
            delta_w_ho = self.lr ...
                * output_gradients ...
                * hidden_outputs';

            % Apply the hidden-to-output weight corrections.
            %
            % Addition is used because the error was defined as:
            %
            %   targets - final_outputs
            %
            % The sign of the error already indicates whether the
            % weight should increase or decrease.
            self.w_ho = self.w_ho + delta_w_ho;

            % =========================================================
            % STEP 4B: UPDATE INPUT-TO-HIDDEN WEIGHTS
            % =========================================================

            % Calculate the hidden-layer gradients.
            %
            % The hidden gradient combines:
            %
            %   - the backpropagated hidden error
            %   - the sigmoid derivative of the hidden neurons
            hidden_gradients = hidden_errors ...
                .* hidden_outputs ...
                .* (1 - hidden_outputs);

            % Calculate the correction for every input-to-hidden weight.
            %
            % Matrix dimensions:
            %
            %   hidden_gradients   = hnodes x 1
            %   inputs'            = 1 x inodes
            %   delta_w_ih         = hnodes x inodes
            %
            % This outer product creates one correction for every
            % input-to-hidden connection.
            delta_w_ih = self.lr ...
                * hidden_gradients ...
                * inputs';

            % Apply the input-to-hidden weight corrections.
            self.w_ih = self.w_ih + delta_w_ih;
        end
    end

    methods (Static)
        function out = actfun(x)
            % actfun
            %
            % Sigmoid activation function.
            %
            % Mathematical definition:
            %
            %             1
            % sigmoid = -------
            %           1 + e^-x
            %
            % The sigmoid maps every real number to a value
            % strictly between 0 and 1.
            %
            % Examples:
            %
            %   sigmoid(-5) is close to 0
            %   sigmoid(0)  equals 0.5
            %   sigmoid(5)  is close to 1
            %
            % The dot in ./ is important.
            %
            % It tells MATLAB to divide element by element,
            % so the function works correctly for vectors and matrices.
            out = 1 ./ (1 + exp(-x));
        end
    end
end
