import numpy as np
import os


# sigmoid activation function
def sigmoid(x, derivative=False):
    sig = 1 / (1 + np.exp(-x))
    if derivative:
        return sig * (1 - sig)
    return sig


# divides a list into n-sized chunks
def chunks(arr, size):
    chunks = list()
    for i in range(0, len(arr), size):
        chunks.append(arr[i:i+size])
    return chunks


# structure for training and using a Neural Network
class NeuralNetwork:

    def __init__(self, **kwargs):

        if 'weights' in kwargs and 'biases' in kwargs:
            weights = kwargs.get("weights")
            biases = kwargs.get("biases")
            self.num_layers = len(weights) + 1
            self.weights = weights
            self.biases = biases

        elif 'input_size' in kwargs and 'output_size' in kwargs:
            input_size = kwargs.get("input_size")
            output_size = kwargs.get("output_size")
            hidden_size = kwargs.get("hidden_size", 1)
            num_hidden_layers = kwargs.get("num_hidden_layers", 1)
            self.num_layers = num_hidden_layers + 2

            # creates a list of layer sizes used to initialize the weights and biases
            layer_sizes = [input_size]
            for i in range(num_hidden_layers):
                layer_sizes.append(hidden_size)
            layer_sizes.append(output_size)

            # initializes the weights and biases with random values between -1 and 1
            weights = list()
            biases = list()
            for i in range(self.num_layers - 1):
                weights.append(2 * np.random.rand(layer_sizes[i + 1], layer_sizes[i]) - 1)
                biases.append(2 * np.random.rand(layer_sizes[i + 1]) - 1)

            self.weights = np.array(weights)
            self.biases = np.array(biases)

        else:
            raise ValueError("Must specify 'input_size' and 'output_size' or 'weights' and 'biases'"
                             "in order to initialize weights and biases")

    @classmethod
    def from_existing_model(cls, directory):
        weights = np.load(os.path.join(directory, "weights.npy"))
        biases = np.load(os.path.join(directory, "biases.npy"))
        return cls(weights=weights, biases=biases)

    # activation function
    def activate(self, value, derivative=False):
        return sigmoid(value, derivative)

    # forward propagation algorithm
    def forward_propagate(self, inputs):
        layer_inputs = list()  # inputs to the hidden and output layers
        layer_activations = list()  # activations of all the layers
        layer_activations.append(inputs)

        current_layer_activations = inputs
        for i in range(self.num_layers - 1):

            next_layer_inputs = self.weights[i].dot(current_layer_activations) + self.biases[i]
            next_layer_activations = self.activate(next_layer_inputs)

            layer_inputs.append(next_layer_inputs)
            layer_activations.append(next_layer_activations)
            current_layer_activations = next_layer_activations

        return {
            "outputs": current_layer_activations,
            "layer_inputs": layer_inputs,
            "layer_activations": layer_activations
        }

    # use the model to make a prediction on the given inputs
    def predict(self, inputs):
        return self.forward_propagate(inputs)["outputs"]

    # returns the error of the outputs relative to the labeled outputs
    def cost(self, predictions, targets, derivative=False):
        difference = predictions - targets
        if derivative:
            return difference * 2
        return np.sum(difference ** 2)

    # trains the model using labeled data
    def train(self, inputs, outputs, batch_size=10, learning_rate=0.01, iterations=100):
        input_batches = chunks(inputs, batch_size)
        output_batches = chunks(outputs, batch_size)

        for i in range(iterations):
            for input_batch, output_batch in zip(input_batches, output_batches):
                self.train_batch(input_batch, output_batch, learning_rate)
            print("iter " + str(i + 1) + "/" + str(iterations) + " complete")

    # trains the model for one iteration on one batch
    def train_batch(self, input_batch, output_batch, learning_rate=0.01):
        weight_adjustments_batch = []
        bias_adjustments_batch = []
        for inputs, outputs in zip(input_batch, output_batch):
            data = self.forward_propagate(inputs)
            weight_gradient, bias_gradient = self.back_propagate(
                        predictions=data["outputs"],
                        targets=outputs,
                        layer_inputs=data["layer_inputs"],
                        layer_activations=data["layer_activations"])

            # gets the adjustments based on the gradient and the learning rate
            weight_adjustments = []
            bias_adjustments = []
            for k in range(len(weight_gradient)):
                weight_adjustments.append(-weight_gradient[k] * learning_rate)
                bias_adjustments.append(-bias_gradient[k] * learning_rate)

            weight_adjustments_batch.append(weight_adjustments)
            bias_adjustments_batch.append(bias_adjustments)

        mean_weight_adjustments = np.array(weight_adjustments_batch).mean(0)
        mean_bias_adjustments = np.array(bias_adjustments_batch).mean(0)

        # adjusts weights and biases
        self.weights += mean_weight_adjustments
        self.biases += mean_bias_adjustments

    # computes the gradient for the weights and biases using the back propagation algorithm
    def back_propagate(self, predictions, targets, layer_inputs, layer_activations):

        # gets the partials of the cost function with respect to the outputs
        ca = self.cost(predictions, targets, derivative=True)

        # gets the partials of the output layer activations with respect to the
        # layer inputs, which are equal to the derivative of the activation function
        current_layer_inputs = layer_inputs[self.num_layers - 2]
        az = self.activate(current_layer_inputs, derivative=True)

        # gets the partials of the output layer inputs with respect to the weights,
        # which are equal to the activations of the previous layer
        current_layer_size = len(layer_activations[self.num_layers - 1])
        previous_layer_activations = layer_activations[self.num_layers - 2]
        zw = np.tile(previous_layer_activations, (current_layer_size, 1))

        # applies the chain rule to get the partials of the cost function with
        # respect to the weights and biases of the output layer
        cw = (ca * az * zw.T).T
        cb = ca * az

        # adds the cost derivatives to the gradient
        weight_gradient = [cw]
        bias_gradient = [cb]

        for i in reversed(range(2, self.num_layers)):

            # gets the partials of the inputs of the next layer with respect to the
            # activations of the current layer, which are equal to the weights between the layers
            za = self.weights[i - 1]

            # gets the partials of the cost function with respect to the activations of
            # the current layer
            ca = (np.multiply(az, ca) * za.T).T.sum(axis=0)

            # gets the partials of the activations of the current layer with
            # respect to the inputs of the current layer, which are equal to the
            # derivative of the activation function
            current_layer_inputs = layer_inputs[i - 2]
            az = self.activate(current_layer_inputs, derivative=True)

            # gets the partials of the output layer inputs with respect to the weights,
            # which are equal to the activations of the previous layer
            current_layer_size = len(layer_activations[i - 1])
            previous_layer_activations = layer_activations[i - 2]
            zw = np.tile(previous_layer_activations, (current_layer_size, 1))

            # applies the chain rule to get the partials of the cost function with
            # respect to the weights and biases of the current layer
            cw = (ca * az * zw.T).T
            cb = ca * az

            # adds the cost derivatives to the gradient
            weight_gradient.append(cw)
            bias_gradient.append(cb)

        # reverse the gradient lists to match the order of the weights and biases
        weight_gradient.reverse()
        bias_gradient.reverse()

        return np.array(weight_gradient), np.array(bias_gradient)

    def save_model(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, "biases.npy"), self.biases)
        np.save(os.path.join(directory, "weights.npy"), self.weights)
