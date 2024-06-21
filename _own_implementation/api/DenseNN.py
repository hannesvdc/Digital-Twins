import jax.numpy as np

import api.Functional as functional

class DenseNN:
    def __init__(self, layers=[], activation=functional.relu):
        self.n_layers = len(layers) - 1 # The first layer determines the size of the input
        self.n_weights = np.sum(np.array(list(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))))
        self.layers = layers
        self.activation = activation

    # Input 'data' must have size (Input_Size, N_data)
    def forward(self, data, weights):
        x = data

        weights_index = 0
        for i in range(self.n_layers):
            # Extract Weights and Biases from weights vector
            input_dim = self.layers[i]
            output_dim = self.layers[i+1]
            W = np.reshape(weights[weights_index:weights_index + input_dim * output_dim], (output_dim, input_dim))
            b = weights[weights_index + input_dim * output_dim : weights_index + input_dim * output_dim + output_dim]

            # Do forward propagation
            x = np.dot(W, x) + b[:,np.newaxis]
            if i != self.n_layers - 1: # Do not apply activation in last layer
                x = self.activation(x)

            # Variable upkeeping for the next iteration
            weights_index += input_dim * output_dim + output_dim

        return x