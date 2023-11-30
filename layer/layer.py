import numpy as np
import pandas as pd
from .weight_initialization import xavier_initialization, normalized_xavier_initialization, he_initialization
from batch_normalization.batch_normalization import BatchNormalization

'''
Base class for dense layers.

To do: Need to change np.zeros to random values
'''


class LayerDense:
    def __init__(self, n_inputs, n_neurons, activation_function, random_bias = False):
        self.neurons = n_neurons
        # Weight initialization with correspond method
        if activation_function == 'relu':
            self.weights = he_initialization(n_inputs, n_neurons)
        else:
            self.weights = xavier_initialization(n_inputs, n_neurons)
        
        # Decide if bias is only 0 or initialized to random values
        if random_bias:
            self.biases = np.random.rand(1, n_neurons)
        else:
            self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        # Check for pandas data type, convert to numpy if data is Series or DataFrame
        if isinstance(inputs, pd.Series) or isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_numpy()

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
        
    def backward(self, dvalues):

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)