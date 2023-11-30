import numpy as np


'''
Batch normalization class. Implements batch normalization (should be apply before layer activation).

This class does not currently work. There is a shape mismatch between neurons and inputs in the forward method. 
Batch normalization is initialized with number of neurons but the forward method uses the input (batch size) which will always
be different.
This needs to be fixed or adapted to the network or redone in some other way.
'''


class BatchNormalization:
    def __init__(self, n_neurons, momentum=0.9, epsilon=1e-5): #n_neurons not n_inputs
        self.gamma = np.ones((1, n_neurons))  # scale parameter
        self.beta = np.zeros((1, n_neurons))  # shift parameter

        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((1, int(n_neurons)))
        self.running_variance = np.ones((1, int(n_neurons)))


    def forward(self, inputs, training=True):
        if training:
            # Compute mean and variance over the mini-batch
            mean = np.mean(inputs, axis=0, keepdims=True)
            variance = np.var(inputs, axis=0, keepdims=True)

            # Normalize input
            self.normalized_input = (inputs - mean) / np.sqrt(variance + self.epsilon)

            # Scale and shift
            print("Gamma shape:", self.gamma.shape)
            print("Beta shape:", self.beta.shape)
            print("Normalized input shape:", self.normalized_input.shape)
            outputs = self.gamma * self.normalized_input + self.beta

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * variance
        else:
            # Use running mean and variance during inference
            self.normalized_input = (inputs - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)
            outputs = self.gamma * self.normalized_input + self.beta

        return outputs

    def backward(self, dvalues):

        # Gradients for gamma and beta
        self.dgamma = np.sum(dvalues * self.normalized_input, axis=0, keepdims=True)
        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient for the normalized input
        dnormalized_input = dvalues * self.gamma

        return dnormalized_input