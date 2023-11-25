import numpy as np


'''
Batch normalization class. Implements batch normalization (should be apply before layer activation)
'''


class BatchNormalization:
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        self.gamma = np.ones((1, input_size))  # scale parameter
        self.beta = np.zeros((1, input_size))  # shift parameter
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_variance = None

    def forward(self, inputs, training=True):
        if training:
            # Compute mean and variance over the mini-batch
            mean = np.mean(inputs, axis=0, keepdims=True)
            variance = np.var(inputs, axis=0, keepdims=True)

            # Normalize input
            self.normalized_input = (inputs - mean) / np.sqrt(variance + self.epsilon)

            # Scale and shift
            outputs = self.gamma * self.normalized_input + self.beta

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * variance
        else:
            # Use running mean and variance during inference
            normalized_input = (inputs - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)
            outputs = self.gamma * normalized_input + self.beta

        return outputs

    def backward(self, dvalues):
        # Gradients for gamma and beta
        self.dgamma = np.sum(dvalues * self.normalized_input, axis=0, keepdims=True)
        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient for the normalized input
        dnormalized_input = dvalues * self.gamma

        return dnormalized_input