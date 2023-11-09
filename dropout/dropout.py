import numpy as np

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
        self.output = None
    # Training needs to be set to false when testing the model (true for training purposes)
    def forward(self, inputs, training):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
            self.output = inputs * self.mask
        else:
            return inputs

    def backward(self, d_values):
        # The gradient is simply the input values that were not dropped out during the forward pass.
        return d_values * self.mask