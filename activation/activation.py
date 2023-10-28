import numpy as np


'''
Activation functions
- Rectified Linear function
- Softmax function
'''


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def forward(self, inputs):
        #axis = 1 = row, 0 = column, default none
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # subtract the max of each batch to avoid overflow (keep values between 0 and 1)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)# normalized value will still be the same at the end
        self.output = probabilities