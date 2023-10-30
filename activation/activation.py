import numpy as np


'''
Activation functions
- Rectified Linear function
- Softmax function
'''


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Derivative of ReLU: 1 for positive values, 0 for non-positive values
        self.dvalues = dvalues.copy()  # Copy the input gradient
        self.dvalues[self.output <= 0] = 0  # Set gradient to 0 for non-positive input values


class ActivationSoftmax:
    def forward(self, inputs):
        #axis = 1 = row, 0 = column, default none
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # subtract the max of each batch to avoid overflow (keep values between 0 and 1)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)# normalized value will still be the same at the end
        self.output = probabilities
        
    def backward(self, dvalues):
        # Derivative of the softmax function
        self.dvalues = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dvalues[index] = np.dot(jacobian_matrix, single_dvalues)


class ActivationTanh:
    def forward(self, input):
        self.output = np.tanh(input)

    def backward(self, dvalues):
        self.dvalues = dvalues * (1 - self.output ** 2)