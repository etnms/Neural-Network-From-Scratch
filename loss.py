import numpy as np
import pandas as pd
import string
'''
Loss classes. Loss class as base and Categorical Cross entropy derived from it
'''

class Loss:
    def calculate(self, output, y):
        # Check for pandas data type, convert to numpy if data is Series or DataFrame
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif isinstance(y, pd.DataFrame):
            y = y.to_numpy()
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    

class LossCategoricalCrossentropy(Loss):
    def __init__(self, lambda_reg=0.01): # lambda = regularization strength
        self.lambda_reg = lambda_reg
        self.params = None  # Store the model parameters for regularization

    def set_params(self, params):
        self.params = params
    
    def calculate_total_weights(self):
        # Create a 1D array containing all the weights of the model
        total_weights = np.concatenate([layer_params['weights'].flatten() for layer_params in self.params])
        
        return total_weights
    
    def forward(self, y_pred, y_true):
        samples = len(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # clip to avoid infinity problem

        if len(y_true.shape) == 1: # check if scalar values have been passed (shape of array = 1 dim)
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # check if one encoded vector  (shape of array = 2 dim)
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # sum on axis 1 = row

        if self.params is None:
            raise ValueError("Model parameters have not been set. Call set_params before forward.")
        
        total_weights = self.calculate_total_weights()
        l1_regularization = self.lambda_reg * np.sum(np.abs(total_weights)) / samples
        l2_regularization = 0.5 * self.lambda_reg * np.sum(total_weights**2) / samples
        
        negative_log_likelihoods = -np.log(correct_confidences) + l2_regularization
        return negative_log_likelihoods

    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues) #number of rows in the dvalues matrix

        # Calculate the gradient
        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues /= samples
              
        total_weights = self.calculate_total_weights()
        # L1 regularization gradient
        l1_gradient = self.lambda_reg * np.sign(total_weights) / samples
        # L2 regularization gradient
        l2_gradient = self.lambda_reg * total_weights / samples 
        l2_gradient = l2_gradient.reshape((1, -1)) # reshape regularization array tp match shape
        self.dvalues += l2_gradient[:, :self.dvalues.shape[1]] # Broadcast to match shape of dvalues (still needed after reshape)


class LossMeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            # Reshape, first convert to np array
            y_true = np.array(y_true).reshape(-1, 1)

        # Calculate MSE  
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1) 
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

         # Ensure y_true is a column vector
        if len(y_true.shape) == 1:
            # Reshape, first convert to np array
            y_true = np.array(y_true).reshape(-1, 1)

        # Gradient of the mean squared error with respect to the predicted values
        self.dvalues = -2 * (y_true - dvalues) / samples