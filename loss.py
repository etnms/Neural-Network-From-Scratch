import numpy as np
import pandas as pd


class Loss:
    def __init__(self, lambda_reg=0.01): 
        """
        Base class for loss functions.

        Parameters:
        - lambda_reg (float): Regularization strength.
        """
        self.lambda_reg = lambda_reg
        self.params = None  # Store the model parameters for regularization

    def set_params(self, params):
        self.params = params

    def calculate(self, output, y, regularization):
        """
        Calculate the loss.

        Parameters:
        - output (numpy.ndarray): Predicted values.
        - y (pandas.Series or pandas.DataFrame): True values.
        - regularization (str or None): Regularization type.

        Returns:
        - float: Calculated loss.
        """
        # Check for pandas data type, convert to numpy if data is Series or DataFrame
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        # Calculate loss
        sample_losses = self.forward(output, y, regularization)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def calculate_total_weights(self):
        # Create a 1D array containing all the weights of the model
        total_weights = np.concatenate([layer_params['weights'].flatten() for layer_params in self.params])
        
        return total_weights

class LossCategoricalCrossentropy(Loss):
    def calculate_regularization_term(self, total_weights, samples, regularization):
        if self.lambda_reg == 0.0:
            return 0.0  # No regularization

        if regularization == 'l1':
            return self.lambda_reg * np.sum(np.abs(total_weights)) / samples
        elif regularization == 'l2':
            return 0.5 * self.lambda_reg * np.sum(total_weights ** 2) / samples
        else:
            raise ValueError("Invalid regularization type")
        
    def forward(self, y_pred, y_true, regularization = None):
        """
        Calculate categorical cross-entropy loss.

        Parameters:
        - y_pred (numpy.ndarray): Predicted probabilities.
        - y_true (numpy.ndarray): True labels.
        - regularization (str or None): Regularization type.

        Returns:
        - numpy.ndarray: Array of negative log-likelihoods.
        """
        samples = len(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # clip to avoid infinity problem
        # Preprocess y_true if needed
        #y_true_numeric = [float(value) for value in y_true]
        #y_true_indices = np.array([value if not np.isnan(value) else 0 for value in y_true_numeric], dtype=int)

        if len(y_true.shape) == 1: # check if scalar values have been passed (shape of array = 1 dim)
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2: # check if one-hot encoded vector  (shape of array = 2 dim)
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # sum on axis 1 = row
        if self.params is None:
            raise ValueError("Model parameters have not been set. Call set_params before forward.")
        
        # total weights for regularization
        if regularization is not None:
            total_weights = self.calculate_total_weights()
            regularization_term = self.calculate_regularization_term(total_weights, len(y_true), regularization)  
            negative_log_likelihoods = -np.log(correct_confidences) + regularization_term
        else:
            negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    
    def backward(self, dvalues, y_true, regularization = None):
        """
        Backward pass to calculate gradients.

        Parameters:
        - dvalues (numpy.ndarray): Gradients of the loss.
        - y_true (numpy.ndarray): True labels.
        - regularization (str or None): Regularization type.
        
        Returns:
        - numpy.ndarray: Gradients of the loss with respect to the input values.
        """
        # Number of samples
        samples = len(dvalues) #number of rows in the dvalues matrix

        # Calculate the gradient
        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues /= samples
        
        if regularization is not None:
            total_weights = self.calculate_total_weights()
            regularization_gradient = self.calculate_regularization_term(total_weights, samples, regularization)
            regularization_gradient = regularization_gradient.reshape((1, -1)) # reshape regularization array tp match shape
            self.dvalues += regularization_gradient[:, :self.dvalues.shape[1]] # Broadcast to match shape of dvalues (still needed after reshape)
        return self.dvalues

class LossMeanSquaredError(Loss):
    def forward(self, y_pred, y_true, regularization=None):
        """
        Calculate mean squared error loss.

        Parameters:
        - y_pred (numpy.ndarray): Predicted values.
        - y_true (numpy.ndarray): True values.
        - regularization (str or None): Regularization type.

        Returns:
        - numpy.ndarray: Array of sample losses.
        """
        if len(y_true.shape) == 1:
            # Reshape, first convert to np array
            y_true = np.array(y_true).reshape(-1, 1)

        # Calculate MSE
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)

        # Regularization term
        if regularization is not None:
            total_weights = self.calculate_total_weights()

            # L1 regularization
            if regularization == 'l1':
                regularization_term = self.lambda_reg * np.sum(np.abs(total_weights)) / len(y_true)
            # L2 regularization
            elif regularization == 'l2':
                regularization_term = 0.5 * self.lambda_reg * np.sum(total_weights ** 2) / len(y_true)
            else:
                regularization_term = 0.0  # No regularization

            sample_losses += regularization_term

        return sample_losses

    def backward(self, dvalues, y_true, regularization=None):
        """
        Backward pass to calculate gradients for mean squared error.

        Parameters:
        - dvalues (numpy.ndarray): Gradients of the loss.
        - y_true (numpy.ndarray): True values.
        - regularization (str or None): Regularization type.

        Returns:
        - numpy.ndarray: Gradients of the loss with respect to the input values.
        """
        samples = len(dvalues)

        # Ensure y_true is a column vector
        if len(y_true.shape) == 1:
            # Reshape, first convert to np array
            y_true = np.array(y_true).reshape(-1, 1)

        # Gradient of the mean squared error with respect to the predicted values
        self.dvalues = -2 * (y_true - dvalues) / samples

        # Regularization term in the gradient
        if regularization is not None:
            total_weights = self.calculate_total_weights()

            # L1 regularization gradient
            if regularization == 'l1':
                regularization_gradient = self.lambda_reg * np.sign(total_weights) / samples
            # L2 regularization gradient
            elif regularization == 'l2':
                regularization_gradient = self.lambda_reg * total_weights / samples
            else:
                regularization_gradient = np.zeros_like(total_weights)

            regularization_gradient = regularization_gradient.reshape((1, -1))
            self.dvalues += regularization_gradient[:, :self.dvalues.shape[1]]
        return self.dvalues