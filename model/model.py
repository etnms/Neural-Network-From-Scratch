from loss import LossCategoricalCrossentropy, LossMeanSquaredError
from utils.utils import convert_to_numpy_arrays, convert_to_python_types
from layer.layer import LayerDense
import numpy as np
import pandas as pd
import os
import string
import json
import sys
from PyQt6.QtWidgets import QApplication
import matplotlib.pyplot as plt
from data_visualization.plot_data import plot_data
from layer.create_modular_layers import ModularLayer
from activation import activation

'''
Model class with train and testing functions. 
It also has a forward_backward function for the training, while the testing only uses a forward pass as it doesn't require
to change the model.

'''


class Model:
    def __init__(self, layers, update_text_callback=None):
        self.layers = layers
        self.best_val_loss = float('inf') # first training will always be less than infinity
        self.no_improvement_count = 0

        # Layer element lists
        self.layer_dense_list = [layer[0] for layer in self.layers]
        self.activation_layer_list = [layer[1] for layer in self.layers]
        self.dropout_layer_list = [layer[2] for layer in self.layers]

        # ONLY for GUI logic
        self.update_text_callback= update_text_callback

    # def forward_and_backward_pass(layers, activations, layer1, activation1, layer2, activation2, data_X, data_y, learning_rate):
    def forward_and_backward_pass(self, data_X, data_y, learning_rate, apply_dropout, training, loss_function_used, regularization):
        x = data_X
        
        model_parameters = [{'weights': layer.weights, 'biases': layer.biases} for layer in self.layer_dense_list]
        # Forward pass
        for layer, activation, dropout in zip(self.layer_dense_list, self.activation_layer_list, self.dropout_layer_list):
            layer.forward(x)
            activation.forward(layer.output)
            x = activation.output

            if apply_dropout and dropout is not None:
                dropout.forward(x, training)
                x = dropout.output
        # Calculate the loss
        loss_function = loss_function_used

        loss_function.set_params(params = model_parameters)
        loss = loss_function.calculate(x, data_y, regularization) # forward pass with loss

        # Backward pass
        dvalues = loss_function.backward(x, data_y, regularization)
        for layer, activation in zip(reversed(self.layer_dense_list), reversed(self.activation_layer_list)):
            activation.backward(dvalues)
            layer.backward(activation.dvalues)
            dvalues = layer.dvalues

        # Optimization step
        for layer in self.layer_dense_list:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases

        return loss, x

    def train_model(self, num_epochs, batch_size, learning_rate, 
                    data_X, data_y, training, loss_function_used = None,
                    early_stopping = False,
                    early_stopping_patience = None, regularization = None,
                    plot_loss = False):
        
        # Default best accuracy for early stopping (saving best parameters in case of stopping)
        self.prev_best_accuracy = 0

        # Select the loss function used
        if loss_function_used == None or loss_function_used == 'CrossEntropy':
            loss_function_used = LossCategoricalCrossentropy()
        elif loss_function_used == 'MSE':
            loss_function_used = LossMeanSquaredError()
        
        data_size = len(data_X)
        
        # Check if dropout values have been used
        apply_dropout = self.check_for_dropout(self.dropout_layer_list)

        # Loss history variables for plotting purposes only
        loss_history = []
        accuracy_history = []

        for epoch in range(num_epochs):
            total_loss = 0
            all_predictions = []
            for i in range(0, data_size, batch_size):
                if isinstance(data_X, pd.Series) or isinstance(data_X, pd.Series):
                    batch_X = data_X[i:i+batch_size].values.reshape(-1, 1)  # Reshape to 2D array if data_X is 1D (pandas series is 1d)
                else:
                    batch_X = data_X[i:i+batch_size]
                batch_y = data_y[i:i+batch_size]
                batch_loss, batch_predictions = self.forward_and_backward_pass(batch_X, batch_y, learning_rate, apply_dropout, 
                                                                               training, loss_function_used, regularization)
                total_loss += batch_loss
                all_predictions.append(batch_predictions)

            # Combine predictions from all batches
            predictions = np.vstack(all_predictions)
            
            # Calculate and print the average loss and accuracy for this epoch
            average_loss = total_loss / (data_size / batch_size)
            # Min class labels for dataset that don't start at 0 since they will run an index error
            min_class_label = np.min(data_y)
            accuracy = np.mean(np.argmax(predictions, axis=1) == (data_y - min_class_label)) # Assuming data_y represents true class labels

            # Append loss and accuracy value to array for plot
            loss_history.append(average_loss)
            accuracy_history.append(accuracy)

            if epoch % 20 == 0: # Compute and print loss every X epochs
                # If using GUI application then update text in app, else update command line text
                if self.update_text_callback is not None:
                    self.update_text_callback(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}, Accuracy: {accuracy}')
                    QApplication.processEvents()
                else:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}, Accuracy: {accuracy}')

                if early_stopping:
                    if average_loss < self.best_val_loss:
                        self.best_val_loss = average_loss
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1

                    # Early stopping saving best model. Check for acuracy, if higher then update which parameters are good
                    if accuracy > self.prev_best_accuracy:
                        self.prev_best_accuracy = accuracy
                        self.save_model(self.layers, early_stopping, early_stopping_patience, regularization,
                                        name= '/early_stopping_outputs/best_early_stopping')

                    if self.no_improvement_count >= early_stopping_patience:
                        print(f'Early stopping at epoch {epoch + 1} as there is no improvement in validation loss.')
                        break

        if plot_loss:
            plot_data(loss_history,'Training Loss', 'Epoch', 'Loss', 'Training Loss Over Epochs')
            plot_data(accuracy_history,'Training Accuracy', 'Epoch', 'Accuracy', 'Training Accuracy Over Epochs')


    def check_for_dropout(self, dropouts):
        if dropouts is None:
            return False
        else:
            return True

    def testing_model(self, data_X):
        x = data_X

        # Forward pass only for testing the model. We are not trying to update the model but simply test its acuracy
        for layer, activation in zip(self.layer_dense_list, self.activation_layer_list):
            layer.forward(x)
            activation.forward(layer.output)
            x = activation.output
        
        return x
    
    def save_model(self, layers, early_stopping, early_stopping_patience, regularization, name: string):
        model_data = {}

        for i, layer in enumerate(layers):
            model_data[f'layer{i+1}_n_neurons'] = layer[0].n_neurons
            model_data[f'layer{i+1}_n_inputs'] = layer[0].n_inputs
            model_data[f'layer{i+1}_weights'] = layer[0].weights
            model_data[f'layer{i+1}_biases'] = layer[0].biases
            #model_data[f'layer{i+1}_activation_param'] = layer[1].__dict__
            model_data[f'layer{i+1}_activation_name'] = layer[1].__class__.__name__
            if layer[2] is not None:
                model_data[f'layer{i+1}_dropout'] = layer[2].__dict__
        model_data[f'early_stopping'] = early_stopping
        model_data[f'early_stopping_patience'] = early_stopping_patience
        model_data[f'regularization'] = regularization

        # Get root of directory    
        root_current_project = os.path.dirname(sys.modules['__main__'].__file__)

        with open(f'{root_current_project}/{name}.json', mode='w') as output:
            # Remove previous content if any
            output.truncate()
            # Write to json
            content = json.dumps(model_data, default=convert_to_python_types)
            output.write(content)
      

    def load_model(self, name: str):
    # Get root of directory
        root_current_project = os.path.dirname(sys.modules['__main__'].__file__)

        with open(f'{root_current_project}/{name}', mode='r') as input_file:
            model_data = json.load(input_file)

        layers = []

        for i in range(1, (len(model_data) - 3) // 5 + 1):
            # Load layer data
            n_neurons = model_data[f'layer{i}_n_neurons']
            n_inputs = model_data[f'layer{i}_n_inputs']
            weights = model_data[f'layer{i}_weights']
            biases = model_data[f'layer{i}_biases']

            # Create layer instance
            layer_dense = LayerDense(n_inputs, n_neurons, activation_function=None)
            layer_dense.weights = weights
            layer_dense.biases = biases

            # Load activation function
            activation_function = model_data[f'layer{i}_activation_name']

            if activation_function == 'ActivationReLU':
                activation_layer = activation.ActivationReLU()
            elif activation_function == 'ActivationSoftmax':
                activation_layer = activation.ActivationSoftmax()
            elif activation_function == 'ActivationTanh':
                activation_layer = activation.ActivationTanh()
            elif activation_function == 'ActivationSigmoid':
                activation_layer = activation.ActivationSigmoid()
            else:
                activation_layer = None

            layers.append((layer_dense, activation_layer, None))

        return layers