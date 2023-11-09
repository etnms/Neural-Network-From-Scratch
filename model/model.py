from loss import LossCategoricalCrossentropy
import numpy as np
import pandas as pd
from dropout import dropout

'''
Model class with train and testing functions. 
It also has a forward_backward function for the training, while the testing only uses a forward pass as it doesn't require
to change the model.

To implement: Save and load functions to save and load a model that has been created.
'''


class Model:
    # def forward_and_backward_pass(layers, activations, layer1, activation1, layer2, activation2, data_X, data_y, learning_rate):
    def forward_and_backward_pass(self, layers, activations, dropouts, data_X, data_y, learning_rate, apply_dropout, training):
        x = data_X
        
        # Forward pass
        for layer, activation, dropout in zip(layers, activations, dropouts):
            layer.forward(x)
            activation.forward(layer.output)
            x = activation.output

            if apply_dropout and dropout is not None:
                dropout.forward(x, training)
                x = dropout.output
        # Calculate the loss
        loss_function = LossCategoricalCrossentropy()
        loss = loss_function.calculate(x, data_y)

        # Backward pass
        loss_function.backward(x, data_y) #activations[-1].output = x, could be used as well since starting from last
        dvalues = loss_function.dvalues

        for layer, activation in zip(reversed(layers), reversed(activations)):
            activation.backward(dvalues)
            layer.backward(activation.dvalues)
            dvalues = layer.dvalues

        # Optimization step
        for layer in layers:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases

        return loss, x

    def train_model(self, layers, activations, dropouts, num_epochs, batch_size, learning_rate, data_X, data_y, training):
        data_size = len(data_X)

        # Check if dropout values have been used
        apply_dropout = self.check_for_dropout(dropouts)

        for epoch in range(num_epochs):
            total_loss = 0
            all_predictions = []
            for i in range(0, data_size, batch_size):
                if isinstance(data_X, pd.Series) or isinstance(data_X, pd.Series):
                    batch_X = data_X[i:i+batch_size].values.reshape(-1, 1)  # Reshape to 2D array if data_X is 1D (pandas series is 1d)
                else:
                    batch_X = data_X[i:i+batch_size]
                batch_y = data_y[i:i+batch_size]
                batch_loss, batch_predictions = self.forward_and_backward_pass(layers, activations, dropouts, batch_X, batch_y, learning_rate, apply_dropout, training)
                total_loss += batch_loss
                all_predictions.append(batch_predictions)

            # Combine predictions from all batches
            predictions = np.vstack(all_predictions)
            
            # Calculate and print the average loss and accuracy for this epoch
            average_loss = total_loss / (data_size / batch_size)
            accuracy = np.mean(np.argmax(predictions, axis=1) == data_y) # Assuming data_y represents true class labels
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}, Accuracy: {accuracy}')

    def check_for_dropout(self, dropouts):
        if dropouts is None:
            return False
        else:
            return True

    def testing_model(self, layers, activations, data_X):
        x = data_X

        # Forward pass only for testig. We are not trying to update the model but simply test its acuracy
        for layer, activation in zip(layers, activations):
            layer.forward(x)
            activation.forward(layer.output)
            x = activation.output
        
        return x
    

'''
def save_model(self):
    model_data = {
    'layer1_weights': self.layer1.weights,
    'layer1_biases': self.layer1.biases,
    'layer2_weights': self.layer2.weights,
    'layer2_biases': self.layer2.biases,
    'activation1_state': self.activation1.__dict__,
    'activation2_state': self.activation2.__dict__,
    }
    print(model_data)
'''