import numpy as np
import pandas as pd
from loss import LossCategoricalCrossentropy


'''
Epoch functions. This needs to be updated in the future to be clearer.
Hard coded values for the layers and activation -> this also needs to be updated.
'''

# def forward_and_backward_pass(layers, activations, layer1, activation1, layer2, activation2, data_X, data_y, learning_rate):
def forward_and_backward_pass(layers, activations, data_X, data_y, learning_rate):
    x = data_X

    # Forward pass
    for layer, activation in zip(layers, activations):
        layer.forward(x)
        activation.forward(layer.output)
        x = activation.output

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

def epoch_function(layers, activations, num_epochs, batch_size, learning_rate, data_X, data_y):
    data_size = len(data_X)

    for epoch in range(num_epochs):
        total_loss = 0
        all_predictions = []
        for i in range(0, data_size, batch_size):
            if isinstance(data_X, pd.Series) or isinstance(data_X, pd.Series):
                batch_X = data_X[i:i+batch_size].values.reshape(-1, 1)  # Reshape to 2D array if data_X is 1D (pandas series is 1d)
            else:
                batch_X = data_X[i:i+batch_size]
            batch_y = data_y[i:i+batch_size]
            batch_loss, batch_predictions = forward_and_backward_pass(layers, activations, batch_X, batch_y, learning_rate)
            total_loss += batch_loss
            all_predictions.append(batch_predictions)

        # Combine predictions from all batches
        predictions = np.vstack(all_predictions)
        
        # Calculate and print the average loss and accuracy for this epoch
        average_loss = total_loss / (data_size / batch_size)
        accuracy = np.mean(np.argmax(predictions, axis=1) == data_y) # Assuming data_y represents true class labels
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}, Accuracy: {accuracy}')