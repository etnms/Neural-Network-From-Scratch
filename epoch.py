import numpy as np
from loss import LossCategoricalCrossentropy
from activation import activation
from layer import layer

# Forward pass 
# Temporary hardcoding for testing purposes

dense1 = layer.LayerDense(2, 512) #2 because 100, 3 from dataset and 3 can be whatever
activation1 = activation.ActivationReLU()

dense2 = layer.LayerDense(512, 3) #3 because previous one is 3 (dense1)
activation2 = activation.ActivationSoftmax()

def forward_and_backward_pass(layer1, activation1, layer2, activation2, data_X, data_y, learning_rate):
    # Forward pass
    layer1.forward(data_X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # Calculate the loss
    loss_function = LossCategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, data_y)

    # Backward pass
    loss_function.backward(activation2.output, data_y)
    activation2.backward(loss_function.dvalues)
    layer2.backward(activation2.dvalues)
    activation1.backward(layer2.dvalues)
    layer1.backward(activation1.dvalues)

    # Optimization step
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases

    return loss, activation2.output

def epoch_function(num_epochs, batch_size, learning_rate, data_X, data_y):
    data_size = len(data_X)

    for epoch in range(num_epochs):
        total_loss = 0
        all_predictions = []
        for i in range(0, data_size, batch_size):
            batch_X = data_X[i:i+batch_size]
            batch_y = data_y[i:i+batch_size]
            batch_loss, batch_predictions = forward_and_backward_pass(dense1, activation1, dense2, activation2, batch_X, batch_y, learning_rate)
            total_loss += batch_loss
            all_predictions.append(batch_predictions)

        # Combine predictions from all batches
        predictions = np.vstack(all_predictions)
        
        # Calculate and print the average loss and accuracy for this epoch
        average_loss = total_loss / (data_size / batch_size)
        accuracy = np.mean(np.argmax(predictions, axis=1) == data_y) # Assuming data_y represents true class labels
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}, Accuracy: {accuracy}')


'''
# without batch
import numpy as np
from loss import LossCategoricalCrossentropy
from activation import activation
from layer import layer

# Forward pass 
# Temporary hardcoding for testing purposes

dense1 = layer.LayerDense(2, 512) #2 because 100, 3 from dataset and 3 can be whatever
activation1 = activation.ActivationReLU()

dense2 = layer.LayerDense(512, 3) #3 because previous one is 3 (dense1)
activation2 = activation.ActivationSoftmax()

def epoch_function(num_epochs:int, learning_rate: float, data_X, data_y):
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        dense1.forward(data_X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Calculate the loss
        loss_function = LossCategoricalCrossentropy()
        loss = loss_function.calculate(activation2.output, data_y)

        # Backward pass
        loss_function.backward(activation2.output, data_y)
        activation2.backward(loss_function.dvalues)
        dense2.backward(activation2.dvalues)
        activation1.backward(dense2.dvalues)
        dense1.backward(activation1.dvalues)

        # Optimization step
        dense1.weights -= learning_rate * dense1.dweights
        dense1.biases -= learning_rate * dense1.dbiases
        dense2.weights -= learning_rate * dense2.dweights
        dense2.biases -= learning_rate * dense2.dbiases

        # Calculate and print the loss and accuracy for this epoch
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == data_y)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Accuracy: {accuracy}')

'''