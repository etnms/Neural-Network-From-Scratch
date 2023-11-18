import numpy as np

# Xavier weight initialization
# Is used with Tanh, Sigmoid, or Softmax (is often the default initialization method)
def xavier_initialization(n_inputs, n_neurons): #input from previous layer
    # calculate the range for the weights
    lower, upper = -(1.0 / np.sqrt(n_inputs)), (1.0 / np.sqrt(n_inputs))
    # generate random numbers
    numbers = np.random.randn(n_inputs, n_neurons)
    # scale to the desired range
    scaled = lower + numbers * (upper - lower)
    return scaled


# Normalized xavier initialization 
def normalized_xavier_initialization(n_inputs, n_neurons):
    # calculate the range for the weights
    lower, upper = -(np.sqrt(6.0) / np.sqrt(n_inputs + n_neurons)), (np.sqrt(6.0) / np.sqrt(n_inputs + n_neurons))
    # generate random numbers
    numbers = np.random.rand(n_inputs, n_neurons)
    # scale to the desired range
    scaled = lower + numbers * (upper - lower)
    return scaled


# He weight initialization
# Is used with relu
def he_initialization(n_inputs, n_neurons):
    # calculate the range for the weights
    std = np.sqrt(2.0 / n_inputs)
    # generate random numbers
    numbers = np.random.randn(n_inputs, n_neurons)
    # scale to the desired range
    scaled = numbers * std
    return scaled