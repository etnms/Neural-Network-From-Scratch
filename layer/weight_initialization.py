from math import sqrt
from numpy import mean
from numpy.random import rand, randn

# Xavier weight initialization
# Is used with Tanh, Sigmoid, or Softmax (is often the default initialization method)
def xavier_initialization(n_inputs, n_neurons): #input from previous layer
    # calculate the range for the weights
    lower, upper = -(1.0 / sqrt(n_inputs)), (1.0 / sqrt(n_inputs))
    # generate random numbers
    numbers = randn(n_inputs, n_neurons)
    # scale to the desired range
    scaled = lower + numbers * (upper - lower)
    return scaled


# Normalized xavier initialization 
def normalized_xavier_initialization(n_neurons, num_neurons_next_layer):
    # calculate the range for the weights
    lower, upper = -(sqrt(6.0) / sqrt(n_neurons + num_neurons_next_layer)), (sqrt(6.0) / sqrt(n_neurons + num_neurons_next_layer))
    # generate random numbers
    numbers = rand(1000)
    # scale to the desired range
    scaled = lower + numbers * (upper - lower)


# He weight initialization
# Is used with relu
def he_initialization(n_neurons):
    # calculate the range for the weights
    std = sqrt(2.0 / n_neurons)
    # generate random numbers
    numbers = randn(1000)
    # scale to the desired range
    scaled = numbers * std