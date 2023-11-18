import string
from layer import layer
from activation import activation
from dropout import dropout

class CreateLayer:
    def create(number_classes: int, number_neurons: int, activation_function: string, dropout_rate: float = None):
        # Create layer
        layer_dense = layer.LayerDense(number_classes, number_neurons, activation_function)

        # Create activation function
        match activation_function:
            case 'relu':
                activation_layer = activation.ActivationReLU()
            case 'softmax':
                activation_layer = activation.ActivationSoftmax()
            case 'tanh':
                activation_layer = activation.ActivationTanh()
            case 'sigmoid':
                activation_layer = activation.ActivationSigmoid()
            case _:
                pass

        # If there is a dropout rate, create a dropout layer
        if (dropout_rate):
            # First make sure the value is correct
            if not (0 <= dropout_rate <= 1):
                raise ValueError("Rate must be between 0 and 1.")
            
            dropout_layer = dropout.Dropout(dropout_rate)
        else:
            dropout_layer = None
            
        return layer_dense, activation_layer, dropout_layer