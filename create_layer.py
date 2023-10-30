import string
from layer import layer
from activation import activation


class CreateLayer:
    def create(number_classes: int, number_neurons: int, activation_function: string):
        layer_dense = layer.LayerDense(number_classes, number_neurons)

        match activation_function:
            case 'relu':
                activation_layer = activation.ActivationReLU()
            case 'softmax':
                activation_layer = activation.ActivationSoftmax()
            case _:
                pass
            
        return layer_dense, activation_layer