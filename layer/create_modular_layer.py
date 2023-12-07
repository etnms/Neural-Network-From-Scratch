import string
from .create_layer import CreateLayer

class ModularLayer:

    def create_modular_layer(list_classes: int, list_neurons: int, list_activations: string):
        layers = []
        for layer in zip(list_classes, list_neurons, list_activations):
            modular_layer = CreateLayer.create(layer[0], layer[1], layer[2])
            layers.append(modular_layer)
        return layers        