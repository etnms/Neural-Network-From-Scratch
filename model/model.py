class Model:
    def __init__(self, layer1, layer2, activation1, activation2):
        self.layer1 = layer1
        self.layer2 = layer2
        self.activation1 = activation1
        self.activation2 = activation2

    def save_model(self, weights, biases):
        model_data = {
        'layer1_weights': self.layer1.weights,
        'layer1_biases': self.layer1.biases,
        'layer2_weights': self.layer2.weights,
        'layer2_biases': self.layer2.biases,
        'activation1_state': self.activation1.__dict__,
        'activation2_state': self.activation2.__dict__,
        }
        print(model_data)