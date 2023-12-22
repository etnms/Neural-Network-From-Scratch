import pprint
import random 
import numpy as np
from model.model import Model
from layer.create_layer import CreateLayer


class RandomSearch:
    def __init__(self, hyperpatameter_ranges):
        self.hyperparameter_ranges = hyperpatameter_ranges
        self.results = []

    def random_search(self, num_random_samples: int, number_classes: int, training_set_X, training_set_y, testing_set_X, testing_set_y, 
                    early_stopping, early_stopping_patience):

        for _ in range(num_random_samples):
            model = Model()
            layer1 = CreateLayer.create(number_classes=number_classes, number_neurons=32, activation_function='relu')
            layer2 = CreateLayer.create(number_classes=32, number_neurons=16, activation_function='relu')
            layer3 = CreateLayer.create(number_classes=16, number_neurons=5, activation_function='softmax')
            layers=[layer1, layer2, layer3]

            layer_dense_list = [layer[0] for layer in layers]
            activation_layer_list = [layer[1] for layer in layers]
            dropout_layer_list = [layer[2] for layer in layers]

            learning_rate = random.uniform(*self.hyperparameter_ranges['learning_rate'])
            num_epochs = random.randint(*self.hyperparameter_ranges['number_epochs'])
            batch_size = random.randint(*self.hyperparameter_ranges['batch_size'])
            #hidden_layers = random.randint(*hyperparameter_ranges['hidden_layers'])
            #neurons_per_layer = random.randint(*hyperparameter_ranges['neurons_per_layer'])
            #activation = random.choice(hyperparameter_ranges['activation'])


            model.train_model(layers = layer_dense_list, activations=activation_layer_list, dropouts=dropout_layer_list,
                        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,data_X=training_set_X,
                        data_y=training_set_y, training=True, early_stopping=early_stopping, 
                        early_stopping_patience=early_stopping_patience)

            predictions = model.testing_model(layers=layer_dense_list, activations=activation_layer_list, data_X=testing_set_X)
            predicted_classes = np.argmax(predictions, axis=1)

            accuracy = np.mean(predicted_classes == testing_set_y)
            print(f"Test accuracy: {accuracy}")
            # Store or print the results as needed
            self.results.append({'learning_rate': learning_rate,  'number_epochs' : num_epochs, 'batch_size' : batch_size,
                            'accuracy': accuracy})

        pprint.pprint(self.results)