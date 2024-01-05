import pprint
import random 
import numpy as np
from model.model import Model
from layer.create_modular_layers import ModularLayer
import json
from utils.utils import convert_to_python_types
import os, sys


class RandomSearch:
    def __init__(self, hyperpatameter_ranges):
        self.hyperparameter_ranges = hyperpatameter_ranges
        self.best_model = None
        self.best_accuracy = 0
        self.results = []

    def random_search(self, num_random_samples: int, number_features: int, number_classes: int, training_set_X, training_set_y, testing_set_X, testing_set_y, 
                    early_stopping, early_stopping_patience):

        for _ in range(num_random_samples):
            # Generate a number of hidden layers
            hidden_layers = random.randint(*self.hyperparameter_ranges['hidden_layers']) #number of hidden layers
            # Create list for the number of features, starts with the value of the sample which cannot be random
            number_features_list = [number_features]
            # For each hidden layer, generate a random number of neurons which will determined the number of features of the next layer
            # since this number follows the number of neurons of the previous layer
            for _ in range(hidden_layers):
                random_n_neurons = random.randint(*self.hyperparameter_ranges['neurons_per_layer'])
                random_number = random.randint(1, random_n_neurons)
                number_features_list.append(random_number)

            # Keep track of the number of neurons by creating a list
            # Copy the number of features since they are the same except for the first one
            # The first number (number of classes of the sample) is removed (pop(0)), and the last number correspond to number of classes of data sample
            number_neurons_list = number_features_list.copy()
            number_neurons_list.pop(0)
            number_neurons_list.append(number_classes)
            # Generate a list of activation functions based on the number of hidden layers (one per layer)
            activation_functions = [f'activation{i}' for i in range(1, hidden_layers + 2)] 
            # +2 in range to account for 1 more needed for each layer since includes output one, and 1 as base list loop (1 + 1)

            # Create empty dictionary to store values
            values_dict = {}

            for function_name in activation_functions:
                values_dict[function_name] = random.choice(self.hyperparameter_ranges['activation'])

            # Array of the actual activation function names (not just activation1, activation2, etc.)
            activation_functions_names = []
            # Accessing the values using the variable names as keys in the dictionary
            for function_index, function_name in values_dict.items():
                activation_functions_names.append(function_name)

            # Other hyperparameterrs
            learning_rate = random.uniform(*self.hyperparameter_ranges['learning_rate'])
            num_epochs = random.randint(*self.hyperparameter_ranges['number_epochs'])
            batch_size = random.randint(*self.hyperparameter_ranges['batch_size'])

            layers = ModularLayer.create_modular_layers(number_features_list, number_neurons_list, activation_functions_names)

            regularization = None
            loss_function_used = None
            training = True

            model = Model(layers)
            model.train_model(num_epochs, batch_size, learning_rate,training_set_X,
                    training_set_y, training, loss_function_used, early_stopping, 
                    early_stopping_patience, regularization)

            predictions = model.testing_model(data_X=testing_set_X)
            predicted_classes = np.argmax(predictions, axis=1)

            # Min class labels for dataset that don't start at 0 since they will run an index error
            min_class_label = np.min(testing_set_y)
            accuracy = np.mean(predicted_classes == (testing_set_y - min_class_label))
            print(f"Test accuracy: {accuracy}")
            # Store or print the results as needed
            self.results.append({'learning_rate': learning_rate,  'number_epochs' : num_epochs, 'batch_size' : batch_size,
                            'accuracy': accuracy, 'activations' : activation_functions_names, 'n_hidden_layers': hidden_layers,
                            'n_neurons_per_layer': number_neurons_list})
            
            # Check what the best model is, if new model is better than update the values
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = {'learning_rate': learning_rate, 'num_epochs': num_epochs,
                                   'batch_size': batch_size, 'activations': activation_functions_names,
                                   'n_hidden_layers': hidden_layers, 'n_neurons_per_layer': number_neurons_list, 'accuracy': accuracy}

        pprint.pprint(self.results)

        if self.best_model:
            root_current_project = os.path.dirname(sys.modules['__main__'].__file__)
            with open(f'{root_current_project}/best_model.json', 'w') as json_file:
                json.dump(self.best_model, json_file, indent=2)
