import numpy as np
from model.model import Model
from split_training_data import split_training_data
from random_search import RandomSearch
from dataset import generate_spiral_set
from layer.create_modular_layers import ModularLayer
from text_processing.text_processing import create_tokenizer, pad_sentences
import argparse
from load_csv_data import load_csv_data

'''
File for testing purposes
'''

X, y, number_features, number_classes = load_csv_data('./dataset/winequality-red.csv')

#X, y = generate_spiral_set.create_data(100, 3)

training_set_X, training_set_y, testing_set_X, testing_set_y = split_training_data(X, y, training_size=0.8)

# Hyperparameters
learning_rate = 0.01
batch_size = 32
num_epochs = 1000  # Specify the number of training epochs
early_stopping = True
early_stopping_patience = 5 # Stop training if validation loss does not improve for 5 consecutive epochs

regularization = None
loss_function_used = None # default to cross entropy
training = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random search')
    parser.add_argument('random_search', type=str, help='Applies random search')
    args = parser.parse_args()

    if args.random_search.lower() == 'false':
        layers = ModularLayer.create_modular_layers([number_features, 128, 128], [128, 128, number_classes], ['sigmoid', 'sigmoid', 'softmax'])
        model = Model(layers)
        model.train_model(num_epochs, batch_size, learning_rate, training_set_X,
                        training_set_y, training, loss_function_used, early_stopping, 
                        early_stopping_patience, regularization)

        # Save model
        model.save_model(layers, early_stopping, early_stopping_patience, regularization, name='model')

        # Load model
        predictions = model.testing_model(data_X=testing_set_X)
        predicted_classes = np.argmax(predictions, axis=1)
        # Min class labels for dataset that don't start at 0 since they will run an index error
        min_class_label = np.min(testing_set_y)
        accuracy = np.mean(predicted_classes == (testing_set_y - min_class_label))
        print(f"Test accuracy: {accuracy}")
    else:
        hyperparameter_ranges = {
                'learning_rate': (0.001, 0.1),
                'hidden_layers': (1, 10),
                'number_epochs': (20, 300),
                'batch_size': (12, 128),
                'neurons_per_layer': (64, 512),
                'activation': ['relu', 'tanh', 'sigmoid', 'softmax'],
                }

        random_search = RandomSearch(hyperparameter_ranges)

        random_search.random_search(1, number_features, number_classes, training_set_X, training_set_y, testing_set_X, testing_set_y, 
                                    early_stopping=True, early_stopping_patience=5)
