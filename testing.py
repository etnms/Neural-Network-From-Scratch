import numpy as np
import pandas as pd
from model.model import Model
from create_layer import CreateLayer
from split_training_data import split_training_data
from random_search import RandomSearch
from dataset import generate_spiral_set
'''
File for testing purposes
'''


model = Model()

data = pd.read_csv('./dataset/phonemes.csv', encoding='utf-8')  

X = data[['V1', 'V2', 'V3', 'V4', 'V5']]

y = data['Class']

X#, y = generate_spiral_set.create_data(100, 3)
training_set_X, training_set_y, testing_set_X, testing_set_y = split_training_data(X, y, training_size=0.8)

number_classes = 5
#number_classes = 2

# Hyperparameters
learning_rate = 0.3
batch_size = 128
num_epochs = 200  # Specify the number of training epochs
early_stopping = True
early_stopping_patience = 5 # Stop training if validation loss does not improve for 5 consecutive epochs


# Number of classes = number of inputs
# In case of second layer until the end it takes the number of neurons from the previous layer
# Ex layer1(2,512), layer2(512,3) layer3(3, 128) etc.
layer1 = CreateLayer.create(number_classes=number_classes, number_neurons=32, activation_function='relu')
layer2 = CreateLayer.create(number_classes=32, number_neurons=16, activation_function='relu')
layer3 = CreateLayer.create(number_classes=16, number_neurons=5, activation_function='softmax')
layers=[layer1, layer2, layer3]

layer_dense_list = [layer[0] for layer in layers]
activation_layer_list = [layer[1] for layer in layers]
dropout_layer_list = [layer[2] for layer in layers]


model.train_model(layers = layer_dense_list, activations=activation_layer_list, dropouts=dropout_layer_list,
                   batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,data_X=training_set_X,
                   data_y=training_set_y, training=True, early_stopping=early_stopping, 
                   early_stopping_patience=early_stopping_patience, regularization='l1')

# Save model
#model.save_model(layers=layers, name='model')

# Load model
#layer_dense_list = model.load_model('model.json')
predictions = model.testing_model(layers=layer_dense_list, activations=activation_layer_list, data_X=testing_set_X)

# For binary classification, the prediction is the index of the maximum value in the last layer's output
    # /!\ need to have something for more than binary classification
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == testing_set_y)
print(f"Test accuracy: {accuracy}")

'''
hyperparameter_ranges = {
        'learning_rate': (0.001, 0.1),
        #'hidden_layers': (1, 3),
        'number_epochs': (10, 200),
        'batch_size': (12, 128)
        #'neurons_per_layer': (64, 256),
        #'activation': ['relu', 'sigmoid']
        }

random_search = RandomSearch(hyperparameter_ranges)
random_search.random_search(5, 5, training_set_X=training_set_X, training_set_y=training_set_y, 
              testing_set_X=testing_set_X, testing_set_y=testing_set_y, early_stopping=True, early_stopping_patience=5)
              '''
