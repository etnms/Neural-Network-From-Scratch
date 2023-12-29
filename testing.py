import numpy as np
import pandas as pd
from model.model import Model
from layer.create_layer import CreateLayer
from split_training_data import split_training_data
from random_search import RandomSearch
from dataset import generate_spiral_set
from layer.create_modular_layer import ModularLayer

'''
File for testing purposes
'''


data = pd.read_csv('./dataset/phonemes.csv', encoding='utf-8')  

X = data[['V1', 'V2', 'V3', 'V4', 'V5']]

y = data['Class']

#X, y = generate_spiral_set.create_data(100, 3)
training_set_X, training_set_y, testing_set_X, testing_set_y = split_training_data(X, y, training_size=0.8)

number_features = 5


# Hyperparameters
learning_rate = 0.3
batch_size = 128
num_epochs = 200  # Specify the number of training epochs
early_stopping = True
early_stopping_patience = 5 # Stop training if validation loss does not improve for 5 consecutive epochs

regularization = 'l2'
loss_function_used = None
training = True
# Number of classes = number of inputs
# In case of second layer until the end it takes the number of neurons from the previous layer
# Ex layer1(2,512), layer2(512,3) layer3(3, 128) etc.


layers = ModularLayer.create_modular_layer([number_features,32,16], [32,16,5], ['relu', 'relu', 'softmax'])

if __name__ == "__main__":
    '''
    model = Model(layers=layers)
    model.train_model(num_epochs, batch_size, learning_rate,training_set_X,
                    training_set_y, training, loss_function_used, early_stopping, 
                    early_stopping_patience, regularization)

    # Save model
    model.save_model(layers, early_stopping, early_stopping_patience, regularization, name='model')

    # Load model
    #layer_dense_list = model.load_model('model.json')
    predictions = model.testing_model(data_X=testing_set_X)

    # For binary classification, the prediction is the index of the maximum value in the last layer's output
        # /!\ need to have something for more than binary classification
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = np.mean(predicted_classes == testing_set_y)
    print(f"Test accuracy: {accuracy}")
'''
    
    hyperparameter_ranges = {
            'learning_rate': (0.001, 0.1),
            'hidden_layers': (1, 3),
            'number_epochs': (10, 200),
            'batch_size': (12, 128),
            'neurons_per_layer': (64, 256),
            'activation': ['relu', 'tanh', 'sigmoid', 'softmax'],
            }

    random_search = RandomSearch(hyperparameter_ranges)
    number_features = 5
    number_classes = 3
    random_search.random_search(2, number_features, number_classes, training_set_X, training_set_y, testing_set_X, testing_set_y, 
                                early_stopping=True, early_stopping_patience=5)
    

