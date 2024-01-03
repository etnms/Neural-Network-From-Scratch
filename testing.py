import numpy as np
import pandas as pd
from model.model import Model
from layer.create_layer import CreateLayer
from split_training_data import split_training_data
from random_search import RandomSearch
from dataset import generate_spiral_set
from layer.create_modular_layer import ModularLayer
from text_processing.text_processing import create_tokenizer, pad_sentences
import argparse


'''
File for testing purposes
'''


#data = pd.read_csv('./dataset/phonemes.csv', encoding='utf-8') 
#X = data[['V1', 'V2', 'V3', 'V4', 'V5']]
#y = data['Class']
# number_features = X.shape[1]

data = pd.read_csv('./dataset/phone.csv', encoding='utf-8') 
# Exclude the last column (assuming it's the target variable)
X = data.iloc[:, :-1]

# The last column is the target variable (y)
y = data.iloc[:, -1]
number_features = X.shape[1]
print(number_features)
number_classes = data.iloc[:, -1].nunique()
print(number_classes)
#data['Stars'] = data['Stars'].astype(float).round().astype(int)
#y = data['Stars']

# Tokenize and pad string features
#style_tokenizer, tokenized_styles = create_tokenizer(X['Style'])
#country_tokenizer, tokenized_countries = create_tokenizer(X['Country'])

# Set the desired length for padding
#desired_length = 10

# Pad tokenized strings
#padded_styles = pad_sentences(tokenized_styles, desired_length)
#padded_countries = pad_sentences(tokenized_countries, desired_length)

# Combine padded features into a single input array
#padded_features = np.column_stack((padded_styles, padded_countries))

# Get the feature names
feature_names = X.columns.tolist()
# Get the number of unique features
#number_features = len(set(feature_names))
#number_features = padded_features.shape[1]

#X, y = generate_spiral_set.create_data(100, 3)
#training_set_X, training_set_y, testing_set_X, testing_set_y = split_training_data(X, y, training_size=0.8)
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
# Number of classes = number of inputs
# In case of second layer until the end it takes the number of neurons from the previous layer
# Ex layer1(2,512), layer2(512,3) layer3(3, 128) etc.


#data = pd.read_csv('./dataset/phonemes.csv', encoding='utf-8')  \
#X = data[['V1', 'V2', 'V3', 'V4', 'V5']]
#y = data['Class']
# number_features = X.shape[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random search')
    parser.add_argument('random_search', type=str, help='Applies random search')
    args = parser.parse_args()

    if args.random_search.lower() == 'false':
        layers = ModularLayer.create_modular_layer([number_features, 128, 128], [128, 128, 4], ['sigmoid', 'sigmoid', 'softmax'])
        model = Model(layers=layers)
        model.train_model(num_epochs, batch_size, learning_rate, training_set_X,
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
    else:
        hyperparameter_ranges = {
                'learning_rate': (0.001, 0.1),
                'hidden_layers': (1, 3),
                'number_epochs': (100, 1000),
                'batch_size': (12, 128),
                'neurons_per_layer': (64, 256),
                'activation': ['relu', 'tanh', 'sigmoid', 'softmax'],
                }

        random_search = RandomSearch(hyperparameter_ranges)

        random_search.random_search(10, number_features, number_classes, training_set_X, training_set_y, testing_set_X, testing_set_y, 
                                    early_stopping=True, early_stopping_patience=5)
