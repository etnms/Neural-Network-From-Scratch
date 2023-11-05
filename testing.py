import numpy as np
import pandas as pd
from epoch import epoch_function
from create_layer import CreateLayer
from dataset import generate_spiral_set
from split_training_data import split_training_data
from data_conversion import data_conversion
from text_processing import text_processing


'''
This class is messy at the moment for testing purposes (but this needs to be cleaned a bit at one point)
'''


#data = pd.read_csv('./dataset/random_int.csv', encoding='utf-8', dtype = str)  
data = pd.read_csv('./dataset/wordlist_random.csv', encoding='utf-8', dtype = str)  
# Remove invalid rows
#data = data.dropna(subset=['Int_value'])
data = data.dropna(subset=['String'])
# Shuffle the data if sample not shuffled already
#data = data.sample(frac=1).reset_index(drop=True)


#data = data[pd.to_numeric(data['Int_value'], errors='coerce').notna()]
#data['Int_value'] = pd.to_numeric(data['Int_value'])
#data_encoded = pd.get_dummies(data, columns=['String'])

# Convert data to values between 0 and 1 with mean/max scaling or standardization 
#data['Int_value_scaled'] = data_conversion.standardization(data, 'Int_value')
tokenizer, tokenized_sentences = text_processing.create_tokenizer(data['String'])

clean_token_data = data_conversion.token_to_int(tokenized_sentences)
data['test'] = clean_token_data

data['string_value_stand'] = data_conversion.standardization(data, 'test')

data['length'] = data['String'].apply(len)
data['alpha'] = data['String'].str.isalpha().astype(int)
data['numeric'] = data['String'].str.isnumeric().astype(int)
# 'Int_value' now contains numeric values
#X = data['Int_value_scaled']
X = data['string_value_stand']

X = data[['length', 'alpha', 'numeric']]
#X = data_encoded
#label_mapping = {'small': 0, 'big': 1}
label_mapping = {'str': 0, 'int': 1}
#y = data['Size'].map(label_mapping).astype(int)
y = data['Category'].map(label_mapping).astype(int)


#X, y = generate_spiral_set.create_data(100, 3)

training_set_X, training_set_y, testing_set_X, testing_set_y = split_training_data(X, y, training_size=0.8)

# Hyperparameters
learning_rate = 0.01
num_epochs = 10  # Specify the number of training epochs
batch_size = 16

# Number of classes = number of inputs
# In case of second layer until the end it takes the number of neurons from the previous layer
# Ex layer1(2,512), layer2(512,3) layer3(3, 128) etc.
layer1 = CreateLayer.create(number_classes=3, number_neurons=16, activation_function='sigmoid')
layer2 = CreateLayer.create(number_classes=16, number_neurons=2, activation_function='sigmoid')
#layer3 = CreateLayer.create(number_classes=4, number_neurons=2, activation_function='softmax')
layers=[layer1, layer2]

layer_dense_list = [layer[0] for layer in layers]
activation_layer_list = [layer[1] for layer in layers]

epoch_function(layers = layer_dense_list, activations=activation_layer_list, batch_size=batch_size, 
               num_epochs=num_epochs, learning_rate=learning_rate,data_X=training_set_X, data_y=training_set_y)