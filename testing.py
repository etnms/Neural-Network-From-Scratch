import numpy as np
import pandas as pd
from epoch import epoch_function
from create_layer import CreateLayer
from dataset import generate_spiral_set
from split_training_data import split_training_data


data = pd.read_csv('./dataset/random_int.csv', encoding='utf-8', dtype = str)  

# Remove invalid rows
#data = data.dropna(subset=['Int_value'])

# Shuffle the data if sample not shuffled already
#data = data.sample(frac=1).reset_index(drop=True)

data = data[pd.to_numeric(data['Int_value'], errors='coerce').notna()]
data['Int_value'] = pd.to_numeric(data['Int_value'])

# 'Int_value' now contains numeric values
X = data['Int_value']
label_mapping = {'small': 0, 'big': 1}
y = data['Size'].map(label_mapping).astype(int)


#X, y = generate_spiral_set.create_data(100, 3)


#training_set_X, training_set_y, testing_set_X, testing_set_y = split_training_data(X, y, training_size=0.8)


# Hyperparameters
learning_rate = 1
num_epochs = 10  # Specify the number of training epochs
batch_size = 500

# Number of classes = number of inputs
# In case of second layer until the end it takes the number of neurons from the previous layer
# Ex layer1(2,512), layer2(512,3) layer3(3, 128) etc.
layer1 = CreateLayer.create(number_classes=1, number_neurons=128, activation_function='relu')
layer2 = CreateLayer.create(number_classes=128, number_neurons=64, activation_function='relu')
layer3 = CreateLayer.create(number_classes=64, number_neurons=2, activation_function='softmax')
layers=[layer1, layer2, layer3]

layer_dense_list = [layer[0] for layer in layers]
activation_layer_list = [layer[1] for layer in layers]

epoch_function(layers = layer_dense_list, activations=activation_layer_list, batch_size=batch_size, 
               num_epochs=num_epochs, learning_rate=learning_rate,data_X=X, data_y=y)