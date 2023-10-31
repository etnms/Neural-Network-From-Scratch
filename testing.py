import numpy as np
import pandas as pd
from epoch import epoch_function
from create_layer import CreateLayer


data = pd.read_csv('./dataset/random_int.csv', encoding='utf-8', dtype = str)  

# Remove invalid rows
#data = data.dropna(subset=['Int_value'])

# Shuffle the data if sample not shuffled already
#data = data.sample(frac=1).reset_index(drop=True)

X = data['Int_value']
y = data['Size']


# Hyperparameters
learning_rate = 0.1
num_epochs = 100  # Specify the number of training epochs
batch_size = 1

# Number of classes = number of inputs
# In case of second layer until the end it takes the number of neurons from the previous layer
# Ex layer1(2,512), layer2(512,3) layer3(3, 128) etc.
layer1 = CreateLayer.create(number_classes=1, number_neurons=512, activation_function='relu')
layer2 = CreateLayer.create(number_classes=512, number_neurons=1, activation_function='softmax')
layers=[layer1, layer2]

layer_dense_list = [layer[0] for layer in layers]
activation_layer_list = [layer[1] for layer in layers]

epoch_function(layers = layer_dense_list, activations=activation_layer_list, batch_size=batch_size, 
               num_epochs=num_epochs, learning_rate=learning_rate,data_X=X, data_y=y)