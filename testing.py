import numpy as np
from epoch import epoch_function
from create_layer import CreateLayer


np.random.seed(0)

# generate data for learning and testing purposes
# Generate spirals with numbers of points and number of classes
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(class_number*4, (class_number+1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

# Hyperparameters
learning_rate = 0.1
num_epochs = 100  # Specify the number of training epochs
batch_size = 8

# Number of classes = number of inputs
# In case of second layer until the end it takes the number of neurons from the previous layer
# Ex layer1(2,512), layer2(512,3) layer3(3, 128) etc.
layer1 = CreateLayer.create(number_classes=2, number_neurons=512, activation_function='relu')
layer2 = CreateLayer.create(number_classes=512, number_neurons=3, activation_function='softmax')
layers=[layer1, layer2]

layer_dense_list = [layer[0] for layer in layers]
activation_layer_list = [layer[1] for layer in layers]

epoch_function(layers = layer_dense_list, activations=activation_layer_list, batch_size=batch_size, 
               num_epochs=num_epochs, learning_rate=learning_rate,data_X=X, data_y=y)