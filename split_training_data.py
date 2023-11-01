import numpy as np

'''
Split dataset to have a training sample and a testing sample
Right now the method takes an input X and y, this should be modified to take an *array to make the code tidier
'''

def split_training_data(X,y, training_size: float or int, testing_size: float or int = None):
    if (training_size < 0 or training_size > 1):
        print('Incorrect size for training_size')
        return
    if (testing_size == None):
        testing_size = 1 - training_size
    if (testing_size < 0 or testing_size > 1):
        print('Incorrect size for testing_size')
        return

    data_length = len(X)
    training_end = int(data_length * training_size)
    testing_start = training_end
    testing_end = testing_start + int(data_length * testing_size)

    training_sample_X = X[:training_end]
    testing_sample_X = X[testing_start:testing_end]
    training_sample_y = y[:training_end]
    testing_sample_y = y[testing_start:testing_end]

    return training_sample_X, training_sample_y, testing_sample_X, testing_sample_y
