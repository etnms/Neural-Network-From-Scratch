import string


'''
Functions to convert data to readable format for the network
'''

# Mean max function 
def mean_max_scaling(data, column_name: string):
    min_value = data[column_name].min()
    max_value = data[column_name].max()
    return data[column_name].apply(lambda x: (x - min_value) / (max_value - min_value))


# Standardization function 
def standardization(data, column_name: string):
    mean_value = data[column_name].mean()
    std_dev = data[column_name].std()
    return data[column_name].apply(lambda x: (x - mean_value) / std_dev)

# Simple functions that convert individual arrays in a tokenized_sentence to its actual int value
def token_to_int(tokenized_sentences):
    return [token[0] for token in tokenized_sentences]
