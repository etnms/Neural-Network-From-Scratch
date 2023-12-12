'''
One-hot encodes a list of labels.

Parameters:
- labels: List of labels to be one-hot encoded.
- num_classes: Number of classes. If not provided, it will be inferred from the unique labels.

Returns:
- A list of one-hot encoded vectors.
'''

def one_hot_encode(labels, num_classes=None):

    unique_labels = set(labels)

    if num_classes is None:
        num_classes = len(unique_labels)

    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    one_hot_encoded = []
    for label in labels:
        vector = [0] * num_classes
        vector[label_to_index[label]] = 1
        one_hot_encoded.append(vector)

    return one_hot_encoded

'''
# Example usage:
labels = ['cat', 'dog', 'bird', 'dog', 'cat']
encoded_labels = one_hot_encode(labels)
print(encoded_labels)
'''
