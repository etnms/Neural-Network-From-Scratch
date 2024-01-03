from model.model import Model
from load_csv_data import load_csv_data
import numpy as np
import argparse
from model.model import Model

def test_model(file_path):
    '''Function to get user data'''

    # Load the testing data

    testing_set_X, testing_set_y, number_features, number_classes = load_csv_data(file_path)
    # Load the trained model
    layer_dense_list = Model.load_model(None, 'model.json')
    model = Model([layer_dense_list])
    print("Number of features:", number_features)
    print("Number of classes:", number_classes)
    print("Input shape:", testing_set_X.shape)
    single_data_point = testing_set_X.iloc[0].values  # Assuming the first data point
    single_data_point = np.expand_dims(single_data_point, axis=0)

    predictions = model.testing_model(data_X=single_data_point)

    print(single_data_point)
    print(single_data_point.shape)
    # For binary classification, the prediction is the index of the maximum value in the last layer's output
    # /!\ need to have something for more than binary classification
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = np.mean(predicted_classes == testing_set_y)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path of a csv file')
    parser.add_argument('file_path', type=str, help='Get the file of csv file to test')
    args = parser.parse_args()

    test_model(args.file_path)