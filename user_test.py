from model.model import Model
from load_csv_data import load_csv_data
import numpy as np
import argparse
from model.model import Model
from split_training_data import split_training_data

def test_model(file_path):
    '''Function to get user data'''

    # Load the testing data

    testing_set_X, testing_set_y, number_features, number_classes = load_csv_data(file_path)
   
    # Load the trained model
    layers = Model.load_model(None, 'model.json')

    model = Model(layers)
    
    predictions = model.testing_model(data_X=testing_set_X)

    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = np.mean(predicted_classes == testing_set_y)
    print(predicted_classes)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path of a csv file')
    parser.add_argument('file_path', type=str, help='Get the file of csv file to test')
    args = parser.parse_args()

    test_model(args.file_path)