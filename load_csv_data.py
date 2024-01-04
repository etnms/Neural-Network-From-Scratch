import pandas as pd


def load_csv_data(path):
    data = pd.read_csv(f'{path}', encoding='utf-8')
    # Exclude the last column (assuming it's the target variable)
    X = data.iloc[:, :-1]
    # The last column is the target variable (y)
    y = data.iloc[:, -1]
    number_features = X.shape[1]
    number_classes = data.iloc[:, -1].nunique()
    return X, y, number_features, number_classes

def load_csv_data_with_strings():
    # Tokenize and pad string features
    # style_tokenizer, tokenized_styles = create_tokenizer(X['Style'])
    # country_tokenizer, tokenized_countries = create_tokenizer(X['Country'])

    # Set the desired length for padding
    # desired_length = 10

    # Pad tokenized strings
    # padded_styles = pad_sentences(tokenized_styles, desired_length)
    # padded_countries = pad_sentences(tokenized_countries, desired_length)

    # Combine padded features into a single input array
    # padded_features = np.column_stack((padded_styles, padded_countries))

    # Get the feature names
    # feature_names = X.columns.tolist()
    # Get the number of unique features
    # number_features = len(set(feature_names))
    # number_features = padded_features.shape[1]
    pass
