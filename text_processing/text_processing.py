import pandas as pd
'''
Tokenizer and padding functions to preprocess text (convert text to numerical values) so it can be read in the network.
'''

# Convert text (sentence or words) into numerical values
def create_tokenizer(sentences):
    tokenizer = {}
    current_index = 1  # Start indexing from 1
    tokenized_sentences = []

    for sentence in sentences:
        if pd.isna(sentence):  # Check for NaN values
            sentence = 'NaN_placeholder'  # Replace NaN with a placeholder string

        tokenized_sentence = []
        for word in sentence.lower().split(): # Convert to lowercase and split into words
            if word not in tokenizer:
                tokenizer[word] = current_index
                current_index += 1
            tokenized_sentence.append(tokenizer[word])
        
        tokenized_sentences.append(tokenized_sentence)

    return tokenizer, tokenized_sentences

# Pad sentences (return a new list of sentences that are padded with zeros or truncated to the desired length)
def pad_sentences(tokenized_sentences, desired_length):
    padded_sentences = []

    for sentence in tokenized_sentences:
        if len(sentence) > desired_length:
            # Truncate the sentence if it's too long
            padded_sentence = sentence[:desired_length]
        else:
            # Pad the sentence with zeros if it's too short
            padding = [0] * (desired_length - len(sentence))
            padded_sentence = sentence + padding

        padded_sentences.append(padded_sentence)

    return padded_sentences