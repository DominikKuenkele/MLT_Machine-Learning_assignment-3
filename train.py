"""
This script will train an ML model based on training samples.

Example:
    python train.py data/train_100000.pickle SVC --output-file data/SVM_100000.pickle
"""

import argparse
import pickle

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

CONSONANTS = 'bcdfghjklmnpqrstvwxyz'


def load_training_data(training_file):
    data = pd.read_pickle(training_file)

    return data[data.columns.difference(['CLASS'])], data['CLASS']


def encode_classes(classes):
    return [CONSONANTS.index(consonant) for consonant in classes]


def train(X_train, y_train, model):
    if model == 'SVC':
        return SVC(kernel='linear', probability=True).fit(X_train, y_train)
    elif model == 'MultinomialNB':
        # MultinomialNB requires numerical classes
        encoded_classes = encode_classes(y_train)
        return MultinomialNB().fit(X_train, encoded_classes)


def save_model(model, vector_features, filename):
    # saves:
    # - model
    # - the features of the vector (to reuse for perplexity calculation)
    # - possible classes (to reuse for numerical encoding)
    with open(filename, 'wb') as file:
        pickle.dump((model, vector_features, CONSONANTS), file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model based on training samples and save it to a file.')
    parser.add_argument('file', type=str, nargs=1, help='A .pickle file containing training samples')
    parser.add_argument('model', choices=['SVC', 'MultinomialNB'],
                        help='model, which will be used for training')
    parser.add_argument('--output-file', default='model.pickle',
                        help='file, in which the model should be stored (default: model.pickle)')

    args = parser.parse_args()

    X_train, y_train = load_training_data(args.file[0])
    model = train(X_train, y_train, args.model)
    save_model(model, X_train.columns.values.tolist(), args.output_file)
