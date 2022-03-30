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
        encoded_classes = encode_classes(y_train)
        return MultinomialNB().fit(X_train, encoded_classes)


def save_model(model, vector_features, filename):
    with open(filename, 'wb') as file:
        pickle.dump((model, vector_features, CONSONANTS), file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file', type=str, nargs=1, help='A pickle file containing training samples')
    parser.add_argument('model', choices=['SVC', 'MultinomialNB'],
                        help='model, which should be used for training')
    parser.add_argument('--output-file', default='data/model.pickle',
                        help='file, where the model should be saved')

    args = parser.parse_args()

    X_train, y_train = load_training_data(args.file[0])
    model = train(X_train, y_train, args.model)
    save_model(model, X_train.columns.values.tolist(), args.output_file)
