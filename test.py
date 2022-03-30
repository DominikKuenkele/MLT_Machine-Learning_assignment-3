"""
This script will take a trained ML model and calculate accuracy, precision, recall and f1-score based on test samples.

Example:
    python test.py data/test_50000.pickle data/SVM_50000.pickle
"""

import argparse
import pickle

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB


def load_test_data(test_file):
    data = pd.read_pickle(test_file)

    return data[data.columns.difference(['CLASS'])], data['CLASS']


def load_model(model_file):
    with open(model_file, 'rb') as file:
        model, _, classes = pickle.load(file)

    return model, classes


def encode_classes(classes, all_classes):
    return [all_classes.index(consonant) for consonant in classes]


def eval_model(model, X_test, y_test, classes, average):
    if isinstance(model, MultinomialNB):
        # MultinomialNB requires numerical classes
        y_test = encode_classes(y_test, classes)

    predictions = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions, average=average))
    print('Recall:', recall_score(y_test, predictions, average=average, ))
    print('F1 Score:', f1_score(y_test, predictions, average=average))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained model on test samples and output different scores.')
    parser.add_argument('test_data_file', type=str, nargs=1, help='A .pickle file containing test samples')
    parser.add_argument('model_file', type=str, nargs=1,
                        help='A .pickle file containing the model and the feature vector')
    parser.add_argument('--average', choices=['micro', 'macro'], default='macro',
                        help='defines how the average for precision, recall and f1-score should be calculated '
                             '(default: macro)')

    args = parser.parse_args()

    X_test, y_test = load_test_data(args.test_data_file[0])
    model, classes = load_model(args.model_file[0])
    eval_model(model, X_test, y_test, classes, args.average)
