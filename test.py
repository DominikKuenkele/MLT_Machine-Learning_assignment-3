import argparse
import pickle

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB


def load_test_data(test_file):
    data = pd.read_pickle(test_file)

    return data[data.columns.difference(['CLASS'])], data['CLASS']


def load_model(model_file):
    with open(model_file, 'rb') as file:
        model, _, classes = pickle.load(file)

    return model, classes


def eval_model(model, X_test, y_test, classes, average):
    if isinstance(model, MultinomialNB):
        y_test = [classes.index(consonant) for consonant in y_test]

    predictions = model.predict(X_test)
    print('Precision:', precision_score(y_test, predictions, average=average))
    print('Recall:', recall_score(y_test, predictions, average=average,))
    print('F1 Score:', f1_score(y_test, predictions, average=average))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('test_data_file', type=str, nargs=1, help='A pickle file containing test samples')
    parser.add_argument('model_file', type=str, nargs=1, help='A pickle file containing the model')
    parser.add_argument('--average', choices=['micro', 'macro'], default='macro',
                        help='defines how average should be calculated')

    args = parser.parse_args()

    X_test, y_test = load_test_data(args.test_data_file[0])
    model, classes = load_model(args.model_file[0])
    eval_model(model, X_test, y_test, classes, args.average)
