"""
This script will calculate the perplexity of a trained model, given a test corpus.

Example:
    python perplexity.py data/UN-english.txt.gz data/SVM_5000.pickle --lines 10000 --random-lines False
"""


import argparse
import math
import pickle

import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from sample import sample_lines, process_sentences, create_samples


def load_test_data(test_file):
    data = pd.read_pickle(test_file)

    return data[data.columns.difference(['CLASS'])], data['CLASS']


def load_model(model_file):
    with open(model_file, 'rb') as file:
        model, vector_features, classes = pickle.load(file)

    return model, vector_features, classes


def create_df(samples, vector_features):
    # dataframe needs to be build differently than in sample.py since vector needs to have the same features as the
    # trained model.
    vectors = []
    classes = []

    for sample in samples:
        vectors.append([0] * len(vector_features))
        classes.append(sample[1])
        for index, letter in enumerate(sample[0]):
            pos_feature = f'{letter}_{index}'
            if pos_feature in vector_features:
                vectors[-1][vector_features.index(pos_feature)] += 1

    return pd.DataFrame(vectors, columns=vector_features), classes


def get_log_probabilities(model, test_vectors, correct_classes):
    model_classes = model.classes_.tolist()
    correct_probs = []
    all_probs = model.predict_log_proba(test_vectors)
    for index, sample_probs in enumerate(all_probs):
        correct_class = correct_classes[index]
        # sklearn uses log to the base e, so it needs to be converted to base 2, to be used in perplexity formula
        if correct_class in model_classes:
            correct_probs.append(sample_probs[model_classes.index(correct_class)] / math.log(2, math.e))
        else:
            correct_probs.append(math.log(1 / len(test_vectors), 2))
    return correct_probs


def encode_classes(classes, all_classes):
    return [all_classes.index(consonant) for consonant in classes]


def perplexity(model, test_vectors, correct_classes, all_classes):
    if isinstance(model, MultinomialNB):
        # MultinomialNB requires numerical classes
        correct_classes = encode_classes(correct_classes, all_classes)

    probs = get_log_probabilities(model, test_vectors, correct_classes)
    return 2 ** -(sum(probs) / len(test_vectors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the perplexity of a model given a test corpus.')
    parser.add_argument('test_corpus', type=str, nargs=1, help='A .txt.gz archive containing the test corpus')
    parser.add_argument('model_file', type=str, nargs=1,
                        help='A .pickle file containing the model and the feature vector')
    parser.add_argument('--average', choices=['micro', 'macro'], default='macro',
                        help='defines how the average for precision, recall and f1-score should be calculated '
                             '(default: macro)')
    parser.add_argument('--lines', type=int, default=-1,
                        help='number of lines to use from the file. -1 for all lines (default: -1)')
    parser.add_argument('--random-lines', type=bool, default=True,
                        help='determines, if lines from corpus should be selected randomly or from the start of file')

    args = parser.parse_args()

    model, vector_features, classes = load_model(args.model_file[0])

    sentences = sample_lines(args.test_corpus[0], args.lines, args.random_lines)
    processed = process_sentences(sentences)
    samples = create_samples(processed, -1)

    test_vectors, test_classes = create_df(samples, vector_features)

    print('Perplexity:', perplexity(model, test_vectors, test_classes, classes))
