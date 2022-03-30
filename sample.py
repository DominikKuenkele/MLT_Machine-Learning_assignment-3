"""
This script will create samples for a next consonant predictor out of a corpus
and split them into test and training samples.
The samples consist of four characters and the next succeeding consonant.

Example:
    python sample.py data/UN-english.txt.gz --lines 100000 --samples 500000 --output-train data/train_500000.pickle --output-test data/test_500000.pickle
"""


import argparse
import gzip
import random

import pandas as pd

CONSONANTS = 'bcdfghjklmnpqrstvwxyz'


def sample_lines(file_name, lines=-1, shuffle=True):
    with gzip.open(file_name, 'rt') as file:
        file_lines = file.read().splitlines()
    if lines > len(file_lines) or lines == -1:
        lines = len(file_lines)

    if shuffle:
        return random.sample(file_lines, lines)
    else:
        return file_lines[:lines]


def process_sentences(lines):
    return [line.lower() for line in lines]


def get_first_consonant(string):
    for char in string:
        if char in CONSONANTS:
            return char
    return ''


def create_samples(sentences, number_of_samples=-1):
    ngram_length = 4

    classifications = []
    for sentence in sentences:
        for index in range(0, len(sentence) - ngram_length):
            end_ngram = index + ngram_length
            ngram = sentence[index:end_ngram]
            next_consonant = get_first_consonant(sentence[end_ngram + 1:])
            if next_consonant != '':
                classifications.append((ngram, next_consonant))

    if number_of_samples > len(classifications) or number_of_samples == -1:
        number_of_samples = len(classifications)

    return random.sample(classifications, number_of_samples)


def create_df(samples):
    features = ['CLASS']
    dataframe = []

    for sample in samples:
        dataframe.append([0] * len(features))
        dataframe[-1][0] = sample[1]
        for index, letter in enumerate(sample[0]):
            pos_feature = f'{letter}_{index}'
            if pos_feature not in features:
                features.append(pos_feature)
                for prev_sample in dataframe:
                    prev_sample.append(0)
            dataframe[-1][features.index(pos_feature)] += 1

    return pd.DataFrame(dataframe, columns=features)


def split_samples(dataframe, train_percent):
    split = round(len(dataframe) * train_percent)

    return dataframe[:split], dataframe[split:]


def save_to_file(data, filename):
    data.to_pickle(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training/test samples for a next consonant predictor.')
    parser.add_argument('file', type=str, nargs=1, help='A .txt.gz archive containing the training data')
    parser.add_argument('--lines', default=100000, type=int,
                        help='number of lines to use from the file. -1 for all lines (default: 100000)')
    parser.add_argument('--samples', default=1000, type=int,
                        help='number of samples to build test/training data (default: 1000)')
    parser.add_argument('--training-split', default=0.8, type=float,
                        help='percentage of samples that should be taken as training data. '
                             'The rest will ne used as test data. (default: 0.8)')
    parser.add_argument('--output-test', default='test.pickle', type=str,
                        help='file, in which test samples should be stored (default: test.pickle)')
    parser.add_argument('--output-train', default='train.pickle', type=str,
                        help='file, in which training samples should be stored (default: train.pickle)')

    args = parser.parse_args()

    sentences = sample_lines(args.file[0], args.lines)
    processed = process_sentences(sentences)
    samples = create_samples(processed, args.samples)
    df = create_df(samples)

    train_data, test_data = split_samples(df, args.training_split)
    save_to_file(train_data, args.output_train)
    save_to_file(test_data, args.output_test)
