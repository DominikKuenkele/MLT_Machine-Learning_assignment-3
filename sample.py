import argparse
import gzip
import random

import pandas as pd

CONSONANTS = 'bcdfghjklmnpqrstvwxyz'


def sample_lines(file_name, lines):
    with gzip.open(file_name, 'rt') as file:
        file_lines = file.read().splitlines()
    if lines > len(file_lines):
        lines = len(file_lines)
    return random.sample(file_lines, lines)


def process_sentences(lines):
    return [line.lower() for line in lines]


def get_first_consonant(string):
    for char in string:
        if char in CONSONANTS:
            return char
    return ''


def create_samples(sentences, number_of_samples):
    ngram_length = 4

    classifications = []
    for sentence in sentences:
        for index in range(0, len(sentence) - ngram_length):
            end_ngram = index + ngram_length
            ngram = sentence[index:end_ngram]
            next_consonant = get_first_consonant(sentence[end_ngram + 1:])
            if next_consonant != '':
                classifications.append((ngram, next_consonant))

    if number_of_samples > len(classifications):
        number_of_samples = len(classifications)
    random_samples = random.sample(classifications, number_of_samples)
    return random_samples


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
    parser.add_argument('file', type=str, nargs=1, help='A .gz archive containing the training data')
    parser.add_argument('--lines', default=100000, type=int, help='number of lines to use from the file')
    parser.add_argument('--samples', default=1000, type=int, help='number of samples to build test/training data')
    parser.add_argument('--training-split', default=0.8, type=float,
                        help='percentage of data that should be taken as training data')

    args = parser.parse_args()

    sentences = sample_lines(args.file[0], args.lines)
    processed = process_sentences(sentences)
    samples = create_samples(processed, args.samples)
    df = create_df(samples)

    train_data, test_data = split_samples(df, args.training_split)
    save_to_file(train_data, 'data/train.pickle')
    save_to_file(test_data, 'data/test.pickle')
