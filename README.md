# LT2222-assignment-3
> Part of the Master in Language Technology at the University of Gothenburg
>
> **Course:** Machine learning for statistical NLP: introduction (LT2222)
>
> **Assignment:** Assignment 3

## Introduction
In this assignment, you will use Python command-line scripts built with the command-line argument parser module, argparse.  You will write and run the scripts from the command line yourself, with no template, and you will document them in a way that explains to us how to run them. You will submit a github repo you created yourself with all your code and two saved model files. You can freely re-use code you used in assignment 2.   In addition, all scripts should use argparse to print out brief explanations of their parameters and options.

The scripts will represent a "next consonant" predictor, where you use four characters (of any kind) to predict the next nearest consonant character, using the standard consonantal sounds of English (including y and w).  You can use the UN-English corpus from Assignment 2 to train and test the models.  The next consonant predictor works over full sentences, lower-cased, but keeping all punctuation. Consider the text

The quick brown fox jumped over the lazy dog -- how much wood could a woodchuck chuck

Some of the samples you could extract include "quic" which predicts "k". However, "uick" predicts "b".  But "ick " (with a space after it) also predicts "b".  " dog" (with a space in front) as well as "dog." predicts "h".  It always predicts the next consonant (lowercase), even if the next consonant is not the fifth in the sequence.  The last characters in a line, if there aren't enough to produce a next consonant, are ignored.

## Scripts to write
All scripts must run on mltgpu.

### sample.py
You will write a script named sample.py which takes a text formatted like the UN-English corpus (but is not necessarily the same file) and produces two lists, a set of training samples and a set of test samples, based on the total number of samples to take and a train/test split.

### train.py
This script will take the training output of sample.py as the input file and learn either a model from sklearn's CategoricalNB or SVC (with a linear kernel), based on a command-line option.  It will save the model to a file, also given on the command line.

### test.py
This script will load the model output of train.py and the test sample output of test.py and calculate accuracy, precision, recall, and F-measure and print these out to the terminal.

### perplexity.py
This script will calculate the perplexity of an output model given a sample text, as full sentences in the UN-English corpus format.

## Explore and experiment
Vary the overall sample size and report changes in all of the measures in test.py and perplexity.py.  Compare the support vector classifier to the naïve Bayes in terms of the test statistics, including perplexity.  Document differences in their ability to learn in relation to the sample size.  Try some larger samples on the order of hundreds of thousands or more, memory permitting -- assume that an hour to train is a reasonable maximum. Document your observations in a Markdown file.

## Bonus
You can write separate training and testing scripts for this with a different name, as necessary.  Develop a feed-forward network in PyTorch that learns the same relation as in train.py.  It should have at least two hidden layers with non-linearities.  Document your efforts and the results.
