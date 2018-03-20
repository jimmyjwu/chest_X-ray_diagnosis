"""
This script uses extracted feature/embedding vectors to classify new examples.
"""
# Python modules
import argparse
import logging
import os

# Scientific and deep learning modules
import numpy
import torch
from torch.autograd import Variable
from tqdm import tqdm
from skmultilearn.adapt.mlknn import MLkNN
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Project modules
import utils
import model.net as net
import model.data_loader as data_loader
import analyze_feature_vectors


# Configure user arguments for this script
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-small',
                             action='store_true', # Sets arguments.small to False by default
                             help='Use small dataset instead of full dataset')

argument_parser.add_argument('--features_directory',
                             default='feature_data',
                             help='Directory containing files related to feature extraction')
argument_parser.add_argument('--dataset_type',
                             default='val',
                             help='Which part of the dataset to evaluate, i.e. "val" (default), or "test"')


def train_and_evaluate_k_nearest_neighbors(X_train, y_train, X_evaluation, y_evaluation, k=10, training_sample_fraction=0.01):
    """
    Trains a multi-label k-nearest neighbors model and evaluates its performance.

    Arguments:
        X_train: (2D NumPy array) where X[i,j] is the j-th feature value for the i-th training example
        y_train: (2D NumPy array) where y[i,j] is 1 if the i-th training example has label j, and 0 otherwise
        X_evaluation: (2D NumPy array) where X[i,j] is the j-th feature value for the i-th evaluation example
        y_evaluation: (2D NumPy array) where y[i,j] is 1 if the i-th evaluation example has label j, and 0 otherwise
        training_sample_fraction: (float) the fraction of training examples to use when training
    """
    logging.info('Starting evaluation of k-nearest neighbors with k=' + str(k))

    # Sample a subset of the data in a way that preserves the proportion of examples with each label
    X_train, y_train = utils.sample_examples_by_class(X_train, y_train, training_sample_fraction)

    logging.info('Fitting model')

    # Fit a k-nearest neighbors model to the training data
    model = MLkNN(k=k)
    model.fit(X_train, y_train)

    logging.info('Evaluating model')

    # Make predictions (a probability for each label) on the training set, compute AUROCs, log the average AUROC
    y_train_predict = model.predict_proba(X_train).toarray() # Convert SciPy sparse matrix to NumPy array
    train_AUROCs = net.accuracy(y_train_predict, y_train)
    logging.info('- Training metrics : mean AUROC: {:05.3f}'.format(numpy.mean(train_AUROCs)))

    # Make predictions (a probability for each label) on the evaluation set, compute AUROCs, log them all
    y_evaluation_predict = model.predict_proba(X_evaluation).toarray() # Convert SciPy sparse matrix to NumPy array
    evaluation_AUROCs = net.accuracy(y_evaluation_predict, y_evaluation)
    logging.info('- Evaluation metrics : mean AUROC: {:05.3f}'.format(numpy.mean(evaluation_AUROCs)))
    utils.print_class_accuracy(evaluation_AUROCs)


def train_and_evaluate_multilabel_classifier_from_binary_classifier(
        BinaryClassifier, X_train, y_train, X_evaluation, y_evaluation,
        binary_classifier_arguments={},
        training_sample_fraction=1.0):
    """
    Instantiates a multi-label model based on a given binary classifier, trains it, and evaluates its performance.

    Arguments:
        BinaryClassifier: (class) a scikit-learn binary classifier class
        X_train: (2D NumPy array) where X[i,j] is the j-th feature value for the i-th training example
        y_train: (2D NumPy array) where y[i,j] is 1 if the i-th training example has label j, and 0 otherwise
        X_evaluation: (2D NumPy array) where X[i,j] is the j-th feature value for the i-th evaluation example
        y_evaluation: (2D NumPy array) where y[i,j] is 1 if the i-th evaluation example has label j, and 0 otherwise
        training_sample_fraction: (float) the fraction of training examples to use when training
    """
    logging.info('Starting evaluation of a multi-label classifier based on ' + BinaryClassifier.__name__ + ' with arguments ' + str(binary_classifier_arguments))

    # Sample a subset of the data in a way that preserves the proportion of examples with each label
    X_train, y_train = utils.sample_examples_by_class(X_train, y_train, training_sample_fraction)

    logging.info('Fitting model')

    # Fit a model to the training data
    model = OneVsRestClassifier(BinaryClassifier(**binary_classifier_arguments))
    model.fit(X_train, y_train)

    logging.info('Evaluating model')

    # Make predictions (a probability for each label) on the training set, compute AUROCs, log the average AUROC
    y_train_predict = model.predict_proba(X_train)
    train_AUROCs = net.accuracy(y_train_predict, y_train)
    logging.info('- Training metrics : mean AUROC: {:05.3f}'.format(numpy.mean(train_AUROCs)))

    # Make predictions (a probability for each label) on the evaluation set, compute AUROCs, log them all
    y_evaluation_predict = model.predict_proba(X_evaluation)
    evaluation_AUROCs = net.accuracy(y_evaluation_predict, y_evaluation)
    logging.info('- Evaluation metrics : mean AUROC: {:05.3f}'.format(numpy.mean(evaluation_AUROCs)))
    utils.print_class_accuracy(evaluation_AUROCs)



def main():
    # Load user arguments
    arguments = argument_parser.parse_args()

    # Record whether GPU is available
    CUDA_is_available = torch.cuda.is_available()

    # Set random seed for reproducible experiments
    torch.manual_seed(230)
    if CUDA_is_available: torch.cuda.manual_seed(230)

    # Configure logger
    utils.set_logger(os.path.join(arguments.features_directory, 'classify_by_cluster.log'))

    # Features file names are of the form
    #       features_directory/{train, val, test}_features_and_labels.txt
    # with 'small_' prepended if user specifies '--small'
    train_features_file_name = ('small_' if arguments.small else '') + 'train_features_and_labels.txt'
    evaluation_features_file_name = ('small_' if arguments.small else '') + arguments.dataset_type + '_features_and_labels.txt'
    train_features_file_path = os.path.join(arguments.features_directory, train_features_file_name)
    evaluation_features_file_path = os.path.join(arguments.features_directory, evaluation_features_file_name)

    # Exit if user-specified feature vector files do not exist
    for features_file_path in [train_features_file_path, evaluation_features_file_path]:
        if not os.path.isfile(features_file_path):
            logging.info('Features file(s) not detected; please generate them by extracting feature vectors from a model, for example by running analyze_feature_vectors.py.')
            return

    # Read features and labels from files
    X_train, y_train = utils.read_feature_and_label_matrices(train_features_file_path)
    X_evaluation, y_evaluation = utils.read_feature_and_label_matrices(evaluation_features_file_path)

    # Evaluate the model
    RANDOM_FOREST_ARGUMENTS = {
        'n_estimators': 10,
        'max_depth': 10,
        'n_jobs': -1, # Use all available CPUs
    }
    train_and_evaluate_multilabel_classifier_from_binary_classifier(
        RandomForestClassifier, X_train, y_train, X_evaluation, y_evaluation,
        RANDOM_FOREST_ARGUMENTS, training_sample_fraction=0.1)
    """
    train_and_evaluate_k_nearest_neighbors(X_train, y_train, X_evaluation, y_evaluation, k=10, training_sample_fraction=0.01)
    """


if __name__ == '__main__':
	main()