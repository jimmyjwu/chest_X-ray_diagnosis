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


def train_and_evaluate_k_nearest_neighbors(X_train, y_train, X_evaluation, y_evaluation):
    """
    Trains a k-nearest neighbors model and evaluates its performance against specified metrics.

    Arguments:
        train_feature_vectors: (list of NumPy arrays) where the i-th array is the i-th training example's features
        train_label_vectors: (list of NumPy arrays) where the i-th array is the i-th training example's labels
        evaluation_feature_vectors: (list of NumPy arrays) where the i-th array is the i-th evaluation example's features
        evaluation_label_vectors: (list of NumPy arrays) where the i-th array is the i-th evaluation example's labels
    """
    logging.info('Starting evaluation of k-nearest neighbors')

    # Fit/"train" a k-nearest neighbors model to the training data
    model = MLkNN(k=10)
    logging.info('Fitting model')
    model.fit(X_train, y_evaluation)

    # Make predictions (a probability for each label) on the evaluation set
    logging.info('Making predictions')
    y_predict = model.predict_proba(X_evaluation).toarray() # Convert SciPy sparse matrix to NumPy array

    # Compute AUROCs for each individual class
    class_AUROCs = net.accuracy(y_predict, y_evaluation)

    return class_AUROCs



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
    class_AUROCs = train_and_evaluate_k_nearest_neighbors(train_feature_vectors, train_label_vectors, evaluation_feature_vectors, evaluation_label_vectors)

    # Print average AUROC and individual class AUROCs
    logging.info('- Evaluation metrics : mean AUROC: {:05.3f}'.format(numpy.mean(AUROCs)))
    utils.print_class_accuracy(class_AUROCs)



if __name__ == '__main__':
	main()