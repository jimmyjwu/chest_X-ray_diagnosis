"""
This script ensembles and evaluates the models.
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Project modules
import utils
import model.net as net
import model.data_loader as data_loader
import classify_by_cluster


# Configure user arguments for this script
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--data_dir', default='data/224x224_images', help='Directory containing the dataset')
argument_parser.add_argument('--model_dir', default='experiments/base_model', help='Directory containing params.json')
argument_parser.add_argument('--restore_file',
                             default='best',
                             help='File in --model_dir containing weights to load, e.g. "best" or "last" (default: "best")')
argument_parser.add_argument('-small',
                             action='store_true', # Sets arguments.small to False by default
                             help='Use small dataset instead of full dataset')
argument_parser.add_argument('-use_tencrop',
                             action='store_true', # Sets arguments.use_tencrop to False by default
                             help='Use ten-cropping when making predictions')

argument_parser.add_argument('--features_directory',
                             default='feature_data',
                             help='Directory containing files related to feature extraction')
argument_parser.add_argument('--dataset_type',
                             default='val',
                             help='Which part of the dataset to evaluate, i.e. "val" (default), or "test"')


def evaluate(neural_network_model, other_models, loss_fn, data_loader, metrics, parameters):
    """
    Evaluates an ensemble of a given neural network model and other scikit-learn models.

    Args:
        neural_network_model: (torch.nn.Module) a neural network
        other_models: (list of sciki-learn models)
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_loader: (torch.utils.data.DataLoader) a DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        parameters: (Params) hyperparameters object
        use_tencrop: (bool) whether to use ten-cropping to make predictions
    """
    # Set model to evaluation mode
    neural_network_model.eval()

    # Summary for current eval loop
    summary = {}
    summary['loss'] = []
    summary['outputs'] = []
    summary['labels'] = []

    # Use tqdm for progress bar
    with tqdm(total=len(data_loader)) as t:
        for input_batch, labels_batch in data_loader:

            # Move data to GPU if available
            if parameters.cuda:
                input_batch, labels_batch = input_batch.cuda(async=True), labels_batch.cuda(async=True)

            # Wrap batch Tensors in Variables
            # Note: Setting "volatile=True" during evaluation prevents needlessly storing gradients, saving memory
            input_batch, labels_batch = Variable(input_batch, volatile=True), Variable(labels_batch, volatile=True)

            # Compute model output
            output_batch, features_batch = neural_network_model(input_batch)

            # Compute loss
            loss = loss_fn(output_batch, labels_batch)

            # Convert feature, prediction, and label Variables to Tensors, move to CPU, and convert to NumPy arrays
            features_batch = features_batch.data.cpu().numpy()
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # List of output batches for neural net and other models
            output_batches = [output_batch]

            # Make predictions using scikit-learn models
            for model in other_models:
                prediction_batch = model.predict_proba(features_batch)
                output_batches.append(prediction_batch)

            # Average predictions
            average_output_batch = numpy.mean(output_batches, axis=0)

            # Record predictions, labels, and losses for this batch
            summary['outputs'].append(average_output_batch)
            summary['labels'].append(labels_batch)
            summary['loss'].append(loss.data[0])

            # Update progress bar
            t.update()

    # Compute mean of all metrics in summary
    AUROCs = metrics['accuracy'](numpy.concatenate(summary['outputs']), numpy.concatenate(summary['labels']))
    metrics_mean = {}
    metrics_mean['accuracy'] = numpy.mean(AUROCs)
    metrics_mean['loss'] = sum(summary['loss'])/float(len(summary['loss']))

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info('- Eval metrics : ' + metrics_string)
    
    return metrics_mean, AUROCs


if __name__ == '__main__':
    """
    Evaluates the model on the test set.
    """
    # Load user arguments
    arguments = argument_parser.parse_args()

    # Load hyperparameters from JSON file
    json_path = os.path.join(arguments.model_dir, 'params.json')
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    parameters = utils.Params(json_path)

    # Record whether GPU is available
    parameters.cuda = torch.cuda.is_available()

    # Set random seed for reproducible experiments
    torch.manual_seed(230)
    if parameters.cuda: torch.cuda.manual_seed(230)

    # Configure logger
    utils.set_logger(os.path.join(arguments.model_dir, 'evaluate_ensemble.log'))

    # Create data loaders for test data
    logging.info('Loading test dataset...')
    test_dataloader = data_loader.fetch_dataloader(['test'], arguments.data_dir, parameters, arguments.small, arguments.use_tencrop)['test']
    logging.info('...done.')

    # Configure model; return feature vectors on each input
    densenet_model = net.DenseNet169(parameters, return_features=True)
    if parameters.cuda: densenet_model = densenet_model.cuda()

    # Load weights from trained model
    utils.load_checkpoint(os.path.join(arguments.model_dir, arguments.restore_file + '.pth.tar'), densenet_model)



    # Features file names are of the form
    #       features_directory/{train, val, test}_features_and_labels.txt
    # with 'small_' prepended if user specifies '--small'
    train_features_file_name = ('small_' if arguments.small else '') + 'train_features_and_labels.txt'
    evaluation_features_file_name = ('small_' if arguments.small else '') + arguments.dataset_type + '_features_and_labels.txt'
    train_features_file_path = os.path.join(arguments.features_directory, train_features_file_name)
    evaluation_features_file_path = os.path.join(arguments.features_directory, evaluation_features_file_name)

    # Read features and labels from files
    X_train, y_train = utils.read_feature_and_label_matrices(train_features_file_path)
    X_evaluation, y_evaluation = utils.read_feature_and_label_matrices(evaluation_features_file_path)

    RANDOM_FOREST_ARGUMENTS = {
        'n_estimators': 100,
        'max_depth': 5,
        'n_jobs': -1, # Use all available CPUs
    }
    random_forest_model = classify_by_cluster.train_and_evaluate_multilabel_classifier_from_binary_classifier(
        RandomForestClassifier, X_train, y_train, X_evaluation, y_evaluation,
        RANDOM_FOREST_ARGUMENTS, training_sample_fraction=1)

    """
    logistic_regression_model = classify_by_cluster.train_and_evaluate_multilabel_classifier_from_binary_classifier(
        LogisticRegression, X_train, y_train, X_evaluation, y_evaluation,
        {}, training_sample_fraction=0.1)

    SVM_model = classify_by_cluster.train_and_evaluate_multilabel_classifier_from_binary_classifier(
        LogisticRegression, X_train, y_train, X_evaluation, y_evaluation,
        {}, training_sample_fraction=0.1)
    """

    # Evaluate the models
    logging.info('Starting evaluation')
    test_metrics, class_auroc = evaluate(densenet_model, [random_forest_model], net.loss_fn, test_dataloader, net.metrics, parameters)
    utils.print_class_accuracy(class_auroc)
    save_path = os.path.join(arguments.model_dir, 'metrics_test_{}.json'.format(arguments.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
