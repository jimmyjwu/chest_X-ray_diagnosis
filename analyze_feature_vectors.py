"""
This script:
- Extracts feature vectors from the training set and stores them in a file.
- Analyzes the clustering properties of the feature vectors.
"""
# Python modules
import argparse
import logging
import os
import itertools
from collections import OrderedDict

# Scientific and deep learning modules
import numpy
import torch
from torch.autograd import Variable
from tqdm import tqdm

# Project modules
import utils
import model.net as net
import model.data_loader as data_loader


# Configure user arguments for this script
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--data_directory', default='data/224x224_images', help='Directory containing the dataset')
argument_parser.add_argument('-small',
                             action='store_true', # Sets arguments.small to False by default
                             help='Use small dataset instead of full dataset')

argument_parser.add_argument('--model_directory', default='experiments/base_model', help='Directory containing params.json')
argument_parser.add_argument('--restore_file',
                             default='best',
                             help='File in --model_directory containing weights to load, e.g. "best" or "last" (default: "best")')

argument_parser.add_argument('--features_directory',
                             default='feature_data',
                             help='Directory containing files related to feature extraction')
argument_parser.add_argument('--features_file',
                             default='train_features_and_labels.txt',
                             help='File in --features_directory to which features should be saved')


def extract_feature_vectors(model, data_loader, parameters, features_file_path):
    """
    Extracts feature vectors from the training set and writes them, along with labels, to a file.

    Arguments:
        model: (torch.nn.Module) a neural network
        data_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        parameters: (Params) hyperparameters object
        features_file_path: (string) name of the file to which the features and labels should be written
    """
    feature_vectors, label_vectors = [], []

    # Set model to evaluation mode
    model.eval()

    # Show progress bar while iterating over mini-batches
    with tqdm(total=len(data_loader)) as progress_bar:
        for i, (X_batch, Y_batch) in enumerate(data_loader):

            # Dimensions of the input Tensor
            batch_size, channels, height, width = X_batch.size()

            # If GPU available, enable CUDA on data
            if parameters.cuda:
                X_batch = X_batch.cuda()
                Y_batch = Y_batch.cuda()

            # Wrap the input tensor in a Torch Variable
            X_batch_variable = Variable(X_batch, volatile=True)

            # Run the model on this batch of inputs, obtaining a Variable of predicted labels and a Variable of features
            Y_predicted, features = model(X_batch_variable)

            # Convert the features Variable (of size [batch_size, 1024]) to a Tensor, move it to
            # CPU, and convert it to a NumPy array
            features_numpy = features.data.cpu().numpy()

            # Move the labels Tensor (of size [batch_size, 14]) to CPU and convert it to a NumPy array
            Y_numpy = Y_batch.cpu().numpy()

            # For each example in the batch, record its features and labels
            for j in range(batch_size):
                feature_vectors.append(features_numpy[j,:])
                label_vectors.append(Y_numpy[j,:])

            progress_bar.update()

    utils.write_feature_and_label_vectors(features_file_path, feature_vectors, label_vectors)


def average_distance_between_vectors(vectors, distance):
    """
    Returns the average distance between pairs of vectors in a given list of vectors.

    Arguments:
        vectors: (list) list of NumPy arrays
        distance: (function) function that takes two NumPy arrays and returns a real number
    """
    average_distance = utils.RunningAverage()
    for vector_1, vector_2 in itertools.combinations(vectors, r=2): # All pairs of vectors
        average_distance.update(distance(vector_1, vector_2))
    return average_distance()


def analyze_feature_vector_clusters(features_file_path, distance=utils.L2_distance):
    """
    Reads feature vectors and labels from a file and prints information about their clustering
    properties. Here, we think of the space of feature vectors, and consider a vector v_i to be in
    cluster j if j is one of the labels for example i.

    TEMPORARY: This function currently only runs in a reasonable amount of time for feature files
    of modest size. We recommend using the features file built from the small training dataset
    (~4000 samples).

    Arguments:
        features_file_path: (string) name of a file in which each line contains one example's features and labels
        distance: (function) a symmetric distance function on pairs of vectors
    """
    feature_vectors, label_vectors = utils.read_feature_and_label_vectors(features_file_path)

    logging.info('Building clusters...')

    # Map from (integer j) --> (list of indices i such that feature_vectors[i] is in cluster j)
    # Cluster 0 indicates no disease
    cluster_member_indices_for_cluster = OrderedDict((i, []) for i in range(15))

    for i, (features, labels) in enumerate(zip(feature_vectors, label_vectors)):

        # Record which disease classes (1-14) this example belongs to
        for j, label in enumerate(labels):
            if label == 1: cluster_member_indices_for_cluster[j+1].append(i)

        # Record whether this example belongs to no classes (i.e. no disease present)
        if all(label == 0 for label in labels):
            cluster_member_indices_for_cluster[0].append(i)

    logging.info('...done.')

    logging.info('Computing global and within-cluster average distances')

    # Compute average distance between vectors overall
    global_average_distance = average_distance_between_vectors(feature_vectors, distance)
    logging.info('Global average ' + distance.__name__ + ' between vectors: ' + str(global_average_distance))

    # Compute average distance within each cluster
    for j, vector_indices in cluster_member_indices_for_cluster.items():
        vectors_in_cluster = [feature_vectors[index] for index in vector_indices]
        average_cluster_distance = average_distance_between_vectors(vectors_in_cluster, distance)
        logging.info('Average ' + distance.__name__ + ' between vectors in cluster ' + str(j) + ': ' + str(average_cluster_distance))



if __name__ == '__main__':

    # Load user arguments
    arguments = argument_parser.parse_args()

    # Load hyperparameters from JSON file
    parameters = utils.Params(os.path.join(arguments.model_directory, 'params.json'))

    # Record whether GPU is available
    parameters.cuda = torch.cuda.is_available()

    # Set random seed for reproducible experiments
    torch.manual_seed(230)
    if parameters.cuda: torch.cuda.manual_seed(230)

    # Configure logger
    utils.set_logger(os.path.join(arguments.features_directory, 'analyze_feature_vectors.log'))

    # Features file should be under features_directory; prepend 'small_' if user specifies '--small'
    features_file_name = ('small_' if arguments.small else '') + arguments.features_file
    features_file_path = os.path.join(arguments.features_directory, features_file_name)

    # Extract feature vectors and write out to user-specified file (if such file does not yet exist)
    if os.path.isfile(features_file_path):
        logging.info('Features file detected; skipping feature extraction')
    else:
        logging.info('Features file not detected; now extracting features...')

        # Create data loader for training data
        logging.info('Loading ' + ('small ' if arguments.small else '') + 'train dataset...')
        train_data_loader = data_loader.fetch_dataloader(['train'], arguments.data_directory, parameters, arguments.small)['train']
        logging.info('...done.')

        # Configure model
        model = net.DenseNet169(parameters, return_features=True)
        if parameters.cuda: model = model.cuda()

        # Load weights from trained model
        utils.load_checkpoint(os.path.join(arguments.model_directory, arguments.restore_file + '.pth.tar'), model)

        extract_feature_vectors(model, train_data_loader, parameters, features_file_path)
        logging.info('...done.')

    # Read feature vectors and labels and print information about them
    logging.info('Analyzing features...')
    analyze_feature_vector_clusters(features_file_path, distance=utils.L2_distance)
    analyze_feature_vector_clusters(features_file_path, distance=utils.L1_distance)
    logging.info('...done.')




