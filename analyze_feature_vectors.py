"""
This script:
- (TODO) Extracts feature vectors from the training set and stores them in a file.
- (TODO) Analyzes the clustering properties of the feature vectors.
"""
# Python modules
import argparse
import logging
import os
import itertools
from collections import defaultdict

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
argument_parser.add_argument('--data_directory', default='data/224x224_images', help="Directory containing the dataset")
argument_parser.add_argument('--model_directory', default='experiments/base_model', help="Directory containing params.json")

argument_parser.add_argument('--features_directory',
                             default='feature_data',
                             help="Directory containing files related to feature extraction")

argument_parser.add_argument('--restore_file',
                             default='CheXNet_model.pth.tar',
                             help="File in --features_directory containing weights to load")

argument_parser.add_argument('--features_file',
                             default='train_features_and_labels.txt',
                             help="File in --features_directory to which features should be saved")

argument_parser.add_argument('-small',
                             action='store_true', # Sets arguments.small to False by default
                             help="Use small dataset instead of full dataset")


def extract_feature_vectors(model, data_loader, parameters, features_file):
    """
    Extracts feature vectors from the training set and stores them, along with labels, in a file.

    Arguments:
        model: (torch.nn.Module) a neural network
        data_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        parameters: (Params) hyperparameters object
        features_file: (file) a Python file object to which the features and labels should be written
    """
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

            """
            Convert the Variable (of size [batch_size, 1024]) of features for this batch to a NumPy array of the same size
            Notes:
                - ".data" returns the Tensor that underlies the Variable
                - ".cpu()" moves the Tensor from the GPU to the CPU
                - ".numpy()" converts a Tensor to a NumPy array
            """
            features_numpy = features.data.cpu().numpy()

            # Move the labels Tensor (of size [batch_size, 14]) to CPU and convert it to a NumPy array
            Y_numpy = Y_batch.cpu().numpy()

            # For each example in the batch, write its features and labels to a file
            for i in range(batch_size):

                # Concatenate the i-th example's features and labels
                features_and_labels = numpy.concatenate((features_numpy[i,:], Y_numpy[i,:]))

                # Convert feature/label values to strings and write them out as a space-separated line
                features_file.write(' '.join(map(str, features_and_labels)) + '\n')

            progress_bar.update()


def L2_distance(vector_1, vector_2):
    """
    Returns the L2/Euclidean distance between two vectors.
    """
    return numpy.linalg.norm(vector_1 - vector_2)


def L1_distance(vector_1, vector_2):
    """
    Returns the L1 distance between two vectors.
    """
    return numpy.linalg.norm(vector_1 - vector_2, ord=1)


def average_distance_between_vectors(vectors, distance):
    """
    Returns the average distance between pairs of vectors in a given list of vectors.
    """
    average_distance = utils.RunningAverage()
    for vector_1, vector_2 in itertools.combinations(vectors, r=2): # All pairs of vectors
        average_distance.update(distance(vector_1, vector_2))
    return average_distance()


def analyze_feature_vector_clusters(features_file_path, distance=L2_distance, number_of_features=1024):
    """
    Loads feature vectors and labels from a file and prints information about their clustering
    properties. Here, we think of the space of feature vectors, and consider a vector v_i to be in
    cluster j if j is one of the labels for example i.

    TEMPORARY: This function currently only runs in a reasonable amount of time for feature files
    of modest size. We recommend using the features file built from the small training dataset
    (~4000 samples).

    Arguments:
        features_file: (file) a file object in which each line contains one example's features and labels
        distance: (function) a symmetric distance function on pairs of vectors
        number_of_features: (int) the number of feature values in each line of feature_file
    """
    with open(features_file_path, 'r') as features_file:

        logging.info('Loading feature vectors and building clusters')

        # List of all feature vectors
        feature_vectors = []

        # Map from (integer j) --> (list of indices i such that feature_vectors[i] is in cluster j)
        cluster_member_indices_for_cluster = defaultdict(list)

        # Each line in features_file contains feature values followed by 14 0/1's indicating labels, separated by spaces
        for i, line in enumerate(features_file):
            features_and_labels = line.split()

            # Record features for this example, casting them as floats and placing them in a NumPy array
            feature_vectors.append( numpy.fromiter( map(float, features_and_labels[0:number_of_features]), float ) )

            # Record classes to which this example belongs
            for j, label in enumerate(features_and_labels[-14:]):
                
                # These numbers look like '1.0' or '0.0', so cast as float
                if float(label) == 1: cluster_member_indices_for_cluster[j].append(i)

        logging.info('Done loading feature vectors and building clusters')

        logging.info('Computing global and within-cluster average distances')

        # Compute average distance between vectors overall
        global_average_distance = average_distance_between_vectors(feature_vectors, distance)
        print('Global average ' + distance.__name__ + ' between vectors: ' + str(global_average_distance))

        # Compute average distance within each cluster
        for j, vector_indices in cluster_member_indices_for_cluster.items():
            vectors_in_cluster = [feature_vectors[index] for index in vector_indices]
            average_cluster_distance = average_distance_between_vectors(vectors_in_cluster, distance)
            print('Average ' + distance.__name__ + ' between vectors in cluster ' + str(j) + ': ' + str(average_cluster_distance))



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

    # Create data loader for training data
    logging.info("Loading dataset")
    train_data_loader = data_loader.fetch_dataloader(['train'], arguments.data_directory, parameters, arguments.small)['train']
    logging.info("Done loading dataset")

    # Initialize the model, using CUDA if GPU available
    # TEMPORARY: Use public CheXNet model instead of our own model so that we can load their weights
    model = net.CheXNet(parameters).cuda() if parameters.cuda else net.CheXNet(parameters)

    # TEMPORARY: Wrap model in DataParallel to match CheXNet code so that we can load their weights
    model = torch.nn.DataParallel(model).cuda()

    # TEMPORARY: Load weights from pre-trained CheXNet model file
    utils.load_checkpoint(os.path.join(arguments.features_directory, arguments.restore_file), model)
    
    logging.info("Extracting features")

    # Features file should be under features_directory; prepend 'small_' if user specifies '--small'
    features_file_name = ('small_' if arguments.small else '') + arguments.features_file
    features_file_path = os.path.join(arguments.features_directory, features_file_name)
    
    """
    TEMPORARY: Lines below commented out since AWS instance already has features file built
    # Extract feature vectors and write out to user-specified file
    with open(features_file_path, 'w') as features_file:
        extract_feature_vectors(model, train_data_loader, parameters, features_file)
    """

    logging.info("Done extracting features")

    logging.info("Analyzing features")

    # Read feature vectors and labels and print information about them
    analyze_feature_vector_clusters(features_file_path, distance=L2_distance)
    analyze_feature_vector_clusters(features_file_path, distance=L1_distance)

    logging.info("Done analyzing features")




