import json
import logging
import os
import shutil
import torch
import random
from collections import OrderedDict

import numpy

CLASS_NAMES = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia'
]

# Fraction of the dataset having each label
LABEL_DISTRIBUTION = numpy.array([0.14026072, 0.03008273, 0.16194535, 0.2529456, 0.07834044,
                               0.0768363, 0.01667084, 0.06317373, 0.05602908, 0.02506894,
                               0.0260717, 0.02080722, 0.04662823, 0.00513913])


class Params():
    """
    Class that loads hyperparameters from a JSON file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """
    Class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return float(self.total) / max(self.steps, 1) # Prevent divide-by-zero error
        
    
def set_logger(log_path):
    """
    Configures a logger with a given log file path, so that any message sent to the logger via
            logging.info('Your message here')
    will be printed to the terminal as well as stored in the permanent log file.

    Arguments:
        log_path: (string) path to the file where logging messages should be stored
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """
    Saves a dictionary of floats in a JSON file

    Arguments:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """
    Saves weights and training parameters for a model in a file named
            checkpoint + 'last.pth.tar'
    If is_best is True, also saves these to
            checkpoint + 'best.pth.tar'

    Arguments:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Arguments:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def print_class_accuracy(class_accuracies):
    """
    Given a list of 14 accuracy numbers, records them to the log.
    """
    for class_name, class_accuracy in zip(CLASS_NAMES, class_accuracies):
        logging.info(class_name + ': ' + str(class_accuracy))


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


def _feature_and_label_matrices_from_lists(feature_vectors, label_vectors, features_data_type=numpy.float64, labels_data_type=int):
    """
    Given features and labels as lists of NumPy arrays, returns them as 2D NumPy arrays.

    Arguments:
        feature_vectors: (list of NumPy arrays) where the i-th array is the i-th example's features
        label_vectors: (list of NumPy arrays) where the i-th array is the i-th example's labels

    Returns:
        X: (2D NumPy array) where X[i,j] is the j-th feature value for the i-th example
        y: (2D NumPy array) where y[i,j] is 1 if the i-th example has label j, and 0 otherwise
    """
    logging.info('Converting features and labels from lists to NumPy matrices...')

    # Copy 1D NumPy arrays into new 2D NumPy arrays
    X = numpy.array(feature_vectors, dtype=features_data_type)
    y = numpy.array(label_vectors, dtype=labels_data_type)

    logging.info('...done.')
    return X, y


def _feature_and_label_lists_from_matrices(X, y):
    """
    Given features and labels as 2D NumPy arrays, returns them as lists of NumPy arrays.

    Arguments:
        X: (2D NumPy array) where X[i,j] is the j-th feature value for the i-th example
        y: (2D NumPy array) where y[i,j] is 1 if the i-th example has label j, and 0 otherwise

    Returns:
        feature_vectors: (list of NumPy arrays) where the i-th array is the i-th example's features
        label_vectors: (list of NumPy arrays) where the i-th array is the i-th example's labels
    """
    logging.info('Converting features and labels from NumPy matrices to lists...')

    assert X.shape[0] == y.shape[0], 'X and y must have the same number of rows'
    number_of_rows = X.shape[0]

    feature_vectors = [X[i,:] for i in range(number_of_rows)]
    label_vectors = [y[i,:] for i in range(number_of_rows)]

    logging.info('...done.')
    return feature_vectors, label_vectors


def write_feature_and_label_vectors(features_file_path, feature_vectors, label_vectors):
    """
    Writes feature vectors and label vectors to a file.

    The file is formatted so that the i-th line contains the i-th example's features followed by
    labels, all separated by spaces.

    Arguments:
        features_file_path: (string) name of the file to write to
        feature_vectors: (list of NumPy arrays) where the i-th array is the i-th example's features
        label_vectors: (list of NumPy arrays) where the i-th array is the i-th example's labels
    """
    logging.info('Writing feature and label vectors to {}...'.format(features_file_path))

    with open(features_file_path, 'w') as features_file:

        for features, labels in zip(feature_vectors, label_vectors):

            # Concatenate this example's features and labels
            features_and_labels = numpy.concatenate((features, labels))

            # Cast feature/label values to strings
            features_and_labels_strings = map(str, features_and_labels)

            # Convert feature/label values to strings and write them out as a space-separated line
            features_file.write(' '.join(features_and_labels_strings) + '\n')

    logging.info('...done.')


def read_feature_and_label_vectors(features_file_path, number_of_labels=14):
    """
    Reads feature vectors and label vectors from a file and returns them.

    The file must formatted as in the output of write_feature_and_label_vectors(), i.e. the i-th
    line contains the i-th example's features followed by labels, all separated by spaces.

    Arguments:
        features_file_path: (string) name of the file to read from
        number_of_features: (int) the number of feature values in each line of feature_file

    Returns:
        feature_vectors: (list of NumPy arrays) where the i-th array is the i-th example's features
        label_vectors: (list of NumPy arrays) where the i-th array is the i-th example's labels
    """
    logging.info('Loading feature and label vectors from {}...'.format(features_file_path))

    feature_vectors, label_vectors = [], []
    with open(features_file_path, 'r') as features_file:

        # Each line in features_file contains feature values (1024 for DenseNet121, 1664 for DenseNet169)
        # followed by 14 space-separated strings (either '0.0' or '1.0') indicating labels
        for line in features_file:
            features_and_labels = line.split()

            # Record features for this example in a NumPy array of floats
            feature_vectors.append(numpy.fromiter(features_and_labels[0:-number_of_labels], float))

            # Record classes for this example in a NumPy array of floats
            label_vectors.append(numpy.fromiter(features_and_labels[-number_of_labels:], float))

    logging.info('...done.')
    return feature_vectors, label_vectors


def read_feature_and_label_matrices(features_and_labels_file_path, features_data_type=numpy.float64, labels_data_type=int, number_of_labels=14):
    """
    Reads feature vectors and label vectors from a file and returns them as X and y matrices
    suitable for immediate use by scikit-learn.

    The file must formatted as in the output of write_feature_and_label_vectors(), i.e. the i-th
    line contains the i-th example's features followed by labels, all separated by spaces.

    Arguments:
        features_file_path: (string) name of the file to read from
        number_of_labels: (int) the number of label values in each line of feature_file

    Returns:
        X: (2D NumPy array) where X[i,j] is the j-th feature value for the i-th example
        y: (2D NumPy array) where y[i,j] is 1 if the i-th example has label j, and 0 otherwise
    """
    # Read features and labels from file
    feature_vectors, label_vectors = read_feature_and_label_vectors(features_and_labels_file_path, number_of_labels)

    # Convert features and labels into matrix form
    return _feature_and_label_matrices_from_lists(feature_vectors, label_vectors, features_data_type, labels_data_type)


def map_labels_to_example_indices(label_vectors):
    """
    Given a list of label vectors as NumPy arrays, returns a dictionary mapping
        (class number j from 0-14) --> (list of indices i such that label_vectors[i] is class j)
    where the class 0 is the "no disease" class.
    """
    indices_for_label = OrderedDict((i, []) for i in range(15))

    for i, labels in enumerate(label_vectors):

        # Record which disease classes (1-14) this example belongs to
        for j, label in enumerate(labels):
            if label == 1: indices_for_label[j+1].append(i)

        # Record whether this example belongs to no classes (i.e. no disease present)
        if all(label == 0 for label in labels):
            indices_for_label[0].append(i)

    return indices_for_label


def sample_examples_by_class(X, y, sample_fraction):
    """
    Given examples as matrices X and y, returns matrices X_sample and y_sample representing a random
    subset of the examples/rows, in such a way that every label occupies the same proportion of the
    sampled data as it does in the original data.
    """
    logging.info('Sampling a ' + str(sample_fraction) + ' fraction of the examples...')

    if sample_fraction >= 1: return X, y

    # Convert features and labels into list form
    feature_vectors, label_vectors = _feature_and_label_lists_from_matrices(X, y)

    # Map from (integer j) --> (list of indices i such that feature_vectors[i] is in cluster j)
    # Cluster 0 indicates no disease
    indices_for_label = map_labels_to_example_indices(label_vectors)

    # Sample a subset of each class
    sampled_indices_for_label = OrderedDict((i, set()) for i in range(15))
    for label, indices in indices_for_label.items():

        # The sampled set should have the full set scaled by the sample factor
        number_to_sample = int(len(indices) * sample_fraction)
        sampled_indices_for_label[i] = set(random.sample(indices, number_to_sample))

    # Union all the sampled indices into one list to remove duplicates
    sampled_indices = list(set().union(*indices_for_label.values()))

    # Keep only the rows at sampled indices
    X_sample = X[sampled_indices,:]
    y_sample = y[sampled_indices,:]

    logging.info('...done.')
    return X_sample, y_sample


