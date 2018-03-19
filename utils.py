import json
import logging
import os
import shutil
import torch

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


def write_feature_and_label_vectors(features_file_path, feature_vectors, label_vectors, number_of_features=1024):
    """
    Writes feature vectors and label vectors to a file.

    The file is formatted so that the i-th line contains the i-th example's features followed by
    labels, all separated by spaces.

    Arguments:
        features_file_path: (string) name of the file to write to
        feature_vectors: (list of NumPy arrays) where the i-th array is the i-th example's features
        label_vectors: (list of NumPy arrays) where the i-th array is the i-th example's labels
        number_of_features: (int) the number of feature values in each line of feature_file
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


def read_feature_and_label_vectors(features_file_path, number_of_features=1024):
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

        # Each line in features_file contains 1024 feature values followed by
        # 14 space-separated strings (either '0.0' or '1.0') indicating labels
        for line in features_file:
            features_and_labels = line.split()

            # Record features for this example in a NumPy array of floats
            feature_vectors.append(numpy.fromiter(features_and_labels[0:number_of_features], float))

            # Record classes for this example in a NumPy array of floats
            label_vectors.append(numpy.fromiter(features_and_labels[-14:], float))

    logging.info('...done.')
    return feature_vectors, label_vectors


def read_feature_and_label_matrices(features_and_labels_file_path, features_data_type=numpy.float64, labels_data_type=int, number_of_features=1024):
    """
    Reads feature vectors and label vectors from a file and returns them as X and y matrices
    suitable for immediate use by scikit-learn.

    The file must formatted as in the output of write_feature_and_label_vectors(), i.e. the i-th
    line contains the i-th example's features followed by labels, all separated by spaces.

    Arguments:
        features_file_path: (string) name of the file to read from
        number_of_features: (int) the number of feature values in each line of feature_file

    Returns:
        X: (2D NumPy array) where X[i,j] is the j-th feature value for the i-th example
        y: (2D NumPy array) where y[i,j] is 1 if the i-th example has label j, and 0 otherwise
    """
    feature_vectors, label_vectors = read_feature_and_label_vectors(features_and_labels_file_path, number_of_features)

    # Copy 1D NumPy arrays into new 2D NumPy arrays
    X = numpy.array(feature_vectors, dtype=features_data_type)
    y = numpy.array(label_vectors, dtype=labels_data_type)

    return X, y




