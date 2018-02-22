"""
This script:
- (TODO) Extracts feature vectors from the training set and stores them in a file.
- (TODO) Analyzes the clustering properties of the feature vectors.
"""
# Python modules
import argparse
import logging
import os

# Scientific and deep learning modules
import numpy
import torch
from torch.autograd import Variable

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
                    help="Directory containing feature data and files useful for feature extraction")

argument_parser.add_argument('--restore_file',
                    default='CheXNet_model.pth.tar',
                    help="Name of the file in --features_directory containing weights to load")

argument_parser.add_argument('--features_file',
                    default='train_features_and_labels.txt',
                    help="Name of the file in --features_directory in which features should be saved")


def extract_feature_vectors(model, data_loader, parameters, features_file):
    """
    Extracts feature vectors from the training set and stores them in a file.

    Arguments:
        model: (torch.nn.Module) a neural network
        data_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        parameters: (Params) hyperparameters object
    """
    # A list that will eventually contain a NumPy feature vector for each example
    feature_vectors = []

    # Set model to evaluation mode
    model.eval()

    for i, (X_batch, Y_batch) in enumerate(test_loader):

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
        Store the feature vector for each example as a NumPy array
        Notes:
            - "features[i]" is the i-th training example packed into the features Variable
            - ".data" returns the Tensor that underlies the Variable
            - ".cpu()" moves the Tensor from the GPU to the CPU
            - ".numpy()" converts a Tensor to a NumPy array

        Since "features" is a [64, 1024]-size Variable, each "feature_vector" below is a 1024-length NumPy array
        """
        features_numpy = features.data.cpu().numpy()

        # For each example in the batch (i.e. each index in the first dimension of features_numpy),
        # write its feature vector and labels to a file
        for feature_vector in features_numpy:

            # TEMPORARY: Instead of writing to file, print to console
            print(feature_vector)

            # TEMPORARY: Break after first example
            break

        # TEMPORARY: Break after first batch
        break



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
    train_data_loader = data_loader.fetch_dataloader(['train'], arguments.data_directory, parameters)['train']
    logging.info("Done loading dataset")

    # Initialize the model, using CUDA if GPU available
    model = net.DenseNet121(parameters).cuda() if parameters.cuda else net.DenseNet121(parameters)

    # TEMPORARY: Wrap model in DataParallel to match CheXNet code to enable loading their weights
    model = torch.nn.DataParallel(model).cuda()

    # Reload weights from pre-trained CheXNet model file
    utils.load_checkpoint(os.path.join(arguments.features_directory, arguments.restore_file), model)
    
    # Extract feature vectors
    logging.info("Extracting features")
    extract_feature_vectors(model, train_data_loader, parameters, arguments.features_file)
    logging.info("Done extracting features")

