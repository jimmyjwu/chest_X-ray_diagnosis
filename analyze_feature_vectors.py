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
                             help="Directory containing feature data and files useful for feature extraction")

argument_parser.add_argument('--restore_file',
                             default='CheXNet_model.pth.tar',
                             help="Name of the file in --features_directory containing weights to load")

argument_parser.add_argument('--features_file',
                             default='train_features_and_labels.txt',
                             help="Name of the file in --features_directory in which features should be saved")

argument_parser.add_argument('-small',
                    action='store_true', # Sets arguments.small to False by default
                    help="(Optional) Use small dataset instead of full dataset")


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
    
    # Extract feature vectors and write out to user-specified file
    with open(features_file_path, 'w') as features_file:
        extract_feature_vectors(model, train_data_loader, parameters, features_file)
    
    logging.info("Done extracting features")

