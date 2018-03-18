"""
This script evaluates the models.
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


def evaluate(model, loss_fn, data_loader, metrics, parameters, use_tencrop=False):
    """
    Evaluates a given model.

    Args:
        model: (torch.nn.Module) a neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_loader: (torch.utils.data.DataLoader) a DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        parameters: (Params) hyperparameters object
        use_tencrop: (bool) whether to use ten-cropping to make predictions
    """
    # Set model to evaluation mode
    model.eval()

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

            if use_tencrop:
                # Due to ten-cropping, input batch is a 5D Tensor
                batch_size, number_of_crops, number_of_channels, height, width = input_batch.size()

                # Fuse batch size and crops
                input_batch = input_batch.view(-1, number_of_channels, height, width)

                # Compute model output
                output_batch_crops = model(input_batch)

                # Average predictions for each set of crops
                output_batch = output_batch_crops.view(batch_size, number_of_crops, -1).mean(1)

            else:
                # Compute model output
                output_batch = model(input_batch)

            # Compute loss
            loss = loss_fn(output_batch, labels_batch)

            # Convert prediction and label Variables to Tensors, move to CPU, and convert to NumPy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # Record predictions, labels, and losses for this batch
            summary['outputs'].append(output_batch)
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
    utils.set_logger(os.path.join(arguments.model_dir, 'evaluate.log'))

    # Create data loaders for test data
    logging.info('Loading test dataset...')
    test_dataloader = data_loader.fetch_dataloader(['test'], arguments.data_dir, parameters, arguments.small, arguments.use_tencrop)['test']
    logging.info('...done.')

    # Configure model
    model = net.DenseNet169(parameters).cuda() if parameters.cuda else net.DenseNet169(parameters)

    # Load weights from trained model
    utils.load_checkpoint(os.path.join(arguments.model_dir, arguments.restore_file + '.pth.tar'), model)

    # Evaluate the model
    logging.info('Starting evaluation')
    test_metrics, class_auroc = evaluate(model, net.loss_fn, test_dataloader, net.metrics, parameters, arguments.use_tencrop)
    utils.print_class_accuracy(class_auroc)
    save_path = os.path.join(arguments.model_dir, 'metrics_test_{}.json'.format(arguments.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
