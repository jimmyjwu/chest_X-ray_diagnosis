"""
This script evaluates the models.
"""
# Python modules
import argparse
import logging
import os

# Scientific and deep learning modules
import numpy as np
import torch
from torch.autograd import Variable

# Project modules
import utils
import model.net as net
import model.data_loader as data_loader


# Configure user arguments for this script
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/224x224_images', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default='best',
                    help="(Optional) File in --model_dir containing weights to load")
parser.add_argument('-small',
                    action='store_true', # Sets args.small to False by default
                    help="Use small dataset instead of full dataset")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluates a given model.

    Args:
        model: (torch.nn.Module) a neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters object
    """
    # Set model to evaluation mode
    model.eval()

    # Summary for current eval loop
    summ = {}
    summ['loss'] = []
    summ['outputs'] = []
    summ['labels'] = []
    
    # Compute metrics over the dataset
    print("Computing metrics over the dataset")
    for data_batch, labels_batch in dataloader:

        # Move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        
        # Fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        # Compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # Extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        summ['outputs'].append(output_batch)
        labels_batch = labels_batch.data.cpu().numpy()
        summ['labels'].append(labels_batch)
        summ['loss'].append(loss.data[0])

    # Compute mean of all metrics in summary
    print("Computing mean of all metrics in summary")
    AUROCs = metrics['accuracy'](np.concatenate(summ['outputs']), np.concatenate(summ['labels']))
    metrics_mean = {}
    metrics_mean['accuracy'] = np.mean(AUROCs)
    metrics_mean['loss'] = sum(summ['loss'])/float(len(summ['loss']))

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    
    return metrics_mean, AUROCs


if __name__ == '__main__':
    """
    Evaluates the model on the test set.
    """
    # Load user arguments
    args = parser.parse_args()

    # Load hyperparameters from JSON file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Record whether GPU is available
    params.cuda = torch.cuda.is_available()

    # Set random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Configure logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))    

    # Create data loaders for test data
    logging.info("Loading the dataset...")
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params, args.small)
    test_dl = dataloaders['test']
    logging.info("- done.")

    # Configure model
    model = net.DenseNet121(params).cuda() if params.cuda else net.DenseNet121(params)

    # Load weights from trained model
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate the model
    logging.info("Starting evaluation")
    test_metrics, class_auroc = evaluate(model, net.loss_fn, test_dl, net.metrics, params)
    utils.print_class_accuracy(class_auroc)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
