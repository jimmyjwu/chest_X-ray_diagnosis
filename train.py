"""
This script trains the models.
"""
# Python modules
import argparse
import logging
import os

# Scientific and deep learning modules
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

# Project modules
import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate


# Configure user arguments for this script
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/224x224_images', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default=None,
                    help="(Optional) File in --model_dir containing weights to load") # 'best' or 'train'
parser.add_argument('-small',
                    action='store_true', # Sets args.small to False by default
                    help="Use small dataset instead of full dataset")


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Trains a given model.

    Args:
        model: (torch.nn.Module) a neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters object
    """
    # Set model to training mode
    model.train()

    # Summary for current training loop and a running average object for loss
    summ = {}
    summ['loss'] = []
    summ['outputs'] = []
    summ['labels'] = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            
            # Move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            
            # Convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # Compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # Clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # Perform updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                # compute all metrics on this batch                
                summ['loss'].append(loss.data[0])

            # Update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute mean of all metrics in summary
    metrics_mean = {}
    metrics_mean['loss'] = sum(summ['loss'])/float(len(summ['loss']))
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """
    Trains a given model and evaluates each epoch against specified metrics.

    Args:
        model: (torch.nn.Module) a neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters object
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # Load weights from pre-trained model if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_auroc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics, val_class_auroc = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_auroc = val_metrics['accuracy']
        is_best = val_auroc>=best_val_auroc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If this is the best AUROC thus far in training, print metrics for every class and save metrics to JSON file
        if is_best:
            logging.info("- Found new best accuracy: " + str(best_val_auroc))
            utils.print_class_accuracy(val_class_auroc)
            best_val_auroc = val_auroc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

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
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create data loaders for training and validation data
    logging.info("Loading the datasets...")
    data_loaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params, args.small)
    train_data_loader = data_loaders['train']
    validation_data_loader = data_loaders['val']
    logging.info("- done.")

    # Configure model and optimizer
    model = net.DenseNet121(params).cuda() if params.cuda else net.DenseNet121(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_data_loader, validation_data_loader, optimizer, net.loss_fn, net.metrics, params, args.model_dir, args.restore_file)
