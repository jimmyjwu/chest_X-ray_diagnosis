"""
This script trains the models.
"""
# Python modules
import argparse
import logging
import os

# Scientific and deep learning modules
import numpy
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

# Project modules
import utils
import model.net as net
import model.neg as neg
import model.data_loader as data_loader
from evaluate import evaluate



# Configure user arguments for this script
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--data_dir', default='data/224x224_images', help='Directory containing the dataset')
argument_parser.add_argument('--model_dir', default='experiments/base_model', help='Directory containing params.json')
argument_parser.add_argument('--restore_file',
                             default=None,
                             help='(Optional) File in --model_dir containing weights to load, e.g. "best" or "last"')
argument_parser.add_argument('--feature_dir', default='data/feature', help='Directory containing feature vector')
argument_parser.add_argument('-small',
                             action='store_true', # Sets arguments.small to False by default
                             help='Use small dataset instead of full dataset')
argument_parser.add_argument('-fixChestNet',
                             action='store_true', # Sets arguments.fixChestNet to False by default
                             help='Use the training result of chexnet and does not change it')


def loadChestNet(model, loss_fn, data_loader, parameters):
    """
    Load feature vector output of chest net model to embedding

    Args:
        model: (torch.nn.Module) a neural network
        loss_fn: a function that takes indices_batch, pos_sample_batch, feature_vectors_batch, negative sampling number and computes the loss for the batch
        data_loader: (torch.utils.data.DataLoader) a DataLoader object that fetches data
        parameters: (Params) hyperparameters object
    """

    label_vectors = []
    # Use tqdm for progress bar
    with tqdm(total=len(data_loader)) as t:
        for i, (train_batch, pos_sample_batch, indices_batch, label_batch) in enumerate(data_loader):
            for i in range(label_batch.shape[0]):
                label_vectors.append(label_batch[i])
            # Move to GPU if available
            if parameters.cuda:
                train_batch = train_batch.cuda(async=True)
            
            # Convert to torch Variables
            train_batch, pos_sample_batch, indices_batch = Variable(train_batch), Variable(pos_sample_batch), Variable(indices_batch, requires_grad=False)

            # Compute model output and loss
            _, feature_vectors_batch = model(train_batch)
            loss = loss_fn(indices_batch, pos_sample_batch, feature_vectors_batch, 10)

            t.update()

    return label_vectors


def train(model, optimizer, loss_fn, data_loader, metrics, parameters, fixChestNet):
    """
    Trains a given model.

    Args:
        model: (torch.nn.Module) a neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes indices_batch, pos_sample_batch, feature_vectors_batch, negative sampling number and computes the loss for the batch
        data_loader: (torch.utils.data.DataLoader) a DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        parameters: (Params) hyperparameters object
    """
    # Set model to evaluate mode if fixChestNet; train mode otherwise
    if fixChestNet:
        model.eval()
    else:
        model.train()

    # Summary for current training loop and a running average object for loss
    summary = {}
    summary['loss'] = []
    summary['outputs'] = []
    summary['labels'] = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(data_loader)) as t:
        for i, (train_batch, pos_sample_batch, indices_batch, label_batch) in enumerate(data_loader):
            
            # Move to GPU if available
            if parameters.cuda:
                train_batch = train_batch.cuda(async=True)
            
            # Convert to torch Variables
            train_batch, pos_sample_batch, indices_batch = Variable(train_batch), Variable(pos_sample_batch), Variable(indices_batch, requires_grad=False)

            # Compute model output and loss
            _, feature_vectors_batch = model(train_batch)
            loss = loss_fn(indices_batch, pos_sample_batch, feature_vectors_batch, 10)
            # Clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # Perform updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % parameters.save_summary_steps == 0:
                # compute all metrics on this batch                
                summary['loss'].append(loss.data[0])

            # Update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute mean of all metrics in summary
    metrics_mean = {}
    metrics_mean['loss'] = sum(summary['loss']) / float(len(summary['loss']))
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, optimizer, scheduler, loss_fn, metrics, parameters, model_dir, feature_dir, restore_file=None, fixChestNet=False):
    """
    Trains a given model and evaluates each epoch against specified metrics.

    Args:
        model: (torch.nn.Module) a neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        parameters: (Params) hyperparameters object
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # Load weights from pre-trained model if specified
    if restore_file is not None:
        print("loading pre-trained wegiths...")
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    lebel_vectors = loadChestNet(model, loss_fn, train_dataloader, parameters)


    for epoch in range(parameters.num_epochs):

        # Train for one epoch
        logging.info('Epoch {}/{}'.format(epoch + 1, parameters.num_epochs))
        train(model, optimizer, loss_fn, train_dataloader, metrics, parameters, fixChestNet)
        if not fixChestNet:
            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()},
                                   is_best=is_best,
                                   checkpoint=model_dir)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, 'metrics_val_last_weights.json')
            utils.save_dict_to_json(val_metrics, last_json_path)

    feature_vectors = loss_fn.input_embeddings()
    write_feature_and_label_vectors(features_file_path, feature_vectors, label_vectors)


if __name__ == '__main__':

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
    utils.set_logger(os.path.join(arguments.model_dir, 'train_embedding.log'))

    # Create data loaders for training and validation data
    logging.info('Loading train datasets...')
    data_loaders = data_loader.fetch_dataloader(['train'], arguments.data_dir, parameters, arguments.small, False, True)
    train_data_loader = data_loaders['train']
    logging.info('...done.')

    # Configure model and optimizer
    model = net.DenseNet169(parameters, True).cuda() if parameters.cuda else net.DenseNet169(parameters, True)
    optimizer = optim.Adam(model.parameters(), lr=parameters.learning_rate, weight_decay=parameters.L2_penalty)

    # Configure schedule for decaying learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=parameters.learning_rate_decay_factor,
                                                     patience=parameters.learning_rate_decay_patience,
                                                     verbose=True) # Print message every time learning rate is reduced
    if arguments.small:
        num_train_data = 3924
    else:
        num_train_data = 78468
    loss = neg.NEG_loss(num_train_data, 1664)

    # Train the model
    logging.info('Starting training for {} epoch(s)'.format(parameters.num_epochs))
    train_and_evaluate(model, train_data_loader, optimizer, scheduler, loss, net.metrics, parameters, arguments.model_dir, arguments.feature_dir, arguments.restore_file, arguments.fixChestNet)
