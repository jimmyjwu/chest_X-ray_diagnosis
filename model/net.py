"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score


class DenseNet169(nn.Module):
    """
    The DenseNet169 model, with the classification layer modified to output 14 classes and forward()
    modified to also return feature vectors.

    Note: This model includes a sigmoid in its last layer. Do not place a sigmoid in the loss or
    accuracy functions.
    """
    def __init__(self, parameters, return_features=False):
        """
        Initializes the layers of the model.

        Arguments:
            parameters.out_size: (int) number of output classes
            output_features: (bool) whether forward() should output the feature vector for a given
                input in addition to the prediction
        """
        super(DenseNet169, self).__init__()

        # Record whether user wants forward() to return feature vectors
        self.return_features = return_features

        # Obtain a standard DenseNet169 model pre-trained on ImageNet
        self.densenet169 = torchvision.models.densenet169(pretrained=True)

        # Train only the last few/classification layers
        # for parameter in self.densenet169.parameters():
        #     parameter.requires_grad = False

        # By default, the input to the final layer has size 1024
        number_of_features = self.densenet169.classifier.in_features

        # Replace the standard DenseNet169 last layer with a linear-sigmoid sequence with 14 outputs
        self.densenet169.classifier = nn.Sequential(
            nn.Linear(number_of_features, parameters.out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Runs a given input x through the network and returns:
            - The output/prediction for x
            - The feature vector for x, as defined in the figure below

        DenseNet runs the input through the following sequence of layers:
                 +----------+   +----+   +---------------+            +----------+
            x -->| features |-->|ReLU|-->|average pooling|--feature-->|classifier|--output-->
                 +----------+   +----+   +---------------+  vector    +----------+
        See documentation:
            https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

        Arguments:
            x: (Variable) a batch of images, of dimensions [batch_size, 3, 224, 224]

        Returns:
            output: (Variable) label probabilities for each image; dimensions [batch_size, 14]
        """
        feature_vector = self.densenet169.features(x)
        feature_vector = F.relu(feature_vector, inplace=True)
        feature_vector = F.avg_pool2d(feature_vector, kernel_size=7, stride=1).view(feature_vector.size(0), -1)

        output = self.densenet169.classifier(feature_vector)

        # Return the feature vector if desired by user
        if self.return_features:
            return output, feature_vector
        else:
            return output


class DenseNet121(nn.Module):
    """
    The CheXNet model, with forward() modified to also return feature vectors.

    Note: This model includes a sigmoid in its last layer. Do not place a sigmoid in the loss or
    accuracy functions.
    """
    def __init__(self, parameters, return_features=False):
        """
        Initializes the layers of the model.

        Arguments:
            parameters.out_size: (int) number of output classes
            output_features: (bool) whether forward() should output the feature vector for a given
                input in addition to the prediction
        """
        super(DenseNet121, self).__init__()

        # Record whether user wants forward() to return feature vectors
        self.return_features = return_features

        # Obtain a standard DenseNet121 model pre-trained on ImageNet
        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        # Train only the last few/classification layers
        # for parameter in self.densenet121.parameters():
        #     parameter.requires_grad = False

        # By default, the input to the final layer has size 1024
        number_of_features = self.densenet121.classifier.in_features

        # Replace the standard DenseNet121 last layer with a linear-sigmoid sequence with 14 outputs
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(number_of_features, parameters.out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Runs a given input x through the network and returns:
            - The output/prediction for x
            - The feature vector for x, as defined in the figure below

        DenseNet runs the input through the following sequence of layers:
                 +----------+   +----+   +---------------+            +----------+
            x -->| features |-->|ReLU|-->|average pooling|--feature-->|classifier|--output-->
                 +----------+   +----+   +---------------+  vector    +----------+
        See documentation:
            https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

        Arguments:
            x: (Variable) a batch of images, of dimensions [batch_size, 3, 224, 224]

        Returns:
            output: (Variable) label probabilities for each image; dimensions [batch_size, 14]
        """
        feature_vector = self.densenet121.features(x)
        feature_vector = F.relu(feature_vector, inplace=True)
        feature_vector = F.avg_pool2d(feature_vector, kernel_size=7, stride=1).view(feature_vector.size(0), -1)
        
        output = self.densenet121.classifier(feature_vector)

        # Return the feature vector if desired by user
        if self.return_features:
            return output, feature_vector
        else:
            return output


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 14 - output of the model
        labels: (Variable) dimension batch_size x 14 - label of every type of disease [0, 1] (1 represents contains such disease;)

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    weight = torch.mean(labels, 0)
    return F.binary_cross_entropy(outputs, labels, weight=weight)

    # -torch.sum(torch.add(torch.mul((1-weight), torch.mul(labels, torch.log(outputs))), 
        # torch.mul(weight, torch.mul(1-labels, torch.log(1-outputs)))))


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 14 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x 14 - label of every type of disease [0, 1] (1 represents contains such disease;)

    Returns: List of AUROCs of all classes.
    """
    AUROCs = []
    for i in range(outputs.shape[1]):
        AUROCs.append(roc_auc_score(labels[:, i], outputs[:, i]))
    return AUROCs

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
