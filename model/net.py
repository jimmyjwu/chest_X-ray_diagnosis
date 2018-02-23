"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score

class DenseNet121(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(DenseNet121, self).__init__()
        self.out_size = params.out_size
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        # for param in self.densenet121.parameters():
        #     param.requires_grad = False
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, self.out_size)


    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 224 x 224 .

        Returns:
            out: (Variable) dimension batch_size x 14 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        s = self.densenet121(s)
        return s


class CheXNet(nn.Module):
    """
    The CheXNet model, with forward() modified to also return feature vectors.

    Note: Unlike the DenseNet121 model above, this model includes a sigmoid in its last layer.
    """
    def __init__(self, params):
        """
        Initializes the layers of the model.
        """
        super(CheXNet, self).__init__()

        # Obtain a standard DenseNet121 model pre-trained on ImageNet
        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        # By default, the input to the final layer has size 1024
        number_of_features = self.densenet121.classifier.in_features

        # Replace the standard DenseNet121 last layer with a linear-sigmoid sequence with 14 outputs
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(number_of_features, params.out_size),
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
        """
        feature_vector = self.densenet121.features(x)
        feature_vector = F.relu(feature_vector, inplace=True)
        feature_vector = F.avg_pool2d(feature_vector, kernel_size=7, stride=1).view(feature_vector.size(0), -1)
        
        output = self.densenet121.classifier(feature_vector)

        return output, feature_vector


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
    return nn.MultiLabelSoftMarginLoss()(outputs, labels)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 14 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x 14 - label of every type of disease [0, 1] (1 represents contains such disease;)

    Returns: (float) accuracy 1 x 14 in [0,1]
    """
    N_CLASSES = 14
    outputs = 1 / (1 + np.exp(-outputs))
    outputs = (outputs > 0.5)
    AUROCs = []
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
