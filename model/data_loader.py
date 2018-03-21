"""
This file contains tools for loading and transforming data for use by PyTorch models.

Tutorial on data loading: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import random
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Statistics from ImageNet dataset, to which we normalize our X-ray images
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

class_dist = np.array([0.14026072, 0.03008273, 0.16194535, 0.2529456, 0.07834044,
                     0.0768363, 0.01667084, 0.06317373, 0.05602908, 0.02506894, 
                     0.0260717, 0.02080722, 0.04662823, 0.00513913])

POS_NUM = 5
CLASS_NUM = 14

# Define an object that transforms a given training set example
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip image horizontally
    transforms.ToTensor(),              # Transform into a Torch tensor
    normalize                           # Normalize
])

# Define an object that transforms a given evaluation (i.e. validation or test) set example
# Note that we do not flip inputs during evaluation
evaluation_transform = transforms.Compose([
    transforms.ToTensor(),  # Transform into a Torch tensor
    normalize               # Normalize
])

"""
Define an object that transforms a given evaluation (i.e. validation or test) set example
including by taking ten crops of the image (ten-cropping)
Notes:
    - We do not flip inputs during evaluation
    - Due to the use of ten-cropping, each transformed image is a 4D tensor rather than 3D

See TenCrop documentation: http://pytorch.org/docs/master/torchvision/transforms.html
"""
evaluation_transform_with_tencrop = transforms.Compose([
    transforms.TenCrop(224),    # Crop image and its horizontal flip into five crops
    transforms.Lambda(          # Transform each crop into a Torch tensor
        lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])
    ),
    transforms.Lambda(          # Normalize each crop
        lambda crops: torch.stack([normalize(crop) for crop in crops])
    )
])


class ChestXRayDataset(Dataset):
    """
    A PyTorch Dataset that is for Chest X Ray.
    """
    def __init__(self, data_dir, image_list_file, transform):
        """
        Stores the filenames of the images in this dataset, and which transforms to apply on them.

        Arguments:
            data_dir: (string) directory containing the dataset
            image_list_file: (string) name of the file containing a list of image names and their labels
            transform: (torchvision.transforms) transformation to apply on each image
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                label = items[1:]
                label = [int(i) for i in label]
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Returns the image at index 'idx' (with transform applied), and its labels, from the dataset.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            labels: (Tensor) corresponding labels of image
        """
        image = Image.open(self.image_names[idx]).convert('RGB')  # PIL image
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(self.labels[idx])

class EmbeddingDataset(Dataset):
    """
    A PyTorch Dataset that is for Embedding.
    """
    def __init__(self, data_dir, image_list_file, transform):
        """
        Stores the filenames of the images in this dataset, and which transforms to apply on them.

        Arguments:
            data_dir: (string) directory containing the dataset
            image_list_file: (string) name of the file containing a list of image names and their labels
            transform: (torchvision.transforms) transformation to apply on each image
        """
        image_names = []
        labels = []
        classIndices = {i: [] for i in range(CLASS_NUM+1)}
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                label = items[1:]
                label = np.array([int(i) for i in label]) 
                labels.append(label)
                for classIdx in range(CLASS_NUM):
                    if label[classIdx] == 1:
                        classIndices[classIdx].append(len(labels)-1)
                if np.sum(label) == 0:
                    classIndices[CLASS_NUM].append(len(labels)-1)
        # print(classIndices)
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.classIndices = classIndices


    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Returns the image at index 'idx' (with transform applied), and its labels, from the dataset.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            labels: (Tensor) corresponding labels of image
        """
        image = Image.open(self.image_names[idx]).convert('RGB')  # PIL image
        if self.transform is not None:
            image = self.transform(image)
        return image, self.positive_sampling(idx), idx, torch.LongTensor(self.labels[idx])


    def positive_sampling(self, idx):

        if np.sum(self.labels[idx]) == 0:
            pos_sample = np.random.choice(self.classIndices[CLASS_NUM], POS_NUM, replace=False)
        else:

            class_num = self.labels[idx] * class_dist
            class_num = class_num / np.sum(class_num) * POS_NUM
            class_num_int = class_num.astype(int)
            pos_sample = np.zeros((POS_NUM, ))
            index = 0
            for classIdx in range(CLASS_NUM):
                if class_num[classIdx] > 0:
                    choice = np.random.choice(self.classIndices[classIdx], class_num_int[classIdx], replace=False)
                    pos_sample[index:index+choice.shape[0]] = choice
        return torch.LongTensor(pos_sample)





def fetch_dataloader(types, data_dir, parameters, small=False, use_tencrop=False, use_embedding=False):
    """
    Returns DataLoaders containing train, val, and/or test data from a given directory.

    Args:
        types: (list) a subset of the list ['train', 'val', 'test'] indicating which parts of dataset are desired
        data_dir: (string) name of directory containing the dataset
        parameters: (Params) hyperparameters object
        small: (bool) whether to use small dataset instead of full dataset
        use_tencrop: (bool) whether to use ten-cropping in val/test transforms

    Returns:
        data: (dict) a map from (a type in types) --> (a DataLoader object containing that data type)
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            image_file = os.path.join(data_dir, "..")

            # Obtain a list of images belonging to this split  
            # If user specified '-small' flag, obtain the small version of the train/val/test list          
            if small:
                labels_dir = "../labels/small_{}_list.txt".format(split)
            else:
                labels_dir = "../labels/{}_list.txt".format(split)
            image_list_file = os.path.join(data_dir, labels_dir)
            
            # Training set and val/test tests use different transforms and shuffling
            if split == 'train':
                transform = train_transform
                shuffle = True
            else:
                shuffle = False
                # If user specified '-use_tencrop' flag, use the evaluation transform that includes ten-cropping
                transform = evaluation_transform_with_tencrop if use_tencrop else evaluation_transform

            if use_embedding:
                dataset = EmbeddingDataset(data_dir=data_dir,
                                           image_list_file=image_list_file,
                                           transform=transform)

            else:
                dataset = ChestXRayDataset(data_dir=data_dir,
                                           image_list_file=image_list_file,
                                           transform=transform)

            dataloader = DataLoader(dataset=dataset,
                                    batch_size=parameters.batch_size, shuffle=shuffle,
                                    num_workers=parameters.num_workers,
                                    pin_memory=parameters.cuda)

            dataloaders[split] = dataloader

    return dataloaders
