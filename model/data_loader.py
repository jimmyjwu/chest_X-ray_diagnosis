import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
<<<<<<< HEAD
    transforms.Resize(224),  # resize the image to 224x224 (remove if images are already 224x224)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor(),  # transform it into a torch tensor
    normalize()])  # normalize

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(224),  # resize the image to 224x224 (remove if images are already 224x224)
    transforms.ToTensor(),  # transform it into a torch tensor
    normalize()])  # 


=======
                                        transforms.Resize(256),  # downscale the size
                                        transforms.TenCrop(224), # four corners and the central crop plus the horizon flipped
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
>>>>>>> 4e53b397accd3b3004bc22c3eac0e8da4d4a4a04
class SIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, image_list_file, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            image_list_file: (string) file list train and test names
            transform: (torchvision.transforms) transformation to apply on image
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
        # return size of dataset
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.image_names[idx]).convert('RGB')  # PIL image
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(self.labels[idx])


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            image_file = os.path.join(data_dir, "images")
            image_list_file = os.path.join(os.path.join(data_dir, "labels"),"{}_list.txt".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SIGNSDataset(data_dir=image_file,
                                             image_list_file=image_list_file,
                                             transform=train_transformer),
                                batch_size=params.batch_size, shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            else:
                dl = DataLoader(SIGNSDataset(data_dir=image_file,
                                             image_list_file=image_list_file,
                                             transform=eval_transformer), 
                                batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
