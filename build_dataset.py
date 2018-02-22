"""
This script processes the raw dataset images and stores the processed images in a new directory.

This script does not split the dataset into train/val/test sets; instead, for benchmarking
purposes, the split is obtained from an official split shared by various researchers who have
worked on this dataset.

The chest X-ray dataset comes into the following format:
    images/
        00000805_000.png
        ...

The original images have size 1024x1024. This script resizes them to 224x224, which reduces the
dataset size from 43 GB to 2.5 GB and speeds up training.
"""
import argparse
import random
import os

from PIL import Image, ImageFile
from tqdm import tqdm

SIZE = 224

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/images', help="Directory containing the chest X-ray dataset")
parser.add_argument('--output_dir', default='data/224x224_images', help="Where to store the processed dataset")

def resize_and_save(filename, output_dir, size=SIZE):
    """
    Resizes the image contained in 'filename' and saves it (with the same name) in the directory 'output_dir'
    """
    image = Image.open(filename)
    image = image.resize((size, size), Image.BILINEAR) # Use bilinear interpolation instead of default "nearest neighbor" method
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    # Parse user-provided arguments
    args = parser.parse_args()

    # Ensure that 'data_dir' is a directory
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames of images in dataset directory
    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith('.png')]

    # Create the output directory; proceed with a warning if it already exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    ImageFile.LOAD_TRUNCATED_IMAGES = True # Fixes an issue in which PIL errors on some images

    # Save files in output directory
    for filename in tqdm(filenames):
        resize_and_save(filename, args.output_dir, size=SIZE)

    print("Done building dataset")
