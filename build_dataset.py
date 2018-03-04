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
import os

from PIL import Image, ImageFile
from tqdm import tqdm


# Configure user arguments for this script
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/images', help="Directory containing the chest X-ray dataset")
parser.add_argument('--output_dir', default='data/224x224_images', help="Where to store the processed dataset")
parser.add_argument('--output_size',
                    default=224,
                    help='The pixel dimension of each side of the square output images',
                    type=int)


def resize_and_save(file_path, output_dir, size):
    """
    Resizes the image contained in 'file_path' to (size)x(size) and saves it (with the same name) in
    the directory 'output_dir'.
    """
    image = Image.open(file_path)
    image = image.resize((size, size), Image.BILINEAR) # Use bilinear interpolation instead of default "nearest neighbor" method
    image.save(os.path.join(output_dir, file_path.split('/')[-1]))


if __name__ == '__main__':
    # Parse user-provided arguments
    arguments = parser.parse_args()

    # Ensure that 'data_dir' is a directory
    assert os.path.isdir(arguments.data_dir), "Couldn't find the dataset at {}".format(arguments.data_dir)

    # Get the file paths of images in dataset directory
    file_names = os.listdir(arguments.data_dir)
    file_paths = [os.path.join(arguments.data_dir, file_name) for file_name in file_names if file_name.endswith('.png')]

    # Create the output directory; proceed with a warning if it already exists
    if not os.path.exists(arguments.output_dir):
        os.mkdir(arguments.output_dir)
    else:
        print("Warning: output dir {} already exists".format(arguments.output_dir))

    ImageFile.LOAD_TRUNCATED_IMAGES = True # Fixes an issue in which PIL errors on some images

    # Save files in output directory
    for file_path in tqdm(file_paths):
        resize_and_save(file_path, arguments.output_dir, size=arguments.output_size)

    print("Done building dataset")
