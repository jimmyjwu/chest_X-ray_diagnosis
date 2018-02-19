"""resize images to 224x224.

The SIGNS dataset comes into the following format:
    images/
        00000805_000.png
        ...

Original images have size (1024, 1024).
Resizing to (224, 224) reduces the dataset size, and loading smaller images
makes training faster.

"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 224

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/images', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/224x224_images', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in directory
    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith('.png')]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Save files in output directory
    for filename in tqdm(filenames):
        resize_and_save(filename, args.output_dir, size=SIZE)

    print("Done building dataset")
