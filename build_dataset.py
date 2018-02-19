"""resize images to 224x224.

The SIGNS dataset comes into the following format:
    train_signs/
        0_IMG_58224.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (1024, 1024).
Resizing to (224, 224) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
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
    image.save(os.path.join(output_dir, filename))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in each directory
    filenames = os.listdir(args.data_dir)
    filenames = [f for f in filenames if f.endswith('.png')]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    for filename in tqdm(filenames):
        resize_and_save(filename, output_dir, size=SIZE)

    print("Done building dataset")
