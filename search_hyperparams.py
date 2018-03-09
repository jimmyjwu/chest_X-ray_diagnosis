"""
This script performs hyperparameter search.
"""
# Python modules
import argparse
import os
import sys
from subprocess import check_call

# Project modules
import utils


PYTHON = sys.executable

# Configure user arguments for this script
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir',
                    default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--data_dir',
                    default='data/224x224_images',
                    help='Directory containing the dataset')
parser.add_argument('-small',
                    action='store_true', # Sets arguments.small to False by default
                    help='Use small dataset instead of full dataset')


def launch_training_job(parent_dir, data_dir, small_flag, job_name, params):
    """
    Launches training of the model with a set of hyperparameters in parent_dir/job_name.

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir} {small_flag}".format(
            python=PYTHON, model_dir=model_dir, data_dir=data_dir, small_flag=small_flag)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    if args.small:
        small_flag = "-small"
    else:
        small_flag = ""
    
    # Perform hypersearch over one parameter
    # learning_rates = [1e-4, 1e-3, 1e-2]
    learning_rates = [1e-5, 5e-5]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.data_dir, small_flag, job_name, params)
