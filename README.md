
# Thoracic Disease Identification from Chest X-Rays

The goal of this project is to develop high-accuracy deep learning models for identifying 14 thoracic diseases from chest X-ray scans, as well as to localize the regions of the scans indicating disease.

__Here is a [report summarizing our methods and findings](https://goo.gl/mua7zS)__. The rest of this page contains instructions for reproducing our results.


---
## How to replicate our work
Using our repository, you can train both deep learning (DenseNet) and shallow models on chest X-ray data, extract and analyze feature vectors, evaluate model performance, and tune hyperparameters in the same way as in our report.

To begin, log in to a UNIX-based machine. We recommend using an [Amazon Web Services (AWS) Deep Learning Base AMI (Ubuntu)](https://aws.amazon.com/marketplace/pp/prodview-dxk3xpeg6znhm), on a GPU-accelerated instance type such as `p2.xlarge`.

Clone this repository into your machine by running
```
git clone https://github.com/jimmyjwu/chest_X-ray_diagnosis.git
cd chest_X-ray_diagnosis
```
Unless otherwise stated, all of the commands in the rest of this document should be executed from within the directory `/chest_X-ray_diagnosis`. To learn about the user arguments for any script, run `[script name].py -h`.


### Prepare the project

#### Install dependencies
Ensure that [pip](https://pip.pypa.io/en/stable/) and [Anaconda](https://www.anaconda.com/) are installed. Then run
```
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

#### Download the dataset
The National Institutes of Health (NIH) hosts the dataset in [this Box folder](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345). Download all the `tar.gz` files under the `images` directory, and extract them to the folder `/data/images` in your local repository. Your directory structure should now look like
```
data/
    images/
        00000001_000.png
        00000001_001.png
        00000001_002.png
        ...
        (112,120 images in total)
```

#### Pre-process the dataset
The DenseNet architecture in our training code only accepts images of size 224x224. Resize the images in the dataset by running
```bash
python build_dataset.py
```
This downscales the images to 224x224 and stores them in `data/224x224_images`. (If desired, the new image size, source directory, and output directory can be specified via user arguments.) Your dataset is now ready for model training.

**Note:** This does not split the dataset into training/validation/test sets; instead, for benchmarking purposes, this repository fixes an [official split](https://github.com/yaoli/chest_xray_14) shared by researchers who have used this dataset.


### Train models

#### Train the DenseNet model
To train a DenseNet169 model on the dataset, ensure that all model hyperparameters are specified in the file `experiments/base_model/params.json`, then run
```bash
python train.py
```
This trains a DenseNet169 model on the dataset and stores the model parameters in a file.

#### Extract feature vectors
Once a DenseNet model has been trained, it can be used to extract feature vectors from the images in the dataset. To extract features, run
```bash
python analyze_feature_vectors.py
```
This runs the dataset through the trained DenseNet model and writes the feature vectors (defined as the output of the second-to-last layer of the DenseNet) to a file. It also prints information on the average distances between vectors in the dataset.

#### Train non-neural models
Once feature vectors have been extracted, they can be used as examples to train various shallow (non-neural) machine learning models. To do this, run
```bash
python classify_by_cluster.py
```
By default, this trains (on the training set) and evaluates (on the validation set) a random forest model and a k-nearest neighbors model.


### Evaluate models

#### Evaluate a DenseNet model
To evaluate your DenseNet model, run
```bash
python evaluate.py
```
This loads the DenseNet169 model trained previously, and evaluates it on the test set.

#### Evaluate non-neural models
To evaluate your non-neural models, run
```bash
python classify_by_cluster.py --dataset_type test
```
The argument `--dataset_type test` instructs the script to evaluate on the test set instead of the validation set (default).

#### Train and evaluate an ensemble model
You can also train and evaluate an _ensemble_ model whose predictions are an average of the predictions of individual models, such as a DenseNet and any number of non-neural models. To train an ensemble model, ensure that a DenseNet model has been trained and stored, and that its feature vectors have been extracted. Then run
```bash
python evaluate_ensemble.py
```
This loads the model and feature vectors, trains a set of non-neural machine learning models on the feature vectors, and evaluates the ensemble (average) of these models on the test set. By default, the only non-neural model trained by this command is a random forest.


### Run experiments

#### Perform hyperparameter search for the DenseNet
The file `experiments/params.json` contains baseline hyperparameter settings. It looks like this:
```json
{
    "learning_rate": 1e-4,
    "learning_rate_decay_factor": 0.2,
    "learning_rate_decay_patience": 1,
    ...
}
```

Suppose you want to search over different learning rates. Create a new directory under `experiments`, such as `experiments/learning_rate`, which will store all the results of this experiment. Then create a `params.json` file in this directory and populate it with the base hyperparameters you want to use for this experiment.

Now open the file `search_hyperparams.py`. This script contains the following code block, which creates a training job for each learning rate:
```python
# Perform search over learning rate
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]

for learning_rate in learning_rates:
    # Modify the relevant parameter in params
    params.learning_rate = learning_rate

    # Launch job (name has to be unique)
    job_name = "learning_rate_{}".format(learning_rate)
    launch_training_job(args.parent_dir, args.data_dir, small_flag, job_name, params)
```
Uncomment this code block and run
```
python search_hyperparams.py
```
The classification results for each choice of learning rate will be stored in a new directory under `experiments/learning_rate`. You can then display the results of this hyperparameter search in a table by running
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

#### Perform experiments on non-neural models
Currently, there is no automated script to perform hyperparameter search for non-neural models. However, this can be manually accomplished by modifying the code in the `main()` function of `classify_by_cluster.py`, then running that script.
