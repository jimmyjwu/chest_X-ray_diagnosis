
# Thoracic Disease Identification from Chest X-Rays

The goal of this project is to develop high-accuracy deep learning models for identifying 14 thoracic diseases from chest X-ray scans, as well as to localize the regions of the scans indicating disease.

__Here is a [paper summarizing our methods and findings](https://goo.gl/mua7zS)__.


---
## How to Replicate Our Work

Notes:
- Unless stated otherwise, run all scripts from the top-level folder of this project.
- We recommend running this project from within an Amazon Web Services (AWS) Deep Learning Base AMI (Ubuntu), using a GPU-accelerated instance type such as `p2.xlarge`.
- To learn about the user arguments for each script, run `[script name].py -h`.


### Preparing the Project

#### Install Dependencies
Ensure that pip and Anaconda are installed. Then run:
```
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

#### Download the Dataset
The NIH hosts the dataset at this [Box folder](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345). Download all the tar.gz files in `images`, and extract them to the folder `/data/images`, so that we have:
```
data/
    images/
        00000001_000.png
        00000001_001.png
        00000001_002.png
        ...
        (112,120 images in total)
```

#### Pre-Process the Dataset
Run the script `build_dataset.py`, which resizes the images to 224x224 and stores them in `data/224x224_images`. (The new image size, source directory, and output directory can be adjusted via user arguments.)

```bash
python build_dataset.py
```


### Training Models

#### Train the DenseNet Model
Run the script `train.py`, which trains a DenseNet169 model on the dataset using the hyperparameters specified in `experiments/base_model/params.json`, then stores the parameters of the model in a file.
```bash
python train.py
```

#### Extract Feature Vectors
Run the script `analyze_feature_vectors.py`, which runs the dataset through the trained DenseNet model and writes the feature vectors (defined as the output of the second-to-last layer of the DenseNet) to a file. It also prints information on the average distances between vectors in the dataset.
```bash
python analyze_feature_vectors.py
```

#### Train Non-Neural Models
Run the script `classify_by_cluster.py`, which uses the feature vectors extracted from the DenseNet as inputs to various traditional machine learning models. By default, this script trains and evaluates (on the validation set) a random forest model and a k-nearest neighbors model.
```bash
python classify_by_cluster.py
```


### Evaluating Models

#### Evaluate the DenseNet Model
Run the script `evaluate.py`, which loads the DenseNet169 model trained previously, and evaluates it on the test set.
```bash
python evaluate.py
```

#### Evaluate the Non-Neural Models
Run the script `classify_by_cluster.py` with an appropriate argument to use the test set instead of the validation set:
```bash
python classify_by_cluster.py --dataset_type test
```

#### Evaluate the Ensemble Model
Run the script `evaluate_ensemble.py`, which loads the DenseNet169 model trained previously, trains any number of non-neural machine learning models on the feature vector set, and evaluates the ensemble of these models on the test set. By default, only non-neural model, namely a random forest, is used in this ensemble.
```bash
python evaluate_ensemble.py
```


### Running Experiments

#### Perform Hyperparameter Search for the DenseNet

Under the `experiments` directory you will find a file called `params.json`, which contains baseline hyperparameter settings. It looks like:
```json
{
    "learning_rate": 1e-4,
    "learning_rate_decay_factor": 0.2,
    "learning_rate_decay_patience": 1,
    ...
}
```

Suppose we wish to search over different learning rates. Create a new directory under `experiments`, say `experiments/learning_rate`, and create a similar `params.json` file under that directory. Then in `search_hyperparams.py`, uncomment the code block that creates a training job for each learning rate; this block looks like:
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
Now run `search_hyperparams.py`. The results for each choice of learning rate will be stored in a directory under `experiments/learning_rate`.

To display the results of this hyperparameter search in a table, run
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

#### Perform Experiments on Non-Neural Models

Currently, there is no automated script to perform hyperparameter search for non-neural models. Instead, manually adjust the code in the `main()` function of `classify_by_cluster.py` and then run that script.

