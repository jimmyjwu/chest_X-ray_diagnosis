
# Thoracic Disease Identification from Chest X-Rays

The goal of this project is to develop high-accuracy deep learning models for identifying 14 thoracic diseases from chest X-ray scans, as well as to localize the regions of the scans indicating disease.

__TODO: Update link below with our report.__
__Here is a [paper summarizing our methods and findings](https://google.com)__.


---
## How to Replicate Our Work

Notes:
- Unless stated otherwise, run all scripts from the top-level folder of this project.
- We recommend running this project from within an Amazon Web Services (AWS) Deep Learning Base AMI (Ubuntu), using a GPU-accelerated instance type such as `p2.xlarge`.


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
Run the script `build_dataset.py`, which resizes the images to 224x224. The resized dataset will be located by default in `data/224x224_images`.

```bash
python build_dataset.py --data_dir data/images --output_dir data/224x224_images
```



## Quickstart (~10 min)

1. __Build the dataset of size 224x224__: make sure you complete this step before training
```bash
python build_dataset.py --data_dir data/images --output_dir data/224x224_images
```

2. __Your first experiment__ We created a `base_model` directory for you under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```
For every new experiment, you will need to create a new directory under `experiments` with a similar `params.json` file.

3. __Train__ your experiment. Simply run
```
python train.py --data_dir data/224x224_images --model_dir experiments/base_model
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the validation set.

4. __Your first hyperparameters search__ We created a new directory `learning_rate` in `experiments` for you. Now, run
```
python search_hyperparams.py --data_dir data/224x224_images --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

5. __Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

6. __Evaluation on the test set__ Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```
python evaluate.py --data_dir data/224x224_images --model_dir experiments/base_model
```


## Guidelines for more advanced use

We recommend reading through `train.py` to get a high-level overview of the training loop steps:
- loading the hyperparameters for the experiment (the `params.json`)
- loading the training and validation data
- creating the model, loss_fn and metrics
- training the model for a given number of epochs by calling `train_and_evaluate(...)`

You can then have a look at `data_loader.py` to understand:
- how jpg images are loaded and transformed to torch Tensors
- how the `data_iterator` creates a batch of data and labels and pads sentences

Once you get the high-level idea, depending on your dataset, you might want to modify
- `model/net.py` to change the neural network, loss function and metrics
- `model/data_loader.py` to suit the data loader to your specific needs
- `train.py` for changing the optimizer
- `train.py` and `evaluate.py` for some changes in the model or input require changes here

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

## Resources

- [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)

[SIGNS]: https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view?usp=sharing
