# MNIST_Classification
This repository implements a supervised classification model, which predicts the labels of the Modified MNIST dataset. This model was used in a [Kaggle competition](https://www.kaggle.com/c/modified-mnist/overview). 
## The Dataset
The Modified MNIST dataset consists of images with a dark background and three handwritten digits from 0 through 9. The digits  were taken from the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). The associated label for each image is the digit with the greatest numerical value. The Modified MNIST dataset can be found [here](https://www.kaggle.com/c/modified-mnist/data).
<p align="center">
<img src="https://github.com/arcaulfield/MNIST_Classification/blob/master/img/MNISTExample.png" width="400"/>
</p>

## The Optimal Strategy
The model with the highest classification accuracy used a Residual Network. The architecture for this model is a modified verison of one that was developped by [John Olafenwa](https://towardsdatascience.com/understanding-residual-networks-9add4b664b03). These modifications included:
* increasing the kernel size of the last average pooling layer, so that it acts as a global average pooling layer
* increasing the depth of the network<br />

In order to augment the data, an _ImageDataGenerator_ from the _Keras_ library was used. Images from the training set were randomly rotated between -10 and 10 degrees, translated by up to 10% of the image width and height, as well as zoomed in or out by up to 10%. <br />

Five models were trained using five different splits of the training data. Ensembling the predictions made from each model, using majority vote, achieved a Kaggle leaderboard accuracy of __99.133%__. The confusion matrix of this strategy is as follows:

<p align="center">
<img src="https://github.com/arcaulfield/MNIST_Classification/blob/master/results/UNPROCESSED_fold2_ResNet_confusion.png" width="400"/>
</p>

The accuracy and loss of the training and test datasets over 50 epochs are as follows:

<p float="left">
<img src="https://github.com/arcaulfield/MNIST_Classification/blob/master/results/UNPROCESSED_fold4_ResNet_loss.png" width="400"/>
<img src="https://github.com/arcaulfield/MNIST_Classification/blob/master/results/UNPROCESSED_fold4_ResNet_acc.png" width="400"/>
</p>

## How to Run the Program
1. Download the training data and the test data from [Kaggle](https://www.kaggle.com/c/modified-mnist/data), and place them in the `data/` folder
2. Open the `src/config.py` file and do the following:
    * While they shouldn't require modification, double check that all filepaths are ok. 
    * Select the model you would like to run. This could be `CNN`, for a convolutional neural netowrk, or `ResNet`, for a residual network. Update the `MODEL` variable accordingly. Note that the optimal strategy uses `ResNet`.
    * If you would like to retrain the model, let `retrain_models = True`.
    * If you would like to perform transfer learning, let `transfer_learning = True`.
    * Indicate the fold number you would like to run, by adjusting `FOLD_NUMBER` accordingly. 
3. Run the `unprocessed_predictions.py` script.
### How to Ensemble Predictions
1. Place all predictions to be ensembled in the `results/ensemble/` folder.
2. Run the `ensemble.py` script. 

## Directory Structure
```
.
├── data
│   
├── models
│
├── results
│   └── ensemble
│
└── src
    ├── config.py
    |
    ├── unprocessed_predictions.py
    ├── isolated_prediction.py
    ├── triplet_predictions.py
    |
    ├── ensemble.py
    |
    ├── data_analysis.ipynb
    ├── results_analysis.ipynb
    |
    ├── data_processing
    │   ├── data_loader.py
    │   └── number_extraction.py
    |
    ├── models
    │   ├── max_mnist_predictor.py
    │   └── models.py
    |
    └── utils
        └── fileio.py
```
The `data/` folder holds the training and testing data, in the form of .csv files. <br />

Any results are placed automatically in the `results/` folder. These results include confusion matrices, loss and accuracy graphs, as well as .csv files with predictions that can be submitted to Kaggle. <br />

* `results/ensemble/` contains all predictions, in the form of .csv files, to be ensembled. 

The `models/` folder contains models that have been trained. Newly trained models are automatically stored here, where they can then be used to make predictions or perform transfer learning. <br />

Files in `src/`:
* `config.py` defines which models are to be run, and allows for specific configurations. 
* `models/models.py` contains the implementations of both a convolutional neural network and a residual network.
* `unprocessed_predictions.py` has the scripts to train a model, using the unprocessed Modified MNIST dataset, and produce predictions in the form of Kaggle results. 
* `isolated_prediction.py` has the scripts to train a model, using the original MNIST dataset, and make predictions on the individual digits of each image. 
* `triplet_predictions.py` has the scripts to train a model and make predictions, using a modified version of the Modified MNIST dataset, where each image is the concatenation of the three isolated digits.
* `util/fileio.py` defines the functionalities needed for reading from and writing to files.
* `ensemble.py` includes the implementation of an ensemble method, which takes all predictions that are stored in the `results/ensemble/` folder and outputs a prediction in the form of a .csv file, which is placed in the `results/` folder. 
* `data_processing/` has the scripts necessary for different data processing strategies. 
* `data_analysis.ipynb` is a jupyter notebook that analyzes the dataset for this classification task.
* `results_analysis.ipynb` is a jupyter notebook that analyzes the types of errors made by the model. 

