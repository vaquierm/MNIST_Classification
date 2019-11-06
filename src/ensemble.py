import glob
import os
import numpy as np
from scipy.stats import mode
from src.util.fileio import load_training_labels, save_kaggle_results
from src.config import results_path, ensemble_folder


def load_predictions():
    """
    loads all the kaggle predictions into one nparray
    :return: nparray
    """
    predictions = list()
    for file in glob.glob(os.path.join(results_path, ensemble_folder, "*.csv")):
        predictions.append(load_training_labels(file))
    return np.vstack(tuple(predictions))


def get_majority_vote(predictions):
    """
    returns the most popular prediction
    :param predictions: a numpy matrix with all the predictions
    :return: a numpy array
    """
    majority = mode(predictions)[0][0]
    return majority


def get_min_predictions(predictions):
    """
    returns the minimum prediction of all the predictions
    :param predictions: a numpy matrix with all the predictions
    :return: a numpy array
    """
    minimum = np.amin(predictions, axis=0)
    return minimum


def get_max_predictions(predictions):
    """
    returns the minimum prediction of all the predictions
    :param predictions: a numpy matrix with all the predictions
    :return: a numpy array
    """
    maximum = np.amax(predictions, axis=0)
    return maximum


def get_ensemble_predictions():
    predictions = load_predictions()
    save_kaggle_results(os.path.join(results_path,  "majority_prediction.csv"), get_majority_vote(predictions))
    save_kaggle_results(os.path.join(results_path,  "maximum_prediction.csv"), get_max_predictions(predictions))
    save_kaggle_results(os.path.join(results_path,  "minimum_prediction.csv"), get_min_predictions(predictions))


if __name__ == '__main__':
    get_ensemble_predictions()
