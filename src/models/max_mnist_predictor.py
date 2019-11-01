from keras.models import Model
import numpy as np
from src.data_processing.number_extraction import extract_k_numbers
from src.data_processing.data_loader import prepare_for_model_training

from src.config import MNIST_PIXEL, NUMBERS_PER_PICTURE


class MaxMNISTPredictor:
    def __init__(self, model: Model):
        self.model = model

    def predict_max_num(self, x: np.ndarray):
        """
        predicts which number of three numbers in an array is the largest
        :param x: a n x 128 x 128 numpy array of images
        :return: a n x 1 numpy array
        """

        # isolate the 3 numbers in each image and save in test_x (n x 3 x 28 x 28 array)
        test_x = np.empty((x.shape[0], NUMBERS_PER_PICTURE, MNIST_PIXEL, MNIST_PIXEL))
        for i in range(x.shape[0]):
            test_x[i] = extract_k_numbers(x[i])

        test_x = prepare_for_model_training(test_x)

        # predict the number in each of the three images and place the highest of the three in y
        predicted_y = np.empty((x.shape[0], 3))
        for i in range(3):
            predicted_y[:, i] = np.argmax(self.model.predict(test_x[:, i]), axis=1)
        predicted_y = np.amax(predicted_y, axis=1)
        return predicted_y

