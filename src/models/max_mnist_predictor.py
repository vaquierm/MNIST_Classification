from keras.models import Model
import numpy as np


class MaxMNISTPredictor:
    def __init__(self, model: Model):
        self.model = model

    def predict_max_num(self, x):
        """
        Predicts the number in each image in a n x 3 array and returns the maximum number in each row
        :param x: a n x 3 x 784 x 1 array of images of individual numbers (each row contains the 3 numbers that are in 1 image)
        :return: a n x 1 numpy array
        """
        predicted_y = np.empty(x.shape(0))
        for i in range(3):
            y = np.argmax(self.model.predict(x[:, i]), axis=1)
            np.append(predicted_y, y, axis=1)
        predicted_y = np.amax(predicted_y, axis=1)
        return predicted_y

