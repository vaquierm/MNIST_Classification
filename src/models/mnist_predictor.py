# This file creates all getters for models to predict the simple MNIST problem
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf

from src.config import MNIST_PIXEL


def get_model(model_name: str):
    """
    Returns a keras model corresponding to the model name
    :param model_name: Model name
    :return: The keras model
    """
    if model_name == "CNN":
        return get_CNN_model()
    else:
        raise Exception("The model name " + model_name + " is unknown")


def get_CNN_model():

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(MNIST_PIXEL, MNIST_PIXEL, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
