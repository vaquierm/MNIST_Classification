# This file creates all getters for models to predict the simple MNIST problem
from keras.models import Sequential, Model
import keras
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, BatchNormalization, AveragePooling2D, Activation, Input
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from keras.losses import categorical_crossentropy

from src.config import MNIST_PIXEL, NUM_CATEGORIES


def get_model(model_name: str, input_shape=(MNIST_PIXEL, MNIST_PIXEL, 1), num_categories=NUM_CATEGORIES):
    """
    Returns a keras model corresponding to the model name
    :param model_name: Model name
    :param input_shape: Shape of input img
    :param num_categories: Number of categories
    :return: The keras model
    """
    if model_name == "CNN":
        return get_CNN_model(input_shape, num_categories)
    else:
        raise Exception("The model name " + model_name + " is unknown")


# Model obtained from https://www.kaggle.com/ankur1401/digit-recognizer-with-cnn-using-keras
def get_CNN_model(input_shape=(MNIST_PIXEL, MNIST_PIXEL, 1), num_categories=NUM_CATEGORIES):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.4))
    model.add(Dense(num_categories, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
