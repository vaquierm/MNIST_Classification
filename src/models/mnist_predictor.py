# This file creates all getters for models to predict the simple MNIST problem
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
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


def get_CNN_model(input_shape=(MNIST_PIXEL, MNIST_PIXEL, 1), num_categories=NUM_CATEGORIES):

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',
                     input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(num_categories, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    return model
