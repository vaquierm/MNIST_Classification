# This file can load the original MNIST dataset as well as a preporcessed version to match the distribution of the extracted images
import numpy as np
from keras.datasets.mnist import load_data as load_MNIST
from keras.preprocessing.image import ImageDataGenerator

from src.data_processing.number_extraction import extract_k_numbers, extract_3_and_paste
from src.config import MNIST_PIXEL, NUMBERS_PER_PICTURE


def get_MNIST(dataset_name: str):
    """
    Load either the processed MNIST dataset or the regular MNIST
    :param dataset_name: (MNIST or PROC_MNIST)
    :return: (x_train, y_train), (x_test, y_test)
    """
    if dataset_name == "MNIST":
        return get_original_MNIST()
    elif dataset_name == "PROC_MNIST":
        return get_processed_MNIST()
    else:
        raise Exception("The dataset " + dataset_name + " is not recognized")


def get_original_MNIST():
    """
    Get the regular MNIST dataset in the correct shape to be trained by a model
    :return: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = load_MNIST()

    x_train = prepare_for_model_training(x_train)
    x_test = prepare_for_model_training(x_test)

    return (x_train, y_train), (x_test, y_test)


def get_processed_MNIST():
    """
    Get the MNIST dataset thresholded the same way that the number extraction module
    thresholds. All data samples will be randomly rotated by an angle between 0 and 40 deg and zoomed in or out by a factor of 0.1
    :return: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = load_MNIST()

    # Iterate through each example, brighten the image by a random amount and apply the same processing as the extracted images
    for i in range(x_train.shape[0]):
        brightness_increase = np.random.randint(70, 85)
        x_train[i] = np.where((255 - x_train[i]) < brightness_increase, 255, x_train[i] + brightness_increase)
        x_train[i] = extract_k_numbers(x_train[i], k=1)[0]

    for i in range(x_test.shape[0]):
        brightness_increase = np.random.randint(70, 85)
        x_train[i] = np.where((255 - x_train[i]) < brightness_increase, 255, x_train[i] + brightness_increase)
        x_train[i] = extract_k_numbers(x_train[i], k=1)[0]

    datagen = ImageDataGenerator(rotation_range=45, zoom_range=0.15, horizontal_flip=False, vertical_flip=False)

    x_train = prepare_for_model_training(x_train)
    x_test = prepare_for_model_training(x_test)

    flow = datagen.flow(x=x_train, y=y_train, batch_size=100000)

    (x_train, y_train) = flow.next()

    flow = datagen.flow(x=x_test, y=y_test, batch_size=20000)

    (x_test, y_test) = flow.next()

    return (x_train, y_train), (x_test, y_test)


def transform_to_trio_MNIST(X: np.ndarray, Y: np.ndarray = None):
    """
    Transforms the input data to the trio MNIST format
    :param X: 128 x 128 unprocessed images from the modified MNIST dataset
    :param Y: Labels associated to X
    :return: X, Y in the form of the trio MNIST format. If Y is non we only return X
    """
    if Y is not None:
        X_trio = np.empty((X.shape[0] * 6, MNIST_PIXEL, NUMBERS_PER_PICTURE * MNIST_PIXEL))
        Y_trio = np.empty(Y.shape[0] * 6).astype(int)

        # Extract the numbers from each image
        for i in range(X.shape[0]):
            Y_trio[i * 6:i * 6 + 6] = Y[i]
            x_extracted = extract_3_and_paste(X[i])
            for j in range(6):
                X_trio[i * 6 + j] = x_extracted[j]

        return X_trio, Y_trio

    if Y is None:
        X_trio = np.empty((X.shape[0], MNIST_PIXEL, NUMBERS_PER_PICTURE * MNIST_PIXEL))

        # Extract the numbers from each image
        for i in range(X.shape[0]):
            X_trio[i] = extract_3_and_paste(X[i], get_permutations=False)

        return X_trio


def prepare_for_model_training(data):
    """
    Processes the data such that it can be fed to the keras model.
    The data is normalized between 0 and 1, and given the appropriate shape to be fed to keras (N, 28, 28, 1)
    :param data: numpy array representing the data (N, 28, 28)
    :return: Data ready to be fed to keras (N, 28, 28, 1)
    """
    data = data.reshape(data.shape + (1,))
    data = data.astype('float32')
    return data / 255
