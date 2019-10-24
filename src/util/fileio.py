import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pkl_file(pkl_file_path: str):
    """
    Loads the pkl file into a 3D numpy array (num_samples, img_width, img_height)
    :param pkl_file_path: The file path of th pkl file
    :return: The numpy representation of all images
    """
    if not os.path.isfile(pkl_file_path):
        raise Exception("The pkl file " + pkl_file_path + " you are trying to load does not exist.")

    return pd.read_pickle(pkl_file_path).astype(np.uint8)


def save_ndarray(file_path: str, images: np.ndarray):
    """
    Saves the numpy array as a pkl file
    :param file_path: File path to save to
    :param images: Images as a numpy array
    """
    if not os.path.isdir(os.path.dirname(file_path)):
        raise Exception("The directory in which you want to save the file " + file_path + " does not exist.")

    np.save(file_path, images)


def load_ndarray(file_path: str):
    """
    Loads a numpy array from file
    :param file_path: File path of numpy array
    :return: The numpy array
    """
    if not os.path.isfile(file_path):
        raise Exception("The numpy array file " + file_path + " you are trying to load does not exist.")

    return np.load(file_path).astype(np.uint8)


def load_training_labels(file_path: str):
    """
    Loads the csv file containing the true labels to the training set
    :param file_path: The file path of the file containing true labels of training data
    :return: A numpy array containing all training labels
    """
    if not os.path.isfile(file_path):
        raise Exception("The training labels file " + file_path + " you are trying to load does not exist.")

    df = pd.read_csv(file_path)
    return np.array(df['Label'])


def show_images(images: list, titles=None):
    """
    Shows multiple images, prints them on the same plot in a line
    :param images: Images to be printed
    :param titles: Titles of all images
    """
    if titles is None:
        titles = []
    if len(images) == 0:
        raise Exception("You must input at least one image")
    if 0 < len(titles) != len(images):
        raise Exception("The number of images does not match the number of titles")
    if len(titles) == 0:
        titles = ["" for i in range(len(images))]

    _, axs = plt.subplots(nrows=1, ncols=len(images), figsize=(12, 12))
    axs = axs.flatten()
    for img, title, ax in zip(images, titles, axs):
        ax.title.set_text(title)
        ax.imshow(img, cmap=plt.get_cmap('gray'))


def show_image(image: np.ndarray, title: str = ""):
    """
    Shows an image represented as a np array
    :param image: numpy array representing image
    :param title: Title of image to be displayed
    """
    plt.title(title)
    if len(image.shape) == 2:
        plt.imshow(image, cmap=plt.get_cmap('gray'))
    elif len(image.shape) == 3:
        plt.imshow(image)
    else:
        raise Exception("You cannot print an image with shape", image.shape)
    plt.show()
