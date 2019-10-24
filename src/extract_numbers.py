# This script loads all test data, extracts all numbers from each image, then saves it into (n, 3, 28, 28) np array that is then serialized

import os
import numpy as np

from src.config import data_path, training_images_file, testing_images_file, training_extracted_file, testing_extracted_file, MNIST_PIXEL, NUMBERS_PER_PICTURE
from src.data_processing.number_extraction import extract_k_numbers
from src.util.fileio import load_pkl_file, save_ndarray


def extract_numbers():
    """
    This method loads both the training and testing dataset, extracts three numbers out of each image to get a numpy array of size
    (Number of samples, 3, MNIST_WIDTH, MNIST_WIDTH)
    :return: Array containing only the images of the extracted numbers from each image
    """
    training_images_file_path = os.path.join(data_path, training_images_file)
    testing_images_file_path = os.path.join(data_path, testing_images_file)
    if not os.path.isfile(training_images_file_path):
        raise Exception("The training images file " + training_images_file_path + " does not exist")
    if not os.path.isfile(testing_images_file_path):
        raise Exception("The testing images file " + testing_images_file_path + " does not exist")

    print("\n\nExtracting numbers from both training and testing datasets")

    print("\tLoading training dataset")
    images = load_pkl_file(training_images_file_path)

    N = images.shape[0]

    print("\tLoading complete, extracting 3 numbers from all ", N, " images")

    images_extracted = np.empty((N, NUMBERS_PER_PICTURE, MNIST_PIXEL, MNIST_PIXEL))

    for i in range(images.shape[0]):
        if i % int(N / 5) == 0:
            print("\t\tCompleted extraction of ", i, " images")

            images_extracted[i] = extract_k_numbers(images[i], k=NUMBERS_PER_PICTURE, show_imgs=False)

    print("\tTraining data number extraction complete, saving data...")

    save_ndarray(os.path.join(data_path, training_extracted_file), images_extracted)

    print("\tLoading testing dataset")
    images = load_pkl_file(testing_images_file_path)

    N = images.shape[0]

    print("\tLoading complete, extracting 3 numbers from all ", N, " images")

    images_extracted = np.empty((N, NUMBERS_PER_PICTURE, MNIST_PIXEL, MNIST_PIXEL))

    for i in range(images.shape[0]):
        if i % int(N / 5) == 0:
            print("\t\tCompleted extraction of ", i, " images")

            images_extracted[i] = extract_k_numbers(images[i], k=NUMBERS_PER_PICTURE, show_imgs=False)

    print("\tTraining data number extraction complete, saving data...")

    save_ndarray(os.path.join(data_path, testing_extracted_file), images_extracted)

    print("All numbers extracted from images")


if __name__ == '__main__':
    extract_numbers()
