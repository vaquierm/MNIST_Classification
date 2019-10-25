from keras.datasets.mnist import load_data as load_MNIST
from src.data_processing.number_extraction import extract_k_numbers


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

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)


def get_processed_MNIST():
    """
    Get the MNIST dataset after going through the same preprocessing as the rectangle extracted from our
    training images
    :return: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = load_MNIST()

    for i in range(x_train.shape[0]):
        x_train[i] = extract_k_numbers(x_train[i], k=1)

    for i in range(x_test.shape[0]):
        x_test[i] = extract_k_numbers(x_test[i], k=1)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)
