import cv2
from keras.datasets.mnist import load_data as load_MNIST
from keras.preprocessing.image import ImageDataGenerator


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

    # Iterate through each example and apply the same threshold applied to the number extraction
    for i in range(x_train.shape[0]):
        x_train[i] = cv2.threshold(x_train[i], 220, 255, cv2.THRESH_TOZERO)[1]

    for i in range(x_test.shape[0]):
        x_test[i] = cv2.threshold(x_test[i], 220, 255, cv2.THRESH_TOZERO)[1]

    datagen = ImageDataGenerator(rotation_range=40, zoom_range=0.1)

    x_train = prepare_for_model_training(x_train)
    x_test = prepare_for_model_training(x_test)

    flow = datagen.flow(x=x_train, y=y_train, batch_size=100000)

    (x_train, y_train) = flow.next()

    flow = datagen.flow(x=x_test, y=y_test, batch_size=20000)

    (x_test, y_test) = flow.next()

    return (x_train, y_train), (x_test, y_test)


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
