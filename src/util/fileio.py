import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model


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


def plot_training_history(history: dict):
    """
    Plot the validation and training accuracy/loss
    :param history: keras model history
    """
    plt.figure(3)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.figure(4)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def save_training_history(history: dict, acc_img_path: str, loss_img_path: str):
    """
    Save the validation and training accuracy/loss
    :param history: keras model history
    :param acc_img_path: File path to save accuracy image
    :param loss_img_path: File path to save loss image
    """
    plt.figure(1)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(acc_img_path)

    # Plot training & validation loss values
    plt.figure(2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(loss_img_path)


def plot_confusion_matrix(cm: np.ndarray, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Plots the configuration matrix
    :param cm: Confusion matrix
    :param classes: List of all classes
    :param normalize: Normalizes the confusion matrix if true
    :param title: Title of figure
    :param cmap: Color map of confusion matrix
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def save_confusion_matrix(cm: np.ndarray, classes,
                          fig_file_path: str,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Saves the configuration matrix to file
    :param cm: Confusion matrix
    :param classes: List of all classes
    :param fig_file_path: File path to save to
    :param normalize: Normalizes the confusion matrix if true
    :param title: Title of figure
    :param cmap: Color map of confusion matrix
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not os.path.isdir(os.path.dirname(fig_file_path)):
        raise Exception("The directory", os.path.dirname(fig_file_path), "you are trying to save your confusion matrix does not exist")

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(fig_file_path)


def save_model_weights(model_path: str, model: Model):
    """
    Saves the keras model weights to a file
    :param model_path: File path to save to
    :param model: The keras model
    """
    if not os.path.isdir(os.path.dirname(model_path)):
        raise Exception("The directory", os.path.dirname(model_path), "in which you want to save your model weights does not exist")
    model.save_weights(model_path)


def load_model(model_path: str, model: Model):
    """
    Loads the keras model weights from file to the model
    :param model_path: File path of keras model
    :param model: Model to load weights to
    """
    if not os.path.isfile(model_path):
        raise Exception("The file", model_path, "from which you are trying to load your model weights does not exist")
    return model.load_weights(model_path)


def dictionary_to_json(json_file_path: str, dictionary: dict):
    """
    Dump the dictionary in a json file
    :param json_file_path: Path of the JSON file to save to
    :param dictionary: Dictionary to save
    """
    if not os.path.join(os.path.dirname(json_file_path)):
        raise Exception("The directory " + os.path.dirname(json_file_path), " you are trying to save your JSON results to does not exist")

    with open(json_file_path, 'w') as fp:
        json.dump(dictionary, fp, indent=4)


def save_kaggle_results(result_file_path: str, Y):
    """
    Save the Kaggle predictions to a file
    :param result_file_path: File path to save to
    :param Y: Prediction results Y
    """
    if not os.path.isdir(os.path.dirname(result_file_path)):
        raise Exception("The directory " + os.path.dirname(result_file_path) + " to which you want to save Kaggle predictions does not exist")

    ids = np.arange(Y.shape[0])
    Y = list(map(lambda pred: str(int(pred)), Y))

    # Create a dataframe
    df = pd.DataFrame({'Id': ids, 'Label': Y})

    # Save to csv
    df.to_csv(result_file_path, mode='w', index=False, quoting=csv.QUOTE_NONNUMERIC)
