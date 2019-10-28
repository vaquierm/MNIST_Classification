import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.models import Model

from src.models.mnist_predictor import get_model
from src.data_processing.MNIST import prepare_for_model_training, transform_to_trio_MNIST
from src.config import NUM_CATEGORIES, MNIST_PIXEL, retrain_models, data_path, models_path, MNIST_model_names, results_path, training_images_file, training_labels_file_name
from src.util.fileio import load_model, load_pkl_file, load_training_labels, save_model_weights, plot_training_history, save_training_history, plot_confusion_matrix, save_confusion_matrix, dictionary_to_json


def evaluate_trio_MNIST_model(model_str: str, generate_results: bool = True, show_graphs: bool = False):
    """
    Evaluate a model predicting on all three extracted numbers at once
    :param model_str: Name of model to use
    :param generate_results: Generate results such as confusion matrix if True
    :param show_graphs: If true show the graphs
    """
    print("\nEvaluating model " + model_str + " on the TRIO dataset")

    # Get the file paths of the training data
    training_images_file_path = os.path.join(data_path, training_images_file)
    training_labels_file_path = os.path.join(data_path, training_labels_file_name)

    # Load the training data
    X = load_pkl_file(training_images_file_path)
    Y = load_training_labels(training_labels_file_path)

    X, Y = transform_to_trio_MNIST(X, Y)

    X = prepare_for_model_training(X)

    split = int(X.shape[0] * 0.8)
    # Split into training ad testing set
    X_train = X[:split]
    y_train = Y[:split]
    X_test = X[split:]
    y_test = Y[split:]

    del X
    del X

    # If the models need to be trained, do so
    if not retrain_models:
        try:
            model = get_model(model_str)
            model_path = os.path.join(models_path, model_str + "_TRIO.h5")
            load_model(model_path, model)
            model.summary()
        except:
            print("\tThe model file cannot be found at " + model_path + " so it will be retrained.")
            model = train_model(model_str, X_train, y_train, X_test, y_test)

    else:
        model = train_model(model_str, X_train, y_train, X_test, y_test)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\nValidation accuracy:", accuracy_score(y_test, y_pred))

    if generate_results:
        conf_mat_file_path = os.path.join(results_path, model_str + "_TRIO_confusion.png")
        save_confusion_matrix(confusion_matrix(y_test, y_pred), list(map(lambda x: str(x), range(10))),
                              conf_mat_file_path, title="Confusion matrix of " + model_str + " with dataset TRIO")
    if show_graphs:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred), list(map(lambda x: str(x), range(10))),
                              title="Confusion matrix of " + model_str + " with dataset TRIO")


def train_model(model_str: str, x_train=None, y_train=None, x_test=None, y_test=None, generate_results: bool = True, show_graphs: bool = False):
    """
    Train the model, generate graphs of training and validation error and loss per epoch
    :param model_str: String code for the model to evaluate (CNN, RNN)
    :param dataset: Dataset code for what data to train on (MNIST, PROC_MNIST)
    :param generate_results: If true, the results of the training are saved in the results folder
    :param show_graphs: If true, the graphs are shown to the user
    :return: The optimal model
    """

    if x_train is None and y_train is None and x_test is None and y_test is None:
        # Get the file paths of the training data
        training_images_file_path = os.path.join(data_path, training_images_file)
        training_labels_file_path = os.path.join(data_path, training_labels_file_name)

        # Load the training data
        X = load_pkl_file(training_images_file_path)
        Y = load_training_labels(training_labels_file_path)

        X, Y = transform_to_trio_MNIST(X, Y)

        X = prepare_for_model_training(X)

        split = int(X.shape[0] * 0.95)
        # Split into training ad testing set
        x_train = X[:split]
        y_train = Y[:split]
        x_test = X[split:]
        y_test = Y[split:]

    print("\tTraining model " + model_str + " with dataset TRIO")

    # Make the data categorical
    y_train = to_categorical(y_train, NUM_CATEGORIES)
    y_test = to_categorical(y_test, NUM_CATEGORIES)

    # Keep track of the validation and training accuracies as well as loss
    history = {'model': model_str, 'dataset': "TRIO", 'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

    model: Model = get_model(model_str, (MNIST_PIXEL, 3 * MNIST_PIXEL, 1))
    model.summary()
    model_path = os.path.join(models_path, model_str + "_TRIO.h5")

    best_accuracy = 0.
    for i in range(50):

        # Perform one epoch
        model.fit(x=x_train, y=y_train, epochs=1, verbose=0)

        # Evaluate the model
        results = model.evaluate(x_train, y_train, verbose=0)

        history['loss'].append(results[0])
        history['acc'].append(results[1])

        print("\t\tEpoch " + str(i+1) + "/50: training accuracy=" + str(results[1]), ", training loss=" + str(results[0]))

        results = model.evaluate(x_test, y_test, verbose=0)

        history['val_loss'].append(results[0])
        history['val_acc'].append(results[1])

        print("\t\tEpoch " + str(i+1) + "/50: validation accuracy=" + str(results[1]), ", validation loss=" + str(results[0]))

        if best_accuracy < results[1]:
            save_model_weights(model_path, model)
            best_accuracy = results[1]

    # Plot the training history if requested
    if show_graphs:
        plot_training_history(history)
    if generate_results:
        acc_img_path = os.path.join(results_path, model_str + "_TRIO_acc.png")
        loss_img_path = os.path.join(results_path, model_str + "_TRIO_loss.png")
        save_training_history(history, acc_img_path, loss_img_path)
        results_file_path = os.path.join(results_path, model_str + "_TRIO_results.json")
        dictionary_to_json(results_file_path, history)

    load_model(model_path, model)

    return model


def evaluate_all_trio_MNIST_model():
    for model_str in MNIST_model_names:
        evaluate_trio_MNIST_model(model_str)


if __name__ == '__main__':
    evaluate_all_trio_MNIST_model()
