# This script will evaluate all simple MNIST models specified in the config file as well as the dataset to run them against
import os
import numpy as np
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score

from src.data_processing.MNIST import get_MNIST
from src.models.mnist_predictor import get_model
from src.config import NUM_CATEGORIES, retrain_models, models_path, results_path, MNIST_model_names, MNIST_datasets
from src.util.fileio import load_model, save_model_weights, plot_training_history, save_training_history, plot_confusion_matrix, save_confusion_matrix, dictionary_to_json
from keras.utils import to_categorical


def evaluate_MNIST_model(model_str: str, dataset: str, generate_results: bool = True, show_graphs: bool = False):
    """
    Evaluate the input model for the accuracy metric
    The results such as the confusion matrix will be saved to the results folder
    :param model_str: String code for the model to evaluate (CNN, RNN)
    :param dataset: String code for which dataset to train on (MNIST, PROC_MNIST)
    :param generate_results: If true, the results of the training are saved in the results folder
    :param show_graphs: If true, the graphs are shown to the user
    """
    print("\nEvaluating model " + model_str + " with dataset " + dataset)

    if not retrain_models:
        try:
            model = get_model(model_str)
            model_path = os.path.join(models_path, model_str + "_" + dataset + ".h5")
            load_model(model_path, model)
            model.summary()
        except:
            print("\tThe model file cannot be found at " + model_path + " so it will be retrained.")
            model = train_model(model_str, dataset)

    else:
        model = train_model(model_str, dataset)

    (x_test, y_test) = get_MNIST(dataset)[1]

    # Predict using the model
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    print("\nValidation accuracy:", accuracy_score(y_test, y_pred))

    if generate_results:
        conf_mat_file_path = os.path.join(results_path, model_str + "_" + dataset + "_confusion.png")
        save_confusion_matrix(confusion_matrix(y_test, y_pred), list(map(lambda x: str(x), range(10))), conf_mat_file_path, title="Confusion matrix of " + model_str + " with dataset " + dataset)
    if show_graphs:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred), list(map(lambda x: str(x), range(10))), title="Confusion matrix of " + model_str + " with dataset " + dataset)


def train_model(model_str: str, dataset: str, generate_results: bool = True, show_graphs: bool = False):
    """
    Train the model, generate graphs of training and validation error and loss per epoch
    :param model_str: String code for the model to evaluate (CNN, RNN)
    :param dataset: Dataset code for what data to train on (MNIST, PROC_MNIST)
    :param generate_results: If true, the results of the training are saved in the results folder
    :param show_graphs: If true, the graphs are shown to the user
    :return: The optimal model
    """

    print("\tTraining model " + model_str + " with dataset " + dataset)

    (x_train, y_train), (x_test, y_test) = get_MNIST(dataset)

    # Make the data categorical
    y_train = to_categorical(y_train, NUM_CATEGORIES)
    y_test = to_categorical(y_test, NUM_CATEGORIES)

    # Keep track of the validation and training accuracies as well as loss
    history = {'model': model_str, 'dataset': dataset, 'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

    model: Model = get_model(model_str)
    model.summary()
    model_path = os.path.join(models_path, model_str + "_" + dataset + ".h5")

    best_accuracy = 0.
    for i in range(50):

        # Perform one epoch
        model.fit(x=x_train, y=y_train, epochs=1, verbose=1)

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
        acc_img_path = os.path.join(results_path, model_str + "_" + dataset + "_acc.png")
        loss_img_path = os.path.join(results_path, model_str + "_" + dataset + "_loss.png")
        save_training_history(history, acc_img_path, loss_img_path)
        results_file_path = os.path.join(results_path, model_str + "_" + dataset + "_results.json")
        dictionary_to_json(results_file_path, history)

    load_model(model_path, model)

    return model


def evaluate_all_MNIST_models():
    print("\nEvaluating models for the simple MNIST problem")
    for dataset in MNIST_datasets:
        for model_str in MNIST_model_names:
            evaluate_MNIST_model(model_str, dataset, show_graphs=False, generate_results=True)


if __name__ == '__main__':
    evaluate_all_MNIST_models()
