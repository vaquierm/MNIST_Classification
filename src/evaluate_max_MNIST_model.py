import os
from sklearn.metrics import confusion_matrix, accuracy_score

from src.models.mnist_predictor import get_model
from src.config import models_path, results_path, MNIST_model_names, MNIST_datasets, training_images_file, training_labels_file_name, data_path
from src.util.fileio import load_model, plot_confusion_matrix, save_confusion_matrix, dictionary_to_json
from src.evaluate_MNIST_models import train_model
from src.models.max_mnist_predictor import MaxMNISTPredictor
from src.util.fileio import load_pkl_file, load_training_labels


def evaluate_max_mnist_model(model_str: str, dataset: str, generate_results: bool = True, show_graphs: bool = False):
    """
    Evaluate the meta model MAX_MNIST
    The results such as the confusion matrix will be saved to the results folder
    :param model_str: String code for the model to evaluate (CNN, RNN)
    :param dataset: String code for which dataset to train on (MNIST, PROC_MNIST)
    :param generate_results: If true, the results of the training are saved in the results folder
    :param show_graphs: If true, the graphs are shown to the user
    """
    print("\nEvaluating MAX MNIST meta model " + model_str + " with dataset " + dataset)

    try:
        model = get_model(model_str)
        model_path = os.path.join(models_path, model_str + "_" + dataset + ".h5")
        load_model(model_path, model)
    except:
        print("\tThe model file cannot be found at " + model_path + " so it will be retrained.")
        model = train_model(model_str, dataset)

    training_images_file_path = os.path.join(data_path, training_images_file)
    training_labels_file_path = os.path.join(data_path, training_labels_file_name)

    x_test = load_pkl_file(training_images_file_path)
    y_test = load_training_labels(training_labels_file_path)
    # predict the output
    max_predictor = MaxMNISTPredictor(model)
    y_pred = max_predictor.predict_max_num(x_test)

    if generate_results:
        conf_mat_file_path = os.path.join(results_path, "MAX_MNIST_" +  model_str + "_" + dataset + "_confusion.png")
        save_confusion_matrix(confusion_matrix(y_test, y_pred), list(map(lambda x: str(x), range(10))),
                              conf_mat_file_path, title="MAX MNIST Confusion matrix with inner model " + model_str + " trained on dataset " + dataset)
    if show_graphs:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred), list(map(lambda x: str(x), range(10))),
                              title="MAX MNIST Confusion matrix with inner model " + model_str + " trained on dataset " + dataset)

    acc = accuracy_score(y_test, y_pred)
    print("\nValidation accuracy:", acc)

    return acc


def evaluate_all_max_mnist_models():
    print("\nEvaluating models for the MAX MNIST problem")
    results = {}
    for dataset in MNIST_datasets:
        for model_str in MNIST_model_names:
            results['MAX_MNIST_' + model_str + "_" + dataset] = evaluate_max_mnist_model(model_str, dataset,
                                                                                         show_graphs=False,
                                                                                         generate_results=True)
    result_file_path = os.path.join(results_path, "MAX_MNIST_results.json")
    dictionary_to_json(result_file_path, results)

if __name__ == '__main__':
    evaluate_all_max_mnist_models()
