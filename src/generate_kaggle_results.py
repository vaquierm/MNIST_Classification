import os

from src.evaluate_MNIST_models import train_model as train_model_MNIST
from src.evaluate_trio_mnist import train_model as train_model_trio_MNIST
from src.models.max_mnist_predictor import MaxMNISTPredictor
from src.util.fileio import save_kaggle_results, load_pkl_file, load_model, show_image
from src.config import results_path, kaggle_dataset, kaggle_model, data_path, testing_images_file, models_path, \
    retrain_models, MNIST_PIXEL
from src.models.mnist_predictor import get_model
from src.data_processing.MNIST import transform_to_trio_MNIST, prepare_for_model_training


def generate_kaggle_results():

    print("\n\n Generating Kaggle submission with model: " + kaggle_model + " and dataset: " + kaggle_dataset)

    if not retrain_models:
        try:
            model = get_model(kaggle_model, (MNIST_PIXEL, 3 * MNIST_PIXEL, 1) if kaggle_dataset == "TRIO" else (MNIST_PIXEL, MNIST_PIXEL, 1))
            model_path = os.path.join(models_path, kaggle_model + "_" + kaggle_dataset + ".h5")
            load_model(model_path, model)
            model.summary()
        except:
            print("\tThe model file cannot be found at " + model_path + " so it will be retrained.")
            if kaggle_dataset == "TRIO":
                model = train_model_trio_MNIST(kaggle_model, generate_results=False)
            else:
                model = train_model_MNIST(kaggle_model, kaggle_dataset)
    else:
        if kaggle_dataset == "TRIO":
            model = train_model_trio_MNIST(kaggle_model, generate_results=False)
        else:
            model = train_model_MNIST(kaggle_model, kaggle_dataset)

    # Load the test data
    print("\tLoading test data...")
    test_images_file_path = os.path.join(data_path, testing_images_file)
    x_test = load_pkl_file(test_images_file_path)

    # Predict output
    print("\tPredicting data to model: " + kaggle_model)
    if kaggle_dataset == "TRIO":
        x_test = transform_to_trio_MNIST(x_test)
        x_test = prepare_for_model_training(x_test)
        y_predicted = model.predict(x_test).argmax(axis=1)
    else:
        y_predicted = MaxMNISTPredictor(model).predict_max_num(x_test)

    # Save the predicted values to the results folder
    print("\tSaving predictions...")
    results_file_path = os.path.join(results_path, "predictions.csv")
    save_kaggle_results(results_file_path, y_predicted)


if __name__ == '__main__':
    generate_kaggle_results()
