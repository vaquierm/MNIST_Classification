import os

from src.evaluate_MNIST_models import train_model
from src.models.max_mnist_predictor import MaxMNISTPredictor
from src.util.fileio import save_kaggle_results, load_pkl_file, load_model
from src.config import results_path, kaggle_dataset, kaggle_model, data_path, testing_images_file, models_path, \
    retrain_models
from src.models.mnist_predictor import get_model
from src.data_processing.MNIST import get_MNIST

def generate_kaggle_results():

    print("\n\n Generating Kaggle submission with model: " + kaggle_model + " and dataset: " + kaggle_dataset)

    # Train model
    print("\tTraining model: " + kaggle_model + " with dataset: " + kaggle_dataset)
    if not retrain_models:
        try:
            model = get_model(kaggle_model)
            model_path = os.path.join(models_path, kaggle_model + "_" + kaggle_dataset + ".h5")
            load_model(model_path, model)
            model.summary()
        except:
            print("\tThe model file cannot be found at " + model_path + " so it will be retrained.")
            model = train_model(kaggle_model, kaggle_dataset)
    else:
        model = train_model(kaggle_model, kaggle_dataset)

    # Load the test data
    print("\tLoading test data...")
    test_images_file_path = os.path.join(data_path, testing_images_file)
    x_test = load_pkl_file(test_images_file_path)

    # Predict output
    print("\tPredicting data to model: " + kaggle_model)
    y_predicted = MaxMNISTPredictor(model).predict_max_num(x_test)

    # Save the predicted values to the results folder
    print("\tSaving predictions...")
    results_file_path = os.path.join(results_path, "predictions.csv")
    save_kaggle_results(results_file_path, y_predicted)


if __name__ == '__main__':
    generate_kaggle_results()