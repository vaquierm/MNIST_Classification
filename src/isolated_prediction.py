import os
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from src.data_processing.data_loader import get_MNIST
from src.models.models import get_model
from src.models.max_mnist_predictor import MaxMNISTPredictor
from src.config import models_path, results_path, NUM_CATEGORIES, MNIST_PIXEL, retrain_models, MODEL, ISOLATED_PRED_DATASET, EPOCH, transfer_learning
from src.util.fileio import load_model, save_confusion_matrix, load_modified_MNIST_training, load_modified_MNIST_test, save_kaggle_results, save_training_history, dictionary_to_json


def run():
    if MODEL == "ResNet":
        raise Exception("The triplet predictions can only be done using the CNN, please change the MODEL parameter in the config file")

    print("Evaluating Independent predictions with model " + MODEL + " with dataset " + ISOLATED_PRED_DATASET)
    # Instantiate the appropriate model
    model = get_model(MODEL, input_shape=(MNIST_PIXEL, MNIST_PIXEL, 1), num_categories=NUM_CATEGORIES)
    model_path = os.path.join(models_path, "ISOLATED_" + MODEL + "_" + ISOLATED_PRED_DATASET + ".h5")
    if not retrain_models:
        try:
            # Try to load the weights if we do not want to retrain
            load_model(model_path, model)
            model.summary()
        except:
            print("\tThe model file cannot be found at " + model_path + " so it will be retrained.")
            train(model)
    else:
        if transfer_learning:
            try:
                load_model(model_path, model)
                print("Transfer learning enabled, loaded old weights")
            except:
                print("Transfer learning enabled but no old weights exist")
        train(model)

    print("Loading modified MNIST training dataset...")
    x_test, y_test = load_modified_MNIST_training()

    print("Predicting training data...")
    # predict the output
    max_predictor = MaxMNISTPredictor(model)
    y_pred = max_predictor.predict_max_num(x_test)

    # Save a confusion matrix
    conf_mat_file_path = os.path.join(results_path, "ISOLATED_" + MODEL + "_" + ISOLATED_PRED_DATASET + "_confusion.png")
    save_confusion_matrix(confusion_matrix(y_test, y_pred), list(map(lambda x: str(x), range(10))),
                          conf_mat_file_path,
                          title="Isolated predictions with inner model " + MODEL + " trained on dataset " + ISOLATED_PRED_DATASET)

    print("Validation accuracy: ", accuracy_score(y_test, y_pred))

    produce_kaggle_results(model)


def produce_kaggle_results(model: Model):
    """
    Produces a csv file that can be submitted to kaggle
    :param model: model to predict
    """
    print("Loading modified MNIST test data...")
    x_test = load_modified_MNIST_test()

    print("Predicting Kaggle test data...")
    y_pred = MaxMNISTPredictor(model).predict_max_num(x_test)

    # Save the predicted values to the results folder
    print("Saving predictions...")
    results_file_path = os.path.join(results_path, "ISOLATED_" + MODEL + "_" + ISOLATED_PRED_DATASET + "_predictions.csv")
    save_kaggle_results(results_file_path, y_pred)


def train(model: Model):
    """
    Trains the model with the correct MNIST dataset and loads it with the best weights
    :param model: Model to be trained
    """
    model_path = os.path.join(models_path, "ISOLATED_" + MODEL + "_" + ISOLATED_PRED_DATASET + ".h5")
    mc = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', verbose=1,
                         save_best_only=True)

    (x_train, y_train), (x_test, y_test) = get_MNIST(ISOLATED_PRED_DATASET)

    print("Training " + MODEL + " on " + ISOLATED_PRED_DATASET + " dataset")
    history = model.fit(x=x_train, y=to_categorical(y_train), batch_size=128, epochs=EPOCH, verbose=2, callbacks=[mc], validation_data=(x_test, to_categorical(y_test)))

    # Save the training history
    save_training_history(history.history, os.path.join(results_path, "ISOLATED_" + MODEL + "_" + ISOLATED_PRED_DATASET + "acc.png"), os.path.join(results_path, "ISOLATED_" + MODEL + "_" + ISOLATED_PRED_DATASET + "loss.png"))
    dictionary_to_json(os.path.join(results_path, "ISOLATED_" + MODEL + "_" + ISOLATED_PRED_DATASET + "results.json"), history.history)

    # Load the model with the best weights
    load_model(model_path, model)


if __name__ == '__main__':
    run()
