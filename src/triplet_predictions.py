import os
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from src.data_processing.data_loader import transform_to_trio_MNIST, prepare_for_model_training
from src.models.models import get_model
from src.config import models_path, results_path, NUM_CATEGORIES, MNIST_PIXEL, retrain_models, MODEL, NUMBERS_PER_PICTURE, REMOVE_BACKGROUND_TRIO, EPOCH, transfer_learning
from src.util.fileio import load_model, save_confusion_matrix, load_modified_MNIST_training, save_kaggle_results, load_modified_MNIST_test, save_training_history, dictionary_to_json


def run():
    if MODEL == "ResNet":
        raise Exception("The triplet predictions can only be done using the CNN, please change the MODEL parameter in the config file")

    print("Evaluating Triplet predictions with model " + MODEL + " and with background removal", REMOVE_BACKGROUND_TRIO)
    # Instantiate the appropriate model
    model = get_model(MODEL, input_shape=(MNIST_PIXEL, NUMBERS_PER_PICTURE * MNIST_PIXEL, 1), num_categories=NUM_CATEGORIES)
    model_path = os.path.join(models_path, "TRIPLET_" + MODEL + "_removeback" + str(REMOVE_BACKGROUND_TRIO) + ".h5")

    print("Loading modified MNIST train dataset")
    x_train, y_train = load_modified_MNIST_training()

    print("Transforming training set to Triplet set")
    x_triplet, y_triplet = transform_to_trio_MNIST(x_train, y_train)
    x_triplet = prepare_for_model_training(x_triplet)
    del x_train
    del y_train

    split = 0.8

    if not retrain_models:
        try:
            # Try to load the weights if we do not want to retrain
            load_model(model_path, model)
            model.summary()
        except:
            print("\tThe model file cannot be found at " + model_path + " so it will be retrained.")
            train(model, x_triplet, y_triplet, split)
    else:
        if transfer_learning:
            try:
                load_model(model_path, model)
                print("Transfer learning enabled, loaded old weights")
            except:
                print("Transfer learning enabled but no old weights exist")
        train(model, x_triplet, y_triplet, split)

    print("Predicting training data...")
    # predict the output
    y_pred = model.predict(x_triplet[int(x_triplet.shape[0] * split):]).argmax(axis=1)
    y_true = y_triplet[int(y_triplet.shape[0] * split):]

    # Save a confusion matrix
    conf_mat_file_path = os.path.join(results_path, "TRIPLET_" + MODEL + "_removeback" + str(REMOVE_BACKGROUND_TRIO) + "_confusion.png")
    save_confusion_matrix(confusion_matrix(y_true, y_pred), list(map(lambda x: str(x), range(10))),
                          conf_mat_file_path,
                          title="Triplet predictions with model " + MODEL + ", removed background: " + str(REMOVE_BACKGROUND_TRIO))

    print("Validation accuracy: ", accuracy_score(y_true, y_pred))

    produce_kaggle_results(model)


def produce_kaggle_results(model: Model):
    """
    Produces a csv file that can be submitted to kaggle
    :param model: model to predict
    """
    print("Loading modified MNIST test data...")
    x_test = load_modified_MNIST_test()

    x_test = transform_to_trio_MNIST(x_test)
    x_test = prepare_for_model_training(x_test)

    print("Predicting Kaggle test data...")
    y_pred = model.predict(x_test).argmax(axis=1)

    # Save the predicted values to the results folder
    print("Saving predictions...")
    results_file_path = os.path.join(results_path,
                                     "TRIPLET_" + MODEL + "_removeback" + str(REMOVE_BACKGROUND_TRIO) + "_predictions.csv")
    save_kaggle_results(results_file_path, y_pred)


def train(model: Model, x_triplet, y_triplet, split: float):
    """
    Trains the model with the triplet MNIST dataset and loads it with the best weights
    :param model: Model to be trained
    :param x_triplet: X triplet dataset
    :param y_triplet: Y triplet dataset
    :param split: Percentage of the data to be used for the training
    """
    model_path = os.path.join(models_path, "TRIPLET_" + MODEL + "_removeback" + str(REMOVE_BACKGROUND_TRIO) + ".h5")
    mc = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', verbose=1,
                         save_best_only=True)

    split = int(x_triplet.shape[0] * split)
    # Split into training ad testing set
    x_train = x_triplet[:split]
    y_train = y_triplet[:split]
    x_test = x_triplet[split:]
    y_test = y_triplet[split:]

    print("Training Triplet " + MODEL + " on with background removed as " + str(REMOVE_BACKGROUND_TRIO))
    history = model.fit(x=x_train, y=to_categorical(y_train), batch_size=128, epochs=EPOCH, verbose=2, callbacks=[mc], validation_data=(x_test, to_categorical(y_test)))

    # Save the training history
    save_training_history(history.history, os.path.join(results_path, "TRIPLET_" + MODEL + "_removeback" + str(REMOVE_BACKGROUND_TRIO) + "acc.png"), os.path.join(results_path, "TRIPLET_" + MODEL + "_removeback" + str(REMOVE_BACKGROUND_TRIO) + "loss.png"))
    dictionary_to_json(os.path.join(results_path, "TRIPLET_" + MODEL + "_removeback" + str(REMOVE_BACKGROUND_TRIO) + "results.json"), history.history)

    # Load the model with the best weights
    load_model(model_path, model)


if __name__ == '__main__':
    run()
