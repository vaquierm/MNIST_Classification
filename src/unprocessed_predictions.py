import os
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from src.data_processing.data_loader import prepare_for_model_training
from src.models.models import get_model
from src.config import models_path, results_path, NUM_CATEGORIES, retrain_models, MODEL, MOD_MNIST_PIXEL, EPOCH, transfer_learning
from src.util.fileio import load_model, save_confusion_matrix, load_modified_MNIST_training, save_kaggle_results, load_modified_MNIST_test, save_training_history, dictionary_to_json


def run():
    print("Evaluating predictions with model " + MODEL + " on unprocessed dataset")
    # Instantiate the appropriate model
    model = get_model(MODEL, input_shape=(MOD_MNIST_PIXEL, MOD_MNIST_PIXEL, 1),
                      num_categories=NUM_CATEGORIES)
    model_path = os.path.join(models_path, "UNPROCESSED_" + MODEL + ".h5")

    print("Loading modified MNIST train dataset")
    x_train, y_train = load_modified_MNIST_training()

    x_train = prepare_for_model_training(x_train)
    (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.15, random_state=1, shuffle=True, stratify=y_train)
    if not retrain_models:
        try:
            # Try to load the weights if we do not want to retrain
            load_model(model_path, model)
            model.summary()
        except:
            print("\tThe model file cannot be found at " + model_path + " so it will be retrained.")
            train(model, x_train, x_test, y_train, y_test)
    else:
        if transfer_learning:
            try:
                load_model(model_path, model)
                print("Transfer learning enabled, loaded old weights")
            except:
                print("Transfer learning enabled but no old weights exist")
        train(model, x_train, x_test, y_train, y_test)

    print("Predicting training data...")
    # predict the output
    y_pred = model.predict(x_test).argmax(axis=1)

    # Save a confusion matrix
    conf_mat_file_path = os.path.join(results_path, "UNPROCESSED_" + MODEL + "_confusion.png")
    save_confusion_matrix(confusion_matrix(y_test, y_pred), list(map(lambda x: str(x), range(10))),
                          conf_mat_file_path,
                          title="Predictions with model " + MODEL)

    print("Validation accuracy: ", accuracy_score(y_test, y_pred))

    produce_kaggle_results(model)


def produce_kaggle_results(model: Model):
    """
    Produces a csv file that can be submitted to kaggle
    :param model: model to predict
    """
    print("Loading modified MNIST test data...")
    x_test = load_modified_MNIST_test()
    x_test = prepare_for_model_training(x_test)

    print("Predicting Kaggle test data...")
    y_pred = model.predict(x_test).argmax(axis=1)

    # Save the predicted values to the results folder
    print("Saving predictions...")
    results_file_path = os.path.join(results_path, "UNPROCESSED_" + MODEL + "_predictions.csv")
    save_kaggle_results(results_file_path, y_pred)


def train(model: Model, x_train, x_test, y_train, y_test):
    """
    Trains the model with the modified MNIST dataset and loads it with the best weights
    :param model: Model to be trained
    :param x_train: Training X data from modified MNIST
    :param x_test: Test X data from modified MNIST
    :param y_train: Training Y data from modified MNIST
    :param y_test: Test Y data from modified MNIST
    """
    model_path = os.path.join(models_path, "UNPROCESSED_" + MODEL + ".h5")
    mc = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Make the y inputs categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the data generator to perform data augmentation
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(x_train)

    print("Training unprocessed data with " + MODEL)
    batch_size = 128
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=datagen.flow(x_test, y_test), epochs=EPOCH, steps_per_epoch=int(x_train.shape[0]/batch_size), verbose=2, callbacks=[mc])

    # Save the training history
    save_training_history(history.history, os.path.join(results_path, "UNPROCESSED_" + MODEL + "acc.png"), os.path.join(results_path, "UNPROCESSED_" + MODEL + "loss.png"))
    dictionary_to_json(os.path.join(results_path, "UNPROCESSED_" + MODEL + "results.json"), history.history)

    # Load the model with the best weights
    load_model(model_path, model)


if __name__ == '__main__':
    run()
