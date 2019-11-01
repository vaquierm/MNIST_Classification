# This file contains all configurations like file paths and configurations to run

# General dir paths
data_path = "../data"
results_path = "../results"
models_path = "../models"

# Raw training data file names
training_labels_file_name = "train_max_y.csv"
training_images_file = "train_max_x"
testing_images_file = "test_max_x"

################### General constants #################

MNIST_PIXEL = 28
NUMBERS_PER_PICTURE = 3
NUM_CATEGORIES = 10
MOD_MNIST_PIXEL = 128


############ Options specific to all methods ###################

# Model used to perform predictions (CNN or ResNet).
# The ResNet is only compatible with the fully unprocessed data.
MODEL = "CNN"
# Define the number of epochs to do
EPOCH = 50
# If true, the models are retrained from scratch and the best models are saved to file
retrain_models = True
# Enabling transfer learning will load an existing model if it already exists and continue the training from there
transfer_learning = False


############## Options for specific methods ######################

# Options for Isolated predictions (options: MNITS, PROC_MNIST)
ISOLATED_PRED_DATASET = "PROC_MNIST"

# Options for triplet predictions
REMOVE_BACKGROUND_TRIO = False
