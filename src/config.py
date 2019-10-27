# This file contains all configurations like file paths and configurations to run

# General dir paths
data_path = "../data"
results_path = "../results"
models_path = "../models"

# Raw training data file names
training_labels_file_name = "train_max_y.csv"
training_images_file = "train_max_x"
testing_images_file = "test_max_x"

# MNIST predictor model names to evaluate (options: CNN)
MNIST_model_names = ["CNN"]
# MNIST datasets to run against the above models (options: MNITS, PROC_MNIST)
MNIST_datasets = ["PROC_MNIST"]
# If true, the models are retrained from scratch and the best models are saved to file
retrain_models = True


MNIST_PIXEL = 28
NUMBERS_PER_PICTURE = 3
NUM_CATEGORIES = 10
