# This file creates all getters for models to predict the simple MNIST problem
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, BatchNormalization, AveragePooling2D, Activation, Input, add
from keras.optimizers import Adam


def get_model(model_name: str, input_shape, num_categories):
    """
    Returns a keras model corresponding to the model name
    :param model_name: Model name
    :param input_shape: Shape of input img
    :param num_categories: Number of categories
    :return: The keras model
    """
    if model_name == "CNN":
        model = get_CNN_model(input_shape, num_categories)
    elif model_name == "ResNet":
        model = get_ResNet_Model(input_shape, num_categories)
    else:
        raise Exception("The model name " + model_name + " is unknown")
    model.summary()
    return model


# Model obtained from https://www.kaggle.com/ankur1401/digit-recognizer-with-cnn-using-keras
def get_CNN_model(input_shape, num_categories):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_categories, activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# Model obtained from https://towardsdatascience.com/understanding-residual-networks-9add4b664b03
def get_ResNet_Model(input_shape, num_categories):
    # Define the unit that can pass the input tenser to the output to create the skip connection
    def Unit(x, filters, pool=False):
        res = x
        if pool:
            x = MaxPool2D(pool_size=(2, 2))(x)
            res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)
        out = BatchNormalization()(x)
        out = Activation("relu")(out)
        out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

        out = add([res, out])

        return out

    images = Input(input_shape)
    net = Conv2D(filters=32, kernel_size=[5, 5], strides=[1, 1], padding="same")(images)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)

    net = Unit(net, 64, pool=True)
    net = Unit(net, 64)
    net = Unit(net, 64)

    net = Unit(net, 128, pool=True)
    net = Unit(net, 128)
    net = Unit(net, 128)

    net = Unit(net, 256, pool=True)
    net = Unit(net, 256)
    net = Unit(net, 256)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.25)(net)

    net = AveragePooling2D(pool_size=(4, 4))(net)
    net = Flatten()(net)
    net = Dense(units=num_categories, activation="softmax")(net)

    model = Model(inputs=images, outputs=net)

    model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    return model
