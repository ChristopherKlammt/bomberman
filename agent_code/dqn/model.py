from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.optimizers import Adam

from .parameters import LEARNING_RATE, INPUT_SHAPE

def create_model():
    NUM_ACTIONS = 6

    # create keras model
    model = Sequential()
    # model.add(Conv2D(8, input_shape=INPUT_SHAPE, kernel_size=(5, 5), padding="same", activation="relu"))
    # model.add(Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Flatten())
    model.add(Dense(64, input_shape=INPUT_SHAPE, kernel_initializer="random_normal", activation="relu"))
    # model.add(Dense(128, activation="relu"))
    model.add(Dense(NUM_ACTIONS, activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=LEARNING_RATE))

    return model
