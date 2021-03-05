import os
import pickle
import random

import numpy as np

from tensorflow import keras


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if os.path.exists("my_agent.model"):
        self.logger.info("Loading model from saved state.")
        self.model = keras.models.load_model("my_agent.model")
    elif self.train:
        self.logger.info("Models will be generated in train.py")
    else:
        self.logger.info("No trained model available.")
        exit()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        choice = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.logger.debug(f"Chose action: {choice}")
        return choice
    
    self.logger.debug("Querying model for action.")
    choice = np.random.choice(ACTIONS, p=self.model.predict(np.array([state_to_features(game_state)]))[0])
    # choice = ACTIONS[np.argmax(self.model.predict(np.array([state_to_features(game_state)]))[0])]
    self.logger.debug(self.model.predict(np.array([state_to_features(game_state)]))[0])
    self.logger.debug(f"Chose action: {choice}")
    return choice

VIEW_SIZE = 9

def field_to_small_view(field, x, y):
    step = int((VIEW_SIZE - 1) / 2)
    central_view = []
    for x_i in range(x - step, x + step + 1):
        row = []
        for y_i in range(y - step, y + step + 1):
            if x_i < 17 and y_i < 17:
                row.append(field[x_i][y_i])
            else:
                row.append(-1)
        central_view.append(row)
    central_view = np.array(central_view)
    return central_view


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []

    # chanels consist of: 
    #       - walls
    #       - coins
    # each channel is a 5x5 view of the field around the player

    name, score, bomb, (x_pos, y_pos) = game_state["self"]

    # create walls channel
    walls = field_to_small_view(game_state["field"], x_pos, y_pos)
    channels.append(walls)

    # create coins channel
    coin_map = np.zeros_like(game_state["field"])
    for x, y in game_state["coins"]:
        coin_map[x][y] = 1
    coins = field_to_small_view(coin_map, x_pos, y_pos)
    channels.append(coins)

    # create bomb channel
    bomb_map = np.zeros_like(game_state["field"])
    for (x, y), countdown in game_state["bombs"]:
        bomb_map[x][y] = countdown
    bombs = field_to_small_view(bomb_map, x_pos, y_pos)
    channels.append(bombs)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
