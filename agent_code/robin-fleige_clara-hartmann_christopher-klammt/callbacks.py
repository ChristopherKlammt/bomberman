import os
import pickle
import random

import numpy as np

from tensorflow import keras

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# returns probabilites for an array of game_states
def get_valid_probabilities_list(self, states, features):
    probabilities = self.model.predict(np.array(features))
    self.logger.debug(probabilities)
    for i in range(len(probabilities)):
        if min(probabilities[i]) < 0:
            probabilities[i] += abs(min(probabilities[i]))
        probabilities[i] *= get_valid_actions(states[i]) # only allow valid actions
        probabilities[i] /= probabilities[i].sum() # normalize to statistical vector (= sums up to 1)
    return probabilities


def get_valid_probabilities(self, game_state):
    probabilities = get_valid_probabilities_list(self, [game_state], [state_to_features(game_state)])[0]
    return probabilities

def get_next_action(self, game_state):
    probabilities = get_valid_probabilities(self, game_state)
    choice = np.random.choice(ACTIONS, p=probabilities)
    # choice = ACTIONS[np.argmax(probabilities)]
    
    return probabilities, choice

def get_valid_actions(game_state):
    _, _, bomb, (x, y) = game_state["self"]
    walls = game_state["field"]
    bombs = list(map(lambda x: x[0], game_state["bombs"]))

    actions = np.ones(6)

    if walls[x][y-1] != 0 or (x, y-1) in bombs:
        actions[0] = 0 # can't go up
    if walls[x+1][y] != 0 or (x+1, y) in bombs:
        actions[1] = 0 # can't go right
    if walls[x][y+1] != 0 or (x, y+1) in bombs:
        actions[2] = 0 # can't go down
    if walls[x-1][y] != 0 or (x-1, y) in bombs:
        actions[3] = 0 # can't go left

    if True:
    # if not bomb:
        actions[5] = 0 # can't plant bomb
    
    return actions

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
    probabilities, choice = get_next_action(self, game_state)
    # choice = ACTIONS[np.argmax(self.model.predict(np.array([state_to_features(game_state)]))[0])]
    self.logger.debug(probabilities)
    self.logger.debug(f"Chose action: {choice}")
    return choice


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
    #       - crates
    #       - coins
    #       - players
    #       - bombs

    # create walls channel
    wall_map = np.zeros_like(game_state["field"])
    for x, y in zip(*np.where(game_state["field"] == -1)):
        wall_map[x][y] = 1
    channels.append(wall_map)

    # create crates channel
    crate_map = np.zeros_like(wall_map)
    for x, y in zip(*np.where(game_state["field"] == 1)):
        crate_map[x][y] = 1
    channels.append(crate_map)    

    # create coins channel
    coin_map = np.zeros_like(wall_map)
    for x, y in game_state["coins"]:
        coin_map[x][y] = 1
    channels.append(coin_map)

    # create bomb channel
    bomb_map = np.zeros_like(wall_map)
    for x, y in zip(*np.where(game_state["explosion_map"] == 1)):
        bomb_map[x][y] = 1
        # bomb basically is a wall -> player can't move past it
        wall_map[x][y] = 1

    for (x, y), countdown in game_state["bombs"]:
        bomb_map[x][y] = 1
        for i in range(1, 4):
            if x+i < len(bomb_map[x]) - 1:
                bomb_map[x+i][y] = 1
            if x-i > 0:
                bomb_map[x-i][y] = 1
            if y+i < len(bomb_map[:,y]) - 1:
                bomb_map[x][y+i] = 1
            if y-i > 0:
                bomb_map[x][y-i] = 1

    channels.append(bomb_map)

    # create player channel
    player_map = np.zeros_like(wall_map)
    _, _, _, (x, y) = game_state["self"]
    player_map[x][y] = 1
    for _, _, _, (x, y) in game_state["others"]:
        player_map[x][y] = -1
    channels.append(player_map)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels).astype(float)
    return stacked_channels
