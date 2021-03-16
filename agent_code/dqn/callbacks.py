import os
import pickle
import random
from collections import deque

import numpy as np

from tensorflow import keras

from .parameters import (
    ACTIONS, 
    TRANSITION_HISTORY_SIZE,
    RULE_BASED_PROB_MAX,
    RULE_BASED_PROB_MIN, 
    RULE_BASED_PROB_STEP
)

from .model import create_model
from .rule_based_agent import rb_act, rb_setup
from .state import state_to_features

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

# returns probabilites for an array of game_states
def get_valid_probabilities_list(self, states, features):
    probabilities = self.model.predict(np.array(features))
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

    # if True:
    if not bomb:
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
    
    rb_setup(self)

    if os.path.exists("my_agent.model"):
        self.logger.info("Loading model from saved state.")
        self.model = keras.models.load_model("my_agent.model")
        if self.train:
            self.target_model = create_model()
            self.target_model.set_weights(self.model.get_weights())
    elif self.train:
        self.logger.info("Setting up model from scratch...")
        self.model = create_model()
        self.target_model = create_model()
        self.target_model.set_weights(self.model.get_weights())
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
    self.trainingStrength = game_state['round']
    choice = None
    #Rule Based Agent
    if self.train and random.random() < RULE_BASED_PROB_MAX-RULE_BASED_PROB_STEP*self.trainingStrength:
        choice = rb_act(self, game_state)

    if choice is None:
        self.logger.debug("Querying model for action.")
        probabilities, choice = get_next_action(self, game_state)
        self.logger.debug(probabilities)
        self.logger.debug(f"Chose action: {choice}")
    
    self.trainingStrength +=1
    return choice
