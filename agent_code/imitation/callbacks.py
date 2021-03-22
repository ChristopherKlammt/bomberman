import os
import pickle
import random
from collections import deque

import numpy as np

from tensorflow import keras

from .parameters import (
    ACTIONS, 
    TRANSITION_HISTORY_SIZE,
    EPSILON
)

from .model import create_model
from .rule_based_agent import rb_act, rb_setup
from .state import state_to_features

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

def get_next_action(self, game_state):
    probabilities = self.model.predict(np.array([state_to_features(game_state)]))[0]
    choice = np.random.choice(ACTIONS, p=probabilities)
    # choice = ACTIONS[np.argmax(probabilities)]
    
    return probabilities, choice

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
    # always use Rule Based Agent for training
    if self.train:
        choice = rb_act(self, game_state)
        probabilities, _ = get_next_action(self, game_state)
        self.logger.debug(probabilities)

    if choice is None:
        self.logger.debug("Querying model for action.")
        probabilities, choice = get_next_action(self, game_state)
        self.logger.debug(probabilities)
        self.logger.debug(f"Chose action: {choice}")
        rule_based_choice = rb_act(self, game_state)
        self.logger.debug(f"Rule-based agent chose: {rule_based_choice}")
    
    self.trainingStrength +=1
    return choice
