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
    RULE_BASED_PROB_STEP,
    TEMPERATURE,
    EPS, 
    TRAINING_STEP
)

from .model import create_model
from .rule_based_agent import rb_act, rb_setup
from .state import state_to_features

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

# returns probabilites for an array of game_states
def get_valid_probabilities_list(self, states, features):
    probabilities = self.model.predict(np.array(features))
    return probabilities


def get_valid_probabilities(self, game_state):
    probabilities = get_valid_probabilities_list(self, [game_state], [state_to_features(game_state)])[0]
    return probabilities
    
def max_boltzmann(self, probabilities):
    distribution = []
    for i in range(len(probabilities)):
        # if probabilities[i]>0:
            distribution.append(np.exp((probabilities[i]/TEMPERATURE)))
        # else: 
        #     distribution.append(0)
    distribution /= np.sum(distribution)
    if np.random.random() >= EPS: # return choice of highest probability
        self.logger.debug(f"Chose exploiting action.")
        return probabilities, ACTIONS[np.argmax(probabilities)]
    else: 
        self.logger.debug(f"Chose exploring action.")
        return distribution, np.random.choice(ACTIONS, p=distribution)

def get_next_action(self, game_state):
    # Using Max-Boltzmann exploration
    # probabilities, choice = max_boltzmann(get_valid_probabilities(self, game_state))
    probabilities = get_valid_probabilities(self, game_state)
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
    # self.trainingStrength = game_state['round']
    choice = None
    #Rule Based Agent
    if self.train: # and random.random() < max(RULE_BASED_PROB_MAX-RULE_BASED_PROB_STEP*self.trainingStrength, RULE_BASED_PROB_MIN):
        choice = rb_act(self, game_state)
        probabilities, model_choice = get_next_action(self, game_state)
        self.logger.debug(probabilities)
        self.logger.debug(f"Model choice: {model_choice}")

    if choice is None:
        self.logger.debug("Querying model for action.")
        probabilities, choice = get_next_action(self, game_state)
        self.logger.debug(probabilities)
        self.logger.debug(f"Chose action: {choice}")
        
    if self.train:
        self.trainingStrength += TRAINING_STEP
    return choice
