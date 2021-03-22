import pickle
import random
from collections import namedtuple, deque
from typing import List
from datetime import datetime

import events as e
import numpy as np

from tqdm.keras import TqdmCallback
import keras
import tensorflow as tf

from .experience import Experience
from .callbacks import get_next_action
from .model import create_model
from .state import state_to_features
from .rule_based_agent import rb_act
from .parameters import (
    ACTIONS,
    ACTIONS_TO_NUMBER,
    NUMBER_EPOCHS
)

def train(self):
    if self.experience.filled:
        features_states, targets = self.experience.get()

        self.model.fit(features_states, targets, initial_epoch=self.current_epoch, epochs=self.current_epoch + NUMBER_EPOCHS, verbose=2, callbacks=[self.tensorboard_callback])
        self.current_epoch += NUMBER_EPOCHS

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.experience = Experience()

    log_dir = "tf-logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    self.current_epoch = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None:
        target = [1 if a == rb_act(self, old_game_state) else 0 for a in ACTIONS]
        self.experience.remember(old_game_state, target)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    print(f"Number of steps: {last_game_state['step']}")

    for i in range(10):
        train(self)
    
    # Store the model
    self.model.save("my_agent.model")
