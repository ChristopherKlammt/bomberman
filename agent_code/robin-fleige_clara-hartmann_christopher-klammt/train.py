import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

from tqdm.keras import TqdmCallback
import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = .99
MINIBATCH_SIZE = 32
LEARNING_RATE=0.05

actions_to_number = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "BOMB": 4, "WAIT": 5}

# Events
SURVIVED_ROUND = "SURVIVED_ROUND"

def create_model():
    # parameters
    num_actions = 6
    num_channels = 3
    view_size = 9
    input_dim = num_channels * view_size ** 2

    # create keras model
    model = Sequential()
    
    model.add(Dense(64, input_dim=input_dim, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_actions, activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Adam())

    return model

def train(self):
    minibatch = np.random.choice(self.transitions, MINIBATCH_SIZE, replace=True)

    # states = np.array(list(map(lambda x: x.state, minibatch)))
    # actions = np.array(list(map(lambda x: x.action, minibatch)))
    # next_states = np.array(list(map(lambda x: x.next_state, minibatch)))
    # rewards = np.array(list(map(lambda x: x.reward, minibatch)))
    states = np.array(list(map(lambda x: x["s"], minibatch)))
    actions = np.array(list(map(lambda x: x["a"], minibatch)))
    next_states = np.array(list(map(lambda x: x["sprime"], minibatch)))
    rewards = np.array(list(map(lambda x: x["r"], minibatch)))

    qvals_next_states = self.model.predict(next_states)
    target_f = self.model.predict(states)

    # q-update target
    for i, (state, action, reward, qvals_next_state) in enumerate(zip(states, actions, rewards, qvals_next_states)):
        if qvals_next_state is None:
            target = reward
        else:
            target = reward + GAMMA * np.max(qvals_next_state)
        target_f[i][actions_to_number[action]] = target

    self.model.fit(states, target_f, epochs=1, verbose=0)
    return self.model

def save_model(model, fn):
    model.save(fn)

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.transitions = []

    self.reward_sum = 0

    if not hasattr(self, "model"):
        self.logger.debug("create new models for training")
        self.model = create_model()
        # self.target_model = create_model()
    else:
        self.logger.debug("load model as target_model as well")
        # self.target_model = keras.models.load_model("my_agent.model")

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

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(SURVIVED_ROUND)

    if len(self.transitions) > TRANSITION_HISTORY_SIZE:
            self.transitions.pop(0)

    # state_to_features is defined in callbacks.py
    if old_game_state is not None:
        self.reward_sum += reward_from_events(self, events)
        self.transitions.append({"s": state_to_features(old_game_state), "a": self_action, "sprime": state_to_features(new_game_state), "r": reward_from_events(self, events)})
        # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

        train(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.reward_sum += reward_from_events(self, events)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append({"s": state_to_features(last_game_state), "a": last_action, "sprime": None, "r": reward_from_events(self, events)})
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, state_to_features(None), reward_from_events(self, events)))

    print("Total reward:", self.reward_sum)
    self.reward_sum = 0
    self.transitions = []

    # Store the model
    save_model(self.model, "my_agent.model")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: .5,
        e.MOVED_RIGHT: .5,
        e.MOVED_UP: .5,
        e.MOVED_DOWN: .5,
        e.COIN_COLLECTED: 10,
        # e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -2,
        e.WAITED: -2,
        # SURVIVED_ROUND: .1,
        e.KILLED_SELF: -100,
        # e.BOMB_DROPPED: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
