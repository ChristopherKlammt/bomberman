import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features, get_next_action, get_valid_probabilities_list

import numpy as np

from tqdm.keras import TqdmCallback
import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.optimizers import Adam

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = .99
MINIBATCH_SIZE = 32
LEARNING_RATE = 0.05

actions_to_number = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "BOMB": 4, "WAIT": 5}

# Events
SURVIVED_ROUND = "SURVIVED_ROUND"
ALREADY_VISITED = "ALREADY_VISITED"
NEW_LOCATION_VISITED = "NEW_LOCATION_VISITED"
SURVIVED_BOMB = "SURVIVED_BOMB"

def create_model():
    # parameters
    num_actions = 6

    # create keras model
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_actions))
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=LEARNING_RATE))

    return model

def train(self):
    batch_size = MINIBATCH_SIZE
    if MINIBATCH_SIZE > len(self.transitions):
        batch_size = len(self.transitions)
    minibatch = np.random.choice(self.transitions, batch_size, replace=True)

    states = np.array(list(map(lambda x: x["state"], minibatch)))
    feature_states = np.array(list(map(lambda x: state_to_features(x["state"]), minibatch)))
    actions = np.array(list(map(lambda x: x["action"], minibatch)))
    next_states = np.array(list(map(lambda x: state_to_features(x["next_state"]), minibatch)))
    rewards = np.array(list(map(lambda x: x["reward"], minibatch)))
    game_overs = np.array(list(map(lambda x: x["game_over"], minibatch)))

    qvals_next_states = self.model.predict(next_states)
    target_f = get_valid_probabilities_list(self, states, feature_states)

    # q-update target
    for i, (state, action, reward, qvals_next_state, game_over) in enumerate(zip(feature_states, actions, rewards, qvals_next_states, game_overs)):
        # add total reward to each step ??
        # reward = reward + self.reward_sum
        if game_over:
            target = reward
        else:
            target = reward + GAMMA * np.max(qvals_next_state)
        target_f[i][actions_to_number[action]] = target

    self.model.fit(feature_states, target_f, epochs=1, verbose=0)

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
    self.transitions = []

    self.visited_coords = []

    self.reward_sum = 0
    self.rewards = []

    if not hasattr(self, "model"):
        self.logger.debug("create new models for training")
        self.model = create_model()

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

    _, _, _, coords = new_game_state["self"]
    if coords in self.visited_coords:
        events.append(ALREADY_VISITED)
    else:
        events.append(NEW_LOCATION_VISITED)
        self.visited_coords.append(coords)

    if SURVIVED_ROUND in events and e.BOMB_EXPLODED in events:
        events.append(SURVIVED_BOMB)

    if len(self.transitions) > TRANSITION_HISTORY_SIZE:
            self.transitions.pop()

    # state_to_features is defined in callbacks.py
    if old_game_state is not None:
        self.reward_sum += reward_from_events(self, events)
        self.transitions.append({"state": old_game_state, "action": self_action, "next_state": new_game_state, "reward": reward_from_events(self, events), "game_over": False})


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
    self.transitions.append({"state": last_game_state, "action": last_action, "next_state": last_game_state, "reward": reward_from_events(self, events), "game_over": True})

    for i in range(10): # train after each game with 10 different minibatches
        train(self)

    self.rewards.append(self.reward_sum)
    print(f"Number of steps: {len(self.transitions)}")
    # print(f"Positive reward quota: {np.count_nonzero(np.array(self.rewards) > 0) / len(self.rewards) * 100 : .1f} % ({self.reward_sum : .1f} in {len(self.transitions)} rounds)")
    self.reward_sum = 0
    self.transitions = []
    self.visited_coords = []

    # Store the model
    save_model(self.model, "my_agent.model")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # e.MOVED_LEFT: .1,
        # e.MOVED_RIGHT: .1,
        # e.MOVED_UP: .1,
        # e.MOVED_DOWN: .1,
        e.COIN_COLLECTED: 10,
        # e.KILLED_OPPONENT: 5,
        e.WAITED: -2,
        # e.KILLED_SELF: -10,
        # e.BOMB_EXPLODED: 10,
        # e.BOMB_DROPPED: -5,
        SURVIVED_ROUND: -.5,
        # e.CRATE_DESTROYED: 3,
        # e.COIN_FOUND: 5,
        NEW_LOCATION_VISITED: 1,
        ALREADY_VISITED: -.5,
        # SURVIVED_BOMB: 10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
