import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
import numpy as np

from tqdm.keras import TqdmCallback
import keras
import tensorflow as tf

import tables

from .experience import Experience
from .callbacks import get_next_action, get_valid_probabilities_list
from .model import create_model
from .state import state_to_features

from settings import (
    MAX_STEPS,
    SQRT_OF_COINS,
    MAX_AGENTS,
    REWARD_COIN,
    REWARD_KILL
    )

from .parameters import (
    ACTIONS_TO_NUMBER,
    TRANSITION_HISTORY_SIZE,
    GAMMA,
    MINIBATCH_SIZE, 
    LEARNING_RATE,
    SURVIVED_ROUND,
    ALREADY_VISITED,
    NEW_LOCATION_VISITED,
    SURVIVED_BOMB,
    FILENAME,
    EVALUATION
)

def train(self):
    if not self.experience.filled:
        return

    states, features_states, actions, features_next_states, rewards = self.experience.get_sample()

    qvals_next_states = self.model.predict(features_next_states)
    qvals_next_states_target = self.target_model.predict(features_next_states)
    target_f = get_valid_probabilities_list(self, states, features_states)

    index = np.arange(MINIBATCH_SIZE)
    target_f[index, actions] = rewards + GAMMA * qvals_next_states_target[index, np.argmax(qvals_next_states, axis=1)]

    self.model.fit(features_states, target_f, verbose=0)

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    self.eval_eor_file = tables.open_file('../../'+FILENAME, mode='a')
    self.experience = Experience()

    self.visited_coords = []

    self.reward_sum = 0
    self.rewards = []
    self.trainingStrength = 0
    self.collected_coins = 0 # number of coins collected during one round
    self.killed_opponents = 0 # number of killed opponents
    self.self_kill = 0 # ==1 if he killed himself
    self.number_of_crates_destroyed = 0
    self.points = 0
    self.points_all = {}
    self.count_invalid_actions = 0
    self.survived = 100

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
    add_custom_events(self, new_game_state, events)

    if old_game_state is not None:
        self.reward_sum += reward_from_events(self, events)
        self.experience.remember(old_game_state, ACTIONS_TO_NUMBER[self_action], new_game_state, reward_from_events(self, events))

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
    add_custom_events(self, last_game_state, events)
    self.reward_sum += reward_from_events(self, events)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # evaluate current training using total rewards and number of steps
    if EVALUATION:
        self.last_game_state = last_game_state
        evaluate_training_eor(self)

    self.rewards.append(self.reward_sum)
    print(f"Number of steps: {last_game_state['step']}")
    self.reward_sum = 0
    self.visited_coords = []
    self.collected_coins = 0
    self.killed_opponents = 0
    self.self_kill = 0 
    self.number_of_crates_destroyed = 0
    self.points = 0
    self.points_all = {}
    self.count_invalid_actions = 0
    self.survived = 100

    # update target model
    self.target_model.set_weights(self.model.get_weights())

    # Store the model
    self.model.save("my_agent.model")

def add_custom_events(self, new_game_state, events):
    events.append(SURVIVED_ROUND)

    _, _, _, coords = new_game_state["self"]
    if coords in self.visited_coords:
        events.append(ALREADY_VISITED)
    else:
        events.append(NEW_LOCATION_VISITED)
        self.visited_coords.append(coords)

    if SURVIVED_ROUND in events and e.BOMB_EXPLODED in events:
        events.append(SURVIVED_BOMB)

    # for evaluation purposes
    if e.COIN_COLLECTED in events:
        self.collected_coins += 1
        self.points += REWARD_COIN
    if e.KILLED_OPPONENT in events:
        self.killed_opponents += 1
        self.points += REWARD_KILL
    if e.KILLED_SELF in events:
        self.self_kill = 1
    for event in events:
        if event == e.CRATE_DESTROYED:
            self.number_of_crates_destroyed +=1
    for other in new_game_state['others']:
        self.points_all[other[0]] = other[1]
    if e.INVALID_ACTION in events:
        self.count_invalid_actions +=1
    if e.GOT_KILLED in events:
        self.survived = 0

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.05,
        e.INVALID_ACTION: -0.05,
        e.BOMB_DROPPED: 0.3, 
        # e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0.1,
        e.COIN_FOUND: 0.1,
        e.COIN_COLLECTED: 0.5,
        e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -1,
        e.GOT_KILLED: -1,
        e.OPPONENT_ELIMINATED: 0.1,
        # SURVIVED_ROUND: 0.01,
        SURVIVED_BOMB: 0.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def evaluate_training_eor(self):
    values = []
    values.append(self.last_game_state['step']/MAX_STEPS*100)           # Number of survived steps
    values.append(self.reward_sum/self.last_game_state['step']*200+50)                   # Reward sum
    values.append(self.collected_coins/SQRT_OF_COINS/SQRT_OF_COINS*100) # Number of collected coins
    if MAX_AGENTS > 1:
        values.append(self.killed_opponents/(MAX_AGENTS-1)*100)         # Number of killed enemies
    else:
        values.append(0) 
    values.append(self.self_kill*100)                                   # Did he killed himself?
    values.append(self.number_of_crates_destroyed)                      # Number of crates destroyed
    values.append(self.points*10)                                       # Number of points
    
    points = 0
    if len(self.points_all) != 0:
       for point in self.points_all:
           points += self.points_all[point]
       points = points / len(self.points_all)
    values.append(points*10)                                            # average points of enemies
    values.append(self.count_invalid_actions/self.last_game_state['step']*100)             # number of invalid actions
    values.append(self.survived)                                        # survived

    print(values)
    self.eval_eor_file.root.data.append(np.reshape(np.array(values), (1,len(values))))
