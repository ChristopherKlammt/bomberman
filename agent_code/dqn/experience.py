from collections import deque
import numpy as np

from .callbacks import state_to_features
from .parameters import (
    TRANSITION_HISTORY_SIZE, 
    MINIBATCH_SIZE
)

class Experience:
    def __init__(self):
        self.size = TRANSITION_HISTORY_SIZE
        self.filled = False

        self.states = deque([], self.size)
        self.features_states = deque([], self.size)
        self.actions = deque([], self.size)
        self.features_next_states = deque([], self.size)
        self.rewards = deque([], self.size)

    def remember(self, state, action, next_state, reward):
        self.states.append(state)
        self.features_states.append(state_to_features(state))
        self.actions.append(action)
        self.features_next_states.append(state_to_features(next_state))
        self.rewards.append(reward)

        if self.size == len(self.states):
            self.filled = True

    def get_sample(self, batch_size=MINIBATCH_SIZE):
        if self.filled:
            indices = np.random.choice(self.size, batch_size)
        else:
            indices = np.random.choice(len(self.states), batch_size)

        return np.array(self.states)[indices], np.array(self.features_states)[indices], np.array(self.actions)[indices], np.array(self.features_next_states)[indices], np.array(self.rewards)[indices]
