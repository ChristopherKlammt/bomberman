from collections import deque
import numpy as np

from .callbacks import state_to_features
from .parameters import BATCH_SIZE, TRANSITION_HISTORY_SIZE

class Experience:
    def __init__(self):
        self.size = TRANSITION_HISTORY_SIZE
        self.features_states = deque([], self.size)
        self.targets = deque([], self.size)
        self.filled = False

    def remember(self, state, target,):
        self.features_states.append(state_to_features(state))
        self.targets.append(target)

        if len(self.targets) == self.size:
            self.filled = True

    def get(self, batch_size=BATCH_SIZE):
        if self.filled:
            indices = np.random.choice(self.size, batch_size)
        else:
            indices = np.random.choice(len(self.states), batch_size)
            
        return np.array(self.features_states)[indices], np.array(self.targets)[indices]

    def clear(self):
        self.features_states.clear()
        self.targets.clear()

