ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_TO_NUMBER = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}

INPUT_SHAPE = (20,)

TRANSITION_HISTORY_SIZE = 300
GAMMA = .99
MINIBATCH_SIZE = 64
LEARNING_RATE = 0.0005

RULE_BASED_PROB_MAX = 0.20
RULE_BASED_PROB_MIN = 0.20
RULE_BASED_PROB_STEP = (RULE_BASED_PROB_MAX -RULE_BASED_PROB_MIN) / TRANSITION_HISTORY_SIZE

EPS = 0.3 # Hyperparameters for Max-Boltzmann
TEMPERATURE = 10

# Events
SURVIVED_ROUND = "SURVIVED_ROUND"
ALREADY_VISITED = "ALREADY_VISITED"
NEW_LOCATION_VISITED = "NEW_LOCATION_VISITED"
SURVIVED_BOMB = "SURVIVED_BOMB"

TRAINING_STEP = 0.2

# File to evaluate data
FILENAME = 'evaluation/new_state.h5'
ROW_SIZE = 5
EVALUATION = True