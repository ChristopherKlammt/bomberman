ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_TO_NUMBER = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}

INPUT_SHAPE = (5, 17, 17)

TRANSITION_HISTORY_SIZE = 1000
GAMMA = .99
MINIBATCH_SIZE = 64
LEARNING_RATE = 0.0005

RULE_BASED_PROB_MAX = 1.20
RULE_BASED_PROB_MIN = 0.20
RULE_BASED_PROB_STEP = (RULE_BASED_PROB_MAX -RULE_BASED_PROB_MIN) / TRANSITION_HISTORY_SIZE

# Events
SURVIVED_ROUND = "SURVIVED_ROUND"
ALREADY_VISITED = "ALREADY_VISITED"
NEW_LOCATION_VISITED = "NEW_LOCATION_VISITED"
SURVIVED_BOMB = "SURVIVED_BOMB"
