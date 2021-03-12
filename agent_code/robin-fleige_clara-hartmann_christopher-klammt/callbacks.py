import os
import pickle
import random
from collections import deque
from random import shuffle

import numpy as np

from tensorflow import keras

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

TRANSITION_HISTORY_SIZE = 100
RULE_BASED_PROB_MAX = 1.20
RULE_BASED_PROB_MIN = 0.20
RULE_BASED_PROB_STEP = (RULE_BASED_PROB_MAX -RULE_BASED_PROB_MIN) / TRANSITION_HISTORY_SIZE

# returns probabilites for an array of game_states
def get_valid_probabilities_list(self, states, features):
    probabilities = self.model.predict(np.array(features))
    for i in range(len(probabilities)):
        if min(probabilities[i]) < 0:
            probabilities[i] += abs(min(probabilities[i]))
        probabilities[i] *= get_valid_actions(states[i]) # only allow valid actions
        probabilities[i] /= probabilities[i].sum() # normalize to statistical vector (= sums up to 1)
    return probabilities


def get_valid_probabilities(self, game_state):
    probabilities = get_valid_probabilities_list(self, [game_state], [state_to_features(game_state)])[0]
    return probabilities

def get_next_action(self, game_state):
    probabilities = get_valid_probabilities(self, game_state)
    choice = np.random.choice(ACTIONS, p=probabilities)
    # choice = ACTIONS[np.argmax(probabilities)]
    
    return probabilities, choice

def get_valid_actions(game_state):
    _, _, bomb, (x, y) = game_state["self"]
    walls = game_state["field"]
    bombs = list(map(lambda x: x[0], game_state["bombs"]))

    actions = np.ones(6)

    if walls[x][y-1] != 0 or (x, y-1) in bombs:
        actions[0] = 0 # can't go up
    if walls[x+1][y] != 0 or (x+1, y) in bombs:
        actions[1] = 0 # can't go right
    if walls[x][y+1] != 0 or (x, y+1) in bombs:
        actions[2] = 0 # can't go down
    if walls[x-1][y] != 0 or (x-1, y) in bombs:
        actions[3] = 0 # can't go left

    if True:
    # if not bomb:
        actions[5] = 0 # can't plant bomb
    
    return actions

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
    elif self.train:
        self.logger.info("Models will be generated in train.py")
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
    #Rule Based Agent
    if self.train and random.random() < RULE_BASED_PROB_MAX-RULE_BASED_PROB_STEP*self.trainingStrength:
        choice = rb_act(self, game_state)
    
    #Random Agent
    # todo Exploration vs exploitation
    #random_prob = .1
    #if self.train and random.random() < random_prob:
    #    self.logger.debug("Choosing action purely at random.")
    #    choice = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    #    self.logger.debug(f"Chose action: {choice}")
    
    if choice is None:
        self.logger.debug("Querying model for action.")
        probabilities, choice = get_next_action(self, game_state)
        #choice = ACTIONS[np.argmax(self.model.predict(np.array([state_to_features(game_state)]))[0])]
        self.logger.debug(probabilities)
        self.logger.debug(f"Chose action: {choice}")
    
    self.trainingStrength +=1
    return choice


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []

    # chanels consist of: 
    #       - walls
    #       - crates
    #       - coins
    #       - players
    #       - bombs

    # create walls channel
    wall_map = np.zeros_like(game_state["field"])
    for x, y in zip(*np.where(game_state["field"] == -1)):
        wall_map[x][y] = 1
    channels.append(wall_map)

    # create crates channel
    crate_map = np.zeros_like(wall_map)
    for x, y in zip(*np.where(game_state["field"] == 1)):
        crate_map[x][y] = 1
    channels.append(crate_map)    

    # create coins channel
    coin_map = np.zeros_like(wall_map)
    for x, y in game_state["coins"]:
        coin_map[x][y] = 1
    channels.append(coin_map)

    # create bomb channel
    bomb_map = np.zeros_like(wall_map)
    for x, y in zip(*np.where(game_state["explosion_map"] == 1)):
        bomb_map[x][y] = 1
        # bomb basically is a wall -> player can't move past it
        wall_map[x][y] = 1

    for (x, y), countdown in game_state["bombs"]:
        bomb_map[x][y] = 1
        for i in range(1, 4):
            if x+i < len(bomb_map[x]) - 1:
                bomb_map[x+i][y] = 1
            if x-i > 0:
                bomb_map[x-i][y] = 1
            if y+i < len(bomb_map[:,y]) - 1:
                bomb_map[x][y+i] = 1
            if y-i > 0:
                bomb_map[x][y-i] = 1

    channels.append(bomb_map)

    # create player channel
    player_map = np.zeros_like(wall_map)
    _, _, _, (x, y) = game_state["self"]
    player_map[x][y] = 1
    for _, _, _, (x, y) in game_state["others"]:
        player_map[x][y] = -1
    channels.append(player_map)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels).astype(float)
    return stacked_channels

#Rule Based Agent Logic

def rb_setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0

def rb_look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
        
def rb_reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    
def rb_act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        rb_reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = rb_look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a
