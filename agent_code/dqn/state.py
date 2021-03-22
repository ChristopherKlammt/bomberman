import numpy as np

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
