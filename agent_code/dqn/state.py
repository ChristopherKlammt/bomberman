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
    #       - movable direction
    #       - danger
    #       - coins
    #       - players
    #       - bomb strength
    
    # entries in chanels are for self, top, right, down, left
    
    # prepare data
    player_position = np.array([game_state["self"][3][0],game_state["self"][3][1]])
    
    adjusted_map = np.array(game_state["field"])
    adjusted_map[player_position[0]][player_position[1]] = -2
    # Add coins to adjusted map
    adjusted_map[tuple(np.array(game_state["coins"]).T)] = 2
    # Add others
    for i in range(len(game_state["others"])):
        adjusted_map[game_state["others"][i][3]] = 4
    # Add bombs
    for i in range(len(game_state["bombs"])):
        adjusted_map[game_state["bombs"][i][0]] = 3
    #print("Adjusted Map")
    #print(adjusted_map)
    # Player = -2
    # Wall = -1
    # Crate = 1
    # Free = 0
    # Coins = 2
    # Bombs = 3
    # Others = 4
    
    # print(adjusted_map)

    
    
    neighbouring_fields = []
    neighbouring_fields.append((player_position[0]  ,    player_position[1]     )) # self
    neighbouring_fields.append((player_position[0]-1,    player_position[1]     )) # left
    neighbouring_fields.append((player_position[0],      player_position[1]+1   )) # down
    neighbouring_fields.append((player_position[0]+1,    player_position[1]     )) # right
    neighbouring_fields.append((player_position[0],      player_position[1]-1   )) # up
    #print("Neighbouring Fields")
    #print(neighbouring_fields)
    
    # is it possible to move in that direction?
    movable = np.zeros(5)
    for i, position in enumerate(neighbouring_fields):
        field = adjusted_map[int(position[0])][int(position[1])]
        movable[i] = field == 0 or field == 2 or field == -2
    #print("Moveable")
    #print(movable)
    
    # is the field dangered by a bomb? if yes, how many turns until explosion?
    dangered = np.zeros(5)
    for i, position in enumerate(neighbouring_fields):
        dangered[i] = 4
        for bomb in game_state["bombs"]:
            if position in get_dangered_fields_by_bomb(bomb[0],adjusted_map):
                if dangered[i] > bomb[1]:
                        dangered[i] = bomb[1]
    #print("Dangered")
    #print(dangered)
    
    # distance to the next coin
    coin_distance = np.zeros(5)
    for i, position in enumerate(neighbouring_fields):
        coin_distance[i] = get_closest(position,adjusted_map, 2)
    #print("Coin Distance")
    #print(coin_distance)
                    
    # distance to the next enemy
    enemy_distance = np.zeros(5)
    for i, position in enumerate(neighbouring_fields):
        enemy_distance[i] = get_closest(position,adjusted_map, 4)
    #print("Enemy Distance")
    #print(enemy_distance)

    # number of crates in bomb range
    bomb_strength = np.zeros(5)
    for i, position in enumerate(neighbouring_fields):
        dangered_fields = get_dangered_fields_by_bomb(game_state["self"][3],adjusted_map)
        count = 0
        for field in dangered_fields:
            if(adjusted_map[field[0]][field[1]]) == 1:
                count = count + 1
            if(adjusted_map[field[0]][field[1]]) == 4:
                count = count + 3
        bomb_strength[i] = count
    #print("Bomb Strength")
    #print(bomb_strength)    
        

    # concatenate them as a feature tensor (they must have the same shape), ...
    channels.append(movable)
    channels.append(dangered)
    channels.append(coin_distance)
    channels.append(enemy_distance)
    channels.append(bomb_strength)
    stacked_channels = np.stack(channels).astype(float)
    return stacked_channels.reshape(-1)

def get_closest(position, adjusted_map, goal):
    MAX = 100
    distance_when_to_abort = 20
    to_move = [(position[0],position[1],0)]
    while len(to_move) > 0:
        position = to_move.pop(0)
        # if position[2] > distance_when_to_abort:
        #     return MAX
        if position[0] > 0 and position[1] > 0 and position[0] < adjusted_map.shape[0] and position[1] < adjusted_map.shape[1]:
             if adjusted_map[position[0]][position[1]] == goal:
                 return position[2]+1
             else:
                 if adjusted_map[position[0]][position[1]] == 0 or adjusted_map[position[0]][position[1]] == 2:
                     adjusted_map[position[0]][position[1]] = -1
                     to_move.append((position[0]+1,position[1]  ,position[2]+1))
                     to_move.append((position[0]-1,position[1]  ,position[2]+1))
                     to_move.append((position[0]  ,position[1]+1,position[2]+1))
                     to_move.append((position[0]  ,position[1]-1,position[2]+1))
    return adjusted_map.shape[0]+adjusted_map.shape[1]

def get_dangered_fields_by_bomb(bomb,adjusted_map):
    dangered_fields = []
    shape = adjusted_map.shape
    for i in range(1,4):
        position = (bomb[0]+i,bomb[1]  )
        if position[0] > 0 and position[1] > 0 and position[0] < shape[0] and position[1] <shape[1]:
            if adjusted_map[position[0]][position[1]] == -1:
                i = 4
            else:
                dangered_fields.append(position)
    for i in range(1,4):
        position = (bomb[0]-i,bomb[1]  )
        if position[0] > 0 and position[1] > 0 and position[0] < shape[0] and position[1] <shape[1]:
            if adjusted_map[position[0]][position[1]] == -1:
                i = 4
            else:
                dangered_fields.append(position)
            dangered_fields.append(position)
    for i in range(1,4):
        position = (bomb[0]  ,bomb[1]+i)
        if position[0] > 0 and position[1] > 0 and position[0] < shape[0] and position[1] <shape[1]:
            if adjusted_map[position[0]][position[1]] == -1:
                i = 4
            else:
                dangered_fields.append(position)
            dangered_fields.append(position)
    for i in range(1,4):
        position = (bomb[0]  ,bomb[1]-i)
        if position[0] > 0 and position[1] > 0 and position[0] < shape[0] and position[1] <shape[1]:
            if adjusted_map[position[0]][position[1]] == -1:
                i = 4
            else:
                dangered_fields.append(position)
            dangered_fields.append(position)
    return dangered_fields
