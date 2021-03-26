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
    #       - bomb strnegth
    
    # entries in chanels are for top, right, down, left in that order
    
    # prepare data
    player_position = np.array([game_state["self"][3][0],game_state["self"][3][1]])
    
    adjusted_map = np.array(game_state["field"])
    adjusted_map[player_position[0]][player_position[1]] = -2
    print("Adjusted Map")
    print(adjusted_map)
    # Player = -2
    # Wall = -1
    # Crate = 1
    # Free = 0
    #TODO: Add Coins = 2
    #TODO: Add Bombs = 3
    #TODO: Add Others = 4
    
    
    neighbouring_fields = []
    neighbouring_fields.append((player_position[0]-1,    player_position[1]     ))
    neighbouring_fields.append((player_position[0],      player_position[1]+1   ))
    neighbouring_fields.append((player_position[0]+1,    player_position[1]     ))
    neighbouring_fields.append((player_position[0],      player_position[1]-1   ))
    print("Neighbouring Fields")
    print(neighbouring_fields)
    
    # is it possible to move in that direction?
    movable = np.zeros(4)
    for i, position in enumerate(neighbouring_fields):
        field = adjusted_map[int(position[0])][int(position[1])]
        movable[i] = field == 0 or field == 2
    print("Moveable")
    print(movable)
    
    # is the field dangered by a bomb? if yes, how many turn until explosion?
    dangered = np.zeros(4)
    for i, position in enumerate(neighbouring_fields):
        dangered[i] = 4
        for bomb in game_state["bombs"]:
            if get_dangered_fields_by_bomb(bomb[0],adjusted_map.shape).contains(position):
                if dangered[i] > bomb[1]:
                        dangered = bomb[1]
    print("Dangered")
    print(dangered)
    
    # distance to the next coin
    coin_distance = np.zeros(4)
    for i, position in enumerate(neighbouring_fields):
        coin_distance[i] = get_closest(position,adjusted_map, 2)
    print("Coin Distance")
    print(coin_distance)
                    
    # distance to the next enemy
    enemy_distance = np.zeros(4)
    for i, position in enumerate(neighbouring_fields):
        enemy_distance[i] = get_closest(position,adjusted_map, 4)
    print("Enemy Distance")
    print(enemy_distance)

    # number of crates in bomb range
    bomb_strength = np.zeros(4)
    for i, position in enumerate(neighbouring_fields):
        dangered_fields = get_dangered_fields_by_bomb(game_state["self"][3],adjusted_map.shape)
        count = 0
        for field in dangered_fields:
            if(adjusted_map[field[0]][field[1]]) == 1:
                count = count + 1
        bomb_strength[i] = count
    print("Bomb Strength")
    print(bomb_strength)    
        

    # concatenate them as a feature tensor (they must have the same shape), ...
    channels.append(movable)
    channels.append(dangered)
    channels.append(coin_distance)
    channels.append(enemy_distance)
    channels.append(bomb_strength)
    stacked_channels = np.stack(channels).astype(float)
    return stacked_channels.reshape(-1)

def get_closest(position, adjusted_map, goal):
    to_move = [(position[0],position[1],0)]
    while len(to_move) > 0:
        position = to_move.pop(0)
        if position[0] > 0 and position[1] > 0 and position[0] < adjusted_map.shape[0] and position[1] <adjusted_map.shape[1]:
             if adjusted_map[position[0]][position[1]] == goal:
                 return position[2]+1
             else:
                 if adjusted_map[position[0]][position[1]] == 0 or adjusted_map[position[0]][position[1]] == 2:
                     adjusted_map[position[0]][position[1]] = -1
                     to_move.append((position[0]+1,position[1]  ,position[2]+1))
                     to_move.append((position[0]-1,position[1]  ,position[2]+1))
                     to_move.append((position[0]  ,position[1]+1,position[2]+1))
                     to_move.append((position[0]  ,position[1]-1,position[2]+1))
    return -1

def get_dangered_fields_by_bomb(bomb,shape):
    dangered_fields = []
    for i in range(3):
        position = (bomb[0]+i,bomb[1]  )
        if position[0] > 0 and position[1] > 0 and position[0] < shape[0] and position[1] <shape[1]:
            dangered_fields.append(position)
        position = (bomb[0]-i,bomb[1]  )
        if position[0] > 0 and position[1] > 0 and position[0] < shape[0] and position[1] <shape[1]:
            dangered_fields.append(position)
        position = (bomb[0]  ,bomb[1]+i)
        if position[0] > 0 and position[1] > 0 and position[0] < shape[0] and position[1] <shape[1]:
            dangered_fields.append(position)
        position = (bomb[0]  ,bomb[1]-i)
        if position[0] > 0 and position[1] > 0 and position[0] < shape[0] and position[1] <shape[1]:
            dangered_fields.append(position)
    return dangered_fields
