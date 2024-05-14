from portable.utils.ale_utils import get_object_position, get_skull_position

bottom_of_ladder = [
    (20, 148, 1), # bottom of left ladder room 1
    (76, 192, 1), # bottom of middle ladder room 1
    (133, 148, 1), # bottom right ladder room 1
    (77, 235, 4), # middle ladder room 4
    (77, 235, 6), # bottom ladder room 6
    (77, 235, 9), # bottom of ladder room 9
    (77, 235, 10), # bottom of ladder room 10
    (77, 236, 11), # middle of ladder room 11
    (77, 237, 13), # middle of ladder room 13
    (77, 235, 19), # bottom ladder room 19
    (76, 235, 21), # bottom ladder room 21
    (77, 235, 22), # bottom ladder room 22
]

top_of_ladder = [
    (21, 192, 1), # top of left ladder room 1
    (77, 235, 1), # top of middle ladder room 1
    (129, 192, 1), # top right ladder room 1
    (74, 235, 0), # top ladder room 0
    (79, 235, 2), # top ladder room 2
    (76, 235, 3), # top ladder room 3
    (77, 235, 4), # middle ladder room 4
    (77, 157, 5), # top ladder room 5
    (79, 235, 7), # top ladder room 7
    (77, 235, 11), # middle of ladder room 11
    (77, 235, 13), # middle of ladder room 13
    (80, 160, 14), # top ladder room 14
]

def check_termination_bottom_ladder(state, env):
    info = env.get_current_info({})
    if info["dead"]:
        return False
    
    position = info["player_pos"]
    
    for pos in bottom_of_ladder:
        if in_epsilon_square(pos, position):
            return True
    return False

def check_termination_top_ladder(state, env):
    info = env.get_current_info({})
    if info["dead"]:
        return False
    
    position = info["player_pos"]
    
    for pos in top_of_ladder:
        if in_epsilon_square(pos, position):
            return True
    return False

def get_snake_x_left(position):
    if position[2] == 9:
        return 11
    if position[2] == 11:
        return 35
    if position[2] == 22:
        return 25

def get_snake_x_right(position):
    if position[2] == 9:
        return 60
    if position[2] == 11:
        return 118
    if position[2] == 22:
        return 25

def in_epsilon_square(current_position, final_position):
    epsilon = 2
    if current_position[0] <= (final_position[0] + epsilon) and \
        current_position[0] >= (final_position[0] - epsilon) and \
        current_position[1] <= (final_position[1] + epsilon) and \
        current_position[1] >= (final_position[1] - epsilon):
        return True
    return False  

def check_termination_correct_enemy_left(state, env):
    info = env.get_current_info({})
    if info["dead"]:
        return False
    
    position = info["player_pos"]

    room = position[2]
    ram = env.unwrapped.ale.getRAM()
    if room in [2,3]:
        # dancing skulls
        skull_x = get_skull_position(ram)
        if position[0] < skull_x-25 and position[1] <= 235:
            return True
        else:
            return False
    if room in [1,5,18]:
        # rolling skulls
        skull_x = get_skull_position(ram)
        if room == 1:
            ground_y = 148
        elif room == 5:
            ground_y = 195
        elif room == 18:
            ground_y = 235
        if position[0] < skull_x-6 and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        ground_y = 235
        if position[0] < spider_x and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [9,11,22]:
        # snakes
        snake_x = get_snake_x_left(position)
        ground_y = 235
        if position[0] < snake_x and position[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False

def check_termination_correct_enemy_right(state, env):
    info = env.get_current_info({})
    if info["dead"]:
        return False
    
    position = info["player_pos"]
    room = position[2]
    ram = env.unwrapped.ale.getRAM()
    if room in [2,3]:
        # dancing skulls
        skull_x = get_skull_position(ram)
        if position[0] > skull_x+25 and position[1] <= 235:
            return True
        else:
            return False
    if room in [1,5,18]:
        # rolling skulls
        skull_x = get_skull_position(ram)
        if room == 1:
            ground_y = 148
        elif room == 5:
            ground_y = 195
        elif room == 18:
            ground_y = 235
        if position[0] > skull_x+6 and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        ground_y = 235
        if position[0] > spider_x and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [9,11,22]:
        # snakes
        snake_x = get_snake_x_right(position)
        ground_y = 235
        if position[0] > snake_x and position[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False






