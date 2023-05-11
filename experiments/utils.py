from portable.utils.ale_utils import get_object_position, get_skull_position

def get_snake_x(position):
    if position[2] == 9:
        if position[1] <= 42:
            return 42
        else:
            return 11
    if position[2] == 11:
        if position[1] <= 97:
            return 97
        else:
            return 35
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

def get_percent_completed_enemy(start_pos, final_pos, terminations, env):
    def manhatten(a,b):
        return sum(abs(val1-val2) for val1, val2 in zip((a[0], a[1]),(b[0],b[1])))

    if start_pos[2] != final_pos[2]:
        return 0

    info = env.get_current_info({})
    if info["dead"]:
        return 0

    room = start_pos[2]
    ground_y = start_pos[1]
    ram = env.unwrapped.ale.getRAM()
    true_distance = 0
    completed_distance = 0
    if final_pos[2] != room:
        return 0
    if final_pos[2] == 4 and final_pos[0] == 5:
        return 0
    if room in [2,3]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-25, ground_y)
        if final_pos[0] < skull_x-25 and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    if room in [0,1,18]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-6, ground_y)
        if final_pos[0] < skull_x-6 and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        end_pos = (spider_x - 6, ground_y)
        if final_pos[0] < spider_x and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    elif room in [9,11,22]:
        # snakes
        end_pos = terminations
        if in_epsilon_square(final_pos, end_pos):
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    else:
        return 0

    return completed_distance/(true_distance+1e-5)

def check_termination_correct_enemy(final_pos, terminations, env):
    if terminations[2] != final_pos[2]:
        return False
    
    info = env.get_current_info({})
    if info["dead"]:
        return False

    room = terminations[2]
    ground_y = terminations[1]
    ram = env.unwrapped.ale.getRAM()
    if final_pos[2] != room:
        return False
    if final_pos[2] == 4 and final_pos[0] == 5:
        return False
    if room in [2,3]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-25, ground_y)
        if final_pos[0] < skull_x-25 and final_pos[1] <= ground_y:
            return True
        else:
            return False
    if room in [0,1,18]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-6, ground_y)
        if final_pos[0] < skull_x-6 and final_pos[1] <= ground_y:
            return True
        else:
            return False
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        end_pos = (spider_x - 6, ground_y)
        if final_pos[0] < spider_x and final_pos[1] <= ground_y and final_pos[2] == room:
            return True
        else:
            return False
    elif room in [9,11,22]:
        # snakes
        test_pos = (final_pos[0]-3, final_pos[1], final_pos[2])
        snake_x = get_snake_x(test_pos)
        end_pos = (snake_x, ground_y)
        if final_pos[0] < snake_x and final_pos[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False

