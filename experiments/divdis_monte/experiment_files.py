# monte experiment files

monte_positive_files = [
    ["resources/monte_images/climb_down_ladder_room1_screen_termination_positive.npy"],
    [],
    [],
    [],
]

monte_negative_files = [
    ["resources/monte_images/climb_down_ladder_room1_screen_termination_negative.npy"],
    [],
    [],
    [],
]

monte_unlabelled_files = [
    ["resources/monte_images/climb_down_ladder_room0_screen_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room0_screen_termination_positive.npy",
     "resources/monte_images/climb_down_ladder_room2_screen_termination_negative.npy",
     "resources/monte_images/climb_down_ladder_room2_screen_termination_positive.npy"],
    [],
    [],
    [],
]

monte_test_positive_files = [
    [],
    [],
    [],
    [],
]

monte_test_negative_files = [
    [],
    [],
    [],
    [],
]

def epsilon_ball_termination(current_position, term_position, env):
    epsilon = 2
    if current_position[0] <= (term_position[0] + epsilon) and \
        current_position[0] >= (term_position[0] - epsilon) and \
        current_position[1] <= (term_position[1] + epsilon) and \
        current_position[1] >= (term_position[1] - epsilon):
        return True
    return False 


def in_room_termination(current_position, term_position, env):
    return current_position[2] == term_position[2]
    


