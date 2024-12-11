# monte experiment files
from experiments.divdis_monte.core.monte_terminations import check_termination_correct_enemy_left, check_termination_correct_enemy_right

monte_positive_files = [
    ["resources/monte_images/screen_climb_down_ladder_termination_positive.npy"],
    ["resources/monte_images/climb_up_ladder_room1_termination_positive.npy"],
    ["resources/monte_images/move_left_enemy_room1_termination_positive.npy"],
    ["resources/monte_images/move_right_enemy_room1_termination_positive.npy"],
]

monte_negative_files = [
    ["resources/monte_images/screen_climb_down_ladder_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/climb_up_ladder_room1_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/move_left_enemy_room1_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
    ["resources/monte_images/move_right_enemy_room1_termination_negative.npy",
     "resources/monte_images/screen_death_1.npy",
     "resources/monte_images/screen_death_2.npy",
     "resources/monte_images/screen_death_3.npy",
     "resources/monte_images/screen_death_4.npy"],
]

monte_unlabelled_files = [
    ["resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_positive.npy",],
    ["resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_positive.npy",],
    ["resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_positive.npy",],
    ["resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/climb_down_ladder_room0_initiation_positive.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_negative.npy",
     "resources/monte_images/move_left_enemy_room11left_termination_positive.npy",],
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

def enemy_right_termination(current_position, term_position, env):
    return check_termination_correct_enemy_right(current_position, env)

def enemy_left_termination(current_position, term_position, env):
    return check_termination_correct_enemy_left(current_position, env)


