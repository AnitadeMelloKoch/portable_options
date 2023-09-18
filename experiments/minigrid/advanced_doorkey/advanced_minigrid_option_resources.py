from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import environment_builder 

training_seed = 0

initiation_positive_files = [
    # get red key
    [
        'resources/minigrid_images/adv_doorkey_getredkey_doorred_0_initiation_positive.npy'
    ],
    # get blue key
    [
        'resources/minigrid_images/adv_doorkey_getbluekey_doorblue_0_initiation_positive.npy'
    ],
    # get green key
    [
        'resources/minigrid_images/adv_doorkey_getgreenkey_doorgreen_0_initiation_positive.npy'
    ],
    # get purple key
    [
        'resources/minigrid_images/adv_doorkey_getpurplekey_doorpurple_0_initiation_positive.npy'
    ],
    # get yellow key
    [
        'resources/minigrid_images/adv_doorkey_getyellowkey_dooryellow_0_initiation_positive.npy'
    ],
    # get grey key
    [
        'resources/minigrid_images/adv_doorkey_getgreykey_doorgrey_0_initiation_positive.npy'
    ],
    # open red door
    [
        'resources/minigrid_images/adv_doorkey_openreddoor_doorred_0_initiation_positive.npy'
    ],
    # open blue door
    [
        'resources/minigrid_images/adv_doorkey_openbluedoor_doorblue_0_initiation_positive.npy'
    ],
    # open green door
    [
        'resources/minigrid_images/adv_doorkey_opengreendoor_doorgreen_0_initiation_positive.npy'
    ],
    # open purple door
    [
        'resources/minigrid_images/adv_doorkey_openpurpledoor_doorpurple_0_initiation_positive.npy'
    ],
    # open yellow door
    [
        'resources/minigrid_images/adv_doorkey_openyellowdoor_dooryellow_0_initiation_positive.npy'
    ],
    # open grey door
    [
        'resources/minigrid_images/adv_doorkey_opengreydoor_doorgrey_0_initiation_positive.npy'
    ],
]
initiation_negative_files = [
    # get red key
    [
        'resources/minigrid_images/adv_doorkey_getredkey_doorred_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_red.npy',
    ],
    # get blue key
    [
        'resources/minigrid_images/adv_doorkey_getbluekey_doorblue_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_blue.npy',
    ],
    # get green key
    [
        'resources/minigrid_images/adv_doorkey_getgreenkey_doorgreen_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_green.npy',
    ],
    # get purple key
    [
        'resources/minigrid_images/adv_doorkey_getpurplekey_doorpurple_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_purple.npy',
    ],
    # get yellow key
    [
        'resources/minigrid_images/adv_doorkey_getyellowkey_dooryellow_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_yellow.npy',
    ],
    # get grey key
    [
        'resources/minigrid_images/adv_doorkey_getgreykey_doorgrey_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_grey.npy',
    ],
    # open red door
    [
        'resources/minigrid_images/adv_doorkey_openreddoor_doorred_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_red.npy',
    ],
    # open blue door
    [
        'resources/minigrid_images/adv_doorkey_openbluedoor_doorblue_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_blue.npy',
    ],
    # open green door
    [
        'resources/minigrid_images/adv_doorkey_opengreendoor_doorgreen_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_green.npy',
    ],
    # open purple door
    [
        'resources/minigrid_images/adv_doorkey_openpurpledoor_doorpurple_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_purple.npy',
    ],
    # open yellow door
    [
        'resources/minigrid_images/adv_doorkey_openyellowdoor_dooryellow_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_yellow.npy',
    ],
    # open grey door
    [
        'resources/minigrid_images/adv_doorkey_opengreydoor_doorgrey_0_initiation_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_grey.npy',
    ],
]
termination_positive_files = [
    # get red key
    [
        'resources/minigrid_images/adv_doorkey_getredkey_doorred_0_termination_positive.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_red.npy',
    ],
    # get blue key
    [
        'resources/minigrid_images/adv_doorkey_getbluekey_doorblue_0_termination_positive.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_blue.npy',
    ],
    # get green key
    [
        'resources/minigrid_images/adv_doorkey_getgreenkey_doorgreen_0_termination_positive.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_green.npy',
    ],
    # get purple key
    [
        'resources/minigrid_images/adv_doorkey_getpurplekey_doorpurple_0_termination_positive.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_purple.npy',
    ],
    # get yellow key
    [
        'resources/minigrid_images/adv_doorkey_getyellowkey_dooryellow_0_termination_positive.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_yellow.npy',
    ],
    # get grey key
    [
        'resources/minigrid_images/adv_doorkey_getgreykey_doorgrey_0_termination_positive.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_grey.npy',
    ],
    # open red door
    [
        'resources/minigrid_images/adv_doorkey_openreddoor_doorred_0_termination_positive.npy',
    ],
    # open blue door
    [
        'resources/minigrid_images/adv_doorkey_openbluedoor_doorblue_0_termination_positive.npy',
    ],
    # open green door
    [
        'resources/minigrid_images/adv_doorkey_opengreendoor_doorgreen_0_termination_positive.npy',
    ],
    # open purple door
    [
        'resources/minigrid_images/adv_doorkey_openpurpledoor_doorpurple_0_termination_positive.npy',
    ],
    # open yellow door
    [
        'resources/minigrid_images/adv_doorkey_openyellowdoor_dooryellow_0_termination_positive.npy',
    ],
    # open grey door
    [
        'resources/minigrid_images/adv_doorkey_opengreydoor_doorgrey_0_termination_positive.npy',
    ],
]
termination_negative_files = [
    # get red key
    [
        'resources/minigrid_images/adv_doorkey_getredkey_doorred_0_termination_negative.npy'
    ],
    # get blue key
    [
        'resources/minigrid_images/adv_doorkey_getbluekey_doorblue_0_termination_negative.npy'
    ],
    # get green key
    [
        'resources/minigrid_images/adv_doorkey_getgreenkey_doorgreen_0_termination_negative.npy'
    ],
    # get purple key
    [
        'resources/minigrid_images/adv_doorkey_getpurplekey_doorpurple_0_termination_negative.npy'
    ],
    # get yellow key
    [
        'resources/minigrid_images/adv_doorkey_getyellowkey_dooryellow_0_termination_negative.npy'
    ],
    # get grey key
    [
        'resources/minigrid_images/adv_doorkey_getgreykey_doorgrey_0_termination_negative.npy'
    ],
    # open red door
    [
        'resources/minigrid_images/adv_doorkey_openreddoor_doorred_0_termination_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_red.npy',
    ],
    # open blue door
    [
        'resources/minigrid_images/adv_doorkey_openbluedoor_doorblue_0_termination_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_blue.npy',
    ],
    # open green door
    [
        'resources/minigrid_images/adv_doorkey_opengreendoor_doorgreen_0_termination_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_green.npy',
    ],
    # open purple door
    [
        'resources/minigrid_images/adv_doorkey_openpurpledoor_doorpurple_0_termination_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_purple.npy',
    ],
    # open yellow door
    [
        'resources/minigrid_images/adv_doorkey_openyellowdoor_dooryellow_0_termination_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_yellow.npy',
    ],
    # open grey door
    [
        'resources/minigrid_images/adv_doorkey_opengreydoor_doorgrey_0_termination_negative.npy',
        'resources/minigrid_images/adv_doorkey_random_states_no_grey.npy',
    ],
]

def check_got_redkey(env):
    if env.unwrapped.carrying is not None:
        if env.unwrapped.carrying.type == 'key':
            return env.unwrapped.carrying.color == 'red'

def check_got_bluekey(env):
    if env.unwrapped.carrying is not None:
        if env.unwrapped.carrying.type == 'key':
            return env.unwrapped.carrying.color == 'blue'

def check_got_greenkey(env):
    if env.unwrapped.carrying is not None:
        if env.unwrapped.carrying.type == 'key':
            return env.unwrapped.carrying.color == 'green'

def check_got_purplekey(env):
    if env.unwrapped.carrying is not None:
        if env.unwrapped.carrying.type == 'key':
            return env.unwrapped.carrying.color == 'purple'

def check_got_yellowkey(env):
    if env.unwrapped.carrying is not None:
        if env.unwrapped.carrying.type == 'key':
            return env.unwrapped.carrying.color == 'yellow'

def check_got_greykey(env):
    if env.unwrapped.carrying is not None:
        if env.unwrapped.carrying.type == 'key':
            return env.unwrapped.carrying.color == 'grey'

def check_dooropen(env):
    door = env.get_door_obj()

    return door.is_open

