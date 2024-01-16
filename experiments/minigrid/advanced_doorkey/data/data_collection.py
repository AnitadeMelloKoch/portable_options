import re
from turtle import title

from sklearn import base
from experiments.minigrid.utils import environment_builder, actions
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import matplotlib.pyplot as plt 
import numpy as np

from portable import agent 


class MiniGridDataCollector:
    def __init__(self, training_seed):
        self.training_seed = training_seed
        self.colours = ["red", "green", "blue", "purple", "yellow", "grey"]
        self.tasks = ["get_key", "open_door"]

        
    def process_loc_input(self, input_str):
        try:
            # Split the input string by comma
            row_str, col_str = input_str.split(',')

            # Convert each part to an integer
            row_num = int(row_str.strip())  # strip() removes any leading/trailing spaces
            col_num = int(col_str.strip())

            return (row_num, col_num)
        except ValueError:
            # Handle the error if input is not in the expected format
            print("Invalid input format. Please enter in the format 'row_num, col_num'.")
            return None
        except Exception as e:
            # Handle any other unexpected errors
            print(f"An error occurred: {e}")
            return None


    def init_env(self, door_colour, other_keys_colour):
        env = environment_builder('AdvancedDoorKey-8x8-v0', seed=self.training_seed, grayscale=False)
        # env = AdvancedDoorKeyPolicyTrainWrapper(env,
        #                                         door_colour=door_colour)
     
        env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                door_colour=door_colour,
                                                key_colours=other_keys_colour)
        
        # env = AdvancedDoorKeyPolicyTrainWrapper(
        #     factored_environment_builder(
        #         'AdvancedDoorKey-8x8-v0',
        #         seed=training_seed
        #     ),
        #     door_colour=door_colour
        # )
        
        return env

    
    def collect_env(self):
        #TODO: finish
        
        # Init env, not collected, just for visualisation
        env = self.init_env('blue', ['blue','blue','blue'])
        state, _ = env.reset()
        state = state.numpy()
        screen = env.render()
        ax.imshow(screen)
        plt.show(block=False, title="Visualisation to get agent, key, door locations")

        
        agent_loc = self.process_loc_input(input("Agent location: (row_num, col_num) || e.g. 5,4 || "
                                            "NOTE: row increases down 1-6, col increases right 1-6. "))
        agent_facing = input(f"Agent current facing: u/d/l/r. || e.g. u || NOTE: up, down, left, right ")
        num_keys = int(input("Number of keys: "))
        target_key_loc = self.process_loc_input(input("Target key location: (row_num, col_num) "))
        keys_loc = [target_key_loc]
        for i in range(num_keys-1):
            keys_loc.append(self.process_loc_input(input(f"Other Key {i+1} location: (row_num, col_num) ")))
        door_loc = self.process_loc_input(input("Door location: (row_num, col_num) "))
        goal_loc = self.process_loc_input(input("Goal location: (row_num, col_num) "))
        show_turns = input("Show cell collection turns? (y/n) ")
        show_path = input("Show data collection path? (y/n) ")
        show_turns = True if show_turns == "y" else False
        show_path = True if show_path == "y" else False

        for t_idx in range(2):
            task = self.tasks[t_idx]
            for c_idx in range(len(self.colours)):
                door_colour = self.colours[c_idx]
                other_keys_colour = self.colours[:c_idx] + self.colours[c_idx+1:]

                env = self.init_env(door_colour, other_keys_colour)
                grid = GridEnv(env, task, agent_loc, agent_facing, keys_loc, door_loc, s)
                
               
            




        
        
        self.show = show
        grid = GridEnv(env, task, agent_loc, agent_facing, keys_loc, door_loc, goal_loc, self.show)
        grid.collect_data(task, show_move=False, show_turn=False)








class GridEnv:
    def __init__(self, task, env, agent_loc, agent_facing, keys_loc, door_loc, show_path, show_turns, vis_title):
        self.task = task
        
        self.agent_loc = agent_loc
        self.agent_facing = agent_facing
        
        self.target_key_loc = keys_loc[0]
        self.other_keys_loc = keys_loc[1:]
        
        self.door_loc = door_loc
        self.wall_col = door_loc[1]
        #self.goal_loc = goal_loc
        
        self.init_positive_image = []
        self.init_negative_image = []
        self.term_positive_image = []
        self.term_negative_image = []

        self.init_pos = None
        self.term_pos = None

        self.show_turns = show_turns
        self.show_path = show_path
        
        fig = plt.figure(num=1, clear=True, title=vis_title)
        self.ax = fig.add_subplot()  

        self.env = env
        self.state, _ = self.env.reset()
        self.state = self.state.numpy()
        self.screen = self.env.render()
        self.ax.imshow(self.screen)
        plt.show(block=False)

        """user_input = input("Initiation: (y) positive (n) negative ")
        if user_input == "y":
            self.init_positive_image.append(self.state)
        elif user_input == "n":
            self.init_negative_image.append(self.state)
        else:
            print("Not saved to either")

        user_input = input("Termination: (y) positive (n) negative ")
        if user_input == "y":
            self.term_positive_image.append(self.state)
        elif user_input == "n":
            self.term_negative_image.append(self.state)
        else:
            print("Not saved to either")"""


    def collect_data(self, task):
        #TODO: finish
        # Your existing collect_data code goes here, using self.env
        if task.lower()==('get_key' or 'getkey'):
            # SETUP
            wall_col = door_loc[1]
            target_key_loc = keys_loc[0]
            other_keys = keys_loc[1:]

            # GET TO TOP LEFT CORNER
            facing = goto(agent_loc, (1,1), agent_facing, init_pos=True, term_pos=False, show=self.show_path)
        
            # SWEEP ROOM
            loc, facing = sweep((1,1), (6,wall_col-1), facing, init_pos=True, term_pos=False, show=self.show_path) # agent end up in (6,1) facing down
                
            # PICK UP KEY
            key_loc, facing_needed = key_pick_loc(target_key_loc, other_keys, wall_col)
            facing = goto(loc, key_loc, facing, init_pos=True, term_pos=False, show=show_move)
            
            # UNLOCK DOOR
            # OPEN DOOR

            # SAVE IMAGES
            self.save_to_file()
            

        
        elif task.lower()==('open_door' or 'opendoor'):

            # SETUP
            wall_col = door_loc[1]

            # GET TO TOP LEFT CORNER
            
            # SWEEP ROOM
            # PICK UP KEY
            # UNLOCK DOOR
            # OPEN DOOR
            pass
        
        pass



    def goto(self, loc_end, show):
        dist_row = loc_end[0] - self.agent_loc[0]
        dist_col = loc_end[1] - self.agent_loc[1]
        #self.agent_facing = self.agent_facing  # Initialize facing_end with current facing direction
        
        # Vertical movement (up or down)
        if dist_row > 0:  # Moving down
            self.turn_direction('d', show)
            self.perform_action(actions.FORWARD, dist_row, show=show)
        elif dist_row < 0:  # Moving up
            self.turn_direction('u', show)
            self.perform_action(actions.FORWARD, -dist_row, show=show)

        # Horizontal movement (left or right)
        if dist_col > 0:  # Moving right
            self.turn_direction('r', show)
            self.perform_action(actions.FORWARD, dist_col, show=show)
        elif dist_col < 0:  # Moving left
            self.turn_direction('l', show)
            self.perform_action(actions.FORWARD, -dist_col, show=show)


    def sweep(self, loc_start, loc_end, show):
        #TODO: fix perform action
        # Calculate distances and directions
        if self.agent_loc != loc_start:
            raise ValueError('Agent not at start location!')
        dist_row, dist_col = abs(loc_end[0] - loc_start[0]), abs(loc_end[1] - loc_start[1])
        row_direction = 1 if loc_end[0] >= loc_start[0] else -1
        col_direction = 1 if loc_end[1] >= loc_start[1] else -1

        cur_row, cur_col = loc_start[0], loc_start[1]

        # Determine initial turn direction based on row direction
        initial_turn = 'd' if row_direction == 1 else 'u'
        self.turn_direction(initial_turn, show)

        for _ in range(dist_row + 1):
            if cur_row % 2 == 1:  # odd row
                # Adjust turning direction and movement based on col_direction
                turn_action = actions.RIGHT if col_direction == 1 else actions.LEFT
                self.perform_action(env, turn_action, 3, init_positive=init_pos, term_positive=term_pos, show=show)

                for _ in range(dist_col):
                    self.perform_action(env, actions.FORWARD, 1, init_positive=init_pos, term_positive=term_pos, show=show)
                    self.collect_cell(show)
                    cur_col += col_direction

                if cur_row != loc_end[0]:
                    self.perform_action(env, turn_action, 1, init_positive=init_pos, term_positive=term_pos, show=show)
                    self.perform_action(env, actions.FORWARD, 1, init_positive=init_pos, term_positive=term_pos, show=show)
                    cur_row += row_direction

            elif cur_row % 2 == 0:  # even row
                # Adjust turning direction and movement based on col_direction
                turn_action = actions.LEFT if col_direction == 1 else actions.RIGHT
                self.perform_action(env, turn_action, 3, init_positive=init_pos, term_positive=term_pos, show=show)

                for _ in range(dist_col):
                    self.perform_action(env, actions.FORWARD, 1, init_positive=init_pos, term_positive=term_pos, show=show)
                    collect_cell(init_pos, term_pos, show)
                    cur_col -= col_direction

                if cur_row != loc_end[0]:
                    self.perform_action(env, turn_action, 1, init_positive=init_pos, term_positive=term_pos, show=show)
                    self.perform_action(env, actions.FORWARD, 1, init_positive=init_pos, term_positive=term_pos, show=show)
                    cur_row += row_direction

            else:
                raise ValueError('sweep error')

        # Determine final facing direction based on column direction
        if col_direction == 1:
            final_facing = 'r'
        else:
            final_facing = 'l'
            
        return (cur_row, cur_col), final_facing

    def key_pick_loc(self, target_key_loc):
        #TODO: fix
        # Your existing key_pick_loc code goes here, using self.env
        target_row, target_col = target_key_loc
        potential_loc = [(target_row-1, target_col),(target_row+1, target_col),(target_row, target_col-1),(target_row, target_col+1)]
        facing_needed = ['u','d','l','r']
        final_loc = [loc for loc in potential_loc if ((loc[0]!=0) and (loc[0]!=7) and (loc[1]!=0) and (loc[1]!=wall_col) 
                            and (loc!=other_keys_loc[0]) and (loc!=other_keys_loc[1]))]
        loc_idx = potential_loc.index(final_loc[0])
        facing_needed = facing_needed[loc_idx]
        return final_loc[0], facing_needed

    
    def turn_direction(self, target_direction, show):

        turn_map = {
            'l': {'r': 'turn around', 'u': 'right', 'd': 'left'},
            'r': {'l': 'turn around', 'u': 'left', 'd': 'right'},
            'u': {'d': 'turn around', 'l': 'left', 'r': 'right'},
            'd': {'u': 'turn around', 'l': 'right', 'r': 'left'}
        }

        turn_action = turn_map.get(self.agent_facing, {}).get(target_direction, 'No turn needed')
        
        if turn_action == 'right':
            self.perform_action(actions.RIGHT, 1, show=show)
        elif turn_action == 'left':
            self.perform_action(actions.LEFT, 1, show=show)
        elif turn_action == 'turn around':
            self.perform_action(actions.LEFT, 2, show=show)
        else:
            print('No turn needed')

        self.agent_facing = target_direction

    
    def collect_cell(self, show):
        # collect data for current cell, just turn to each direction and take a picture
        self.perform_action(actions.RIGHT, 4, show=show)


    def perform_action(self, action, steps, show=True, prompt=False):
        # TODO: maybe add option for NOT saving some actions (like duplicates)
        # if prompt, use user input to decide whether to save to init or term
        if prompt:
            init_positive = None
            term_positive = None
        else:
            init_positive = self.init_pos
            term_positive = self.term_pos
            
        for _ in range(steps):
            
            state, _, terminated, _  = self.env.step(action)
            state = state.numpy()
            
            if show:
                screen = self.env.render()
                self.ax.imshow(screen)
                plt.show(block=False)
                plt.pause(0.2)
            
            if init_positive is None:
                user_input = input("Initiation: (y) positive (n) negative")
                if user_input == "y":
                    self.init_positive_image.append(state)
                elif user_input == "n":
                    self.init_negative_image.append(state)
                else:
                    print("Not saved to either")

            if term_positive is None:
                user_input = input("Termination: (y) positive (n) negative")
                if user_input == "y":
                    self.term_positive_image.append(state)
                elif user_input == "n":
                    self.term_negative_image.append(state)
                else:
                    print("Not saved to either")
        
            if init_positive is True:
                print("in init set True")
                self.init_positive_image.append(state)
            elif init_positive is False:
                print("in init set False")
                self.init_negative_image.append(state)
            else:
                print("Not saved to either init")
            
            if term_positive is True:
                print("in term set True")
                self.term_positive_image.append(state)
            elif term_positive is False:
                print("in term set False")
                self.term_negative_image.append(state)
            else:
                print("Not saved to either term")


    def save_to_file(self):
        # Task specific base name
        if self.task.lower()==('get_key' or 'getkey'):
            base_file_name = "adv_doorkey_8x8_get{}key_door{}_{}".format(door_colour, door_colour, training_seed)
        elif self.task.lower()==('open_door' or 'opendoor'):
            base_file_name = "adv_doorkey_8x8_open{}door_door{}_{}".format(door_colour, door_colour, training_seed)
        else:
            raise ValueError("Task not recognised! Use either")

        # Print number of images collected
        print(f'===DATA COLLECTION SUMMARY===')
        print(f'Task: {self.task}')
        print(f'Initiation: {len(self.init_positive_image)} positive, {len(self.init_negative_image)} negative.')
        print(f'Termination: {len(self.term_positive_image)} positive, {len(self.term_negative_image)} negative.') 
        print(f'Check image shape: {self.init_positive_image[0].shape}') 
        
        # Save each set of images to file
        if len(self.init_positive_image) > 0:
            np.save('resources/minigrid_images/{}_initiation_positive.npy'.format(base_file_name), self.init_positive_image)
        if len(self.init_negative_image) > 0:
            np.save('resources/minigrid_images/{}_initiation_negative.npy'.format(base_file_name), self.init_negative_image)
        if len(self.term_positive_image) > 0:
            np.save('resources/minigrid_images/{}_termination_positive.npy'.format(base_file_name), self.term_positive_image)
        if len(self.term_negative_image) > 0:
            np.save('resources/minigrid_images/{}_termination_negative.npy'.format(base_file_name), self.term_negative_image)
















###############################################################################################
###############################################################################################
###############################################################################################
#---------------------------------------------------------------------------------------------#
training_seed = 1
colours = ["red", "green", "blue", "purple", "yellow", "grey"]
# door_colour = 'grey'
key_colour = 'purple'
door_colour = 'red'

SHOW_MOVE = True
SHOW_TURN = False
#---------------------------------------------------------------------------------------------#
###############################################################################################
###############################################################################################
###############################################################################################

env = environment_builder('AdvancedDoorKey-8x8-v0', seed=training_seed, grayscale=False)
# env = AdvancedDoorKeyPolicyTrainWrapper(env,
#                                         door_colour=door_colour)
env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                        door_colour=door_colour,
                                        key_colours=colours)
                                                    #[key_colour,
                                                     #"yellow",
                                                     #"grey"
                                                    # ])

# env = AdvancedDoorKeyPolicyTrainWrapper(
#     factored_environment_builder(
#         'AdvancedDoorKey-8x8-v0',
#         seed=training_seed
#     ),
#     door_colour=door_colour
# )

state, _ = env.reset()

state = state.numpy()

screen = env.render()
ax.imshow(screen)
plt.show(block=False)

user_input = input("Initiation: (y) positive (n) negative ")
if user_input == "y":
    init_positive_image.append(state)
elif user_input == "n":
    init_negative_image.append(state)
else:
    print("Not saved to either")

user_input = input("Termination: (y) positive (n) negative ")
if user_input == "y":
    term_positive_image.append(state)
elif user_input == "n":
    term_negative_image.append(state)
else:
    print("Not saved to either")
    

#######################################################################################################
#######################################################################################################
####                                        SEED 1                                                 ####
#######################################################################################################
#######################################################################################################

self.perform_action(env, actions.LEFT, 3             , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3            , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3            , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 6            , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 1             , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=False, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 4             , init_positive=False, term_positive=False, show=False)

#######################################################################################
#######################################################################################
##                                PICK UP KEY                                        ##
#######################################################################################
#######################################################################################

self.perform_action(env, actions.PICKUP, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 2             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 2          , init_positive=True, term_positive=False, show=False)

self.perform_action(env, actions.LEFT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3            , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 4            , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3            , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3            , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3            , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 6            , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1          , init_positive=True, term_positive=False, show=False)

#######################################################################################
#######################################################################################
##                                UNLOCK DOOR                                        ##
#######################################################################################
#######################################################################################

self.perform_action(env, actions.TOGGLE, 1            , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.TOGGLE, 1            , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 1             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 5           , init_positive=True, term_positive=False, show=False)

self.perform_action(env, actions.LEFT, 3              , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 3              , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 3              , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 4             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 3              , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.LEFT, 3              , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 3             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.RIGHT, 6             , init_positive=True, term_positive=False, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=True, term_positive=False, show=False)


#######################################################################################
#######################################################################################
##                                OPEN DOOR                                          ##
#######################################################################################
#######################################################################################

self.perform_action(env, actions.TOGGLE, 1            , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 1             , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 5           , init_positive=False, term_positive=True, show=False)

self.perform_action(env, actions.LEFT, 3              , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.LEFT, 3              , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.LEFT, 3              , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 3             , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 4             , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 3             , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.LEFT, 3              , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.LEFT, 3              , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 3             , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 3             , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 1           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 6             , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 2           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 4             , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.FORWARD, 3           , init_positive=False, term_positive=True, show=False)
self.perform_action(env, actions.RIGHT, 6             , init_positive=False, term_positive=True, show=False)



for _ in range(3):
    self.perform_action(env, actions.FORWARD, 1       , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.RIGHT, 4         , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.FORWARD, 1       , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.RIGHT, 3         , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.FORWARD, 1       , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.RIGHT, 3         , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.FORWARD, 1       , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.RIGHT, 4         , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.FORWARD, 1       , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.LEFT, 3          , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.FORWARD, 1       , init_positive=False, term_positive=True, show=True)
    self.perform_action(env, actions.LEFT, 3          , init_positive=False, term_positive=True, show=True)

self.perform_action(env, actions.PICKUP, 1)
# 64

print(f'Saved {len(init_positive_image)} init positive, {len(init_negative_image)} init negative;{len(term_positive_image)} term positive, {len(term_negative_image)} term negative images.')

print(init_positive_image[0].shape)
print(init_positive_image[0])

base_file_name = "adv_doorkey_8x8_open{}door_door{}_{}".format(door_colour, door_colour, training_seed)

if len(init_positive_image) > 0:
    np.save('resources/minigrid_images/{}_initiation_positive.npy'.format(base_file_name), init_positive_image)
if len(init_negative_image) > 0:
    np.save('resources/minigrid_images/{}_initiation_negative.npy'.format(base_file_name), init_negative_image)
if len(term_positive_image) > 0:
    np.save('resources/minigrid_images/{}_termination_positive.npy'.format(base_file_name), term_positive_image)
if len(term_negative_image) > 0:
    np.save('resources/minigrid_images/{}_termination_negative.npy'.format(base_file_name), term_negative_image)














