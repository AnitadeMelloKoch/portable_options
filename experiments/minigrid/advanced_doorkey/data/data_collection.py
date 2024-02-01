from experiments.minigrid.utils import environment_builder, actions, factored_environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import matplotlib.pyplot as plt 
import numpy as np


class MiniGridDataCollector:
    def __init__(self):
        self.env_mode = None
        self.data_mode = None
        
        self.colors = ["red", "green", "blue", "purple", "yellow", "grey"]
        self.tasks = ["get_key", "open_door"]
        self.training_seed = None

    
    def collect_envs(self):
        self.training_seed = int(input("Training seed: "))
        self.env_mode = int(input("Environment: (1) advanced doorkey (2) factored doorkey: "))
        self.data_mode = int(input("Mode: (1) get_key & open_door (2) get_diff_key: "))
        
        if self.data_mode == 1:
            # Init env, not collected, just for visualisation
            fig = plt.figure(num=1, clear=True)
            ax = fig.add_subplot()
            ax.set_title("Init visualisation to get agent, key, door locations")
                
            env = self.init_env('blue', ['red','grey','grey']) 
            state, _ = env.reset()
            state = state.numpy()
            screen = env.render()
            ax.imshow(screen)
            plt.show(block=False)
            
            if self.training_seed == 0:
                agent_loc = self.process_loc_input('4,3')
                agent_facing = 'd'
                num_keys = 3
                target_key_loc = self.process_loc_input('4,2')
                keys_loc = [target_key_loc]
                keys_loc.append(self.process_loc_input('4,4'))
                keys_loc.append(self.process_loc_input('1,6'))
                
                door_loc = self.process_loc_input('2,5')
                #goal_loc = self.process_loc_input(input("Goal location: (row_num, col_num) "))
                show_path = True 
                show_turns = False
            else:
                # User input for agent, key, door locations, etc.
                agent_loc = self.process_loc_input(input("Agent location: (row_num, col_num) [e.g. 5,4]"
                                                    " [NOTE: row increases down 1-6, col increases right 1-6]: "))
                agent_facing = input(f"Agent current facing (u/d/l/r) [e.g. u] [NOTE: up, down, left, right]: ")
                
                num_keys = int(input("Number of keys: "))
                #correct_key_loc = self.process_loc_input(input("Correct key (unlocks door) location (row_num, col_num): "))
                target_key_loc = self.process_loc_input(input("Target key  location (row_num, col_num) [NOTE: can be same as correct key]: "))
                keys_loc = [target_key_loc]
                for i in range(num_keys-1):
                    keys_loc.append(self.process_loc_input(input(f"Other Key {i+1} location (row_num, col_num): ")))
                    
                door_loc = self.process_loc_input(input("Door location (row_num, col_num): "))
                #goal_loc = self.process_loc_input(input("Goal location: (row_num, col_num) "))
                show_path = True if input("Show data collection path? (y/n): ") == "y" else False
                show_turns = True if input("Show cell collection turns? (y/n): ") == "y" else False
        
            for t_idx in range(2):
                task = self.tasks[t_idx]
                for c_idx in range(len(self.colors)):
                    door_color = self.colors[c_idx]
                    key_color = door_color
                    other_keys_colour = self.colors[:c_idx] + self.colors[c_idx+1:]

                    env = self.init_env(door_color, other_keys_colour)
                    grid = GridEnv(env, task, self.training_seed, key_color, door_color, agent_loc, agent_facing, target_key_loc, keys_loc, door_loc, 
                                show_path, show_turns)
                    print(f'======START DATA COLLECTION======')
                    print(f"Collecting data for {task} task, {door_color} door, {door_color} key.")
                    grid.collect_data()

                    
        elif self.data_mode == 2:
        
            task = "get_diff_key"
            print(f"Available colors: {self.colors}")
            door_color = input("Door color: ")
            key_color = input("Key color: ")
            
            # Init env, not collected, just for visualisation
            fig = plt.figure(num=1, clear=True)
            ax = fig.add_subplot()
            ax.set_title("Init visualisation to get agent, key, door locations")

            other_rand_colors = [c for c in self.colors if c not in [door_color, key_color]]
            other_keys_colour = np.random.choice(other_rand_colors, size=2, replace=False)
            other_keys_colour = list(other_keys_colour)
            other_keys_colour.append(key_color)
            other_keys_colour = other_keys_colour[::-1]
            
            env = self.init_env(door_color, other_keys_colour) 
            state, _ = env.reset()
            state = state.numpy()
            screen = env.render()
            ax.imshow(screen)
            plt.show(block=False)
            
            # User input for agent, key, door locations, etc.
            agent_loc = self.process_loc_input(input("Agent location: (row_num, col_num) [e.g. 5,4]"
                                                " [NOTE: row increases down 1-6, col increases right 1-6]: "))
            agent_facing = input(f"Agent current facing (u/d/l/r) [e.g. u] [NOTE: up, down, left, right]: ")
            
            num_keys = int(input("Number of keys: "))
            correct_key_loc = self.process_loc_input(input("Correct key (unlocks door) location (row_num, col_num): "))
            target_key_loc = self.process_loc_input(input("Target key  location (row_num, col_num) [NOTE: can be same as correct key]: "))
            keys_loc = [correct_key_loc]
            for i in range(num_keys-1):
                keys_loc.append(self.process_loc_input(input(f"Other Key {i+1} location (row_num, col_num): ")))
                
            door_loc = self.process_loc_input(input("Door location (row_num, col_num): "))
            #goal_loc = self.process_loc_input(input("Goal location: (row_num, col_num) "))
            show_path = True if input("Show data collection path? (y/n): ") == "y" else False
            show_turns = True if input("Show cell collection turns? (y/n): ") == "y" else False
            
            env = self.init_env(door_color, other_keys_colour)
            grid = GridEnv(env, task, self.training_seed, key_color, door_color, agent_loc, agent_facing, target_key_loc, keys_loc, door_loc, 
                            show_path, show_turns)
            print(f"Collecting data for {task} task, {door_color} door, {door_color} key.")
            grid.collect_data()
            
        else:
            raise ValueError("Data collection mode not recognised! Use either 1 or 2")


    def init_env(self, door_colour, other_keys_colour):
        if self.env_mode == 1:
            env = environment_builder('AdvancedDoorKey-8x8-v0', seed=self.training_seed, grayscale=False)
            env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                door_colour=door_colour,
                                                key_colours=other_keys_colour)
        elif self.env_mode == 2:
            env = factored_environment_builder('AdvancedDoorKey-8x8-v0', seed=self.training_seed)
            env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                door_colour=door_colour,
                                                key_colours=other_keys_colour)
        else:
            raise ValueError("Environment mode not recognised! Use either 1 or 2")
        
        return env

                           
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



class GridEnv:
    def __init__(self, env, task, training_seed, key_color, door_color, agent_loc, agent_facing, target_key_loc, keys_loc, door_loc, show_path, show_turns):
        self.task = task
        self.training_seed = training_seed
        
        self.key_color = key_color
        self.door_color = door_color
        
        self.agent_loc = agent_loc
        self.agent_facing = agent_facing

        self.target_key_loc = target_key_loc
        self.correct_key_loc = keys_loc[0]
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
        
        self.fig = plt.figure(num=1, clear=True)
        self.ax = self.fig.add_subplot()  
        self.ax.set_title(f"{task} task, {self.key_color} key, {self.door_color} door.")

        self.env = env
        self.state, _ = self.env.reset()
        self.state = self.state.numpy()
        self.screen = self.env.render()
        self.ax.imshow(self.screen)
        plt.show(block=False)


    def collect_data(self):
        if self.task.lower()==('get_key' or 'getkey'):
            # SETUP
            self.init_pos, self.term_pos = True, False

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))
            
            # PICK UP KEY
            pickup_loc, facing_needed = self.key_pick_loc(self.target_key_loc)
            self.go_to(pickup_loc)
            self.turn_to(facing_needed)
            self.init_pos, self.term_pos = False, True
            self.perform_action(actions.PICKUP, 1)
            

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))
            
            # UNLOCK DOOR
            unlock_loc = (self.door_loc[0], self.door_loc[1]-1)
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.perform_action(actions.TOGGLE, 1)
            self.perform_action(actions.TOGGLE, 1)

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))
            
            # OPEN DOOR
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.perform_action(actions.TOGGLE, 1)

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))

            # ENTER OTHER ROOM
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.forward(1)
            self.collect_cell()
            self.go_to((1, self.wall_col+1))

            # SWEEP OTHER ROOM
            self.sweep((1, self.wall_col+1), (6,6))
            
            # SAVE IMAGES
            self.save_to_file()
            
        
        elif self.task.lower()==('open_door' or 'opendoor'):
            # SETUP
            self.init_pos, self.term_pos = False, False

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))
            
            # PICK UP KEY
            pickup_loc, facing_needed = self.key_pick_loc(self.target_key_loc)
            self.go_to(pickup_loc)
            self.turn_to(facing_needed)
            self.init_pos, self.term_pos = True, False
            self.perform_action(actions.PICKUP, 1)
            

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))
            
            # UNLOCK DOOR
            unlock_loc = (self.door_loc[0], self.door_loc[1]-1)
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.init_pos, self.term_pos = False, True
            self.perform_action(actions.TOGGLE, 1)
            self.init_pos, self.term_pos = True, False
            self.perform_action(actions.TOGGLE, 1)

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))
            
            # OPEN DOOR
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.init_pos, self.term_pos = False, True
            self.perform_action(actions.TOGGLE, 1)

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))

            # ENTER OTHER ROOM
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.forward(1)
            self.collect_cell()
            self.go_to((1, self.wall_col+1))

            # SWEEP OTHER ROOM
            self.sweep((1, self.wall_col+1), (6,6))
            
            # SAVE IMAGES
            self.save_to_file()

        elif self.task.lower()==('get_diff_key' or 'getdiffkey'):
            ## LOCK & CLOSED
            # SETUP
            self.init_pos, self.term_pos = True, False

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))
            
            # PICK UP KEY
            pickup_loc, facing_needed = self.key_pick_loc(self.target_key_loc)
            self.go_to(pickup_loc)
            self.turn_to(facing_needed)
            self.init_pos, self.term_pos = False, True
            self.perform_action(actions.PICKUP, 1)
            

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))

            # PICK UP CORRECT KEY
            pickup_loc, facing_needed = self.key_pick_loc(self.correct_key_loc)
            self.go_to(pickup_loc)
            self.turn_to(facing_needed)
            self.perform_action(actions.PICKUP, 1)
            
            ## UNLOCK DOOR
            unlock_loc = (self.door_loc[0], self.door_loc[1]-1)
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.perform_action(actions.TOGGLE, 1)
            self.perform_action(actions.TOGGLE, 1)

            ## UNLOCKED & CLOSED
            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))

            ## UNLOCKED & OPEN
            # OPEN DOOR
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.perform_action(actions.TOGGLE, 1)

            # GET TO TOP LEFT CORNER & SWEEP ROOM
            self.go_to((1,1))
            self.sweep((1,1), (6,self.wall_col-1))

            # ENTER OTHER ROOM
            self.go_to(unlock_loc)
            self.turn_to('r')
            self.forward(1)
            self.collect_cell()
            self.go_to((1, self.wall_col+1))

            # SWEEP OTHER ROOM
            self.sweep((1, self.wall_col+1), (6,6))
            
            # SAVE IMAGES
            self.save_to_file()

        else:
            raise ValueError("Task not recognised! Use either 'get_key' or 'open_door' or 'get_diff_key'")



    def go_to(self, loc_end):
        dist_row = loc_end[0] - self.agent_loc[0]
        dist_col = loc_end[1] - self.agent_loc[1]

        # check if agent is at wall col, then horizontal movement first, then vertical
        if self.agent_loc[1] == self.wall_col:
            # Horizontal movement (left or right)
            if dist_col > 0:  # Moving right
                self.turn_to('r')
                self.forward(dist_col)
            elif dist_col < 0:  # Moving left
                self.turn_to('l')
                self.forward(-dist_col)
                
            # Vertical movement (up or down)
            if dist_row > 0:  # Moving down
                self.turn_to('d')
                self.forward(dist_row)
            elif dist_row < 0:  # Moving up
                self.turn_to('u')
                self.forward(-dist_row)

        else: # otherwise vertical movement first, then horizontal            
            # Vertical movement (up or down)
            if dist_row > 0:  # Moving down
                self.turn_to('d')
                self.forward(dist_row)
            elif dist_row < 0:  # Moving up
                self.turn_to('u')
                self.forward(-dist_row)

            # Horizontal movement (left or right)
            if dist_col > 0:  # Moving right
                self.turn_to('r')
                self.forward(dist_col)
            elif dist_col < 0:  # Moving left
                self.turn_to('l')
                self.forward(-dist_col)

        # check if agent had arrived at target location correctly
        if self.agent_loc != loc_end:
            self.fig.savefig('experiments/minigrid/advanced_doorkey/data/go_to_error.png')
            raise ValueError('Agent not at target location!')


    def sweep(self, loc_start, loc_end):
        # Calculate distances and directions
        
        if self.agent_loc != loc_start:
            self.fig.savefig('experiments/minigrid/advanced_doorkey/data/sweep_error.png')
            raise ValueError('Agent not at start location!')
        
        dist_row, dist_col = abs(loc_end[0] - loc_start[0]), abs(loc_end[1] - loc_start[1])
        row_direction = 1 if loc_end[0] >= loc_start[0] else -1
        col_direction = 1 if loc_end[1] >= loc_start[1] else -1

        cur_row, cur_col = loc_start[0], loc_start[1]

        # Determine initial turn direction based on row direction
        initial_turn = 'd' if row_direction == 1 else 'u'
        self.turn_to(initial_turn)

        for _ in range(dist_row + 1):
            if cur_row % 2 == 1:  # odd row
                # Adjust turning direction and movement based on col_direction
                turn_action = actions.RIGHT if col_direction == 1 else actions.LEFT
                self.perform_action(turn_action, 3, show=self.show_path)
                self.agent_facing = 'r' if col_direction == 1 else 'l'

                for _ in range(dist_col):
                    #TODO: update forward
                    self.forward(1)
                    self.collect_cell()
                    cur_col += col_direction

                if cur_row != loc_end[0]:
                    self.perform_action(turn_action, 1, show=self.show_path)
                    self.agent_facing = 'd' if col_direction == 1 else 'u'
                    self.forward(1)
                    cur_row += row_direction

            elif cur_row % 2 == 0:  # even row
                # Adjust turning direction and movement based on col_direction
                turn_action = actions.LEFT if col_direction == 1 else actions.RIGHT
                self.perform_action(turn_action, 3, show=self.show_path)
                self.agent_facing = 'l' if col_direction == 1 else 'r'

                for _ in range(dist_col):
                    self.forward(1)
                    self.collect_cell()
                    cur_col -= col_direction

                if cur_row != loc_end[0]:
                    self.perform_action(turn_action, 1, show=self.show_path)
                    self.agent_facing = 'd' if col_direction == 1 else 'u'
                    self.forward(1)
                    cur_row += row_direction
                    
            else:
                raise ValueError('sweep error')

        if self.agent_loc[0] != loc_end[0]:
            self.fig.savefig('experiments/minigrid/advanced_doorkey/data/sweep_error.png')
            print(f"Agent row: {self.agent_loc[0]} | End row: {loc_end[0]}")
            raise ValueError('Agent not at end row!')


    def key_pick_loc(self, key_loc):
        # return the loc agent should go to and facing needed to pick up target key
        target_row, target_col = key_loc
        potential_loc = [(target_row-1, target_col),(target_row+1, target_col),(target_row, target_col-1),(target_row, target_col+1)]
        facing_needed = ['d','u','r','l']
        #final_loc = [loc for loc in potential_loc if ((loc[0]!=0) and (loc[0]!=7) and (loc[1]!=0) and (loc[1]!=self.wall_col) 
        #                                                and (loc!=self.other_keys_loc[0]) and (loc!=self.other_keys_loc[1]))]
        final_loc = [loc for loc in potential_loc if ((loc[0] != 0) and (loc[0] != 7) and (loc[1] != 0) and (loc[1] != self.wall_col))]

        loc_idx = potential_loc.index(final_loc[0])
        facing_needed = facing_needed[loc_idx]
        return final_loc[0], facing_needed

    def forward(self, steps):
        if steps < 0:
            raise ValueError('steps must be positive')
        
        if self.agent_facing == 'u':
            self.agent_loc = (self.agent_loc[0]-steps, self.agent_loc[1])
        elif self.agent_facing == 'd':
            self.agent_loc = (self.agent_loc[0]+steps, self.agent_loc[1])
        elif self.agent_facing == 'l':
            self.agent_loc = (self.agent_loc[0], self.agent_loc[1]-steps)
        elif self.agent_facing == 'r':
            self.agent_loc = (self.agent_loc[0], self.agent_loc[1]+steps)
            
        self.perform_action(actions.FORWARD, steps, show=self.show_path)

        
    def turn_to(self, target_direction):
        turn_map = {
            'l': {'r': 'turn around', 'u': 'right', 'd': 'left'},
            'r': {'l': 'turn around', 'u': 'left', 'd': 'right'},
            'u': {'d': 'turn around', 'l': 'left', 'r': 'right'},
            'd': {'u': 'turn around', 'l': 'right', 'r': 'left'}
        }
                
        turn_action = turn_map.get(self.agent_facing, {}).get(target_direction, 'No turn needed')
        self.agent_facing = target_direction
        
        if turn_action == 'right':
            self.perform_action(actions.RIGHT, 1, show=self.show_path)
        elif turn_action == 'left':
            self.perform_action(actions.LEFT, 1, show=self.show_path)
        elif turn_action == 'turn around':
            self.perform_action(actions.LEFT, 2, show=self.show_path)
        else:
            print('No turn needed')

        if self.agent_facing != target_direction:
            self.fig.savefig('experiments/minigrid/advanced_doorkey/data/turn_error.png')
            raise ValueError('Agent not facing target direction!')

    
    def collect_cell(self):
        # collect data for current cell, just turn to each direction and take a picture
        self.perform_action(actions.RIGHT, 4, show=self.show_turns)


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
                self.ax.clear()  # Clear the axes
                self.ax.imshow(screen)  # Update the image
                plt.draw()  # Redraw only the necessary parts
                plt.pause(0.005)  # Short pause for the update
            
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
                self.init_positive_image.append(state)
                init_msg = "in init set True"
            elif init_positive is False:
                self.init_negative_image.append(state)
                init_msg = "in init set False"
            else:
                init_msg = "Not saved to either init"
            
            if term_positive is True:
                self.term_positive_image.append(state)
                term_msg = "in term set True"
            elif term_positive is False:
                self.term_negative_image.append(state)
                term_msg = "in term set False"
            else:
                term_msg = "Not saved to either term"

            # map agent facing to arrows
            facing_map = {
                'u': '^',
                'd': 'v',
                'l': '<',
                'r': '>'
            }
            cur_facing = facing_map.get(self.agent_facing, 'No facing')
            print(f"{init_msg} | {term_msg} | {cur_facing} | {self.agent_loc} | {str(action)}")


    def save_to_file(self):
        # Task specific base name
        if self.task.lower()==('get_key' or 'getkey'):
            base_file_name = f"adv_doorkey_8x8_v2_get{self.key_color}key_door{self.door_color}_{self.training_seed}"
        elif self.task.lower()==('open_door' or 'opendoor'):
            base_file_name = f"adv_doorkey_8x8_v2_open{self.door_color}door_door{self.door_color}_{self.training_seed}"
        elif self.task.lower()==('get_diff_key' or 'getdiffkey'):
            base_file_name = f"adv_doorkey_8x8_v2_get{self.key_color}key_door{self.door_color}_{self.training_seed}"
        else:
            raise ValueError("Task not recognised! Use either")

        # Print number of images collected
        print(f'===DATA COLLECTION SUMMARY===')
        print(f'Task: {self.task}')
        print(f'Initiation: {len(self.init_positive_image)} positive, {len(self.init_negative_image)} negative.')
        print(f'Termination: {len(self.term_positive_image)} positive, {len(self.term_negative_image)} negative.') 
        print(f'Check saved image/state shape: {self.init_positive_image[0].shape}') 
        print(f'Check one image/state: {self.init_positive_image[0]}')
        
        # Save each set of images to file
        if len(self.init_positive_image) > 0:
            np.save('resources/minigrid_images/{}_initiation_positive.npy'.format(base_file_name), self.init_positive_image)
        if len(self.init_negative_image) > 0:
            np.save('resources/minigrid_images/{}_initiation_negative.npy'.format(base_file_name), self.init_negative_image)
        if len(self.term_positive_image) > 0:
            np.save('resources/minigrid_images/{}_termination_positive.npy'.format(base_file_name), self.term_positive_image)
        if len(self.term_negative_image) > 0:
            np.save('resources/minigrid_images/{}_termination_negative.npy'.format(base_file_name), self.term_negative_image)


if __name__ == "__main__":
    meta_data_collector = MiniGridDataCollector()
    meta_data_collector.collect_envs()
















