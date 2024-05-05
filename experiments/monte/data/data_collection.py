import curses
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pfrl.wrappers import atari_wrappers
from skimage import color

from .port_wrapper import MontezumaPortWrapper, actions



def make_env(env_name, seed, max_frames):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=max_frames),
        episode_life=False,
        clip_rewards=False,
        frame_stack=False
    )
    env.seed(seed)
    
    return MontezumaPortWrapper(env)


def update_state(player_pic, state):
    state = np.roll(state, -1, 0)
    state[-1, ...] = player_pic
    
    return state


def set_ram(env, ram_state):
    env.reset()
    state_ref = env.unwrapped.ale.cloneState()
    env.unwrapped.ale.deleteState(state_ref)
    
    new_state_ref = env.unwrapped.ale.decodeState(ram_state)
    env.unwrapped.ale.restoreState(new_state_ref)
    env.unwrapped.ale.deleteState(new_state_ref)
    obs, _, _, _ = env.step(0)  # NO-OP action to update the RAM state


def get_player_position(ram):
    """
    given the ram state, get the position of the player
    """
    def _getIndex(address):
        assert type(address) == str and len(address) == 2
        row, col = tuple(address)
        row = int(row, 16) - 8
        col = int(col, 16)
        return row * 16 + col
    def getByte(ram, address):
        # Return the byte at the specified emulator RAM location
        idx = _getIndex(address)
        return ram[idx]
    # return the player position at a particular state
    x = int(getByte(ram, 'aa'))
    y = int(getByte(ram, 'ab'))
    room = int(getByte(ram, '83'))
    return x, y, room


def set_player_position(env, x, y, room):
    """
    set the player position, specifically made for monte envs
    """
    state_ref = env.unwrapped.ale.cloneState()
    state = env.unwrapped.ale.encodeState(state_ref)
    env.unwrapped.ale.deleteState(state_ref)

    state[331] = x
    state[335] = y
    state[175] = room

    new_state_ref = env.unwrapped.ale.decodeState(state)
    env.unwrapped.ale.restoreState(new_state_ref)
    env.unwrapped.ale.deleteState(new_state_ref)
    env.step(0)  # NO-OP action to update the RAM state



class MonteDataCollector:
    def __init__(self, env_name, seed, max_frames):
        self.env = make_env(env_name, seed, max_frames)
        self.env.reset()
        self.state = np.zeros((4, 84, 84))
        self.save_dir = 'resources/monte_images/'
        
        self.init_positive_states = []
        self.init_negative_states = []
        self.term_positive_states = []
        self.term_negative_states = []
        self.uncertain_states = []
        self.INITIATION = False
        self.TERMINATION = False
        self.UNCERTAIN = False

        # init visualization
        self.fig = plt.figure(num=1, figsize=(10,10), clear=True)
        self.ax = self.fig.add_subplot()
        screen = self.env.render('rgb_array')
        self.ax.clear()  # Clear the axes
        self.ax.imshow(screen)  # Update the image
        plt.show(block=False)


    def update_state(self, player_pic, state):
        state = np.roll(state, -1, 0)
        state[-1, ...] = player_pic
        return state


    def visualize_env(self, pause=0.05):
        # update env visualization for current state
        screen = self.env.render("rgb_array")
        self.ax.clear()
        self.ax.imshow(screen)
        plt.draw()
        plt.pause(pause)


    def set_ram(self, ram):
        ram_state, agent_state, state, position = ram.values()
        self.env.reset()
        state_ref = self.env.unwrapped.ale.cloneState()
        self.env.unwrapped.ale.deleteState(state_ref)

        new_state_ref = self.env.unwrapped.ale.decodeState(ram_state)
        self.env.unwrapped.ale.restoreState(new_state_ref)
        self.env.unwrapped.ale.deleteState(new_state_ref)
        obs, _, _, _ = self.env.step(0)  # NO-OP action to update the RAM state

        # check state shape
        try:
            assert state.shape == (4, 84, 84)
        except AssertionError:
            state = state.squeeze.numpy()

        if self.INITIATION:
            self.init_positive_states = [state]
        else:
            self.init_negative_states = [state]

        if self.TERMINATION:
            self.term_positive_states = [state]
        else:
            self.term_negative_states = [state]

        self.visualize_env()
        

    def collect_data(self, stdscr):
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.clear()
        stdscr.scrollok(True)
        stdscr.addstr("Now collecting data! Press ESC to exit.\n")
        stdscr.addstr("Press W, A, S, D to move. Q, E, Z, C for combined movements.\n")
        stdscr.addstr("Use arrow keys for jumping movements.\n")
        stdscr.addstr("Press I to toggle initiation, T to toggle termination. Press V to clear saved dataset. Press B to save data.\n")
        stdscr.addstr(f"Current Unitiation: {self.INITIATION}, Termination: {self.TERMINATION}, Uncertainty: {self.UNCERTAIN}\n")
        i = len(self.init_positive_states)+len(self.init_negative_states)

        self.visualize_env()

        while True:
            key = stdscr.getch()  # Get a single key press
            stdscr.addstr(f"t={i}, pressed {chr(key)} | Initiation: {self.INITIATION}, Termination: {self.TERMINATION}, Uncertainty: {self.UNCERTAIN}\n")
            
            if key == 27:  # ESC key to exit
                break
            
            elif key == ord('i'):
                self.INITIATION = True
                self.TERMINATION = False
                self.UNCERTAIN = False
                stdscr.addstr(f"Initiation set to: {self.INITIATION}\n")
            elif key == ord('t'):
                self.TERMINATION = True
                self.INITIATION = False
                self.UNCERTAIN = False
                stdscr.addstr(f"Termination set to: {self.TERMINATION}\n")
            elif key == ord('u'):
                self.UNCERTAIN = True
                self.INITIATION = False
                self.TERMINATION = False
                stdscr.addstr(f"Uncertainty set to: {self.UNCERTAIN}\n")
            elif key == ord('f'):
                self.INITIATION = False
                self.TERMINATION = False
                self.UNCERTAIN = False
                stdscr.addstr(f"All set to: {self.INITIATION}\n")
                

            elif key == ord('b'):
                stdscr.addstr(f"Save data? Dataset will be reset. (y/n)\n")
                key = stdscr.getch()
                
                if key == ord('y'):
                    # ask for save file name prefix
                    stdscr.addstr(f"Enter save file name prefix: (e.g. 'climb_down_ladder_room0')\n")
                    prefix = stdscr.getstr().decode('utf-8')
                    stdscr.addstr(f"Saving data to {self.save_dir+prefix}_initiation_positive.npy and 3 other .npy files\n")
                    self.save_data(f'{prefix}')
                    stdscr.addstr("Data saved!\n")
                else:
                    stdscr.addstr("Data not saved. Still collecting data.\n")
                
            elif key == (ord('w') or ord('W')):
                i += 1
                self.perform_action(actions.UP, 1)
            elif key == (ord('s') or ord('S')):
                i += 1
                self.perform_action(actions.DOWN, 1)
            elif key == (ord('a') or ord('A')):
                i += 1
                self.perform_action(actions.LEFT, 1)
            elif key == (ord('d') or ord('D')):
                i += 1
                self.perform_action(actions.RIGHT, 1)

            elif key == ord('q'):
                i += 1
                self.perform_action(actions.UP_LEFT, 1)
            elif key == ord('e'):
                i += 1
                self.perform_action(actions.UP_RIGHT, 1)
            elif key == ord('z'):
                i += 1
                self.perform_action(actions.DOWN_LEFT, 1)
            elif key == ord('c'):
                i += 1
                self.perform_action(actions.DOWN_RIGHT, 1)

            elif key == ord('f'): # jump
                i += 1
                self.perform_action(actions.FIRE, 1)

            elif key == curses.KEY_UP:
                i += 1
                self.perform_action(actions.UP_FIRE, 1)
            elif key == curses.KEY_DOWN:
                i += 1
                self.perform_action(actions.DOWN_FIRE, 1)
            elif key == curses.KEY_LEFT:
                i += 1
                self.perform_action(actions.LEFT_FIRE, 1)
            elif key == curses.KEY_RIGHT:
                i += 1
                self.perform_action(actions.RIGHT_FIRE, 1)
            elif key == ord(' '): 
                i += 1
                self.perform_action(actions.NOOP, 1)
                
            stdscr.refresh()


    def perform_action(self, action, steps):

        for _ in range(steps):
            # env step and update state images
            obs, _, _, _ = self.env.step(action)
            self.state = update_state(obs, self.state)

            # store in init or term dataset
            if self.INITIATION:
                self.init_positive_states.append(self.state.copy())
            else:
                self.init_negative_states.append(self.state.copy())

            if self.TERMINATION:
                self.term_positive_states.append(self.state.copy())
            else:
                self.term_negative_states.append(self.state.copy())

            if self.UNCERTAIN:
                self.uncertain_states.append(self.state.copy())

            self.visualize_env()
            

    def save_data(self, prefix):       
        if len(self.init_positive_states) > 0:
            np.save(f'{self.save_dir+prefix}_initiation_positive.npy', np.array(self.init_positive_states))
        if len(self.init_negative_states) > 0:
            np.save(f'{self.save_dir+prefix}_initiation_negative.npy', np.array(self.init_negative_states))
        if len(self.term_positive_states) > 0:
            np.save(f'{self.save_dir+prefix}_termination_positive.npy', np.array(self.term_positive_states))
        if len(self.term_negative_states) > 0:
            np.save(f'{self.save_dir+prefix}_termination_negative.npy', np.array(self.term_negative_states))
        if len(self.uncertain_states) > 0:
            np.save(f'{self.save_dir+prefix}_uncertain.npy', np.array(self.uncertain_states))


    def run(self):
        curses.wrapper(self.collect_data)



if __name__ == "__main__":
    collector = MonteDataCollector('MontezumaRevengeNoFrameskip-v4', 0, 30*60*60)
    collector.INITIATION = False
    collector.TERMINATION = False

    #start_filename = "resources/monte_env_states/room22/ladder/bottom_0.pkl"
    #start_filename = "resources/monte_env_states/room18/platforms/right.pkl"
    start_filename = "resources/monte_env_states/room18/bridge/right_bridge.pkl"
    
    with open(start_filename, "rb") as f:
        start_ram = pickle.load(f)
    collector.set_ram(start_ram)

    collector.run()
