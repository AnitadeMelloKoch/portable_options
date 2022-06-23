from pathlib import Path
from collections import defaultdict

import gym
import cv2
import pfrl

from portable import utils

cv2.ocl.setUseOpenCL(False)


class SingleOptionTrial(utils.BaseTrial):
    """
    a base class for every class that deals with training/executing a single option
    This class should only be used for training on Montezuma
    """
    def __init__(self):
        pass

    def get_common_arg_parser(self):
        parser = super().get_common_arg_parser()
        # defaults
        parser.set_defaults(experiment_name='monte', 
                            environment='MontezumaRevengeNoFrameskip-v4', 
                            hyperparams='hyperparams/atari.csv')
    
        # environments
        parser.add_argument("--agent_space", action='store_true', default=False,
                            help="train with the agent space")
        
        # skill type
        parser.add_argument("--skill_type", "-s", type=str, default="enemy", 
                            choices=['skull', 'snake', 'spider', 'enemy', 'ladder', 'finish_game'], 
                            help="the type of skill to train")

        # start state
        parser.add_argument("--ram_dir", type=Path, default="resources/monte_ram",
                            help="where the monte ram (encoded) files are stored")
        parser.add_argument("--start_state", type=str, default=None,
                            help="""filename that saved the starting state RAM. 
                                    This should not include the whole path or the .npy extension.
                                    e.g: room1_right_ladder_top""")
        return parser
    
    def find_start_state_ram_file(self, start_state):
        """
        given the start state string, find the corresponding RAM file in ram_dir
        when the skill type is enemey, look through all the enemies directories
        """
        real_skill_type = self._get_real_skill_type(start_state)
        start_state_path = self.params['ram_dir'] / real_skill_type / f'{start_state}.npy'
        if not start_state_path.exists():
            raise FileNotFoundError(f'{start_state_path} does not exist')
        return start_state_path
    
    def _get_real_skill_type(self, saved_ram_file):
        """
        when the entered skill type is enemy, the real skil type is the actual enemy type
        """
        if self.params['skill_type'] == 'enemy':
            enemies = ['skull', 'snake', 'spider']
            for enemy in enemies:
                saved_ram_path = self.params['ram_dir'] / f'{enemy}' / f'{saved_ram_file}.npy'
                if saved_ram_path.exists():
                    real_skill = enemy
            try:
                real_skill
            except NameError:
                raise RuntimeError(f"Could not find real skill type for {saved_ram_file} and entered skill {self.params['skill_type']}")
        else:
            real_skill = self.params['skill_type']
        self.real_skill_type = real_skill
        return real_skill

    def make_env(self, env_name, env_seed, start_state=None):
        """
        Make a monte environemnt for training skills
        Args:
            goal: None or (x, y)
        """
        from portable.environment import EpisodicLifeEnv, \
            MonteAgentSpace, MonteDeepMindAgentSpace, \
            MonteForwarding, MonteLadderGoalWrapper, MonteSkullGoalWrapper, \
            MonteSpiderGoalWrapper, MonteSnakeGoalWrapper

        assert env_name == 'MontezumaRevengeNoFrameskip-v4'

        if self.params['use_deepmind_wrappers']:
            # ContinuingTimeLimit, NoopResetEnv, MaxAndSkipEnv
            env = pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60)  # 30 min with 60 fps
            # deepmind wrappers without EpisodicLife, use the custom one
            env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
                env,
                episode_life=False,
                clip_rewards=True,
                frame_stack=True,
                scale=False,
                fire_reset=False,
                channel_order="chw",
                flicker=False,
            )
            # episodic life
            env = EpisodicLifeEnv(env)
            # make agent space
            if self.params['agent_space']:
                env = MonteDeepMindAgentSpace(env)
                print('using the agent space to train the optin right now')
        else:
            env = gym.make(env_name)
            # make agent space
            if self.params['agent_space']:
                env = MonteAgentSpace(env)
                print('using the agent space to train the option right now')
        # make the agent start in another place if needed
        if start_state is not None:
            start_state_path = self.find_start_state_ram_file(start_state)
            # MonteForwarding should be after EpisodicLifeEnv so that reset() is correct
            # this does not need to be enforced once test uses the timeout wrapper
            env = MonteForwarding(env, start_state_path)
        self._get_real_skill_type(start_state)
        if self.real_skill_type == 'ladder':
            # ladder goals
            # should go after the forwarding wrappers, because the goals depend on the position of 
            # the agent in the starting state
            env = MonteLadderGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'])
            print('pursuing ladder skills')
        elif self.real_skill_type == 'skull':
            env = MonteSkullGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'])
            print('pursuing skull skills')
        elif self.real_skill_type == 'spider':
            env = MonteSpiderGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'])
            print('pursuing spider skills')
        elif self.real_skill_type == 'snake':
            env = MonteSnakeGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'])
            print('pursuing snake skills')
        print(f'making environment {env_name}')
        env.seed(env_seed)
        env.action_space.seed(env_seed)
        return env


def _getIndex(address):
    """
    helper function for parsing ram address
    get the index of the ram address using teh row and column format
    """
    assert type(address) == str and len(address) == 2
    row, col = tuple(address)
    row = int(row, 16) - 8
    col = int(col, 16)
    return row * 16 + col


def getByte(ram, address):
    """Return the byte at the specified emulator RAM location"""
    idx = _getIndex(address)
    return ram[idx]


def get_player_position(ram):
    """
    given the ram state, get the position of the player
    """
    # return the player position at a particular state
    x = int(getByte(ram, 'aa'))
    y = int(getByte(ram, 'ab'))
    return x, y


def get_skull_position(ram):
    """
    given the ram state, get the x position of the skull
    """
    x = int(getByte(ram, 'af'))
    level = 0
    screen = get_player_room_number(ram)
    skull_offset = defaultdict(lambda: 33, {
        18: [1,23,12][level],
    })[screen]
    # Note: up to some rounding, player dies when |player_x - skull_x| <= 6
    return x + skull_offset


def get_level(ram):
    return int(getByte(ram, 'b9'))


def get_object_position(ram):
    x = int(getByte(ram, 'ac'))
    y = int(getByte(ram, 'ad'))
    return x, y


def get_in_air(ram):
    # jump: 255 is on the ground, when initiating jump, turns to 16, 12, 8, 4, 0, 255, ...
    jump = getByte(ram, 'd6')
    # fall: 0 is on the groud, even positive numbers are falling (jumping is not included)
    fall = getByte(ram, 'd8')
    return jump != 255, fall > 0


def set_player_position(env, x, y):
    """
    set the player position, specifically made for monte envs
    """
    state_ref = env.unwrapped.ale.cloneState()
    state = env.unwrapped.ale.encodeState(state_ref)
    env.unwrapped.ale.deleteState(state_ref)

    state[331] = x
    state[335] = y

    new_state_ref = env.unwrapped.ale.decodeState(state)
    env.unwrapped.ale.restoreState(new_state_ref)
    env.unwrapped.ale.deleteState(new_state_ref)
    env.step(0)  # NO-OP action to update the RAM state


def get_player_room_number(ram):
    """
    given the ram state, get the room number of the player
    """
    return int(getByte(ram, '83'))


def set_player_ram(env, ram_state):
    """
    completely override the ram with a saved ram state
    """
    state_ref = env.unwrapped.ale.cloneState()
    env.unwrapped.ale.deleteState(state_ref)

    new_state_ref = env.unwrapped.ale.decodeState(ram_state)
    env.unwrapped.ale.restoreState(new_state_ref)
    env.unwrapped.ale.deleteState(new_state_ref)
    obs, _, _, _ = env.step(0)  # NO-OP action to update the RAM state
    return obs
