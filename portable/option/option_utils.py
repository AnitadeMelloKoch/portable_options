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
        super().__init__()
    
    def setup(self):
        self._expand_agent_name()
    
    def check_params_validity(self):
        if self.params['skill_type'] == 'ladder':
            print(f"changing epsilon to ladder specific one: {self.params['ladder_epsilon_tol']}")
            self.params['goal_epsilon_tol'] = self.params['ladder_epsilon_tol']

    def get_common_arg_parser(self):
        parser = super().get_common_arg_parser()
        # defaults
        parser.set_defaults(experiment_name='monte', 
                            environment='MontezumaRevengeNoFrameskip-v4', 
                            hyperparams='hyperparams/atari.csv')
    
        # environments
        parser.add_argument("--agent_space", action='store_true', default=False,
                            help="train with the agent space")

        # ensemble
        parser.add_argument("--action_selection_strat", type=str, default="ucb_leader",
                            choices=['vote', 'uniform_leader', 'greedy_leader', 'ucb_leader', 'add_qvals'],
                            help="the action selection strategy when using ensemble agent")
        
        # skill type
        parser.add_argument("--skill_type", "-s", type=str, default="ladder", 
                            choices=['skull', 'snake', 'spider', 'enemy', 'ladder', 'finish_game'], 
                            help="the type of skill to train")

        # start state
        parser.add_argument("--ram_dir", type=Path, default="resources/monte_ram",
                            help="where the monte ram (encoded) files are stored")
        parser.add_argument("--start_state", type=str, default=None,
                            help="""filename that saved the starting state RAM. 
                                    This should not include the whole path or the .npy extension.
                                    e.g: room1_right_ladder_top""")
        
        # classifiers
        parser.add_argument("--initiation_clf", "-i", action="store_true", default=False,
                            help="whether to use the initiation clf to determine when to train the skill")
        parser.add_argument("--termination_clf", "-c", action='store_true', default=False,
                            help="whether to use the trained termination classifier to determine episodic done.")
        parser.add_argument("--confidence_based_reward", action='store_true', default=False,
                            help="whether to use the confidence based reward when using the trained termination classifer")
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
    
    def _expand_agent_name(self):
        """
        expand the agent name to include information such as whether using agent space.
        """
        agent = self.params['agent']
        if agent == 'ensemble':
            agent += f"-{self.params['num_policies']}"
        if self.params['agent_space']:
            agent += '-agent-space'
        if self.params['action_selection_strat'] == 'add_qvals':
            agent += '-add-qvals'

        self.detailed_agent_name = agent

        if self.params['initiation_clf']:
            agent += '-initclf'
        if self.params['termination_clf']:
            agent += '-termclf'
            agent += f"-highconf-{self.params['termination_num_agreeing_votes']}"
        if self.params['confidence_based_reward']:
            agent += '-cbr'
        self.expanded_agent_name = agent
    
    def _set_saving_dir(self):
        self._expand_agent_name()
        return Path(self.params['results_dir']).joinpath(self.params['experiment_name']).joinpath(self.expanded_agent_name)

    def make_env(self, env_name, env_seed, eval=False, start_state=None):
        """
        Make a monte environemnt for training skills
        Args:
            goal: None or (x, y)
        """
        from portable.environment import MonteAgentWrapper, SaveOriginalFrame, NoopResetEnv, \
            MaxAndSkipEnv, wrap_deepmind, MonteForwarding, MonteTerminationSetWrapper, MonteInitiationSetWrapper, \
            MonteLadderGoalWrapper, MonteSpiderGoalWrapper, MonteSkullGoalWrapper, MonteSnakeGoalWrapper
        assert env_name == 'MontezumaRevengeNoFrameskip-v4'
        env = gym.make(env_name)
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env  # unwrap TimeLimit because we use our own in Agent wrapper
        # make agent space
        if self.params['agent_space']:
            print('using the agent space to train the option right now')
        env = MonteAgentWrapper(
            env, 
            agent_space=self.params['agent_space'],
            max_steps=self.params['eval_max_step_limit'] if eval else self.params['training_max_step_limit'],
        )
        # basic wrappers
        env = SaveOriginalFrame(env)
        env = NoopResetEnv(env, noop_max=100)
        env = MaxAndSkipEnv(env, skip=4)
        if self.params['use_deepmind_wrappers']:
            env = wrap_deepmind(
                env,
                warp_frames=not self.params['agent_space'],
                episode_life=not eval,
                clip_rewards=True,
                frame_stack=True,
                scale=False,
                fire_reset=False,
                channel_order="chw",
                flicker=False,
            )
        # starting state wrappers
        if start_state is not None:
            start_state_path = self.find_start_state_ram_file(start_state)
            # MonteForwaring should immediately follow Framestack to access the _get_ob() method
            env = MonteForwarding(env, start_state_path)
        # termination wrappers
        if self.params['termination_clf']:
            env = MonteTerminationSetWrapper(env, eval=eval, num_agreeing_votes=self.params['termination_num_agreeing_votes'], confidence_based_reward=self.params['confidence_based_reward'], device=self.params['device'])
            print('using trained termination classifier')
        # initiation wrappers
        if self.params['initiation_clf']:
            env = MonteInitiationSetWrapper(env, device=self.params['device'])
            print('using trained initiation classifier')
        # skills and goals
        self._get_real_skill_type(start_state)
        use_ground_truth = not self.params['termination_clf'] and not self.params['initiation_clf']
        if self.real_skill_type == 'ladder':
            # ladder goals
            # should go after the forwarding wrappers, because the goals depend on the position of 
            # the agent in the starting state
            env = MonteLadderGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'], info_only=not eval and not use_ground_truth)
            print('pursuing ladder skills')
        elif self.real_skill_type == 'skull':
            env = MonteSkullGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'], info_only=not eval and not use_ground_truth)
            print('pursuing skull skills')
        elif self.real_skill_type == 'spider':
            env = MonteSpiderGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'], info_only=not eval and not use_ground_truth)
            print('pursuing spider skills')
        elif self.real_skill_type == 'snake':
            env = MonteSnakeGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'], info_only=not eval and not use_ground_truth)
            print('pursuing snake skills')
        print(f'making environment {env_name}')
        env.seed(env_seed)
        env.action_space.seed(env_seed)
        return env
