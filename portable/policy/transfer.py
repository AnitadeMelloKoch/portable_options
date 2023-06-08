"""
acknowledgement:
this code is adapted from 
https://github.com/lerrytang/train-procgen-pfrl/
"""

import os
import time
import argparse
from pathlib import Path
from collections import deque

import numpy as np
from procgen import ProcgenEnv

from portable.policy.vec_env import VecExtractDictObs, VecNormalize, VecChannelOrder, VecMonitor
from portable.policy.ensemble import AttentionEmbedding
from portable.policy.agents import PPO, EnsembleAgent
from portable.policy.models import PPOMLP
from portable.utils import BaseTrial
from portable.policy import logger
from portable.utils import utils
from portable.policy.procgen_curriculum import procgen_game_curriculum


class ProcgenTransferTrial(BaseTrial):
    """
    trial for training procgen
    """
    def __init__(self):
        super().__init__()
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        self.setup()

    def parse_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[self.get_common_arg_parser()]
        )
        # defaults
        parser.set_defaults(hyperparams='procgen_ensemble')

        # training
        parser.add_argument('--transfer_steps', type=int, default=500_000)
        parser.add_argument('--bandit_exploration_weight', type=float, default=500)
        parser.add_argument('--individual_spatial_feature_extractor', '-s', action='store_true', default=False,
                            help='use individual spatial feature extractor for each attention module')
        parser.add_argument('--individual_global_feature_extractor', '-g', action='store_true', default=False,
                            help='use individual global feature extractor for each attention module')

        # procgen environment
        parser.add_argument('--env', type=str, required=True,
                            help='name of the procgen environment')
        parser.add_argument('--distribution_mode', '-d', type=str, default='easy',
                            choices=['easy', 'hard', 'exploration', 'memory', 'extreme'],
                            help='distribution mode of procgen')
        parser.add_argument('--num_envs', type=int, default=8,
                            help='number of environments to run in parallel')
        parser.add_argument('--start_level', type=int, default=0,
                            help='start level of the procgen environment')
        parser.add_argument('--num_levels', type=int, default=20,
                            help='number of different levels to generate during training')
        parser.add_argument('--curriculum', action='store_true', default=False,
                            help='arrange the levels in an order of increasing difficulty. Currently only available to a few games')
        
        # agent
        parser.add_argument('--agent', type=str, default='ensemble',
                            choices=['ensemble'])
        parser.add_argument('--fix_attention_masks', action='store_true', default=False,
                            help='fix the attention mask and no longer train them')
        parser.add_argument('--load', type=str, default=None,
                            help='directory to load the saved agent and attention masks')
        parser.add_argument('--remove_feature_learner', action='store_true', default=False,
                            help='only use 1 attention mask to get 1 feature, but could be multiple policies')
        parser.add_argument('--action_selection_strat', type=str, default='ucb_leader',
                            choices=['ucb_leader', 'greedy_leader', 'uniform_leader', 
                                     'exp3_leader', "ucb_57", "ucb_window_size", "ucb_gestation"],
                            help='how to select the ensemble-action to take')
        
        args = self.parse_common_args(parser)
        # auto fill
        if args.experiment_name is None:
            args.experiment_name = args.env
        return args

    def check_params_validity(self):
        """
        check whether the params entered by the user is valid
        """
        if self.params['fix_attention_masks']:
            assert self.params['load'] is not None, "must load a saved agent if fix_attention_masks is True"
    
    def make_vector_env(self, level_index, eval=False):
        """
        make a procgen environment such that the training env only focus on 1 particular level
        the testing env will sample randomly from a number of levels

        NOTE: Warning: the eval environment does not guarantee sequential transition between `num_levels` levels.
        The following level is sampled randomly, so evaluation is not deterministic, and performed on god knows
        which levels.
        """
        venv = ProcgenEnv(
            num_envs=self.params['num_envs'],
            env_name=self.params['env'],
            num_levels=self.params['num_levels'] if eval else 1,
            start_level=0 if eval else level_index,
            distribution_mode=self.params['distribution_mode'],
            num_threads=4,
            center_agent=True,
            rand_seed=self.params['seed'],
        )
        venv = VecChannelOrder(venv, channel_order='chw')
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        return venv

    def _make_ppo_agent(self, policy, optimizer, phi=lambda x: x):
        ppo_agent = PPO(
            model=policy,
            optimizer=optimizer,
            gpu=-1 if self.params['device']=='cpu' else 0,
            gamma=self.params['gamma'],
            lambd=self.params['lambda'],
            phi=phi,
            value_func_coef=self.params['value_function_coef'],
            entropy_coef=self.params['entropy_coef'],
            update_interval=self.params['nsteps'] * self.params['num_envs'],  # nsteps is the number of parallel-env steps till an update
            minibatch_size=self.params['batch_size'],
            epochs=self.params['nepochs'],
            clip_eps=self.params['clip_range'],
            clip_eps_vf=self.params['clip_range'],
            max_grad_norm=self.params['max_grad_norm'],
        )
        return ppo_agent

    def make_agent(self, env):
        assert self.params['agent'] == 'ensemble'

        attention_embedding = AttentionEmbedding(
            embedding_size=64,
            attention_depth=32,
            num_attention_modules=1 if self.params['remove_feature_learner'] else self.params['num_policies'],
            use_individual_spatial_feature=self.params['individual_spatial_feature_extractor'],
            use_individual_global_feature=self.params['individual_global_feature_extractor'],
            plot_dir=self.params['plots_dir'],
        )
        def _make_policy_and_opt():
            policy = PPOMLP(
                output_size=env.action_space.n,
            )
            optimizer = None
            return policy, optimizer
        base_learners = [
            self._make_ppo_agent(*_make_policy_and_opt(), phi=lambda x: x) 
            for _ in range(self.params['num_policies'])
        ]
        agent = EnsembleAgent(
            attention_model=attention_embedding,
            learning_rate=5e-4,
            learners=base_learners,
            device=self.params['device'],
            warmup_steps=self.params['warmup_steps'],
            batch_size=self.params['attention_batch_size'],
            action_selection_strategy=self.params['action_selection_strat'],
            phi=lambda x: x.astype(np.float32) / 255,
            buffer_length=self.params['buffer_length'],
            update_interval=self.params['attention_update_interval'],
            discount_rate=self.params['gamma'],
            num_modules=self.params['num_policies'],
            embedding_plot_freq=self.params['embedding_plot_freq'],
            bandit_exploration_weight=self.params['bandit_exploration_weight'],
            fix_attention_mask=self.params['fix_attention_masks'],
            use_feature_learner=not self.params['remove_feature_learner'],
            saving_dir=self.saving_dir,
        )
        if self.params['fix_attention_masks']:
            load_path = os.path.join(self.params['load'], self.expanded_agent_name, str(self.params['seed']))
            agent.load_attention_mask(load_path)
        return agent
    
    def _expand_agent_name(self):
        agent = self.params['agent']
        if agent == 'ensemble':
            agent += f"-{self.params['num_policies']}"
        self.expanded_agent_name = agent
    
    def _set_saving_dir(self):
        self._expand_agent_name()
        return Path(self.params['results_dir'], self.params['experiment_name'], self.expanded_agent_name, str(self.params['seed']))

    def make_logger(self, log_dir):
        return logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])
    
    def setup(self):
        self.check_params_validity()
        self.make_deterministic(self.params['seed'])

        # set up saving dir
        self.saving_dir = self._set_saving_dir()
        utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['saving_dir'] = self.saving_dir
        self.params['plots_dir'] = os.path.join(self.saving_dir, 'plots')
        os.mkdir(self.params['plots_dir'])

        # save hyperparams
        utils.save_hyperparams(self.saving_dir.joinpath('hyperparams.csv'), self.params)

        # logger
        self.logger = self.make_logger(self.saving_dir)

        # env
        self.train_env = self.make_vector_env(level_index=0, eval=False)
        self.eval_env = self.make_vector_env(level_index= 0, eval=True)

        # agent
        self.agent = self.make_agent(self.train_env)
    
    def transfer(self):
        # whether to use curriculum
        if self.params['curriculum']:
            level_order = procgen_game_curriculum[self.params['env']]
            assert self.params['start_level'] == 0
            assert self.params['num_levels'] == len(level_order)  # level_order only designed for 20 levels
        else:
            level_order = range(self.params['start_level'], self.params['start_level'] + self.params['num_levels'])
        # loop through training
        for i, i_level in enumerate(level_order):
            self.train_env = self.make_vector_env(level_index=i_level, eval=False)
            train_with_eval(
                agent=self.agent,
                train_env=self.train_env,
                test_env=self.eval_env,
                num_envs=self.params['num_envs'],
                max_steps=self.params['transfer_steps'],
                level_index=i_level,
                steps_offset=i * self.params['transfer_steps'],
                log_interval=100,
                logger=self.logger,
            )
            # reset the agent
            # self.agent.reset()  # if we use this, should tune down bandit exploration
        
        # save agent
        save_agent(self.agent, self.saving_dir, self.logger)


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def rollout_one_step(agent, env, obs, steps, env_max_steps=1000):

    # Step once.
    action = agent.batch_act(obs)
    new_obs, reward, done, infos = env.step(action)
    steps += 1
    reset = steps == env_max_steps
    steps[done] = 0

    # Save experience.
    agent.batch_observe(
        batch_obs=new_obs,
        batch_reward=reward,
        batch_done=done,
        batch_reset=reset,
    )

    # Get rollout statistics.
    epinfo = []
    for info in infos:
        maybe_epinfo = info.get('episode')
        if maybe_epinfo:
            epinfo.append(maybe_epinfo)

    return new_obs, steps, epinfo


def train_with_eval(
    agent,
    train_env, 
    test_env,
    num_envs,
    max_steps,
    steps_offset=0,
    level_index=0,
    log_interval=100,
    logger=None,
):
    logger.info('Train agent from scratch.')

    train_epinfo_buf = deque(maxlen=100)
    train_obs = train_env.reset()
    train_steps = np.zeros(num_envs, dtype=int)

    test_epinfo_buf = deque(maxlen=100)
    test_obs = test_env.reset()
    test_steps = np.zeros(num_envs, dtype=int)

    max_steps = max_steps // num_envs

    tstart = time.perf_counter()
    for step_cnt in range(max_steps):

        # Roll-out in the training environments.
        assert agent.training
        train_obs, train_steps, train_epinfo = rollout_one_step(
            agent=agent,
            env=train_env,
            obs=train_obs,
            steps=train_steps,
        )
        train_epinfo_buf.extend(train_epinfo)

        # Roll-out in the test environments.
        with agent.eval_mode():
            assert not agent.training
            test_obs, test_steps, test_epinfo = rollout_one_step(
                agent=agent,
                env=test_env,
                obs=test_obs,
                steps=test_steps,
            )
            test_epinfo_buf.extend(test_epinfo)

        assert agent.training

        if (step_cnt + 1) % log_interval == 0:
            tnow = time.perf_counter()
            fps = int((step_cnt + 1) * num_envs / (tnow - tstart))

            logger.logkv('level_index', level_index)
            logger.logkv('steps', step_cnt + 1)
            logger.logkv('level_total_steps', (step_cnt + 1) * num_envs)
            logger.logkv('total_steps', (step_cnt + 1) * num_envs + steps_offset)
            logger.logkv('steps_per_second', fps)
            logger.logkv('ep_reward_mean',
                         safe_mean([info['r'] for info in train_epinfo_buf]))
            logger.logkv('ep_len_mean',
                         safe_mean([info['l'] for info in train_epinfo_buf]))
            logger.logkv('eval_ep_reward_mean',
                         safe_mean([info['r'] for info in test_epinfo_buf]))
            logger.logkv('eval_ep_len_mean',
                         safe_mean([info['l'] for info in test_epinfo_buf]))
            train_stats = agent.get_statistics()
            for stats in train_stats:
                logger.logkv(stats[0], stats[1])
            logger.dumpkvs()

            tstart = time.perf_counter()


def save_agent(agent, saving_dir, logger):
    if type(agent) == PPO:
        model_path = os.path.join(saving_dir, 'model.pt')
        agent.model.save_to_file(model_path)
        logger.info(f"Model saved to {model_path}")
    elif type(agent) == EnsembleAgent:
        agent.save(saving_dir)
        logger.info(f"Model saved to {saving_dir}/agent.pkl")
    else:
        raise RuntimeError 


if __name__ == '__main__':
    trial = ProcgenTransferTrial()
    trial.transfer()
