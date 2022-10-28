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

import torch
import torch.nn as nn
from torch import distributions
import numpy as np
import pfrl
from pfrl.nn.lmbda import Lambda
from procgen import ProcgenEnv
import pfrl
from pfrl.agents import DoubleDQN
from pfrl.utils import set_random_seed
from pfrl.q_functions import DiscreteActionValueHead

from portable.policy.vec_env import VecExtractDictObs, VecNormalize, VecChannelOrder, VecMonitor
from portable.policy.ensemble import AttentionEmbedding
from portable.policy.agents import PPO, EnsembleAgent, SAC, SingleSharedBias
from portable.policy.envs import make_ant_env
from portable.policy.models import ProcgenCNN, ImpalaCNN, PPOMLP
from portable.policy.plot import plot_reward_curve
from portable.utils import BaseTrial
from portable.policy import logger
from portable.utils import utils

class ProcgenAntTrial(BaseTrial):
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
        parser.set_defaults(hyperparams='procgen_ppo')

        # procgen environment
        parser.add_argument('--env', type=str, required=True,
                            help='name of the procgen environment')
        parser.add_argument('--distribution_mode', '-d', type=str, default='easy',
                            choices=['easy', 'hard', 'exploration, memeory', 'extreme'],
                            help='distribution mode of procgen')
        parser.add_argument('--num_envs', type=int, default=64,
                            help='number of environments to run in parallel')
        parser.add_argument('--num-threads', type=int, default=4)
        parser.add_argument('--num_levels', type=int, default=200,
                            help='number of different levels to generate during training')
        parser.add_argument('--start_level', type=int, default=0,
                            help='seed to start level generation')
        
        # agent
        parser.add_argument('--agent', type=str, default='ppo',
                            choices=['ppo', 'ensemble', 'dqn', 'sac'])
        parser.add_argument('--load', '-l', type=str, default=None,
                            help='path to load agent')
        
        args = self.parse_common_args(parser)
        # auto fill
        if args.experiment_name is None:
            args.experiment_name = args.env
        if args.agent == 'ppo':
            args.hyperparams = 'procgen_ppo'
        elif args.agent == 'ensemble':
            args.hyperparams = 'procgen_ensemble'
        elif args.agent == 'dqn':
            args.hyperparams = 'procgen_dqn'
        elif args.agent == 'sac':
            args.hyperparams = 'procgen_sac'
        return args

    def check_params_validity(self):
        """
        check whether the params entered by the user is valid
        """
        pass
    
    def make_vector_env(self, eval=False):
        """vector environment for mujoco and procgen"""
        if 'ant' in self.params['env']:
            # ant mujoco env
            venv = make_ant_env(self.params['env'], self.params['num_envs'], eval=eval)
        else:
            # procgen env
            venv = ProcgenEnv(
                num_envs=self.params['num_envs'],
                env_name=self.params['env'],
                num_levels=0 if eval else self.params['num_levels'],
                start_level=0 if eval else self.params['start_level'],
                distribution_mode=self.params['distribution_mode'],
                num_threads=self.params['num_threads'],
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
        if self.params['agent'] == 'ppo':
            policy = ImpalaCNN(
                input_shape=env.observation_space.shape,
                num_outputs=env.action_space.n,
            )
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.params['learning_rate'], eps=1e-5)
            return self._make_ppo_agent(policy, optimizer, phi=lambda x: x.astype(np.float32) / 255)

        elif self.params['agent'] == 'ensemble':
            attention_embedding = AttentionEmbedding(
                embedding_size=64,
                attention_depth=32,
                num_attention_modules=self.params['num_policies'],
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
            )
            return agent
        
        elif self.params['agent'] == 'dqn':
            n_actions = env.action_space.n
            q_func = nn.Sequential(
                ProcgenCNN(obs_space=env.observation_space, num_outputs=n_actions),  # includes linear layer for q function
                SingleSharedBias(),
                DiscreteActionValueHead(),
            )
            explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
                1.0,
                0.01,  # final epsilon
                10**6,  # final_exploration_frames
                lambda: np.random.randint(n_actions),
            )
            # Use the Nature paper's hyperparameters
            opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
                q_func.parameters(),
                lr=2.5e-4,
                alpha=0.95,
                momentum=0.0,
                eps=1e-2,
                centered=True,
            )
            agent = DoubleDQN(
                q_function=q_func,
                optimizer=opt,
                replay_buffer=pfrl.replay_buffers.ReplayBuffer(self.params['buffer_length']),
                gamma=self.params['gamma'],
                explorer=explorer,
                phi=lambda x: x.astype(np.float32) / 255,
                gpu=-1 if self.params['device']=='cpu' else 0,
                replay_start_size=self.params['warmup_steps'],
                minibatch_size=self.params['batch_size'],
                update_interval=self.params['update_interval'],
                target_update_interval=self.params['target_update_interval'],
                clip_delta=True,
                n_times_update=1,
                batch_accumulator="mean",
            )
            return agent

        elif self.params['agent'] == 'sac':
            action_space = env.action_space[0]  # get the first env's action space
            obs_size = env.observation_space.shape[-1]  # get rid of the num_envs dimension
            action_size = action_space.shape[0]  # get the only elt in the tuple

            def squashed_diagonal_gaussian_head(x):
                assert x.shape[-1] == action_size * 2
                mean, log_scale = torch.chunk(x, 2, dim=1)
                log_scale = torch.clamp(log_scale, -20.0, 2.0)
                var = torch.exp(log_scale * 2)
                base_distribution = distributions.Independent(
                    distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
                )
                # cache_size=1 is required for numerical stability
                return distributions.transformed_distribution.TransformedDistribution(
                    base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
                )

            policy = nn.Sequential(
                nn.Linear(obs_size, self.params['n_hidden_channels']),
                nn.ReLU(),
                nn.Linear(self.params['n_hidden_channels'], self.params['n_hidden_channels']),
                nn.ReLU(),
                nn.Linear(self.params['n_hidden_channels'], action_size * 2),
                Lambda(squashed_diagonal_gaussian_head),
            )
            torch.nn.init.xavier_uniform_(policy[0].weight)
            torch.nn.init.xavier_uniform_(policy[2].weight)
            torch.nn.init.xavier_uniform_(policy[4].weight)
            policy_optimizer = torch.optim.Adam(
                policy.parameters(), lr=self.params['learning_rate'], eps=self.params['adam_eps']
            )

            def make_q_func_with_optimizer():
                q_func = nn.Sequential(
                    pfrl.nn.ConcatObsAndAction(),
                    nn.Linear(obs_size + action_size, self.params['n_hidden_channels']),
                    nn.ReLU(),
                    nn.Linear(self.params['n_hidden_channels'], self.params['n_hidden_channels']),
                    nn.ReLU(),
                    nn.Linear(self.params['n_hidden_channels'], 1),
                )
                torch.nn.init.xavier_uniform_(q_func[1].weight)
                torch.nn.init.xavier_uniform_(q_func[3].weight)
                torch.nn.init.xavier_uniform_(q_func[5].weight)
                q_func_optimizer = torch.optim.Adam(
                    q_func.parameters(), lr=self.params['learning_rate'], eps=self.params['adam_eps']
                )
                return q_func, q_func_optimizer

            q_func1, q_func1_optimizer = make_q_func_with_optimizer()
            q_func2, q_func2_optimizer = make_q_func_with_optimizer()

            rbuf = pfrl.replay_buffers.ReplayBuffer(10**6, num_steps=self.params['n_step_return'])

            def burnin_action_func():
                """Select random actions until model is updated one or more times."""
                return action_space.sample()

            # Hyperparameters in http://arxiv.org/abs/1802.09477
            agent = SAC(
                policy,
                q_func1,
                q_func2,
                policy_optimizer,
                q_func1_optimizer,
                q_func2_optimizer,
                rbuf,
                gamma=self.params['discount'],
                update_interval=self.params['update_interval'],
                replay_start_size=self.params['replay_start_size'],
                gpu=-1 if self.params['device']=='cpu' else 0,
                minibatch_size=self.params['batch_size'],
                burnin_action_func=burnin_action_func,
                entropy_target=-action_size,
                temperature_optimizer_lr=self.params['learning_rate'],
            )
            return agent

        else:
            raise NotImplementedError('Unsupported agent')
    
    def _expand_agent_name(self):
        agent = self.params['agent']
        if agent == 'ensemble':
            agent += f"-{self.params['num_policies']}"
        self.expanded_agent_name = agent
    
    def _set_saving_dir(self):
        self._expand_agent_name()
        return Path(self.params['results_dir'], self.params['experiment_name'], self.expanded_agent_name, str(self.params['seed']))

    def make_logger(self, log_dir):
        logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])
    
    def setup(self):
        self.check_params_validity()
        set_random_seed(self.params['seed'])
        torch.backends.cudnn.benchmark = True

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
        self.train_env = self.make_vector_env(eval=False)
        self.eval_env = self.make_vector_env(eval=True)

        # agent
        self.agent = self.make_agent(self.train_env)
    
    def train(self):
        train_with_eval(
            agent=self.agent,
            train_env=self.train_env,
            test_env=self.eval_env,
            num_envs=self.params['num_envs'],
            max_steps=self.params['max_steps'],
            model_dir=self.saving_dir,
            model_file=self.params['load'],
            log_interval=100,
            save_interval=self.params['save_interval'],
        )
        plot_reward_curve(self.saving_dir)


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
    model_dir,
    model_file=None,
    log_interval=100,
    save_interval=20_000,
):
    if model_file is not None:
        load_agent(agent, model_file, plot_dir=os.path.join(model_dir, 'plots'))
    else:
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

            logger.logkv('steps', step_cnt + 1)
            logger.logkv('total_steps', (step_cnt + 1) * num_envs)
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
        
        if (step_cnt + 1) % save_interval == 0:
            save_agent(agent, model_dir)

    # Save the final model.
    logger.info('Training done.')
    save_agent(agent, model_dir)


def save_agent(agent, saving_dir):
    if type(agent) == PPO:
        model_path = os.path.join(saving_dir, 'model.pt')
        agent.model.save_to_file(model_path)
        logger.info(f"Model saved to {model_path}")
    elif type(agent) == EnsembleAgent:
        agent.save(saving_dir)
        logger.info(f"Model saved to {saving_dir}/agent.pkl")
    elif type(agent) == SAC:
        agent.save(saving_dir)
        logger.info(f"Model saved to {saving_dir}")
    else:
        raise RuntimeError 


def load_agent(agent, load_path, plot_dir=None):
    if type(agent) == PPO:
        agent.model.load_from_file(load_path)
        logger.info(f"Model loaded from {load_path}")
    elif type(agent) == EnsembleAgent:
        EnsembleAgent.load(load_path, plot_dir=plot_dir)
    elif type(agent) == SAC:
        agent.load(load_path)
        logger.info(f"Model loaded from {load_path}")
    else:
        raise RuntimeError


if __name__ == '__main__':
    trial = ProcgenAntTrial()
    trial.train()
