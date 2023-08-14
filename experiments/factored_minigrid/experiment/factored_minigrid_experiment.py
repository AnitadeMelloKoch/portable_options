import logging 
import datetime 
import os 
import random 
import gin 
import torch 
import lzma 
import dill 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from portable.option.sets.utils import VOTE_FUNCTION_NAMES

@gin.configurable 
class FactoredMinigridExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 training_seed,
                 random_seed,
                 create_env_function,
                 num_levels,
                 create_agent_function,
                 action_space,
                 num_frames_train=10**5,
                 lr=1e-4,
                 initiation_vote_function="weighted_vote_low",
                 termination_vote_function="weighted_vote_low",
                 device_type="cpu",
                 sigma=0.5):
        
        assert device_type in ["cpu", "cuda"]
        assert initiation_vote_function in VOTE_FUNCTION_NAMES
        assert termination_vote_function in VOTE_FUNCTION_NAMES
        
        self.device = torch.device(device_type)
        
        self.make_env = create_env_function
        
        random.seed(random_seed)
        self.random_seed = random_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.random_seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.agent = create_agent_function(action_space,
                                        #    save_dir=os.path.join(self.save_dir, "option_agent"),
                                           gpu=0,
                                           n_input_channels=6,
                                           lr=lr,
                                           sigma=sigma)

        self.training_env = self.make_env(training_seed)
        
        self.training_env_seed = training_seed
        self.test_env_seeds = random.choices(np.arange(1000), k=num_levels)
        self.test_env_seeds = [int(x) for x in self.test_env_seeds]
        self.num_levels = num_levels
        self.num_frames_train = num_frames_train
        
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.random_seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Training base rainbow agent with no options")
        logging.info("Train seed: {}".format(training_seed))
        logging.info("Number of levels in experiment: {}".format(num_levels))
        logging.info("Test seeds: {}".format(self.test_env_seeds))
        
        self.trial_data = pd.DataFrame([],
                                       columns=[
                                           'reward',
                                           'seed',
                                           'frames',
                                           'env_num'
                                       ])
        
    def save(self):
        self.agent.save()
        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        with lzma.open(filename, 'wb') as f:
            dill.dump(self.trial_data, f)
            
    def load(self):
        self.option.load(self.save_dir)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        if os.path.exists(filename):
            with lzma.open(filename, 'rb') as f:
                self.trial_data = dill.load(f)
    
    def train_options(self):
        raise NotImplementedError('train_options')
    
    def train_test_envs(self):
        for idx, env_seed in enumerate(self.test_env_seeds):
            env = self.make_env(env_seed)
            total_steps = 0
            ep = 0
            while total_steps < self.num_frames_train:
                episode_rewards, steps = self.run_episode(env)
                undiscounted_return = sum(episode_rewards)
                total_steps += steps
                
                d = pd.DataFrame([{
                    "reward": episode_rewards,
                    "frames": total_steps,
                    "seed": env_seed,
                    "env_num": idx
                }])
                self.trial_data.append(d)
                
                print(100 * '-')
                print(f'Episode: {ep} for env seed: {env_seed}',
                f"Steps: {total_steps}",
                f'Reward: {undiscounted_return}')
                print(100 * '-')
                
                ep += 1
        
    
    def run_episode(self, env):
        obs, info = env.reset()
        done = False
        episode_reward = 0.
        rewards = []
        trajectory = []
        steps = 0
        
        
        while not done:
            action = self.agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            trajectory.append((obs, action, reward, next_obs, done, info["needs_reset"]))
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        self.agent.experience_replay(trajectory)
        return rewards, steps



