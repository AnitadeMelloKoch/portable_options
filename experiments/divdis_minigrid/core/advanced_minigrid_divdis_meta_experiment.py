import logging 
import datetime 
import os 
import gin 
import numpy as np 
from portable.utils.utils import set_seed 
from torch.utils.tensorboard import SummaryWriter 
import torch 
from collections import deque
import random
import matplotlib.pyplot as plt
import pickle

from portable.option.divdis.divdis_mock_option import DivDisMockOption
from experiments.experiment_logger import VideoGenerator
from portable.agent.model.ppo import ActionPPO 
from portable.option.policy.intrinsic_motivation.tabular_count import TabularCount
import math

@gin.configurable
class AdvancedMinigridDivDisMetaExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 option_policy_phi,
                 agent_phi,
                 use_gpu,
                 action_policy,
                 action_vf,
                 terminations,
                 num_options,
                 num_primitive_actions,
                 discount_rate=0.9,
                 make_videos=False):
        
        self.name = experiment_name,
        self.seed = seed 
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        set_seed(seed)
        
        log_file = os.path.join(self.log_dir, 
                                "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        
        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None
        
        self.meta_agent = ActionPPO(use_gpu=use_gpu,
                                    policy=action_policy,
                                    value_function=action_vf,
                                    phi=agent_phi)
        
        self.options = []
        assert len(terminations) == num_options
        self.num_options = num_options
        self.num_primitive_actions = num_primitive_actions
        
        self._cumulative_discount_vector = np.array(
            [math.pow(discount_rate, n) for n in range(100)]
        )
        
        for idx, termination_list in enumerate(terminations):
            self.options.append(DivDisMockOption(use_gpu=use_gpu,
                                                 log_dir=os.path.join(self.log_dir, "option_{}".format(idx)),
                                                 save_dir=os.path.join(self.save_dir, "option_{}".format(idx)),
                                                 terminations=termination_list,
                                                 policy_phi=option_policy_phi,
                                                 video_generator=self.video_generator,
                                                 plot_dir=os.path.join(self.plot_dir, "option_{}".format(idx)),
                                                 use_seed_for_initiation=True))
        
        if len(self.options) > 0:
            self.num_heads = self.options[0].num_heads
        else:
            self.num_heads = 0
        self.gamma = discount_rate
        
        self.experiment_data = []
        
    
    def save(self):
        for option in self.options:
            option.save()
        
        with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'wb') as f:
            pickle.dump(self.experiment_data, f)
    
    def load(self):
        for option in self.options:
            option.load()
        
        with open(os.path.join(self.save_dir, "experiment_results.pkl"), 'rb') as f:
            self.experiment_data = pickle.load(f)
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def train_option_policies(self, 
                              train_envs_list, 
                              seed, 
                              max_steps):
        for o_idx, option_train_envs in enumerate(train_envs_list):
            for t_idx, term_train_envs in enumerate(option_train_envs):
                self.options[o_idx].bootstrap_policy(t_idx,
                                                     term_train_envs,
                                                     max_steps,
                                                     0.98,
                                                     seed)
    
    def get_masks_from_seed(self,
                            seed):
        action_mask = [True]*(self.num_primitive_actions)
        
        for option in self.options:
            option_mask = option.find_possible_policy(seed)
            action_mask.extend(option_mask)
        
        return action_mask
    
    def act(self, obs):
        # TODO Add action mask 
        action, q_vals = self.meta_agent.act(obs)
        
        return action, q_vals
    
    def observe(self, 
                obs, 
                rewards, 
                done):
        
        if len(rewards) > len(self._cumulative_discount_vector):
            self._cumulative_discount_vector = np.array(
                [math.pow(self.gamma, n) for n in range(len(rewards))]
            )
        
        reward = np.sum(self._cumulative_discount_vector[:len(rewards)]*rewards)
        
        self.meta_agent.observe(obs,
                                reward,
                                done,
                                done)
    
    def save_image(self, env):
        if self.video_generator is not None:
            img = env.render()
            self.video_generator.make_image(img)
    
    def train_meta_agent(self,
                         env,
                         seed,
                         max_steps,
                         min_performance):
        total_steps = 0
        episode_rewards = deque(maxlen=200)
        episode = 0
        undiscounted_rewards = []
        
        while total_steps < max_steps:
            undiscounted_reward = 0
            done = False
            
            if self.video_generator is not None:
                self.video_generator.episode_start()
            
            obs, info = env.reset()
            
            while not done:
                self.save_image(env)
                if type(obs) == np.ndarray:
                    obs = torch.from_numpy(obs).float()
                action_mask = self.get_masks_from_seed(seed)
                action, q_vals = self.act(obs)
                
                self._video_log("action: {}".format(action))
                self._video_log("action q vals: {}".format(q_vals))
                
                
                if action < self.num_primitive_actions:
                    next_obs, reward, done, info = env.step(action)
                    undiscounted_reward += reward
                    rewards = [reward]
                    # total_steps += 1
                    steps = 0
                else:
                    # if (action_mask[action] is False):
                    #     print("no actions")
                    #     next_obs, reward, done, info = env.step(6)
                    #     steps = 1
                    #     rewards = [reward]
                    # else:
                    action_offset = action-self.num_primitive_actions
                    option_num = int(action_offset/self.num_heads)
                    option_head = action_offset%self.num_heads
                    next_obs, info, done, steps, rewards, _, _, _ = self.options[option_num].train_policy(option_head,
                                                                                                            env,
                                                                                                            obs,
                                                                                                            info,
                                                                                                            seed,
                                                                                                            max_steps=50,
                                                                                                            make_video=True)
                undiscounted_reward += np.sum(rewards)
                total_steps += 1
                
                self.experiment_data.append({
                    "meta_step": total_steps,
                    "option_length": steps,
                    "option_rewards": rewards
                })
                
                self.observe(obs,
                            rewards,
                            done)
                obs = next_obs
            logging.info("Episode {} total steps: {}  average undiscounted reward: {}".format(episode,
                                                                                     total_steps,
                                                                                     np.mean(episode_rewards)))
            
            if (undiscounted_reward > 0 or episode%10==0) and self.video_generator is not None:
                self.video_generator.episode_end("episode_{}".format(episode))
            
            undiscounted_rewards.append(undiscounted_reward)
            episode += 1
            episode_rewards.append(undiscounted_reward)
            
            self.plot_learning_curve(episode_rewards)
            
            self.meta_agent.save(os.path.join(self.save_dir, "action_agent"))
            self.save()
            
            if total_steps > 1e6 and np.mean(episode_rewards) > min_performance:
                logging.info("Meta agent reached min performance {} in {} steps".format(np.mean(episode_rewards),
                                                                                        total_steps))
                return
    
    def plot_learning_curve(self,
                            rewards):
        x = np.arange(len(rewards))
        fig, ax = plt.subplots()
        ax.plot(x, rewards)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Sum Undiscounted Rewards")
        
        fig.savefig(os.path.join(self.plot_dir, "learning_curve.png"))
        
        plt.close(fig)
        
    
    def eval_meta_agent(self,
                        env,
                        seed,
                        num_runs):
        undiscounted_rewards = []
        
        with self.meta_agent.agent.eval_mode():
            for run in range(num_runs):
                total_steps = 0
                undiscounted_reward = 0
                done = False
                if self.video_generator is not None:
                    self.video_generator.episode_start()
                obs, info = env.reset()
                while not done:
                    self.save_image(env)
                    if type(obs) == np.ndarray:
                        obs = torch.from_numpy(obs).float()
                    action_mask = self.get_masks_from_seed(seed)
                    action, q_vals = self.act(obs)
                    
                    self._video_log("[meta] action: {} option: {}".format(action, option))
                    self._video_log("[meta] action q values")
                    for idx in range(len(q_vals[0])):
                        self._video_log("[meta] action {} value {}".format(idx, q_vals[0][idx]))
                    
                    if action < self.num_primitive_actions:
                        next_obs, reward, done, info = env.step(action)
                        undiscounted_reward += reward
                        rewards = [reward]
                        total_steps += 1
                    else:
                        if (action_mask[action] is False):
                            self._video_log("[meta] action+option not executable. Perform no-op")
                            next_obs, reward, done, info = env.step(6)
                            steps = 1
                            rewards = [reward]
                        else:
                            self._video_log("[meta] selected option {}".format(action-self.num_primitive_actions))
                            next_obs, info, done, steps, rewards, _, _, _ = self.options[action-self.num_primitive_actions].eval_policy(option,
                                                                                                                                  env,
                                                                                                                                  obs,
                                                                                                                                  info,
                                                                                                                                  seed)
                        undiscounted_reward += np.sum(rewards)
                        total_steps += steps
                    
                        self.observe(obs,
                                    rewards,
                                    done)
                        obs = next_obs
                
                logging.info("Eval {} total steps: {} undiscounted reward: {}".format(run,
                                                                                      total_steps,
                                                                                      undiscounted_reward))
            
                if self.video_generator is not None:
                    self.video_generator.episode_end("eval_{}".format(run))
                
                undiscounted_rewards.append(undiscounted_reward)
    
    
    
    
    













