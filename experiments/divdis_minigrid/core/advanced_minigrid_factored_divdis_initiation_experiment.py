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

from portable.option.divdis.divdis_mock_option import DivDisMockOption
from experiments.experiment_logger import VideoGenerator
from portable.agent.model.ppo import ActionPPO, OptionPPO
import math

@gin.configurable
class FactoredAdvancedMinigridDivDisInitiationExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 policy_phi,
                 use_gpu,
                 terminations,
                 make_videos=False):
        
        self.name = experiment_name
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
        
        self.option = DivDisMockOption(use_gpu=use_gpu,
                                       log_dir=os.path.join(self.log_dir, "option"),
                                       save_dir=os.path.join(self.save_dir, "option"),
                                       terminations=terminations,
                                       policy_phi=policy_phi,
                                       video_generator=self.video_generator,
                                       plot_dir=os.path.join(self.plot_dir, "option"),
                                       use_seed_for_initiation=False)
    
    def save(self):
        self.option.save()
    
    def load(self):
        self.option.load()
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def run_rollout(self,
                    env,
                    obs,
                    info,
                    idx,
                    generate_video):
        if generate_video and self.video_generator is not None:
            self.video_generator.episode_start()

        _, _, steps, _, option_rewards, _, _ = self.option.train_policy(0,
                                                                        env,
                                                                        obs,
                                                                        info,
                                                                        idx,
                                                                        make_video=generate_video)
        
        if generate_video and self.video_generator is not None:
            self.video_generator.episode_end("train")
        
        return steps, option_rewards
    
    def train(self,
              envs,
              max_steps):
        total_steps = 0
        option_rewards = deque(maxlen=200)
        episode = 0
        
        while total_steps < max_steps:
            env = random.choice(envs)
            rand_num = np.random.randint(low=0, high=50)
            obs, info = env.reset(agent_reposition_attempts=rand_num,
                                  random_start=True)
            
            possible_policies = self.option.find_possible_policy(obs)[0]
            if len(possible_policies) > 0:
                policy_idx = random.choice(possible_policies)
            else:
                policy_idx = 0
            
            steps, rewards = self.run_rollout(env,
                                              obs,
                                              info,
                                              policy_idx,
                                              generate_video=True)
            
            total_steps += steps
            option_rewards.append(sum(rewards))
            
            if episode % 50 == 0:
                logging.info("steps: {} average reward: {}".format(total_steps,
                                                                          np.mean(option_rewards)))

            episode += 1



