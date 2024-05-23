import logging
import datetime
import os 
import gin 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from portable.utils.utils import set_seed 
import torch 
from collections import deque 

from experiments.experiment_logger import VideoGenerator
from portable.option.divdis.divdis_option import DivDisOption
from portable.option.divdis.divdis_mock_option import DivDisMockOption

OPTION_TYPES = [
    "mock",
    "full"
]

@gin.configurable
class MonteDivDisOptionExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 policy_phi,
                 option_type,
                 option_timeout,
                 terminations=[],
                 num_heads=4,
                 gpu=0,
                 make_videos=False) -> None:
        
        self.name = experiment_name
        self.seed = seed 
        self.use_gpu = gpu
        self.option_timeout = option_timeout
        
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
        
        if option_type == "mock":
            self.option = DivDisMockOption(use_gpu=gpu,
                                           log_dir=os.path.join(self.log_dir, "option"),
                                           save_dir=os.path.join(self.save_dir, "option"),
                                           terminations=terminations,
                                           policy_phi=policy_phi,
                                           video_generator=self.video_generator,
                                           plot_dir=os.path.join(self.plot_dir, "option"),
                                           use_seed_for_initiation=True)
        else:
            self.option = DivDisOption(use_gpu=[gpu]*num_heads,
                                       log_dir=os.path.join(self.log_dir, "option"),
                                       save_dir=os.path.join(self.save_dir, "option"),
                                       num_heads=num_heads,
                                       policy_phi=policy_phi,
                                       video_generator=self.video_generator,
                                       plot_dir=os.path.join(self.plot_dir, "option"),
                                       use_seed_for_initiation=True)
    
    def add_data(self,
                 true_list,
                 false_list,
                 unlabelled_list):
        self.option.add_datafiles(positive_files=true_list,
                                  negative_files=false_list,
                                  unlabelled_files=unlabelled_list)
    
    def train_classifier(self, epochs):
        self.option.terminations.train(epochs)
    
    def train_option(self,
                     env,
                     env_seed,
                     max_steps=1e6):
        for head_idx in range(self.option.num_heads):
            train_rewards = deque(maxlen=200)
            episode = 0
            total_steps = 0
            while total_steps < max_steps:
                rand_num = np.random.randint(low=0, high=50)
                obs, info = env.reset(agent_reposition_attempts=rand_num)
                _, _, _, steps, _, rewards, _, _ = self.option.train_policy(head_idx,
                                                                            env,
                                                                            obs, 
                                                                            info,
                                                                            env_seed)
                total_steps += steps
                train_rewards.append(sum(rewards))
                if episode % 200 == 0:
                    logging.info("idx {} steps: {} average train rewards: {}".format(head_idx,
                                                                                     total_steps,
                                                                                     np.mean(train_rewards)))
                episode += 1
            logging.info("idx {} finished -> steps: {} average train reward: {}".format(head_idx,
                                                                                        total_steps,
                                                                                        np.mean(train_rewards)))
        
        self.option.save()
    


