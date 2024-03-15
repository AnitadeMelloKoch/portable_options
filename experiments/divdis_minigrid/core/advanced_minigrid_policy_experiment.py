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
class AdvancedMinigridDivDisOptionExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 policy_phi,
                 use_gpu,
                 option_type="mock",
                 make_videos=False) -> None:
        
        self.name = experiment_name
        self.seed = seed 
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, "logs")
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
        
        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None
        
        assert option_type in OPTION_TYPES
        self.option_type = option_type
        self.policy_phi = policy_phi
    
    def evaluate_vf(self,
                    env,
                    env_seed,
                    head_num,
                    terminations=None,
                    positive_files=None,
                    negative_files=None,
                    unlabelled_files=None):
        if self.option_type == "mock":
            assert terminations is not None
            assert len(terminations) == head_num
        
        if self.option_type == "mock":
            option = DivDisMockOption(
                use_gpu=self.use_gpu,
                terminations=terminations,
                log_dir=os.path.join(self.log_dir, "option"),
                save_dir=os.path.join(self.save_dir, "option"),
                use_seed_for_initiation=True,
                policy_phi=self.policy_phi,
                video_generator=self.video_generator
            )
        elif self.option_type == "full":
            option = DivDisOption(
                use_gpu=self.use_gpu,
                log_dir=os.path.join(self.log_dir, "option"),
                save_dir=os.path.join(self.save_dir, "option"),
                num_heads=head_num,
                policy_phi=self.policy_phi,
                video_generator=self.video_generator
            )
            
            option.add_datafiles(positive_files,
                                 negative_files,
                                 unlabelled_files)
            
            option.terminations.train(epochs=300)
        
        for head in range(head_num):
            total_steps = 0
            train_rewards = deque(maxlen=200)
            episode = 0
            while total_steps < 2e6:
                rand_num = np.random.randint(50)
                obs, info = env.reset(agent_reposition_attempts=rand_num)
                _, _, _, steps, _, rewards, _, _ = self.option.train_policy(
                    head,
                    env,
                    obs,
                    info,
                    env_seed
                )
                total_steps += steps
                train_rewards.append(sum(rewards))
                if episode % 200 == 0:
                    logging.info("idx {} steps: {} average train reward: {}".format(head,
                                                                                    total_steps,
                                                                                    np.mean(train_rewards)))
                
                episode += 1
            
            logging.info("idx {} finished -> steps: {} average train reward: {}".format(head,
                                                                                        total_steps,
                                                                                        np.mean(train_rewards)))
        option.save()
    
    
    


