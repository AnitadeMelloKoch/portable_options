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
from portable.option.divdis.policy_evaluation import get_wasserstain_distance, get_kl_distance

from experiments.experiment_logger import VideoGenerator
from portable.option.divdis.divdis_option import DivDisOption
from portable.option.divdis.divdis_mock_option import DivDisMockOption
from portable.option.divdis.policy.policy_and_initiation import PolicyWithInitiation

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
    
    def evaluate_diff_policies_mock_option(self,
                                           env_1,
                                           env_2,
                                           env_seed_1,
                                           env_seed_2,
                                           terminations,
                                           evaluation_type,
                                           evaluate_num=1):
        # terminations should be a two element list of lists of 
        # terminations for the two options
        base_option = DivDisMockOption(use_gpu=self.use_gpu,
                                       terminations=terminations[0],
                                       log_dir=os.path.join(self.log_dir, "base_option"),
                                       save_dir=os.path.join(self.save_dir, "base_option"),
                                       use_seed_for_initiation=True,
                                       policy_phi=self.policy_phi,
                                       video_generator=self.video_generator)
        
        rand_policy = PolicyWithInitiation(use_gpu=self.use_gpu,
                                           policy_phi=self.policy_phi,
                                           learn_initiation=False)
        rand_policy.move_to_gpu()
        
        trained_option = DivDisMockOption(use_gpu=self.use_gpu,
                                       terminations=terminations[1],
                                       log_dir=os.path.join(self.log_dir, "trained_option"),
                                       save_dir=os.path.join(self.save_dir, "trained_option"),
                                       use_seed_for_initiation=True,
                                       policy_phi=self.policy_phi,
                                       video_generator=self.video_generator)
        
        self.train_policy(base_option, env_1, env_seed_1, max_steps=1e6)
        self.train_policy(trained_option, env_2, env_seed_2, max_steps=1e6)
        
        for _ in range(evaluate_num):
        
            test_buffer = self.get_test_buffer(base_option, 
                                            env_1, 
                                            1000,
                                            0,
                                            env_seed_1)
            
            _, base_q_values = base_option.evaluate_states(0,
                                                        test_buffer,
                                                        env_seed_1)
            
            _, rand_q_values = rand_policy.batch_act(test_buffer)
            
            _, trained_q_values = trained_option.evaluate_states(0,
                                                                test_buffer,
                                                                env_seed_2)
            
            base_q_values = base_q_values.detach().cpu().squeeze().numpy()
            rand_q_values = rand_q_values.detach().cpu().squeeze().numpy()
            trained_q_values = trained_q_values.detach().cpu().squeeze().numpy()
            
            if evaluation_type == "wass": 
                rand_wass = get_wasserstain_distance(base_q_values, rand_q_values)
                trained_wass = get_wasserstain_distance(base_q_values, trained_q_values)
                
                print("random wass:", rand_wass)
                logging.info("random wass:", rand_wass)
                print("trained wass", trained_wass)
                logging.info("trained wass", trained_wass)
            
            if evaluation_type == "kl": 
                rand_kl = get_kl_distance(base_q_values, rand_q_values)
                trained_kl = get_kl_distance(base_q_values, trained_q_values)
                
                print("random kl:", rand_kl)
                logging.info("random kl:", rand_kl)
                print("trained kl", trained_kl)
                logging.info("trained kl", trained_kl)
            
    
    def get_test_buffer(self, option, env, num_states, head_idx, env_seed):
        test_states = []
        while len(test_states) < num_states:
            rand_num = np.random.randint(80)
            obs, info = env.reset(agent_reposition_attempts=rand_num)
            _, _, _, _, _, _, states, _ = option.eval_policy(head_idx,
                                                             env,
                                                             obs,
                                                             info,
                                                             env_seed)
            for idx in range(len(states)):
                states[idx] = states[idx].unsqueeze(0)
            test_states.extend(states)
        
        test_states = torch.cat(test_states, dim=0)
        
        return test_states
            
    
    def train_policy(self, option, env, env_seed, max_steps=1e6):
        total_steps = 0
        train_rewards = deque(maxlen=200)
        episode = 0
        for head_idx in range(option.num_heads):
            while total_steps < max_steps:
                rand_num = np.random.randint(low=0, high=50)
                obs, info = env.reset(agent_reposition_attempts=rand_num)
                _, _, _, steps, _, rewards, _, _ = option.train_policy(head_idx,
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
        option.save()
            
    
    
    


