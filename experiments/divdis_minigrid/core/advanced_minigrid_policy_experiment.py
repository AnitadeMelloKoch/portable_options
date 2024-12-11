import logging
import datetime
import os
import time 
import gin 
import pickle 
import numpy as np 
import matplotlib.pyplot as plt
from portable.utils.utils import set_seed
import torch 
from collections import deque
from portable.option.divdis.policy_evaluation import get_wasserstain_distance, get_kl_distance, get_policy_similarity_metric

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
        
        
        self.train_policy(base_option, env_1, env_seed_1, max_steps=5e4)
        self.train_policy(trained_option, env_2, env_seed_2, max_steps=5e4)
        
        for _ in range(evaluate_num):
            if evaluation_type != "psm":
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
                
                base_q_values = base_q_values.detach().cpu().squeeze()
                rand_q_values = rand_q_values.detach().cpu().squeeze()
                trained_q_values = trained_q_values.detach().cpu().squeeze()
                
                if evaluation_type == "wass": 
                    rand_wass = get_wasserstain_distance(base_q_values, rand_q_values)
                    trained_wass = get_wasserstain_distance(base_q_values, trained_q_values)
                    
                    print("random wass:", rand_wass)
                    logging.info("random wass: {}".format(rand_wass))
                    print("trained wass", trained_wass)
                    logging.info("trained wass: {}".format(trained_wass))
                
                if evaluation_type == "kl": 
                    rand_kl = get_kl_distance(base_q_values, rand_q_values)
                    trained_kl = get_kl_distance(base_q_values, trained_q_values)
                    
                    print("random kl:", rand_kl)
                    logging.info("random kl: {}".format(rand_kl))
                    print("trained kl", trained_kl)
                    logging.info("trained kl {}".format(trained_kl))

            else:
                test_buffer_traj = self.get_test_buffer_trajectories(base_option, 
                                                env_1, 
                                                0,
                                                env_seed_1)
                test_buffer_traj_trained = self.get_test_buffer_trajectories(trained_option,
                                                            env_2,
                                                            0,
                                                            env_seed_2)
                
                base_q_values = [base_option.evaluate_states(0,traj,env_seed_1)[1] for traj in test_buffer_traj]
                trained_q_values = [trained_option.evaluate_states(0,traj,env_seed_2)[1] for traj in test_buffer_traj_trained]
                rand_q_values = [rand_policy.batch_act(traj)[1] for traj in test_buffer_traj]
                
                base_q_values = [q_values.detach().cpu().squeeze() for q_values in base_q_values]
                rand_q_values = [q_values.detach().cpu().squeeze() for q_values in rand_q_values]
                trained_q_values = [q_values.detach().cpu().squeeze() for q_values in trained_q_values]
                
                rand_psm = 0
                trained_psm = 0
                for traj_idx in range(5):
                    rand_psm += get_policy_similarity_metric(base_q_values[traj_idx], rand_q_values[traj_idx], use_gpu=self.use_gpu)
                    trained_psm += get_policy_similarity_metric(base_q_values[traj_idx], trained_q_values[traj_idx], use_gpu=self.use_gpu)
                
                rand_psm /= 5
                trained_psm /= 5
                
                print("random psm:", rand_psm)
                logging.info("random psm: {}".format(rand_psm))
                print("trained psm", trained_psm)
                logging.info("trained psm {}".format(trained_psm))
    
    def evaluate_two_policies_mock_option(self,
                                           env_1,
                                           env_2_list,
                                           env_3_list,
                                           env_seed_1,
                                           env_seed_2_list,
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
        
        self.train_policy(base_option, env_1, env_seed_1, max_steps=3e4)
        
        for idx, seed in enumerate(env_seed_2_list):
            trained_option = DivDisMockOption(use_gpu=self.use_gpu,
                                        terminations=terminations[1],
                                        log_dir=os.path.join(self.log_dir, "trained_option"),
                                        save_dir=os.path.join(self.save_dir, "trained_option"),
                                        use_seed_for_initiation=True,
                                        policy_phi=self.policy_phi,
                                        video_generator=self.video_generator)
            wrong_trained_option = DivDisMockOption(use_gpu=self.use_gpu,
                                                    terminations=terminations[2],
                                                    log_dir=os.path.join(self.log_dir, "wrong_option"),
                                                    save_dir=os.path.join(self.save_dir, "wrong_option"),
                                                    use_seed_for_initiation=True,
                                                    policy_phi=self.policy_phi,
                                                    video_generator=self.video_generator)
            logging.info(f"Now training new options for seed: {seed}")
            
            self.train_policy(trained_option, env_2_list[idx], seed, max_steps=3e4)
            self.train_policy(wrong_trained_option, env_3_list[idx], seed, max_steps=3e4)

            wrong_psm_list = []
            trained_psm_list = []
            rand_psm_list = []
            
            for _ in range(evaluate_num):
                if evaluation_type != "psm":
                    test_buffer = self.get_test_buffer(base_option, 
                                                env_1, 
                                                500,
                                                0,
                                                env_seed_1)
                    test_buffer = test_buffer.to("cuda")
                    
                    _, base_q_values = base_option.evaluate_states(0,
                                                            test_buffer,
                                                            env_seed_1)
                    _, trained_q_values = trained_option.evaluate_states(0,
                                                                        test_buffer,
                                                                        seed)
                    
                    _, wrong_q_values = wrong_trained_option.evaluate_states(0,
                                                                            test_buffer,
                                                                            seed)
                    _, rand_q_values = rand_policy.batch_act(test_buffer)

                
                    base_q_values = base_q_values.detach().cpu().squeeze()
                    wrong_q_values = wrong_q_values.detach().cpu().squeeze()
                    trained_q_values = trained_q_values.detach().cpu().squeeze()
                    rand_q_values = rand_q_values.detach().cpu().squeeze()
                
                    if evaluation_type == "wass": 
                        wrong_wass = get_wasserstain_distance(base_q_values, wrong_q_values)
                        trained_wass = get_wasserstain_distance(base_q_values, trained_q_values)
                        rand_wass = get_wasserstain_distance(base_q_values, rand_q_values)
                        
                        print("Seed:", seed)
                        logging.info(f"Seed: {seed}")
                        print("wrong wass:", wrong_wass)
                        logging.info("wrong wass: {}".format(wrong_wass))
                        print("right wass", trained_wass)
                        logging.info("right wass: {}".format(trained_wass))
                        print("rand wass: {}".format(rand_wass))
                        logging.info("rand wass: {}".format(rand_wass))
                    
                    if evaluation_type == "kl": 
                        wrong_kl = get_kl_distance(base_q_values, wrong_q_values)
                        trained_kl = get_kl_distance(base_q_values, trained_q_values)
                        rand_kl = get_kl_distance(base_q_values, rand_q_values)
                        
                        print("Seed:", seed)
                        logging.info(f"Seed: {seed}")
                        print("wrong kl:", wrong_kl)
                        logging.info("wrong kl: {}".format(wrong_kl))
                        print("trained kl", trained_kl)
                        logging.info("trained kl {}".format(trained_kl))
                        print("rand kl", rand_kl)
                        logging.info("rand kl: {}".format(rand_kl))
                        
                else:
                    logging.info(f"=======================================")
                    logging.info(f"Now evaluating seed: {seed}")
                    test_buffer_traj = self.get_test_buffer_trajectories(base_option, 
                                                env_1, 
                                                20,
                                                0,
                                                env_seed_1)
                    test_buffer_traj_trained = self.get_test_buffer_trajectories(trained_option,
                                                                env_2_list[idx],
                                                                20,
                                                                0,
                                                                seed)
                    test_buffer_traj_wrong = self.get_test_buffer_trajectories(wrong_trained_option,
                                                                env_3_list[idx],
                                                                20,
                                                                0,
                                                                seed)
                    
                    
                    test_buffer_traj = [traj.to("cuda") for traj in test_buffer_traj]
                    test_buffer_traj_trained = [traj.to("cuda") for traj in test_buffer_traj_trained]
                    test_buffer_traj_wrong = [traj.to("cuda") for traj in test_buffer_traj_wrong]

                    base_q_values = [base_option.evaluate_states(0,traj,env_seed_1)[1] for traj in test_buffer_traj]
                    trained_q_values = [trained_option.evaluate_states(0,traj,seed)[1] for traj in test_buffer_traj_trained]
                    wrong_q_values = [wrong_trained_option.evaluate_states(0,traj,seed)[1] for traj in test_buffer_traj_wrong]
                    rand_q_values = [rand_policy.batch_act(traj)[1] for traj in test_buffer_traj]
                
                    base_q_values = [q_values.detach().cpu().squeeze() for q_values in base_q_values]
                    wrong_q_values = [q_values.detach().cpu().squeeze() for q_values in wrong_q_values]
                    trained_q_values = [q_values.detach().cpu().squeeze() for q_values in trained_q_values]
                    rand_q_values = [q_values.detach().cpu().squeeze() for q_values in rand_q_values]
                    
                    wrong_psm = []
                    trained_psm = []
                    rand_psm = []

                    # compare all traj pairs between base and (wrong, trained, random)
                    for base_idx in range(20):
                        for other_idx in range(20):
                            wrong_psm.append(get_policy_similarity_metric(base_q_values[base_idx], wrong_q_values[other_idx], use_gpu=self.use_gpu).detach().cpu())
                            trained_psm.append(get_policy_similarity_metric(base_q_values[base_idx], trained_q_values[other_idx], use_gpu=self.use_gpu).detach().cpu())
                            rand_psm.append(get_policy_similarity_metric(base_q_values[base_idx], rand_q_values[other_idx], use_gpu=self.use_gpu).detach().cpu())

                    wrong_psm = sum(wrong_psm) / len(wrong_psm)
                    trained_psm = sum(trained_psm) / len(trained_psm)
                    rand_psm = sum(rand_psm) / len(rand_psm)
                    
                    print("Seed:", seed)
                    logging.info(f"Seed: {seed}")
                    print("wrong psm:", wrong_psm)
                    logging.info("wrong psm: {}".format(wrong_psm))
                    print("trained psm", trained_psm)
                    logging.info("trained psm {}".format(trained_psm))
                    print("rand psm", rand_psm)
                    logging.info("rand psm: {}".format(rand_psm))

                    wrong_psm_list.append(wrong_psm)
                    trained_psm_list.append(trained_psm)
                    rand_psm_list.append(rand_psm)

            logging.info("Mean stats for Seed: {}".format(seed))
            logging.info(f"Wrong PSM: {np.mean(wrong_psm_list), np.std(wrong_psm_list)}")
            logging.info(f"Trained PSM: {np.mean(trained_psm_list), np.std(trained_psm_list)}")
            logging.info(f"Rand PSM: {np.mean(rand_psm_list), np.std(rand_psm_list)}")
            
    
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

    def get_test_buffer_trajectories(self, option, env, num_traj, head_idx, env_seed, max_steps=100):
        test_trajectories  = []
        while len(test_trajectories) < num_traj:
            rand_num = np.random.randint(80)
            obs, info = env.reset(agent_reposition_attempts=rand_num)
            _, _, _, _, _, _, states, _ = option.eval_policy(head_idx,
                                                             env,
                                                             obs,
                                                             info,
                                                             env_seed,
                                                             max_steps=max_steps)
            trajectory = [state.unsqueeze(0) for state in states]
            trajectory = torch.cat(trajectory, dim=0)
            if trajectory.dim() == 1: # in the rare case that the trajectory is only one state
                trajectory = trajectory.unsqueeze(0)
            test_trajectories.append(trajectory)
        
        return test_trajectories
    
    def train_policy(self, option, env, env_seed, max_steps=1e6):
        for head_idx in range(option.num_heads):
            train_rewards = deque(maxlen=200)
            episode = 0
            total_steps = 0
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
            
    
    def evaluate_option(self,
                        env_1,
                        seed_1,
                        env_2_builder,
                        seeds_2,
                        termination_1,
                        terminations_2,
                        evaluation_type):
        
        original_option = DivDisMockOption(use_gpu=self.use_gpu,
                                           terminations=termination_1,
                                           log_dir=os.path.join(self.log_dir, "original"),
                                           save_dir=os.path.join(self.save_dir, "original"),
                                           use_seed_for_initiation=True,
                                           policy_phi=self.policy_phi,
                                           video_generator=self.video_generator)
        
        self.train_policy(original_option, env_1, seed_1, max_steps=2e4)
        
        head_scores = np.zeros(len(terminations_2))
        
        for seed_2 in seeds_2:

            new_option = DivDisMockOption(use_gpu=self.use_gpu,
                                        terminations=terminations_2,
                                        log_dir=os.path.join(self.log_dir, "new"),
                                        save_dir=os.path.join(self.save_dir, "new"),
                                        use_seed_for_initiation=True,
                                        policy_phi=self.policy_phi,
                                        video_generator=self.video_generator)
            env_2 = env_2_builder(seed_2)
            self.train_policy(new_option, env_2, seed_2, max_steps=2e4)
            for head_idx in range(new_option.num_heads):
                test_buffer = self.get_test_buffer(new_option,
                                                env_2,
                                                500,
                                                head_idx,
                                                seed_2)
                
                test_buffer = test_buffer.to("cuda")
                
                _, original_q_values = original_option.evaluate_states(0,
                                                                    test_buffer,
                                                                    seed_1)
                
                _, new_q_values = new_option.evaluate_states(head_idx,
                                                             test_buffer,
                                                             seed_2)
                
                original_q_values = original_q_values.detach().cpu().squeeze()
                new_q_values = new_q_values.detach().cpu().squeeze()
                
                if evaluation_type == "wass":
                    score = get_wasserstain_distance(original_q_values, new_q_values)
                if evaluation_type == "kl":
                    score = get_kl_distance(original_q_values, new_q_values)
                
                print("head {} score {}".format(head_idx, score))
                logging.info("head {} score {}".format(head_idx, score))
                head_scores[head_idx] = score
            
            confidence_scores = np.zeros(new_option.num_heads)
            confidence_scores[np.argmax(head_scores)] = 1
            
            new_option.update_confidences(confidence_scores)

        
        print("final confidences: {}".format(new_option.get_confidences()))
        logging.info("final confidences: {}".format(new_option.get_confidences()))
        
    


