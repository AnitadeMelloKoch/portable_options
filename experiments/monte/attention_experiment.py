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
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from portable.utils.utils import set_seed

from portable.option import AttentionOption
from portable.option.ensemble.custom_attention import AutoEncoder
from portable.utils import set_player_ram
from portable.utils import load_init_states

@gin.configurable
class MonteExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 experiment_seed,
                 markov_option_builder,
                 policy_phi,
                 check_termination_true,
                 dataset_transform_function=None,
                 initiation_epochs=300,
                 termination_epochs=300,
                 policy_lr=1e-4,
                 policy_max_steps=1e6,
                 policy_success_threshold=0.98,
                 use_gpu=True,
                 max_episodes_in_trial=100,
                 use_agent_state=False,
                 max_option_tries=5,
                 train_options=True):
        
        self.initiation_epochs = initiation_epochs
        self.termination_epochs = termination_epochs
        self.policy_lr = policy_lr
        self.policy_max_steps = policy_max_steps
        self.policy_success_threshold = policy_success_threshold
        self.use_gpu = use_gpu
        self.max_episodes_in_trial = max_episodes_in_trial
        self.use_agent_state = use_agent_state
        self.max_option_tries = max_option_tries
        self._check_termination_true = check_termination_true
        self.train_options = train_options
        
        if self.use_gpu:
            self.embedding = AutoEncoder().to("cuda")
        else:
            self.embedding = AutoEncoder()
        self.embedding_loaded = False
        
        set_seed(experiment_seed)
        self.seed = experiment_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, experiment_name, str(experiment_seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)      
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Experiment seed: {}".format(experiment_seed))
        
        # self.trial_data = pd.DataFrame([],
        #                                columns=['start_position'
        #                                         'end_position',
        #                                         'true_terminations',
        #                                         'completed',
        #                                         'dead',
        #                                         'steps',
        #                                         'trained_instance',
        #                                         'current_instance'])
        
        self.trial_data = []
        
        option_save_dir = os.path.join(self.save_dir, 'option')
        
        self.option = AttentionOption(use_gpu=use_gpu,
                                      log_dir=os.path.join(self.log_dir, 'option'),
                                      markov_option_builder=markov_option_builder,
                                      embedding=self.embedding,
                                      policy_phi=policy_phi,
                                      dataset_transform_function=dataset_transform_function,
                                      save_dir=option_save_dir)

    def save(self):
        self.option.save()
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        with lzma.open(filename, 'wb') as f:
            dill.dump(self.trial_data, f)
        
    def load(self):
        self.option.load()
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        if os.path.exists(filename):
            with lzma.open(filename, 'rb') as f:
                self.trial_data = dill.load(f)
    
    def load_embedding(self, load_dir=None):
        if load_dir is None:
            load_dir = os.path.join(self.save_dir, 'embedding', 'model.ckpt')
        logging.info("[experiment embedding] Embedding loaded from {}".format(load_dir))
        self.embedding.load_state_dict(torch.load(load_dir))
        self.embedding_loaded = True
    
    def train_embedding(self, 
                        train_data,
                        epochs,
                        lr):
        optimizer = torch.optim.Adam(self.embedding.parameters(), lr=lr)
        mse_loss = torch.nn.MSELoss()
        base_dir = os.path.join(self.save_dir, 'embedding')
        os.makedirs(base_dir, exist_ok=True)
        save_file = os.path.join(self.save_dir, 'embedding', 'model.ckpt')
        train_x = None
        
        for epoch in range(epochs):
            train_data.shuffle()
            loss = 0
            counter_train = 0
            logging.info("[experiment embedding] Training embedding")
            for b_idx in range(train_data.num_batches):
                counter_train += 1
                x, _ = train_data.get_batch()
                x = x.to(self.device)
                train_x = x[:5]
                pred = self.embedding(x)
                mse_loss = mse_loss(pred, x)
                loss += mse_loss.item()
                mse_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            self.writer.add_scalar('embedding/mse_loss', loss/counter_train, epoch)
            for i in range(5):
                fig, axes = plt.subplots(ncols=2)
                sample = train_x[i]
                with torch.no_grad():
                    axes[0].set_axis_off()
                    axes[1].set_axis_off()
                    with torch.no_grad():
                        pred = self.embedding(sample.unsqueeze(0)).cpu().numpy()
                    axes[0].imshow(np.transpose(sample.cpu().numpy(), axes=(1,2,0)))
                    axes[1].imshow(np.transpose(pred[0], axes=(1,2,0)))

                    fig.savefig(os.path.join(base_dir, "{}.png".format(i)))
                plt.close(fig)
        torch.save(self.embedding.state_dict(), save_file)
        self.embedding_loaded = True
    
    def add_datafiles(self,
                      initiation_positive_files,
                      initiation_negative_files,
                      termination_positive_files,
                      termination_negative_files):
        self.option.initiation.add_data_from_files(initiation_positive_files,
                                                   initiation_negative_files)
        self.option.termination.add_data_from_files(termination_positive_files,
                                                    termination_negative_files)
        
    def train_option(self, training_env):
        if self.train_options is False:
            self.option.load()
            return
        logging.info('[experiment] Training option')
        self.option.initiation.train(self.initiation_epochs)
        self.option.termination.train(self.termination_epochs)
        self.option.bootstrap_policy(training_env,
                                     self.policy_max_steps,
                                     self.policy_success_threshold)
        
        self.option.save()
    
    @staticmethod
    def _set_env_ram(env, 
                     ram, 
                     state, 
                     agent_state, 
                     use_agent_space):
        env.reset()
        _ = set_player_ram(env, ram)
        env.stacked_state = state
        env.stacked_agent_state = agent_state
        
        if use_agent_space:
            return agent_state
        else:
            return state
    
    def run_instance(self, 
                    env,
                    possible_starts,
                    true_terminations,
                    eval,
                    train_instance,
                    current_instance):
        
        episode_count = 0
        results = []
        start_poses = []
        end_poses = []
        correct_terminations = []
        completeds = []
        deads = []
        final_steps = []
        instance_well_trained = False
        instantiation_instances = set()
        
        possible_rams = load_init_states(possible_starts)
        
        while episode_count < self.max_episodes_in_trial and (not instance_well_trained):
            logging.info("[run instance] Episode {}/{}".format(episode_count, self.max_episodes_in_trial))
            completed = False
            
            rand_idx = random.randint(0, len(possible_starts)-1)
            rand_state = possible_rams[rand_idx]
            state = self._set_env_ram(env,
                                      rand_state["ram"],
                                      rand_state["state"],
                                      rand_state["agent_state"],
                                      self.use_agent_state)
            
            agent_state = rand_state["agent_state"]
            start_pos = rand_state["position"]
            info = env.get_current_info({})
            attempt = 0
            timedout = 0
            must_break = False
            
            while (attempt < self.max_option_tries) and (timedout < 3) and not must_break:
                attempt += 1
                logging.info("[run instance] Attempt {}/{}".format(attempt, self.max_option_tries))
                
                option_result = self.option.run(env,
                                                state,
                                                info,
                                                eval,
                                                [])
                
                if self.option.markov_idx is not None:
                    instantiation_instances.add(self.option.markov_idx)
                
                if option_result is None:
                    logging.info("[run instance] Option did not initiate")
                    result = 0
                    must_break = True
                    position = info["position"]
                    completed = False
                    steps = 0
                else:
                    _, _, done, info, steps = option_result
                    
                    if info["needs_reset"]:
                        logging.info("[run instance] Environment needs reset")
                        must_break = True
                    if info["option_timed_out"]:
                        timedout += 1
                        logging.info("[run instance] Option timed out {}/{}".format(timedout, 3))
                    
                    if done:
                        must_break = True
                    
                    agent_state = info["stacked_agent_state"]
                    position = info["position"]

                    completed = self._check_termination_true(position, true_terminations[rand_idx], env)
                    if completed:
                        must_break = True
        
            episode_count += 1
            instance_well_trained = all([self.option.markov_instantiations[instance].is_well_trained() for instance in instantiation_instances])
            if len(instantiation_instances) == 0:
                instance_well_trained = False
            
            start_poses.append(start_pos)
            end_poses.append(position)
            correct_terminations.append(true_terminations[rand_idx])
            completeds.append(completed)
            deads.append(info["dead"])
            final_steps.append(steps)
            logging.info("Succeeded: {}".format(completed))
            
            results.append(int(completed))
            
        
        self.trial_data.append({"start_position": start_poses,
                                "end_position": end_poses,
                                "true_terminations": correct_terminations,
                                "completed": completeds,
                                "dead": deads,
                                "steps": final_steps,
                                "trained_instance": train_instance,
                                "current_instance": current_instance,
                                "eval": eval})
        
        logging.info("[experiment:run_trial] Finished trial performance: {}".format(
            np.mean(results)
            ))
        logging.info("[experiment:run_trial] All instances well trained: {}".format(
            instance_well_trained))
        if not instance_well_trained:
            logging.info("[experiment:run_trial] instance success rates:"
                .format([self.option.markov_instantiations[instance].is_well_trained() for instance in instantiation_instances])
            )

        print("[experiment] Finished trial performance: {}".format(
            np.mean(results)
            ))
        
        return instantiation_instances
    
    def bootstrap_from_room(self,
                            env,
                            possible_inits,
                            true_terminations,
                            number_episodes_in_trial,
                            use_agent_space=False):
        
        assert isinstance(possible_inits, list)
        assert isinstance(true_terminations, list)
        
        logging.info("Bootstrapping termination confidences from training room")
        print("Bootstrapping termination confidences from training room")
        
        for x in range(number_episodes_in_trial):
            logging.info("Episode {}/{}".format(x, number_episodes_in_trial))
            rand_idx = random.randint(0, len(possible_inits)-1)
            rand_state = possible_inits[rand_idx]
            state = self._set_env_ram(env,
                                      rand_state["ram"],
                                      rand_state["state"],
                                      rand_state["agent_state"],
                                      self.use_agent_state)
            
            info = env.get_current_info({})
            done = False
            
            count = 0
            timedout = 0
            
            while (not done) and (not info["needs_reset"]) and (count < 100) and (timedout<3):
                count += 1
                
                option_result = self.option.run(env,
                                                state,
                                                info,
                                                True,
                                                [])
                
                if option_result is None:
                    logging.info("initiation was not triggered")
                    self.option.initiation_update_confidence(was_successful=False, votes=self.option.initiation.votes)
                    break
                
                _, _, done, info, _ = option_result
                
                if info["needs_reset"]:
                    break
                
                if info["option_timed_out"]:
                    timedout += 1
                
                position = info["position"]
                
                if self._check_termination_true(position, true_terminations[rand_idx], env):
                    self.option.termination_update_confidence(was_successful=True, votes=self.option.termination.votes)
                    logging.info("correct termination was found")
                    break
                else:
                    self.option.termination_update_confidence(was_successful=False, votes=self.option.termination.votes)
            
    
    
    