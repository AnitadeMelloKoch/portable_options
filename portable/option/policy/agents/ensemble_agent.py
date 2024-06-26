import copy
import os
import lzma
import dill

import torch
import numpy as np
from pfrl import explorers
from pfrl import replay_buffers
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils.batch_states import batch_states

from portable.option.policy.agents import Agent
from portable.option.policy.attention_value_ensemble import AttentionValueEnsemble
from pfrl.collections.prioritized import PrioritizedBuffer
from portable.option.markov.markov_policy import MarkovAgent
import warnings

class EnsembleAgent(Agent):
    """
    an Agent that keeps an ensemble of policies
    an agent needs to support two methods: observe() and act()

    this class currently doesn't support batched observe() and act()
    """
    def __init__(self, 
                use_gpu, 
                warmup_steps,
                batch_size,
                phi,
                prioritized_replay_anneal_steps,
                embedding, 
                divergence_loss_scale=0.0005,
                buffer_length=100000,
                update_interval=4,
                q_target_update_interval=40,
                learning_rate=2.5e-4,
                final_epsilon=0.01,
                final_exploration_frames=10**6,
                discount_rate=0.9,
                num_modules=8, 
                num_actions=18,
                c=100,
                summary_writer=None,
                factored_obs=False):
        # vars
        self.use_gpu = use_gpu
        self.phi = phi
        self.prioritized_replay_anneal_steps = prioritized_replay_anneal_steps
        self.buffer_length = buffer_length
        self.learning_rate = learning_rate
        self.final_epsilon = final_epsilon
        self.final_exploration_frames = final_exploration_frames
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.q_target_update_interval = q_target_update_interval
        self.update_interval = update_interval
        self.num_actions = num_actions
        self.num_modules = num_modules
        self.step_numbers = [0]*self.num_modules
        self.episode_number = 0
        self.update_epochs_per_step = 1
        self.discount_rate = discount_rate
        self.c = c
        self.embedding = embedding
        
        # ensemble
        self.value_ensemble = AttentionValueEnsemble(
            use_gpu=use_gpu,
            embedding=embedding,
            learning_rate=learning_rate,
            discount_rate=discount_rate,
            attention_module_num=num_modules,
            c=c,
            num_actions=num_actions,
            divergence_loss_scale=divergence_loss_scale,
            summary_writer=summary_writer,
            factored_obs=factored_obs
        )

        # explorer
        self.explorers = [explorers.LinearDecayEpsilonGreedy(
            1.0,
            final_epsilon,
            final_exploration_frames,
            lambda: np.random.randint(num_actions),
        ) for _ in range(num_modules)]

        # Prioritized Replay
        # Anneal beta from beta0 to 1 throughout training
        # taken from https://github.com/pfnet/pfrl/blob/master/examples/atari/reproduction/rainbow/train_rainbow.py
        # self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(
        #     capacity=buffer_length,
        #     alpha=0.5,  # Exponent of errors to compute probabilities to sample
        #     beta0=0.4,  # Initial value of beta
        #     betasteps=prioritized_replay_anneal_steps,  # Steps to anneal beta to 1
        #     normalize_by_max="memory",  # method to normalize the weight
        # )
        # we use a replay buffer for each policy
        self.replay_buffers = [
            replay_buffers.PrioritizedReplayBuffer(
                capacity=buffer_length,
                alpha=0.5,
                beta0=0.4,
                betasteps=prioritized_replay_anneal_steps,
                normalize_by_max="memory"
            ) for _ in range(self.num_modules)
        ]
        # self.replay_updater = ReplayUpdater(
        #     replay_buffer=self.replay_buffer,
        #     update_func=self.update,
        #     batchsize=batch_size,
        #     episodic_update=False,
        #     episodic_update_len=None,
        #     n_times_update=1,
        #     replay_start_size=warmup_steps,
        #     update_interval=update_interval,
        # )
        self.replay_updaters = [
            ReplayUpdater(
                replay_buffer=self.replay_buffers[i],
                update_func=self.update,
                batchsize=batch_size,
                episodic_update=False,
                episodic_update_len=None,
                n_times_update=1,
                replay_start_size=warmup_steps,
                update_interval=update_interval 
            ) for i in range(self.num_modules)
        ]
        self.replay_buffers_loaded = [True] * self.num_modules
        super().__init__()
    
    def initialize_new_policy(self):

        new_policy = MarkovAgent(use_gpu=self.use_gpu,
                                 warmup_steps=self.warmup_steps,
                                 batch_size=self.batch_size,
                                 phi=self.phi,
                                 prioritized_replay_anneal_steps=self.prioritized_replay_anneal_steps,
                                 embedding=self.embedding,
                                 initial_policy=self.value_ensemble,
                                 policy_idx=self.value_ensemble.action_leader,
                                 update_interval=self.update_interval,
                                 q_target_update_interval=self.q_target_update_interval,
                                 learning_rate=2.5e-4,
                                 final_epsilon=self.final_epsilon,
                                 discount_rate=self.discount_rate,
                                 num_actions=self.num_actions)

        return new_policy
    
    def set_action_leader(self, idx):
        self.value_ensemble.action_leader = idx
    
    def action_leader(self):
        return self.value_ensemble.action_leader
    
    def move_to_gpu(self):
        self.value_ensemble.move_to_gpu()
    
    def move_to_cpu(self):
        self.value_ensemble.move_to_cpu()
    
    def store_buffer(self, save_folder):
        action_leader = self.action_leader()
        save_file = os.path.join(save_folder, '{}.pkl'.format(action_leader))
        self.replay_buffers[action_leader].save(save_file)
        self.replay_buffers[action_leader].memory = PrioritizedBuffer()
        
        self.replay_buffers_loaded[action_leader] = False
    
    def load_buffer(self, save_folder):
        action_leader = self.action_leader()
        save_file = os.path.join(save_folder, '{}.pkl'.format(action_leader))
        self.replay_buffers[action_leader].load(save_file)
        
        self.replay_buffers_loaded[action_leader] = True
    
    def update_step(self):
        self.step_numbers[self.action_leader()] += 1

    def train(self, epochs):
        if self.replay_buffers_loaded[self.action_leader()] is False:
            raise Exception("replay buffer is not loaded")
        if len(self.replay_buffers[self.action_leader()]) < self.batch_size*epochs:
            return False
        

        for _ in range(epochs):
            transitions = self.replay_buffers[self.action_leader()].sample(self.batch_size)
            self.replay_updaters[self.action_leader()].update_func(transitions)

        return True

    def observe(self, 
                obs, 
                action, 
                reward, 
                next_obs, 
                terminal, 
                update_policy=True,
                update_bandit=True):
        """
        store the experience tuple into the replayreplay_buffer buffer
        and update the agent if necessary
        """
        if self.replay_buffers_loaded[self.action_leader()] is False:
            warnings.warn("Replay buffer is not loaded. This may not be intended.")
        
        self.update_step()

        # update replay buffer 
        if self.training:
            transition = {
                "state": obs,
                "action": action,
                "reward": reward,
                "next_state": next_obs,
                "next_action": None,
                "is_state_terminal": terminal,
            }
            self.replay_buffers[self.action_leader()].append(**transition)
            if terminal:
                self.replay_buffers[self.action_leader()].stop_current_episode()

            if update_policy is True:
                self.replay_updaters[self.action_leader()].update_if_necessary(self.step_numbers[self.action_leader()])
            if update_bandit is True:
                self.value_ensemble.step()
                self.value_ensemble.update_accumulated_rewards(reward)
            
        # new episode
        if terminal:
            self.episode_number += 1
    
    def begin_rollout(self, buffer_folder, policy_leader=None, load_buffer=True):
        if policy_leader is not None:
            self.set_action_leader(policy_leader)
        else:
            self.value_ensemble.update_leader()
        
        if load_buffer:
            self.load_buffer(buffer_folder)
        self.move_to_gpu()
    
    def end_rollout(self, buffer_folder):
        self.store_buffer(buffer_folder)
        self.move_to_cpu()
    
    def update(self, experiences, errors_out=None):
        """
        update the model
        accepts as argument a list of transition dicts
        args:
            transitions (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.
        """
        if self.replay_buffers_loaded[self.action_leader()] is False:
            warnings.warn("Replay buffer is not loaded. This may not be intended.")
        
        if self.training:
            has_weight = "weight" in experiences[0][0]
            if self.use_gpu:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            exp_batch = batch_experiences(
                experiences,
                device=device,
                phi=self.phi,
                gamma=self.discount_rate,
                batch_states=batch_states,
            )
            # get weights for prioritized experience replay
            if has_weight:
                exp_batch["weights"] = torch.tensor(
                    [elem[0]["weight"] for elem in experiences],
                    device=device,
                    dtype=torch.float32,
                )
                if errors_out is None:
                    errors_out = []
            # actual update
            update_target_net =  self.step_numbers[self.action_leader()] % self.q_target_update_interval == 0
            self.value_ensemble.train(exp_batch, errors_out, update_target_net)
            # update prioritiy
            if has_weight:
                assert isinstance(self.replay_buffers[self.action_leader()], replay_buffers.PrioritizedReplayBuffer)
                self.replay_buffers[self.action_leader()].update_errors(errors_out)

    def act(self, obs, return_ensemble_info=False):
        """
        epsilon-greedy policy
        args:
            obs (object): Observation from the environment.
            return_ensemble_info (bool): when set to true, this function returns
                (action_selected, actions_selected_by_each_learner, q_values_of_each_actions_selected)
        """
        if self.replay_buffers_loaded[self.action_leader()] is False:
            warnings.warn("Replay buffer is not loaded. This may not be intended.")
        
        if self.use_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        obs = batch_states([obs], device, self.phi)
        action, action_q_val = self.value_ensemble.predict_actions(obs, return_q_values=True)
        # action selection strategy
        action_selection_func = lambda a: a
        
        # epsilon-greedy
        if self.training:
            a = self.explorers[self.action_leader()].select_action(
                self.step_numbers[self.action_leader()],
                greedy_action_func=lambda: action_selection_func(action),
            )
        else:
            randval = np.random.rand()
            if randval > 0.05:
                a = action_selection_func(action)
            else:
                a = np.random.randint(0, self.num_actions)
        if return_ensemble_info:
            return a, action_q_val
        return a

    def save(self, save_dir):
        if self.replay_buffers_loaded[self.action_leader()] is True:
            warnings.warn("Replay buffer is loaded. This may not be intended. Please save replay buffer separately")
        
        self.value_ensemble.save(save_dir)

    def load(self, load_path):
        self.value_ensemble.load(load_path)

        self.replay_buffers_loaded = [False]*self.num_modules
        
        