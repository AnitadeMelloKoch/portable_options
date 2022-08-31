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
from portable.option.policy.value_ensemble import ValueEnsemble
from portable.option.policy.aggregate import choose_leader, choose_most_popular, \
    upper_confidence_bound, choose_max_sum_qvals


class EnsembleAgent(Agent):
    """
    an Agent that keeps an ensemble of policies
    an agent needs to support two methods: observe() and act()

    this class currently doesn't support batched observe() and act()
    """
    def __init__(self, 
                device, 
                warmup_steps,
                batch_size,
                phi,
                action_selection_strategy,
                prioritized_replay_anneal_steps,
                buffer_length=100000,
                update_interval=4,
                q_target_update_interval=40,
                embedding_output_size=64, 
                learning_rate=2.5e-4,
                final_epsilon=0.01,
                final_exploration_frames=10**6,
                discount_rate=0.9,
                num_modules=8, 
                num_output_classes=18,
                plot_dir=None,
                embedding_plot_freq=10000,
                verbose=False,):
        # vars
        self.device = device
        self.phi = phi
        self.action_selection_strategy = action_selection_strategy
        print(f"using action selection strategy: {self.action_selection_strategy}")
        if self._using_leader():
            self.action_leader = np.random.choice(num_modules)
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.q_target_update_interval = q_target_update_interval
        self.update_interval = update_interval
        self.num_output_classes = num_output_classes
        self.num_modules = num_modules
        self.step_number = 0
        self.episode_number = 0
        self.learner_accumulated_reward = np.ones(self.num_modules)  # laplace smoothing
        self.learner_selection_count = np.ones(self.num_modules)  # laplace smoothing
        self.update_epochs_per_step = 1
        self.embedding_plot_freq = embedding_plot_freq
        self.discount_rate = discount_rate
        
        # ensemble
        self.value_ensemble = ValueEnsemble(
            device=device,
            embedding_output_size=embedding_output_size,
            learning_rate=learning_rate,
            discount_rate=discount_rate,
            num_modules=num_modules,
            num_output_classes=num_output_classes,
            plot_dir=plot_dir,
            verbose=verbose,
        )

        # explorer
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            final_epsilon,
            final_exploration_frames,
            lambda: np.random.randint(num_output_classes),
        )

        # Prioritized Replay
        # Anneal beta from beta0 to 1 throughout training
        # taken from https://github.com/pfnet/pfrl/blob/master/examples/atari/reproduction/rainbow/train_rainbow.py
        self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            capacity=buffer_length,
            alpha=0.5,  # Exponent of errors to compute probabilities to sample
            beta0=0.4,  # Initial value of beta
            betasteps=prioritized_replay_anneal_steps,  # Steps to anneal beta to 1
            normalize_by_max="memory",  # method to normalize the weight
        )
        self.replay_updater = ReplayUpdater(
            replay_buffer=self.replay_buffer,
            update_func=self.update,
            batchsize=batch_size,
            episodic_update=False,
            episodic_update_len=None,
            n_times_update=1,
            replay_start_size=warmup_steps,
            update_interval=update_interval,
        )
    
    def _using_leader(self):
        return self.action_selection_strategy in ['ucb_leader', 'greedy_leader', 'uniform_leader']
    
    def observe(self, obs, action, reward, next_obs, terminal):
        """
        store the experience tuple into the replay buffer
        and update the agent if necessary
        """
        self.step_number += 1

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
            self.replay_buffer.append(**transition)
            if terminal:
                self.replay_buffer.stop_current_episode()

            self.replay_updater.update_if_necessary(self.step_number)
            if self._using_leader():
                self.learner_accumulated_reward[self.action_leader] += reward
            
        # new episode
        if terminal:
            self.episode_number += 1
            if self._using_leader():
                self._set_action_leader()

    def _set_action_leader(self):
        """choose which learner in the ensemble gets to lead the action selection process"""
        if self.action_selection_strategy == 'uniform_leader':
            # choose a random leader
            self.action_leader = np.random.choice(self.num_modules)
        elif self.action_selection_strategy == 'greedy_leader':
            # greedily choose the leader based on the cumulated reward
            normalized_reward = self.learner_accumulated_reward - self.learner_accumulated_reward.max()
            probability = np.exp(normalized_reward) / np.exp(normalized_reward).sum()  # softmax
            self.action_leader = np.random.choice(self.num_modules, p=probability)
        elif self.action_selection_strategy == 'ucb_leader':
            # choose a leader based on the Upper Condfience Bound algorithm 
            self.action_leader = upper_confidence_bound(values=self.learner_accumulated_reward, t=self.step_number, visitation_count=self.learner_selection_count, c=100)
            self.learner_selection_count[self.action_leader] += 1

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
        if self.training:
            has_weight = "weight" in experiences[0][0]
            exp_batch = batch_experiences(
                experiences,
                device=self.device,
                phi=self.phi,
                gamma=self.discount_rate,
                batch_states=batch_states,
            )
            # get weights for prioritized experience replay
            if has_weight:
                exp_batch["weights"] = torch.tensor(
                    [elem[0]["weight"] for elem in experiences],
                    device=self.device,
                    dtype=torch.float32,
                )
                if errors_out is None:
                    errors_out = []
            # actual update
            update_target_net =  self.step_number % self.q_target_update_interval == 0
            self.value_ensemble.train(exp_batch, errors_out, update_target_net, plot_embedding=(self.step_number % self.embedding_plot_freq == 0))
            # update prioritiy
            if has_weight:
                assert isinstance(self.replay_buffer, replay_buffers.PrioritizedReplayBuffer)
                self.replay_buffer.update_errors(errors_out)

    def act(self, obs, return_ensemble_info=False):
        """
        epsilon-greedy policy
        args:
            obs (object): Observation from the environment.
            return_ensemble_info (bool): when set to true, this function returns
                (action_selected, actions_selected_by_each_learner, q_values_of_each_actions_selected)
        """
        obs = batch_states([obs], self.device, self.phi)
        actions, action_q_vals, all_q_vals = self.value_ensemble.predict_actions(obs, return_q_values=True)
        # action selection strategy
        if self.action_selection_strategy == 'vote':
            action_selection_func = lambda a, qvals: choose_most_popular(a)
        elif self.action_selection_strategy in ['ucb_leader', 'greedy_leader', 'uniform_leader']:
            action_selection_func = lambda a, qvals: choose_leader(a, leader=self.action_leader)
        elif self.action_selection_strategy == 'add_qvals':
            action_selection_func = lambda a, qvals: choose_max_sum_qvals(qvals)
        else:
            raise NotImplementedError("action selection strat not supported")
        # epsilon-greedy
        if self.training:
            a = self.explorer.select_action(
                self.step_number,
                greedy_action_func=lambda: action_selection_func(actions, all_q_vals),
            )
        else:
            a = action_selection_func(actions, all_q_vals)
        if return_ensemble_info:
            return a, actions, action_q_vals
        return a

    def save(self, save_dir):
        path = os.path.join(save_dir, "agent.pkl")
        with lzma.open(path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, load_path, reset=False, plot_dir=None):
        with lzma.open(load_path, 'rb') as f:
            agent = dill.load(f)
        # hack to change the plot_dir of the agent
        agent.value_ensemble.embedding.plot_dir = plot_dir
        # reset defaults
        if reset:
            agent.learner_accumulated_reward = np.ones_like(agent.learner_accumulated_reward)
            agent.learner_selection_count = np.ones_like(agent.learner_selection_count)
        return agent
