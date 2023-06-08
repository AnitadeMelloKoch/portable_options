import os
import lzma
import dill
import itertools
from collections import deque
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
from pfrl import replay_buffers
from pfrl.replay_buffer import batch_experiences
from pfrl.utils.batch_states import batch_states

from portable.policy import logger
from portable.policy.ensemble.criterion import batched_L_divergence
from portable.policy.agents.abstract_agent import Agent, evaluating
from portable.policy.ensemble.aggregate import choose_most_popular, choose_leader, \
    choose_max_sum_qvals, upper_confidence_bound, exp3_bandit_algorithm, \
    upper_confidence_bound_agent_57, upper_confidence_bound_with_window_size, \
    upper_confidence_bound_with_gestation



class EnsembleAgent(Agent):
    """
    an Agent that keeps an ensemble of policies
    an agent needs to support two methods: observe() and act()
    """
    def __init__(self, 
                attention_model,
                learning_rate,
                learners,
                device, 
                warmup_steps,
                batch_size,
                action_selection_strategy,
                phi=lambda x: x,
                buffer_length=100000,
                update_interval=4,
                discount_rate=0.9,
                num_modules=8, 
                embedding_plot_freq=10000,
                bandit_exploration_weight=500,
                fix_attention_mask=False,
                use_feature_learner=True,
                saving_dir=None,):
        # vars
        self.device = device
        self.batch_size = batch_size
        self.phi = phi
        self.action_selection_strategy = action_selection_strategy
        print(f"using action selection strategy: {self.action_selection_strategy}")
        if self._using_leader():
            self.action_leader = np.random.choice(num_modules)
        self.num_modules = num_modules
        self.step_number = 0
        self.episode_number = 0
        self.n_updates = 0
        if self._using_leader():
            self.learner_accumulated_reward = np.ones(self.num_modules)  # laplace smoothing
            self.learner_selection_count = np.ones(self.num_modules)  # laplace smoothing
        if self.action_selection_strategy == "ucb_window_size":
            self.ucb_window_size = 90  # from the agent 57 paper
            self.learner_accumulated_reward_queue = [deque(maxlen=self.ucb_window_size) for _ in range(self.num_modules)]
            self.learner_selection_count_queue = [deque(maxlen=self.ucb_window_size) for _ in range(self.num_modules)]
        if self.action_selection_strategy == 'ucb_gestation':
            self.gestation_period = 1_000_000
            self.gestation_bandit_reset = False
        self.embedding_plot_freq = embedding_plot_freq
        self.discount_rate = discount_rate
        self.bandit_exploration_weight = bandit_exploration_weight
        self.fix_attention_mask = fix_attention_mask
        self.use_feature_learner = use_feature_learner
        self.saving_dir = saving_dir
        if self.saving_dir is not None and self._using_leader():
            self.logger = logger.configure(dir=self.saving_dir, format_strs=['csv'], log_suffix="_bandit_stats")
            self.loss_logger = logger.configure(dir=self.saving_dir, format_strs=['csv', 'stdout'], log_suffix="_loss_stats")
        
        # ensemble
        self.attention_model = attention_model.to(self.device)
        self.attention_optimizer = torch.optim.Adam(self.attention_model.parameters(), lr=learning_rate)
        self.learners = learners
        learnable_params = list(itertools.chain.from_iterable([list(learner.model.parameters()) for learner in self.learners]))
        if not self.fix_attention_mask:
            learnable_params = list(self.attention_model.parameters()) + learnable_params
        self.optimizer = torch.optim.Adam(
            learnable_params,
            lr=learning_rate
        )
        self.num_learners = len(learners)

        self.replay_buffer = replay_buffers.ReplayBuffer(capacity=buffer_length)
        # self.replay_updater = ReplayUpdater(
        #     replay_buffer=self.replay_buffer,
        #     update_func=self.update_attention,
        #     batchsize=batch_size,
        #     episodic_update=False,
        #     episodic_update_len=None,
        #     n_times_update=1,
        #     replay_start_size=warmup_steps,
        #     update_interval=update_interval,
        # )
    
    @contextmanager
    def set_evaluating(self):
        """set the evaluation mode of the learners to be the same as this agent"""
        istrain = self.training
        original_status = [learner.training for learner in self.learners]
        try:
            for learner in self.learners:
                learner.training = istrain
            yield
        finally:
            for learner, status in zip(self.learners, original_status):
                learner.training = status

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        with self.set_evaluating():

            # learners
            embedded_obs = self._attention_embed_obs(batch_obs)
            losses = []
            for i, learner in enumerate(self.learners):
                maybe_loss = learner.batch_observe(embedded_obs[i], batch_reward, batch_done, batch_reset)
                losses.append(maybe_loss)
            assert np.sum([loss is None for loss in losses]) == 0 or np.sum([loss is None for loss in losses]) == self.num_learners

            # add experience to buffer and update action leader
            if self.training:
                self._batch_observe_train(
                    batch_obs, batch_reward, batch_done, batch_reset
                )
            else:
                self._batch_observe_eval(
                    batch_obs, batch_reward, batch_done, batch_reset
                )

            # actual update 
            if np.sum([loss is not None for loss in losses]) == self.num_learners:
                learner_loss = torch.stack(losses).sum()
                div_loss = self.update_attention(experiences=self.replay_buffer.sample(self.batch_size), compute_loss_only=True)
                loss = learner_loss + div_loss
                
                # log the stats
                self.loss_logger.logkv("step_number", self.step_number)
                self.loss_logger.logkv('episode_number', self.episode_number)
                self.loss_logger.logkv("learner_loss", learner_loss.item())
                self.loss_logger.logkv("div_loss", div_loss.item())
                self.loss_logger.dumpkvs()

                if not self.fix_attention_mask:
                    self.attention_model.train()
                self.optimizer.zero_grad()
                loss.backward()
                for learner in self.learners:
                    if hasattr(learner, 'max_grad_norm') and learner.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(learner.model.parameters(), learner.max_grad_norm)
                self.optimizer.step()
                self.attention_model.eval()
    
    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        """
        currently, this just adds experience to replay buffer and sets action leader as neccessary
        the actual update is done in self.update_attention()
        """
        for i in range(len(batch_obs)):
            self.step_number += 1

            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)

            # self.replay_updater.update_if_necessary(self.step_number)

        # action leader
        if batch_reset.any() or batch_done.any():
            self.episode_number += np.logical_or(batch_reset, batch_done).sum()
            self._set_action_leader(batch_reward.clip(0, 1).sum())
            self._update_learner_stats(batch_reward.clip(0, 1).sum())  # assume non-zero reward only at episode end
            if self.episode_number % 20 == 0:
                self._log_bandit_stats()

    def _batch_observe_eval(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass
    
    def _using_leader(self):
        return self.action_selection_strategy in ['exp3_leader', 'ucb_leader', 'greedy_leader', 
            'uniform_leader', 'ucb_57', 'ucb_window_size', 'ucb_gestation']

    def _update_learner_stats(self, reward):
        def safe_mean(x):
            return np.mean(x) if len(x) > 0 else 0
        
        # if need to reset bandit counts
        if self.action_selection_strategy == "ucb_gestation":
            if self.step_number > self.gestation_period and not self.gestation_bandit_reset:
                self.learner_accumulated_reward = np.ones_like(self.learner_accumulated_reward)
                self.learner_selection_count = np.ones_like(self.learner_selection_count)
                self.gestation_bandit_reset = True
        
        if self.action_selection_strategy == "ucb_window_size":
            # udpate queue
            for i in range(self.num_learners):
                if i == self.action_leader:
                    self.learner_accumulated_reward_queue[i].append(reward)
                    self.learner_selection_count_queue[i].append(1)
                else:
                    self.learner_accumulated_reward_queue[i].append(0)
                    self.learner_selection_count_queue[i].append(0)
            # update the stats
            for i in range(self.num_learners):
                self.learner_accumulated_reward[i] = safe_mean(self.learner_accumulated_reward_queue[i])
                self.learner_selection_count[i] = np.sum(self.learner_selection_count_queue[i])
        else:
            self.learner_accumulated_reward[self.action_leader] += np.clip(reward, a_min=None, a_max=1)
            self.learner_selection_count[self.action_leader] += 1

    def _attention_embed_obs(self, batch_obs):
        obs = torch.as_tensor(batch_obs.copy(), dtype=torch.float32, device=self.device)
        embedded_obs = self.attention_model(obs)
        embedded_obs = [emb.cpu() for emb in embedded_obs]
        if not self.use_feature_learner:
            embedded_obs = embedded_obs * self.num_learners  # fead same feature to all learners
        return embedded_obs
    
    def observe(self, obs, action, reward, next_obs, terminal):
        """
        DEPRECATED
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
                self._set_action_leader(reward)
                self._log_bandit_stats()
    
    def _log_bandit_stats(self):
        self.logger.logkv('episode_number', self.episode_number)
        self.logger.logkv('time_step', self.step_number)
        self.logger.logkv('action_leader', self.action_leader)
        self.logger.logkv('num_learners', self.num_learners)
        for i in range(self.num_modules):
            self.logger.logkv(f'learner_{i}_reward', self.learner_accumulated_reward[i])
            self.logger.logkv(f'learner_{i}_selection_count', self.learner_selection_count[i])
        self.logger.dumpkvs()

    def _set_action_leader(self, reward):
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
            self.action_leader = upper_confidence_bound(
                values=self.learner_accumulated_reward, 
                t=self.step_number, 
                visitation_count=self.learner_selection_count, 
                c=self.bandit_exploration_weight
            )
        elif self.action_selection_strategy == 'ucb_gestation':
            self.action_leader = upper_confidence_bound_with_gestation(
                values=self.learner_accumulated_reward,
                t=self.step_number,
                visitation_count=self.learner_selection_count,
                gestation_period=self.gestation_period,
                c=self.bandit_exploration_weight,
            )
        elif self.action_selection_strategy == 'ucb_57':
            self.action_leader = upper_confidence_bound_agent_57(
                mean_rewards=self.learner_accumulated_reward / self.learner_selection_count,
                t=self.step_number,
                visitation_count=self.learner_selection_count,
                beta=self.bandit_exploration_weight,
            )
        elif self.action_selection_strategy == 'ucb_window_size':
            self.action_leader = upper_confidence_bound_with_window_size(
                mean_rewards=self.learner_accumulated_reward,
                t=self.step_number,
                visitation_count=self.learner_selection_count,
                beta=self.bandit_exploration_weight,
                epsilon=0.1,
            )
        elif self.action_selection_strategy == 'exp3_leader':
            # choose a leader based on the EXP3 algorithm
            self.action_leader = exp3_bandit_algorithm(
                reward=reward,
                num_arms=self.num_modules,
                gamma=0.1,  # exploration parameter, in (0, 1]
            )

    def update_attention(self, experiences, compute_loss_only=False, errors_out=None):
        """
        update the attention model
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

            # compute loss
            self.attention_model.train()
            batch_states = exp_batch["state"]
            state_embeddings = self.attention_model(batch_states, plot=(self.n_updates % self.embedding_plot_freq == 0))  # num_modules x (batch_size, C, H, W)
            state_embedding_flatten = torch.cat([embedding.unsqueeze(1) for embedding in state_embeddings], dim=1)  # (batch_size, num_modules, C, H, W)
            state_embedding_flatten = state_embedding_flatten.view(self.batch_size, len(self.attention_model.attention_modules), -1)  # (batch_size, num_modules, d)
            div_loss = batched_L_divergence(state_embedding_flatten)

            if not compute_loss_only:
                self.attention_optimizer.zero_grad()
                div_loss.backward()
                self.attention_optimizer.step()
            self.attention_model.eval()

            # update prioritiy
            if has_weight:
                assert isinstance(self.replay_buffer, replay_buffers.PrioritizedReplayBuffer)
                self.replay_buffer.update_errors(errors_out)
            
            self.n_updates += 1

            if self.fix_attention_mask:
                return 0

            return div_loss

    def batch_act(self, batch_obs):
        with self.set_evaluating():
            # action selection strategy
            if self.action_selection_strategy == 'vote':
                action_selection_func = choose_most_popular
            elif self._using_leader():
                action_selection_func = lambda a: choose_leader(a, leader=self.action_leader)
            else:
                raise NotImplementedError("action selection strat not supported")

            # learners choose actions
            embedded_obs = self._attention_embed_obs(batch_obs)
            batch_actions = [
                self.learners[i].batch_act(embedded_obs[i])
                for i in range(self.num_learners)
            ]
            batch_action = action_selection_func(batch_actions)

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)

            return batch_action

    def act(self, obs, return_ensemble_info=False):
        """
        epsilon-greedy policy
        args:
            obs (object): Observation from the environment.
            return_ensemble_info (bool): when set to true, this function returns
                (action_selected, actions_selected_by_each_learner, q_values_of_each_actions_selected)
        """
        with torch.no_grad(), evaluating(self):
            obs = batch_states([obs], self.device, self.phi)
            actions, action_q_vals, all_q_vals = self.value_ensemble.predict_actions(obs, return_q_values=True)
            actions, action_q_vals, all_q_vals = actions[0], action_q_vals[0], all_q_vals[0]  # get rid of batch dimension
        # action selection strategy
        if self.action_selection_strategy == 'vote':
            action_selection_func = lambda a, qvals: choose_most_popular(a)
        elif self.action_selection_strategy in ['ucb_leader', 'greedy_leader', 'uniform_leader', 'exp3_leader']:
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
    
    def get_statistics(self):
        return []

    def save(self, save_dir):
        # save the bandit counts
        if self._using_leader():
            np.savetxt(os.path.join(save_dir, 'learner_selection_count.txt'), self.learner_selection_count)
            np.savetxt(os.path.join(save_dir, 'learner_accumulated_reward.txt'), self.learner_accumulated_reward)
        # save agent
        path = os.path.join(save_dir, "agent.pkl")
        with lzma.open(path, 'wb') as f:
            dill.dump(self, f)
    
    def _reset_learner_stats(self):
        # note that this does NOT reset self.step_number
        assert self._using_leader()
        self.learner_selection_count = np.ones_like(self.learner_selection_count)
        self.learner_accumulated_reward = np.ones_like(self.learner_accumulated_reward)

    def _reset_step_number(self):
        self.step_number = 0
    
    def reset(self):
        self._reset_step_number()
        self._reset_learner_stats()

    @classmethod
    def load(cls, load_path, reset=False, plot_dir=None):
        with lzma.open(load_path, 'rb') as f:
            agent = dill.load(f)
        # hack to change the plot_dir of the agent
        agent.value_ensemble.embedding.plot_dir = plot_dir
        # reset if needed
        if reset:
            agent.reset()
        return agent
    
    def load_attention_mask(self, load_dir):
        """
        load attention mask from a experiment saving dir
        learner_selection_count.txt: n numbers that tell us which learner is selected how many times
        agent.pkl: saved agent
        """
        # find the portable feature
        with open(os.path.join(load_dir, 'learner_selection_count.txt'), 'r') as f:
            learner_selection_count = np.loadtxt(f)
        portable_feature = np.argmax(learner_selection_count)

        # load the saved attention model 
        with lzma.open(os.path.join(load_dir, 'agent.pkl'), 'rb') as f:
            agent = dill.load(f)
        saved_attention_model = agent.attention_model

        # make all attention the portable one
        portable_attention_mask = saved_attention_model.attention_modules[portable_feature]
        attention_modules = nn.ModuleList(
            [
                portable_attention_mask for _ in range(len(saved_attention_model.attention_modules))
            ]
        )
        
        # load the attention model
        self.attention_model.attention_modules = attention_modules
