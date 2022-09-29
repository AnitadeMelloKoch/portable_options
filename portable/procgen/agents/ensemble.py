import os
import lzma
import dill
import itertools
from contextlib import contextmanager

import torch
import numpy as np
from pfrl import replay_buffers
from pfrl.replay_buffer import batch_experiences
from pfrl.utils.batch_states import batch_states

from portable.procgen.ensemble.criterion import batched_L_divergence
from portable.procgen.agents.abstract_agent import Agent, evaluating
from portable.procgen.ensemble.aggregate import choose_most_popular, choose_leader, \
    choose_max_sum_qvals, upper_confidence_bound


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
                bandit_exploration_weight=500):
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
        self.embedding_plot_freq = embedding_plot_freq
        self.discount_rate = discount_rate
        self.bandit_exploration_weight = bandit_exploration_weight
        
        # ensemble
        self.attention_model = attention_model.to(self.device)
        self.attention_optimizer = torch.optim.Adam(self.attention_model.parameters(), lr=learning_rate)
        self.learners = learners
        self.optimizer = torch.optim.Adam(
            list(self.attention_model.parameters()) + list(itertools.chain.from_iterable([list(learner.model.parameters()) for learner in self.learners])),
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

            # for attention model
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

                self.attention_model.train()
                self.optimizer.zero_grad()
                loss.backward()
                for learner in self.learners:
                    if hasattr(learner, 'max_grad_norm') and learner.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(learner.model.parameters(), learner.max_grad_norm)
                self.optimizer.step()
                self.attention_model.eval()
    
    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
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
        self.learner_accumulated_reward[self.action_leader] += batch_reward.clip(0, 1).sum()
        if batch_reset.any() or batch_done.any():
            self.episode_number += np.logical_or(batch_reset, batch_done).sum()
            self._set_action_leader()

    def _batch_observe_eval(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass
    
    def _using_leader(self):
        return self.action_selection_strategy in ['ucb_leader', 'greedy_leader', 'uniform_leader']

    def _attention_embed_obs(self, batch_obs):
        obs = torch.as_tensor(batch_obs.copy(), dtype=torch.float32, device=self.device)
        embedded_obs = self.attention_model(obs)
        embedded_obs = [emb.cpu() for emb in embedded_obs]
        return embedded_obs
    
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
            self.action_leader = upper_confidence_bound(
                values=self.learner_accumulated_reward, 
                t=self.step_number, 
                visitation_count=self.learner_selection_count, 
                c=self.bandit_exploration_weight
            )
            self.learner_selection_count[self.action_leader] += 1

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
            state_embedding_flatten = state_embedding_flatten.view(self.batch_size, self.num_modules, -1)  # (batch_size, num_modules, d)
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

            return div_loss

    def batch_act(self, batch_obs):
        with self.set_evaluating():
            # action selection strategy
            if self.action_selection_strategy == 'vote':
                action_selection_func = choose_most_popular
            elif self.action_selection_strategy in ['ucb_leader', 'greedy_leader', 'uniform_leader']:
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
