from pfrl.utils.batch_states import batch_states
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from pfrl.agents import PPO
from pfrl.agents.ppo import _yield_minibatches
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
)
from pfrl.utils.batch_states import batch_states
import torch
import pfrl
import gin
import os

class MaskablePPO(PPO):
    
    def __init__(self, 
                 model, 
                 action_dim,
                 optimizer, 
                 obs_normalizer=None, 
                 gpu=None, 
                 gamma=0.99, 
                 lambd=0.95, 
                 phi=lambda x: x, 
                 value_func_coef=1, 
                 entropy_coef=0.01, 
                 update_interval=2048, 
                 minibatch_size=64, 
                 epochs=10, 
                 clip_eps=0.2, 
                 clip_eps_vf=None, 
                 standardize_advantages=True, 
                 batch_states=batch_states, 
                 recurrent=False, 
                 max_recurrent_sequence_len=None, 
                 act_deterministically=False, 
                 max_grad_norm=None, 
                 value_stats_window=1000, 
                 entropy_stats_window=1000, 
                 value_loss_stats_window=100, 
                 policy_loss_stats_window=100):
        super().__init__(model, 
                         optimizer, 
                         obs_normalizer, 
                         gpu, 
                         gamma, 
                         lambd, 
                         phi, 
                         value_func_coef, 
                         entropy_coef, 
                         update_interval, 
                         minibatch_size, 
                         epochs, 
                         clip_eps, 
                         clip_eps_vf, 
                         standardize_advantages, 
                         batch_states, 
                         recurrent, 
                         max_recurrent_sequence_len, 
                         act_deterministically, 
                         max_grad_norm, 
                         value_stats_window, 
                         entropy_stats_window, 
                         value_loss_stats_window, 
                         policy_loss_stats_window)
        
        self.action_dist = MaskableCategoricalDistribution(action_dim=action_dim)
    
    def batch_observe(self, 
                      batch_obs, 
                      batch_reward, 
                      batch_done, 
                      batch_reset,
                      batch_mask):
        if self.training:
            self._batch_observe_train(batch_obs,
                                      batch_reward,
                                      batch_done,
                                      batch_reset,
                                      batch_mask)
        else:
            self._batch_observe_eval(batch_obs,
                                     batch_reward,
                                     batch_done,
                                     batch_reset)
    
    def _batch_observe_train(self, 
                             batch_obs, 
                             batch_reward, 
                             batch_done, 
                             batch_reset,
                             batch_mask):
        assert self.training
        
        for i, (state, action, reward, next_state, done, reset, mask) in enumerate(
            zip(
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset,
                batch_mask
            )
        ):
            if state is not None:
                assert action is not None
                transition = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0,
                    "mask": mask
                }
                if self.recurrent:
                    transition["recurrent_state"] = get_recurrent_state_at(
                        self.train_prev_recurrent_states, i, detach=True
                    )
                    transition["next_recurrent_state"] = get_recurrent_state_at(
                        self.train_recurrent_states, i, detach=True
                    )
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None
        
        self.train_prev_recurrent_states = None
        
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.train_recurrent_states = mask_recurrent_state_at(
                    self.train_recurrent_states, indices_that_ended
                )

        self._update_if_dataset_is_ready()
    
    def batch_act(self, batch_obs, batch_mask=None):
        if self.training:
            return self._batch_act_train(batch_obs, batch_mask)
        else:
            return self._batch_act_eval(batch_obs, batch_mask)
    
    def _batch_act_eval(self, batch_obs, batch_mask):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)
        
        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)
        
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                (action_logits, _), self.test_recurrent_states = one_step_forward(
                    self.model, b_state, self.test_recurrent_states
                )
            else:
                action_logits, _ = self.model(b_state)
            distribution = self.action_dist.proba_distribution(action_logits)
            if batch_mask is not None:
                distribution.apply_masking(batch_mask)
            action = distribution.get_actions(deterministic=self.act_deterministically).cpu().numpy()
            
        return action
            
    
    def _batch_act_train(self, batch_obs, batch_mask):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)
        
        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)
        
        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs
        
        # action distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (
                    (action_logits, batch_value),
                    self.train_recurrent_states,
                ) = one_step_forward(
                    self.model, b_state, self.train_prev_recurrent_states
                )
            else:
                action_logits, batch_value = self.model(b_state)
            distribution = self.action_dist.proba_distribution(action_logits)
            if batch_mask is not None:
                distribution.apply_masking(batch_mask)
            batch_action = distribution.get_actions().cpu().numpy()
            self.entropy_record.extend(distribution.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())
        
        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)
        
        return batch_action
    
    def _update(self, dataset):
        """Update both the policy and the value function."""

        device = self.device

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        assert "state" in dataset[0]
        assert "v_teacher" in dataset[0]

        if self.standardize_advantages:
            all_advs = torch.tensor([b["adv"] for b in dataset], device=device)
            std_advs, mean_advs = torch.std_mean(all_advs, unbiased=False)

        for batch in _yield_minibatches(
            dataset, minibatch_size=self.minibatch_size, num_epochs=self.epochs
        ):
            states = self.batch_states(
                [b["state"] for b in batch], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            masks = torch.tensor([b["mask"] for b in batch], device=device)
            actions = torch.tensor([b["action"] for b in batch], device=device)
            logits, vs_pred = self.model(states)
            distribs = self.action_dist.proba_distribution(logits)
            distribs.apply_masking(masks)

            advs = torch.tensor(
                [b["adv"] for b in batch], dtype=torch.float32, device=device
            )
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = torch.tensor(
                [b["log_prob"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            vs_pred_old = torch.tensor(
                [b["v_pred"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            vs_teacher = torch.tensor(
                [b["v_teacher"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.model.zero_grad()
            loss = self._lossfun(
                distribs.entropy(),
                vs_pred,
                distribs.log_prob(actions),
                vs_pred_old=vs_pred_old,
                log_probs_old=log_probs_old,
                advs=advs,
                vs_teacher=vs_teacher,
            )
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self.n_updates += 1
    
    def _update_once_recurrent(self, episodes, mean_advs, std_advs):
        assert std_advs is None or std_advs > 0

        device = self.device

        # Sort desc by lengths so that pack_sequence does not change the order
        episodes = sorted(episodes, key=len, reverse=True)

        flat_transitions = flatten_sequences_time_first(episodes)

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in episodes:
            states = self.batch_states(
                [transition["state"] for transition in ep],
                self.device,
                self.phi,
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_masks = torch.tensor(
            [transition["mask"] for transition in flat_transitions],
            device=device
        )
        
        flat_actions = torch.tensor(
            [transition["action"] for transition in flat_transitions],
            device=device,
        )
        flat_advs = torch.tensor(
            [transition["adv"] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        if self.standardize_advantages:
            flat_advs = (flat_advs - mean_advs) / (std_advs + 1e-8)
        flat_log_probs_old = torch.tensor(
            [transition["log_prob"] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        flat_vs_pred_old = torch.tensor(
            [[transition["v_pred"]] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        flat_vs_teacher = torch.tensor(
            [[transition["v_teacher"]] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            rs = concatenate_recurrent_states(
                [ep[0]["recurrent_state"] for ep in episodes]
            )

        (flat_logits, flat_vs_pred), _ = pack_and_forward(self.model, seqs_states, rs)
        flat_distribs = self.action_dist.proba_distribution(flat_logits)
        flat_distribs.apply_masking(flat_masks)
        flat_log_probs = flat_distribs.log_prob(flat_actions)
        flat_entropy = flat_distribs.entropy()

        self.model.zero_grad()
        loss = self._lossfun(
            entropy=flat_entropy,
            vs_pred=flat_vs_pred,
            log_probs=flat_log_probs,
            vs_pred_old=flat_vs_pred_old,
            log_probs_old=flat_log_probs_old,
            advs=flat_advs,
            vs_teacher=flat_vs_teacher,
        )
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.n_updates += 1

@gin.configurable
class MaskablePPOAgent():
    def __init__(self,
                 use_gpu,
                 learning_rate,
                 state_shape,
                 phi,
                 num_actions,
                 policy=None,
                 value_function=None,
                 model=None,
                 epochs_per_update=10,
                 clip_eps_vf=None,
                 entropy_coef=0,
                 standardize_advantages=True,
                 gamma=0.9,
                 lambd=0.97,
                 minibatch_size=64,
                 update_interval=2048) -> None:
        self.returns_vals = False
        if model is None:
            assert policy is not None
            assert value_function is not None
            model = pfrl.nn.Branched(policy, value_function)
            self.returns_vals = True
        opt = torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               eps=1e-5)
        
        obs_normalizer = pfrl.nn.EmpiricalNormalization(state_shape,
                                                        clip_threshold=5)
        
        self.agent = MaskablePPO(model,
                                 num_actions,
                                 opt,
                                 obs_normalizer=obs_normalizer,
                                 gpu=use_gpu,
                                 phi=phi,
                                 entropy_coef=entropy_coef,
                                 update_interval=update_interval,
                                 minibatch_size=minibatch_size,
                                 epochs=epochs_per_update,
                                 clip_eps_vf=clip_eps_vf,
                                 standardize_advantages=standardize_advantages,
                                 gamma=gamma,
                                 lambd=lambd)
        
        self.num_actions = num_actions
        
        self.step = 0
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.agent.save(dir)
    
    def load(self, dir):
        self.agent.load(dir)
    
    def act(self, obs, mask):
        self.step += 1
        out = self.agent.batch_act([obs], [mask])
        out = torch.from_numpy(out)
        
        if not self.returns_vals:
            return out, None
        
        if self.agent.training:
            action = torch.argmax(out)
        
        return action, out
    
    def q_function(self, obs):
        return self.agent.batch_act(obs)
    
    def observe(self, obs, mask, reward, done, reset):
        obs = obs.unsqueeze(0)
        mask = [mask]
        reward = [reward]
        done = [done]
        reset = [reset]
        
        return self.agent.batch_observe(obs,
                                        reward,
                                        done,
                                        reset,
                                        mask)





