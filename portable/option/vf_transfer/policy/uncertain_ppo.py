import os 
import logging 
import gin 
import torch 
import pickle 
import itertools
import numpy as np 
import torch.optim as optim
import torch.nn as nn
logger = logging.getLogger(__name__)
from collections import deque 
from copy import deepcopy 
import torch.nn.functional as F
import pfrl

from portable.option.policy.agents import Agent 
from portable.option.policy.models.ppo import PPO, _compute_explained_variance
from portable.option.vf_transfer.models.uncertain_ppo_model import UncertainPPOModel, VFEnsembleFull

from pfrl.utils.batch_states import batch_states
from pfrl.utils.recurrent import get_recurrent_state_at, mask_recurrent_state_at

def _calc_alpha(vf_task_uncert, 
                vf_meta_uncert,
                k,              # temperature: how sharp the transition is
                z0):            # offset for sigmoid: how much certainty do we need to throw away V meta
    z = torch.log(vf_meta_uncert.clamp_min(1e-6)) - torch.log(vf_task_uncert.clamp_min(1e-6))
    alpha = torch.sigmoid(-k*(z-z0))
    
    return alpha
    

"""
These are all overwritten functions to add the new advantage and meta v function computations
"""

def _add_log_prob_and_value_to_episodes(episodes,
                                        model,
                                        phi,
                                        batch_states,
                                        obs_normalizer,
                                        device,
                                        k,
                                        z0,
                                        meta_vf=None,
                                        meta_vf_ready=False,
                                        meta_vf_weight=1.0,
                                        alpha_out=None):
    dataset = list(itertools.chain.from_iterable(episodes))

    states = batch_states([b["state"] for b in dataset], device, phi)
    next_states = batch_states([b["next_state"] for b in dataset], device, phi)

    if obs_normalizer:
        states = obs_normalizer(states, update=False)
        next_states = obs_normalizer(next_states, update=False)

    with torch.no_grad(), pfrl.utils.evaluating(model):
        distribs, vs_pred = model(states)
        _, next_vs_pred = model(next_states)

        if meta_vf_ready and meta_vf is not None:
            own_unc = model.vf_uncertainty(next_states)
            meta_unc = meta_vf.uncertainty(next_states)
            alpha = _calc_alpha(own_unc,
                                meta_unc,
                                k,
                                z0)
            if alpha_out is not None:
                alpha_out.append((
                    alpha.mean().item(), alpha.min().item(), alpha.max().item(),
                    own_unc.mean().item(), meta_unc.mean().item(),
                ))
            # Clip meta VF predictions to the plausible return range.
            # Per-head loss causes large weight magnitudes that extrapolate wildly
            # on OOD transfer states; without clipping this triggers a divergence loop.
            meta_vs = meta_vf(next_states).clamp(-10.0, 10.0)
            bootstrap_vs = alpha * meta_vs + (1 - alpha) * next_vs_pred
        else:
            bootstrap_vs = next_vs_pred

        actions = torch.tensor([b["action"] for b in dataset], device=device)
        log_probs = distribs.log_prob(actions).cpu().numpy()
        vs_pred = vs_pred.cpu().numpy().ravel()
        bootstrap_vs = bootstrap_vs.cpu().numpy().ravel()

    for transition, log_prob, v_pred, bootstrap in zip(
        dataset, log_probs, vs_pred, bootstrap_vs
    ):
        transition["log_prob"] = log_prob
        transition["v_pred"] = v_pred
        transition["bootstrap"] = float(bootstrap)


def _add_advantage_and_value_target_to_episode(episode, gamma, lambd):
    adv = 0.0
    for transition in reversed(episode):
        td_err = (
            transition["reward"]
            + (gamma*transition["nonterminal"]*transition["bootstrap"])
            - transition["v_pred"]
        )
        adv = td_err + gamma * lambd * adv
        transition["adv"] = adv
        # TD(0) Bellman backup: when alpha=1 this is r + γ*meta_vf(next),
        # the exact fixed point the task VF should learn under HPP.
        # Using GAE (adv + v_pred) here amplifies v_pred errors into the target
        # via the lambda-return sum, causing the observed 2e9 explosion.
        transition["v_teacher"] = (
            transition["reward"]
            + gamma * transition["nonterminal"] * transition["bootstrap"]
        )
        

def _add_advantage_and_value_target_to_episodes(episodes, gamma, lambd):
    for episode in episodes:
        _add_advantage_and_value_target_to_episode(episode, gamma=gamma, lambd=lambd)

def _make_dataset(episodes,
                  model,
                  phi,
                  batch_states,
                  obs_normalizer,
                  gamma,
                  lambd,
                  device,
                  k,
                  z0,
                  meta_vf=None,
                  meta_vf_ready=False,
                  meta_vf_weight=1.0,
                  alpha_out=None):
    _add_log_prob_and_value_to_episodes(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        device=device,
        k=k,
        z0=z0,
        meta_vf=meta_vf,
        meta_vf_ready=meta_vf_ready,
        meta_vf_weight=meta_vf_weight,
        alpha_out=alpha_out,
    )

    _add_advantage_and_value_target_to_episodes(episodes, gamma=gamma, lambd=lambd)

    return list(itertools.chain.from_iterable(episodes))

@gin.configurable
class UncertainPPO(PPO):
    def __init__(self,
                 num_actions,
                 num_vf_heads,
                 learning_rate,
                 meta_vf_model,
                 k=5,
                 z0=0,
                 meta_vf_weight=1,
                 prior_scale=1.0,
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

        self.model = UncertainPPOModel(num_actions=num_actions,
                                       vf_heads=num_vf_heads,
                                       prior_scale=prior_scale)
        
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=learning_rate,
                                     eps=1e-5)
        
        self.meta_vf = meta_vf_model
        self.meta_vf_ready = False
        self.meta_vf_weight = meta_vf_weight
        self.last_mean_alpha = None
        self.last_own_unc = None
        self.last_meta_unc = None
        self.alpha_log = []

        self.k = k
        self.z0 = z0
        
        super().__init__(self.model, 
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
    
    def _batch_observe_train(self, 
                             batch_obs, 
                             batch_reward, 
                             batch_done, 
                             batch_reset):
        assert self.training
        
        for i, (state, action, reward, next_state, done, reset) in enumerate(
            zip(
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset
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
            self.batch_last_action[i] =None

        self.train_prev_recurrent_states = None
        
        if self.recurrent:
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.train_recurrent_states = mask_recurrent_state_at(
                    self.train_recurrent_states, indices_that_ended
                )
        
        maybe_loss = self._update_if_dataset_is_ready()
        return maybe_loss
    
    def _lossfun(self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher):
        prob_ratio = torch.exp(log_probs - log_probs_old)
        loss_policy = -torch.mean(torch.min(
            prob_ratio * advs,
            torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs,
        ))
        self.policy_loss_record.append(float(loss_policy))

        # Per-head VF loss: each head gets a different gradient signal (via its unique prior),
        # so heads independently converge toward V*(x), driving uncertainty to zero on training data.
        head_losses = torch.stack([
            F.mse_loss(hv, vs_teacher) for hv in self.model._last_head_vs
        ])
        loss_value_func = 0.5 * head_losses.mean()
        self.value_loss_record.append(float(loss_value_func))

        loss_entropy = -torch.mean(entropy)
        return loss_policy + self.value_func_coef * loss_value_func + self.entropy_coef * loss_entropy

    def _update_if_dataset_is_ready(self):
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (
                0
                if self.batch_last_episode is None 
                else sum(len(episode) for episode in self.batch_last_episode)
            )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            assert not self.recurrent
            alpha_out = []
            dataset = _make_dataset(
                episodes=self.memory,
                model=self.model,
                phi=self.phi,
                batch_states=self.batch_states,
                obs_normalizer=self.obs_normalizer,
                gamma=self.gamma,
                lambd=self.lambd,
                k=self.k,
                z0=self.z0,
                device=self.device,
                meta_vf=self.meta_vf,
                meta_vf_ready=self.meta_vf_ready,
                meta_vf_weight=self.meta_vf_weight,
                alpha_out=alpha_out,
            )
            if alpha_out:
                self.last_mean_alpha = alpha_out[0][0]
                self.last_own_unc = alpha_out[0][3]
                self.last_meta_unc = alpha_out[0][4]
                self.alpha_log.append(alpha_out[0])
            assert len(dataset) == dataset_size
            maybe_loss = self._update(dataset)
            self.explained_variance = _compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory))
            )
            self.memory=[]
            return maybe_loss

@gin.configurable
class PPOEnsembleVFAgent(Agent):
    def __init__(self,
                 meta_vf: VFEnsembleFull,
                 num_actions,
                 num_vf_heads,
                 gpu_id=-1,
                 phi=lambda x:x):
        super().__init__()
        
        assert isinstance(meta_vf, VFEnsembleFull)
        
        self.meta_vf = meta_vf
        
        self.agent = UncertainPPO(gpu=gpu_id,
                                  phi=phi,
                                  meta_vf_model=meta_vf,
                                  num_actions=num_actions,
                                  num_vf_heads=num_vf_heads)
    
    def save(self, dirname):
        self.agent.save(dirname=dirname)
    
    def load(self, dirname):
        self.agent.load(dirname=dirname)
        
    def observe(self, 
                obs,
                reward,
                terminal):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        obs = obs.unsqueeze(0)
        reward = [reward]
        done = [terminal]
        reset = [terminal]
        
        self.agent.batch_observe(obs, 
                                 reward,
                                 done,
                                 reset)
    
    def act(self, obs):
        out = self.agent.batch_act([obs])
        
        return out
    
    def run_episode(self, env):
        step_rewards = []
        states = []
        infos = []
        done = False
        steps = 0
        
        state, info = env.reset()
        
        states.append(state)
        infos.append(info)
        
        while not done:
            action = self.act(state)
            steps += 1
            next_state, reward, done, info = env.step(action)
            
            self.observe(next_state,
                         reward,
                         done)
            
            state = next_state
            
            step_rewards.append(reward)
            states.append(state)
            infos.append(info)
            
        return step_rewards, states, infos, steps
        
        
    
    def run_until_termination(self, env, term, reset_env=True, state=None):
        step_rewards = []
        states = []
        infos = []
        done = False
        steps = 0
        
        if not reset_env:
            assert state is not None
        
        if reset_env:
            state, info = env.reset()
            infos.append(info)
        
        states.append(state)
        
        while not term(state):
            action = self.act(state)
            steps += 1
            next_state, reward, done, info = env.step(action)
            
            self.observe(next_state,
                         reward,
                         done)
            
            state = next_state
            
            step_rewards.append(reward)
            states.append(state)
            infos.append(info)
            
        return step_rewards, states, infos, steps













