import torch
import random
import numpy as np
from pfrl import nn as pnn
from pfrl import replay_buffers
from pfrl import agents, explorers
from pfrl.wrappers import atari_wrappers
from pfrl.q_functions import DistributionalDuelingDQN
from pfrl.utils import batch_states as pfrl_batch_states
from PIL import Image


class Rainbow:
    def __init__(self, n_actions, n_atoms, v_min, v_max, noisy_net_sigma, lr, 
            n_steps, betasteps, replay_start_size, replay_buffer_size, gpu,
            n_obs_channels, use_custom_batch_states=True,
            epsilon_decay_steps=None, final_epsilon=0.1,
            update_interval: int = 4):
        self.n_actions = n_actions
        n_channels = n_obs_channels
        self.use_custom_batch_states = use_custom_batch_states
        self.update_interval = update_interval

        self.q_func = DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max, n_input_channels=n_channels)
        pnn.to_factorized_noisy(self.q_func, sigma_scale=noisy_net_sigma)

        if epsilon_decay_steps in (None, 0) and final_epsilon is not None:
            explorer = explorers.ConstantEpsilonGreedy(
                epsilon=final_epsilon,
                random_action_func=lambda: random.randint(0, n_actions - 1)
            )
        elif epsilon_decay_steps is not None and final_epsilon is not None:
            print(f'Number of epsilon decay steps = {epsilon_decay_steps}')
            explorer = explorers.LinearDecayEpsilonGreedy(
                start_epsilon=1.0,
                end_epsilon=final_epsilon,
                decay_steps=epsilon_decay_steps,
                random_action_func=lambda: random.randint(0, n_actions - 1)
            )
        else:
            explorer = explorers.Greedy()
            print(f'[Rainbow] Using exploration strategy {explorer}')

        opt = torch.optim.Adam(self.q_func.parameters(), lr, eps=1.5e-4)

        self.rbuf = replay_buffers.PrioritizedReplayBuffer(
        replay_buffer_size,
        alpha=0.5, 
        beta0=0.4,
        betasteps=betasteps,
        num_steps=n_steps,
        normalize_by_max="memory"
        )

        self.agent = agents.CategoricalDoubleDQN(
        self.q_func,
        opt,
        self.rbuf,
        gpu=gpu,
        gamma=0.99,
        explorer=explorer,
        minibatch_size=32,
        replay_start_size=replay_start_size,
        target_update_interval=32_000,
        update_interval=update_interval,
        batch_accumulator="mean",
        phi=self.phi,
        batch_states=self.batch_states if use_custom_batch_states else pfrl_batch_states
        )

        self.T = 0
        self.device = torch.device(f"cuda:{gpu}" if gpu > -1 else "cpu")

    def eval(self):
        self.q_func.eval()
    
    def parameters(self):
        return self.q_func.parameters()
    
    @staticmethod
    def batch_states(states, device, phi):
        assert isinstance(states, list), type(states)
        features = np.asarray([phi(s) for s in states])
        return torch.as_tensor(features).to(device)

    @staticmethod
    def phi(x):
        """ Observation pre-processing for convolutional layers. """
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=np.float32) / 255.
        assert x.dtype == torch.float32 and x.max().item() <= 1, f'{x.dtype, x.max()}'
        
        return x

    @staticmethod
    def transform_obs(x):
        if type(x) is torch.Tensor:
            x = x.cpu().numpy()
        if len(x.shape) == 3:
            num_channels, _, _ = x.shape
            num_batches = 1
            remove_batches=True
            x = np.expand_dims(x, 0)
        else:
            num_batches, num_channels, _, _ = x.shape
            remove_batches = False
        frames = np.zeros((num_batches, num_channels, 84, 84))
        for batch in range(num_batches):
            for channel in range(num_channels):
                img = Image.fromarray(x[batch, channel, :, :])
                frames[batch, channel, :, :] = np.asarray(img.resize((84, 84), Image.BILINEAR))
        
        if remove_batches:
            frames = np.squeeze(frames)
        
        return frames

    def act(self, state):
        """ Action selection method at the current state. """
        state = self.transform_obs(state)
        return self.agent.act(state)

    def step(self, state, action, reward, next_state, done, reset):
        """ Learning update based on a given transition from the environment. """
        state = self.transform_obs(state)
        next_state = self.transform_obs(next_state)
        reset = reset['needs_reset'] if isinstance(reset, dict) else reset
        assert isinstance(reset, bool), type(reset)
        self._overwrite_pfrl_state(state, action)
        self.agent.observe(next_state, reward, done, reset)

    def _overwrite_pfrl_state(self, state, action):
        """ Hack the pfrl state so that we can call act() consecutively during an episode before calling step(). """
        state = self.transform_obs(state)
        self.agent.batch_last_obs = [state]
        self.agent.batch_last_action = [action]

    @torch.no_grad()
    def value_function(self, states):
        states = self.transform_obs(states)
        batch_states = self.agent.batch_states(states, self.device, self.phi)
        action_values = self.agent.model(batch_states).q_values
        return action_values.max(dim=1).values

    @torch.no_grad()
    def q_function(self, state):
        state = self.transform_obs(state)
        batch_states = self.agent.batch_states(state, self.device, self.phi)
        return self.agent.model(batch_states).q_values
    
    
    def experience_replay(self, trajectory):
        """ Add trajectory to the replay buffer and perform agent learning updates. """

        for transition in trajectory:
            self.step(*transition)