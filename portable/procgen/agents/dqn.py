import os
import dill
import lzma

import numpy as np
import torch
import torch.nn as nn
import pfrl
from pfrl import agents, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers
from pfrl.utils import evaluating
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN

from portable.procgen.models.small_cnn import SmallCNN
from portable.procgen.agents.abstract_agent import Agent


class DoubleDQN(agents.DoubleDQN, Agent):
    """
    a customr DQN
    such that the observe() method takes in actions as well

    need it so that actions don't need to come from act()

    this class doesn't support recurrent DQNs
    """
    def observe(
        self,
        obs,
        action,
        reward,
        next_obs,
        termimal,
    ):
        return self.batch_observe([obs], [action], [reward], [next_obs], [termimal])

    def batch_observe(
        self,
        batch_obs,
        batch_action,
        batch_reward,
        batch_next_obs,
        batch_terminal,
    ) -> None:
        if self.training:
            return self._batch_observe_train(
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal
            )

    def _batch_observe_train(
        self,
        batch_obs,
        batch_action,
        batch_reward,
        batch_next_obs,
        batch_terminal,
    ) -> None:

        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            # Add a transition to the replay buffer
            transition = {
                "state": batch_obs[i],
                "action": batch_action[i],
                "reward": batch_reward[i],
                "next_state": batch_next_obs[i],
                "next_action": None,
                "is_state_terminal": batch_terminal[i],
            }
            self.replay_buffer.append(env_id=i, **transition)
            if batch_terminal[i]:
                self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)
    
    def extract_features(self, state):
        with torch.no_grad(), evaluating(self.model):
            cnn = self.model[0]
            batch_xs = self.batch_states([state], self.device, self.phi)
            return cnn(batch_xs).cpu().numpy().flatten()
    
    def save(self, save_dir):
        path = os.path.join(save_dir, "agent.pkl")
        with lzma.open(path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, load_path):
        with lzma.open(load_path, 'rb') as f:
            agent = dill.load(f)
        return agent


class SingleSharedBias(nn.Module):
    """
    Single shared bias used in the Double DQN paper.
    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.
    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

    def __call__(self, x):
        return x + self.bias.expand_as(x)


def parse_arch(arch, n_actions):
    """
    return the architecture of the Q agent
    """
    if arch == "custom":
        return nn.Sequential(
            SmallCNN(),
            init_chainer_default(nn.Linear(24, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    elif arch == "nature":
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "doubledqn":
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    elif arch == "nips":
        return nn.Sequential(
            pnn.SmallAtariCNN(),
            init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "dueling":
        return DuelingDQN(n_actions)
    else:
        raise NotImplementedError(f"Not supported architecture: {arch}")


def parse_agent(agent):
    """
    which DQN agent to use
    """
    return {"DQN": agents.DQN, "DoubleDQN": DoubleDQN, "PAL": agents.PAL}[agent]


def make_dqn_agent(q_agent_type,
                    arch,
                    n_actions,
                    phi,
                    lr=2.5e-4,
                    noisy_net_sigma=None,
                    buffer_length=10 ** 6,
                    final_epsilon=0.01,
                    final_exploration_frames=10 ** 6,
                    use_gpu=0,
                    replay_start_size=5 * 10 **4,
                    target_update_interval=3 * 10**4,
                    update_interval=4,
                    ):
    """
    given an architecture and an specific dqn 
    return the agent

    args:
        q_agent_type: choices=["DQN", "DoubleDQN", "PAL"]
        arch: choices=["nature", "nips", "dueling", "doubledqn"]
        n_actions: number of actions
        phi: the function to apply to the state, usually a feature extractor
        final_epsilon: Final value of epsilon during training
        final_exploration_frames: Timesteps after which we stop annealing exploration rate
        replay_start_size: Minimum replay buffer size before performing gradient updates.
        target_update_interval: Frequency (in timesteps) at which the target network is updated
        update_interval: Frequency (in timesteps) of network updates.
    """
    # q function 
    q_func = parse_arch(arch, n_actions)

    # explorer
    if noisy_net_sigma is not None:
        pnn.to_factorized_noisy(q_func, sigma_scale=noisy_net_sigma)
        # turn off explorer
        explorer = explorers.Greedy()
    else:
        # deafult option
        explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            final_epsilon,
            final_exploration_frames,
            lambda: np.random.randint(n_actions),
        )
    
    # optimizer
    # Use the Nature paper's hyperparameters
    opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
        q_func.parameters(),
        lr=lr,
        alpha=0.95,
        momentum=0.0,
        eps=1e-2,
        centered=True,
    )

    # replay_buffer
    rbuf = replay_buffers.ReplayBuffer(buffer_length)     

    Agent = parse_agent(q_agent_type)
    agent = Agent(
        q_func,
        opt,
        rbuf,
        gpu=use_gpu,  # 0 or -1
        gamma=0.99,
        explorer=explorer,
        replay_start_size=replay_start_size,
        target_update_interval=target_update_interval,
        clip_delta=True,
        update_interval=update_interval,
        batch_accumulator="sum",
        phi=phi,
    )

    return agent
