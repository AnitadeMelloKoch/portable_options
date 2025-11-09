# skills_policy.py
import os
import numpy as np
import torch
from torch import nn
import pfrl
from pfrl.agents import PPO
import gymnasium as gym
import matplotlib.pyplot as plt

from skills import SkillEnv 

def phi(obs):
    x = obs["image"] if isinstance(obs, dict) else obs
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] >= 3:
        if x.max() > 1.1:
            x /= 255.0
        x = x[..., :3]
        x = np.transpose(x, (2, 0, 1))  
    elif x.ndim == 2:
        x = x[None, ...]
    return x

class SimpleCNNPolicyValue(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  
        )
        self.pi = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, num_actions))
        self.v  = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        z = self.backbone(x)
        logits = self.pi(z)
        value = self.v(z)
        return pfrl.policies.SoftmaxCategoricalHead()(logits), value
    

def make_env():
    base = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    return SkillEnv(base, option_reward=1.0, max_skill_horizon=200)

def train(
    total_steps=200_00,     
    lr=3e-4,
    update_interval=1024,    
    minibatch_size=256,
    epochs=5,
    entropy_coef=0.01,
    gamma=0.99,
    lambd=0.95,
    seed=0,
    gpu=-1,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    obs, info = env.reset()
    num_actions = len(env.skills)

    dummy = phi(obs)
    c, h, w = dummy.shape
    model = SimpleCNNPolicyValue(num_actions)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    obs_norm = pfrl.nn.EmpiricalNormalization((c, h, w), clip_threshold=5)

    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_norm,
        gpu=gpu,
        phi=phi,
        entropy_coef=entropy_coef,
        update_interval=update_interval,
        minibatch_size=minibatch_size,
        epochs=epochs,
        clip_eps_vf=None,
        max_grad_norm=0.5,
        standardize_advantages=True,
        gamma=gamma,
        lambd=lambd,
    )

    steps = 0
    ep = 0
    while steps < total_steps:
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        while not done and steps < total_steps:
            a = int(agent.batch_act([obs])[0])
            next_obs, r, done, _ = env.step(a)
            agent.batch_observe([next_obs], [r], [done], [done])
            obs = next_obs
            ep_return += r
            ep_len += 1
            steps += 1
        ep += 1
        if ep % 5 == 0:
            print(f"[ep {ep:04d}] steps={steps} return={ep_return:.2f} len={ep_len}")

    os.makedirs("ckpts", exist_ok=True)
    agent.save("ckpts/ppo_option_manager")
    print("Saved to ckpts/ppo_option_manager")


if __name__ == "__main__":
    train()
   
