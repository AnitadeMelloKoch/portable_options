import os
from portable.option.divdis.policy.policy_and_initiation import PolicyWithInitiation
from experiments.sokoban.utils import environment_builder
import numpy as np
from collections import deque
import torch
from portable.option.policy.intrinsic_motivation.tabular_count import TabularCount

def get_unique(states):
    if len(states) < 2:
        return states
    stacked_states = np.stack(states)
    stacked_states = np.unique(stacked_states, axis=0)
    
    array_list = [state for state in stacked_states]
    
    return array_list

def collect_states(seed=0, episodes=500):
    box_1_pos = []
    box_1_neg = []
    box_2_pos = []
    box_2_neg = []
    box_3_pos = []
    box_3_neg = []
    
    save_dir = 'resources/sokoban_images'
    os.makedirs(save_dir, exist_ok=True)
    
    
    
    ave_rewards = deque(maxlen=100)
    env = environment_builder(level_name="PushAndPull-Sokoban-v0", seed=seed)    
    total_steps = 0
    
    intrinsic_bonus = TabularCount(beta=0.01)
    
    policy = PolicyWithInitiation(use_gpu=0,
                                  warmup_steps=1024,
                                  prioritized_replay_anneal_steps=500000,
                                  buffer_length=100000,
                                  update_interval=1,
                                  q_target_update_interval=100,
                                  learning_rate=2.5e-4,
                                  final_epsilon=0.1,
                                  final_exploration_frames=1e5,
                                  batch_size=32,
                                  num_actions=13,
                                  policy_infeature_size=18496,
                                  policy_phi=lambda x: x,
                                  gru_hidden_size=512,
                                  q_hidden_size=128,
                                  )
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        
        obs = torch.from_numpy(obs/255.0).float()
        obs = obs.permute((2,0,1))
        # obs = obs.unsqueeze(0).float()
        rewards = 0
        
        while not done:
            action = policy.act(obs)
            
            next_obs, reward, done, info = env.step(action)
            total_steps += 1
            
            if "in_target_box_locations" in info:
                box_in = info["in_target_box_locations"]
                if len(box_in) == 1:
                    box_1_pos.append(next_obs)
                    box_2_neg.append(next_obs)
                    box_3_neg.append(next_obs)
                elif len(box_in) == 2:
                    box_1_pos.append(next_obs)
                    box_2_pos.append(next_obs)
                    box_3_neg.append(next_obs)
                elif len(box_in) == 3:
                    box_1_pos.append(next_obs)
                    box_2_pos.append(next_obs)
                    box_3_pos.append(next_obs)
            else:
                box_1_neg.append(next_obs)
                box_2_neg.append(next_obs)
                box_3_neg.append(next_obs)
            
            next_obs = torch.from_numpy(next_obs/255.0)
            next_obs = next_obs.permute((2,0,1))
            # next_obs = next_obs.unsqueeze(0).float()
            
            rewards += reward
            
            reward += intrinsic_bonus.get_bonus(tuple(info["state"].flatten()))
            policy.observe(obs,
                           action,
                           reward,
                           next_obs,
                           done)
        
        ave_rewards.append(rewards)
        print(f"Episode {episode} steps {total_steps} ave reward {np.mean(ave_rewards)}")
        
        
        
        box_1_pos = get_unique(box_1_pos)
        box_1_neg = get_unique(box_1_neg)
        
        box_2_pos = get_unique(box_2_pos)
        box_2_neg = get_unique(box_2_neg)
        
        box_3_pos = get_unique(box_3_pos)
        box_3_neg = get_unique(box_3_neg)
        
        # np.save(f'{save_dir}/box_1_{seed}_positive.npy', box_1_pos)
        # np.save(f'{save_dir}/box_1_{seed}_negative.npy', box_1_neg)
        print(f'box 1 pos len {len(box_1_pos)} box 1 neg len {len(box_1_neg)}')
        
        # np.save(f'{save_dir}/box_2_{seed}_positive.npy', box_2_pos)
        # np.save(f'{save_dir}/box_2_{seed}_negative.npy', box_2_neg)
        print(f'box 2 pos len {len(box_2_pos)} box 2 neg len {len(box_2_neg)}')
        
        # np.save(f'{save_dir}/box_3_{seed}_positive.npy', box_3_pos)
        # np.save(f'{save_dir}/box_3_{seed}_negative.npy', box_3_neg)
        print(f'box 3 pos len {len(box_3_pos)} box 3 neg len {len(box_3_neg)}')

collect_states(0, 40000)           
    
    
    
