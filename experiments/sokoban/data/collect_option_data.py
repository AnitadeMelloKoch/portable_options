from experiments.sokoban.utils import environment_builder
import numpy as np
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt

def get_unique(states):
    if len(states) < 2:
        return states
    stacked_states = np.stack(states)
    stacked_states = np.unique(stacked_states, axis=0)
    
    array_list = [state for state in stacked_states]
    
    return array_list


def collect_data(seed=0):
    box_1_pos = []
    box_1_neg = []
    box_2_pos = []
    box_2_neg = []
    box_3_pos = []
    box_3_neg = []
    
    save_dir = 'resources/sokoban_images/2_box'
    os.makedirs(save_dir, exist_ok=True)
    env = environment_builder(level_name="PushAndPull-Sokoban-v2", seed = seed)
    
    if os.path.exists(f'{save_dir}/box_1_1_{seed}_positive.npy'):
        box_1_pos = np.load(f'{save_dir}/box_1_1_{seed}_positive.npy')
        box_1_pos = [state for state in box_1_pos]
    
    if os.path.exists(f'{save_dir}/box_1_1_{seed}_negative.npy'):
        box_1_neg = np.load(f'{save_dir}/box_1_1_{seed}_negative.npy')
        box_1_neg = [state for state in box_1_neg]
    
    if os.path.exists(f'{save_dir}/box_2_1_{seed}_positive.npy'):
        box_2_pos = np.load(f'{save_dir}/box_2_1_{seed}_positive.npy')
        box_2_pos = [state for state in box_2_pos]
    
    if os.path.exists(f'{save_dir}/box_2_1_{seed}_negative.npy'):
        box_2_neg = np.load(f'{save_dir}/box_2_1_{seed}_negative.npy')
        box_2_neg = [state for state in box_2_neg]
    
    if os.path.exists(f'{save_dir}/box_3_1_{seed}_positive.npy'):
        box_3_pos = np.load(f'{save_dir}/box_3_1_{seed}_positive.npy')
        box_3_pos = [state for state in box_3_pos]
    
    if os.path.exists(f'{save_dir}/box_3_1_{seed}_negative.npy'):
        box_3_neg = np.load(f'{save_dir}/box_3_1_{seed}_negative.npy')
        box_3_neg = [state for state in box_3_neg]
    
    
    
    random_steps = 0
    for _ in tqdm(range(4000)):
        obs, info = env.reset()
        plt.imshow(obs)
        plt.show(block=False)
        for _ in range(1000):            
            if "in_target_box_locations" in info:
                box_in = info["in_target_box_locations"]
                if len(box_in) == 1:
                    box_1_pos.append(obs)
                    box_2_neg.append(obs)
                    box_3_neg.append(obs)
                elif len(box_in) == 2:
                    box_1_pos.append(obs)
                    box_2_pos.append(obs)
                    box_3_neg.append(obs)
                elif len(box_in) == 3:
                    box_1_pos.append(obs)
                    box_2_pos.append(obs)
                    box_3_pos.append(obs)
            else:
                box_1_neg.append(obs)
                box_2_neg.append(obs)
                box_3_neg.append(obs)
            
            if random_steps == 0:
                action = input("action: ")
                
                if action == "q":
                    break
                if action == "r":
                    random_steps = int(input("num random steps: "))
                    action = random.randint(0,12)
            else:
                action = random.randint(0,12)
                random_steps -= 1
            
            
            
            obs, _, _, info = env.step(action)
            plt.imshow(obs)
            plt.show(block=False)
            plt.pause(0.1)
            
        
        box_1_pos = get_unique(box_1_pos)
        box_1_neg = get_unique(box_1_neg)
        
        box_2_pos = get_unique(box_2_pos)
        box_2_neg = get_unique(box_2_neg)
        
        box_3_pos = get_unique(box_3_pos)
        box_3_neg = get_unique(box_3_neg)
        
        if len(box_1_neg) > 0:
            np.save(f'{save_dir}/box_1_1_{seed}_positive.npy', box_1_pos)
        if len(box_1_pos) > 0:
            np.save(f'{save_dir}/box_1_1_{seed}_negative.npy', box_1_neg)
        print(f'box 1 pos len {len(box_1_pos)} box 1 neg len {len(box_1_neg)}')
        
        if len(box_2_neg) > 0:
            np.save(f'{save_dir}/box_2_1_{seed}_negative.npy', box_2_neg)
        if len(box_2_pos) > 0:
            np.save(f'{save_dir}/box_2_1_{seed}_positive.npy', box_2_pos)
        print(f'box 2 pos len {len(box_2_pos)} box 2 neg len {len(box_2_neg)}')
        
        if len(box_3_neg) > 0:
            np.save(f'{save_dir}/box_3_1_{seed}_negative.npy', box_3_neg)
        if len(box_3_pos) > 0:
            np.save(f'{save_dir}/box_3_1_{seed}_positive.npy', box_3_pos)
        print(f'box 3 pos len {len(box_3_pos)} box 3 neg len {len(box_3_neg)}')

collect_data(6)

