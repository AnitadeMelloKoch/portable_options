import torch 
import matplotlib.pyplot as plt
import argparse
from experiments.minigrid.utils import environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from portable.utils.utils import load_gin_configs
from portable.agent.model.ppo import create_cnn_policy, create_cnn_vf
from portable.agent.model.ppo import ActionPPO, OptionPPO
import numpy as np
import random
import os 

def get_states(env, num_states):
    states = []
    infos = []
    while len(states) < num_states:
        agent_x = random.randint(1,6)
        agent_y = random.randint(1,6)
        agent_reposition_attempts = random.randint(0,50)
        
        obs, info = env.reset(random_start=False,
                              agent_reposition_attempts=agent_reposition_attempts,
                              agent_position=(agent_x, agent_y),
                              collect_key=np.random.normal()<0.5,
                              door_unlocked=np.random.normal()<0.5,
                              door_open=np.random.normal()<0.5)
        
        states.append(obs)
        infos.append(info)
    
    return states, infos

def idx_with_colour(infos, colour):
    idxs = []
    for info in infos:
        has_key = False
        for key in info["keys"]:
            if key.colour == colour:
                has_key = True
        idxs.append(has_key)
    return idxs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)
    
    load_dir = "runs/oscar/meta_no_options/checkpoints/action_agent"
    plot_dir = "runs/vf_plots/test"
    buffer_size = 5000
    env_seed = 0
    
    def policy_phi(x):
        if type(x) is np.ndarray:
            if np.max(x) > 1:
                x = x/255.0
            x = x.astype(np.float32)
        else:
            if torch.max(x) > 1:
                x = x/255.0
        return x
    
    meta_action_agent = ActionPPO(use_gpu=False,
                                  policy=create_cnn_policy(3, 7),
                                  value_function=create_cnn_vf(3),
                                  phi=policy_phi)
    
    meta_action_agent.load(load_dir)
    
    env = AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder('AdvancedDoorKey-8x8-v0',
                            seed=env_seed,
                            grayscale=False),
        image_input=True,
        door_colour="red"
    )
    
    states, infos = get_states(env, buffer_size)
    
    q_vals = meta_action_agent.q_function(states)
    
    idx_red_key = idx_with_colour(infos, "red")
    idx_yellow_key = idx_with_colour(infos, "yellow")
    idx_grey_key = idx_with_colour(infos, "grey")
    idx_door_open = [
        info["door_open"] for info in infos
    ]
    
    player_x = [
        info["player_x"] for info in infos
    ]
    noise = np.random.normal(0, 0.1, len(player_x))
    player_x = np.array(player_x)
    player_x_noise = player_x + noise
    
    player_y = [
        info["player_y"] for info in infos
    ]
    noise = np.random.normal(0, 0.1, len(player_y))
    player_y = np.array(player_y)
    player_y_noise = player_y + noise
    
    os.makedirs(plot_dir, exist_ok=True)
    
    action_vals = q_vals[:,4]
    
    sc = plt.scatter(
        player_x_noise[idx_red_key],
        player_y_noise[idx_red_key],
        c=action_vals[idx_red_key]
    )
    plt.colorbar(sc)
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "action_4_red_key.png"))
    sc.remove()
    plt.close()
    plt.cla()
    
    nsc = plt.scatter(
        player_x_noise[np.logical_not(idx_red_key)],
        player_y_noise[np.logical_not(idx_red_key)],
        c=action_vals[np.logical_not(idx_red_key)]
    )
    plt.colorbar(nsc)
    plt.grid()
    
    plt.savefig(os.path.join(plot_dir, "action_4_no_red_key.png"))
    plt.cla()















