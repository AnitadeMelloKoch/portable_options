from experiments.minigrid.doorkey.core.minigrid_option_experiment import MinigridOptionExperiment
from experiments.minigrid.doorkey.core.agents.rainbow import Rainbow
import os
import numpy as np 
from experiments.minigrid.utils import environment_builder
import argparse 
from experiments.minigrid.doorkey.core.policy_train_wrapper import DoorKeyPolicyTrainWrapper

from portable.utils.utils import load_gin_configs
import torch

initiation_positive_files = [
    # key
    ["resources/minigrid_images/doorkey_gokey_1_initiation_positive.npy"],
    # door
    ["resources/minigrid_images/doorkey_godoor_1_initiation_positive.npy",
     "resources/minigrid_images/doorkey_godoor_2_initiation_positive.npy"],
    # goal
    ["resources/minigrid_images/doorkey_gogoal_1_initiation_positive.npy"]
]
initiation_negative_files = [
    # key
    ["resources/minigrid_images/doorkey_gokey_1_initiation_negative.npy"],
    # door
    ["resources/minigrid_images/doorkey_godoor_1_initiation_negative.npy",
     "resources/minigrid_images/doorkey_godoor_2_initiation_negative.npy"],
    # goal
    ["resources/minigrid_images/doorkey_gogoal_1_initiation_negative.npy"]
]
initiation_priority_negative_files = [
    # key
    [],
    # door
    [],
    # goal
    []
]

termination_positive_files = [
    # key
    ["resources/minigrid_images/doorkey_gokey_1_termination_positive.npy"],
    # door
    ["resources/minigrid_images/doorkey_godoor_1_termination_positive.npy",
     "resources/minigrid_images/doorkey_godoor_2_termination_positive.npy"],
    # goal
    ["resources/minigrid_images/doorkey_gogoal_1_termination_positive.npy"]
]
termination_negative_files = [
    # key
    ["resources/minigrid_images/doorkey_gokey_1_termination_negative.npy"],
    # door
    ["resources/minigrid_images/doorkey_godoor_1_termination_negative.npy",
     "resources/minigrid_images/doorkey_godoor_1_termination_negative.npy"],
    # goal
    ["resources/minigrid_images/doorkey_gogoal_1_termination_negative.npy"]
]
termination_priority_negative_files = [
    # key
    [],
    # door
    [],
    # goal
    []
]

train_seed = 0

def policy_phi(x):
    if np.max(x) > 1:
        x = x/255.0
    x = x.astype(np.float32)
    return x

def create_env(seed):
        return environment_builder('MiniGrid-DoorKey-8x8-v0', 
                                   seed=seed,
                                   grayscale=False)

def at_key(info):
    obj_x, obj_y = info["key_pos"]
    player_x = info["player_x"]
    player_y = info["player_y"]
    
    if (abs(obj_x-player_x) + abs(obj_y-player_y)) <= 1:
        return True
    return False

def at_door(info):
    obj_x, obj_y = info["door_pos"]
    player_x = info["player_x"]
    player_y = info["player_y"]
    
    if (abs(obj_x-player_x) + abs(obj_y-player_y)) <= 1:
        return True
    return False

def at_goal(info):
    obj_x, obj_y = info["goal_pos"]
    player_x = info["player_x"]
    player_y = info["player_y"]
    
    if (abs(obj_x-player_x) + abs(obj_y-player_y)) <= 1:
        return True
    return False

start_pos = [
    [(1,1), (2,1), (3,1), (4,1),
    (1,2), (2,2), (3,2), (4,2),
    (1,3), (2,3), (3,3), (4,3),
    (1,4), (2,4), (3,4), 
    (1,5), (2,5), 
    (1,6), (2,6)],
    [(1,1), (2,1), (3,1), (4,1), (6,1),
    (1,2), (2,2), (3,2), (4,2), 
    (1,3), (2,3), (3,3), (4,3), (6,3),
    (1,4), (2,4), (3,4), (4,4), (6,4),
    (1,5), (2,5), (3,5),
    (1,6), (2,6), (3,6),(4,6)],
    [(1,1), (2,1), (3,1), (4,1), (6,1),
    (1,2), (2,2), (3,2), (4,2), (6,2),
    (1,3), (2,3), (3,3), (4,3), (6,3),
    (1,4), (2,4), (3,4), (4,4), 
    (1,5), (2,5), (3,5),]
]

train_envs = [
    [DoorKeyPolicyTrainWrapper(environment_builder('MiniGrid-DoorKey-8x8-v0', 
                          seed=train_seed,
                          grayscale=False,
                          random_reset=True,
                          max_steps=50,
                          random_starts=start_pos[0]) , at_key)],
    [DoorKeyPolicyTrainWrapper(environment_builder('MiniGrid-DoorKey-8x8-v0', 
                          seed=train_seed,
                          grayscale=False,
                          random_reset=True,
                          max_steps=50,
                          random_starts=start_pos[1]) , at_door),
    DoorKeyPolicyTrainWrapper(environment_builder('MiniGrid-DoorKey-8x8-v0', 
                          seed=train_seed,
                          grayscale=False,
                          random_reset=True,
                          max_steps=50,
                          random_starts=start_pos[1]) , at_door, key_picked_up=True),
    DoorKeyPolicyTrainWrapper(environment_builder('MiniGrid-DoorKey-8x8-v0', 
                          seed=train_seed,
                          grayscale=False,
                          random_reset=True,
                          max_steps=50,
                          random_starts=start_pos[1]) , at_door, door_unlocked=True), 
    DoorKeyPolicyTrainWrapper(environment_builder('MiniGrid-DoorKey-8x8-v0', 
                          seed=train_seed,
                          grayscale=False,
                          random_reset=True,
                          max_steps=50,
                          random_starts=start_pos[1]) , at_door, door_open=True)],
    [DoorKeyPolicyTrainWrapper(environment_builder('MiniGrid-DoorKey-8x8-v0', 
                          seed=train_seed,
                          grayscale=False,
                          random_reset=True,
                          max_steps=50,
                          random_starts=start_pos[2]), at_goal, door_open=True)],
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    
    load_gin_configs(args.config_file, args.gin_bindings)
    
    def create_agent(
            n_actions,
            gpu,
            n_input_channels,
            env_steps=500_000,
            lr=1e-4,
            sigma=0.5
        ):
        kwargs = dict(
            n_atoms=51, v_max=10., v_min=-10.,
            noisy_net_sigma=sigma, lr=lr, n_steps=3,
            betasteps=env_steps // 4,
            replay_start_size=1024, 
            replay_buffer_size=int(3e5),
            gpu=gpu, n_obs_channels=n_input_channels,
            use_custom_batch_states=False,
            epsilon_decay_steps=12500 # don't forget to change
        )
        return Rainbow(n_actions, **kwargs)
    
    
    
    experiment = MinigridOptionExperiment(base_dir=args.base_dir,
                            random_seed=args.seed,
                            create_env_function=create_env,
                            create_agent_function=create_agent,
                            action_space=10,
                            initiation_positive_files=initiation_positive_files,
                            initiation_negative_files=initiation_negative_files,
                            initiation_priority_negative_files=initiation_priority_negative_files,
                            termination_positive_files=termination_positive_files,
                            termination_negative_files=termination_negative_files,
                            termination_priority_negative_files=termination_priority_negative_files,
                            training_envs=train_envs,
                            policy_phi=policy_phi
                            )
    
    experiment.train_test_envs()