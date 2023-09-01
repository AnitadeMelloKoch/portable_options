from experiments.minigrid.advanced_doorkey.core.advanced_minigrid_experiment import AdvancedMinigridExperiment
import argparse
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import environment_builder 
from portable.utils.utils import load_gin_configs
from experiments.minigrid.doorkey.core.agents.rainbow import Rainbow
from portable.option.markov.nn_markov_option import NNMarkovOption
import numpy as np
from experiments.minigrid.advanced_doorkey.advanced_minigrid_option_resources import *
import random
import torch

# env = environment_builder("MiniGrid-MultiRoom-N6-v0", seed=1, grayscale=False)
# env = environment_builder("MiniGrid-LockedRoom-v0", seed=1, grayscale=False)
# env = environment_builder("AdvancedDoorKey-16x16-v0", 
#                           seed=1, 
#                           grayscale=False,
#                           pad_obs=True,
#                           final_image_size=(200,200))

# (3, 200, 200)

training_envs = [
    # get red key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_got_redkey,
            door_colour="red",
            time_limit=150
        )
    ],
    # get blue key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_got_bluekey,
            door_colour="blue",
            time_limit=150
        )
    ],
    # get green key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_got_greenkey,
            door_colour="green",
            time_limit=150
        )
    ],
    # get purple key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_got_purplekey,
            door_colour="purple",
            time_limit=150
        )
    ],
    # get yellow key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_got_yellowkey,
            door_colour="yellow",
            time_limit=150
        )
    ],
    # get grey key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_got_greykey,
            door_colour="grey",
            time_limit=150
        )
    ],
    # open red door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_dooropen,
            door_colour="red",
            key_collected=True,
            time_limit=150
        )
    ],
    # open blue door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_dooropen,
            door_colour="blue",
            key_collected=True,
            time_limit=150
        )
    ],
    # open green door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_dooropen,
            door_colour="green",
            key_collected=True,
            time_limit=150
        )
    ],
    # open purple door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_dooropen,
            door_colour="purple",
            key_collected=True,
            time_limit=150
        )
    ],
    # open yellow door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_dooropen,
            door_colour="yellow",
            key_collected=True,
            time_limit=150
        )
    ],
    # open grey door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-16x16-v0',
                seed=training_seed,
                grayscale=False,
                pad_obs=True,
                final_image_size=(200, 200)
            ),
            check_option_complete=check_dooropen,
            door_colour="grey",
            key_collected=True,
            time_limit=150
        )
    ],
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--frames_per_env", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    
    load_gin_configs(args.config_file, args.gin_bindings)
    
    def dataset_transform(x):
        x = x/255.0
        
        padding = ((36,36), (36,36))
        
        padded_obs = np.stack([
            np.pad(x[:,:,idx], padding, mode="constant", constant_values=0) for idx in range(3)
        ], axis=-1)
        
        return padded_obs
    
    def policy_phi(x):
        if type(x) is np.ndarray:
            if np.max(x) > 1:
                x = x/255.0
            x = x.astype(np.float32)
        else:
            if torch.max(x) > 1:
                x = x/255.0
        return x
    
    def create_agent(n_actions,
                     gpu,
                     n_input_channels,
                     env_steps=500_000,
                     lr=1e-4,
                     sigma=0.5):
        kwargs = dict(
            n_atoms=51, v_max=10., v_min=-10.,
            noisy_net_sigma=sigma, lr=lr, n_steps=3,
            betasteps=env_steps // 4,
            replay_start_size=1024, 
            replay_buffer_size=int(3e5),
            gpu=gpu, n_obs_channels=n_input_channels,
            use_custom_batch_states=False,
            epsilon_decay_steps=125000 # don't forget to change
        )
        return Rainbow(n_actions, **kwargs)
    
    def create_markov_option(states,
                             termination_state,
                             initiation_votes,
                             termination_votes,
                             false_states,
                             initial_policy,
                             device):
        labels = [1]*len(states) + [0]*len(false_states)
        data = list(states) + list(false_states)
        
        option = NNMarkovOption(device=device,
                                initiation_states=data,
                                initiation_labels=labels,
                                termination=termination_state,
                                initial_policy=initial_policy,
                                initiation_votes=initiation_votes,
                                termination_votes=termination_votes)
        
        return option
    
    def create_env(seed):
        env = environment_builder(
            'MiniGrid-LockedRoom-v0',
            seed=seed,
            grayscale=False
        )
        
        return env
    
    experiment = AdvancedMinigridExperiment(base_dir=args.base_dir,
                                            training_seed=training_seed,
                                            experiment_seed=args.seed,
                                            create_agent_function=create_agent,
                                            num_options=12,
                                            markov_option_builder=create_markov_option,
                                            policy_phi=policy_phi)
    
    experiment.add_datafiles(initiation_positive_files=initiation_positive_files,
                             initiation_negative_files=initiation_negative_files,
                             termination_positive_files=termination_positive_files,
                             termination_negative_files=termination_negative_files)
    
    # should train a new embedding? probs not
    experiment.load_embedding(load_dir="")
    
    experiment.train_options(training_envs=training_envs)
    
    experiment.run(make_env=create_env,
                   num_envs=args.num_envs,
                   frames_per_env=args.frames_per_env)
