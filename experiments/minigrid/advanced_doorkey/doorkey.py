from experiments.minigrid.advanced_doorkey.core.advanced_minigrid_experiment import AdvancedMinigridExperiment
import argparse
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import environment_builder 
from portable.utils.utils import load_gin_configs
from portable.agent.model.rainbow import Rainbow
from portable.option.markov.nn_markov_option import NNMarkovOption
from experiments.minigrid.advanced_doorkey.advanced_minigrid_option_resources import *
import numpy as np
import torch
import random
from portable.agent.model.option_q import OptionDQN

def make_random_getkey_env(train_colour, check_option_complete):
    colours = ["red", "green", "blue", "purple", "yellow", "grey"]
    possible_key_colours = list(filter(lambda c: c!= train_colour, colours))
    
    door_colour = random.choice(possible_key_colours)
    possible_key_colours = list(filter(lambda c: c!= door_colour, possible_key_colours))
    other_col = random.choice(possible_key_colours)
    key_cols = [train_colour, other_col]
    random.shuffle(key_cols)
    
    return AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=training_seed,
            grayscale=False
        ),
        check_option_complete=check_option_complete,
        door_colour=door_colour,
        key_colours=key_cols,
        time_limit=50
    )

def make_random_opendoor_env(train_colour, check_option_complete):
    colours = ["red", "green", "blue", "purple", "yellow", "grey"]
    possible_key_colours = list(filter(lambda c: c!= train_colour, colours))
    
    key_cols = random.choices(possible_key_colours, k=2)
    random.shuffle(key_cols)
    
    return AdvancedDoorKeyPolicyTrainWrapper(
        environment_builder(
            'AdvancedDoorKey-8x8-v0',
            seed=training_seed,
            grayscale=False
        ),
        check_option_complete=check_option_complete,
        door_colour=train_colour,
        key_colours=key_cols,
        key_collected=True,
        time_limit=50
    )

training_envs = [
    # get red key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_got_redkey,
            door_colour="red",
            time_limit=50
        ),
        make_random_getkey_env("red", check_got_redkey),
        make_random_getkey_env("red", check_got_redkey),
        make_random_getkey_env("red", check_got_redkey),
    ],
    # get blue key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_got_bluekey,
            door_colour="blue",
            time_limit=50
        ),
        make_random_getkey_env("blue", check_got_bluekey),
        make_random_getkey_env("blue", check_got_bluekey),
        make_random_getkey_env("blue", check_got_bluekey),
        
    ],
    # get green key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_got_greenkey,
            door_colour="green",
            time_limit=50
        ),
        make_random_getkey_env("green", check_got_greenkey),
        make_random_getkey_env("green", check_got_greenkey),
        make_random_getkey_env("green", check_got_greenkey),
    ],
    # get purple key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_got_purplekey,
            door_colour="purple",
            time_limit=50
        ),
        make_random_getkey_env("purple", check_got_purplekey),
        make_random_getkey_env("purple", check_got_purplekey),
        make_random_getkey_env("purple", check_got_purplekey),
    ],
    # get yellow key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_got_yellowkey,
            door_colour="yellow",
            time_limit=50
        ),
        make_random_getkey_env("yellow", check_got_yellowkey),
        make_random_getkey_env("yellow", check_got_yellowkey),
        make_random_getkey_env("yellow", check_got_yellowkey),
    ],
    # get grey key
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_got_greykey,
            door_colour="grey",
            time_limit=50
        ),
        make_random_getkey_env("grey", check_got_greykey),
        make_random_getkey_env("grey", check_got_greykey),
        make_random_getkey_env("grey", check_got_greykey),
    ],
    # open red door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_dooropen,
            door_colour="red",
            key_collected=True,
            time_limit=50
        ),
        make_random_opendoor_env("red", check_dooropen),
        make_random_opendoor_env("red", check_dooropen),
        make_random_opendoor_env("red", check_dooropen),
    ],
    # open blue door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_dooropen,
            door_colour="blue",
            key_collected=True,
            time_limit=50
        ),
        make_random_opendoor_env("blue", check_dooropen),
        make_random_opendoor_env("blue", check_dooropen),
        make_random_opendoor_env("blue", check_dooropen),
    ],
    # open green door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_dooropen,
            door_colour="green",
            key_collected=True,
            time_limit=50
        ),
        make_random_opendoor_env("green", check_dooropen),
        make_random_opendoor_env("green", check_dooropen),
        make_random_opendoor_env("green", check_dooropen),
    ],
    # open purple door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_dooropen,
            door_colour="purple",
            key_collected=True,
            time_limit=50
        ),
        make_random_opendoor_env("purple", check_dooropen),
        make_random_opendoor_env("purple", check_dooropen),
        make_random_opendoor_env("purple", check_dooropen),
    ],
    # open yellow door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_dooropen,
            door_colour="yellow",
            key_collected=True,
            time_limit=50
        ),
        make_random_opendoor_env("yellow", check_dooropen),
        make_random_opendoor_env("yellow", check_dooropen),
        make_random_opendoor_env("yellow", check_dooropen),
    ],
    # open grey door
    [
        AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=training_seed,
                grayscale=False
            ),
            check_option_complete=check_dooropen,
            door_colour="grey",
            key_collected=True,
            time_limit=50
        ),
        make_random_opendoor_env("grey", check_dooropen),
        make_random_opendoor_env("grey", check_dooropen),
        make_random_opendoor_env("grey", check_dooropen),
    ],
]

names = [
    "get red key",
    "get blue key",
    "get green key",
    "get purple key",
    "get yellow key",
    "get grey key",
    "open red door",
    "open blue door",
    "open green door",
    "open purple door",
    "open yellow door",
    "open grey door",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--frames_per_env", type=int, required=True)
    parser.add_argument("--skip_train_options", action='store_true')
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    
    load_gin_configs(args.config_file, args.gin_bindings)
    
    def policy_phi(x):
        if type(x) is np.ndarray:
            if np.max(x) > 1:
                x = x/255.0
            x = x.astype(np.float32)
        else:
            if torch.max(x) > 1:
                x = x/255.0
        return x
    
    def create_agent(n_actions):
        kwargs = dict(
            n_atoms=51, v_max=10., v_min=-10.,
            noisy_net_sigma=0.5, lr=1e-4, n_steps=3,
            betasteps=500_000 // 4,
            replay_start_size=1024, 
            replay_buffer_size=int(3e5),
            gpu=0, n_obs_channels=3,
            use_custom_batch_states=False,
            epsilon_decay_steps=125000 # don't forget to change
        )
        return Rainbow(n_actions, **kwargs)
    
    def create_markov_option(states,
                             infos,
                             termination_state,
                             termination_info,
                             initiation_votes,
                             termination_votes,
                             false_states,
                             initial_policy,
                             use_gpu,
                             save_file):
        labels = [1]*len(states) + [0]*len(false_states)
        data = list(states) + list(false_states)
        
        option = NNMarkovOption(use_gpu=use_gpu,
                                initiation_states=data,
                                initiation_labels=labels,
                                termination=termination_state,
                                initial_policy=initial_policy,
                                initiation_votes=initiation_votes,
                                termination_votes=termination_votes,
                                save_file=save_file)
        
        return option
    
    def create_env(seed):
        colours = ["red", "green", "blue", "purple", "yellow", "grey"]
        door_colour = random.choice(colours)
        possible_key_colours = list(filter(lambda c: c != door_colour, colours))
        random.shuffle(possible_key_colours)
        
        env = AdvancedDoorKeyPolicyTrainWrapper(
            environment_builder(
                'AdvancedDoorKey-8x8-v0',
                seed=seed,
                grayscale=False
            ),
            door_colour=door_colour,
            key_colours=possible_key_colours
        )
        
        return env
    
    experiment = AdvancedMinigridExperiment(base_dir=args.base_dir,
                                            training_seed=training_seed,
                                            experiment_seed=args.seed,
                                            action_agent=create_agent(13),
                                            option_agent=OptionDQN(height=64,
                                                                   width=64,
                                                                   channel_num=3,
                                                                   action_vector_size=13,
                                                                   num_options=20),
                                            global_option=create_agent(7),
                                            num_options=12,
                                            markov_option_builder=create_markov_option,
                                            policy_phi=policy_phi,
                                            names=names,
                                            termination_oracles=termination_oracles_doorkey)
    
    experiment.add_datafiles(initiation_positive_files=initiation_positive_files,
                             initiation_negative_files=initiation_negative_files,
                             termination_positive_files=termination_positive_files,
                             termination_negative_files=termination_negative_files)
    
    # should train a new embedding? probs not
    experiment.load_embedding(load_dir="resources/encoders/doorkey_8x8/encoder.ckpt")
    
    if not args.skip_train_options:
        experiment.train_options(training_envs=training_envs)
    else:
        experiment.load()
    
    experiment.run(make_env=create_env,
                   num_envs=args.num_envs,
                   frames_per_env=args.frames_per_env)






