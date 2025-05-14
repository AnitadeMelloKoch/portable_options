# Learning Transferable Subgoals by Hypothesizing Generalizing Features

Note you need to install the custom_minigrid library and Atari Montezuma from gym. We run in python 3.9.12. The training data is expected to be in the resources folder in the root of this repo (untar resources.tar.gz in root will result in the correct structure).

All configuration files are found in the configs folder. 


## Montezuma ClimbDownLadder Experiment
```
python -m experiments.divdis_monte.divdis_monte_policy_ladder_down_experiment --base_dir results --seed 0 --num_rooms 1 --config_file configs/divdis_monte_ladder_policy.gin
```

## Minigrid DoorMultiKey
```
python -um experiments.divdis_minigrid.divdis_image_meta_full --config_file configs/divdis_image_meta_full_10.gin --base_dir runs --seed 0
```

Minigrid also has a one_head variant and ppo and dqn baselines