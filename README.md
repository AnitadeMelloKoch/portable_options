# portable_options

## ant experiments
```bash
python3 -m portable.policy.transfer --agent sac --num_envs 16 --max_steps 2_000_000 --env ant_box
```
available envs: `ant_box`, `ant_bridge`, `ant_goal`. `ant_mixed` is available but not tested

## procgen experiments
to do transfer learning, for example:
```bash
python -m portable.policy.transfer -e experiment_name --num_levels 20 --transfer_steps 500000 --env ENV --num_policies 3 --seed 0
# fix the attention masks
python -m portable.policy.transfer -e experiment_name --num_levels 20 --transfer_steps 500000 --env ENV --num_policies 3 --seed 0 --fix_attention_masks --load ./results/saved_experiment
# remove feature learners but keep ensemble policy
python -m portable.policy.transfer -e experiment_name --num_levels 20 --transfer_steps 500000 --env ENV --num_policies 3 --seed 0 --remove_feature_learners
# plotting
python -m portable.policy.plot -f [-u] -l /results/path 
```

## montezuma experiments
```bash
python -m experiments.monte_move_left_spider --base_dir runs --seed {} --config_file configs/monte_move_left_spider.gin

python -m experiments.monte_move_left_snake --base_dir runs --seed {} --config_file configs/monte_move_left_snake.gin

python -m experiments.monte_move_left_rolling_skull --base_dir runs --seed {} --config_file configs/monte_move_left_rolling_skull.gin

python -m experiments.monte_move_left_dancing_skull --base_dir runs --seed {} --config_file configs/monte_move_left_dancing_skull.gin 
```