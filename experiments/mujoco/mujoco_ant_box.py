from experiments.mujoco import MujocoExperiment
import argparse

from portable.utils.utils import load_gin_configs

initiation_positive_files = [
    "resources/mujoco_images/box_initiation_positive.npy",
    "resources/mujoco_images/box_policy.npy",
]
initiation_negative_files = [
    "resources/mujoco_images/box_initiation_negative.npy",
]
initiation_priority_negative_files = []
termination_positive_files = [
    "resources/mujoco_images/box_termination_positive.npy",
]
termination_negative_files = [
    "resources/mujoco_images/box_termination_negative.npy",
]
termination_priority_negative_files = [
    "resources/mujoco_images/box_initiation_positive.npy",
    "resources/mujoco_images/box_policy.npy",
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

    experiment = MujocoExperiment(
        base_dir=args.base_dir,
        seed=args.seed,
        env_name='ant_box',
        initiation_positive_files=initiation_positive_files,
        initiation_negative_files=initiation_negative_files,
        initiation_priority_negative_files=initiation_priority_negative_files,
        termination_positive_files=termination_positive_files,
        termination_negative_files=termination_negative_files,
        termination_priority_negative_files=termination_priority_negative_files
    )

    experiment.save()

    for x in range(5):

        experiment.run_trial(
            100,
            eval=True,
            trial_name="env {}".format(x)
        )

        experiment.save()
