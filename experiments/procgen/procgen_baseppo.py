from experiments.procgen.core.base_procgen_experiment import ProcgenBasePPOExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch

def phi(x):
    x = x/255.
    x = torch.from_numpy(x).float()
    return x

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
    
    experiment = ProcgenBasePPOExperiment(base_dir=args.base_dir,
                                          experiment_seed=args.seed,
                                          policy_phi=phi)
    
    experiment.run()




