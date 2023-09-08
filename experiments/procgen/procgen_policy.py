from experiments.procgen.core.procgen_experiment import ProcgenExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch

def phi(x):
    
    return x

def embedding_phi(x, use_gpu):
    x = x/255.
    x = torch.from_numpy(x).float()
    if use_gpu:
        x = x.to("cuda")
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
    
    experiment = ProcgenExperiment(base_dir=args.base_dir,
                                   experiment_seed=args.seed,
                                   policy_phi=phi,
                                   embedding_phi=embedding_phi)
    
    experiment.load_embedding("resources/encoders/procgen/encoder.ckpt")
    
    experiment.run()
    
    




