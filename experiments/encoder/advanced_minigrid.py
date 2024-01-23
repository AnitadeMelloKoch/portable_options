from experiments.encoder.core.embedding_classifier_experiment import EmbeddingClassifierExperiment
import argparse
from portable.utils.utils import load_gin_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    
    

