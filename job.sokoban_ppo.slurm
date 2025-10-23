#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=55
#SBATCH --mem=110G
#SBATCH --time=2-12:00:00
#SBATCH --partition=pancake
#SBATCH --job-name=sokoban_ppo
#SBATCH --error=sokoban_ppo.%j.err
#SBATCH --output=sokoban_ppo.%j.out
#SBATCH --gres gpu:1

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate port-options

python -m experiments.sokoban.baseline_ppo --base_dir results --seed 1 --config_file configs/sokoban_ppo.gin --gin_bindings ActionPPO.entropy_coef=0.1