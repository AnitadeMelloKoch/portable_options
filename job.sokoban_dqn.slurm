#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=55
#SBATCH --mem=110G
#SBATCH --time=4-12:00:00
#SBATCH --partition=pancake
#SBATCH --job-name=sokoban_dqn
#SBATCH --error=sokoban_dqn.%j.err
#SBATCH --output=sokoban_dqn.%j.out
#SBATCH --gres gpu:1

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate port-options

python -m experiments.sokoban.baseline_dqn --base_dir results --seed 10 --config_file configs/sokoban_dqn.gin --gin_bindings TabularCount.beta=0.001