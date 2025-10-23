#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=4-12:00:00
#SBATCH --partition=pancake
#SBATCH --job-name=sokoban_perfect 
#SBATCH --error=sokoban_perfect.%j.err
#SBATCH --output=sokoban_perfect.%j.out
#SBATCH --gres gpu:2

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate port-options

python -m experiments.sokoban.sokoban_perfect_term --base_dir results --seed 12 --config_file configs/sokoban_perfect_term.gin